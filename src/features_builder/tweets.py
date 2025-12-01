from pathlib import Path

from duckdb import DuckDBPyConnection
import pandas as pd
import torch
from loguru import logger
from tqdm.auto import tqdm

from src.utils.cache import save_tensor, load_tensor
from src.utils.text_embed import TextEmbedder


class TweetBuilder:
    """
    Builds tweet text embeddings in memory-safe chunks.
    Fully cache-aware and uses utils for I/O.
    """

    def __init__(
        self, con: DuckDBPyConnection, out_dir: Path, embedder: TextEmbedder, cfg
    ):
        self.con = con
        self.out_dir = out_dir
        self.embedder = embedder
        self.cfg = cfg

    def embed_k_user_tweets(self, device: str, k: int):
        """
        Embed ONLY the k most recent tweets per user.
        Produces:
            - tweet_ids.pt
            - tweet_embs.pt
            - user_<k>_tweet_embs.pt
        """

        out_ids = self.out_dir / "tweet_ids.pt"
        out_embs = self.out_dir / "tweet_text_embeddings.pt"

        if out_ids.exists() and out_embs.exists():
            logger.info("Loading cached tweet embeddings → tweet_text_embeddings.pt...")
            tweet_ids = load_tensor(out_ids, not_tensor=True)
            tweet_embs = load_tensor(out_embs, device=device)

            return tweet_ids, tweet_embs

        logger.info(f"Querying top-{k} recent tweets per user...")

        # Query only needed tweets
        df = self.con.execute(
            f"""
            SELECT
                users.id AS user_id,
                tweets.id AS tweet_id,
                tweets.text AS text,
                tweets.created_at AS created_at
            FROM users
            LEFT JOIN tweets
                ON ('u' || tweets.author_id) = users.id
            WHERE tweets.text IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY users.id
                ORDER BY tweets.created_at DESC
            ) <= {k}
            ORDER BY users.id, tweets.created_at DESC
            """
        ).df()

        logger.success(f"Fetched {len(df):,} tweets (~{k} per user)")

        tweet_ids_list = df["tweet_id"].astype(str).tolist()
        tweet_texts = df["text"].fillna("").astype(str).tolist()

        logger.info("Embedding recent tweets...")

        chunk_size = self.cfg["tweet_processing"]["chunk_size"]
        num_chunks = (len(tweet_texts) + chunk_size - 1) // chunk_size

        all_embs = []

        for chunk_idx, offset in enumerate(
            range(0, len(tweet_texts), chunk_size), start=1
        ):
            end = min(offset + chunk_size, len(tweet_texts))

            logger.debug(
                f"[Chunk {chunk_idx}/{num_chunks}] "
                f"Embedding tweets {offset:,} → {end:,} "
                f"({(end/len(tweet_texts))*100:.1f}% done)"
            )

            chunk_texts = tweet_texts[offset:end]
            chunk_emb = self.embedder.embed_batch(chunk_texts, show_progress=True)

            all_embs.append(chunk_emb.cpu())

        tweet_embs = torch.cat(all_embs, dim=0)  # (M, 768)
        save_tensor(tweet_embs, out_embs)
        save_tensor(tweet_ids_list, out_ids, not_tensor=True)

        logger.success(f"Saved {out_ids}")
        logger.success(f"Saved {out_embs} with shape {tweet_embs.shape}")

        return tweet_ids_list, tweet_embs

    def embed_recent_tweets_per_user(
        self, user_df: pd.DataFrame, device: str, k: int = 20
    ):
        """
        Returns:
            user_tweet_embs : (num_users, hidden_dim=768)
        """

        path = self.out_dir / "user_tweet_embeddings.pt"

        if path.exists():
            logger.info("Loading cached tweet embeddings → user_tweet_embeddings.pt...")
            user_tweet_embs = load_tensor(path, device=device)

            return user_tweet_embs

        emb_path = self.out_dir / "tweet_text_embeddings.pt"
        ids_path = self.out_dir / "tweet_ids.pt"
        if not emb_path.exists() or not ids_path.exists():
            raise RuntimeError(
                "Tweet embeddings not found! Run embed_all_tweets() first."
            )

        tweet_embs = load_tensor(emb_path, device=device)  # (M, 768)
        tweet_ids = load_tensor(ids_path, not_tensor=True)  # list of M strings
        id_to_idx = {tid: i for i, tid in enumerate(tweet_ids)}

        logger.info("Selecting top-K recent tweets per user...")

        # Query: (user_id, tweet_id, created_at)
        df = self.con.execute(
            """
            SELECT
                users.id AS user_id,
                tweets.id AS tweet_id,
                tweets.created_at
            FROM users
            LEFT JOIN tweets
                ON ('u' || tweets.author_id) = users.id
            WHERE tweets.text IS NOT NULL
            ORDER BY users.id, tweets.created_at DESC
            """
        ).df()

        tids_grouped_by_user = df.groupby("user_id")["tweet_id"].apply(
            lambda s: s.head(k).tolist()
        )

        tids_grouped_by_user = tids_grouped_by_user.reindex(user_df["id"]).apply(
            lambda x: x if isinstance(x, list) else []
        )

        # pool embeddings per user
        pooled = []

        for tids in tids_grouped_by_user.tolist():
            if len(tids) == 0:
                pooled.append(torch.zeros(768, device=device))
                continue

            idxs = [id_to_idx.get(t) for t in tids if t in id_to_idx]
            if len(idxs) == 0:
                logger.warning("Funny! No tweets found for a user.")
                pooled.append(torch.zeros(768, device=device))
                continue

            mat = tweet_embs[idxs]  # (k, 768)
            agg_tweets = mat.mean(dim=0)  # (768,)

            pooled.append(agg_tweets)

        user_tweet_embs = torch.stack(pooled)  # (N, 768)

        save_tensor(user_tweet_embs, path)
        logger.success(f"Saved {path} with shape {user_tweet_embs.shape}")

        return user_tweet_embs
    
    def embed_all_tweets_sorted(self):
        """
        Returns:
            user_tweet_embs : (num_users, hidden_dim=768)
        """
        
        out_ids = self.out_dir / "all_tweet_ids_sorted.pt"
        out_embs = self.out_dir / "all_tweet_embeddings_sorted.pt"
        
        # Check cache
        if out_ids.exists() and out_embs.exists():
            logger.info("Loading cached sorted tweet embeddings...")
            tweet_ids = load_tensor(out_ids, not_tensor=True)
            tweet_embs = load_tensor(out_embs)
            logger.success(f"Loaded {len(tweet_ids):,} sorted tweet embeddings with shape {tweet_embs.shape}")
            return tweet_ids, tweet_embs
        
        logger.info("Streaming tweets from database...")
        
        # Get total count for progress
        total_tweets = self.con.execute(
            "SELECT COUNT(*) FROM tweets WHERE text IS NOT NULL"
        ).fetchone()[0]
        
        logger.info(f"Processing {total_tweets:,} tweets...")
        
        chunk_size = self.cfg["tweet_processing"]["chunk_size"]
        all_embs = []
        all_ids = []
        
        # Stream from database in chunks
        offset = 0
        chunk_idx = 0
        
        with tqdm(total=total_tweets, desc="Embedding tweets") as pbar:
            while offset < total_tweets:
                chunk_idx += 1
                
                # Fetch chunk directly from database
                df_chunk = self.con.execute(
                    f"""
                    SELECT id AS tweet_id, text
                    FROM tweets
                    WHERE text IS NOT NULL
                    ORDER BY id ASC
                    LIMIT {chunk_size} OFFSET {offset}
                    """
                ).df()
                
                if len(df_chunk) == 0:
                    break
                
                chunk_ids = df_chunk["tweet_id"].astype(str).tolist()
                chunk_texts = df_chunk["text"].fillna("").astype(str).tolist()
                
                # Process this chunk
                chunk_emb = self.embedder.embed_batch(
                    chunk_texts, 
                    show_progress=False  # We have outer progress bar
                )
                
                all_embs.append(chunk_emb.cpu())
                all_ids.extend(chunk_ids)
                
                offset += len(df_chunk)
                pbar.update(len(df_chunk))
                
                # Free memory
                del df_chunk, chunk_ids, chunk_texts
        
        tweet_embs = torch.cat(all_embs, dim=0)
        
        save_tensor(tweet_embs, out_embs)
        save_tensor(all_ids, out_ids, not_tensor=True)
        
        logger.success(f"Saved {out_ids}")
        logger.success(f"Saved {out_embs} with shape {tweet_embs.shape}")
        
        return all_ids, tweet_embs