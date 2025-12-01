import torch
import torch.nn as nn
from loguru import logger
from collections import defaultdict
from tqdm import tqdm
import pickle

class NodeEmbeddingBuilder:
    """
    Projects multiple feature blocks (desc, tweet, numeric, categorical)
    into a unified embedding of size cfg["embedding"]["user_embedding_dim"].
    """

    def __init__(self, cfg, out_dir, device):
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = device

        self.desc_proj = None
        self.tweet_proj = None
        self.numeric_proj = None
        self.categorical_proj = None
        self.final_proj = None

    def _init_layers(self, desc_dim, numeric_dim, categorical_dim, tweet_dim=None):
        emb_dim = self.cfg["embedding"]["user_embedding_dim"]
        quarter = emb_dim // 4

        activation = (
            nn.LeakyReLU
            if self.cfg["embedding"]["projection_activation"] == "leaky_relu"
            else nn.ReLU
        )

        self.desc_proj = nn.Sequential(nn.Linear(desc_dim, quarter), activation()).to(
            self.device
        )
        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_dim, quarter), activation()
        ).to(self.device)
        self.categorical_proj = nn.Sequential(
            nn.Linear(categorical_dim, quarter), activation()
        ).to(self.device)

        if tweet_dim is not None:
            self.tweet_proj = nn.Sequential(
                nn.Linear(tweet_dim, quarter), activation()
            ).to(self.device)
        else:
            self.tweet_proj = None

        # fusion MLP
        self.final_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            activation(),
        ).to(self.device)

    # final node embedding
    def static_fuse(self, desc, tweet, numeric, categorical):
        """
        Args:
            desc:         (N, D_desc)
            tweet:        (N, D_tweet) OR None
            numeric:      (N, D_num)
            categorical:  (N, D_cat)

        Returns:
            fused_user_embeddings: (N, embedding_dim)
        """

        emb_dim = self.cfg["embedding"]["user_embedding_dim"]
        fuse_tweets = self.cfg["features"]["include_tweet_in_users"]

        # Build output filename
        fname = f"user_node_embeddings_{emb_dim}_{self.cfg['version_name']}.pt"
        out_path = self.out_dir / fname

        if out_path.exists():
            logger.info(f"Loading cached fused node embeddings → {fname}")
            return load_tensor(out_path, device=device)

        if self.desc_proj is None:
            logger.info("Initializing node embedding projection layers...")

            self._init_layers(
                desc_dim=desc.size(1),
                numeric_dim=numeric.size(1),
                categorical_dim=categorical.size(1),
                tweet_dim=(
                    tweet.size(1) if (tweet is not None and fuse_tweets) else None
                ),
            )

        d = self.desc_proj(desc)
        n = self.numeric_proj(numeric)
        c = self.categorical_proj(categorical)

        if fuse_tweets and (tweet is not None):
            t = self.tweet_proj(tweet)
            combined = torch.cat([d, t, n, c], dim=1)  # (N, 128)
            fused = self.final_proj(combined)
            save_tensor(fused, out_path)
            logger.success(f"Saved fused node embeddings {out_path}")
            return fused

        combined = torch.cat([d, n, c], dim=1)  # (N, 96)
        # No final projection
        save_tensor(combined, out_path)
        logger.success(f"Saved non-tweet node embeddings (raw 96-dim) {out_path}")
        return combined

    # temporalized node embeddings
    def temporal_fuse(self, snapshots_path, max_snapshots: int = 800):
        user_emb_path = self.out_dir / f"user_node_embeddings_{self.cfg['embedding']['user_embedding_dim']}_{self.cfg['version_name']}.pt"
        tweet_emb_path = self.out_dir / "all_tweet_embeddings_sorted.pt"

        if not user_emb_path.exists() or not tweet_emb_path.exists():
            raise RuntimeError(
                "User or tweet embeddings not found! Run build_all() and build_all_tweets() first."
            )
        
        user_embeddings = torch.load(user_emb_path, map_location=self.device, weights_only=False)  # (N, 96)
        print("Loaded user embeddings shape:", user_embeddings.shape)
        tweet_embeddings_roberta = torch.load(tweet_emb_path, map_location=self.device, weights_only=False)  # (M, 768) and dtype=torch.float16
        print("Loaded tweet embeddings shape:", tweet_embeddings_roberta.shape)

        tweet_projector = nn.Sequential(
            nn.Linear(tweet_embeddings_roberta.size(1), self.cfg["embedding"]["tweet_embedding_dim"] // 4),
            nn.LeakyReLU(),
        ).to(self.device).half()

        # Process in batches
        batch_size = 100000
        tweet_embeddings_list = []

        for i in tqdm(range(0, len(tweet_embeddings_roberta), batch_size), 
                    desc="Projecting tweet embeddings"):
            batch = tweet_embeddings_roberta[i:i + batch_size]
            with torch.no_grad():
                batch_out = tweet_projector(batch).float()
                tweet_embeddings_list.append(batch_out.cpu())
            
        tweet_embeddings = torch.cat(tweet_embeddings_list).to(self.device) # (M, 32)

        user_ids_path = self.out_dir / "user_ids.pt"
        tweet_ids_path = self.out_dir / "tweet_ids.pt"

        if not user_ids_path.exists() or not tweet_ids_path.exists():
            raise RuntimeError(
                "User or tweet IDs not found! Run UserLoader and TweetBuilder first."
            )
        
        user_ids = torch.load(user_ids_path, map_location="cpu", weights_only=False) # (N,)
        tweet_ids = torch.load(tweet_ids_path, map_location="cpu", weights_only=False) # (M,)

        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        tweet_id_to_idx = {tid: idx for idx, tid in enumerate(tweet_ids)}

        with open(snapshots_path, 'rb') as f:
            snapshot_idx = 0
            pbar = tqdm(total=max_snapshots, desc="Processing snapshots", unit="snapshot")
            
            while snapshot_idx < max_snapshots:
                try:
                    snapshot = pickle.load(f)
                except EOFError:
                    logger.warning(f"Reached end of file after {snapshot_idx} snapshots")
                    break

                active_user_ids = snapshot['user_ids']
                active_tweet_ids = snapshot['tweet_ids']
                active_edges = snapshot['edges']

                final_embeddings_path = self.out_dir / "temporal" / f"snap_{snapshot_idx:04d}_embeddings.pt"

                # User embeddings for active users (batch indexing)
                valid_active_user_ids = [uid for uid in active_user_ids if uid in user_id_to_idx]
                active_user_global_indices = torch.tensor([user_id_to_idx[uid] for uid in valid_active_user_ids])
                active_user_embeddings = user_embeddings[active_user_global_indices]  # (num_active_users, 96)
                active_user_embeddings = active_user_embeddings.to(self.device)

                # Tweets by user (only 'post' relation for owned tweets)
                user_to_tweet_indices = defaultdict(list)
                
                for user_id, relation, tweet_id in active_edges:
                    if relation == 'post':
                        global_user_idx = user_id_to_idx[user_id]
                        global_tweet_idx = tweet_id_to_idx[tweet_id]
                        user_to_tweet_indices[global_user_idx].append(global_tweet_idx)

                # Create averaged tweet embeddings for each active user
                tweet_emb_list = []
                for user_id in valid_active_user_ids:
                    global_user_idx = user_id_to_idx[user_id]
                    if user_to_tweet_indices[global_user_idx]:
                        tweet_indices = user_to_tweet_indices[global_user_idx]
                        tweet_embs = tweet_embeddings[tweet_indices]  # (num_user_tweets, 32)
                        avg_tweet_emb = tweet_embs.mean(dim=0)  # (32,)
                    else:
                        avg_tweet_emb = torch.zeros(32, device=self.device)
                    tweet_emb_list.append(avg_tweet_emb)

                active_tweet_embeddings = torch.stack(tweet_emb_list) # (num_active_users, 32)

                # Concatenate user and tweet embeddings
                final_embeddings = torch.cat([active_user_embeddings, active_tweet_embeddings], dim=1)  # (num_active_users, 128)

                if not final_embeddings_path.parent.exists():
                    final_embeddings_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save(final_embeddings, final_embeddings_path)
                
                pbar.set_postfix({
                    'users': len(valid_active_user_ids),
                    'edges': len(active_edges)
                })
                pbar.update(1)

                snapshot_idx += 1

                del final_embeddings
                torch.cuda.empty_cache()
            
            pbar.close()

        logger.success("Completed temporalized embeddings for all snapshots.")
