import datetime as dt
import os
from typing import Optional, List

import duckdb
from loguru import logger
import torch
import torch.nn as nn
from transformers import pipeline
from tqdm import tqdm
import pandas as pd

class Twibot22DataBuilder:
    """
    Node embedding and label builder for Twibot-22.

    Design:
    - One node = one user
    - Features per user:
        * (text) user's profile description RoBERTa embedding
        * (text) user's aggregated tweets RoBERTa embedding
        * (numeric) followers, following, tweet_count, active_days, screen_name_len
        * (categorical) protected, verified, default_profile_image (0/1)
    - The features are fused through a single layer MLP and transformed into `embedding_dim`.
    """

    def __init__(
        self,
        db_path: str = "db/twitter_graph.duckdb",
        out: str = "./dataset/",
        load_model: bool = True,
        model_name: str = "roberta-base",
        embedding_dim: int = 128,
        max_tweets_per_user: int = 20,
        text_batch_size: int = 32,
        build_like_count: bool = False,
    ):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.build_like_count = build_like_count

        self.con = duckdb.connect(db_path)

        self.users: Optional[pd.DataFrame] = None
        self.embedding_dim = embedding_dim
        self.max_tweets_per_user = max_tweets_per_user
        self.text_batch_size = text_batch_size

        # Projection layers
        self.linear_relu_desc: Optional[nn.Module] = None
        self.linear_relu_tweets: Optional[nn.Module] = None
        self.linear_relu_num_prop: Optional[nn.Module] = None
        self.linear_relu_cat_prop: Optional[nn.Module] = None
        self.linear_relu_input: Optional[nn.Module] = None

        # Device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            pipeline_device = 0
            logger.debug("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            pipeline_device = "mps"
            logger.debug("Using Apple MPS backend")
        else:
            self.device = "cpu"
            pipeline_device = -1
            logger.debug("Using CPU")

        # Load transformer model
        if load_model:
            logger.info(f"Loading tokenizer/model [{model_name}]...")
            self.model = pipeline(
                "feature-extraction",
                model=model_name,
                tokenizer=model_name,
                device=pipeline_device,
            )
        else:
            logger.info("Skipping model load (test mode).")
            self.model = None

    def _init_projection_layers(
        self, desc_size: int, tweet_size: int, num_prop_size: int, cat_prop_size: int
    ):
        """Initialize projection layers for each feature block."""
        quarter = self.embedding_dim // 4

        self.linear_relu_desc = nn.Sequential(
            nn.Linear(desc_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_tweets = nn.Sequential(
            nn.Linear(tweet_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_input = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(),
        ).to(self.device)

    def _embed_text_batch(
        self, texts: List[str], show_progress: bool = False
    ) -> torch.Tensor:
        """
        Efficiently embed a list of strings using the HF pipeline in batches.

        Returns: (N, hidden_dim) tensor on selected device
        """
        if self.model is None:
            logger.warning("Text embedding model is not initialized. Returning zeros.")
            if len(texts) == 0:
                return torch.zeros(0, 768, device=self.device)  # (0, 768)
            return torch.zeros(len(texts), 768, device=self.device)  # (N, 768)

        all_embs = []  # list of (batch_size, 768)
        iterator = range(0, len(texts), self.text_batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Embedding text",
                total=(len(texts) - 1) // self.text_batch_size + 1,
            )

        for start in iterator:
            batch = texts[start : start + self.text_batch_size]
            batch = [
                t if isinstance(t, str) and t.strip() else "" for t in batch
            ]  # list[str], len=batch_size

            features = self.model(
                batch, truncation=True
            )  # [ [seq_len_f * 768], ... ] length=batch_size

            pooled_batch = []
            for f in features:
                t = torch.tensor(f, dtype=torch.float32)  # (1, seq_len_f, 768)
                t = t.squeeze(0) if t.ndim == 3 else t

                pooled = t.mean(dim=0)  # (768,)
                pooled_batch.append(pooled)

            batch_embs = torch.stack(pooled_batch, dim=0).to(self.device)
            all_embs.append(batch_embs)  # list of (batch_size, 768)

        if not all_embs:
            return torch.zeros(0, 768, device=self.device)

        return torch.cat(all_embs, dim=0)  # (N, 768)

    def load_users(self):
        """Load and cache users table. Keep a fixed order."""
        logger.info("Loading users from DuckDB...")

        df = self.con.execute(
            """
            SELECT *
            FROM users
            ORDER BY id
            """
        ).df()

        self.users = df
        logger.success(f"Users loaded: {len(df)} rows")

    def build_user_desc_embeddings(self) -> torch.Tensor:
        """
        Build / load RoBERTa embeddings for user descriptions.

        Returns:
        torch.Tensor of shape (num_users, hidden_dim)
        """
        if self.users is None:
            raise RuntimeError(
                "Call load_users() before building description embeddings."
            )

        path = os.path.join(self.out, "user_desc_embeddings.pt")
        if os.path.exists(path):
            logger.info("Loading cached user description embeddings...")
            desc_vectors = torch.load(path, map_location="cpu").to(self.device)
            logger.success("Loaded user_desc_embeddings.pt")
            return desc_vectors

        logger.info("Encoding user descriptions (batched)...")

        desc_texts = self.users["description"].fillna("").astype(str).tolist()
        desc_vectors = self._embed_text_batch(desc_texts, show_progress=True)

        torch.save(desc_vectors.cpu(), path)
        logger.success("Saved user_desc_embeddings.pt")
        return desc_vectors

    def build_user_tweet_embeddings(self) -> torch.Tensor:
        """
        Build / load a tweet-based RoBERTa embedding per user by:
        - taking up to `max_tweets_per_user` recent tweets per user,
        - concatenating them into a single string per user,
        - embedding that big single string with the text model.

        Returns:
        torch.Tensor of shape (num_users, hidden_dim)
        """
        if self.users is None:
            raise RuntimeError("Call load_users() before building tweet embeddings.")

        filename = (
            f"user_{self.max_tweets_per_user}_tweets_embeddings.pt"
            if self.max_tweets_per_user
            else "user_all_tweets_embeddings.pt"
        )
        path = os.path.join(self.out, filename)

        if os.path.exists(path):
            logger.info("Loading cached user tweet embeddings...")
            tweet_vectors = torch.load(path, map_location="cpu").to(self.device)
            logger.success(f"Loaded {filename}")

            return tweet_vectors

        logger.info("Aggregating tweets per user in DuckDB...")

        if self.max_tweets_per_user:
            query = f"""
                WITH limited_tweets AS (
                    SELECT
                        author_id,
                        text,
                        ROW_NUMBER() OVER (
                            PARTITION BY author_id
                            ORDER BY created_at DESC
                        ) AS rn
                    FROM tweets
                    WHERE text IS NOT NULL
                )
                SELECT
                    u.id AS user_id,
                    string_agg(lt.text, ' ') AS agg_text
                FROM users u
                LEFT JOIN limited_tweets lt
                    ON ('u' || lt.author_id) = u.id
                  AND lt.rn <= {self.max_tweets_per_user}
                GROUP BY u.id
                ORDER BY u.id;
            """
        else:
            query = """
                WITH limited_tweets AS (
                    SELECT author_id, text
                    FROM tweets
                    WHERE text IS NOT NULL
                )
                SELECT
                    u.id AS user_id,
                    string_agg(lt.text, ' ') AS agg_text
                FROM users u
                LEFT JOIN limited_tweets lt
                    ON ('u' || lt.author_id) = u.id
                GROUP BY u.id
                ORDER BY u.id;
            """

        tweet_df = self.con.execute(query).df()

        # verify or make sure order matches self.users["id"]
        if not (tweet_df["user_id"].tolist() == self.users["id"].tolist()):
            logger.warning(
                "User ID order mismatch between aggregated tweet_df and users df. "
                "Re-aligning via merge, this is slightly slower."
            )
            tweet_df = (
                self.users[["id"]]
                .merge(tweet_df, left_on="id", right_on="user_id", how="left")
                .sort_values("id")
            )

        tweet_texts = tweet_df["agg_text"].fillna("").astype(str).tolist()
        logger.info("Embedding aggregated tweet text per user (batched)...")
        tweet_vectors = self._embed_text_batch(tweet_texts, show_progress=True)

        torch.save(tweet_vectors.cpu(), path)
        logger.success(f"Saved {filename}")
        return tweet_vectors

    def build_numeric_properties(self) -> torch.Tensor:
        """
        Build standardized numeric user features.

        Base features (always included):
            - followers_count
            - following_count
            - tweet_count
            - active_days (account age in days relative to reference date)
            - screen_name_len

        Optional feature (enabled with self.build_like_count=True):
            - unique_like_count (number of distinct tweets the user has liked)

        All features are z-scored independently.

        Returns:
            torch.Tensor of shape:
                (num_users, 5)  if self.build_like_count=False
                (num_users, 6)  if self.build_like_count=True
        """

        if self.users is None:
            raise RuntimeError("Call load_users() before building numeric features.")

        filename = (
            "user_numeric_features_with_like_count.pt"
            if self.build_like_count
            else "user_numeric_features.pt"
        )
        path = os.path.join(self.out, filename)

        if os.path.exists(path):
            logger.info("Loading cached numeric properties...")
            numeric_properties = torch.load(path, map_location="cpu").to(self.device)
            logger.success(f"Loaded {filename}")
            return numeric_properties

        logger.info("Processing user numeric properties (vectorized)...")

        metrics = pd.json_normalize(self.users["public_metrics"])
        followers = metrics.get("followers_count", 0).fillna(0).astype("float32")
        following = metrics.get("following_count", 0).fillna(0).astype("float32")
        statuses = metrics.get("tweet_count", 0).fillna(0).astype("float32")

        created_at = pd.to_datetime(self.users["created_at"], errors="coerce")
        ref_date = dt.datetime(2020, 9, 5, tzinfo=dt.timezone.utc)
        active_days_raw = (ref_date - created_at).dt.days
        active_days_raw = active_days_raw.fillna(0).astype("float32")

        screen_name_len_raw = (
            self.users["name"].fillna("").astype(str).apply(len).astype("float32")
        )

        def _zscore(series: pd.Series) -> pd.Series:
            mu = series.mean()
            sigma = series.std() + 1e-8
            return (series - mu) / sigma

        feature_list = [
            _zscore(followers),
            _zscore(following),
            _zscore(statuses),
            _zscore(active_days_raw),
            _zscore(screen_name_len_raw),
        ]

        if self.build_like_count:
            likes_df = self.con.execute(
                """
              SELECT
                  source_id AS id,
                  COUNT(DISTINCT target_id) AS like_count
              FROM edges
              WHERE relation = 'like'
              GROUP BY source_id
          """
            ).df()

            users_with_likes = self.users.merge(likes_df, on="id", how="left")
            like_count_raw = users_with_likes["like_count"].fillna(0).astype("float32")

            feature_list.append(_zscore(like_count_raw))

        numeric_np = pd.concat(feature_list, axis=1).to_numpy("float32")
        numeric_properties = torch.from_numpy(numeric_np).to(self.device)

        torch.save(numeric_properties.cpu(), path)
        logger.success(f"Saved {filename}")

        return numeric_properties

    def build_categorical_properties(self) -> torch.Tensor:
        """
        Build categorical features (protected, verified, default_profile_image).

        Returns:
        torch.Tensor of shape (num_users, 3)
        """
        if self.users is None:
            raise RuntimeError(
                "Call load_users() before building categorical features."
            )

        path = os.path.join(self.out, "user_cat_features.pt")
        if os.path.exists(path):
            logger.info("Loading cached categorical properties...")
            category_properties = torch.load(path, map_location="cpu").to(self.device)
            logger.success("Loaded user_cat_features.pt")
            return category_properties

        logger.info("Processing user categorical properties (vectorized)...")

        protected = self.users["protected"].fillna(False).astype(bool).astype("float32")
        verified = self.users["verified"].fillna(False).astype(bool).astype("float32")

        # Default profile image heuristic
        default_profile_image = []
        for url in self.users["profile_image_url"]:
            if url is None:
                default_profile_image.append(1.0)
            else:
                url = str(url)
                if (
                    url.strip() == ""
                    or url
                    == "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"
                ):
                    default_profile_image.append(1.0)
                else:
                    default_profile_image.append(0.0)

        default_profile_image = pd.Series(default_profile_image, dtype="float32")

        cat_np = pd.concat(
            [protected, verified, default_profile_image], axis=1
        ).to_numpy(dtype="float32")

        category_properties = torch.from_numpy(cat_np).to(self.device)

        torch.save(category_properties.cpu(), path)
        logger.success("Saved user_cat_features.pt")
        return category_properties

    def build_node_embeddings(self) -> torch.Tensor:
        """
        Combine all feature blocks into a unified 128-dim node embedding per user.

        Returns:
            x: (num_users, embedding_dim) tensor on self.device
        """
        if self.users is None:
            self.load_users()

        desc = self.build_user_desc_embeddings()  # (N, hidden_dim)
        tweet = self.build_user_tweet_embeddings()  # (N, hidden_dim)
        num_prop = self.build_numeric_properties()  # (N, 5)
        cat_prop = self.build_categorical_properties()  # (N, 3)

        # Lazy init
        if self.linear_relu_desc is None:
            logger.info("Initializing projection layers...")
            self._init_projection_layers(
                desc_size=desc.size(1),
                tweet_size=tweet.size(1),
                num_prop_size=num_prop.size(1),
                cat_prop_size=cat_prop.size(1),
            )

        d = self.linear_relu_desc(desc)
        t = self.linear_relu_tweets(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)

        x = torch.cat([d, t, n, c], dim=1)
        x = self.linear_relu_input(x)

        out_path = os.path.join(self.out, "user_node_embeddings_128.pt")
        torch.save(x.cpu(), out_path)
        logger.success(f"Saved {out_path}")

        return x

    def build_labels(self):
        """
        Build aligned label vector: (N,)

        Label mapping: human (0), bot (1)
        """

        if self.users is None:
            raise RuntimeError(
                "Call load_users() before building description embeddings."
            )

        # Load bot_labels as dataframe
        path = os.path.join(self.out, "labels.pt")
        if os.path.exists(path):
            logger.info("Loading cached labels.pt...")
            labels_tensor = torch.load(path, map_location="cpu")
            logger.success("Loaded labels.pt")
            return labels_tensor

        logger.info("Loading bot_labels table from DuckDB...")
        self.labels = self.con.execute(
            """
            SELECT id, label
            FROM bot_labels
        """
        ).df()
        logger.success(f"Loaded {len(self.labels)} label rows from bot_labels")

        # Build lookup dictionary: user_id -> label_string
        label_dict = dict(zip(self.labels["id"], self.labels["label"]))

        mapped = []

        logger.info("Mapping labels to users in ORDER BY id ...")

        for uid in self.users["id"]:
            lbl = label_dict.get(uid, None)

            if lbl == "bot":
                mapped.append(1)

            elif lbl == "human":
                mapped.append(0)

            else:
                raise RuntimeError(f"Invalid label value received: {lbl}.")

        labels_tensor = torch.tensor(mapped, dtype=torch.long)

        torch.save(labels_tensor, os.path.join(self.out, "labels.pt"))

        logger.success("Saved labels.pt")

        return labels_tensor
