import datetime as dt
import os

import duckdb
from loguru import logger
import torch
import torch.nn as nn
from transformers import pipeline
from tqdm import tqdm
import pandas as pd


class Twibot22EmbeddingBuilder:
    def __init__(
        self, db_path="db/twitter_graph.duckdb", out="./dataset/", load_model=True
    ):
        self.out = out
        os.makedirs(out, exist_ok=True)

        self.con = duckdb.connect(db_path)

        self.users = None
        self.tweets = None
        self.embedding_dim = 128
        self.linear_relu_desc = None
        self.linear_relu_text = None
        self.linear_relu_num_prop = None
        self.linear_relu_cat_prop = None
        self.linear_relu_input = None

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

        if load_model:
            logger.info("Loading tokenizer/model...")
            self.model = pipeline(
                "feature-extraction",
                model="roberta-base",
                tokenizer="roberta-base",
                device=pipeline_device,
            )
        else:
            logger.info("Skipping model load (test mode).")
            self.model = None

    def _init_projection_layers(
        self, desc_size, text_size, num_prop_size, cat_prop_size
    ):
        """Initialize projection layers after loading feature sizes."""

        quarter = self.embedding_dim // 4

        self.linear_relu_desc = nn.Sequential(
            nn.Linear(desc_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_text = nn.Sequential(
            nn.Linear(text_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, quarter), nn.LeakyReLU()
        ).to(self.device)

        self.linear_relu_input = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), nn.LeakyReLU()
        ).to(self.device)

    def load_users(self):
        """Load users from DuckDB."""

        logger.info("Loading user tables from DuckDB...")

        df = self.con.execute(
            """
            SELECT *
            FROM users;
        """
        ).df()

        self.users = df
        logger.success("Users loaded:", len(df))

    def load_tweets(self):
        """Load tweets from DuckDB."""

        logger.info("Loading tweet tables...")
        df = self.con.execute(
            """
            SELECT *
            FROM tweets;
        """
        ).df()

        self.tweets = df
        logger.success("Tweets loaded:", len(df))

    def embed_text(self, text):
        """Average pooling of transformer token outputs."""
        if self.model is None:
            logger.warning("Text embedding model is not initialized.")
            return torch.zeros(768)

        if text is None or text.strip() == "":
            return torch.zeros(768)

        out = torch.tensor(self.model(text))  # (1, seq_len, 768)
        emb = out.mean(dim=1).squeeze(0)  # (768,)

        return emb

    def build_user_desc_embeddings(self):
        """User description embeddings."""

        if self.users is None:
            raise RuntimeError(
                "Call load_users() before building description embeddings."
            )

        logger.info("Encoding user descriptions...")
        path = os.path.join(self.out, "user_desc_embeddings.pt")

        if os.path.exists(path):
            desc_vectors = torch.load(path, map_location="cpu").to(self.device)
            logger.success("Loaded cached user description embeddings.")
            return desc_vectors

        desc_vectors = []

        for txt in tqdm(self.users["description"]):
            desc_vectors.append(self.embed_text(txt))

        desc_vectors = torch.stack(desc_vectors)
        torch.save(desc_vectors.cpu(), path)
        logger.success("Saved user_desc_embeddings.pt")

        return desc_vectors

    def build_tweet_text_embeddings(self):
        """Tweet text embeddings."""

        if self.tweets is None:
            raise RuntimeError("Call load_tweets() before building tweet embeddings.")

        logger.info("Processing tweets texts...")
        path = os.path.join(self.out, "tweet_text_embeddings.pt")

        if os.path.exists(path):
            tweet_vecs = torch.load(path, map_location="cpu").to(self.device)
            logger.success("Loaded cached tweet text embeddings.")
            return tweet_vecs

        tweet_vecs = []

        for txt in tqdm(self.tweets["text"]):
            tweet_vecs.append(self.embed_text(txt))

        tweet_vecs = torch.stack(tweet_vecs)

        torch.save(tweet_vecs.cpu(), path)
        logger.success("Saved tweet_text_embeddings.pt")

        return tweet_vecs

    def build_numeric_properties(self):

        if self.users is None:
            raise RuntimeError(
                "Call load_users() before building description embeddings."
            )

        logger.info("Processing user numeric properties...")
        path = os.path.join(self.out, "user_numeric_features.pt")

        if os.path.exists(path):
            numeric_properties = torch.load(path, map_location="cpu").to(self.device)
            logger.success("Loaded cached numeric properties.")
            return numeric_properties

        metrics = pd.json_normalize(self.users["public_metrics"])
        followers = metrics["followers_count"].fillna(0).astype("float32")
        followers = (followers - followers.mean()) / (followers.std() + 1e-8)
        followers = torch.tensor(followers.to_numpy(), dtype=torch.float32)

        following_raw = []
        for m in self.users["public_metrics"]:
            if isinstance(m, dict) and m.get("following_count") is not None:
                following_raw.append(int(m["following_count"]))
            else:
                following_raw.append(0)
        following = pd.Series(following_raw, dtype="float32")
        following = (following - following.mean()) / (following.std() + 1e-8)
        following = torch.tensor(following.to_numpy(), dtype=torch.float32)

        created_at = pd.to_datetime(self.users["created_at"], unit="s", errors="coerce")
        date0 = dt.strptime("Tue Sep 5 00:00:00 +0000 2020 ", "%a %b %d %X %z %Y ")
        active_days_raw = (date0 - created_at).dt.days
        active_days_raw = active_days_raw.fillna(0).astype("float32")
        active_days = (active_days_raw - active_days_raw.mean()) / (
            active_days_raw.std() + 1e-8
        )
        active_days = torch.tensor(active_days.to_numpy(), dtype=torch.float32)

        screen_name_len_raw = (
            self.users["name"].fillna("").apply(lambda x: len(str(x))).astype("float32")
        )
        screen_name_len = (screen_name_len_raw - screen_name_len_raw.mean()) / (
            screen_name_len_raw.std() + 1e-8
        )
        screen_name_len = torch.tensor(screen_name_len.to_numpy(), dtype=torch.float32)

        statuses_raw = []
        for m in self.users["public_metrics"]:
            if isinstance(m, dict) and m.get("tweet_count") is not None:
                statuses_raw.append(int(m["tweet_count"]))
            else:
                statuses_raw.append(0)
        statuses = pd.Series(statuses_raw, dtype="float32")
        statuses = (statuses - statuses.mean()) / (statuses.std() + 1e-8)
        statuses = torch.tensor(statuses.to_numpy(), dtype=torch.float32)

        numeric_properties = torch.cat(
            [
                followers,
                following,
                active_days,
                screen_name_len,
                statuses,
            ],
            dim=1,
        ).to(self.device)

        torch.save(numeric_properties.cpu(), path)
        logger.success("Saved user_numeric_features.pt")

        return numeric_properties

    def cat_prop_preprocess(self):
        if self.users is None:
            raise RuntimeError(
                "Call load_users() before building description embeddings."
            )

        logger.info("Processing user categorical properties...")
        path = os.path.join(self.out, "user_cat_features.pt")

        if os.path.exists(path):
            category_properties = torch.load(path, map_location="cpu").to(self.device)
            logger.success("Loaded cached categorical properties.")
            return category_properties

        protected_list = (
            self.users["protected"]
            .fillna(False)
            .astype(bool)
            .astype("float32")
            .to_list()
        )
        protected_tensor = torch.tensor(protected_list, dtype=torch.float32)

        verified_list = (
            self.users["verified"]
            .fillna(False)
            .astype(bool)
            .astype("float32")
            .to_list()
        )
        verified_tensor = torch.tensor(verified_list, dtype=torch.float32)

        default_profile_image = []
        for url in self.users["profile_image_url"]:
            if url is None:
                default_profile_image.append(1)
            else:
                url = str(url)
                if (
                    url
                    == "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"
                    or url.strip() == ""
                ):
                    default_profile_image.append(1)
                else:
                    default_profile_image.append(0)
        default_profile_image_tensor = torch.tensor(
            default_profile_image, dtype=torch.float32
        )

        category_properties = torch.cat(
            [protected_tensor, verified_tensor, default_profile_image_tensor], dim=1
        ).to(self.device)

        torch.save(category_properties.cpu(), path)
        logger.success("Saved user_categorical_features.pt")

        return category_properties

    def build_node_embeddings(self):
        """Combine all features into a unified 128-dim node embedding."""

        desc = self.build_user_desc_embeddings()
        text = self.build_tweet_text_embeddings()
        num_prop = self.build_numeric_properties()
        cat_prop = self.cat_prop_preprocess()

        if self.linear_relu_desc is None:
            self._init_projection_layers(
                desc_size=desc.size(1),
                text_size=text.size(1),
                num_prop_size=num_prop.size(1),
                cat_prop_size=cat_prop.size(1),
            )

        d = self.linear_relu_desc(desc)
        t = self.linear_relu_text(text)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)

        x = torch.cat((d, t, n, c), dim=1)

        x = self.linear_relu_input(x)

        out_path = os.path.join(self.out, "node_embeddings_128.pt")
        torch.save(x.cpu(), out_path)
        logger.success(f"Saved {out_path}")

        return x
