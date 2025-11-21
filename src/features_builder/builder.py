from pathlib import Path

from loguru import logger

from model.node_embeddings import NodeEmbeddingBuilder

from src.features_builder.labels import LabelBuilder
from src.features_builder.tweets import TweetBuilder
from src.features_builder.users.numeric import NumericFeatureBuilder
from src.features_builder.users.categorical import CategoricalFeatureBuilder
from src.features_builder.users.users import UserLoader

from src.utils.text_embed import TextEmbedder
from src.utils.cache import load_tensor, save_tensor
from src.utils.duck import connect


class Twibot22DataBuilder:
    """
    High-level orchestrator that:
        - Loads users
        - Builds textual, numeric, categorical features
        - Builds tweet-level embeddings
        - Fuses into final node embeddings
        - Builds labels aligned by ID
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # DB connection
        self.con = connect(cfg["paths"]["db_path"])

        # Output dir
        self.out_dir = Path(cfg["paths"]["out_dir"])
        self.out_dir.mkdir(exist_ok=True)

        # Text embedding model
        self.embedder = TextEmbedder(
            cfg["text"]["model_name"],
            cfg["text"]["batch_size"],
            cfg["text"]["pipeline_device"],
        )

        # Builders
        self.users = UserLoader(self.con, self.out_dir)
        self.tweets = TweetBuilder(self.con, self.out_dir, self.embedder, cfg)
        self.numeric = NumericFeatureBuilder(self.con, self.out_dir, cfg)
        self.categorical = CategoricalFeatureBuilder(self.out_dir)
        self.node_emb = NodeEmbeddingBuilder(cfg, self.out_dir, cfg["device"])
        self.labels = LabelBuilder()

    def build_all(self):
        user_df, _user_ids = self.users.load()

        desc_path = self.out_dir / "user_desc_embeddings.pt"
        if desc_path.exists():
            desc = load_tensor(desc_path)
            logger.info(
                "Loading cached user description embeddings → user_desc_embeddings.pt"
            )
        else:
            desc_texts = user_df["description"].fillna("").astype(str).tolist()
            desc = self.embedder.embed_batch(desc_texts, show_progress=True)
            save_tensor(desc.cpu(), desc_path)

        numeric = self.numeric.build(user_df)
        categorical = self.categorical.build(user_df)

        tweet = None

        k = self.cfg["features"]["max_tweets_per_user"]

        _tweet_ids, _tweet_embs = self.tweets.embed_k_user_tweets(k=k)
        if self.cfg["features"]["include_tweet_in_users"]:
            tweet = self.tweets.embed_recent_tweets_per_user(user_df, k=k)

        fused_user_embeddings = self.node_emb.fuse(
            desc=desc,
            tweet=tweet,
            numeric=numeric,
            categorical=categorical,
        )

        label_tensor = self.labels.build(user_df, self.con, self.out_dir)

        self.con.close()

        return fused_user_embeddings, label_tensor
