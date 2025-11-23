import torch
import torch.nn as nn
from loguru import logger

from src.utils.cache import save_tensor, load_tensor


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
    def fuse(self, desc, tweet, numeric, categorical, device):
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
