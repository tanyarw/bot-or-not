import datetime as dt
from pathlib import Path

from duckdb import DuckDBPyConnection
import torch
import pandas as pd
from loguru import logger

from src.utils.cache import save_tensor, load_tensor


class NumericFeatureBuilder:
    """
    Builds standardized numeric user features.

    Base features (always included):
        - followers_count
        - following_count
        - active_days
        - screen_name_len
        - like_count

    Optional (controlled by cfg["features"]):
        - tweet_count


    Returns:
        torch.Tensor of shape (N, D)
    """

    def __init__(self, con: DuckDBPyConnection, out_dir: Path, cfg):
        self.con = con
        self.out_dir = out_dir
        self.cfg = cfg

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        mu = series.mean()
        sigma = series.std() + 1e-8
        return (series - mu) / sigma

    def build(self, users_df: pd.DataFrame):
        """
        Args:
            users_df: DataFrame already ordered by id.

        Returns:
            numeric_features: torch.Tensor on CPU
        """

        # Build filename based on active features
        parts = ["numeric_features"]

        include_tweet = self.cfg["features"]["include_tweet_in_users"]

        if include_tweet:
            parts.append("tweetcount")

        fname = "_".join(parts) + ".pt"
        out_path = self.out_dir / fname

        if out_path.exists():
            logger.info(f"Loading cached numeric features → {fname}")
            return load_tensor(out_path)

        logger.info("Processing numeric user features...")

        metrics = pd.json_normalize(users_df["public_metrics"])

        followers = metrics.get("followers_count", 0).fillna(0).astype("float32")
        following = metrics.get("following_count", 0).fillna(0).astype("float32")

        # account age
        created_at = pd.to_datetime(users_df["created_at"], errors="coerce")
        ref_date = dt.datetime(2020, 9, 5, tzinfo=dt.timezone.utc)
        active_days = (ref_date - created_at).dt.days.fillna(0).astype("float32")

        # screen name length
        screen_name_len = (
            users_df["name"].fillna("").astype(str).apply(len).astype("float32")
        )

        logger.info("Loading like-counts from edges table...")
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
        merged = users_df.merge(likes_df, on="id", how="left")
        like_count = merged["like_count"].fillna(0).astype("float32")

        feature_list = [
            self._zscore(followers),
            self._zscore(following),
            self._zscore(active_days),
            self._zscore(screen_name_len),
            self._zscore(like_count),
        ]

        if include_tweet:
            tweet_count = metrics.get("tweet_count", 0).fillna(0).astype("float32")
            feature_list.append(self._zscore(tweet_count))
        else:
            logger.debug("Skipping tweet count.")

        numeric_np = pd.concat(feature_list, axis=1).to_numpy("float32")
        numeric_tensor = torch.from_numpy(numeric_np)

        save_tensor(numeric_tensor, out_path)
        logger.success(f"Saved numeric features → {out_path}")

        return numeric_tensor
