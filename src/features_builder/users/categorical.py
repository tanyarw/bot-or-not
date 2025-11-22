from pathlib import Path
import torch
import pandas as pd
from loguru import logger

from src.utils.cache import save_tensor, load_tensor


class CategoricalFeatureBuilder:
    """
    Build categorical features:
        - protected (0/1)
        - verified (0/1)
        - default_profile_image (0/1)

    Output:
        Tensor shape (N, 3)
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def build(self, users_df: pd.DataFrame, device: str):
        """
        Args:
            users_df: DataFrame ordered by id

        Returns:
            torch.Tensor of shape (num_users, 3)
        """
        out_path = self.out_dir / "categorical_features.pt"

        if out_path.exists():
            logger.info("Loading cached categorical features → categorical_features.pt")
            return load_tensor(out_path, device=device)

        logger.info("Building categorical user features...")

        protected = users_df["protected"].fillna(False).astype(bool).astype("float32")

        verified = users_df["verified"].fillna(False).astype(bool).astype("float32")

        default_img = "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"

        default_image_flags = []
        for url in users_df["profile_image_url"]:
            if url is None:
                default_image_flags.append(1.0)
            else:
                url = str(url)
                if url.strip() == "" or url == default_img:
                    default_image_flags.append(1.0)
                else:
                    default_image_flags.append(0.0)

        default_image_series = pd.Series(default_image_flags, dtype="float32")

        cat_np = pd.concat(
            [protected, verified, default_image_series], axis=1
        ).to_numpy("float32")

        cat_tensor = torch.from_numpy(cat_np)

        save_tensor(cat_tensor, out_path)
        logger.success(f"Saved categorical_features.pt → {out_path}")

        return cat_tensor
