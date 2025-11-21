from pathlib import Path

from duckdb import DuckDBPyConnection
import pandas as pd
import torch
from loguru import logger


class LabelBuilder:
    """
    Build aligned user labels compatible with ordered users_df.
    Labels: human -> 0, bot -> 1
    """

    def __init__(self):
        pass

    def build(self, users_df: pd.DataFrame, con: DuckDBPyConnection, out_dir: Path):
        """
        Build or load label tensor aligned to users_df["id"] ordering.

        Args:
            users_df (pd.DataFrame): already ordered by id
            con (duckdb.Connection): database connection
            out_dir (Path): output directory for caching

        Returns:
            torch.Tensor (N,) with 0/1 labels
        """

        path = out_dir / "labels.pt"

        if path.exists():
            logger.info("Loading cached labels → labels.pt...")
            labels_tensor = torch.load(path, map_location="cpu")
            return labels_tensor

        logger.info("Loading bot_labels table from DuckDB...")
        labels_df = con.execute(
            """
            SELECT id, label
            FROM bot_labels
            """
        ).df()
        logger.success(f"Loaded {len(labels_df)} rows from bot_labels")

        label_dict = dict(zip(labels_df["id"], labels_df["label"]))
        mapped_labels = []

        logger.info("Mapping labels to users in ORDER BY id...")

        for uid in users_df["id"]:
            lbl = label_dict.get(uid, None)

            if lbl == "bot":
                mapped_labels.append(1)
            elif lbl == "human":
                mapped_labels.append(0)
            else:
                raise RuntimeError(f"Invalid or missing label for user {uid}: {lbl}")

        labels_tensor = torch.tensor(mapped_labels, dtype=torch.long)

        torch.save(labels_tensor, path)
        logger.success("Saved labels.pt")

        return labels_tensor
