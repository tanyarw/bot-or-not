import sys
import os
from loguru import logger

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.databuilder import Twibot22DataBuilder

if __name__ == "__main__":

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        pipeline_device = 0
        logger.debug("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        pipeline_device = "mps"
        logger.debug("Using Apple MPS backend")
    else:
        device = "cpu"
        pipeline_device = -1
        logger.debug("Using CPU")

    builder = Twibot22DataBuilder(
        version_name="v04",
        device=device,
        pipeline_device=pipeline_device,
        db_path="db/twitter_graph.duckdb",
        out="./dataset/",
        max_tweets_per_user=None,
        build_like_count=True,
    )

    # Run only if embeddings do not pre-exist
    # builder.load_users()

    node_emb = builder.build_node_embeddings()
    logger.success(f"Node embeddings shape: {tuple(node_emb.shape)}")

    labels = builder.build_labels()
    logger.success(f"Labels shape: {tuple(node_emb.shape)}")

    builder.con.close()
