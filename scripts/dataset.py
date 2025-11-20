import sys
import os
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.databuilder import Twibot22DataBuilder

if __name__ == "__main__":
    builder = Twibot22DataBuilder(db_path="db/twitter_graph.duckdb", out="./dataset/")

    logger.info("Loading tables...")
    builder.load_users()

    logger.info("Building node embeddings...")
    node_emb = builder.build_node_embeddings()

    logger.success(f"Node embeddings shape: {tuple(node_emb.shape)}")

    logger.info("Building labels...")
    labels = builder.build_labels()

    logger.success(f"Labels shape: {tuple(node_emb.shape)}")
