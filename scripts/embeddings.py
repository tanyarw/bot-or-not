import sys
import os
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedding_builder import Twibot22EmbeddingBuilder

if __name__ == "__main__":
    builder = Twibot22EmbeddingBuilder(
        db_path="db/twitter_graph.duckdb", out="./dataset/"
    )

    logger.info("Loading tables...")
    builder.load_users()

    logger.info("Building node embeddings...")
    node_emb = builder.build_node_embeddings()

    logger.success(f"Node embeddings shape: {tuple(node_emb.shape)}")
