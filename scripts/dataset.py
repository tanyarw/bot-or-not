from pathlib import Path
import sys
import os

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH")).expanduser().resolve()
DATASET_ROOT = Path(os.getenv("DATASET_ROOT")).expanduser().resolve()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import load_config
from metadata import write_metadata
from src.features_builder.builder import Twibot22DataBuilder
from src.static_user_graph import StaticUserGraph
from src.utils.cache import save_tensor


def main():
    cfg = load_config("config.yaml")

    cfg["paths"]["db_path"] = str(DB_PATH)
    cfg["paths"]["out_dir"] = str(DATASET_ROOT)

    builder = Twibot22DataBuilder(cfg)
    fused, labels = builder.build_all()

    write_metadata(cfg, DATASET_ROOT)

    logger.success("Finished building TwiBot22 dataset.")
    logger.success(f"   - Node embeddings shape:{fused.shape}")
    logger.success(f"   - Labels shape:{labels.shape}")

    logger.info("Building Static User Graph")

    graph_builder = StaticUserGraph(
        db_path=DB_PATH,
        user_ids_path=DATASET_ROOT / "user_ids.pt",
        node_embeddings_path=DATASET_ROOT / "user_node_embeddings_128_v2.0.0.pt",
        labels_path=DATASET_ROOT / "labels.pt",
    )

    data = graph_builder.build_pyg_graph()

    path = DATASET_ROOT / "static_user_graph.pt"
    save_tensor(data, path)

    logger.success(f"Saved PyG graph → {path}")
    logger.success(f" Nodes: {data.num_nodes}")
    logger.success(f" Edges: {data.edge_index.size(1)}")
    logger.success(f" Features: {data.x.shape}")


if __name__ == "__main__":
    main()
