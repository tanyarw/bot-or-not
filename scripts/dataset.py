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


def main():
    cfg = load_config("config.yaml")

    cfg["paths"]["db_path"] = str(DB_PATH)
    cfg["paths"]["out_dir"] = str(DATASET_ROOT)

    builder = Twibot22DataBuilder(cfg)
    if True:
        snapshots_path = DATASET_ROOT / "temporalized" / "snapshots.pkl"
        builder.build_temporalized_embeddings(snapshots_path, cfg["embedding"]["max_snapshots"])
        return
    fused, labels = builder.build_all()

    write_metadata(cfg, DATASET_ROOT)

    logger.success("Finished building TwiBot22 dataset.")
    logger.success(f"   - Node embeddings shape:{fused.shape}")
    logger.success(f"   - Labels shape:{labels.shape}")


if __name__ == "__main__":
    main()
