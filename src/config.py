from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# Load configurations from .env
DATA_DIR = os.path.join(os.getcwd(), os.getenv("DATA_DIR", "data"))
DB_PATH = os.path.join(os.getcwd(), os.getenv("DB_PATH", "db"), "twitter_graph.duckdb")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0001"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
NUM_NEIGHBORS = [int(x) for x in os.getenv("NUM_NEIGHBORS", "10,5").split(",")]

# Data split ratios
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.6"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.2"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.2"))

# Model architecture
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "128"))
HIDDEN_CHANNELS = int(os.getenv("HIDDEN_CHANNELS", "128"))
DROPOUT = float(os.getenv("DROPOUT", "0.5"))

# Early stopping
EARLY_STOPPING_PATIENCE = int(os.getenv("PATIENCE", "5"))
MIN_DELTA = float(os.getenv("MIN_DELTA", "0.001"))

# Logging and saving
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "10"))
SAVE_MODEL = os.getenv("SAVE_MODEL", "True").lower() == "true"
MODEL_SAVE_PATH = os.path.join(os.getcwd(), os.getenv("MODEL_SAVE_PATH", "checkpoints"))

# Snapshot processing
MIN_USERS_PER_SNAPSHOT = int(os.getenv("MIN_USERS_PER_SNAPSHOT", "10"))
SKIP_SMALL_SNAPSHOTS = os.getenv("SKIP_SMALL_SNAPSHOTS", "True").lower() == "true"