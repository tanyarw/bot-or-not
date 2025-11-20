"""
DuckDB Graph Loader Script

This script expects a `.env` file in the project root containing:

    DATA_ROOT=/absolute/or/relative/path/to/data
    DB_PATH=/absolute/or/relative/path/to/db/twitter_graph.duckdb

Example:

    DATA_ROOT=~/OneDrive/data
    DB_PATH=./db/twitter_graph.duckdb

Directory structure under DATA_ROOT must be:

    DATA_ROOT/
        static/
            sampled_users.jsonl
            sampled_tweets.jsonl
            sampled_edges.csv
            bot_labels.csv
        temporal/
            (event-based JSONL/CSV files, optional)

What this script does:
- Loads DATA_ROOT and DB_PATH from the `.env` file
- Ensures the parent directory of DB_PATH exists
- Reads JSONL/CSV files from DATA_ROOT/static
- Creates (or overwrites) a DuckDB database at DB_PATH
- Creates tables: users, tweets, edges, bot_labels
- Adds useful indexes for efficient graph queries and joins

Make sure you have a valid `.env` file before running this script.
"""

from pathlib import Path
import os

import duckdb
from loguru import logger

from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = Path(os.getenv("DATA_ROOT")).expanduser().resolve()
logger.debug(f"DATA_ROOT = {DATA_ROOT}")

DB_PATH = Path(os.getenv("DB_PATH")).expanduser().resolve()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
logger.debug(f"DB_PATH = {DB_PATH}")

users_path = DATA_ROOT / "static" / "sampled_users.jsonl"
tweets_path = DATA_ROOT / "static" / "sampled_tweets.jsonl"
edges_path = DATA_ROOT / "static" / "sampled_edges.csv"
labels_path = DATA_ROOT / "static" / "bot_labels.csv"

conn = duckdb.connect(str(DB_PATH))

# USERS
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS users AS
    SELECT * FROM read_json_auto('{users_path}')
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON users(id)")
logger.success("Users table created!")

# TWEETS
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS tweets AS
    SELECT * FROM read_json_auto('{tweets_path}', union_by_name=true, sample_size=-1)
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_tweet_id ON tweets(id)")
logger.success("Tweets table created!")

# EDGES
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS edges AS
    SELECT * FROM read_csv_auto('{edges_path}')
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id)")
logger.success("Edges table created!")

# BOT LABELS
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS bot_labels AS
    SELECT * FROM read_csv_auto('{labels_path}')
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_node ON bot_labels(id)")
logger.success("Bot labels table created!")

conn.close()
logger.success("DB setup completed!")
