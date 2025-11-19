"""
DuckDB Graph Loader Script

This script expects a `.env` file in the project root containing:

    DATA_ROOT=/absolute/path/to/your/data

`DATA_ROOT` must point to the directory that contains the `static/` and `temporal/` folder, e.g.:

    DATA_ROOT=~/OneDrive/data

Directory structure should look like:

    DATA_ROOT/
        static/
            sampled_users.jsonl
            sampled_tweets.jsonl
            sampled_edges.csv
            bot_labels.csv
        temporal/

The script will:
- Load DATA_ROOT from .env
- Read all JSONL/CSV files from DATA_ROOT/static
- Create a local DuckDB database at ./db/twitter_graph.duckdb
- Create tables: users, tweets, edges, bot_labels
- Create useful indexes for fast lookup

Make sure you have a valid `.env` file before running this script.
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import duckdb

load_dotenv()

DATA_ROOT = Path(os.getenv("DATA_ROOT")).expanduser().resolve()
print("DATA_ROOT =", DATA_ROOT)

users_path = DATA_ROOT / "static" / "sampled_users.jsonl"
tweets_path = DATA_ROOT / "static" / "sampled_tweets.jsonl"
edges_path = DATA_ROOT / "static" / "sampled_edges.csv"
labels_path = DATA_ROOT / "static" / "bot_labels.csv"

os.makedirs("db", exist_ok=True)

db_path = os.path.join(os.getcwd(), 'db', 'twitter_graph.duckdb')
conn = duckdb.connect(db_path)

print("Database will be stored at:", db_path)

conn = duckdb.connect(db_path)

# USERS
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS users AS
    SELECT * FROM read_json_auto('{users_path}')
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON users(id)")
print("Users table created")

# TWEETS
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS tweets AS
    SELECT * FROM read_json_auto('{tweets_path}', union_by_name=true, sample_size=-1)
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_tweet_id ON tweets(id)")
print("Tweets table created")

# EDGES
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS edges AS
    SELECT * FROM read_csv_auto('{edges_path}')
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id)")
print("Edges table created")

# BOT LABELS
conn.execute(
    f"""
    CREATE TABLE IF NOT EXISTS bot_labels AS
    SELECT * FROM read_csv_auto('{labels_path}')
"""
)
conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_node ON bot_labels(id)")
print("Bot labels table created")

conn.close()
print("DB setup completed")
