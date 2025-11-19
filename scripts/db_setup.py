import duckdb
import os

db_path = os.path.join(os.getcwd(), 'db', 'twitter_graph.duckdb')
conn = duckdb.connect(db_path)

conn.execute("CREATE TABLE IF NOT EXISTS users AS SELECT * FROM read_json_auto('./data/static/sampled_users.jsonl')")
conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON users(id)")
print("Users table created")

conn.execute("CREATE TABLE IF NOT EXISTS tweets AS SELECT * FROM read_json_auto('./data/static/sampled_tweets.jsonl', union_by_name=true, sample_size=-1)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_tweet_id ON tweets(id)")
print("Tweets table created")

conn.execute("CREATE TABLE IF NOT EXISTS edges AS SELECT * FROM read_csv_auto('./data/static/sampled_edges.csv')")
conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id)")
print("Edges table created")

conn.execute("CREATE TABLE IF NOT EXISTS bot_labels AS SELECT * FROM read_csv_auto('./data/static/bot_labels.csv')")
conn.execute("CREATE INDEX IF NOT EXISTS idx_bot_node ON bot_labels(id)")
print("Bot labels table created")

conn.close()
print("DB setup completed")