import pandas as pd
import orjson
import pickle
import os
from tqdm.auto import tqdm


def load_users(filepath):
    users = []
    total_users = 21359 # Pre-known total number of users
    with open(filepath) as f:
        for line in tqdm(f, desc='Loading users', total=total_users):
            user = orjson.loads(line)
            users.append({
                'id': user['id'],
                'created_at': pd.to_datetime(user['created_at']),
                'features': user
            })
    return pd.DataFrame(users)


def load_tweets_meta(filepath):
    tweets = []
    total_tweets = 8369615
    
    with open(filepath, 'rb') as f:
        for line in tqdm(f, desc='Loading tweets', total=total_tweets):
            tweet = orjson.loads(line)
            tweets.append({
                'id': tweet['id'],
                'author_id': f"u{tweet['author_id']}",
                'created_at': tweet['created_at']
            })
    
    df = pd.DataFrame(tweets)
    df['created_at'] = pd.to_datetime(df['created_at'])
    return df


def create_snapshots(users_df: pd.DataFrame, tweets_meta_df: pd.DataFrame, edges_df: pd.DataFrame, output_dir: str = None):
    # Mapping from IDs to creation times
    user_times = users_df.set_index('id')['created_at'].to_dict() # Format: {user_id: created_at}
    tweet_times = tweets_meta_df.set_index('id')['created_at'].to_dict()

    edges_df = edges_df[edges_df['relation'].isin(['following', 'followers', 'post', 'like', 'retweeted', 'quoted', 'replied', 'mentioned', 'pinned'])].copy()

    # Calculate start times for edges
    tqdm.pandas(desc="Calculating active edges")
    edges_df['active_from'] = edges_df.progress_apply(
        lambda row: max(
            user_times.get(row['source_id']) or tweet_times.get(row['source_id']),
            user_times.get(row['target_id']) or tweet_times.get(row['target_id'])
        ), axis=1
    )

    min_time = users_df['created_at'].min()
    max_time = max(users_df['created_at'].max(), tweets_meta_df['created_at'].max())
    timestamps = pd.date_range(start=min_time, end=max_time, freq='W')

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    snapshots_path = os.path.join(output_dir, 'snapshots.pkl')
    with open(snapshots_path, 'wb') as f:
        for ts in tqdm(timestamps, desc='Snapshots', total=len(timestamps)):
            active_users = users_df[users_df['created_at'] <= ts]['id'].tolist()
            active_users_set = set(active_users)

            active_tweets = tweets_meta_df[(tweets_meta_df['created_at'] <= ts) & (tweets_meta_df['author_id'].isin(active_users_set))]['id'].tolist()

            active_edges = edges_df[(edges_df['active_from'] <= ts)][['source_id', 'relation', 'target_id']].values.tolist()

            snapshot = {
                'timestamp': ts,
                'user_ids': active_users,
                'tweet_ids': active_tweets,
                'edges': active_edges,
                'num_users': len(active_users),
                'num_tweets': len(active_tweets),
                'num_edges': len(active_edges)
            }
            pickle.dump(snapshot, f)

    return len(timestamps)


def save_graph(num_snapshots: int, users_df: pd.DataFrame, tweets_meta_df: pd.DataFrame, output_dir: str, tweets_jsonl_path: str):
    output_dir = os.path.abspath(output_dir)

    # Saving all user features, format: [id -> feature dict]
    user_features_path = os.path.join(output_dir, 'user_features.pkl')
    user_features = users_df.set_index('id')['features'].to_dict()
    with open(user_features_path, 'wb') as f:
        pickle.dump(user_features, f)
    
    # Save all tweet features, format: [id -> feature dict]
    total_tweets = 8369615 # Pre-known total number of tweets
    tweet_features_path = os.path.join(output_dir, 'tweet_features.pkl')
    tweet_features = {}
    with open(tweets_jsonl_path) as f:
        for line in tqdm(f, desc='Saving Tweets', total=total_tweets):
            tweet = orjson.loads(line)
            tweet_features[tweet['id']] = tweet
            
            # Saving in batches
            if len(tweet_features) >= 100000:
                with open(tweet_features_path, 'ab') as out:
                    pickle.dump(tweet_features, out)
                tweet_features = {}

    # Save remaining tweets
    if tweet_features:
        with open(tweet_features_path, 'ab') as out:
            pickle.dump(tweet_features, out)    

    # Log summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'Total snapshots: {num_snapshots}\n')
        f.write(f'Total users: {len(users_df)}\n')
        f.write(f'Total tweets: {len(tweets_meta_df)}\n')



def static_to_temporal(input_dir, output_dir):
    input_dir = os.path.abspath(input_dir)

    users_jsonl_path = os.path.join(input_dir, 'sampled_users.jsonl')
    tweets_jsonl_path = os.path.join(input_dir, 'sampled_tweets.jsonl')
    edges_csv_path = os.path.join(input_dir, 'sampled_edges.csv')
    
    users_df = load_users(users_jsonl_path)
    tweets_meta_df = load_tweets_meta(tweets_jsonl_path)
    edges_df = pd.read_csv(edges_csv_path)
    
    num_snapshots = create_snapshots(users_df, tweets_meta_df, edges_df, output_dir)
    save_graph(num_snapshots, users_df, tweets_meta_df, output_dir, tweets_jsonl_path)
    
    return num_snapshots