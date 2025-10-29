import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json as json_lib
from collections import defaultdict
import gc
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "./data"
OUTPUT_DIR = "./sampled_graph"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_USERS = 20000
SEED_RATIO = 0.10
HUB_RATIO = 0.70
BOT_RATIO_TARGET = 0.14
CHUNK_SIZE = 100000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_users():
    user_file = os.path.join(DATA_DIR, "user.json")
    label_file = os.path.join(DATA_DIR, "label.csv")
    
    # Load labels
    labels = pd.read_csv(label_file)
    label_dict = dict(zip(labels['id'], labels['label']))
    print(f"Loaded {len(label_dict):,} labels")
    
    users_list = []
    
    with open(user_file, 'r', encoding='utf-8') as f:
        users_data = json_lib.load(f)

    print(f"Processing {len(users_data):,} users")
    pbar = tqdm(
        total=len(users_data) // CHUNK_SIZE + 1,
        desc="  Processing users",
        unit="chunk"
    )
    
    for i, user in enumerate(users_data):
        try:
            user_id = user.get('id')
            
            if user_id not in label_dict:
                continue
            
            metrics = user.get('public_metrics', {})
            
            user_data = {
                'user_id': user_id,
                'created_at': user.get('created_at'),
                'followers_count': metrics.get('followers_count', 0),
                'following_count': metrics.get('following_count', 0),
                'label': label_dict[user_id]
            }
            
            users_list.append(user_data)
            
        except Exception as e:
            continue
        
        if (i + 1) % 100000 == 0:
            pbar.update(1)
    
    pbar.update(1)
    pbar.close()
    del users_data
    gc.collect()
    
    users_df = pd.DataFrame(users_list)
    users_df['created_at'] = pd.to_datetime(users_df['created_at'], errors='coerce')
    users_df = users_df.dropna(subset=['created_at'])
    
    print(f"Loaded {len(users_df):,} users")
    print(f"  Humans: {(users_df['label']=='human').sum():,}")
    print(f"  Bots: {(users_df['label']=='bot').sum():,}")
    
    del users_list
    gc.collect()
    
    return users_df


def calculate_degrees(users_df):   
    edge_file = os.path.join(DATA_DIR, "edge.csv")
    degree_count = defaultdict(int)
    
    print("Calculating degrees")
    
    pbar = tqdm(
        desc="  Processing edges",
        unit="chunk"
    )
    
    for chunk in pd.read_csv(edge_file, chunksize=CHUNK_SIZE):
        user_edges = chunk[chunk['relation'].isin(['following', 'followers'])]
        
        source_counts = user_edges['source_id'].value_counts()
        target_counts = user_edges['target_id'].value_counts()

        for user_id, count in source_counts.items():
            degree_count[user_id] += count
        for user_id, count in target_counts.items():
            degree_count[user_id] += count
        
        pbar.update(1)

    pbar.close()
    
    users_df['degree'] = users_df['user_id'].map(degree_count).fillna(0).astype(int)
    gc.collect()
    return users_df


def select_seeds(users_df):
    users_df['time_bin'] = pd.qcut(
        users_df['created_at'], 
        q=20, 
        labels=False, 
        duplicates='drop'
    )
    
    n_seeds = int(TARGET_USERS * SEED_RATIO)
    time_bins = sorted(users_df['time_bin'].dropna().unique())
    seeds_per_bin = max(1, n_seeds // len(time_bins))
    
    seeds = set()
    
    for bin_id in time_bins:
        bin_users = users_df[users_df['time_bin'] == bin_id]
        
        for label in ['human', 'bot']:
            label_users = bin_users[bin_users['label'] == label]
            
            if len(label_users) == 0:
                continue
            
            n = seeds_per_bin // 2
            n_hubs = int(n * HUB_RATIO)
            n_random = n - n_hubs
            
            hubs = label_users.nlargest(
                min(n_hubs, len(label_users)), 
                'degree'
            )['user_id'].values
            seeds.update(hubs)
            
            remaining = label_users[~label_users['user_id'].isin(hubs)]
            if len(remaining) > 0:
                random_sample = remaining.sample(
                    min(n_random, len(remaining)),
                    random_state=RANDOM_SEED
                )['user_id'].values
                seeds.update(random_sample)
    
    return seeds, users_df


def build_adjacency(relevant_users):
    adjacency = defaultdict(set)
    edge_file = os.path.join(DATA_DIR, "edge.csv")
    relevant_users = set(relevant_users)
    
    for chunk in pd.read_csv(edge_file, chunksize=CHUNK_SIZE):
        user_edges = chunk[
            chunk['relation'].isin(['following', 'followers']) &
            (chunk['source_id'].isin(relevant_users) | 
             chunk['target_id'].isin(relevant_users))
        ]
        
        for _, row in user_edges.iterrows():
            src, tgt = row['source_id'], row['target_id']
            adjacency[src].add(tgt)
            adjacency[tgt].add(src)
    
    return dict(adjacency)


def expand_sample(seeds, users_df):
    sampled = seeds.copy()
    adjacency = build_adjacency(sampled)
    
    user_meta = users_df.set_index('user_id')[['label', 'degree']].to_dict('index')
    candidate_pool = list(seeds)
    
    print("Expanding sample")
    iteration = 0
    while len(sampled) < TARGET_USERS and candidate_pool:
        current = candidate_pool.pop(0)
        
        neighbors = adjacency.get(current, set())
        unsampled = [n for n in neighbors if n not in sampled]
        
        if not unsampled:
            if len(sampled) % 5000 == 0:
                adjacency = build_adjacency(sampled)
            continue
        
        labels = [user_meta.get(u, {}).get('label', 'human') for u in sampled]
        bot_ratio = sum(1 for l in labels if l == 'bot') / len(labels)
        
        valid = []
        for n in unsampled:
            if n not in user_meta:
                continue
            is_bot = user_meta[n].get('label') == 'bot'
            
            if is_bot and bot_ratio > (BOT_RATIO_TARGET + 0.02):
                continue
            if not is_bot and bot_ratio < (BOT_RATIO_TARGET - 0.02):
                continue
            
            valid.append(n)
        
        if not valid:
            continue
        
        degrees = [user_meta.get(n, {}).get('degree', 1) for n in valid]
        total_deg = sum(degrees)
        probs = np.array(degrees) / total_deg if total_deg > 0 else None
        
        n_add = min(5, len(valid), TARGET_USERS - len(sampled))
        selected = np.random.choice(valid, size=n_add, replace=False, p=probs)
        
        sampled.update(selected)
        candidate_pool.extend(selected)
        
        iteration += 1
        if iteration % 1000 == 0:
            print(f"  Progress: {len(sampled):,}/{TARGET_USERS:,} users")
    
    return sampled


def extract_entities(sampled_user_ids):   
    edge_file = os.path.join(DATA_DIR, "edge.csv")
    
    tweets = set()
    lists = set()
    hashtags = set()
    all_edges_list = []
    
    print("Extracting entities")
    file_size = os.path.getsize(edge_file)
    estimated_chunks = int(file_size / (CHUNK_SIZE * 50))
    
    pbar = tqdm(
        total=estimated_chunks,
        desc="  Processing edges",
        unit="chunk",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]'
    )
    
    for chunk in pd.read_csv(edge_file, chunksize=CHUNK_SIZE):
        relevant = chunk[
            chunk['source_id'].isin(sampled_user_ids) |
            chunk['target_id'].isin(sampled_user_ids)
        ].copy()
        
        if len(relevant) == 0:
            pbar.update(1)
            continue
        
        for _, row in relevant.iterrows():
            rel = row['relation']
            src, tgt = row['source_id'], row['target_id']
            
            if rel in ['post', 'pinned', 'like']:
                if src in sampled_user_ids:
                    tweets.add(tgt)
            
            elif rel == 'mentioned':
                if tgt in sampled_user_ids:
                    tweets.add(src)
            
            elif rel in ['retweeted', 'quoted', 'replied']:
                if src in sampled_user_ids or tgt in sampled_user_ids:
                    tweets.add(src)
                    tweets.add(tgt)
            
            elif rel in ['own', 'followed', 'membership']:
                if src in sampled_user_ids:
                    lists.add(tgt)
                if tgt in sampled_user_ids:
                    lists.add(src)
            
            elif rel == 'contain':
                if src in lists or tgt in tweets:
                    lists.add(src)
                    tweets.add(tgt)
            
            elif rel == 'discuss':
                if src in tweets:
                    hashtags.add(tgt)
        
        keep = relevant[
            (relevant['source_id'].isin(sampled_user_ids) |
             relevant['source_id'].isin(tweets) |
             relevant['source_id'].isin(lists)) &
            (relevant['target_id'].isin(sampled_user_ids) |
             relevant['target_id'].isin(tweets) |
             relevant['target_id'].isin(lists) |
             relevant['target_id'].isin(hashtags))
        ]
        
        all_edges_list.append(keep)
        
        if len(all_edges_list) > 50:
            temp_edges = pd.concat(all_edges_list, ignore_index=True)
            all_edges_list = [temp_edges]
            gc.collect()
        
        pbar.set_postfix({
            'tweets': f"{len(tweets):,}",
            'lists': f"{len(lists):,}",
            'hashtags': f"{len(hashtags):,}"
        })
        pbar.update(1)
    
    pbar.close()
    
    all_edges_df = pd.concat(all_edges_list, ignore_index=True)
    
    entity_ids = {
        'users': sampled_user_ids,
        'tweets': tweets,
        'lists': lists,
        'hashtags': hashtags
    }
    
    print(f"Extracted {len(tweets):,} tweets, {len(lists):,} lists, {len(hashtags):,} hashtags")
    
    return entity_ids, all_edges_df


def save_results(entity_ids, all_edges, users_df):
    sampled_users = users_df[users_df['user_id'].isin(entity_ids['users'])]
    print("Saving")
    
    users_file = os.path.join(OUTPUT_DIR, 'sampled_users.csv')
    sampled_users.to_csv(users_file, index=False)
    
    edges_file = os.path.join(OUTPUT_DIR, 'sampled_edges.csv')
    all_edges.to_csv(edges_file, index=False)
    
    tweets_file = os.path.join(OUTPUT_DIR, 'sampled_tweet_ids.csv')
    pd.DataFrame({'tweet_id': list(entity_ids['tweets'])}).to_csv(tweets_file, index=False)
    
    lists_file = os.path.join(OUTPUT_DIR, 'sampled_list_ids.csv')
    pd.DataFrame({'list_id': list(entity_ids['lists'])}).to_csv(lists_file, index=False)
    
    hashtags_file = os.path.join(OUTPUT_DIR, 'sampled_hashtag_ids.csv')
    pd.DataFrame({'hashtag_id': list(entity_ids['hashtags'])}).to_csv(hashtags_file, index=False)
    
    print(f"Saved to {os.path.abspath(OUTPUT_DIR)}")
    
    summary = {
        'target_users': TARGET_USERS,
        'actual_users': len(entity_ids['users']),
        'tweets': len(entity_ids['tweets']),
        'lists': len(entity_ids['lists']),
        'hashtags': len(entity_ids['hashtags']),
        'edges': len(all_edges),
        'bot_ratio': float((sampled_users['label'] == 'bot').mean())
    }
    
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def plot_distributions(original_df, sampled_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    orig_counts = original_df['label'].value_counts()
    samp_counts = sampled_df['label'].value_counts()
    
    x = [0, 1]
    axes[0].bar([i-0.2 for i in x], [orig_counts.get('human', 0), orig_counts.get('bot', 0)], 
                width=0.4, label='Original', alpha=0.7)
    axes[0].bar([i+0.2 for i in x], [samp_counts.get('human', 0), samp_counts.get('bot', 0)], 
                width=0.4, label='Sampled', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Human', 'Bot'])
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution')
    axes[0].legend()
    
    axes[1].hist(original_df['degree'], bins=50, alpha=0.5, label='Original', density=True)
    axes[1].hist(sampled_df['degree'], bins=50, alpha=0.5, label='Sampled', density=True)
    axes[1].set_xlabel('Degree')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Degree Distribution')
    axes[1].set_xlim(0, original_df['degree'].quantile(0.95))
    axes[1].legend()
    
    orig_bins = original_df.groupby('time_bin').size()
    samp_bins = sampled_df.groupby('time_bin').size()
    
    bins = range(20)
    axes[2].plot(bins, [orig_bins.get(b, 0) for b in bins], marker='o', label='Original', alpha=0.7)
    axes[2].plot(bins, [samp_bins.get(b, 0) for b in bins], marker='s', label='Sampled', alpha=0.7)
    axes[2].set_xlabel('Time Bin')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Temporal Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Starting sampling")
    
    users_df = load_users()
    users_df = calculate_degrees(users_df)
    
    seeds, users_df = select_seeds(users_df)
    sampled_user_ids = expand_sample(seeds, users_df)
    
    sampled_df = users_df[users_df['user_id'].isin(sampled_user_ids)]
    bot_ratio = (sampled_df['label'] == 'bot').mean()
    print(f"Sampled {len(sampled_user_ids):,} users (bot ratio: {bot_ratio:.3f})")
    
    entity_ids, all_edges = extract_entities(sampled_user_ids)

    plot_distributions(users_df, sampled_df)
    
    save_results(entity_ids, all_edges, users_df)
    
    print("Done.")