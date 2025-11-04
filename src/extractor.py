import pandas as pd
import json as json_lib
import os
from tqdm import tqdm
import gc
import ijson


TWEET_FILES = [
    "tweet_0.json", "tweet_1.json", "tweet_2.json",
    "tweet_3.json", "tweet_4.json", "tweet_5.json",
    "tweet_6.json", "tweet_7.json", "tweet_8.json"
]


def extract_tweets(input_dir, tweet_file, sampled_ids, output_handle):
    input_dir = os.path.abspath(input_dir)
    file_path = os.path.join(input_dir, tweet_file)
    
    if not os.path.exists(file_path):
        return 0
    
    matched_count = 0
    
    with open(file_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        
        for tweet in tqdm(parser, desc=f"  {tweet_file}", unit=" tweets"):
            tweet_id = tweet.get('id')
            
            if tweet_id in sampled_ids:
                json_lib.dump(tweet, output_handle, default=str)
                output_handle.write('\n')
                
                matched_count += 1
                sampled_ids.remove(tweet_id)
            
            if len(sampled_ids) == 0:
                break
    
    gc.collect()
    return matched_count


def extract_users(input_dir, output_dir):
    input_dir = os.path.abspath(input_dir)
    user_file = os.path.join(input_dir, "user.json")
    
    output_dir = os.path.abspath(output_dir)
    sampled_users_file = os.path.join(output_dir, "sampled_users.csv")
    users_df = pd.read_csv(sampled_users_file)
    sampled_ids = set(users_df['user_id'].values)
    print(f"Looking for {len(sampled_ids):,} users")
    
    output_file = os.path.join(output_dir, "sampled_users.jsonl")
    matched_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(user_file, 'r', encoding='utf-8') as f:
            users_data = json_lib.load(f)
        
        for user in tqdm(users_data, desc="  Extracting", unit=" users"):
            user_id = user.get('id')
            
            if user_id in sampled_ids:
                json_lib.dump(user, out_f, default=str)
                out_f.write('\n')
                matched_count += 1
                sampled_ids.remove(user_id)
            
            if len(sampled_ids) == 0:
                break
        
        del users_data
        gc.collect()
    
    print(f"Found {matched_count:,} users")


def extract(input_dir, output_dir):
    print("Extracting tweets")
    
    tweet_ids_file = os.path.join(output_dir, "sampled_tweet_ids.csv")
    tweet_ids_df = pd.read_csv(tweet_ids_file)
    sampled_ids = set(tweet_ids_df['tweet_id'].values)
    print(f"Looking for {len(sampled_ids):,} tweets")
    
    output_file = os.path.join(output_dir, "sampled_tweets.jsonl")
    total_found = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for tweet_file in TWEET_FILES:
            found = extract_tweets(input_dir, tweet_file, sampled_ids, out_f)
            total_found += found
            
            if len(sampled_ids) == 0:
                break
    
    print(f"Found {total_found:,} tweets")

    extract_users(input_dir, output_dir)
    
    print(f"Saved to {output_dir}")