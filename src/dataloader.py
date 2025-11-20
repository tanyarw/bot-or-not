import numpy as np
import pickle
import duckdb
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm
import csv
import gc


class GraphDataset:
    def __init__(self, embeddings_path, snapshots_path, cache_snapshots=False):
        # Load pre-computed embeddings
        self.embeddings = np.load(embeddings_path, mmap_mode='r')
        self.embedding_dim = self.embeddings.shape[1]
        
        print(f"Loaded embeddings: {self.embeddings.shape} ({self.embeddings.nbytes / 1024**3:.2f} GB)")
        
        # Build node mapping from CSVs
        self.node_to_idx, self.num_nodes, self.num_snapshots = self._node_mapping()
        
        self.snapshots_path = snapshots_path
        self.snapshots = self._load_snapshots() if cache_snapshots else None
        
        # Load labels
        self.labels = self._load_labels()
        
        # Edge relation types
        self.relation_map = {
            'following': 0, 'followers': 1, 'post': 2, 'like': 3,
            'retweeted': 4, 'quoted': 5, 'replied': 6, 'mentioned': 7, 'pinned': 8
        }
        
        print(f"Dataset: {self.num_snapshots} snapshots, {self.num_nodes:,} nodes")
        print(f"Bot labels: {self.labels.sum():.0f} / {self.num_nodes:,}")
    
    def _node_mapping(self):
        with open('./data/temporalized/summary.txt', 'r') as f:
            lines = f.readlines()
            num_snapshots = int(lines[0].split(':')[1].strip())
        
        print(f"Loading node IDs from CSVs...")
        
        # Load user IDs
        users = []
        with open('./data/static/sampled_user_ids.csv', 'r') as f:
            reader = csv.DictReader(f)
            users = [row['user_id'] for row in reader]
        
        # Load tweet IDs
        tweets = []
        with open('./data/static/sampled_tweet_ids.csv', 'r') as f:
            reader = csv.DictReader(f)
            tweets = [row['tweet_id'] for row in reader]

        users = sorted(users)
        tweets = sorted(tweets)
        nodes = users + tweets
        
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        print(f"Node mapping built: {len(users):,} users + {len(tweets):,} tweets = {len(nodes):,} nodes")
        
        return node_to_idx, len(nodes), num_snapshots
    
    def _load_snapshots(self):
        snapshots = []
        with open(self.snapshots_path, 'rb') as f:
            for _ in tqdm(range(self.num_snapshots), desc="Loading snapshots"):
                snapshots.append(pickle.load(f))
        return snapshots
    
    def _get_snapshot(self, idx):
        if self.snapshots:
            return self.snapshots[idx]
        
        # Stream from disk
        with open(self.snapshots_path, 'rb') as f:
            for i in range(idx + 1):
                snap = pickle.load(f)
            return snap
    
    def _load_labels(self):        
        labels = np.zeros(self.num_nodes, dtype=np.float32)
        
        conn = duckdb.connect('./db/twitter_graph.duckdb', read_only=True)
        results = conn.execute("SELECT id, label FROM bot_labels").fetchall()
        conn.close()
        
        for node_id, label in results:
            if node_id in self.node_to_idx:
                labels[self.node_to_idx[node_id]] = float(label == 'bot')
        
        return labels
    
    def __len__(self):
        return self.num_snapshots
    
    def __getitem__(self, idx):
        snap = self._get_snapshot(idx)
        
        # Pre-computed embeddings
        features = torch.from_numpy(np.array(self.embeddings[:]))
        
        # Edges with relation types
        edges = []
        edge_attrs = []
        
        for src, rel, tgt in snap.get('edges', []):
            if src in self.node_to_idx and tgt in self.node_to_idx:
                edges.append([self.node_to_idx[src], self.node_to_idx[tgt]])
                edge_attrs.append(self.relation_map.get(rel, 0))
        
        edge_index = (torch.tensor(edges, dtype=torch.long).T 
                     if edges else torch.zeros((2, 0), dtype=torch.long))
        edge_attr = (torch.tensor(edge_attrs, dtype=torch.float32) 
                    if edge_attrs else torch.zeros(0, dtype=torch.float32))
        
        # PyG Data object
        data = Data(
            x=features,
            edge_index=edge_index,
            edge_attr=edge_attr,  # Relation types for heterogeneous graph
            y=torch.tensor(self.labels, dtype=torch.float32),
            num_nodes=self.num_nodes
        )
        
        # Cleanup
        del edges, edge_attrs
        if not self.snapshots:
            del snap
        gc.collect()
        
        return data
    
    def close(self):
        if hasattr(self, 'embeddings'):
            del self.embeddings
        if self.snapshots:
            self.snapshots.clear()
        gc.collect()
    
    def __del__(self):
        self.close()


def load_data(
    embeddings_path='./data/embeddings/node_embeddings.npy',
    snapshots_path='./data/temporalized/snapshots.pkl',
    cache_snapshots=False
):
    return GraphDataset(embeddings_path, snapshots_path, cache_snapshots)