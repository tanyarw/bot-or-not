import pickle
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from pathlib import Path


class SnapshotDataset:
    def __init__(
        self,
        snapshots_path: Path,
        user_embeddings_path: Path,\
        tweet_embeddings_path: Path,
        labels_path: Path,
        user_ids_sorted: List[str],
        tweet_ids_sorted: List[str],
        device: str = "cpu"
    ):
        self.snapshots_path = snapshots_path
        self.device = device
        
        self.embeddings_user = torch.load(user_embeddings_path, map_location=device, weights_only=False)  # (N, 128)
        self.embeddings_tweet = torch.load(tweet_embeddings_path, map_location=device, weights_only=False)  # (M, 128)
        self.labels = torch.load(labels_path, map_location=device, weights_only=False)  # (N,)
        

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids_sorted)}
        self.tweet_id_to_idx = {tid: idx for idx, tid in enumerate(tweet_ids_sorted)}
        
        self.num_snapshots = 834
    
    def __len__(self) -> int:
        return self.num_snapshots
    
    def get_snapshot(self, idx: int) -> Dict:
        with open(self.snapshots_path, 'rb') as f:
            for i in range(idx + 1):
                snapshot = pickle.load(f)
        return snapshot
    
    def create_hetero_data(self, snapshot: Dict) -> Tuple[HeteroData, Dict[str, int]]:
        data = HeteroData()
        
        active_user_ids = snapshot['user_ids']
        active_user_indices = [self.user_id_to_idx[uid] for uid in active_user_ids]

        local_user_id_to_idx = {uid: local_idx for local_idx, uid in enumerate(active_user_ids)}
        
        user_embeddings = self.embeddings_user[active_user_indices]  # (num_active_users, 128)
        user_labels = self.labels[active_user_indices]  # (num_active_users,)
        
        data['user'].x = user_embeddings
        data['user'].y = user_labels
        data['user'].num_nodes = len(active_user_ids)
        

        active_tweet_ids = snapshot['tweet_ids']
        active_tweet_indices = [self.tweet_id_to_idx[tid] for tid in active_tweet_ids]

        local_tweet_id_to_idx = {tid: local_idx for local_idx, tid in enumerate(active_tweet_ids)}
        
        tweet_embeddings = self.embeddings_tweet[active_tweet_indices]  # (num_active_tweets, 128)
        
        # data['tweet'].x = tweet_embeddings
        # data['tweet'].num_nodes = len(active_tweet_ids)
        
        # Process edges and create edge indices by relation type
        edges_by_relation = self._group_edges_by_relation(
            snapshot['edges'],
            local_user_id_to_idx,
            local_tweet_id_to_idx
        )
        
        # Add edges to HeteroData
        for (src_type, relation, dst_type), edge_list in edges_by_relation.items():
            if len(edge_list) > 0:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                data[src_type, relation, dst_type].edge_index = edge_index
        
        return data
    
    def _group_edges_by_relation(
        self,
        edges: List[List],
        user_mapping: Dict[str, int],
        tweet_mapping: Dict[str, int]
    ) -> Dict[Tuple[str, str, str], List[List[int]]]:
        # Define edge type mappings

        # user -> user relations
        user_user_relations = {'following', 'followers'}
        # user -> tweet relations
        # user_tweet_relations = {'post', 'like', 'retweeted', 'quoted', 'replied', 'mentioned', 'pinned'}
        
        edges_dict = {}
        
        for source_id, relation, target_id in edges:
            # Determine source and target types
            src_is_user = source_id in user_mapping
            dst_is_user = target_id in user_mapping
            src_is_tweet = source_id in tweet_mapping
            dst_is_tweet = target_id in tweet_mapping
            
            # Skip if nodes don't exist in this snapshot
            if not ((src_is_user or src_is_tweet) and (dst_is_user or dst_is_tweet)):
                continue
            
            # Map to local indices based on node types
            if relation in user_user_relations:
                if src_is_user and dst_is_user:
                    src_idx = user_mapping[source_id]
                    dst_idx = user_mapping[target_id]
                    key = ('user', relation, 'user')
                else:
                    continue
            # elif relation in user_tweet_relations:
            #     if src_is_user and dst_is_tweet:
            #         src_idx = user_mapping[source_id]
            #         dst_idx = tweet_mapping[target_id]
            #         key = ('user', relation, 'tweet')
            #     else:
            #         continue
            else:
                continue
            
            if key not in edges_dict:
                edges_dict[key] = []
            edges_dict[key].append([src_idx, dst_idx])
        
        return edges_dict
    
    def iter_snapshots(self):
        with open(self.snapshots_path, 'rb') as f:
            try:
                while True:
                    snapshot = pickle.load(f)
                    hetero_data = self.create_hetero_data(snapshot)
                    yield snapshot, hetero_data
            except EOFError:
                pass