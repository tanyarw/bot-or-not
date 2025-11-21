from pathlib import Path


from loguru import logger
import torch
from torch_geometric.data import Data

from src.utils.cache import load_tensor
from src.utils.duck import connect


class StaticUserGraph:
    """
    Build a PyG graph with:
        - Only USER nodes
        - Node features = user embeddings
        - Node labels = human(0)/bot(1)
        - Edges: 'following' and 'follower'
    """

    def __init__(
        self,
        db_path: Path,
        user_ids_path: Path,
        node_embeddings_path: Path,
        labels_path: Path,
    ):
        self.con = connect(db_path)

        self.user_ids = load_tensor(user_ids_path, not_tensor=True)
        self.x = load_tensor(node_embeddings_path)
        self.y = load_tensor(labels_path)

        logger.info(f"Loaded {len(self.user_ids)} users")
        logger.info(f"x: {tuple(self.x.shape)}, y: {tuple(self.y.shape)}")

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}

    def _load_edges(self):
        """
        Loads two directed relations for user→user graph:
            - following
            - follower
        and filters out tweet nodes.
        """

        logger.info("Loading user->user edges ('following', 'follower') from DuckDB...")

        df = self.con.execute(
            """
            SELECT source_id, relation, target_id
            FROM edges
            WHERE relation IN ('following', 'follower')
            """
        ).df()

        logger.debug(f"Raw edges loaded: {len(df):,}")

        # Keep only user-to-user edges
        df = df[
            df["source_id"].str.startswith("u") & df["target_id"].str.startswith("u")
        ]
        logger.debug(f"user->user edges kept: {len(df):,}")

        src_idx = []
        dst_idx = []
        rel_types = []

        relation_to_int = {"following": 0, "follower": 1}

        for s, r, d in zip(df["source_id"], df["relation"], df["target_id"]):
            if s in self.user_id_to_idx and d in self.user_id_to_idx:
                src_idx.append(self.user_id_to_idx[s])
                dst_idx.append(self.user_id_to_idx[d])
                rel_types.append(relation_to_int[r])

        logger.success(f"Final edge count (user->user): {len(src_idx):,}")

        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_type = torch.tensor(rel_types, dtype=torch.long)

        return edge_index, edge_type

    def build_pyg_graph(self):
        """
        Returns a PyTorch Geometric Data object:
            data.x:     (N, F)
            data.y:     (N,)
            data.edge_index: (2, E)
            data.edge_type:  (E,)  # for RGCN or HGT
        """

        edge_index, edge_type = self._load_edges()

        logger.info("Building PyG graph object...")

        data = Data(
            x=self.x,
            y=self.y,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=self.x.size(0),
        )

        logger.success("Graph built!")
        logger.success(f" Nodes: {data.num_nodes}")
        logger.success(f" Edges: {data.edge_index.size(1)}")
        logger.success(f" Node features: {tuple(data.x.shape)}")

        return data
