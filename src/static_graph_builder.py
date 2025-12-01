from pathlib import Path
from loguru import logger
import torch
from torch_geometric.data import HeteroData

from src.utils.cache import load_tensor
from src.utils.duck import connect


class StaticUserGraph:
    """
    Build a PyG HeteroData graph with:
        - Node type: 'user'
        - Node features = user embeddings
        - Node labels = human(0)/bot(1)
        - Edge types: ('user', 'following', 'user') and ('user', 'followers', 'user')
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
        self.x = load_tensor(node_embeddings_path)  # (N, F)
        self.y = load_tensor(labels_path)  # (N,)

        logger.info(f"Loaded {len(self.user_ids)} users")
        logger.info(f"x: {tuple(self.x.shape)}, y: {tuple(self.y.shape)}")

        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}

    def _load_edges(self):
        """
        Loads directed user→user edges for:
            - following
            - followers
        """

        logger.info("Loading user->user edges from DuckDB...")

        df = self.con.execute(
            """
            SELECT source_id, relation, target_id
            FROM edges
            WHERE relation IN ('following', 'followers')
            """
        ).df()

        logger.debug(f"Raw edges loaded: {len(df):,}")

        # only user nodes
        df = df[
            df["source_id"].str.startswith("u") & df["target_id"].str.startswith("u")
        ]
        logger.debug(f"user->user edges kept: {len(df):,}")

        # separate lists for each relation
        following_src, following_dst = [], []
        followers_src, followers_dst = [], []

        for s, r, d in zip(df["source_id"], df["relation"], df["target_id"]):
            if s in self.user_id_to_idx and d in self.user_id_to_idx:
                s_idx = self.user_id_to_idx[s]
                d_idx = self.user_id_to_idx[d]

                if r == "following":
                    following_src.append(s_idx)
                    following_dst.append(d_idx)
                elif r == "followers":
                    followers_src.append(s_idx)
                    followers_dst.append(d_idx)

        logger.success(f"Following edges: {len(following_src):,}")
        logger.success(f"Followers edges: {len(followers_src):,}")

        return (
            torch.tensor(following_src),
            torch.tensor(following_dst),
            torch.tensor(followers_src),
            torch.tensor(followers_dst),
        )

    def build_pyg_graph(self):
        """
        Returns a PyTorch Geometric HeteroData object.
        """

        (
            fol_src,
            fol_dst,
            fw_src,
            fw_dst,
        ) = self._load_edges()

        logger.info("Building PyG HeteroData graph ...")

        data = HeteroData()

        data["user"].x = self.x  # (N, embedding_dim)
        data["user"].y = self.y  # (N,)
        data["user"].num_nodes = self.x.size(0)
        data["user"].id = self.user_ids

        # user -> following -> user
        data[("user", "following", "user")].edge_index = torch.stack(
            [fol_src, fol_dst], dim=0
        )

        # user -> followers -> user
        data[("user", "followers", "user")].edge_index = torch.stack(
            [fw_src, fw_dst], dim=0
        )

        logger.success("Hetero graph built!")
        logger.success(f" User nodes: {data['user'].num_nodes}")
        logger.success(
            f" Following edges: {data['user','following','user'].edge_index.size(1)}"
        )
        logger.success(
            f" Followers edges: {data['user','followers','user'].edge_index.size(1)}"
        )

        return data
