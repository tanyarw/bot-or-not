import os
import sys
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.loader import NodeLoader
from loguru import logger
import duckdb

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloader import SnapshotDataset
from model.bot_ergcn import BotEvolvingRGCN
import config as cfg


class Evaluator:
    def __init__(self, num_classes: int, device: str):
        self.acc_metric = BinaryAccuracy().to(device)
        self.auc_metric = BinaryAUROC().to(device)

    @torch.no_grad()
    def evaluate_snapshot(
        self, model: torch.nn.Module, hetero_data, mask_name: str, device: str
    ):
        model.eval()

        # Extract homogeneous graph data
        x, edge_index, edge_type, y = extract_homo_data(hetero_data, device)

        # Get evaluation mask
        mask = hetero_data["user"][mask_name]
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0

        mask = mask.to(device)

        # Forward pass
        out = model(x, edge_index, edge_type)

        loss = F.cross_entropy(out[mask], y[mask])
        probs = F.softmax(out[mask], dim=-1)

        accuracy = self.acc_metric(probs[:, 1], y[mask])
        auc = self.auc_metric(probs[:, 1], y[mask])

        return loss.item(), accuracy, auc


def load_sorted_ids(db_path: Path) -> tuple:
    logger.info("Loading sorted user and tweet IDs from database...")
    con = duckdb.connect(str(db_path))

    user_ids = (
        con.execute(
            """
        SELECT id
        FROM users
        ORDER BY id
        """
        )
        .df()["id"]
        .tolist()
    )

    tweet_ids = (
        con.execute(
            """
        SELECT id
        FROM tweets
        ORDER BY id
        """
        )
        .df()["id"]
        .tolist()
    )

    con.close()
    logger.success(f"Loaded {len(user_ids)} user IDs and {len(tweet_ids)} tweet IDs")
    return user_ids, tweet_ids


def extract_homo_data(hetero_data, device="cpu"):
    # Get user data
    user_x = hetero_data["user"].x
    user_x = torch.from_numpy(user_x)
    user_y = hetero_data["user"].y
    user_y = torch.from_numpy(user_y)

    # Define relation type mapping for user-user edges only
    relation_map = {
        ("user", "followers", "user"): 0,
        ("user", "following", "user"): 1,
    }

    # Collect all user-user edges
    edge_indices = []
    edge_types = []

    for edge_key, edge_data in hetero_data.edge_items():
        if edge_key not in relation_map:
            continue

        if "edge_index" not in edge_data:
            continue

        edge_index = edge_data["edge_index"]
        relation_type = relation_map[edge_key]

        edge_indices.append(edge_index)
        edge_types.append(
            torch.full((edge_index.size(1),), relation_type, dtype=torch.long)
        )

    # Concatenate all edges
    if len(edge_indices) > 0:
        edge_index = torch.cat(edge_indices, dim=1).to(device)
        edge_type = torch.cat(edge_types).to(device)
    else:
        # Create empty tensors if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)

    user_x = user_x.to(device)
    user_y = user_y.to(device)

    return user_x, edge_index, edge_type, user_y


def train_snapshot(
    model: torch.nn.Module, hetero_data, optimizer: torch.optim.Optimizer, device: str
):
    model.train()

    # Extract homogeneous graph data
    x, edge_index, edge_type, y = extract_homo_data(hetero_data, device)

    # Get train mask
    train_mask = hetero_data["user"].train_mask
    if train_mask.sum() == 0:
        return 0.0

    train_mask = train_mask.to(device)

    optimizer.zero_grad()

    # Forward pass with evolving weights
    out = model(x, edge_index, edge_type)

    # Compute loss only on training nodes
    loss = F.cross_entropy(out[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()


def train_epoch_sequential(
    model: torch.nn.Module,
    dataset: SnapshotDataset,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
):
    logger.info(f"=== Epoch {epoch + 1}/{cfg.NUM_EPOCHS} ===")

    epoch_train_loss = 0
    epoch_val_loss = 0
    epoch_val_acc = 0
    epoch_val_auc = 0
    valid_snapshots = 0

    snapshot_idx = 0
    for snapshot_dict, hetero_data in dataset.iter_snapshots():
        if snapshot_idx >= 150:
            break
        if (snapshot_idx) % 50 == 0:
            logger.info(f"Processing snapshot {snapshot_idx + 1}/{len(dataset)}")
            logger.info(f"  Timestamp: {snapshot_dict['timestamp']}")
            logger.info(
                f"  Users: {snapshot_dict['num_users']}, "
                f"Edges: {snapshot_dict['num_edges']}"
            )

        # Skip snapshots with too few users
        if (
            cfg.SKIP_SMALL_SNAPSHOTS
            and hetero_data["user"].num_nodes < cfg.MIN_USERS_PER_SNAPSHOT
        ):
            logger.warning(
                f"  Skipping snapshot (too few users: {hetero_data['user'].num_nodes})"
            )
            snapshot_idx += 1
            continue

        # Apply random node split for train/val/test
        transform = RandomNodeSplit(
            split="train_rest",
            num_splits=1,
            num_val=cfg.VAL_RATIO,
            num_test=cfg.TEST_RATIO,
            key="y",
        )
        hetero_data = transform(hetero_data)

        # Train on this snapshot (weights evolve here)
        train_loss = train_snapshot(model, hetero_data, optimizer, device)

        # Evaluate on validation set
        evaluator = Evaluator(num_classes=2, device="cuda")
        val_loss, val_acc, val_auc = evaluator.evaluate_snapshot(
            model, hetero_data, "val_mask", device
        )

        if (snapshot_idx) % 50 == 0:
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(
                f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}"
            )

        epoch_train_loss += train_loss
        epoch_val_loss += val_loss
        epoch_val_acc += val_acc
        epoch_val_auc += val_auc
        valid_snapshots += 1

        snapshot_idx += 1

    # Return average metrics
    if valid_snapshots > 0:
        return (
            epoch_train_loss / valid_snapshots,
            epoch_val_loss / valid_snapshots,
            epoch_val_acc / valid_snapshots,
            epoch_val_auc / valid_snapshots,
        )
    return 0, 0, 0


if __name__ == "__main__":
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
        logger.debug("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.debug("Using Apple MPS backend")
    else:
        device = "cpu"
        logger.debug("Using CPU")

    # Paths
    snapshots_path = os.path.join(cfg.DATA_DIR, "temporalized", "snapshots.pkl")
    user_embeddings_path = os.path.join(
        cfg.DATA_DIR, "embeddings", f"user_node_embeddings_128_test.pt"
    )
    tweet_embeddings_path = os.path.join(
        cfg.DATA_DIR, "embeddings", f"tweet_node_embeddings_128_test.pt"
    )
    labels_path = os.path.join(cfg.DATA_DIR, "embeddings", "labels.pt")

    # Load sorted IDs (only users now)
    user_ids_sorted, tweet_ids_sorted = load_sorted_ids(cfg.DB_PATH)

    # Initialize dataset (no tweet IDs needed anymore)
    logger.info("Initializing SnapshotDataset...")
    dataset = SnapshotDataset(
        snapshots_path=snapshots_path,
        user_embeddings_path=user_embeddings_path,
        tweet_embeddings_path=tweet_embeddings_path,
        labels_path=labels_path,
        user_ids_sorted=user_ids_sorted,
        tweet_ids_sorted=tweet_ids_sorted,
        device=device,
    )
    logger.success(f"Dataset initialized with {len(dataset)} snapshots")

    # Initialize EVOLVING RGCN model
    model = BotEvolvingRGCN(
        in_channels=cfg.EMBEDDING_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        num_relations=2,
        out_channels=2,
        dropout=cfg.DROPOUT,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )
    logger.info(f"Number of relation types: 2 (followers, following)")

    # Create model save directory
    if cfg.SAVE_MODEL:
        os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(cfg.NUM_EPOCHS):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        train_loss, val_loss, val_acc, val_auc = train_epoch_sequential(
            model, dataset, optimizer, device, epoch
        )

        logger.info(
            f"Epoch {epoch + 1} Summary - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f},"
            f" Val AUC: {val_auc:.4f}"
        )

        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        total_time = time.time() - time.mktime(
            time.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        )
        logger.info(
            f"Epoch {epoch + 1} completed at {end_time} (Duration: {total_time/60:.2f} minutes)"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            if cfg.SAVE_MODEL:
                model_path = os.path.join(cfg.MODEL_SAVE_PATH, f"evolving_rgcn_best.pt")
                torch.save(model.state_dict(), model_path)
                logger.success(
                    f"Best model saved (Val Acc: {val_acc:.4f} and Val AUC: {val_auc:.4f})"
                )
        else:
            patience_counter += 1

        # Early stopping
        if (
            hasattr(cfg, "EARLY_STOPPING_PATIENCE")
            and patience_counter >= cfg.EARLY_STOPPING_PATIENCE
        ):
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    logger.success(f"Training completed. Best Val Acc: {best_val_acc:.4f}")

    # Save final model
    if cfg.SAVE_MODEL:
        model_path = os.path.join(cfg.MODEL_SAVE_PATH, f"evolving_rgcn_best.pt")
        torch.save(model.state_dict(), model_path)
        logger.success(f"Final model saved to {model_path}")
