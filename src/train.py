import os
import sys
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torch_geometric.transforms import RandomNodeSplit
from loguru import logger
import duckdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataloader import SnapshotDataset
from model.bot_ergcn import BotEvolveRGCN
import src.config as cfg


def build_sequence_batch(
    snapshots: list,
    device: str
):
    A_list = []
    edge_type_list = []
    Nodes_list = []
    mask_list = []
    y_list = []
    train_mask_list = []
    val_mask_list = []
    
    # Find max nodes
    max_nodes = max(
        hetero_data['user'].num_nodes 
        for _, hetero_data in snapshots
    )
    
    for _, hetero_data in snapshots:
        transform = RandomNodeSplit(
            split='train_rest',
            num_splits=1,
            num_val=cfg.VAL_RATIO,
            num_test=cfg.TEST_RATIO,
            key='y'
        )
        hetero_data = transform(hetero_data)

        x, edge_index, edge_type, y = extract_homo_data(hetero_data, device) 
        num_nodes = x.size(0)
        
        # Pad to max_nodes
        if num_nodes < max_nodes:
            padding = max_nodes - num_nodes
            x = F.pad(x, (0, 0, 0, padding), value=0)
            y = F.pad(y, (0, padding), value=0)
            
            train_mask = hetero_data['user'].train_mask
            val_mask = hetero_data['user'].val_mask
            train_mask = F.pad(train_mask.to(device), (0, padding), value=False)
            val_mask = F.pad(val_mask.to(device), (0, padding), value=False)
        else:
            train_mask = hetero_data['user'].train_mask.to(device)
            val_mask = hetero_data['user'].val_mask.to(device)
        
        # Mask for TopK
        topk_mask = torch.zeros(max_nodes, device=device)
        if num_nodes < max_nodes:
            topk_mask[num_nodes:] = -float('inf')
        
        A_list.append(edge_index)
        edge_type_list.append(edge_type)
        Nodes_list.append(x)
        mask_list.append(topk_mask)
        y_list.append(y)
        train_mask_list.append(train_mask)
        val_mask_list.append(val_mask)
    
    return (
        A_list, edge_type_list, Nodes_list, mask_list,
        y_list, train_mask_list, val_mask_list
    )


def train_sequence(
    model: torch.nn.Module,
    snapshots: list,
    optimizer: torch.optim.Optimizer,
    device: str
):
    model.train()
    (A_list, edge_type_list, Nodes_list, mask_list, y_list, train_mask_list, _) = build_sequence_batch(snapshots, device)
    
    if len(A_list) == 0:
        return 0.0
    
    optimizer.zero_grad()

    out_seq = model(A_list, edge_type_list, Nodes_list, mask_list)
    
    total_loss = 0.0
    num_losses = 0
    
    for t in range(len(out_seq)):
        if train_mask_list[t].sum() == 0:
            continue
        
        loss_t = F.cross_entropy(
            out_seq[t][train_mask_list[t]], 
            y_list[t][train_mask_list[t]]
        )
        
        total_loss += loss_t
        num_losses += 1
    
    if num_losses > 0:
        avg_loss = total_loss / num_losses
    else:
        return 0.0
    
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return avg_loss.item()


@torch.no_grad()
def evaluate_sequence(
    model: torch.nn.Module,
    snapshots: list,
    device: str,
    mask_type: str = 'val_mask'
):
    model.eval()

    (A_list, edge_type_list, Nodes_list, mask_list, y_list, train_mask_list, val_mask_list) = build_sequence_batch(snapshots, device)
    
    if len(A_list) == 0:
        return 0.0, 0.0, 0.0
    
    out_seq = model(A_list, edge_type_list, Nodes_list, mask_list)
    
    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    num_evals = 0
    
    for t in range(len(out_seq)):
        eval_mask = val_mask_list[t] if mask_type == 'val_mask' else train_mask_list[t]
        
        if eval_mask.sum() == 0:
            continue
        
        loss_t = F.cross_entropy(out_seq[t][eval_mask], y_list[t][eval_mask])
        probs = F.softmax(out_seq[t][eval_mask], dim=-1)
        acc = acc_metric(probs[:, 1], y_list[t][eval_mask])
        f1 = f1_metric(probs[:, 1], y_list[t][eval_mask])
        
        total_loss += loss_t.item()
        total_acc += acc.item()
        total_f1 += f1.item()
        num_evals += 1
    
    if num_evals > 0:
        return (
            total_loss / num_evals,
            total_acc / num_evals,
            total_f1 / num_evals
        )
    return 0.0, 0.0, 0.0


def train_epoch_sequences(
    model: torch.nn.Module,
    dataset: SnapshotDataset,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    sequence_length: int = 10
):
    logger.info(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")
    
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    epoch_val_acc = 0.0
    epoch_val_f1 = 0.0
    num_sequences = 0
    
    # Create sequence of snapshots
    current_sequence = []
    snapshot_idx = 0
    total_snapshots = 0
    
    for snapshot_dict, hetero_data in dataset.iter_snapshots():
        if snapshot_idx >= cfg.MAX_SNAPSHOTS:
            break
        
        # Skipping small snapshots
        if cfg.SKIP_SMALL_SNAPSHOTS and hetero_data['user'].num_nodes < cfg.MIN_USERS_PER_SNAPSHOT:
            logger.warning(f"Skipping snapshot {snapshot_idx} (too few users)")
            snapshot_idx += 1
            continue
        
        current_sequence.append((snapshot_dict, hetero_data))
        total_snapshots += 1
        
        if len(current_sequence) > sequence_length:
            logger.info(
                f"Processing sequence {num_sequences + 1}: "
                f"Snapshots {snapshot_idx - len(current_sequence) + 1}-{snapshot_idx + 1}"
            )
            
            train_loss = train_sequence(
                model, current_sequence, optimizer, device
            )
            
            val_loss, val_acc, val_f1 = evaluate_sequence(
                model, current_sequence, device, mask_type='val_mask'
            )
            
            logger.info(
                f"  Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )

            logger.info(
                f"Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}"
            )
            
            epoch_train_loss += train_loss
            epoch_val_loss += val_loss
            epoch_val_acc += val_acc
            epoch_val_f1 += val_f1
            num_sequences += 1
            
            # Clear
            current_sequence = []
        
        snapshot_idx += 1
    
    # Process remaining snapshots (if any)
    if len(current_sequence) > 0:
        logger.info(
            f"Processing final sequence {num_sequences + 1}: "
            f"{len(current_sequence)} snapshots"
        )
        
        train_loss = train_sequence(model, current_sequence, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_sequence(
            model, current_sequence, device, mask_type='val_mask'
        )
        
        logger.info(
            f"  Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Val F1: {val_f1:.4f}"
        )
        
        epoch_train_loss += train_loss
        epoch_val_loss += val_loss
        epoch_val_acc += val_acc
        epoch_val_f1 += val_f1
        num_sequences += 1
    
    logger.info(f"Processed {total_snapshots} snapshots in {num_sequences} sequences")
    
    if num_sequences > 0:
        return (
            epoch_train_loss / num_sequences,
            epoch_val_loss / num_sequences,
            epoch_val_acc / num_sequences,
            epoch_val_f1 / num_sequences
        )
    return 0.0, 0.0, 0.0, 0.0

def load_sorted_ids(db_path: Path) -> tuple:
    logger.info("Loading sorted user and tweet IDs from database...")
    con = duckdb.connect(str(db_path))
    
    user_ids = con.execute(
        """
        SELECT id
        FROM users
        ORDER BY id
        """
    ).df()['id'].tolist()
    
    tweet_ids = con.execute(
        """
        SELECT id
        FROM tweets
        ORDER BY id
        """
    ).df()['id'].tolist()
    
    con.close()
    logger.success(f"Loaded {len(user_ids)} user IDs and {len(tweet_ids)} tweet IDs")
    return user_ids, tweet_ids


def extract_homo_data(hetero_data, device='cpu'):
    # Get user data
    user_x = hetero_data['user'].x
    user_y = hetero_data['user'].y
    
    # Relation type mapping (user-user edges only)
    relation_map = {
        ('user', 'followers', 'user'): 0,
        ('user', 'following', 'user'): 1,
    }
    
    # Collect all user-user edges
    edge_indices = []
    edge_types = []
    
    for edge_key, edge_data in hetero_data.edge_items():
        if edge_key not in relation_map:
            continue
        
        if 'edge_index' not in edge_data:
            continue
            
        edge_index = edge_data['edge_index']
        relation_type = relation_map[edge_key]
        
        edge_indices.append(edge_index)
        edge_types.append(torch.full((edge_index.size(1),), relation_type, dtype=torch.long))
    
    # Concatenate all edges
    if len(edge_indices) > 0:
        edge_index = torch.cat(edge_indices, dim=1).to(device)
        edge_type = torch.cat(edge_types).to(device)
    else:
        # Empty tensors if no edges
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_type = torch.empty((0,), dtype=torch.long, device=device)

    user_x = user_x.to(device)
    user_y = user_y.to(device)
    
    return user_x, edge_index, edge_type, user_y


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        logger.debug("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.debug("Using Apple MPS backend")
    else:
        device = "cpu"
        logger.debug("Using CPU")
    
    snapshots_path = os.path.join(cfg.DATA_DIR, "temporalized", "snapshots.pkl")
    user_embeddings_folder = os.path.join(cfg.DATA_DIR, "embeddings", "temporal")
    tweet_embeddings_path = os.path.join(cfg.DATA_DIR, "embeddings", "all_tweet_embeddings_sorted.pt")
    labels_path = os.path.join(cfg.DATA_DIR, "embeddings", "labels.pt")
    
    user_ids_sorted, tweet_ids_sorted = load_sorted_ids(cfg.DB_PATH)
    
    dataset = SnapshotDataset(
        snapshots_path=snapshots_path,
        user_embeddings_folder=user_embeddings_folder,
        tweet_embeddings_path=tweet_embeddings_path,
        labels_path=labels_path,
        user_ids_sorted=user_ids_sorted,
        tweet_ids_sorted=tweet_ids_sorted,
        device=device
    )
    
    model = BotEvolveRGCN(
        in_channels=cfg.EMBEDDING_DIM,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        num_relations=2,  # followers + following
        out_channels=2,   # bot vs human
        dropout=cfg.DROPOUT,
        num_layers=2,
        device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Number of relations: 2 (followers, following)")
    
    if cfg.SAVE_MODEL:
        os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)
    
    SEQUENCE_LENGTH = getattr(cfg, 'SEQUENCE_LENGTH', 10)
    logger.info(f"Processing {SEQUENCE_LENGTH} snapshots per sequence")
    
    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(cfg.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss, val_loss, val_acc, val_f1 = train_epoch_sequences(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            sequence_length=SEQUENCE_LENGTH
        )
        
        epoch_time = time.time() - start_time
        
        logger.info(
            f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS} Summary - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Val F1: {val_f1:.4f} "
            f"(Duration: {epoch_time/60:.2f} min)"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            
            if cfg.SAVE_MODEL:
                model_path = os.path.join(
                    cfg.MODEL_SAVE_PATH, 
                    "evolvegcn_rgcn_best.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'sequence_length': SEQUENCE_LENGTH,
                }, model_path)
                logger.success(
                    f"Best model saved - "
                    f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
                )
        else:
            patience_counter += 1
        
        # Early stopping
        if hasattr(cfg, 'EARLY_STOPPING_PATIENCE'):
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    logger.success(
        f"Training completed! "
        f"Best Val Acc: {best_val_acc:.4f}, "
        f"Best Val F1: {best_val_f1:.4f}"
    )