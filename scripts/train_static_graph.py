import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
import csv

import torch
from torch_geometric.data import HeteroData
from torch.utils.tensorboard import SummaryWriter

load_dotenv()
DATASET_ROOT = Path(os.getenv("DATASET_ROOT")).expanduser().resolve()
LOG_ROOT = Path(os.getenv("LOG_ROOT")).expanduser().resolve()
BOTRGCN_ROOT = Path(os.getenv("BOTRGCN_ROOT")).expanduser().resolve()
TEST_NODES_PATH = Path(os.getenv("TEST_NODES_PATH")).expanduser().resolve()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import BotRGCN
from config import load_config

# train step
def train_step(
    model,
    x,
    edge_index,
    edge_type,
    y,
    train_mask,
    optimizer,
    loss_fn,
):
    """Single training step on FULL graph, masking only nodes in loss."""
    model.train()
    optimizer.zero_grad()

    out = model(x, edge_index, edge_type)  # (N, 2)

    loss = loss_fn(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


# eval step
def eval_step(model, x, edge_index, edge_type, y, mask):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_type)

    logits = out[mask]
    preds = logits.argmax(dim=1)

    acc = accuracy_score(y[mask].cpu(), preds.cpu())
    f1 = f1_score(y[mask].cpu(), preds.cpu(), average="macro")

    return acc, f1, preds.cpu(), logits.cpu()


if __name__ == "__main__":

    cfg = load_config("config.yaml")
    cfg["paths"]["out_dir"] = str(DATASET_ROOT)
    cfg["botrgcn"]["out_dir"] = str(BOTRGCN_ROOT)
    cfg["botrgcn"]["log_dir"] = str(LOG_ROOT)

    input_dir = Path(cfg["paths"]["out_dir"])
    out_dir = Path(cfg["botrgcn"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["botrgcn"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = log_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_acc",
                "train_f1_macro",
                "val_acc",
                "val_f1_macro",
                "val_f1_minority",
                "weighted_f1",
                "test_acc",
                "test_f1",
            ]
        )
    writer_tb = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    data_path = input_dir / "static_user_hetero_graph.pt"
    logger.info(f"Loading graph from {data_path}")

    raw_data = torch.load(data_path, weights_only=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if isinstance(raw_data, HeteroData):
        logger.info("Detected HeteroData. Converting to homogeneous tensors...")

        hetero_data: HeteroData = raw_data.to(device)

        # test user IDs
        test_user_ids = torch.load(TEST_NODES_PATH)

        num_nodes = hetero_data["user"].num_nodes

        all_user_ids = hetero_data["user"].id
        print(len(all_user_ids))
        id_to_index = {uid: idx for idx, uid in enumerate(all_user_ids)}
        test_node_indices = torch.tensor(
            [id_to_index[uid] for uid in test_user_ids], dtype=torch.long, device=device
        )

        num_nodes = hetero_data["user"].num_nodes
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask[test_node_indices] = True

        remaining_mask = ~test_mask
        remaining_indices = remaining_mask.nonzero(as_tuple=True)[0]

        perm = torch.randperm(remaining_indices.size(0), device=device)
        remaining_indices = remaining_indices[perm]

        num_remaining = remaining_indices.size(0)
        val_ratio = cfg["botrgcn"]["split"]["val"]
        num_val = int(val_ratio * num_remaining)

        val_ids = remaining_indices[:num_val]
        train_ids = remaining_indices[num_val:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        train_mask[train_ids] = True
        val_mask[val_ids] = True

        hetero_data["user"].train_mask = train_mask
        hetero_data["user"].val_mask = val_mask
        hetero_data["user"].test_mask = test_mask

        logger.info(
            f"Mask sizes — Train={train_mask.sum()}  Val={val_mask.sum()}  Test={test_mask.sum()}"
        )

        x = hetero_data["user"].x  # (N, embedding_dim)
        y = hetero_data["user"].y  # (N,)

        train_mask = hetero_data["user"].train_mask
        val_mask = hetero_data["user"].val_mask
        test_mask = hetero_data["user"].test_mask

        edge_index_followers = hetero_data[("user", "followers", "user")].edge_index
        edge_index_following = hetero_data[("user", "following", "user")].edge_index
        edge_index = torch.cat(
            [edge_index_followers, edge_index_following],
            dim=1,
        )

        num_followers_edges = edge_index_followers.size(1)
        num_following_edges = edge_index_following.size(1)

        edge_type = torch.cat(
            [
                torch.zeros(num_followers_edges, dtype=torch.long, device=device),  # 0
                torch.ones(num_following_edges, dtype=torch.long, device=device),  # 1
            ],
            dim=0,
        )

    else:
        raise TypeError(f"Unsupported data type: {type(raw_data)}")

    model = BotRGCN(
        in_channels=x.shape[1],
        hidden_channels=128,
        num_relations=2,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5 * 1e-3, weight_decay=1e-4)
    best_model_path = out_dir / "bot_rgcn.pt"

    # class weights
    num_pos = (y == 1).sum().item()
    num_neg = (y == 0).sum().item()

    weight_pos = num_neg / (num_pos + num_neg)
    weight_neg = num_pos / (num_pos + num_neg)

    class_weights = torch.tensor([weight_neg, weight_pos], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_score = 0.0
    best_epoch = 0
    patience = 10
    min_delta = 1e-4
    patience_counter = 10

    for epoch in range(1, 301):

        loss = train_step(
            model,
            x,
            edge_index,
            edge_type,
            y,
            train_mask,
            optimizer,
            loss_fn,
        )

        train_acc, train_f1, _, _ = eval_step(
            model, x, edge_index, edge_type, y, val_mask
        )
        val_acc, val_f1, val_preds, _ = eval_step(
            model, x, edge_index, edge_type, y, val_mask
        )

        val_f1_minority = f1_score(
            y[val_mask].cpu(),
            val_preds.cpu(),
            average="binary",
            pos_label=1,
        )
        val_f1_macro = val_f1

        combined_score = 0.7 * val_f1_minority + 0.3 * val_f1_macro

        logger.info(
            f"Epoch {epoch:03d} | Loss={loss:.4f} | "
            f"TrainAcc={train_acc:.4f} | TrainMacroF1={train_f1:.4f} | "
            f"ValAcc={val_acc:.4f} | ValMacroF1={val_f1_macro:.4f} | "
            f"ValMinorityF1={val_f1_minority:.4f} | WeightedF1={combined_score:.4f}"
        )

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    float(train_acc),
                    float(train_f1),
                    float(val_acc),
                    float(val_f1_macro),
                    float(val_f1_minority),
                    float(combined_score),
                    "",
                    "",
                ]
            )

        writer_tb.add_scalar("Loss/train", loss, epoch)
        writer_tb.add_scalar("Accuracy/train", train_acc, epoch)
        writer_tb.add_scalar("F1/train_macro", train_f1, epoch)
        writer_tb.add_scalar("Accuracy/val", val_acc, epoch)
        writer_tb.add_scalar("F1/val_macro", val_f1_macro, epoch)
        writer_tb.add_scalar("F1/val_minority", val_f1_minority, epoch)
        writer_tb.add_scalar("Score/weighted_f1", combined_score, epoch)

        if combined_score > best_score + min_delta:
            best_score = combined_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.success(
                f"New best model at epoch {epoch} — WeightedF1Score={combined_score:.4f}"
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.warning(
                f"Early stopping: no improvement for {patience} epochs "
                f"(best epoch = {best_epoch}, best score = {best_score:.4f})"
            )
            break

    logger.info("Loading best model checkpoint...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_acc, test_f1, test_preds, test_logits = eval_step(
        model,
        x,
        edge_index,
        edge_type,
        y,
        test_mask,
    )

    logger.success(f"TEST — Acc={test_acc:.4f}, F1={test_f1:.4f}")

    inference_dict = {
        "test_idx": test_mask.cpu(),
        "true_labels": y[test_mask].cpu(),
        "preds": test_preds,
        "logits": test_logits,
    }

    torch.save(inference_dict, out_dir / "inference_results.pt")
    logger.info("Saved inference_results.pt")
