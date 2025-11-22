import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RandomNodeSplit

load_dotenv()
DATASET_ROOT = Path(os.getenv("DATASET_ROOT")).expanduser().resolve()
BOTRGCN_ROOT = Path(os.getenv("BOTRGCN_ROOT")).expanduser().resolve()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import BotRGCN
from config import load_config


# -----------------------
# Helper: one train step
# -----------------------
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


# -----------------------
# Helper: one eval step
# -----------------------
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

    input_dir = Path(cfg["paths"]["out_dir"])
    out_dir = Path(cfg["botrgcn"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = input_dir / "static_user_hetero_graph.pt"
    logger.info(f"Loading graph from {data_path}")

    raw_data = torch.load(data_path, weights_only=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")


    if isinstance(raw_data, HeteroData):
        logger.info("Detected HeteroData. Converting to homogeneous tensors...")

        hetero_data: HeteroData = raw_data.to(device)

        # assume only 'user' nodes are present / relevant
        num_nodes = hetero_data["user"].num_nodes

        val_ratio = cfg["botrgcn"]["split"]["val"]
        test_ratio = cfg["botrgcn"]["split"]["test"]
        num_val = int(val_ratio * num_nodes)
        num_test = int(test_ratio * num_nodes)

        splitter = RandomNodeSplit(
            split="train_rest",
            num_val=num_val,
            num_test=num_test,
            # key="user",
        )
        hetero_data = splitter(hetero_data)

        logger.info(
            f"Mask sizes (user) — "
            f"Train={hetero_data['user'].train_mask.sum()} "
            f"Val={hetero_data['user'].val_mask.sum()} "
            f"Test={hetero_data['user'].test_mask.sum()}"
        )

        # Node features & labels
        x = hetero_data["user"].x  # (N, F)
        y = hetero_data["user"].y  # (N,)

        train_mask = hetero_data["user"].train_mask
        val_mask = hetero_data["user"].val_mask
        test_mask = hetero_data["user"].test_mask

        # ('user', 'followers', 'user') → relation id 0
        # ('user', 'following', 'user') → relation id 1
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

    elif isinstance(raw_data, Data):
        logger.info("Detected homogeneous Data. Using directly.")

        data: Data = raw_data.to(device)

        num_nodes = data.num_nodes
        val_ratio = cfg["botrgcn"]["split"]["val"]
        test_ratio = cfg["botrgcn"]["split"]["test"]
        num_val = int(val_ratio * num_nodes)
        num_test = int(test_ratio * num_nodes)

        splitter = RandomNodeSplit(
            split="train_rest",
            num_val=num_val,
            num_test=num_test,
        )
        data = splitter(data)

        logger.info(
            f"Mask sizes — "
            f"Train={data.train_mask.sum()} "
            f"Val={data.val_mask.sum()} "
            f"Test={data.test_mask.sum()}"
        )

        x = data.x
        y = data.y
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

        edge_index = data.edge_index
        if hasattr(data, "edge_type"):
            edge_type = data.edge_type
        else:
            raise ValueError("Homogeneous Data object is missing `edge_type`.")

        x = x.to(device)
        y = y.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)

    else:
        raise TypeError(f"Unsupported data type: {type(raw_data)}")

    model = BotRGCN(
        in_channels=x.shape[1],
        hidden_channels=128,
        num_relations=2,  # 0: followers, 1: following
        dropout=0.30,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
    patience_counter = 0

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

        val_acc, val_f1, val_preds, _ = eval_step(
            model,
            x,
            edge_index,
            edge_type,
            y,
            val_mask,
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
            f"MacroF1={val_f1_macro:.4f} | "
            f"MinorityF1={val_f1_minority:.4f} | "
            f"Score={combined_score:.4f}"
        )

        if combined_score > best_score + min_delta:
            best_score = combined_score
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.success(
                f"New best model at epoch {epoch} — Score={combined_score:.4f}"
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
