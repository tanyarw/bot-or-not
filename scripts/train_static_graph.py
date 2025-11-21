import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = Path(os.getenv("DATASET_ROOT")).expanduser().resolve()
BOTRGCN_ROOT = Path(os.getenv("BOTRGCN_ROOT")).expanduser().resolve()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import BotRGCN
from config import load_config


# ----------------------------------------------------------
# Split helper
# ----------------------------------------------------------
def create_splits(num_nodes, train_ratio=0.6, val_ratio=0.2):
    """
    Randomly shuffle node indices and split into train/val/test.
    """
    perm = torch.randperm(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    return train_idx, val_idx, test_idx


# ----------------------------------------------------------
# Train step
# ----------------------------------------------------------
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()


# ----------------------------------------------------------
# Evaluation
# ----------------------------------------------------------
def evaluate(model, data, idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_type)
        logits = out[idx]
        preds = logits.argmax(dim=1)
        y_true = data.y[idx].cpu()

        acc = accuracy_score(y_true, preds.cpu())
        f1 = f1_score(y_true, preds.cpu(), average="macro")

    return acc, f1, preds.cpu(), logits.cpu()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    # ----------------------------
    # Load config
    # ----------------------------
    cfg = load_config("config.yaml")
    cfg["paths"]["out_dir"] = str(DATASET_ROOT)
    cfg["botrgcn"]["out_dir"] = str(BOTRGCN_ROOT)

    cfg_split = cfg["botrgcn"]["split"]

    input_dir = Path(cfg["paths"]["out_dir"])
    out_dir = Path(cfg["botrgcn"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    data_path = Path(input_dir) / "static_user_graph.pt"
    logger.info(f"Loading data from: {data_path}")

    data: Data = torch.load(data_path, weights_only=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    data = data.to(device)

    model = BotRGCN(
        in_channels=data.x.shape[1],
        hidden_channels=128,
        num_relations=2,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # create splits
    train_idx, val_idx, test_idx = create_splits(
        num_nodes=data.x.shape[0],
        train_ratio=cfg_split["train"],
        val_ratio=cfg_split["val"],
    )

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    logger.info(
        f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    # training
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0

    best_model_path = out_dir / "bot_rgcn.pt"

    for epoch in range(1, 100 + 1):
        loss = train(model, data, train_idx, optimizer)
        val_acc, val_f1, preds, _ = evaluate(model, data, val_idx)
        print(torch.bincount(preds))

        logger.info(
            f"Epoch {epoch:03d} | Loss={loss:.4f} | ValAcc={val_acc:.4f} | ValF1={val_f1:.4f}"
        )

        # early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.success("Saved new best model")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.warning("Early stopping triggered.")
            break

    # evaluate
    logger.info("Loading best model checkpoint...")
    model.load_state_dict(torch.load(best_model_path))

    test_acc, test_f1, test_preds, test_logits = evaluate(model, data, test_idx)
    logger.success(f"TEST RESULTS — Acc={test_acc:.4f}, F1={test_f1:.4f}")

    # inference results
    inference_dict = {
        "test_idx": test_idx.cpu(),
        "true_labels": data.y[test_idx].cpu(),
        "preds": test_preds,
        "logits": test_logits,
    }

    torch.save(inference_dict, out_dir / "inference_results.pt")
    logger.info("Saved inference_results.pt")


if __name__ == "__main__":
    main()
