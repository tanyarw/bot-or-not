from pathlib import Path
import torch


def save_tensor(tensor: torch.Tensor, path: Path, not_tensor=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not_tensor:
        torch.save(tensor, path)
    else:
        torch.save(tensor.cpu(), path)


def load_tensor(path: Path, device="mps", not_tensor=False):

    if not_tensor:
        return torch.load(path)
    return torch.load(path, map_location="cpu").to(device)
