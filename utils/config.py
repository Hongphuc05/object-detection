import os
from pathlib import Path

import torch


DEFAULT_CLASSES = ["person", "car", "dog", "cat", "chair"]

CHECKPOINT_FILES = {
    "best": "best.pth",
    "best_loss": "best_loss.pth",
    "last": "last.pth",
}


def detect_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_name(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    return torch.cuda.get_device_name(device)


def is_a100(device: torch.device) -> bool:
    return device.type == "cuda" and "A100" in get_gpu_name(device).upper()


def configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def resolve_batch_size(requested_batch_size: int | None, device: torch.device) -> int:
    if requested_batch_size is not None:
        return requested_batch_size
    return 64 if is_a100(device) else 8


def resolve_num_workers(requested_num_workers: int | None, device: torch.device) -> int:
    if requested_num_workers is not None:
        return requested_num_workers

    cpu_count = os.cpu_count() or 4
    if device.type != "cuda":
        return min(4, cpu_count)
    if is_a100(device):
        return min(12, cpu_count)
    return min(8, cpu_count)


def get_dataloader_kwargs(num_workers: int, device: torch.device) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 6 if is_a100(device) else 2
    return kwargs


def resolve_checkpoint_path(weights: str, checkpoint_dir: str, prefer_last: bool = False) -> Path | None:
    if weights:
        candidate = Path(weights)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            search_order = (
                [CHECKPOINT_FILES["last"], CHECKPOINT_FILES["best"], CHECKPOINT_FILES["best_loss"]]
                if prefer_last
                else [CHECKPOINT_FILES["best"], CHECKPOINT_FILES["best_loss"], CHECKPOINT_FILES["last"]]
            )
            for filename in search_order:
                resolved = candidate / filename
                if resolved.exists():
                    return resolved
            return None
        return None

    checkpoint_root = Path(checkpoint_dir)
    search_order = (
        [CHECKPOINT_FILES["last"], CHECKPOINT_FILES["best"], CHECKPOINT_FILES["best_loss"]]
        if prefer_last
        else [CHECKPOINT_FILES["best"], CHECKPOINT_FILES["best_loss"], CHECKPOINT_FILES["last"]]
    )
    for filename in search_order:
        resolved = checkpoint_root / filename
        if resolved.exists():
            return resolved
    return None


def torch_load_compat(path: str | Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint
