import hashlib
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def seed_everything(seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": str(np.__version__),
            "os": platform.platform(), "cpu": platform.processor(),
            "threads": torch.get_num_threads(), "device": "cpu",
            "logical_cpus": os.cpu_count()}


def make_loaders(data_dir, batch_size, seed, workers=0, test=False):
    norm = transforms.Normalize(MEAN, STD)
    evaluation = transforms.Compose([transforms.ToTensor(), norm])
    gen = torch.Generator().manual_seed(seed)
    if test:
        ds = datasets.CIFAR10(data_dir, train=False, transform=evaluation, download=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    augmentation = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), norm])
    train = datasets.CIFAR10(data_dir, train=True, transform=augmentation, download=False)
    val = datasets.CIFAR10(data_dir, train=True, transform=evaluation, download=False)
    split_path = ROOT / "configs" / "split_seed2026.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        targets = np.array(train.targets)
        rng = np.random.default_rng(2026)
        train_ids, val_ids = [], []
        for cls in range(10):
            ids = rng.permutation(np.flatnonzero(targets == cls))
            val_ids.extend(ids[:500].tolist())
            train_ids.extend(ids[500:].tolist())
        split = {"seed": 2026, "train": train_ids, "val": val_ids,
                 "description": "4500 train + 500 validation per class; official test untouched"}
        write_json(split_path, split)
    train_ids, val_ids = split["train"], split["val"]
    assert len(train_ids) == 45000 and len(val_ids) == 5000
    assert len(set(train_ids) | set(val_ids)) == 50000
    assert set(train_ids).isdisjoint(val_ids)
    return (DataLoader(Subset(train, train_ids), batch_size=batch_size, shuffle=True,
                       num_workers=workers, generator=gen),
            DataLoader(Subset(val, val_ids), batch_size=batch_size, shuffle=False,
                       num_workers=workers))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for x, y in loader:
        logits = model(x)
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.numel()
        confusion += torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
    return {"loss": loss_sum / n, "accuracy_pct": 100.0 * correct / n,
            "n": n, "confusion": confusion.tolist()}
