"""CNN-005 controlled recovery: A=CE, B=CE+teacher KD.

Smoke mode runs one training batch, no validation, and writes no files.
Full mode trains on frozen 45k/5k split, never loads CIFAR-10 test data.
Run from repository root with python -m scripts.run_cnn005_recovery.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from scripts.prepare_cnn005 import EXPECTED, MANIFEST, TEACHER, load_student, sha256
from scripts.run_cnn_experiment_cnn003b_validation import build_model, evaluate

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loaders(root, batch_size, workers):
    manifest = json.loads(MANIFEST.read_text())
    train_indices, val_indices = manifest["train_indices"], manifest["val_indices"]
    assert manifest["seed"] == 42
    assert len(train_indices) == 45000 and len(val_indices) == 5000
    assert len(set(train_indices)) == 45000 and len(set(val_indices)) == 5000
    assert set(train_indices).isdisjoint(val_indices)
    assert set(train_indices) | set(val_indices) == set(range(50000))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    # Explicitly train=True; never construct the held-out test dataset.
    train_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=train_tf)
    val_ds = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=val_tf)
    assert len(train_ds) == len(val_ds) == 50000
    train_loader = DataLoader(Subset(train_ds, train_indices), batch_size=batch_size,
                              shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(val_ds, val_indices), batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--smoke", action="store_true", help="One train batch, no output files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    args = parser.parse_args()
    assert args.epochs > 0 and args.batch_size > 0 and args.workers >= 0
    if not args.smoke:
        assert args.epochs == 20, "Frozen protocol uses 20 epochs; change only with documented amendment"
        free_gb = shutil.disk_usage(".").free / 1024**3
        if free_gb < args.min_free_gb:
            raise RuntimeError(f"Only {free_gb:.2f} GiB free; require {args.min_free_gb:.2f} GiB for full run")
    assert torch.cuda.is_available(), "CNN-005 protocol requires CUDA T4; aborting"
    assert torch.version.cuda is not None
    import torch_pruning as tp
    assert tp.__version__ == "1.6.0"
    for path, expected in EXPECTED.items():
        assert sha256(path) == expected, f"Frozen input SHA mismatch: {path}"
    seed_everything(42)
    student, stripped = load_student()
    assert stripped == 36
    # load_student reseeds torch for deterministic pruning reconstruction.
    seed_everything(42)
    train_loader, val_loader = loaders(args.data_root, args.batch_size, args.workers)
    device = torch.device("cuda")
    student.to(device)
    teacher = None
    if args.arm == "B":
        teacher = build_model("resnet50", 10)
        teacher.load_state_dict(torch.load(TEACHER, map_location="cpu", weights_only=True), strict=True)
        teacher.to(device).eval()
        teacher.requires_grad_(False)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 16], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    name = "CNN-005" + args.arm
    output = Path("checkpoints") / name
    result = Path("results") / name
    if not args.smoke:
        if output.exists() or result.exists():
            raise FileExistsError(f"Refusing to overwrite prior {name} output")
        output.mkdir(parents=True)
        result.mkdir(parents=True)
    best_accuracy, best_epoch = -1.0, None
    history = []
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        student.train()
        total_loss, count = 0.0, 0
        for batch_index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = student(images)
                ce = F.cross_entropy(logits, labels)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                    # Float32 softmax/KL for numerical stability under AMP.
                    temperature = 4.0
                    kd = F.kl_div(
                        F.log_softmax(logits.float() / temperature, dim=1),
                        F.softmax(teacher_logits.float() / temperature, dim=1),
                        reduction="batchmean",
                    ) * temperature**2
                    loss = 0.5 * ce + 0.5 * kd
                else:
                    loss = ce
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item()
            count += 1
            if args.smoke:
                break
        if args.smoke:
            print(f"{name} SMOKE PASS: one training batch; loss={total_loss:.5f}; no files written; no test access")
            return
        accuracy, _, _ = evaluate(student, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / count,
               "val_accuracy": accuracy, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row), flush=True)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(student.state_dict(), output / "best_model.pt")
        scheduler.step()
    torch.save(student.state_dict(), output / "final_model.pt")
    summary = {
        "experiment": name, "arm": "supervised_ce" if args.arm == "A" else "teacher_kd",
        "seed": 42, "epochs": args.epochs, "batch_size": args.batch_size,
        "optimizer": "SGD", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4,
        "lr_milestones": [10, 16], "gamma": 0.1, "amp": True,
        "kd_temperature": 4.0 if args.arm == "B" else None,
        "kd_alpha": 0.5 if args.arm == "B" else None,
        "split_manifest_sha256": EXPECTED[MANIFEST],
        "student_initial_sha256": EXPECTED[next(p for p in EXPECTED if "pruned_before_recovery" in str(p))],
        "teacher_sha256": EXPECTED[TEACHER],
        "train_samples": 45000, "validation_samples": 5000,
        "test_evaluated": False, "best_val_accuracy": best_accuracy, "best_epoch": best_epoch,
        "history": history,
    }
    (result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{name} FINISHED: best validation={best_accuracy:.2f}% at epoch {best_epoch}; held-out test NOT accessed")


if __name__ == "__main__":
    main()
