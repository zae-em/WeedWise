"""Train only through an explicit CLI invocation; imports never start training."""
import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm.auto import tqdm

from .data import FolderImages, discover, image_transforms, NORMALIZE_MEAN, NORMALIZE_STD


@dataclass
class Config:
    data_dir: str = "dataset/extracted"
    output_dir: str = "models/mobilenet_v2"
    evaluation_dir: str = "evaluation/mobilenet_v2"
    image_size: int = 224
    batch_size: int = 32
    head_epochs: int = 5
    finetune_epochs: int = 20
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-4
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True
    resume: str | None = None
    finetune_from: int = 14
    class_balance: bool = True

    def validate(self):
        if self.image_size < 32 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("image_size>=32, batch_size>=1 and num_workers>=0 required")
        if self.head_epochs < 0 or self.finetune_epochs < 0 or self.head_epochs + self.finetune_epochs < 1:
            raise ValueError("At least one non-negative training phase must have epochs")
        if self.patience < 1 or self.head_lr <= 0 or self.finetune_lr <= 0 or self.min_delta < 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer or early stopping configuration")


def inspect_data(config):
    cfg = Config(**config) if isinstance(config, dict) else config
    classes, splits, _, fingerprint = discover(cfg.data_dir, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    return {"classes": classes, "counts": {k: len(v) for k, v in splits.items()}, "fingerprint": fingerprint}


def build_model(num_classes, pretrained=True):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def choose_device(requested):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Connect a Colab GPU kernel.")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def topk_counts(logits, targets):
    k = min(5, logits.shape[1])
    predicted = logits.topk(k, dim=1).indices
    top1 = predicted[:, 0].eq(targets).sum().item()
    top5 = predicted.eq(targets[:, None]).any(dim=1).sum().item()
    return int(top1), int(top5), k


def run_epoch(model, loader, device, optimizer=None, frozen_backbone=False, class_weights=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training and frozen_backbone:
        model.features.eval()  # Keep frozen batch norm statistics fixed.
    elif training and hasattr(model, "features"):
        for block in model.features.children():
            if not any(p.requires_grad for p in block.parameters()):
                block.eval()
    loss_sum = top1 = top5 = count = 0
    actual, predicted = [], []
    criterion = nn.CrossEntropyLoss(weight=class_weights if training else None)
    with torch.set_grad_enabled(training):
        for images, targets in tqdm(loader, leave=False, desc="train" if training else "evaluate"):
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; inspect inputs and learning rate before resuming.")
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            n = targets.size(0)
            a, b, k = topk_counts(logits.detach(), targets)
            count += n
            loss_sum += loss.item() * n
            top1 += a
            top5 += b
            actual.extend(targets.detach().cpu().tolist())
            predicted.extend(logits.detach().argmax(1).cpu().tolist())
    if not count:
        raise ValueError("Empty data loader")
    return {"loss": loss_sum/count, "top1": top1/count, "top5": top5/count, "top5_effective_k": k}, actual, predicted


def atomic_save(value, target):
    target = Path(target)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
            "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    generator.set_state(state["loader"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def evaluate(model, loader, device, classes, destination, validation_loader=None, history=None):
    if validation_loader is not None:
        from .evaluation import evaluate_complete
        return evaluate_complete(model, loader, validation_loader, device, classes, destination, history or [])
    metrics, actual, predicted = run_epoch(model, loader, device)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(classes)))
    report = classification_report(actual, predicted, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    (destination / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (destination / "classification_report.json").write_text(json.dumps(report, indent=2))
    np.savetxt(destination / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    (destination / "class_names.json").write_text(json.dumps(classes, indent=2))
    # Bound figure dimensions for large species inventories; CSV retains full precision.
    side = min(30, max(8, len(classes) * 0.3))
    fig, ax = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, xticks_rotation=90, include_values=len(classes) <= 25, colorbar=False)
    fig.tight_layout()
    fig.savefig(destination / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    return metrics


def train(cfg):
    cfg.validate()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = choose_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    classes, splits, manifest, fingerprint = discover(cfg.data_dir, cfg.val_fraction, cfg.test_fraction, cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and any((output / name).exists() for name in ["last.pt", "best.pt"]):
        raise FileExistsError("Existing checkpoints: use resume or choose a new output_dir.")
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = {name: DataLoader(FolderImages(rows, image_transforms(cfg.image_size, name == "train")),
                               batch_size=cfg.batch_size, shuffle=name == "train", num_workers=cfg.num_workers,
                               pin_memory=device.type == "cuda", generator=generator if name == "train" else None)
               for name, rows in splits.items()}
    model = build_model(len(classes), cfg.pretrained and not cfg.resume).to(device)
    start, best_loss, bad_epochs, history = 0, float("inf"), 0, []
    best_model, best_epoch = None, None
    previous_stage, optimizer, checkpoint = None, None, None
    scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    counts = torch.bincount(torch.tensor([row["label"] for row in splits["train"]]), minlength=len(classes)).float()
    class_weights = (counts.sum() / (len(classes) * counts)).to(device) if cfg.class_balance else None
    if cfg.resume:
        # Load checkpoints created by this project. weights_only prevents arbitrary pickle execution.
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_kind") != "last":
            raise ValueError("Resume from last.pt, which contains matching optimizer and recovery state.")
        if checkpoint["classes"] != classes or checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Resume dataset/classes differ from the saved checkpoint")
        critical = ["image_size", "batch_size", "head_epochs", "head_lr", "finetune_lr", "weight_decay", "seed", "val_fraction", "test_fraction", "min_delta", "patience", "pretrained", "finetune_from", "class_balance"]
        if any(checkpoint["config"][key] != getattr(cfg, key) for key in critical):
            raise ValueError("Resume requires the same training and split configuration")
        model.load_state_dict(checkpoint["model"])
        start, best_loss, bad_epochs = checkpoint["epoch"] + 1, checkpoint["best_loss"], checkpoint["bad_epochs"]
        history, previous_stage = checkpoint["history"], checkpoint["stage"]
        best_model, best_epoch = checkpoint["best_model"], checkpoint["best_epoch"]
        atomic_save({"checkpoint_kind": "best", "model": best_model, "classes": classes,
                     "fingerprint": fingerprint, "best_loss": best_loss, "epoch": best_epoch}, output / "best.pt")
        restore_rng(checkpoint["rng"], generator)
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (output / "classes.json").write_text(json.dumps(classes, indent=2))
    print(f"Device: {device}; GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}")
    for epoch in range(start, cfg.head_epochs + cfg.finetune_epochs):
        stage = "head" if epoch < cfg.head_epochs else "finetune"
        if epoch == cfg.head_epochs and previous_stage == "head":
            bad_epochs = 0  # Give fine-tuning its own patience window.
        if bad_epochs >= cfg.patience and stage == previous_stage:
            if stage == "head" and cfg.finetune_epochs:
                continue
            print("Saved checkpoint already reached early stopping.")
            break
        for parameter in model.features.parameters():
            parameter.requires_grad = stage == "finetune"
        if stage == "finetune":
            for index, block in enumerate(model.features.children()):
                for parameter in block.parameters():
                    parameter.requires_grad = index >= cfg.finetune_from
        if optimizer is None or stage != previous_stage:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                         lr=cfg.head_lr if stage == "head" else cfg.finetune_lr, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
            if checkpoint is not None and stage == checkpoint["stage"]:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
        train_metrics, _, _ = run_epoch(model, loaders["train"], device, optimizer, stage == "head", class_weights, scaler)
        val_metrics, _, _ = run_epoch(model, loaders["val"], device)
        scheduler.step(val_metrics["loss"])
        improved = val_metrics["loss"] < best_loss - cfg.min_delta
        if improved:
            best_loss, bad_epochs = val_metrics["loss"], 0
            best_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "stage": stage, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics})
        state = {"format_version": 1, "checkpoint_kind": "last", "architecture": "mobilenet_v2", "epoch": epoch, "stage": stage,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_loss": best_loss,
                 "best_model": best_model, "best_epoch": best_epoch,
                 "bad_epochs": bad_epochs, "classes": classes, "fingerprint": fingerprint,
                 "config": asdict(cfg), "history": history, "rng": rng_state(generator)}
        state.update(scheduler=scheduler.state_dict(), scaler=scaler.state_dict())
        if improved:
            atomic_save({"checkpoint_kind": "best", "model": best_model, "classes": classes,
                         "fingerprint": fingerprint, "best_loss": best_loss, "epoch": best_epoch}, output / "best.pt")
        atomic_save(state, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
        (output / "training_status.json").write_text(json.dumps({"state":"training","completed_epoch":epoch+1,"total_epochs":cfg.head_epochs+cfg.finetune_epochs,"stage":stage,"latest":history[-1],"best_validation_loss":best_loss},indent=2))
        print(json.dumps(history[-1]))
        previous_stage, checkpoint = stage, None
        if bad_epochs >= cfg.patience:
            if stage == "head" and cfg.finetune_epochs:
                print("Head patience reached; skipping remaining head epochs before fine-tuning.")
            else:
                print("Early stopping")
                break
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    metrics = evaluate(model, loaders["test"], device, classes, cfg.evaluation_dir, validation_loader=loaders["val"], history=history)
    calibration_path = Path(cfg.evaluation_dir) / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.is_file() else {"temperature":1.0,"unknown_threshold":0.5,"fitted":False}
    atomic_save({"architecture": "mobilenet_v2", "model": model.cpu().state_dict(), "classes": classes,
                 "image_size": cfg.image_size, "mean": NORMALIZE_MEAN, "std": NORMALIZE_STD,
                 "temperature":calibration["temperature"],"unknown_threshold":calibration["unknown_threshold"],"calibration":calibration,
                 "fingerprint":fingerprint,"preprocessing":"letterbox_rgb_224_imagenet_normalization","trained":True}, output / "model.pt")
    (output / "training_status.json").write_text(json.dumps({"state":"complete","completed_epochs":len(history),"best_epoch":best_epoch,"test_metrics":metrics},indent=2))
    print("Held-out test metrics:", json.dumps(metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/config.json")
    parser.add_argument("--inspect-only", action="store_true", help="Validate dataset layout without training")
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.inspect_only:
        print(json.dumps(inspect_data(cfg), indent=2))
    else:
        train(cfg)


if __name__ == "__main__":
    main()
