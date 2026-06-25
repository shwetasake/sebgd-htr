"""
HVLT Training Script
====================
Key training decisions from the paper:
- Optimizer: Adam, lr=5e-5 (constant — no LR schedule, by design)
- Batch size: 32
- 3 independent seeds for reproducibility
- Early stopping based on VALIDATION WAR (not loss!)
  → Paper identifies Epoch 3 as optimal; WAR collapses from Epoch 6
- ACG auxiliary loss weight λ=0.1
- GPU: NVIDIA RTX 4090 (~14 min/epoch)

About the test set without ground truth:
→ We skip it (correct approach). Use the val split for evaluation.
  The test set without labels cannot be used for evaluation.
  We split train.txt into 85% train / 15% val (matching paper's 60/15 split ratio).
"""

import os
import sys
import time
import random
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.amp import GradScaler, autocast

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset   import get_dataloaders, decode_tokens
from models.hvlt    import HVLT, HVLTLoss
from utils.metrics  import MetricTracker, decode_batch_predictions


# ─── Config ────────────────────────────────────────────────────────────────────

def get_config():
    parser = argparse.ArgumentParser(description="Train HVLT on ICDAR Word-Level dataset")

    # Dataset
    parser.add_argument("--train_txt",  type=str,
                        default="/home/mca/Shweta/handwritten text detection/dataset/"
                                "Word_Level_English_Training_Set/Word_Level_Training_Set/train.txt",
                        help="Path to train.txt")
    parser.add_argument("--root_dir",   type=str,
                        default="/home/mca/Shweta/handwritten text detection/dataset/"
                                "Word_Level_English_Training_Set/Word_Level_Training_Set",
                        help="Root dir containing 'image/' subfolder")
    parser.add_argument("--val_split",  type=float, default=0.15)

    # Model
    parser.add_argument("--img_height",        type=int,   default=32)
    parser.add_argument("--img_width",         type=int,   default=128)
    parser.add_argument("--num_fiducial",      type=int,   default=16)   # K=16
    parser.add_argument("--d_model",           type=int,   default=768)
    parser.add_argument("--n_heads",           type=int,   default=12)
    parser.add_argument("--n_layers",          type=int,   default=12)
    parser.add_argument("--vis_seq_len",       type=int,   default=256)
    parser.add_argument("--acg_dropout",       type=float, default=0.3)
    parser.add_argument("--acg_lambda",        type=float, default=0.1)
    parser.add_argument("--no_pretrained_swin",    action="store_true")
    parser.add_argument("--no_pretrained_roberta", action="store_true")

    # Training
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=5e-5)     # Paper: Adam, 5e-5
    parser.add_argument("--max_epochs",  type=int,   default=10)       # Paper stops at epoch 3
    parser.add_argument("--patience",    type=int,   default=3,
                        help="Early stopping patience (WAR-based)")
    parser.add_argument("--seeds",       type=int,   nargs="+",
                        default=[42, 123, 7],                          # 3 seeds as in paper
                        help="Random seeds for 3 independent runs")
    parser.add_argument("--use_amp",     action="store_true",
                        help="Use automatic mixed precision (fp16)")
    parser.add_argument("--num_workers", type=int,   default=4)

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/")
    parser.add_argument("--log_every",  type=int, default=50,
                        help="Log every N batches")

    return parser.parse_args()


# ─── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─── Training Epoch ────────────────────────────────────────────────────────────

# ─── Training Epoch ────────────────────────────────────────────────────────────

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    log_every,
    epoch,
):

    model.train()

    tracker = MetricTracker()

    t0 = time.time()

    for batch_idx, (images, targets, labels) in enumerate(loader):

        images = images.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)

        # Dummy ACG labels
        acg_labels = torch.zeros(
            images.shape[0],
            device=device,
        )

        optimizer.zero_grad()

        # ── Proper autoregressive shifting ────────────────────────────────
        decoder_input  = targets[:, :-1]
        decoder_target = targets[:, 1:]

        # ── Mixed precision ───────────────────────────────────────────────
        if scaler is not None:

            with autocast("cuda"):

                logits, acg_gate = model(
                    images,
                    decoder_input,
                    acg_labels,
                )

                loss_dict = criterion(
                    logits,
                    decoder_target,
                    acg_gate,
                    acg_labels,
                )

            scaler.scale(loss_dict["loss"]).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            logits, acg_gate = model(
                images,
                decoder_input,
                acg_labels,
            )

            loss_dict = criterion(
                logits,
                decoder_target,
                acg_gate,
                acg_labels,
            )

            loss_dict["loss"].backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

        # ── TRAINING metrics (teacher-forced approximation) ──────────────
        with torch.no_grad():

            pred_ids = logits.argmax(dim=-1)

            preds = decode_batch_predictions(pred_ids)

        tracker.update(
            loss_dict,
            preds,
            labels,
        )

        # ── Logging ───────────────────────────────────────────────────────
        if (batch_idx + 1) % log_every == 0:

            stats = tracker.compute()

            elapsed = time.time() - t0

            print(
                f"  [E{epoch} B{batch_idx+1}/{len(loader)}] "
                f"loss={stats['loss']:.4f} "
                f"CAR={stats['CAR']:.2f}% "
                f"WAR={stats['WAR']:.2f}% "
                f"({elapsed:.1f}s)"
            )

    return tracker.compute()


# ─── Validation Epoch ──────────────────────────────────────────────────────────
@torch.no_grad()
def validate_epoch(model, loader, criterion, device):

    model.eval()

    tracker = MetricTracker()

    for images, targets, labels in loader:

        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Dummy ACG labels
        acg_labels = torch.zeros(images.shape[0], device=device)

        # ── Proper autoregressive shifting ───────────────────────────────────
        decoder_input  = targets[:, :-1]
        decoder_target = targets[:, 1:]

        # ── Validation forward pass ──────────────────────────────────────────
        logits, acg_gate = model(
            images,
            decoder_input,
            acg_labels,
        )

        loss_dict = criterion(
            logits,
            decoder_target,
            acg_gate,
            acg_labels,
        )

        # ── REAL autoregressive inference ────────────────────────────────────
        # THIS is the important fix.
        pred_ids = model.predict(images)

        preds = decode_batch_predictions(pred_ids)

        tracker.update(loss_dict, preds, labels)

    return tracker.compute()

# ─── Single Run ────────────────────────────────────────────────────────────────

def run_single_seed(cfg, seed: int, run_idx: int):
    print(f"\n{'='*60}")
    print(f"  Run {run_idx+1}/3  |  Seed: {seed}")
    print(f"{'='*60}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Output directory for this run
    run_dir = Path(cfg.output_dir) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    print("\n[1/4] Loading data...")
    train_loader, val_loader = get_dataloaders(
        train_txt=cfg.train_txt,
        root_dir=cfg.root_dir,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        num_workers=cfg.num_workers,
    )

    # ── Model ──
    print("\n[2/4] Building HVLT model...")
    model = HVLT(
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        num_fiducial=cfg.num_fiducial,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        vis_seq_len=cfg.vis_seq_len,
        acg_dropout=cfg.acg_dropout,
        acg_lambda=cfg.acg_lambda,
        pretrained_swin=not cfg.no_pretrained_swin,
        pretrained_roberta=not cfg.no_pretrained_roberta,
    ).to(device)

    n_params = model.count_parameters()
    print(f"  Parameters: {n_params/1e6:.1f}M")

    # ── Optimizer & Loss ──
    # Paper: Adam, constant lr=5e-5 (no scheduler — by design for divergence analysis)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = HVLTLoss(acg_lambda=cfg.acg_lambda)
    scaler    = GradScaler("cuda") if cfg.use_amp and torch.cuda.is_available() else None

    # ── Training Loop ──
    print("\n[3/4] Training...")
    print("  NOTE: Paper finds optimal checkpoint at Epoch 3.")
    print("        WAR-based early stopping will handle this automatically.\n")

    best_war     = -1.0
    best_epoch   = -1
    patience_cnt = 0
    history      = []

    for epoch in range(1, cfg.max_epochs + 1):
        print(f"\n─── Epoch {epoch}/{cfg.max_epochs} (seed={seed}) ───")
        t_start = time.time()

        train_stats = train_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, cfg.log_every, epoch,
        )
        val_stats = validate_epoch(model, val_loader, criterion, device)

        epoch_time = time.time() - t_start

        print(
            f"  [Epoch {epoch}] "
            f"Train: loss={train_stats['loss']:.4f} "
            f"CAR={train_stats['CAR']:.2f}% "
            f"WAR={train_stats['WAR']:.2f}%  |  "
            f"Val: loss={val_stats['loss']:.4f} "
            f"CAR={val_stats['CAR']:.2f}% "
            f"WAR={val_stats['WAR']:.2f}%  |  "
            f"Time: {epoch_time/60:.1f}min"
        )

        # Log epoch stats
        epoch_log = {
            "epoch": epoch,
            "train": train_stats,
            "val":   val_stats,
            "time":  epoch_time,
        }
        history.append(epoch_log)

        # Save history after each epoch
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── WAR-based Early Stopping ────────────────────────────────────────
        # Critical: paper shows WAR collapses to 0% from epoch 6.
        # We checkpoint on VALIDATION WAR, not loss.
        val_war = val_stats["WAR"]

        if val_war > best_war:
            best_war   = val_war
            best_epoch = epoch
            patience_cnt = 0

            # Save best checkpoint
            ckpt_path = run_dir / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_car":     val_stats["CAR"],
                "val_war":     val_war,
                "train_car":   train_stats["CAR"],
                "train_war":   train_stats["WAR"],
                "seed":        seed,
                "config":      vars(cfg),
            }, ckpt_path)
            print(f"  ✓ Best checkpoint saved (WAR={best_war:.2f}% at epoch {best_epoch})")
        else:
            patience_cnt += 1
            print(f"  ✗ No WAR improvement. Patience: {patience_cnt}/{cfg.patience}")

        # Collapse detection: if WAR drops to near 0, stop immediately
        if epoch > 3 and val_war < 1.0:
            print(f"\n  [!] SEQUENCE MEMORISATION COLLAPSE detected at epoch {epoch}.")
            print(f"      Val WAR={val_war:.2f}% — early stopping NOW.")
            print(f"      Best epoch was {best_epoch} with WAR={best_war:.2f}%")
            break

        if patience_cnt >= cfg.patience:
            print(f"\n  Early stopping triggered (patience={cfg.patience} epochs).")
            break

    print(f"\n  ── Run {run_idx+1} Summary ──")
    print(f"     Best epoch: {best_epoch}")
    print(f"     Best val CAR: {history[best_epoch-1]['val']['CAR']:.2f}%")
    print(f"     Best val WAR: {best_war:.2f}%")

    return {
        "seed":       seed,
        "best_epoch": best_epoch,
        "best_war":   best_war,
        "best_car":   history[best_epoch-1]["val"]["CAR"],
        "run_dir":    str(run_dir),
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = get_config()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  HVLT — Hierarchical Vision-Language Transformer")
    print("  Handwritten Text Recognition on ICDAR Word-Level Dataset")
    print("="*60)
    print(f"\n  Training {len(cfg.seeds)} independent runs (seeds: {cfg.seeds})")
    print(f"  Train: {cfg.train_txt}")
    print(f"  Root:  {cfg.root_dir}")
    print(f"  Test set (no GT) → SKIPPED (correct approach)")
    print(f"  Val split: {cfg.val_split*100:.0f}%")
    print(f"  Batch size: {cfg.batch_size} | LR: {cfg.lr} (constant)")
    print(f"  Early stopping: VAL-WAR based (patience={cfg.patience})")
    print()

    all_results = []
    for i, seed in enumerate(cfg.seeds):
        result = run_single_seed(cfg, seed, i)
        all_results.append(result)

    # ── Aggregate results across seeds (mean ± std) ──
    print("\n" + "="*60)
    print("  FINAL RESULTS (across all seeds)")
    print("="*60)

    cars = [r["best_car"] for r in all_results]
    wars = [r["best_war"] for r in all_results]

    car_mean = np.mean(cars)
    car_std  = np.std(cars)
    war_mean = np.mean(wars)
    war_std  = np.std(wars)

    print(f"\n  CAR: {car_mean:.2f}% ± {car_std:.2f}%")
    print(f"  WAR: {war_mean:.2f}% ± {war_std:.2f}%")
    print()
    for r in all_results:
        print(f"  Seed {r['seed']}: CAR={r['best_car']:.2f}%  WAR={r['best_war']:.2f}%  "
              f"(epoch {r['best_epoch']})")

    # Save final summary
    summary = {
        "runs":     all_results,
        "car_mean": car_mean,
        "car_std":  car_std,
        "war_mean": war_mean,
        "war_std":  war_std,
    }
    with open(Path(cfg.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()