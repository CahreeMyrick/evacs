from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from evacs.config import Config
from evacs.dataset import load_split_csv, Sample
from evacs.io import load_wav
from evacs.preprocess import to_mono, resample_linear, normalize_peak, pad_or_trim
from evacs.features import log_mel
from evacs.torch_cnn import SmallLogMelCNN

class LogMelDataset(Dataset):
    def __init__(self, samples: List[Sample], cfg: Config, label_to_idx: dict, duration_tol: float = 0.05):
        self.samples = samples
        self.cfg = cfg
        self.label_to_idx = label_to_idx
        self.duration_tol = duration_tol

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]

        # allow tiny overshoot; then trim/pad fixes exact length
        buf = load_wav(s.path, max_duration_sec=self.cfg.clip_sec + self.duration_tol)
        buf = to_mono(buf)
        buf = resample_linear(buf, target_sr=self.cfg.target_sr)
        buf = normalize_peak(buf, peak=0.99)
        buf = pad_or_trim(buf, clip_sec=self.cfg.clip_sec)

        feats = log_mel(buf, self.cfg)             # (n_mels, n_frames), float32
        X = feats.X[None, :, :]                    # (1, n_mels, n_frames)

        y = self.label_to_idx[s.label]
        return torch.from_numpy(X).float(), torch.tensor(y, dtype=torch.long)

def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return (correct / total) if total > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data root with class folders")
    ap.add_argument("--config", default=None, help="path to JSON config (optional)")
    ap.add_argument("--out", default="models/cnn_logmel.pt", help="output TorchScript model path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--duration_tol", type=float, default=0.05, help="allow +seconds over clip length")
    ap.add_argument("--patience", type=int, default=7, help="early stop after N epochs with no val improvement")
    ap.add_argument("--min_delta", type=float, default=0.0, help="minimum val improvement to count (e.g., 0.001)")
    ap.add_argument("--splits_dir", default="splits", help="Directory containing train.csv/val.csv/test.csv")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labels = cfg.labels
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    splits_dir = Path(args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv   = splits_dir / "val.csv"
    test_csv  = splits_dir / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise SystemExit(
            f"Missing splits in {splits_dir}. Run:\n"
            f"  python -m scripts.make_splits --data {args.data} --outdir {splits_dir} --seed {args.seed}"
        )

    train_s = load_split_csv(str(train_csv))
    val_s   = load_split_csv(str(val_csv))
    test_s  = load_split_csv(str(test_csv))
    print(f"Samples: train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    
    train_ds = LogMelDataset(train_s, cfg, label_to_idx, duration_tol=args.duration_tol)
    val_ds   = LogMelDataset(val_s, cfg, label_to_idx, duration_tol=args.duration_tol)
    test_ds  = LogMelDataset(test_s, cfg, label_to_idx, duration_tol=args.duration_tol)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = args.device
    model = SmallLogMelCNN(num_classes=len(labels)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running += float(loss.item()) * y.numel()
            n += int(y.numel())

        train_loss = running / max(n, 1)
        val_acc = evaluate(model, val_loader, device)

        improved = (val_acc > best_val + args.min_delta)
        if improved:
            best_val = val_acc
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            train_acc = evaluate(model, train_loader, device)
            print(
                f"epoch {epoch:3d}  loss {train_loss:.4f}  "
                f"acc_tr {train_acc:.3f}  acc_val {val_acc:.3f}  "
                f"best_val {best_val:.3f}  bad {bad_epochs}/{args.patience}"
            )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = evaluate(model, test_loader, device)
    print(f"TEST accuracy: {test_acc:.3f}")

    # Export TorchScript for deployment (no Python class needed at inference)
    model.eval()
    example = torch.zeros((1, 1, cfg.n_mels, 10), dtype=torch.float32)  # frames can vary
    scripted = torch.jit.trace(model.cpu(), example)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(outp))
    print(f"Saved TorchScript model → {outp}")

    # Save labels alongside for inference
    labels_path = outp.with_suffix(".labels.txt")
    labels_path.write_text("\n".join(labels))
    print(f"Saved labels → {labels_path}")

if __name__ == "__main__":
    main()