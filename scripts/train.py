from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import json

from evacs.config import Config
from evacs.dataset import load_split_csv
from evacs.io import load_wav
from evacs.preprocess import to_mono, resample_linear, normalize_peak, pad_or_trim
from evacs.features import log_mel, vectorize_logmel

def _one_hot(y: np.ndarray, K: int) -> np.ndarray:
    Y = np.zeros((y.shape[0], K), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y

def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float32)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    E = np.exp(Z)
    return E / (np.sum(E, axis=1, keepdims=True) + 1e-12)

def _loss_ce(P: np.ndarray, Y: np.ndarray) -> float:
    # cross-entropy
    return float(-np.mean(np.sum(Y * np.log(P + 1e-12), axis=1)))

def _accuracy(P: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(P, axis=1)
    return float(np.mean(pred == y))

def extract_vector(path: str, cfg: Config) -> np.ndarray:
    buf = load_wav(path, max_duration_sec=cfg.clip_sec, tolerance_sec=cfg.duration_tolerance_sec)
    buf = to_mono(buf)
    buf = resample_linear(buf, target_sr=cfg.target_sr)
    buf = normalize_peak(buf, peak=0.99)
    buf = pad_or_trim(buf, clip_sec=cfg.clip_sec)
    feats = log_mel(buf, cfg)
    v = vectorize_logmel(feats)
    return v

def build_dataset(samples, cfg: Config, label_to_idx: dict) -> tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    for s in samples:
        v = extract_vector(s.path, cfg)
        X_list.append(v)
        y_list.append(label_to_idx[s.label])
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y

def main():
    ap = argparse.ArgumentParser(description="Train softmax-linear baseline on log-mel vectors (NumPy only).")
    ap.add_argument("--data", required=True, help="Dataset root with class subfolders")
    ap.add_argument("--config", default=None, help="Path to JSON config (optional)")
    ap.add_argument("--out", default=None, help="Output model path (overrides cfg.model_path)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--splits_dir", default="splits", help="Directory containing train.csv/val.csv/test.csv")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    out_path = args.out if args.out else cfg.model_path

    labels = cfg.labels
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    K = len(labels)

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

    if not (train_s and val_s and test_s):
        raise SystemExit("One or more split files are empty.")

    print(f"Samples: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    print("Extracting features...")
    Xtr, ytr = build_dataset(train_s, cfg, label_to_idx)
    Xva, yva = build_dataset(val_s, cfg, label_to_idx)
    Xte, yte = build_dataset(test_s, cfg, label_to_idx)

    # Standardize using train stats
    x_mean = Xtr.mean(axis=0)
    x_std = Xtr.std(axis=0) + 1e-6
    Xtrn = (Xtr - x_mean) / x_std
    Xvan = (Xva - x_mean) / x_std
    Xten = (Xte - x_mean) / x_std

    N, D = Xtrn.shape
    rng = np.random.default_rng(args.seed)
    W = (0.01 * rng.standard_normal((K, D))).astype(np.float32)
    b = np.zeros((K,), dtype=np.float32)

    Ytr = _one_hot(ytr, K)

    best_val = -1.0
    best = None

    for epoch in range(1, args.epochs + 1):
        # forward
        Z = Xtrn @ W.T + b[None, :]    # (N, K)
        P = _softmax_rows(Z)

        # loss + grads
        loss = _loss_ce(P, Ytr) + 0.5 * args.l2 * float(np.sum(W * W))
        dZ = (P - Ytr) / N             # (N, K)
        dW = (dZ.T @ Xtrn) + args.l2 * W
        db = np.sum(dZ, axis=0)

        # update
        W -= args.lr * dW.astype(np.float32)
        b -= args.lr * db.astype(np.float32)

        # eval
        Pva = _softmax_rows(Xvan @ W.T + b[None, :])
        acc_va = _accuracy(Pva, yva)

        if acc_va > best_val:
            best_val = acc_va
            best = (W.copy(), b.copy())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            Ptr = _softmax_rows(Xtrn @ W.T + b[None, :])
            acc_tr = _accuracy(Ptr, ytr)
            print(f"epoch {epoch:4d}  loss {loss:.4f}  acc_tr {acc_tr:.3f}  acc_val {acc_va:.3f}")

    # restore best
    if best is not None:
        W, b = best

    Pte = _softmax_rows(Xten @ W.T + b[None, :])
    acc_te = _accuracy(Pte, yte)
    print(f"TEST accuracy: {acc_te:.3f}")

    model_artifact = {
        "kind": "softmax_linear",
        "labels": labels,
        "W": W.tolist(),
        "b": b.tolist(),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "feature": "logmel_mean_std",
        "config": cfg.to_dict(),
    }

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(model_artifact, indent=2))
    print(f"Saved model → {outp}")

if __name__ == "__main__":
    main()