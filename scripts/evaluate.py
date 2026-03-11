from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path

from evacs.config import Config
from evacs.dataset import list_samples, dedupe_samples, load_split_csv
from evacs.pipeline import classify_file

def confusion_matrix(y_true, y_pred, labels):
    idx = {lab: i for i, lab in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        C[idx[t], idx[p]] += 1
    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="Root folder with class subfolders")
    ap.add_argument("--config", default=None, help="Path to JSON config")
    ap.add_argument("--model", default=None, help="Override model path")
    ap.add_argument("--splits_dir", default=None, help="Directory containing train.csv/val.csv/test.csv")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which frozen split to evaluate")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.model:
        cfg = Config(**{**cfg.to_dict(), "model_path": args.model})

    # Prefer frozen split evaluation if provided
    if args.splits_dir is not None:
        split_csv = Path(args.splits_dir) / f"{args.split}.csv"
        if not split_csv.exists():
            raise SystemExit(f"Missing split file: {split_csv}")
        samples = load_split_csv(str(split_csv))
        print(f"Evaluating frozen split: {split_csv} ({len(samples)} samples)")
    else:
        if args.data is None:
            raise SystemExit("Provide either --data or --splits_dir")
        samples = dedupe_samples(list_samples(args.data, labels=cfg.labels))
        print(f"Evaluating full dataset folder: {len(samples)} samples")

    if len(samples) == 0:
        raise SystemExit("No samples found.")

    y_true = []
    y_pred = []
    total_ms = []

    for s in samples:
        res = classify_file(s.path, cfg)
        y_true.append(s.label)
        y_pred.append(res.prediction.label)
        total_ms.append(res.stage_times.total_ms)

    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    C = confusion_matrix(y_true, y_pred, cfg.labels)

    print(f"accuracy: {acc:.4f}")
    print("confusion (rows=true, cols=pred):")
    print(cfg.labels)
    print(C)

    total_ms = np.asarray(total_ms, dtype=np.float32)
    print(
        f"latency ms: "
        f"p50={np.percentile(total_ms, 50):.1f} "
        f"p95={np.percentile(total_ms, 95):.1f} "
        f"mean={total_ms.mean():.1f}"
    )

if __name__ == "__main__":
    main()