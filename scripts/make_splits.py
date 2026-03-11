from __future__ import annotations
import argparse
from pathlib import Path

from evacs.config import Config
from evacs.dataset import (
    list_samples,
    dedupe_samples,
    stratified_split_samples,
    save_split_csv,
)

def main():
    ap = argparse.ArgumentParser(description="Create frozen stratified train/val/test splits.")
    ap.add_argument("--data", required=True, help="Dataset root with class subfolders")
    ap.add_argument("--config", default=None, help="Optional JSON config (for labels)")
    ap.add_argument("--outdir", default="splits", help="Output directory for split CSV files")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    samples = dedupe_samples(list_samples(args.data, labels=cfg.labels))
    if not samples:
        raise SystemExit("No samples found. Expect data/{ambulance,firetruck,traffic}/*.wav")

    train_s, val_s, test_s = stratified_split_samples(
        samples, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    outdir = Path(args.outdir)
    save_split_csv(train_s, str(outdir / "train.csv"))
    save_split_csv(val_s, str(outdir / "val.csv"))
    save_split_csv(test_s, str(outdir / "test.csv"))

    print(f"Saved splits to {outdir.resolve()}")
    print(f"Counts: total={len(samples)} train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    # simple label distribution check
    def dist(xs):
        d = {}
        for s in xs:
            d[s.label] = d.get(s.label, 0) + 1
        return d

    print("Label dist train:", dist(train_s))
    print("Label dist val:  ", dist(val_s))
    print("Label dist test: ", dist(test_s))

if __name__ == "__main__":
    main()