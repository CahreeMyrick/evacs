from __future__ import annotations
import argparse
import json
import sys
from .config import Config
from .pipeline import classify_file
from .errors import EVACSError

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evacs", description="Emergency Vehicle Audio Classification System")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify", help="Classify a single WAV file")
    c.add_argument("path", type=str, help="Path to .wav file")
    c.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    c.add_argument("--json", action="store_true", help="Print result as JSON")

    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = Config.load(args.config) if args.config else Config()

        if args.cmd == "classify":
            res = classify_file(args.path, cfg)

            if args.json:
                out = {
                    "label": res.prediction.label,
                    "probs": res.prediction.probs,
                    "times_ms": res.stage_times.times_ms,
                    "total_ms": res.stage_times.total_ms,
                }
                print(json.dumps(out, indent=2))
            else:
                print(f"label: {res.prediction.label}")
                print("probs:")
                for k, v in sorted(res.prediction.probs.items()):
                    print(f"  {k}: {v:.4f}")
                print(f"latency: {res.stage_times.total_ms:.1f} ms")
            return 0

        return 2

    except EVACSError as e:
        print(f"[EVACS ERROR] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}", file=sys.stderr)
        return 3

if __name__ == "__main__":
    raise SystemExit(main())