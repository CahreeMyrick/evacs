from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import hashlib
import numpy as np

@dataclass(frozen=True)
class Sample:
    path: str
    label: str

def _sha1_file(path: Path, max_bytes: int = 2_000_000) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()

def list_samples(data_root: str, labels: List[str]) -> List[Sample]:
    root = Path(data_root)
    out: List[Sample] = []
    for lab in labels:
        folder = root / lab
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.wav")):
            out.append(Sample(path=str(p), label=lab))
    return out

def dedupe_samples(samples: List[Sample]) -> List[Sample]:
    seen = set()
    out: List[Sample] = []
    for s in samples:
        h = _sha1_file(Path(s.path))
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out

def stratified_split_samples(
    samples: List[Sample],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Stratified split by label.
    Produces approx train/val/test fractions overall.
    """
    assert 0 < train_frac < 1
    assert 0 <= val_frac < 1
    assert train_frac + val_frac < 1

    rng = np.random.default_rng(seed)

    by_label: Dict[str, List[Sample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    train, val, test = [], [], []

    for lab, group in by_label.items():
        idx = np.arange(len(group))
        rng.shuffle(idx)

        n = len(group)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        train.extend([group[i] for i in train_idx])
        val.extend([group[i] for i in val_idx])
        test.extend([group[i] for i in test_idx])

    # Shuffle each split so labels are mixed
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test

def save_split_csv(samples: List[Sample], out_csv: str) -> None:
    p = Path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["path,label"]
    for s in samples:
        # keep commas safe: wrap path in quotes if needed
        path = s.path.replace('"', '""')
        lines.append(f"\"{path}\",{s.label}")
    p.write_text("\n".join(lines) + "\n")

def load_split_csv(csv_path: str) -> List[Sample]:
    p = Path(csv_path)
    txt = p.read_text().strip().splitlines()
    if not txt:
        return []
    # expect header "path,label"
    rows = txt[1:] if txt[0].lower().startswith("path") else txt
    out: List[Sample] = []
    for r in rows:
        if not r.strip():
            continue
        # very small CSV parser: "path",label OR path,label
        if r.startswith('"'):
            # path may contain commas; find closing quote
            endq = r.find('",')
            if endq == -1:
                raise ValueError(f"Bad CSV row: {r}")
            path = r[1:endq].replace('""', '"')
            label = r[endq+2:].strip()
        else:
            parts = r.split(",")
            if len(parts) < 2:
                raise ValueError(f"Bad CSV row: {r}")
            path = parts[0].strip()
            label = parts[1].strip()
        out.append(Sample(path=path, label=label))
    return out