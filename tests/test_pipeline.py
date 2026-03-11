import numpy as np
import wave
from pathlib import Path
from evacs.config import Config
from evacs.pipeline import classify_file

def _write_wav(path: Path, sr: int, x: np.ndarray):
    x16 = np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x16.tobytes())

def test_pipeline_runs(tmp_path):
    cfg = Config()
    sr = cfg.target_sr
    x = (0.01 * np.random.randn(int(sr * cfg.clip_sec))).astype(np.float32)
    p = tmp_path / "a.wav"
    _write_wav(p, sr, x)

    res = classify_file(str(p), cfg)
    assert res.prediction.label in cfg.labels
    assert res.stage_times.total_ms >= 0.0