import numpy as np
from evacs.io import AudioBuffer
from evacs.config import Config
from evacs.features import log_mel

def test_log_mel_shape():
    cfg = Config(n_mels=32, n_fft=256, hop_length=64, target_sr=8000)
    sr = cfg.target_sr
    x = np.random.randn(int(sr * cfg.clip_sec)).astype(np.float32) * 0.01
    buf = AudioBuffer(samples=x, sr=sr, duration_sec=cfg.clip_sec)
    feats = log_mel(buf, cfg)
    assert feats.X.shape[0] == cfg.n_mels
    assert feats.X.shape[1] > 0