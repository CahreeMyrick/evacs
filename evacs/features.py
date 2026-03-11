from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .io import AudioBuffer
from .config import Config

@dataclass(frozen=True)
class FeatureTensor:
    X: np.ndarray  # shape (n_mels, n_frames), float32

def _stft_mag(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("STFT expects mono signal")

    if x.shape[0] < n_fft:
        x = np.pad(x, (0, n_fft - x.shape[0]), mode="constant")

    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (x.shape[0] - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1

    frames = []
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]), mode="constant")
        frames.append(frame * window)

    frames = np.stack(frames, axis=1)  # (n_fft, n_frames)
    spec = np.fft.rfft(frames, axis=0)
    power = (np.abs(spec) ** 2).astype(np.float32)
    return power  # (n_freq, n_frames)

def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> np.ndarray:
    n_freq = n_fft // 2 + 1
    if fmax is None:
        fmax = sr / 2.0

    m_min = _hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    m_max = _hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    m_pts = np.linspace(m_min, m_max, num=n_mels + 2, dtype=np.float32)
    hz_pts = _mel_to_hz(m_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_freq - 1)

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bin_pts[m], bin_pts[m + 1], bin_pts[m + 2]
        if center == left:
            center = min(left + 1, n_freq - 1)
        if right == center:
            right = min(center + 1, n_freq - 1)

        for k in range(left, center):
            fb[m, k] = (k - left) / float(center - left + 1e-12)
        for k in range(center, right):
            fb[m, k] = (right - k) / float(right - center + 1e-12)

    return fb

def log_mel(buf: AudioBuffer, cfg: Config) -> FeatureTensor:
    x = np.asarray(buf.samples, dtype=np.float32)
    power = _stft_mag(x, n_fft=cfg.n_fft, hop=cfg.hop_length)  # (n_freq, n_frames)

    fb = _mel_filterbank(sr=buf.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    mel = fb @ power  # (n_mels, n_frames)

    mel = np.maximum(mel, 1e-10)
    logmel = np.log(mel).astype(np.float32)

    if cfg.feature_norm == "per_clip":
        mu = np.mean(logmel)
        sigma = np.std(logmel) + 1e-6
        logmel = (logmel - mu) / sigma

    return FeatureTensor(X=logmel.astype(np.float32))

def vectorize_logmel(feats: FeatureTensor) -> np.ndarray:
    """
    Convert (n_mels, n_frames) into fixed vector:
      [mean over time for each mel] + [std over time for each mel]
    Output shape: (2*n_mels,)
    """
    X = np.asarray(feats.X, dtype=np.float32)
    mu = np.mean(X, axis=1)          # (n_mels,)
    sd = np.std(X, axis=1) + 1e-6    # (n_mels,)
    v = np.concatenate([mu, sd], axis=0).astype(np.float32)
    return v