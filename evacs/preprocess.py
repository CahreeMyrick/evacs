from __future__ import annotations
import numpy as np
from .io import AudioBuffer

def to_mono(buf: AudioBuffer) -> AudioBuffer:
    x = buf.samples
    if x.ndim == 1:
        return buf
    # average channels
    mono = np.mean(x, axis=1).astype(np.float32)
    return AudioBuffer(samples=mono, sr=buf.sr, duration_sec=buf.duration_sec)

def normalize_peak(buf: AudioBuffer, peak: float = 0.99) -> AudioBuffer:
    x = buf.samples.astype(np.float32)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m <= 1e-12:
        return buf
    y = (x / m) * peak
    return AudioBuffer(samples=y.astype(np.float32), sr=buf.sr, duration_sec=buf.duration_sec)

def pad_or_trim(buf: AudioBuffer, clip_sec: float) -> AudioBuffer:
    x = buf.samples.astype(np.float32)
    target_n = int(round(clip_sec * buf.sr))
    if x.shape[0] == target_n:
        return AudioBuffer(samples=x, sr=buf.sr, duration_sec=clip_sec)
    if x.shape[0] > target_n:
        y = x[:target_n]
        return AudioBuffer(samples=y, sr=buf.sr, duration_sec=clip_sec)
    # pad
    pad = target_n - x.shape[0]
    y = np.pad(x, (0, pad), mode="constant")
    return AudioBuffer(samples=y.astype(np.float32), sr=buf.sr, duration_sec=clip_sec)

def resample_linear(buf: AudioBuffer, target_sr: int) -> AudioBuffer:
    """
    Simple linear resampler (dependency-free).
    Good enough for skeleton; swap with librosa/soxr later if desired.
    """
    if buf.sr == target_sr:
        return buf
    x = buf.samples.astype(np.float32)
    n_in = x.shape[0]
    duration = n_in / float(buf.sr)
    n_out = int(round(duration * target_sr))
    if n_out <= 1:
        return AudioBuffer(samples=np.zeros((n_out,), dtype=np.float32), sr=target_sr, duration_sec=duration)

    t_in = np.linspace(0.0, duration, num=n_in, endpoint=False)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return AudioBuffer(samples=y, sr=target_sr, duration_sec=duration)