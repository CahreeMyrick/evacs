from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import wave
import numpy as np

from .errors import InvalidFormatError, InvalidDurationError

@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray  # shape (n,), float32, ~[-1,1]
    sr: int
    duration_sec: float

def _read_wav_pcm(path: Path) -> Tuple[np.ndarray, int]:
    """
    Minimal WAV reader using stdlib `wave`.
    Supports PCM integer WAV (8/16/24/32-bit) and mono/stereo.
    """
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()  # bytes per sample
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except wave.Error as e:
        raise InvalidFormatError(f"Not a valid WAV file: {path}") from e

    if sampwidth not in (1, 2, 3, 4):
        raise InvalidFormatError(f"Unsupported WAV sample width: {sampwidth} bytes")

    # Convert bytes to int32
    if sampwidth == 1:
        # unsigned 8-bit PCM
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        data = data.astype(np.int16)
        peak = 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16)
        peak = 32768.0
    elif sampwidth == 3:
        # 24-bit PCM: manual unpack into int32
        a = np.frombuffer(raw, dtype=np.uint8)
        a = a.reshape(-1, 3)
        data = (a[:, 0].astype(np.int32)
                | (a[:, 1].astype(np.int32) << 8)
                | (a[:, 2].astype(np.int32) << 16))
        # sign extension for 24-bit
        sign_bit = 1 << 23
        data = (data ^ sign_bit) - sign_bit
        peak = float(1 << 23)
    else:  # sampwidth == 4
        data = np.frombuffer(raw, dtype=np.int32)
        peak = float(1 << 31)

    # Deinterleave channels to float32
    if n_channels > 1:
        data = data.reshape(-1, n_channels).astype(np.float32) / peak
        # keep as (n, channels) for now
        return data, sr
    else:
        data = data.astype(np.float32) / peak
        return data, sr

def load_wav(path: str, max_duration_sec: float, tolerance_sec: float = 0.0) -> AudioBuffer:
    p = Path(path)
    if p.suffix.lower() != ".wav":
        raise InvalidFormatError(f"Expected .wav file, got: {p.suffix}")

    samples, sr = _read_wav_pcm(p)

    # If stereo, keep both channels; preprocess will handle mono conversion
    n = samples.shape[0]
    duration = n / float(sr)

    if duration > max_duration_sec + tolerance_sec:
        raise InvalidDurationError(
            f"Audio duration {duration:.3f}s exceeds {max_duration_sec:.3f}s (tol {tolerance_sec:.3f}s)"
        )
    
    return AudioBuffer(samples=np.asarray(samples, dtype=np.float32), sr=sr, duration_sec=duration)