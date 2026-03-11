import numpy as np
from evacs.io import AudioBuffer
from evacs.preprocess import pad_or_trim, to_mono

def test_pad_or_trim_exact_length():
    sr = 10
    clip = 3.0
    x = np.ones(5, dtype=np.float32)
    buf = AudioBuffer(samples=x, sr=sr, duration_sec=len(x)/sr)
    out = pad_or_trim(buf, clip_sec=clip)
    assert out.samples.shape[0] == int(round(sr * clip))

def test_to_mono_stereo():
    sr = 10
    x = np.ones((100, 2), dtype=np.float32)
    buf = AudioBuffer(samples=x, sr=sr, duration_sec=10.0)
    out = to_mono(buf)
    assert out.samples.ndim == 1
    assert out.samples.shape[0] == 100