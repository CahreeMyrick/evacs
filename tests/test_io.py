import pytest
from evacs.io import load_wav
from evacs.errors import InvalidFormatError

def test_rejects_non_wav_extension(tmp_path):
    p = tmp_path / "x.mp3"
    p.write_bytes(b"not real")
    with pytest.raises(InvalidFormatError):
        load_wav(str(p), max_duration_sec=3.0)