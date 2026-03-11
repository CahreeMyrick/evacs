class EVACSError(Exception):
    """Base exception for EVACS."""


class InvalidFormatError(EVACSError):
    """Raised when the input file is not a supported audio format."""


class InvalidDurationError(EVACSError):
    """Raised when the audio duration violates constraints."""


class ModelLoadError(EVACSError):
    """Raised when the model artifact cannot be loaded."""


class InferenceError(EVACSError):
    """Raised when inference fails."""