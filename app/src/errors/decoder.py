"""Domain errors raised while decoding neuron output into predictions."""


class IncorrectOutputCount(Exception):
    """Raised when a decoder's outputs are absent or the wrong number for a prediction."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class IncorrectOutputType(Exception):
    """Raised when a decoder's output types do not match its decoder types."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)
