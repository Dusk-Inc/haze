"""Domain errors raised by the network that wires encoders, mesh, and decoders."""


class IncorrectInputSize(Exception):
    """Signals input whose size the network does not accept; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class NetworkException(Exception):
    """Signals a general network fault; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class IdenticalEncoderException(Exception):
    """Signals an encoder already registered on the network; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class EncoderException(Exception):
    """Signals an encoder fault surfaced by the network; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)
