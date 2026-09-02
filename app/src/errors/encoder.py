"""Domain errors raised while encoding input data onto sensor neurons."""


class NoConnectionError(Exception):
    """Signals a missing encoder connection; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)
