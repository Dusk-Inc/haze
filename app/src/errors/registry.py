"""Domain errors raised by the registry that tracks instantiated connections."""


class UninstantiatedConnectionsError(Exception):
    """Raised when the registry is read before any connection has been registered."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)
