"""Domain errors raised by the terminal that holds a neuron's connection list."""


class NoMatchingConnectionError(Exception):
    """Raised when no connection with the requested id exists in the connection list."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class IdenticalConnectionError(Exception):
    """Raised when a connection would duplicate one already in the connection list."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)
