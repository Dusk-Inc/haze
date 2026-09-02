"""Domain errors raised while connecting neurons through axons and dendrites."""


class ReturningFeatureError(Exception):
    """Signals a returning feature on a connection; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class AxonNotAttachedError(Exception):
    """Signals a connection used before its axon was attached; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class DendriteAlreadyExistsError(Exception):
    """Raised when a connection's dendrite is set a second time."""

    def __init__(self):
        """Set the fixed message describing the already-set dendrite."""
        super().__init__("Connect dendrite has already been set.")


class NoIndexSetError(Exception):
    """Raised when a connection is read before an index has been assigned to it."""

    def __init__(self):
        """Set the fixed message describing the missing index."""
        super().__init__("No index has been set for this connection.")
