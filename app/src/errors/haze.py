"""Domain errors raised by the Haze mesh's top-level learning surface."""


class InvalidRewardError(Exception):
    """Raised when a reward passed to learning falls outside the range 0 to 1."""

    def __init__(self):
        """Set the fixed message describing the accepted reward range."""
        super().__init__(self, "Reward must be a value between 0 and 1.")
