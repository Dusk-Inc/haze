"""Domain errors raised by Haze."""


class HazeError(Exception):
    """Base for every error Haze raises, so a caller can catch the whole surface."""


class PortNotFoundError(HazeError):
    """Raised when an observation or prediction names a key no port has registered."""


class PortAlreadyRegisteredError(HazeError):
    """Raised when a key is registered twice, which would silently rebind its neurons."""


class LabelSpaceError(HazeError):
    """Raised when a decoder's labels are unusable: empty, duplicated, or unhashable."""


class CapacityExceededError(HazeError):
    """Raised when an allocation cannot be satisfied even after capacity growth."""


class SignalRangeError(HazeError):
    """Raised when an encoder emits a value outside the signal band.

    Clipping instead would hide a broken encoder behind plausible-looking output.
    """


class SignalDidNotReachMotorsError(HazeError):
    """Raised when no active motor received signal, so there is no answer to decode.

    Distinct from answering wrongly: it means the mesh has not connected its sensors to its
    motors, which is what drives reverse learning.
    """


class LearningDisabledError(HazeError):
    """Raised when learning is attempted on a model in eval mode."""


class InvalidRewardError(HazeError):
    """Raised when a reward is not finite.

    Without this, the clamp propagates NaN into every strength in the active mask, destroying
    the network with no error and no symptom until predictions become nonsense.
    """


class IncompatibleChainError(HazeError):
    """Raised when one chain stage's output cannot be consumed by the next stage's encoders."""


class CheckpointCorruptError(HazeError):
    """Raised when a checkpoint's config, labels, and tensors disagree.

    Index drift produces a model that loads without complaint and answers wrongly, so every
    load boundary checks rather than trusting.
    """


class UntrustedImplementationError(HazeError):
    """Raised when a checkpoint names an implementation outside the first-party allow-list."""
