"""Domain errors raised while generating, loading, or recording a neuron mesh."""


class InsufficientNeuronCount(Exception):
    """Signals a mesh with too few neurons; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class InsufficientDimensions(Exception):
    """Signals a mesh with too few dimensions; the caller supplies the detail."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)


class NoMeshLoaded(Exception):
    """Raised when the mesh is used before one has been created or loaded."""

    def __init__(self, message):
        """Pass the caller-supplied description through to Exception."""
        super().__init__(message)
