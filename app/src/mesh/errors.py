class InsufficientNeuronCount(Exception):
    def __init__(self, message):
        super().__init__(message)

class InsufficientDimensions(Exception):
    def __init__(self, message):
        super().__init__(message)

class NoMeshLoaded(Exception):
    def __init__(self, message):
        super().__init__(message) 