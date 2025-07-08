class IncorrectOutputCount(Exception):
    def __init__(self, message):
        super().__init__(message)

class IncorrectOutputType(Exception):
    def __init__(self, message):
        super().__init__(message)