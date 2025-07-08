class NoMatchingConnectionError(Exception):
    def __init__(self, message):
        super().__init__(message)

class IdenticalConnectionError(Exception):
    def __init__(self, message):
        super().__init__(message)