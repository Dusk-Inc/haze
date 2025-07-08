class IncorrectInputSize(Exception):
    def __init__(self, message):
        super().__init__(message)

class NetworkException(Exception):
    def __init__(self, message):
        super().__init__(message)

class IdenticalEncoderException(Exception):
    def __init__(self, message):
        super().__init__(message)

class EncoderException(Exception):
    def __init__(self, message):
        super().__init__(message)