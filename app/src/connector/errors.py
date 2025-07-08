class ReturningFeatureError(Exception):
    def __init__(self, message):
        super().__init__(message)

class AxonNotAttachedError(Exception):
    def __init__(self, message):
        super().__init__(message)

class DendriteAlreadyExistsError(Exception):
    def __init__(self):
        super().__init__("Connect dendrite has already been set.")

class NoIndexSetError(Exception):
    def __init__(self):
        super().__init__("No index has been set for this connection.")