class InvalidRewardError(Exception):
    def __init__(self):
        super().__init__(self, "Reward must be a value between 0 and 1.")