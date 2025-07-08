from ..entity.core import Entity
import math

class Signal(Entity):
    def __init__(
            self, 
            value: float, 
            path_length: int = 0,
            sum_log: float = 0
        ):
        Entity.__init__(self)
        self._value: float = value
        self._path_length = path_length
        self._sum_log = sum_log

    def propogate(self, connection_strength: float) -> None:
        self._sum_log = self._sum_log + math.log(connection_strength)
        self._path_length += 1
        self._value = self._value * connection_strength

    def get_geometric_mean(self) -> float:
        if self._path_length == 0:
            return 1.0
        return math.exp(self._sum_log / self._path_length)

    def get_actual(self) -> float:
        return self._value * self.get_geometric_mean()

    def get_value(self) -> float:
        return self._value
    
    def set_value(self, new_value) -> None:
        self._value = new_value

    def get_path_length(self) -> int:
        return self._path_length
    
    def get_sums(self):
        return self._sum_log