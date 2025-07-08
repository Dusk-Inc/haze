from abc import ABC, abstractmethod
from ..entity.core import Entity
from .enums import NeuronType

class INeuron(ABC, Entity):
    @abstractmethod
    def transmit(self, signal) -> None:
        pass

    @abstractmethod
    def get_type(self) -> NeuronType:
        pass

    @abstractmethod
    def get_active(self) -> bool:
        pass