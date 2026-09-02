from abc import ABC, abstractmethod
from ..entity.core import Entity
from ..tokens.neuron import NeuronType

class INeuron(ABC, Entity):
    @abstractmethod
    def transmit(self, signal) -> None:
        """Pass a signal on to this neuron's connections."""
        pass

    @abstractmethod
    def get_type(self) -> NeuronType:
        """Return the NeuronType this neuron is."""
        pass

    @abstractmethod
    def get_active(self) -> bool:
        """Return whether this neuron is currently active."""
        pass