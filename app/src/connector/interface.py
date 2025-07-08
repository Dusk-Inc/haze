from abc import ABC, abstractmethod
from uuid import UUID
from ..neuron.interface import INeuron
from ..transmission.core import Transmission


class IConnector(ABC):
    @abstractmethod
    def set_index(self, index: int) -> None:
        """Sets the location in the registry memory that the connector stores its strength and epsilon values."""
        pass
    
    @abstractmethod
    def get_decay(self) -> float:
        """Returns the decay value."""
        pass

    @abstractmethod
    def set_decay(self, new_value: float) -> None:
        """Sets a new decay value."""
        pass

    @abstractmethod
    def get_strength(self) -> float:
        """Returns the current strength of the connection."""
        pass

    @abstractmethod
    def set_strength(self, reward: float, confidence: float) -> None:
        """Updates the strength of the connection based on reward and confidence."""
        pass

    @abstractmethod
    def save_state(self) -> None:
        """Saves the current state of the connector to persistent storage."""
        pass

    @abstractmethod
    def get_epsilon(self) -> float:
        """Returns the epsilon value."""
        pass

    @abstractmethod
    def get_dendrite(self) -> INeuron:
        """Returns the dendrite neuron connected to this connector."""
        pass

    @abstractmethod
    def set_dentraite(self, dendrite: INeuron) -> None:
        """Sets the dendrite neuron for this connector."""
        pass

    @abstractmethod
    def transmit(self, ingress: Transmission) -> None:
        """Handles the transmission of a signal through the connector."""
        pass

    @abstractmethod
    def clear_history(self) -> None:
        """Clears the history of processed transmissions."""
        pass

    @abstractmethod
    def record(self) -> dict:
        """Returns a dictionary representation of the connector's state."""
        pass