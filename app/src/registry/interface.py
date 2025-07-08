from abc import ABC, abstractmethod
from numpy.typing import NDArray
from ..connector.interface import IConnector


class IRegistry(ABC):
    @abstractmethod
    def get_strength(self, index: int) -> float:
        """Returns the strength value at the specified index."""
        pass

    @abstractmethod
    def set_strength(self, index: int, strength: float) -> float:
        """Sets the strength value at the specified index."""
        pass

    @abstractmethod
    def get_epsilon(self, index: int) -> float:
        """Returns the epsilon value at the specified index."""
        pass

    @abstractmethod
    def set_epsilon(self, index: int, epsilon: float) -> float:
        """Sets the epsilon value at the specified index."""
        pass

    @abstractmethod
    def get_status(self, index: int) -> int:
        """Returns the status (active/inactive) at the specified index."""
        pass

    @abstractmethod
    def get_decay(self) -> float:
        """Returns the decay value for this connection's network."""
        pass

    @abstractmethod
    def activate_connector(self, index: int) -> None:
        """Activates the connector at the specified index."""
        pass

    @abstractmethod
    def add_connector(self, connector: IConnector, strength: float, epsilon: float) -> None:
        """Adds a new connector to the registry."""
        pass

    @abstractmethod
    def learn(self, confidence: float, reward: float) -> None:
        """Performs the learning process, updating strengths and epsilons."""
        pass