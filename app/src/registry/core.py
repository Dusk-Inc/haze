import numpy as np
from numpy.typing import NDArray
from ..errors.registry import UninstantiatedConnectionsError
from ..connector.interface import IConnector
from ..threader.core import Threader
from ..config.core import Config
from ..injector.core import Injector
from ..tokens.injector import GlobalTypes
from typing import Optional
import random

class Registry(Threader):
    def __init__(self, threshold: float = 0.3):
        """Start with empty strength, epsilon, and status arrays and the given firing threshold."""
        Threader.__init__(self)
        self._strength: NDArray = np.array([])
        self._epsilon: NDArray = np.array([])
        self._status: NDArray = np.array([])
        self._decay: float = 0.9
        self._connectors: list[IConnector] = []
        self._threshold: float = threshold
        self._config: Config = Injector.resolve(name=GlobalTypes.CONFIG)

    def get_strength(self, index: int):
        """Return the strength held at `index`."""
        return self._strength[index]
    
    def get_threshold(self):
        """Return the firing threshold."""
        return self._threshold
    
    def get_epsilon(self, index: int):
        """Return the learning rate held at `index`."""
        return self._epsilon[index]
    
    def set_epsilon(self, index: int, epsilon: float = 0.5):
        """Set the learning rate at `index`."""
        self._epsilon[index] = epsilon
    
    def get_status(self, index: int):
        """Return the activation status held at `index`."""
        return self._status[index]
    
    def get_decay(self):
        """Return the decay applied to connector strength."""
        return self._decay
    
    def activate_connector(self, index) -> None:
        """Mark the queued index active, draining the queue under the lock."""
        self.enqueue(index)
        while not self._queue.empty():
            with self._lock:
                saved_index: int = self.dequeue()
            self._status[saved_index] = 1

    def add_connector(self, connector: IConnector, strength: Optional[float] = None, epsilon: Optional[float] = None):
        """Append `connector` with its strength, epsilon, and cleared status, and tell it its index."""
        if strength is None:
            strength = random.uniform(0.4, 0.9)

        self._connectors.append(connector)
        self._strength = np.append(self._strength, strength)
        self._epsilon = np.append(self._epsilon, epsilon or self._config.epsilon_start)
        self._status = np.append(self._status, 0)
        connector.set_index(len(self._strength)-1)
        
    def learn(self, confidence: float, reward: float, reverse=False):
        """Move each active connector's strength toward `reward` by its epsilon, decay those epsilons, then reset and persist."""
        if self._strength.size == 0 or self._epsilon.size == 0:
            raise UninstantiatedConnectionsError("No connections have been registered.")

        if reverse:
            active_mask = ~self._status.astype(bool)
        else:
            active_mask = self._status.astype(bool)
        deltas = self._epsilon[active_mask] * (reward - confidence)
        self._strength[active_mask] = np.clip(self._strength[active_mask] + deltas, 0.1, 0.9)
        self._epsilon[active_mask] *= self._config.epsilon_decay

        self._reset_connectors()

        print(self._strength)

        for c in self._connectors:
            c.save_state()

    def _reset_connectors(self) -> None:
        """Clear every connector's activation status."""
        self._status = np.zeros_like(self._status)