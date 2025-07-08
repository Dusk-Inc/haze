from uuid import UUID
from ..neuron.interface import INeuron
from ..neuron.enums import NeuronType
from ..threader.core import Threader
from ..entity.core import Entity
from .errors import DendriteAlreadyExistsError, NoIndexSetError
from ..context.core import Context
from ..transmission.core import Transmission
from ..core_io.core import CoreIO
from ..registry.interface import IRegistry
from ..injector.core import Injector
from ..injector.enums import GlobalTypes
from ..config.core import Config
import os

class Connector(Threader, Entity):
    def __init__(
            self, 
            dendrite: INeuron = None, 
            id: UUID = None, 
            is_default: bool =False
        ):
        Threader.__init__(self)
        Entity.__init__(self, id)
        self._dendrite: INeuron = dendrite
        self._history: list[UUID] = []
        self.is_default = is_default
        self.core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        self._registry: IRegistry = Injector.resolve(GlobalTypes.REGISTRY)
        self._config: Config = Injector.resolve(GlobalTypes.CONFIG)
        self._index = None

    def reset(self, dendrite: INeuron):
        self._dendrite = dendrite
        self._registry
    
    def get_decay(self):
        if self._index is None:
            raise NoIndexSetError()
        
        return self._registry.get_decay()

    def set_index(self, index: int):
        self._index = index

    def get_strength(self):
        if self._index is None:
            return 0.9
        
        return self._registry.get_strength(self._index)
    
    def set_strength(self, strength: float):
        if self._index is None:
            raise NoIndexSetError()
        
        self._registry.set_strength(self._index, strength)
        self.save_state()
    
    def get_epsilon(self):
        if self._index is None:
            raise NoIndexSetError()
        
        return self._registry.get_epsilon(self._index)
    
    def save_state(self):
        state = self.record()
        path = os.path.join(self.core._connection_path, self.get_id(as_string=True))
        self.core.save_to_file(state, path)
    
    def get_dendrite(self):
        return self._dendrite
    
    def set_dentraite(self, dendrite: INeuron):
        if self._dendrite is not None:
            raise DendriteAlreadyExistsError()
        self._dendrite = dendrite
        
    def transmit(self, ingress: Transmission) -> None:
        if self._dendrite.get_type() == NeuronType.MOTOR:
            if not self._dendrite.get_active():
                return 
            
        self.enqueue(ingress)

        while not self._queue.empty():
            with self._lock:
                egress: Transmission = self.dequeue()
            context: Context = egress.get_context()
            if context.get_id() not in self._history:
                self._history.append(context.get_id())
                signal = egress.get_signal()
                signal.propogate(self.get_strength())

                if signal.get_actual() > self._config.get_threshold():
                    if self._index is not None:
                        self._registry.activate_connector(self._index) 

                    context.enqueue(self._dendrite.transmit, egress)

    def clear_history(self):
        self._history = []

    def record(self):
        connector = {
            "dendrite_id": self._dendrite.get_id(as_string=True),
            "is_default": self.is_default,
            "epsilon": self.get_epsilon(),
            "decay": self.get_decay(),
            "strength": self.get_strength(),
            "id": str(self._id)
        }
        return connector