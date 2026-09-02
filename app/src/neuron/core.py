from uuid import UUID
from ..signal.core import Signal
from .interface import INeuron
from ..threader.core import Threader
from typing import Union
from ..entity.core import Entity
from ..terminal.core import Terminal
from ..tokens.neuron import NeuronType
from ..context.core import Context
from ..transmission.core import Transmission
import numpy as np
from numpy.typing import NDArray
import random
from ..tokens.mesh import MeshType
from ..tokens.decoder import DecoderType
from ..tokens.encoder import EncoderType
from ..core_io.core import CoreIO
from ..injector.core import Injector
from ..tokens.injector import GlobalTypes
from ..config.core import Config
import os

class Neuron(INeuron, Terminal, Entity):
    def __init__(self, 
            type: NeuronType,
            id: UUID = None
        ):
        """Initialise the neuron's terminal, identity, kind, and active flag."""
        Terminal.__init__(self)
        Entity.__init__(self, id)
        self._type = type
        self._active = True
        self._k = 0
        self._config: Config = Injector.resolve(GlobalTypes.CONFIG) 

    def get_type(self):
        """Return the NeuronType this neuron is."""
        return self._type
    
    def get_k(self):
        """Return the fan-out sample size for this neuron."""
        return self._k
    
    def set_k(self, N: int):
        """Set the fan-out to a random value bounded by the log of `N`, with `N` floored at 1000."""
        if N < 1000:
            N = 1000
        self._k = random.randint(2, int(np.log(N)))

    
    def transmit(self) -> None:
        """Pass a signal on; implemented by each subclass."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def record(self) -> dict:
        """Return this neuron's serializable state; implemented by each subclass."""
        raise NotImplementedError('This method should be implemented by subclasses.')
    
    def get_active(self):
        """Return whether this neuron is currently active."""
        return self._active
    
    def set_active(self, value: bool):
        """Set whether this neuron is currently active."""
        self._active = value

    def save_state(self, file_path: str):
        """Write this neuron's recorded state to `file_path`."""
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        state = self.record()
        core.save_to_file(state, file_path)




class Sensor(Neuron):
    def __init__(
            self,
            encoder: EncoderType = None, 
            id: Union[UUID, str] = None
        ):
        """Create a sensor neuron bound to `encoder`."""
        Neuron.__init__(
            self,
            id=id,
            type=NeuronType.SENSOR
        )
        self.encoder = encoder

    def transmit(
            self, 
            context: Context,
            input_value: float
        ):
        
        """Emit `input_value` as a signal along every connection, running each in `context`."""
        for c in self.get_connections():
            signal = Signal(
                value=input_value
            )
            egress = Transmission(
                signal=signal,
                context=context
            )
            context.enqueue(c.transmit, egress)
            context.run()

    def record(self):
        """Return the sensor's type, connection ids, and id."""
        return {
            "type": self._type,
            "connections": [connection.get_id() for connection in self.get_connections()],
            "id": self.get_id(as_string=True),
        }
    
    def save_state(self):
        """Write the sensor's state under its encoder's directory."""
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        super().save_state(os.path.join(core._encoder_path, self.encoder, self.get_id(as_string=True)))

class Motor(Neuron, Threader):
    def __init__(self, 
            answer,
            decoder: DecoderType = None,
            id: Union[UUID, str] = None
        ):
        """Create a motor neuron holding `answer`, with a synchronized transmit."""
        Threader.__init__(self)
        Neuron.__init__(
            self,
            id=id,
            type=NeuronType.MOTOR
        )
        self.answer = answer
        self.transmit = self.synchronized(self.transmit)
        self.decoder = decoder
        self._signals: NDArray = np.array([])

    def transmit(self, ingress: Transmission):
        """Drain the queue, accumulating each transmission's actual signal value."""
        self.enqueue(ingress)
        while not self._queue.empty():
            with self._lock:
                egress: Transmission = self.dequeue()
            signal: Signal = egress.get_signal()
            self._signals = np.append(self._signals, signal.get_actual())

    def get_state(self):
        """Return the summed signal reaching this motor, raising when it is NaN."""
        result = np.sum(self._signals)
        if np.isnan(result):
            raise ValueError("State of motor is NaN.")
        
        return result
    
    def save_state(self):
        """Write the motor's state under its decoder's directory."""
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        super().save_state(os.path.join(core._decoder_path, self.decoder, self.get_id(as_string=True)))
    
    def reset_state(self):
        """Set the motor's state marker back to zero."""
        self._state = 0

    def record(self):
        """Return the motor's id, answer, and connection ids."""
        return {
            "id": self.get_id(as_string=True),
            "answer": self.answer,
            "connections": [connection.get_id() for connection in self.get_connections()]
        }

class Inter(Neuron, Threader):
    def __init__(
            self, 
            mesh: MeshType = None,
            id: Union[UUID, str] = None
        ):
        """Create an interneuron belonging to `mesh`."""
        Threader.__init__(self)
        Neuron.__init__(
            self,
            id=id,
            type=NeuronType.INTER
        )
        self.mesh = mesh
        self._signal_buffer: list[Signal] = []

    def transmit(self, ingress: Transmission):
        """Buffer incoming signals and, once their total clears the firing threshold, emit one merged signal onward."""
        self.enqueue(ingress)

        while not self._queue.empty():
            with self._lock:
                egress: Transmission = self.dequeue()
            
            self._signal_buffer.append(egress.get_signal())

        total_value = sum(s.get_value() for s in self._signal_buffer)

        if total_value >= self._config.neuron_firing_threshold:
            merged_signal = Signal(
                value=total_value,
                path_length=max(s.get_path_length() for s in self._signal_buffer),
                sum_log=sum(s.get_sums() for s in self._signal_buffer)
            )

            for c in self.get_connections():
                emission = Transmission(
                    context=egress.get_context(),
                    signal=merged_signal
                )
                egress.get_context().enqueue(c.transmit, emission)

        self._signal_buffer.clear()

    def record(self):
        """Return the interneuron's type, id, and connection ids."""
        return {
            "type": self._type,
            "id": self.get_id(as_string=True),
            "connections": [connection.get_id() for connection in self.get_connections()]
        }
    
    def save_state(self):
        """Write the interneuron's state under its mesh's directory."""
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        super().save_state(os.path.join(core._mesh_path, self.mesh, self.get_id(as_string=True)))