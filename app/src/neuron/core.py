from uuid import UUID
from ..signal.core import Signal
from .interface import INeuron
from ..threader.core import Threader
from typing import Union
from ..entity.core import Entity
from ..terminal.core import Terminal
from .enums import NeuronType
from ..context.core import Context
from ..transmission.core import Transmission
import numpy as np
from numpy.typing import NDArray
import random
from ..mesh.enums import MeshType
from ..decoder.enums import DecoderType
from ..encoder.enums import EncoderType
from ..core_io.core import CoreIO
from ..injector.core import Injector
from ..injector.enums import GlobalTypes
from ..config.core import Config
import os

class Neuron(INeuron, Terminal, Entity):
    def __init__(self, 
            type: NeuronType,
            id: UUID = None
        ):
        Terminal.__init__(self)
        Entity.__init__(self, id)
        self._type = type
        self._active = True
        self._k = 0
        self._config: Config = Injector.resolve(GlobalTypes.CONFIG) 

    def get_type(self):
        return self._type
    
    def get_k(self):
        return self._k
    
    def set_k(self, N: int):
        if N < 1000:
            N = 1000
        self._k = random.randint(2, int(np.log(N)))

    
    def transmit(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def record(self) -> dict:
        raise NotImplementedError('This method should be implemented by subclasses.')
    
    def get_active(self):
        return self._active
    
    def set_active(self, value: bool):
        self._active = value

    def save_state(self, file_path: str):
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        state = self.record()
        core.save_to_file(state, file_path)




class Sensor(Neuron):
    def __init__(
            self,
            encoder: EncoderType = None, 
            id: Union[UUID, str] = None
        ):
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
        return {
            "type": self._type,
            "connections": [connection.get_id() for connection in self.get_connections()],
            "id": self.get_id(as_string=True),
        }
    
    def save_state(self):
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        super().save_state(os.path.join(core._encoder_path, self.encoder, self.get_id(as_string=True)))

class Motor(Neuron, Threader):
    def __init__(self, 
            answer,
            decoder: DecoderType = None,
            id: Union[UUID, str] = None
        ):
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
        self.enqueue(ingress)
        while not self._queue.empty():
            with self._lock:
                egress: Transmission = self.dequeue()
            signal: Signal = egress.get_signal()
            self._signals = np.append(self._signals, signal.get_actual())

    def get_state(self):
        result = np.sum(self._signals)
        if np.isnan(result):
            raise ValueError("State of motor is NaN.")
        
        return result
    
    def save_state(self):
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        super().save_state(os.path.join(core._decoder_path, self.decoder, self.get_id(as_string=True)))
    
    def reset_state(self):
        self._state = 0

    def record(self):
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
        Threader.__init__(self)
        Neuron.__init__(
            self,
            id=id,
            type=NeuronType.INTER
        )
        self.mesh = mesh
        self._signal_buffer: list[Signal] = []

    def transmit(self, ingress: Transmission):
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
        return {
            "type": self._type,
            "id": self.get_id(as_string=True),
            "connections": [connection.get_id() for connection in self.get_connections()]
        }
    
    def save_state(self):
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        super().save_state(os.path.join(core._mesh_path, self.mesh, self.get_id(as_string=True)))