from ..neuron.core import Sensor
from typing import Any, Type, Union
from ..tokens.encoder import EncoderType
from ..context.core import Context
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor, Future
from ..injector.core import Injector
from ..tokens.injector import GlobalTypes
from ..auditor.core import Auditor
from ..network.core import Network
from ..entity.core import Entity
from ..neuron_io.core import NeuronIO
from ..tokens.neuron_io import TransformerTypes

class Encoder(Entity):
    def __init__(
            self, 
            type: EncoderType, 
            data_types: list[Type]
        ):
        """Bind the encoder's kind and accepted data types, and resolve its shared services."""
        Entity.__init__(self)
        self._data_types: list[Type] = data_types
        self._type = type
        self._auditor: Auditor = Injector.resolve(GlobalTypes.AUDITOR)
        self._network: Network = Injector.resolve(GlobalTypes.NETWORK)
        self._io: NeuronIO = Injector.resolve(GlobalTypes.NEURON_IO)

    def get_type(self):
        """Return the EncoderType this encoder implements."""
        return self._type
    
    def get_sensors(self):
        """Return the sensor neurons registered to this encoder."""
        return self._io.get_neurons(self._type)
    
    def get_data_types(self):
        """Return the input types this encoder accepts."""
        return self._data_types
    
    def set_connector(self, callback: callable):
        """Set the callback used to connect new sensors."""
        self._connector = callback

    def add_sensor(self, sensor: Sensor) -> None:
        """Register `sensor` with this encoder and wire it into the network."""
        sensors = self.get_sensors()
        sensors.append(sensor)
        self._io.set_neurons(neuron_list=sensors, transformer_name=self._type, transformer_type=TransformerTypes.ENCODER)
        self._network.connect_sensor(sensor)
    
    def check_sensors(self, inputs: list[Union[int, float]]):
        """Create and register enough new sensors to cover `inputs`."""
        sensors = self.get_sensors()
        difference = len(inputs) - len(sensors or [])
        if difference > 0:
            for _ in range(difference):
                sensor = Sensor(encoder=self._type)
                self.add_sensor(sensor)
                self._auditor.activity.new_sensors += 1
    
    def propogate(self, input_data: list[Any]):
        """Ensure sensor coverage, then transmit each input through its sensor on a worker thread."""
        sensors = self.get_sensors()
        self.check_sensors(input_data)
        self._auditor.activity.features += len(input_data) - self._auditor.activity.features

        with ThreadPoolExecutor() as executor:
            futures: list[Future] = []

            for i, s in zip(input_data, sensors):
                context = Context()
                futures.append(executor.submit(s.transmit, context, i))

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Task raised an exception: {e}")

class NumericEncoder(Encoder):
    def __init__(self):
        """Configure a numeric encoder over integers and floats."""
        Encoder.__init__(
            self,
            type=EncoderType.NUMERIC,
            data_types=[int, float]
        )

    def normalize(self, data: list[float]):
        """Scale `data` into the range 0.1 to 0.9, creating sensors first when the counts disagree."""
        sensors = self.get_sensors()
        if sensors is None or len(data) != len(sensors):
            self.check_sensors(data)

        np_array = np.array(data)
        normalized = (np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array) + 1e-8)
        scaled: NDArray = 0.1 + normalized * 0.8

        return scaled
    
    def propogate(self, input_data: list[Any]):
        """Normalize `input_data` before propagating it."""
        input_data = self.normalize(input_data)
        return super().propogate(input_data)

class TextEncoder(Encoder):
    def __init__(self):
        """Configure a text encoder over strings."""
        Encoder.__init__(
            self,
            type=EncoderType.TEXT,
            data_types=[str]
        )

class VectorEncoder(Encoder):
    def __init__(self):
        """Configure a vector encoder over lists of floats or integers."""
        Encoder.__init__(
            self,
            type=EncoderType.VECTOR,
            data_types=[list[Union[float, int]]]
        )