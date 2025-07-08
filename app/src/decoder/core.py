from ..neuron.core import Motor
from typing import Any
from .errors import IncorrectOutputCount, IncorrectOutputType
from .enums import DecoderType
from ..core_io.core import CoreIO
from ..injector.core import Injector
from ..injector.enums import GlobalTypes
from ..auditor.core import Auditor
from ..network.core import Network
from ..connector.core import Connector
from ..entity.core import Entity
from ..neuron_io.core import NeuronIO
from ..neuron_io.enums import TransformerTypes
import numpy as np
import math
from typing import Union
from ..terminal.errors import IdenticalConnectionError
from typing import Optional
from ..utils.calculations import normalize

# decoder doesn't need to know about stuff
class Decoder(Entity):
    def __init__(
            self, 
            data_types: list[type],
            type: DecoderType
        ):
        Entity.__init__(self)
        self._outputs: Optional[list[Any]] = None
        self._data_types = data_types
        self._type = type
        self.core: Optional[CoreIO] = Injector.resolve(GlobalTypes.CORE)
        self._auditor: Optional[Auditor] = Injector.resolve(GlobalTypes.AUDITOR)
        self._network: Optional[Network] = Injector.resolve(GlobalTypes.NETWORK)
        self._io: Optional[NeuronIO] = Injector.resolve(GlobalTypes.NEURON_IO)
        self._confidence: float = 0.1

    def get_last_confidence(self):
        return self._confidence

    def get_type(self):
        return self._type
    
    def get_data_types(self):
        return self._data_types

    def get_motors(self) -> list[Motor]:
        return self._io.get_neurons(self._type)

    def get_active_motors(self) -> list[Motor]:
        motors = self.get_motors()
        return [motor for motor in motors if motor.get_active()]

    def get_outputs(self) -> None:
        return self._outputs
    
    def add_motor(self, answer: Any) -> None:
        motors = self.get_motors()
        existing_answers = len([motor.answer for motor in motors if motor.answer == answer])
        if existing_answers > 0:
            print("motor already exists in decoder")
            return 
        
        new_motor = Motor(answer=answer, decoder=self._type)
        motors.append(new_motor)
        self._io.set_neurons(neuron_list=motors, transformer_name=self._type, transformer_type=TransformerTypes.DECODER)
        self._network.connect_motor(new_motor)


    def _check_output_type(self, outputs: list[Any]) -> bool:
        if len([output for output in outputs if type(output) not in self._data_types]) > 0:
            return False
        
        return True
    
    def set_outputs(self, outputs: list[Any]) -> None:
        if not self._check_output_type(outputs):
            raise IncorrectOutputType("Mistmatch between output types and decoder types.")
        
        self._reset_motors()
        motors = self.get_motors()
        existing_answers = [motor.answer for motor in motors]
        for output in outputs:
            if output not in existing_answers:
                self.add_motor(answer=output)
                self._auditor.activity.new_motors += 1

        for motor in motors:
            motor.set_active(motor.answer in outputs)
        
        self._outputs = outputs

        self.check_terminus()

    def _check_decoder(self) -> None:
        if len(self._outputs) == 0:
            raise IncorrectOutputCount("Outputs must be set before prediction can be made.")
        
        active_motors = self.get_active_motors()
        if all(motor.get_state() == 0 for motor in active_motors):
            raise ValueError("No signal reached the motors. All active motors have a state of 0. This can be because signal did not reach motors, or no data was fed to the network.")
    
    def predict(self, *args, **kwargs):
        self._check_decoder()
        result = self._predict_impl(*args, **kwargs)
        self._confidence = self.confidence()
        self._reset_motors()
        return result
    
    def _predict_impl(self, *args, **kwargs):
        raise NotImplementedError("Method must be implemented by a subclass.")
    
    def _reset_motors(self):
        motors = self.get_motors()
        for m in motors:
            m.reset_state()
    
    def confidence(self, epsilon=1e-9):
        motor_states = [motor.get_state() for motor in self.get_motors()]
        signals = np.array(motor_states, dtype=np.float64)
        total = np.sum(signals)

        if total == 0:
            return 0.0

        probs = signals / total

        entropy = -np.sum(probs * np.log(probs + epsilon))
        max_entropy = np.log(len(signals))
        confidence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 1.0)

        return confidence
    
    def check_terminus(self):
        terminus = self._network.mesh.terminus
        inter_ids = [i.get_id() for i in terminus.get_inters()]
        motors: list[Motor] = self.get_motors()
        for m in motors:
            dendrite_ids = [connection.get_dendrite().get_id() for connection in m.get_connections()]
            missing_ids = list(set(inter_ids) - set(dendrite_ids))
            for missing_id in missing_ids:
                
                inter = next(i for i in terminus.get_inters() if i.get_id() == missing_id)
                connector = Connector(dendrite=m)
                try:
                    inter.post_connection(connector)
                except IdenticalConnectionError:
                    continue


class Regressor(Decoder):
    def __init__(self):
        Decoder.__init__(self, type=DecoderType.REGRESSOR, data_types=[int, float])

    def _predict_impl(self):
        motors = self.get_active_motors()
        numerator = sum(motor.get_state() * motor.answer for motor in motors)
        denominator = sum(motor.get_state() for motor in motors)
        
        if denominator == 0:
            raise ValueError("Total activation (sum of states) is zero. Cannot compute center of mass.")
        
        return numerator / denominator
    


class ArgMax(Decoder):
    def __init__(self):
        Decoder.__init__(self, type=DecoderType.ARGMAX, data_types=[str, int])

    def _find_max_motor(self) -> Motor:
        active_motors = self.get_active_motors()
        max_motor = max(active_motors, key=lambda m: m.get_state())
        return max_motor

    def _predict_impl(self):
        max_motor = self._find_max_motor()
        return max_motor.answer


class SoftMax(Decoder):
    def __init__(self):
        Decoder.__init__(self, type=DecoderType.SOFTMAX, data_types=[str, int])

    def softmax(self, values):
        exps = [math.exp(v) for v in values]
        total = sum(exps)
        return [v / total for v in exps]
        
    def _predict_impl(self):
        activations = [motor.get_state() for motor in self.get_active_motors()]
        return self.softmax(activations)
    


class Binary(Decoder):
    def __init__(self):
        Decoder.__init__(self, type=DecoderType.BINARY, data_types=[bool])
        
    def set_outputs(self):
        super().set_outputs([True, False])
    
    def _predict_impl(self):
        answers = [motor.get_state() for motor in self.get_active_motors()]
        index = answers.index(max(answers))
        return self._outputs[index]



class Vector(Decoder):
    def __init__(self):
        Decoder.__init__(self, type=DecoderType.VECTOR, data_types=[list[float]])

    def _check_output_type(self, outputs: list[list[float]]):
        for vector in outputs:
            if not all(isinstance(coordinate, float) for coordinate in vector):
                return False
            
        return True

    def _predict_impl(self):
        self._check_decoder()
        active_motors = self.get_active_motors()
        result_vector = [0.0] * len(active_motors[0].answer)

        for motor in active_motors:
            state = motor.get_state()
            result_vector = [rv + state * ans for rv, ans in zip(result_vector, motor.answer)]

        total_activation = sum(motor.get_state() for motor in active_motors)
        if total_activation > 0:
            result_vector = [value / total_activation for value in result_vector]

        return result_vector



class TopK(Decoder):
    def __init__(self):
        Decoder.__init__(self, type=DecoderType.TOP_K, data_types=[str, int])
        self._k = None

    def _sort_motors(self) -> list[Motor]:
        return sorted(self.get_active_motors(), key=lambda motor: motor.get_state(), reverse=True)
        
    def _predict_impl(self, k: int):
        sorted_motors = self._sort_motors()
        self._k = k
        return [motor.answer for motor in sorted_motors][:k]



class Bitmask(Decoder):
    def __init__(self):
        Decoder.__init__(self, type=DecoderType.BITMASK, data_types=[str, int])
        self.threshold = 0
        
    def _predict_impl(self, threshold):
        self.threshold = threshold
        return [motor.answer for motor in self.get_active_motors() if motor.get_state() > threshold]