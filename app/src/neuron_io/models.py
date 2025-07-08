from dataclasses import dataclass
from ..neuron.core import Sensor, Motor
from typing import Union
from .enums import TransformerTypes
from ..encoder.enums import EncoderType
from ..decoder.enums import DecoderType

@dataclass
class NeuronState:
    neurons: list[Union[Sensor, Motor]]
    transformer: TransformerTypes
    transformer_name: Union[EncoderType, DecoderType]
    is_dirty: bool = False