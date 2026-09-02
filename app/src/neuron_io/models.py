from dataclasses import dataclass
from ..neuron.core import Sensor, Motor
from typing import Union
from ..tokens.neuron_io import TransformerTypes
from ..tokens.encoder import EncoderType
from ..tokens.decoder import DecoderType

@dataclass
class NeuronState:
    neurons: list[Union[Sensor, Motor]]
    transformer: TransformerTypes
    transformer_name: Union[EncoderType, DecoderType]
    is_dirty: bool = False