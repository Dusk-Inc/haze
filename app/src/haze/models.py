from dataclasses import dataclass
from ..neuron.core import Neuron
from ..encoder.core import Encoder
from ..encoder.enums import EncoderType
from ..decoder.core import Decoder
from typing import Any

@dataclass
class NeuronPackageModel:
    neuron: Neuron
    connector_set: list[dict]

@dataclass
class InputModel:
    encoder: EncoderType
    input_data: list[Any]

@dataclass
class DecoderModel:
    decoder: Decoder
    outputs: list[Any]

@dataclass
class IdeaModel:
    encoders: list[Encoder]
    decoders: list[DecoderModel]