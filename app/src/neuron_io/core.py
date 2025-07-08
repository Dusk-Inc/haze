from ..neuron.core import Sensor, Motor
from typing import Callable
from ..core_io.core import CoreIO
from ..encoder.enums import EncoderType
from ..decoder.enums import DecoderType
from .models import NeuronState
from .enums import TransformerTypes
from typing import Union

class NeuronIO:
    def __init__(self):
        self._neuron_registry: list[NeuronState] = []

    def set_neurons(self, neuron_list, transformer_name: Union[EncoderType, DecoderType], transformer_type: TransformerTypes):
        if transformer_name not in [state.transformer_name for state in self._neuron_registry]:
            self._neuron_registry.append(
                NeuronState(
                    neurons=neuron_list,
                    transformer=transformer_type,
                    transformer_name=transformer_name
                )
            )
        else:
            for state in self._neuron_registry:
                if state.transformer_name == transformer_name:
                    state.neurons = neuron_list
                    state.is_dirty = True

    def get_neurons(self, transformer_name: Union[EncoderType, DecoderType]) -> list[Sensor]:
        neuron_list = [state.neurons for state in self._neuron_registry if state.transformer_name == transformer_name]
        if len(neuron_list) == 0:
            return []
        if len(neuron_list) > 1:
            raise ValueError("More than one instance of the encoder or decoder was found in the neuron registry.")
        return neuron_list[0]
    
    def get_neuron_total(self, transformer_type: TransformerTypes):
        neurons = []
        for n in self._neuron_registry:
            if n.transformer == transformer_type:
                neurons.extend(n.neurons)
        return len(neurons)
    
    def get_all_neurons_by_transformer(self, transformer_type: TransformerTypes):
        neurons = []
        for n in self._neuron_registry:
            if n.transformer == transformer_type:
                neurons.extend(n.neurons)
        return neurons

    def clear(self):
        self._neuron_registry.clear()