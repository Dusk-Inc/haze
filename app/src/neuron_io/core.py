from ..neuron.core import Sensor, Motor
from typing import Callable
from ..core_io.core import CoreIO
from ..tokens.encoder import EncoderType
from ..tokens.decoder import DecoderType
from .models import NeuronState
from ..tokens.neuron_io import TransformerTypes
from typing import Union

class NeuronIO:
    def __init__(self):
        """Start with an empty neuron registry."""
        self._neuron_registry: list[NeuronState] = []

    def set_neurons(self, neuron_list, transformer_name: Union[EncoderType, DecoderType], transformer_type: TransformerTypes):
        """Register `neuron_list` under `transformer_name`, replacing and marking dirty when already present."""
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
        """Return the neurons registered under `transformer_name`, raising when more than one entry matches."""
        neuron_list = [state.neurons for state in self._neuron_registry if state.transformer_name == transformer_name]
        if len(neuron_list) == 0:
            return []
        if len(neuron_list) > 1:
            raise ValueError("More than one instance of the encoder or decoder was found in the neuron registry.")
        return neuron_list[0]
    
    def get_neuron_total(self, transformer_type: TransformerTypes):
        """Return how many neurons are registered for `transformer_type`."""
        neurons = []
        for n in self._neuron_registry:
            if n.transformer == transformer_type:
                neurons.extend(n.neurons)
        return len(neurons)
    
    def get_all_neurons_by_transformer(self, transformer_type: TransformerTypes):
        """Return every neuron registered for `transformer_type`."""
        neurons = []
        for n in self._neuron_registry:
            if n.transformer == transformer_type:
                neurons.extend(n.neurons)
        return neurons

    def clear(self):
        """Drop every registered neuron."""
        self._neuron_registry.clear()