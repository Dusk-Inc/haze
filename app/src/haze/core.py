from ..network.core import Network
from ..core_io.core import CoreIO
from ..registry.core import Registry
from ..neuron.core import Motor, Sensor, Inter
from ..tokens.decoder import DecoderType
from ..tokens.mesh import MeshType
from dataclasses import fields
from ..connector.core import Connector
from .models import NeuronPackageModel
from ..tokens.encoder import EncoderType
from ..decoder.core import Decoder
from ..encoder.core import Encoder
from ..errors.haze import InvalidRewardError
from ..config.core import Config
from ..auditor.core import Auditor
from ..injector.core import Injector
from ..tokens.injector import GlobalTypes
from ..auditor.models import AuditResultsModel
from .models import IdeaModel, InputModel
from ..utils.calculations import clamp
from ..neuron_io.core import NeuronIO
from ..tokens.neuron_io import TransformerTypes
import random
from typing import Any
import os

class Haze:
    """Orchestrate the network, the encoders, and the decoders."""
    def __init__(
            self, 
            model_path: str = None,
            persist: bool = True,
            config: Config = Config(),
            sequential: bool = False,
            end_token: str = None,
            seed: int = 42
        ):
        """Register the shared services, then set the end token, sequential mode, and random seed."""
        Injector.register(name=GlobalTypes.NEURON_IO, instance=NeuronIO())
        Injector.register(name=GlobalTypes.CONFIG, instance=config)
        Injector.register(name=GlobalTypes.CORE, instance=CoreIO(path=model_path, persist=persist))
        Injector.register(name=GlobalTypes.REGISTRY, instance=Registry())
        Injector.register(name=GlobalTypes.NETWORK, instance=Network())
        Injector.register(name=GlobalTypes.AUDITOR, instance=Auditor())
        
        
        self.network: Network = Injector.resolve(GlobalTypes.NETWORK)
        self._end_token: str = end_token or "<!END!>"
        self._sequential = sequential
        self._inputs: dict[str, list[Any]] = {}
        self._current_observation: InputModel = None
        self._lexical_chain: list[IdeaModel] = None
        random.seed(seed)

    def set_lexical_chain(self, lexical_chain: list[IdeaModel]):
        """Set the lexical chain, adding an end-token motor to the last layer's decoders when sequential."""
        if self._sequential:
            for d in lexical_chain[-1].decoders:
                d.decoder.add_motor(self._end_token)

        self._lexical_chain = lexical_chain

    def observe(self, input_data: list[Any], encoder: Encoder):
        """Feed `input_data` through `encoder`, forwarding each layer's predictions into the next layer's encoders."""
        self._current_observation = InputModel(input_data=input_data, encoder=encoder)
        encoder_id = encoder.get_id(as_string=True)
        self._inputs[encoder_id] = input_data

        for layer in self._lexical_chain:
            for decoder in layer.decoders:
                decoder.decoder.set_outputs(decoder.outputs)

        encoder.propogate(input_data)

        for i, layer in enumerate(self._lexical_chain):

            decoder_outputs = []

            for decoder_model in layer.decoders:
                decoder = decoder_model.decoder
                decoder_outputs.extend(decoder.predict() if i < len(self._lexical_chain) - 1 else decoder_model.outputs)

            if i == len(self._lexical_chain) - 1:
                break

            next_layer = self._lexical_chain[i + 1]
            for encoder, decoder in zip(next_layer.encoders, layer.decoders):
                if set(encoder.get_data_types()) != set(decoder.decoder.get_data_types()):
                    raise Exception("The decoders of the current layer and encoders in the next layer are incompatible")
                
                encoder.propogate(decoder.decoder.predict())

    def predict(self, limit: int = None, iterations=0):
        """Return the decoders' predictions, re-observing and reverse-learning when no signal reaches the motors."""
        try:
            result = self.call_decoders(limit=limit)
            return result
        except ValueError:
            if limit is not None and iterations > limit:
                raise Exception("Network did not connect signal to motors within the iteration limit.")
            
            self.observe(
                encoder=self._current_observation.encoder, 
                input_data=self._current_observation.input_data
            )
            self.learn(reverse=True)
            iterations += 1
            return self.predict(limit=limit, iterations=iterations)

    def call_decoders(self, limit: int = None):
        """Drive the last layer's decoders until each emits the end token or the iteration limit is reached."""
        if self._lexical_chain is None:
            raise Exception("No lexical chain has been set.")

        iterations: int = 0
        results: dict[str, list] = {}
        stopped_decoders = set()
        last_decoders = [decoder_model.decoder for decoder_model in self._lexical_chain[-1].decoders]
        
        for decoder in last_decoders:
            decoder_id = decoder.get_id(as_string=True)
            if decoder_id not in results.keys():
                results[decoder_id] = []

        while True:
            for idx, decoder in enumerate(last_decoders):
                if idx not in stopped_decoders:
                    output = decoder.predict()
                    if output == self._end_token:
                        stopped_decoders.add(idx)
                    else:
                        results[decoder.get_id(as_string=True)].append(output)

            if len(stopped_decoders) == len(last_decoders):
                break

            iterations += 1
            if limit is not None and iterations >= limit:
                break

            if self._sequential:
                starting_encoders = self._lexical_chain[0].encoders
                for encoder, decoder in zip(starting_encoders, last_decoders):
                    if not any(t in decoder.get_data_types() for t in encoder.get_data_types()):
                        raise Exception("The encoders at the beginning of the lexical set and the decoders at the end are incompatible")
                    current_inputs = [input_data for input_data in self._inputs[encoder.get_id(as_string=True)]]
                    current_inputs.append(output)
                    self.observe(current_inputs, encoder)
            else:
                break
        
        self._inputs.clear()
        return results

    def load(
            self, 
            aperature_size: int = 3,
            nexus_size: int = 3,
            terminus_size: int = 3
        ):
        """Build a fresh network, or rebuild the mesh, neurons, and connections from the saved model."""
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
        neuron_io: NeuronIO = Injector.resolve(GlobalTypes.NEURON_IO)
        if core.is_empty():
            self.network.create_network(
                aperature_size=aperature_size, 
                nexus_size=nexus_size, 
                terminus_size=terminus_size
            )
        else:
            self.network.instantiate_mesh()
            neurons: list[NeuronPackageModel] = []
            terminus_neurons = core.get_neuron_data(core._terminus_path)
            nexus_neurons = core.get_neuron_data(core._nexus_path)
            terminus_package = self._package_inters(terminus_neurons, MeshType.TERMINUS)
            self.network.mesh.terminus._inters = [neuron.neuron for neuron in terminus_package]
            nexus_package = self._package_inters(nexus_neurons, MeshType.NEXUS)
            self.network.mesh.nexus._inters = [neuron.neuron for neuron in nexus_package]
            neurons.extend(terminus_package)
            neurons.extend(nexus_package)

            for encoder_type in EncoderType:
                encoder_path = os.path.join(core._encoder_path, encoder_type)
                if os.path.exists(encoder_path):
                    sensor_data = core.get_neuron_data(encoder_path)
                    sensor_package = self._package_sensors(sensor_data=sensor_data, encoder_type=encoder_type)
                    neurons.extend(sensor_package)
                    sensor_list = [package.neuron for package in sensor_package]
                    neuron_io.set_neurons(neuron_list=sensor_list, transformer_name=encoder_type, transformer_type=TransformerTypes.ENCODER)

            for decoder_type in DecoderType:
                decoder_path = os.path.join(core._decoder_path, decoder_type)
                if os.path.exists(decoder_path):
                    motor_data = core.get_neuron_data(decoder_path)
                    motor_package = self._package_motors(motor_data=motor_data, decoder_type=decoder_type)
                    neurons.extend(motor_package)
                    motor_list = [package.neuron for package in motor_package]
                    neuron_io.set_neurons(neuron_list=motor_list, transformer_name=decoder_type, transformer_type=TransformerTypes.DECODER)
            
            for n in neurons:
                for c in n.connector_set:
                    connection_data = core.get_connection_data(c)
                    connector = Connector(
                        dendrite = [packaged_neuron.neuron for packaged_neuron in neurons if packaged_neuron.neuron.get_id() == connection_data["dendrite_id"]][0],
                        id=connection_data["id"],
                        is_default=bool(connection_data["is_default"])
                    )
                    registry.add_connector(connector=connector, strength=connection_data["strength"], epsilon=connection_data["epsilon"])
                    n.neuron.post_connection(connector)

    def _package_sensors(self, sensor_data: list[dict], encoder_type: EncoderType) -> list[NeuronPackageModel]:
        """Rebuild sensor neurons from saved data, paired with the connection ids to restore."""
        neurons = []
        for i in sensor_data:
                neuron = Sensor(
                    id=i["id"],
                    encoder=encoder_type
                )
                neurons.append(
                    NeuronPackageModel(
                        neuron=neuron,
                        connector_set=i["connections"]
                    )
                )
        return neurons

    def _package_motors(self, motor_data: list[dict], decoder_type: DecoderType) -> list[NeuronPackageModel]:
        """Rebuild motor neurons from saved data, paired with the connection ids to restore."""
        neurons = []
        for i in motor_data:
                neuron = Motor(
                    decoder=decoder_type,
                    id=i["id"],
                    answer=i["answer"]
                )
                neurons.append(
                    NeuronPackageModel(
                        neuron=neuron,
                        connector_set=i["connections"]
                    )
                )
        return neurons

    def _package_inters(self, neuron_data: list[dict], mesh: MeshType) -> list[NeuronPackageModel]:
        """Rebuild interneurons from saved data, paired with the connection ids to restore."""
        neurons = []
        for i in neuron_data:
                neuron = Inter(
                    mesh=mesh,
                    id=i["id"]
                )
                neurons.append(
                    NeuronPackageModel(
                        neuron=neuron,
                        connector_set=i["connections"]
                    )
                )
        return neurons

    
    def learn(self, reward: float = 0.9, reverse: bool = False):
        """Apply `reward` to the registry, grow the network on the auditor's verdict, and drop connections that fell below threshold."""
        reward = clamp(reward)
        registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
        auditor: Auditor = Injector.resolve(GlobalTypes.AUDITOR)
        config: Config = Injector.resolve(GlobalTypes.CONFIG)
        core: CoreIO = Injector.resolve(GlobalTypes.CORE)
        auditor.activity.nexus_size = len(self.network.mesh.nexus.get_inters())
        if reverse:
            registry.learn(reward=reward, confidence=0.1, reverse=True)
        else:
            registry.learn(reward=reward, confidence=auditor.get_confidence_score(), reverse=False)

        aggregate_confidence = self.get_aggregate_confidence()
        growth: AuditResultsModel = auditor.stimulate_growth(reward=reward, confidence=aggregate_confidence)
        self.network.handle_growth(growth)

        all_neurons = self.network.get_all_neurons()
        faulty_connections: list[Connector] = []

        for c in registry._connectors:
            if c.get_strength() * 0.9 <= config.signal_threshold:
                faulty_connections.append(c)
                for n in all_neurons:
                    if c in n._connections:
                        n.delete_connection(c)
                
        for c in faulty_connections:
            core.remove_from_file(c.get_id(as_string=True), core._connection_path)

        for c in faulty_connections:
            registry._connectors.remove(c)

        for n in all_neurons:
            if len(n._connections) < 2 and len(n._connections) > 0:
                core.remove_from_file(n._connections[0].get_id(as_string=True), core._connection_path)
                n.clear_connections()
                if n.mesh == MeshType.NEXUS:
                    self.network.mesh.nexus.connect_neurons([n])
                    self.network.connect_mesh([n], self.network.mesh.terminus.get_inters())
                if n.mesh == MeshType.TERMINUS:
                    self.network.mesh.terminus.connect_neurons(n)

    def get_aggregate_confidence(self):
        """Return the summed last-confidence of the final layer's decoders."""
        last_decoders: list[Decoder] = [decoder_model.decoder for decoder_model in self._lexical_chain[-1].decoders]
        agregate_confidence = sum([decoder.get_last_confidence() for decoder in last_decoders])
        return agregate_confidence