from .errors import IncorrectInputSize, NetworkException, IdenticalEncoderException, EncoderException
from ..mesh.core import Mesh
from ..neuron.core import Inter
from datetime import datetime
from .models import MeshModel
import json
import random
from ..connector.core import Connector
from ..neuron.core import Motor, Sensor
from ..mesh.enums import MeshType
from ..registry.interface import IRegistry
from ..injector.core import Injector
from ..injector.enums import GlobalTypes
from ..auditor.models import AuditResultsModel
from ..neuron_io.core import NeuronIO
from ..neuron_io.enums import TransformerTypes
from ..terminal.errors import IdenticalConnectionError
from typing import Optional

class Network:
    def __init__(self) -> None:
        self.mesh: MeshModel
        self.resources: int = 0
        self.state = []
        self._io: NeuronIO = Injector.resolve(name=GlobalTypes.NEURON_IO)
        self._registry: IRegistry = Injector.resolve(name=GlobalTypes.REGISTRY)

    def get_all_neurons(self) -> list[Inter]:
        neuron_list = []
        neuron_list.extend(self.mesh.nexus.get_inters())
        neuron_list.extend(self.mesh.terminus.get_inters())
        return neuron_list

    def is_empty(self):
        if self.mesh is None:
            return True
        
        return False
    
    def connect_motor(self, motor: Motor):
        for m in self.mesh.terminus.get_inters():
            try:
                connector = Connector(dendrite=motor)
                m.post_connection(connector)
                self._registry.add_connector(connector)
            except IdenticalConnectionError:
                continue
            m.save_state()

        motor.save_state()

    def connect_sensor(self, sensor: Sensor):
        for a in self.mesh.nexus.get_inters():
            try:
                connector = Connector(dendrite=a)
                sensor.post_connection(connector)
                self._registry.add_connector(connector)
            except IdenticalConnectionError:
                continue

        sensor.save_state()

    def create_network(
            self, 
            aperature_size: int = 3, 
            nexus_size: int = 3, 
            terminus_size: int = 3
        ):
        self.instantiate_mesh()
        self.mesh.nexus.add_neurons(nexus_size)
        self.mesh.terminus.add_neurons(terminus_size)
        self.connect_mesh(self.mesh.nexus.get_inters(), self.mesh.terminus.get_inters())

    def instantiate_mesh(self):
        self.mesh = MeshModel(
            nexus=Mesh(mesh=MeshType.NEXUS),
            terminus=Mesh(mesh=MeshType.TERMINUS)
        )

    def connect_mesh(self, axon_inters: list[Inter], dendrite_inters: list[Inter]):
        for n in axon_inters:
            sample_size = min(n.get_k(), len(dendrite_inters))
            samples = random.sample(dendrite_inters, sample_size)
            for s in samples:
                connector = Connector(
                    dendrite=s
                )
                self._registry.add_connector(connector)
                n.post_connection(connector)
                connector.save_state()
                n.save_state()

    def handle_growth(self, auditor_results: AuditResultsModel):
        new_nexus_inters = self.mesh.nexus.add_neurons(auditor_results.nexus_growth)
        self.mesh.terminus.add_neurons(auditor_results.terminus_growth)
        self.connect_mesh(new_nexus_inters, self.mesh.terminus.get_inters())
        sensors = self._io.get_all_neurons_by_transformer(transformer_type=TransformerTypes.ENCODER)
        motors = self._io.get_all_neurons_by_transformer(transformer_type=TransformerTypes.DECODER)

        for sensor in sensors:
            self.connect_sensor(sensor)
        
        for motor in motors:
            self.connect_motor(motor)