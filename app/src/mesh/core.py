from ..neuron.core import Inter
from .errors import InsufficientNeuronCount, NoMeshLoaded
from ..connector.core import Connector
from .enums import MeshType
from ..registry.interface import IRegistry
from ..injector.core import Injector
from ..injector.enums import GlobalTypes
import random

class Mesh:
    def __init__(self, mesh: MeshType):
        self._inters: list[Inter] = []
        self.mesh: MeshType = mesh
        self.registry: IRegistry = Injector.resolve(GlobalTypes.REGISTRY)

    def is_empty(self) -> bool:
        if len(self._inters) == 0:
            return True
        
        return False

    def get_inters(self) -> list[Inter]:
        return self._inters

    def add_neurons(self, neurons: int):
        temp_inters: list[Inter] = []
        
        for _ in range(neurons):
            inter = Inter(mesh=self.mesh)
            temp_inters.append(inter)

        self._inters.extend(temp_inters)
        self.connect_neurons(temp_inters)
        
        return temp_inters
    
    def connect_neurons(self, neurons: list[Inter]):
        k = len(self._inters)
        for i in neurons:
            i.set_k(k)
            reliable_inters = [inter for inter in self._inters if inter.get_id() != i.get_id()]
            sample_size = min(i.get_k(), len(reliable_inters))
            sample = random.sample(reliable_inters, sample_size)
            for s in sample:
                i.set_k(k)
                connector = Connector(dendrite=s)
                self.registry.add_connector(connector)
                i.post_connection(connector)
                connector.save_state()
                
            i.save_state()

    def record(self):
        if len(self._inters) == 0:
            raise NoMeshLoaded("No mesh has been created or loaded.")
        neurons = []
        connections = []

        for i in self._inters:
            neurons.append(
                {
                    "id": str(i.get_id()),
                    "type": i.get_type()
                }
            )
            for c in i.get_connections():
                connections.append(
                    c.record()
                )

        return {"neurons": neurons, "connections": connections}