from ..neuron.core import Inter
from ..errors.mesh import InsufficientNeuronCount, NoMeshLoaded
from ..connector.core import Connector
from ..tokens.mesh import MeshType
from ..registry.interface import IRegistry
from ..injector.core import Injector
from ..tokens.injector import GlobalTypes
import random

class Mesh:
    def __init__(self, mesh: MeshType):
        """Start an empty mesh of the given kind and resolve the registry."""
        self._inters: list[Inter] = []
        self.mesh: MeshType = mesh
        self.registry: IRegistry = Injector.resolve(GlobalTypes.REGISTRY)

    def is_empty(self) -> bool:
        """Return True when the mesh holds no interneurons."""
        if len(self._inters) == 0:
            return True
        
        return False

    def get_inters(self) -> list[Inter]:
        """Return the mesh's interneurons."""
        return self._inters

    def add_neurons(self, neurons: int):
        """Add `neurons` interneurons, wire them into the mesh, and return them."""
        temp_inters: list[Inter] = []
        
        for _ in range(neurons):
            inter = Inter(mesh=self.mesh)
            temp_inters.append(inter)

        self._inters.extend(temp_inters)
        self.connect_neurons(temp_inters)
        
        return temp_inters
    
    def connect_neurons(self, neurons: list[Inter]):
        """Wire each neuron to a random sample of the existing interneurons, registering every connector."""
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
        """Return the mesh's neurons and connections, raising when no mesh is loaded."""
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