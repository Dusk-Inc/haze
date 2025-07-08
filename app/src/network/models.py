from dataclasses import dataclass
from ..connector.core import Connector
from ..neuron.core import Inter, Sensor, Motor
from ..mesh.core import Mesh

@dataclass
class MeshModel:
    nexus: Mesh
    terminus: Mesh