from dataclasses import dataclass
from uuid import UUID
from numpy.typing import NDArray

@dataclass
class NeuronLocationModel:
    n_id: UUID
    location: NDArray

@dataclass
class MeshClusterModel:
    sensors: NDArray
    inters: NDArray
    motors: NDArray