from app.src.mesh.core import Mesh
from app.src.errors.mesh import InsufficientNeuronCount, NoMeshLoaded
import pytest
from app.src.tokens.mesh import MeshType
from app.src.core_io.core import CoreIO
from app.src.registry.core import Registry
from app.src.injector.core import Injector
from app.src.config.core import Config
from app.src.tokens.injector import GlobalTypes

@pytest.fixture(autouse=True)
def reset_injector():
    Injector._instances.clear()
    yield
    Injector._instances.clear()

@pytest.fixture
def mesh() -> Mesh:
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    return Mesh(mesh=MeshType.APERTURE)


def test_generateMesh_doesGenerateMesh(mesh: Mesh):
    mesh.add_neurons(
        neurons=3,
    )

    assert len(mesh.get_inters()) == 3
    
    for i in mesh.get_inters():
        connections = len(i.get_connections())
        assert connections >= 1 and connections <=4
    

def test_recordMesh_doesRecordMeshInDict(mesh: Mesh):
    mesh.add_neurons(
        neurons=7
    )

    network: dict = mesh.record()
    assert len(network["neurons"]) == 7
    assert len(network["connections"]) > 0

def test_recordMesh_raisesErrorWhenNoMeshCreated(mesh: Mesh):
    with pytest.raises(NoMeshLoaded):
        mesh.record()