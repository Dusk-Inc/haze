from app.src.encoder.core import NumericEncoder
from app.src.network.core import Network
from app.src.registry.core import Registry
from app.src.injector.core import Injector
from app.src.injector.enums import GlobalTypes
from app.src.config.core import Config
from app.src.core_io.core import CoreIO
from app.src.auditor.core import Auditor
import pytest

@pytest.fixture(autouse=True)
def reset_injector():
    Injector._instances.clear()
    yield
    Injector._instances.clear()

def test_numeric_doesNormalizeData():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    input_data = [-1,3,77, 1003]
    callback = lambda x: x
    encoder = NumericEncoder()
    encoder.set_connector(callback=callback)

    result = encoder.normalize(input_data)
    for r in result:
        assert r > 0 and r < 1

def test_numeric_doesConnectSensors():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    network = Network()
    network.create_network(3,3,3)
    for n in network.mesh.aperature.get_inters():
        num_conn = len(n.get_connections())
        assert num_conn >= 4
    
    for n in network.mesh.nexus.get_inters():
        num_conn = len(n.get_connections())
        assert num_conn >= 4