from app.src.registry.core import Registry
from app.src.connector.core import Connector
from app.src.neuron.core import Neuron, Inter
from app.src.injector.core import Injector
from app.src.injector.enums import GlobalTypes
from app.src.config.core import Config
from app.src.core_io.core import CoreIO
import pytest

# what does registry need to do?
# registry needs to store connection strengths and epsilons
# add new connection strengths when connections are added
# record connections that fire during prediction
# needs to update the connection strength and decay epsilons after learning
# needs to record this information for future use
@pytest.fixture(autouse=True)
def reset_injector():
    Injector._instances.clear()
    yield
    Injector._instances.clear()

def test_addConnection_doesAddConnection():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    neuron = Inter()
    connector = Connector(dendrite=neuron)
    registry.add_connector(connector, 0.8, 0.5)
    assert connector._index == 0
    assert len(registry._connectors) == 1
    assert len(registry._strength) == 1
    assert len(registry._epsilon) == 1
    assert len(registry._status) == 1
    assert registry._status[0] == 0
    assert registry._epsilon[0] == 0.5

def test_addConnection_addsMultipleConnections():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    neuron_1 = Inter()
    neuron_2 = Inter()
    connector_1 = Connector(dendrite=neuron_1)
    connector_2 = Connector(dendrite=neuron_2)
    registry.add_connector(connector_1, 0.8, 0.5)
    registry.add_connector(connector_2, 0.8, 0.5)
    assert connector_2._index == 1
    assert len(registry._connectors) == 2
    assert len(registry._strength) == 2
    assert len(registry._epsilon) == 2
    assert len(registry._status) == 2
    assert registry._status[1] == 0
    assert registry._epsilon[1] == 0.5