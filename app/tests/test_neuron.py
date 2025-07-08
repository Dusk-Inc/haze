from app.src.neuron.core import Motor, Inter, Sensor
from app.src.terminal.errors import IdenticalConnectionError, NoMatchingConnectionError
import pytest
from app.src.connector.core import Connector
from app.src.signal.core import Signal
from app.src.context.core import Context
from app.src.transmission.core import Transmission
from unittest.mock import MagicMock, patch
from app.src.decoder.enums import DecoderType
from app.src.encoder.enums import EncoderType
from app.src.mesh.enums import MeshType
from app.src.core_io.core import CoreIO
from app.src.registry.core import Registry
from app.src.injector.core import Injector
from app.src.injector.enums import GlobalTypes
from app.src.config.core import Config
from uuid import uuid4

@pytest.fixture(autouse=True)
def reset_injector():
    Injector._instances.clear()
    yield
    Injector._instances.clear()

def test_setConnectionStrength_doesRaiseErrorWhenConnectionNotFound():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    connections = [
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        )
    ]
    for c in connections:
        registry.add_connector(c)
    node = Inter()

    for c in connections:
        node.post_connection(c)

    new_connection = Connector(
        dendrite=Inter()
    )
    registry.add_connector(new_connection)

    with pytest.raises(NoMatchingConnectionError):
        node.put_connection(new_connection)

def test_addConnection_doesAddConnection():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    connections = [
        Connector(
            dendrite=Inter(),
        ),
        Connector(
            dendrite=Inter(),
        ),
        Connector(
            dendrite=Inter(),
        ),
        Connector(
            dendrite=Inter(),
        )
    ]
    for c in connections:
        registry.add_connector(c)
    node = Inter()

    for c in connections:
        node.post_connection(c)

    new_connection = Connector(
        dendrite=Inter()
    )
    registry.add_connector(new_connection)

    node.post_connection(new_connection)
    new_connections = node.get_connections()

    assert new_connections[len(new_connections)-1].get_id() == new_connection.get_id()

def test_addConnection_doesRaiseErrorWhenConnectionAlreadyExists():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    connections = [
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        )
    ]

    for c in connections:
        registry.add_connector(c)

    node = Inter()

    for c in connections:
        node.post_connection(c)

    new_connection = Connector(
        id=connections[0].get_id(), 
        dendrite=Inter()
    )
    
    registry.add_connector(new_connection)

    with pytest.raises(IdenticalConnectionError):
        node.post_connection(new_connection)

def test_removeConnection_doesRemoveConnection():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    connections = [
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        )
    ]

    for c in connections:
        registry.add_connector(c)

    node = Inter()

    for c in connections:
        node.post_connection(c)

    node.delete_connection(connections[0])
    new_connections = node.get_connections()

    for nc in new_connections:
        assert nc.get_id() != connections[0].get_id()

def test_removeConnection_doesRaiseErrorWhenConnectionNotFound():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    connections = [
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        ),
        Connector(
            dendrite=Inter()
        )
    ]

    for c in connections:
        registry.add_connector(c)

    node = Inter()
    for c in connections:
        node.post_connection(c)

    bad_connection = Connector(
        dendrite=Inter()
    )

    with pytest.raises(NoMatchingConnectionError):
        node.delete_connection(bad_connection)

def test_setState_doesChangeState():
    neuron = Motor(answer='foobar')
    signal = Signal(0.5)
    context = Context()
    egress = Transmission(context, signal)
    neuron.transmit(egress)
    assert neuron.get_state() == 0.5

def test_chooseDefault_doesChooseDefault():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    neuron = Inter()
    connections = [
        Connector(dendrite=Inter()),
        Connector(dendrite=Inter()),
        Connector(dendrite=Inter()),
        Connector(dendrite=Inter()),
        Connector(dendrite=Inter())
    ]
    for x, c in enumerate(connections):
        registry.add_connector(c, strength=x)
        neuron.post_connection(c)

    neuron.choose_default()

    for c in neuron.get_connections():
        if c.get_strength() == 4:
            assert c.is_default
        else:
            assert c.is_default == False


def test_inter_save_state():
    mock_core = MagicMock(spec=CoreIO)
    mock_core._mesh_path = "mock_mesh_path"
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=mock_core)
    inter = Inter(mesh=MeshType.APERTURE, id=uuid4())
    inter.record = MagicMock(return_value={
        "type": "INTER",
        "id": str(inter.get_id(as_string=True)),
        "connections": []
    })
    inter.save_state()
    mock_core.save_to_file.assert_called_once_with(
        inter.record(),
        f"mock_mesh_path/aperture/{inter.get_id(as_string=True)}"
    )