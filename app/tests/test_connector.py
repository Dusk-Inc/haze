from app.src.connector.core import Connector
from app.src.neuron.core import Inter, Motor, Sensor
from app.src.context.core import Context
from app.src.transmission.core import Transmission
from app.src.core_io.core import CoreIO
from unittest.mock import MagicMock, patch
from app.src.registry.core import Registry
from app.src.injector.core import Injector
from app.src.injector.enums import GlobalTypes
from app.src.config.core import Config
from uuid import uuid4
import pytest

@pytest.fixture(autouse=True)
def reset_injector():
    Injector._instances.clear()
    yield
    Injector._instances.clear()

def test_transmit_handlesSingleBranch():
    sensor = Sensor()
    inter = Inter()
    motor = Motor(answer='test')
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    sensor_con = Connector(
        dendrite=inter
    )
    inter_con = Connector(
        dendrite=motor
    )
    registry.add_connector(sensor_con)
    registry.add_connector(inter_con)
    sensor.post_connection(sensor_con)
    inter.post_connection(inter_con)
    sensor.transmit(
        context=Context(),
        input_value=1
    )
    motor_state = motor.get_state()
    assert motor_state != 1

def test_transmit_handlesMultipleBranches():
    sensor = Sensor()
    inter_1 = Inter()
    inter_2 = Inter()
    motor_1 = Motor(answer='test')
    motor_2 = Motor(answer='test1')
    motor_3 = Motor(answer='test1')
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    connection_list = [
        Connector(dendrite=inter_1),
        Connector(dendrite=inter_2),
        Connector(dendrite=motor_1),
        Connector(dendrite=motor_2),
        Connector(dendrite=motor_3),
        Connector(dendrite=motor_1),
        Connector(dendrite=motor_2),
        Connector(dendrite=motor_3)
    ]
    for c in connection_list:
        registry.add_connector(c)

    sensor.post_connection(connection=connection_list[0])
    sensor.post_connection(connection=connection_list[1])
    inter_1.post_connection(connection=connection_list[2])
    inter_1.post_connection(connection=connection_list[3])
    inter_1.post_connection(connection=connection_list[4])
    inter_2.post_connection(connection=connection_list[5])
    inter_2.post_connection(connection=connection_list[6])
    inter_2.post_connection(connection=connection_list[7])

    sensor.transmit(
        context=Context(),
        input_value=1
    )

    results = [motor_1.get_state(), motor_2.get_state(), motor_3.get_state()]

    assert any(result != 1 for result in results)

def test_transmit_doesNotCauseRecursion():
    sensor = Sensor()
    inter_1 = Inter()
    inter_2 = Inter()
    inter_3 = Inter()
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)

    connection_list = [
        Connector(dendrite=inter_1),
        Connector(dendrite=inter_2),
        Connector(dendrite=inter_3),
        Connector(dendrite=inter_1)
    ]

    for c in connection_list:
        registry.add_connector(c)

    sensor.post_connection(connection_list[0])
    inter_1.post_connection(connection_list[1])
    inter_2.post_connection(connection_list[2])
    inter_3.post_connection(connection_list[3])

    try:
        sensor.transmit(
            input_value=1,
            context=Context()
        )
    except RecursionError:
        pytest.fail("RecursionError occurred during signal transmission")

def test_record_doesRecordConnector():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    dendrite = Inter()
    connector = Connector(dendrite=dendrite)
    registry.add_connector(connector)
    record = connector.record()
    assert record["dendrite_id"] == dendrite.get_id(as_string=True)
    assert record["strength"] == connector.get_strength()
    assert record["epsilon"] == connector.get_epsilon()
    assert record["decay"] == connector.get_decay()
    assert record["is_default"] == connector.is_default

def test_saveState_doesReturnExpectedStateInFile():
    mock_core = MagicMock(spec=CoreIO)
    mock_core._connection_path = "mock_connection_path"
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=mock_core)
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)


    # Create a Connector instance with the mocked CoreIO
    connector = Connector(id=uuid4())
    registry.add_connector(connector)
    connector._dendrite = MagicMock()  # Mock the dendrite to avoid dependency issues
    connector._dendrite.get_id.return_value = "mock_dendrite_id"

    # Mock the record method to return a sample state
    connector.record = MagicMock(return_value={
        "dendrite_id": "mock_dendrite_id",
        "is_default": False,
        "epsilon": 0.5,
        "decay": 0.9,
        "strength": 0.7,
        "id": str(connector.get_id(as_string=True))
    })

    # Call save_state
    connector.save_state()

    # Assert that save_to_file was called with the correct arguments
    mock_core.save_to_file.assert_called_once_with(
        connector.record(),
        f"mock_connection_path/{connector.get_id(as_string=True)}"
    )
