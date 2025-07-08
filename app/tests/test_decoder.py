from app.src.decoder.core import Regressor, ArgMax, Binary, SoftMax, TopK, Bitmask, Vector
from app.src.neuron.core import Motor
from app.src.signal.core import Signal
from app.src.transmission.core import Transmission
from app.src.context.core import Context
from app.src.core_io.core import CoreIO
from app.src.injector.core import Injector
from app.src.injector.enums import GlobalTypes
from app.src.registry.core import Registry
from app.src.config.core import Config
from app.src.auditor.core import Auditor
from app.src.network.core import Network
import pytest

@pytest.fixture(autouse=True)
def reset_injector():
    Injector._instances.clear()
    yield
    Injector._instances.clear()

# these require encoders now to run properly
def test_regressor_doesPredictNumber():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    signal_1 = Signal(
        value=0.3
    )
    signal_2 = Signal(
        value=0.7
    )
    transmission_1 = Transmission(
        signal=signal_1,
        context=Context()
    )
    transmission_2 = Transmission(
        signal=signal_2,
        context=Context()
    )
    expected_answer = 7.3
    regressor = Regressor()
    regressor.set_outputs([1,10])
    regressor._motors[0].transmit(transmission_1)
    regressor._motors[1].transmit(transmission_2)
    actual_result = regressor.predict()
    assert expected_answer == actual_result

def test_argmax_doesPredictWords():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    outputs = ['foo', 'bar']
    signal_1 = Signal(
        value=0.3
    )
    signal_2 = Signal(
        value=0.7
    )
    transmission_1 = Transmission(
        signal=signal_1,
        context=Context()
    )
    transmission_2 = Transmission(
        signal=signal_2,
        context=Context()
    )
    expected_answer = outputs[1]
    classifier = ArgMax()
    classifier.set_outputs(outputs=outputs)
    classifier._motors[0].transmit(transmission_1)
    classifier._motors[1].transmit(transmission_2)
    actual_result = classifier.predict()
    assert expected_answer == actual_result

def test_binary_doesPredictAnswer():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    signal_1 = Signal(
        value=0.3
    )
    signal_2 = Signal(
        value=0.7
    )
    expected_answer = False
    transmission_1 = Transmission(
        signal=signal_1,
        context=Context()
    )
    transmission_2 = Transmission(
        signal=signal_2,
        context=Context()
    )
    binary = Binary()
    binary.set_outputs()
    binary._motors[0].transmit(transmission_1)
    binary._motors[1].transmit(transmission_2)
    actual_answer = binary.predict()
    assert expected_answer == actual_answer

def test_softMax_doesPredictAnswer():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    outputs = ['foo', 'bar']
    signal_1 = Signal(
        value=0.3
    )
    signal_2 = Signal(
        value=0.7
    )
    transmission_1 = Transmission(
        signal=signal_1,
        context=Context()
    )
    transmission_2 = Transmission(
        signal=signal_2,
        context=Context()
    )
    softmax = SoftMax()
    softmax.set_outputs(outputs=outputs)
    softmax._motors[0].transmit(transmission_1)
    softmax._motors[1].transmit(transmission_2)
    actual_result = softmax.predict()
    assert len(actual_result) == 2

def test_topK_doesPredictAnswer():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    outputs = ['foo', 'bar', 'doe']
    signal_1 = Signal(
        value=0.3
    )
    signal_2 = Signal(
        value=0.7
    )
    signal_3 = Signal(
        value=0.8
    )
    transmission_1 = Transmission(
        signal=signal_1,
        context=Context()
    )
    transmission_2 = Transmission(
        signal=signal_2,
        context=Context()
    )
    transmission_3 = Transmission(
        signal=signal_3,
        context=Context()
    )
    expected_answer = ['doe', 'bar']
    top_k = TopK()
    top_k.set_outputs(outputs=outputs)
    top_k._motors[0].transmit(transmission_1)
    top_k._motors[1].transmit(transmission_2)
    top_k._motors[2].transmit(transmission_3)
    actual_result = top_k.predict(2)
    assert expected_answer == actual_result

def test_bitMask_doesPredictAnswer():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    outputs = ['foo', 'bar', 'doe']
    signal_1 = Signal(
        value=0.3
    )
    signal_2 = Signal(
        value=0.7
    )
    signal_3 = Signal(
        value=0.8
    )
    transmission_1 = Transmission(
        signal=signal_1,
        context=Context()
    )
    transmission_2 = Transmission(
        signal=signal_2,
        context=Context()
    )
    transmission_3 = Transmission(
        signal=signal_3,
        context=Context()
    )
    expected_answer = ['bar', 'doe']
    bitmask = Bitmask()
    bitmask.set_outputs(outputs=outputs)
    bitmask._motors[0].transmit(transmission_1)
    bitmask._motors[1].transmit(transmission_2)
    bitmask._motors[2].transmit(transmission_3)
    actual_result = bitmask.predict(threshold=0.5)
    assert expected_answer == actual_result

def test_vector_doesPredictAnswer():
    Injector.register(GlobalTypes.REGISTRY, instance=Registry())
    Injector.register(GlobalTypes.CONFIG, instance=Config())
    Injector.register(GlobalTypes.CORE, instance=CoreIO())
    Injector.register(GlobalTypes.AUDITOR, instance=Auditor())
    Injector.register(GlobalTypes.NETWORK, instance=Network())
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    network.create_network()
    outputs = [[1.0, 0.0], [0.0, 1.0]]
    signal_1 = Signal(
        value=0.3
    )
    signal_2 = Signal(
        value=0.7
    )
    transmission_1 = Transmission(
        signal=signal_1,
        context=Context()
    )
    transmission_2 = Transmission(
        signal=signal_2,
        context=Context()
    )
    expected_answer = [0.3, 0.7]
    vector = Vector()
    vector.set_outputs(outputs=outputs)
    vector._motors[0].transmit(transmission_1)
    vector._motors[1].transmit(transmission_2)
    actual_result = vector.predict()
    assert expected_answer == actual_result