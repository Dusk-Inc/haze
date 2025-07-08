from app.src.haze.core import Haze
import numpy as np
from app.src.registry.core import Registry
from app.src.injector.core import Injector
from app.src.injector.enums import GlobalTypes
from app.src.network.core import Network
from app.src.neuron.core import Inter
from app.src.connector.core import Connector
from app.src.mesh.enums import MeshType
from app.src.encoder.core import NumericEncoder, TextEncoder
from app.src.encoder.enums import EncoderType
from app.src.decoder.core import ArgMax, Regressor
from app.src.haze.models import InputModel, IdeaModel, DecoderModel
import pytest

@pytest.fixture
def haze() -> Haze:
    return Haze(persist=False)

def test_predict_doesReturnValue(haze: Haze):
    input_data = [1,3,5,7,9]
    outputs = ["foo", "bar"]
    haze.load()
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder()],
            decoders=[DecoderModel(decoder=ArgMax(), outputs=outputs)
            ]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    haze.observe(input_data=input_data, encoder=lexical_set[0].encoders[0])
    result = haze.predict()[lexical_set[0].decoders[0].decoder.get_id(as_string=True)][0]
    assert result in outputs

def test_predict_doesReturnRegressionValue(haze: Haze):
    haze.load()
    input_data = [1,2,3,4,5,6]
    outputs = [1, 10]
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder()],
            decoders=[DecoderModel(decoder=Regressor(), outputs=outputs)
            ]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    haze.observe(input_data=input_data, encoder=lexical_set[0].encoders[0])
    result = haze.predict()[lexical_set[0].decoders[0].decoder.get_id(as_string=True)][0]
    assert result > outputs[0] and result < outputs[1]

def test_predict_doesHandleMultipleEncoders(haze: Haze):
    haze.load()
    numeric_input_data = [1,2,3,4,5,6,7,8,9]
    text_data = ['foo', 'bar']
    outputs = [1, 10]
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder(), TextEncoder()],
            decoders=[DecoderModel(decoder=Regressor(), outputs=outputs)]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    haze.observe(input_data=numeric_input_data, encoder=lexical_set[0].encoders[0])
    haze.observe(input_data=text_data, encoder=lexical_set[0].encoders[1])
    result = haze.predict()[lexical_set[0].decoders[0].decoder.get_id(as_string=True)][0]
    assert result > outputs[0] and result < outputs[1]

def test_predict_doesHandleMultipleInputSizes(haze: Haze):
    haze.load()
    first_numeric_input_data = [1,2,3,4,5,6,7,8,9]
    first_text_data = ['foo', 'bar']
    second_numeric_input_data = [1,2,3,4,5,6,7,8,9]
    second_text_data = ['john', 'doe']
    outputs = [1, 10]
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder(), TextEncoder()],
            decoders=[DecoderModel(decoder=Regressor(), outputs=outputs)]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    haze.observe(input_data=first_numeric_input_data, encoder=lexical_set[0].encoders[0])
    haze.observe(input_data=first_text_data, encoder=lexical_set[0].encoders[1])
    result_one = haze.predict()[lexical_set[0].decoders[0].decoder.get_id(as_string=True)][0]
    haze.observe(input_data=second_numeric_input_data, encoder=lexical_set[0].encoders[0])
    haze.observe(input_data=second_text_data, encoder=lexical_set[0].encoders[1])
    result_two = haze.predict()[lexical_set[0].decoders[0].decoder.get_id(as_string=True)][0]
    assert result_one > outputs[0] and result_one < outputs[1]
    assert result_two > outputs[0] and result_two < outputs[1]

def test_predict_canProduceSequentialOutput(haze: Haze):
    haze.load()
    haze._sequential = True
    input_data = [0,1,0,1]
    outputs = [0, 1]
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder()],
            decoders=[DecoderModel(decoder=ArgMax(), outputs=outputs)]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    haze.observe(input_data=input_data, encoder=lexical_set[0].encoders[0])
    result = haze.predict(limit=4)[lexical_set[0].decoders[0].decoder.get_id(as_string=True)]
    assert len(result) > 2 and len(result) < 5

# this one isn't going to work until auditor is fully integrated
def test_learn_doesChangeStrengthsAndEpsilonsWhenTargetAndRewardAreAvailable(haze: Haze):
    haze.load()
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    numeric_input_data = [1,2,3,4,5,6,7,8,9]
    outputs = [1, 10]
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder()],
            decoders=[DecoderModel(decoder=Regressor(), outputs=outputs)
            ]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    haze.observe(input_data=numeric_input_data, encoder=lexical_set[0].encoders[0])
    old_strengths = registry._strength
    haze.predict()
    haze.learn(reward=0.1)
    new_strengths = registry._strength
    assert not np.array_equal(old_strengths, new_strengths)
    assert len(haze.network.mesh.nexus.get_inters()) > 3

def test_learn_doesClearConnectionsWhenNeuronIsUnderConnected(haze: Haze):
    haze.load()
    registry: Registry = Injector.resolve(GlobalTypes.REGISTRY)
    network: Network = Injector.resolve(GlobalTypes.NETWORK)
    underconnected_inter = Inter(mesh=MeshType.NEXUS)
    sparse_connector = Connector(dendrite=network.mesh.nexus.get_inters()[0])
    registry.add_connector(sparse_connector)
    underconnected_inter.post_connection(sparse_connector)
    network.mesh.nexus._inters.append(underconnected_inter)
    numeric_input_data = [1,2,3,4,5,6,7,8,9]
    outputs = [1, 10]
    lexical_set = [
        IdeaModel(
            encoders=[NumericEncoder()],
            decoders=[DecoderModel(decoder=Regressor(), outputs=outputs)
            ]
        )
    ]
    haze.set_lexical_chain(lexical_chain=lexical_set)
    haze.observe(input_data=numeric_input_data, encoder=lexical_set[0].encoders[0])
    haze.predict()
    haze.learn(reward=0.1)
    assert len(underconnected_inter._connections) > 1