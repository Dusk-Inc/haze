from src.network.core import Network
from src.network.errors import NetworkException
from unittest.mock import mock_open, patch, MagicMock
from app.src.decoder.core import ArgMax
import pytest
import json

# def test_predict_doesProduceAPrediction():
#     input_data = [-1, 20, 3, 10054]
#     expected_output = ['foo', 'bar']
#     network = Network()
#     network.attach_decoder(ArgMax(expected_output))
#     network.create_network(
#         mesh_size=3
#     )
    
#     result = network.predict(input_data)
#     assert result in expected_output