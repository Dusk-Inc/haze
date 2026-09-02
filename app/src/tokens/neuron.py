"""Tokens naming the roles a neuron may take in the mesh."""
from enum import StrEnum


class NeuronType(StrEnum):
    """The role a neuron plays between input, interior, and output."""

    SENSOR = "sensor"
    INTER = "inter"
    MOTOR = "motor"
