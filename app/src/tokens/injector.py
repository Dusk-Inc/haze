"""Tokens naming the globally injectable components the injector resolves."""
from enum import StrEnum


class GlobalTypes(StrEnum):
    """The global component an injector lookup resolves to."""

    AUDITOR = "auditor"
    REGISTRY = "registry"
    CONFIG = "config"
    CORE = "core"
    NETWORK = "network"
    NEURON_IO = "neuron_io"
