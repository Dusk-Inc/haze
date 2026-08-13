"""Shared fixtures and stand-in ports, so tests do not depend on Phase 4 encoders."""

from typing import Any

import pytest
import torch

from haze import Haze, makeHaze

TENSOR_BUFFERS = (
    "kind",
    "owner",
    "active",
    "alive_n",
    "src",
    "dst",
    "strength",
    "log_str",
    "epsilon",
    "edge_id",
    "alive_e",
    "step_count",
    "next_edge_id",
    "reward_bar",
    "learn_count",
)


class StubEncoder:
    """A minimal encoder standing in for the real ones until Phase 4."""

    def __init__(self, width: int = 4) -> None:
        """Records how many sensors this encoder claims."""
        self.width = width
        self.key = ""

    def encodeFeatures(self, value: Any) -> torch.Tensor:
        """Returns the observation as a float tensor, without normalizing it."""
        return torch.tensor(value, dtype=torch.float32)

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this encoder."""
        return {"width": self.width}


class StubDecoder:
    """A minimal decoder standing in for the real ones until Phase 4."""

    def __init__(self, labels: list[Any] | None = None) -> None:
        """Records the labels this decoder will claim motors for."""
        self.labels = list(labels) if labels else [0, 1]
        self.key = ""

    def toPortParams(self) -> dict[str, Any]:
        """Returns the constructor arguments needed to rebuild this decoder."""
        return {"labels": self.labels}


@pytest.fixture
def mesh_model() -> Haze:
    """Returns a small model with one encoder and one decoder already registered."""
    model = makeHaze(nexus_size=12, terminus_size=6, seed=11)
    model.registerEncoder("bits", StubEncoder(width=4))
    model.registerDecoder("bit", StubDecoder(labels=[0, 1]))
    model.ports.ensureSensorCapacity("bits", 4)
    model.ports.setLabelsActive("bit", [0, 1])
    return model


def assertMeshEqual(left: Haze, right: Haze) -> None:
    """Asserts two models hold identical mesh state, buffer by buffer."""
    for name in TENSOR_BUFFERS:
        a, b = getattr(left.mesh, name), getattr(right.mesh, name)
        assert a.shape == b.shape, f"{name} shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert torch.equal(a, b), f"{name} contents differ"
    assert torch.equal(left.mesh.generator.get_state(), right.mesh.generator.get_state())
    assert left.calcMeshSize() == right.calcMeshSize()
    assert left.ports.toLabelTable() == right.ports.toLabelTable()
