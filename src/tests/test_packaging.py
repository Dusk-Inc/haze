"""Domain tests pinning that Haze installs as one import root with its runtime present."""

import importlib

import pytest

SUBPACKAGES = [
    "haze.tokens",
    "haze.models",
    "haze.interfaces",
    "haze.modules",
    "haze.modules.encoders",
    "haze.modules.decoders",
    "haze.functions",
    "haze.errors",
    "haze.other",
]


def test_import_doesResolveUnderOneRoot():
    """Asserts the on-disk src/ tree installs as the single import root `haze`."""
    haze = importlib.import_module("haze")
    assert haze.__version__ == "0.2.0"
    assert haze.__file__.endswith("/src/__init__.py")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_import_doesResolveEverySubpackage(name: str):
    """Asserts each convention folder is importable as a `haze.` subpackage, not a top-level one."""
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} carries no module docstring"


@pytest.mark.parametrize("name", sorted({n.split(".")[1] for n in SUBPACKAGES}))
def test_import_doesNotLeakConventionFoldersToTopLevel(name: str):
    """Asserts the package-dir mapping did not install `models`, `functions`, and friends globally."""
    try:
        leaked = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return
    assert leaked is None or "/haze/" not in (leaked.origin or ""), (
        f"convention folder {name!r} is importable as a top-level name from haze"
    )


def test_runtime_doesProvideTensorStack():
    """Asserts the tensor, serialization, validation, and hub dependencies are installed."""
    import huggingface_hub
    import pydantic
    import safetensors
    import torch

    assert torch.__version__
    assert pydantic.__version__.startswith("2.")
    assert safetensors.__version__
    assert huggingface_hub.__version__


def test_runtime_doesSupportTheOperationsPropagationNeeds():
    """Asserts index_add_ and scatter-style selection behave as the engine assumes."""
    import torch

    val = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    dst = torch.tensor([2, 3, 3, 2])
    contrib = torch.tensor([[0.5, 0.25, 0.25, 0.5]])

    out = torch.zeros_like(val).index_add_(1, dst, contrib)

    assert out.tolist() == [[0.0, 0.0, 1.0, 0.5]], "index_add_ must sum repeated destinations"
    assert torch.where(out > 0.6, out, torch.zeros_like(out)).tolist() == [[0.0, 0.0, 1.0, 0.0]]
