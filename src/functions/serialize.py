"""Reading and writing the checkpoint payload, and the checks that keep its indices honest."""

import importlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from ..errors import CheckpointCorruptError, UntrustedImplementationError
from ..models import HazeConfig, LabelEntry, PortSpec
from ..tokens import NeuronKind, defaults


def toStateTensors(mesh: Any) -> dict[str, Tensor]:
    """Returns every buffer a checkpoint carries, each a contiguous copy.

    Buffers reallocated during growth are views over shared storage, which safetensors refuses;
    cloning is what makes a grown model saveable at all. See specs/packaging.md.
    """
    names = (
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
    payload = {name: getattr(mesh, name).detach().cpu().contiguous().clone() for name in names}
    payload["rng_state"] = mesh.generator.get_state().clone()
    return payload


def saveHazeCheckpoint(
    directory: Path,
    config: HazeConfig,
    tensors: dict[str, Tensor],
    labels: dict[str, LabelEntry],
) -> None:
    """Writes a checkpoint atomically, so a crash mid-write never leaves a partial one."""
    directory = Path(directory)
    staging = directory.parent / f"{directory.name}.tmp"
    if staging.exists():
        for leftover in staging.iterdir():
            leftover.unlink()
    else:
        staging.mkdir(parents=True)

    (staging / defaults.CONFIG_FILE).write_text(config.model_dump_json(indent=2))
    (staging / defaults.LABELS_FILE).write_text(
        json.dumps({k: v.model_dump() for k, v in labels.items()}, indent=2)
    )
    save_file(tensors, str(staging / defaults.WEIGHTS_FILE))

    if directory.exists():
        for leftover in directory.iterdir():
            leftover.unlink()
        directory.rmdir()
    staging.rename(directory)


def readHazeConfig(directory: Path) -> HazeConfig:
    """Reads a checkpoint's config, which must be parsed before any tensor is allocated."""
    path = Path(directory) / defaults.CONFIG_FILE
    if not path.exists():
        raise CheckpointCorruptError(f"{path} is missing; a checkpoint cannot be sized without it")
    try:
        config = HazeConfig.model_validate_json(path.read_text())
    except Exception as cause:
        raise CheckpointCorruptError(f"{path} is not a readable Haze config: {cause}") from cause
    if config.architecture != defaults.ARCHITECTURE:
        raise CheckpointCorruptError(
            f"{path} declares architecture {config.architecture!r}, not {defaults.ARCHITECTURE!r}"
        )
    if config.format_version > defaults.FORMAT_VERSION:
        raise CheckpointCorruptError(
            f"{path} is format version {config.format_version}; this build reads up to "
            f"{defaults.FORMAT_VERSION}"
        )
    return config


def readHazeLabels(directory: Path) -> dict[str, LabelEntry]:
    """Reads a checkpoint's label tables."""
    path = Path(directory) / defaults.LABELS_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {k: LabelEntry.model_validate(v) for k, v in raw.items()}
    except CheckpointCorruptError:
        raise
    except Exception as cause:
        raise CheckpointCorruptError(f"{path} is not a readable label table: {cause}") from cause


def readHazeTensors(directory: Path) -> dict[str, Tensor]:
    """Reads a checkpoint's tensor payload."""
    path = Path(directory) / defaults.WEIGHTS_FILE
    if not path.exists():
        raise CheckpointCorruptError(f"{path} is missing")
    try:
        return load_file(str(path))
    except Exception as cause:
        raise CheckpointCorruptError(f"{path} could not be read: {cause}") from cause


def loadStateTensors(mesh: Any, tensors: dict[str, Tensor]) -> None:
    """Copies a checkpoint payload into a mesh allocated at the matching capacity."""
    rng = tensors.pop("rng_state", None)
    for name, value in tensors.items():
        current = getattr(mesh, name, None)
        if current is None:
            raise CheckpointCorruptError(f"checkpoint carries an unknown tensor {name!r}")
        if tuple(current.shape) != tuple(value.shape):
            raise CheckpointCorruptError(
                f"tensor {name!r} has shape {tuple(value.shape)} but the config allocated "
                f"{tuple(current.shape)}"
            )
        current.copy_(value.to(current.dtype))
    if rng is not None:
        mesh.generator.set_state(rng.to(torch.uint8))


def ensureCheckpointCoherent(mesh: Any, labels: dict[str, LabelEntry], ports: list[PortSpec]) -> None:
    """Raises if a checkpoint's labels, ports, and tensors disagree about which neuron is what.

    Index drift is this design's most dangerous failure: it produces a model that loads without
    complaint and answers wrongly. It is checked at every load boundary rather than trusted.
    """
    slots = {spec.key: spec.slot for spec in ports}
    for key, entry in labels.items():
        if key not in slots:
            raise CheckpointCorruptError(
                f"label table names port {key!r}, which the config does not declare"
            )
        slot = slots[key]
        for value, motor in zip(entry.values, entry.motor_ids):
            if motor < 0 or motor >= int(mesh.kind.numel()):
                raise CheckpointCorruptError(
                    f"port {key!r} binds label {value!r} to motor {motor}, outside the slab"
                )
            if int(mesh.kind[motor].item()) != int(NeuronKind.MOTOR):
                raise CheckpointCorruptError(
                    f"port {key!r} binds label {value!r} to neuron {motor}, which is not a motor"
                )
            if int(mesh.owner[motor].item()) != slot:
                raise CheckpointCorruptError(
                    f"port {key!r} binds label {value!r} to motor {motor}, owned by slot "
                    f"{int(mesh.owner[motor].item())} rather than {slot}"
                )
            if not bool(mesh.alive_n[motor].item()):
                raise CheckpointCorruptError(
                    f"port {key!r} binds label {value!r} to motor {motor}, which is not live"
                )


def loadPortImpl(spec: PortSpec, trust_remote_code: bool = False) -> Any:
    """Constructs a port's implementation from the import path a checkpoint names.

    A config field naming an import path is arbitrary code execution on load, so anything
    outside the first-party allow-list requires the caller to opt in explicitly.
    """
    if not spec.impl.startswith(defaults.TRUSTED_IMPL_PREFIXES) and not trust_remote_code:
        raise UntrustedImplementationError(
            f"port {spec.key!r} names implementation {spec.impl!r}, which is outside the "
            "first-party allow-list. Loading it would execute code from the checkpoint's "
            "author. Pass trust_remote_code=True only if you trust that author."
        )
    module_path, _, attribute = spec.impl.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as cause:
        raise CheckpointCorruptError(
            f"port {spec.key!r} names implementation {spec.impl!r}, which cannot be imported: "
            f"{cause}"
        ) from cause
    return factory(**spec.params)
