"""The port registry: which encoder or decoder owns which neurons, keyed by a caller's name."""

from typing import Any, Iterable

import torch
from torch import Tensor

from ..errors import LabelSpaceError, PortAlreadyRegisteredError, PortNotFoundError
from ..models import LabelEntry, PortSpec
from ..tokens import LabelType, NeuronKind, PortRole
from .mesh import MeshState

LABEL_TYPES: dict[type, LabelType] = {
    bool: LabelType.BOOL,
    int: LabelType.INT,
    float: LabelType.FLOAT,
    str: LabelType.STR,
}


def calcLabelType(values: Iterable[Any]) -> LabelType:
    """Returns how a label set round-trips through JSON, which erases Python types.

    Checked before int because bool is a subclass of int, and a decoder that distinguishes
    `True` from `1` would otherwise silently get the wrong one back from a checkpoint.
    """
    values = list(values)
    if not values:
        raise LabelSpaceError("a decoder must declare at least one label")
    first = values[0]
    if isinstance(first, (list, tuple)):
        return LabelType.VECTOR
    for py_type, label_type in LABEL_TYPES.items():
        if isinstance(first, py_type):
            return label_type
    raise LabelSpaceError(f"labels of type {type(first).__name__!r} cannot be persisted")


def toLabelKey(value: Any) -> Any:
    """Returns a hashable form of a label, so vector labels can index a table."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


class PortRegistry:
    """Owns port slots and the neurons each slot has been allocated.

    Keyed by a caller-supplied string rather than by an encoder's class, which is what lets two
    instances of the same encoder hold disjoint sensors and makes genuine multi-modal use
    possible. See specs/ports.md.
    """

    def __init__(self, mesh: MeshState) -> None:
        """Creates an empty registry bound to a mesh."""
        self.mesh = mesh
        self.specs: dict[str, PortSpec] = {}
        self.impls: dict[str, Any] = {}
        self.labels: dict[str, LabelEntry] = {}
        self.sensor_ids: dict[str, Tensor] = {}
        self._label_index: dict[str, dict[Any, int]] = {}


    def registerPort(self, key: str, impl: Any, role: PortRole, width: int | None = None) -> int:
        """Registers an encoder or decoder under a key and returns its slot."""
        if key in self.specs:
            raise PortAlreadyRegisteredError(
                f"port {key!r} is already registered; re-registering would rebind its neurons "
                "and discard everything the mesh learned about them"
            )
        slot = len(self.specs)
        self.specs[key] = PortSpec(
            slot=slot,
            role=role,
            key=key,
            impl=f"{type(impl).__module__}.{type(impl).__qualname__}",
            width=width,
            params=impl.toPortParams() if hasattr(impl, "toPortParams") else {},
        )
        self.impls[key] = impl
        return slot

    def findPortSpec(self, key: str) -> PortSpec:
        """Returns a port's spec, or raises naming the key."""
        try:
            return self.specs[key]
        except KeyError:
            known = ", ".join(sorted(self.specs)) or "none"
            raise PortNotFoundError(
                f"no port registered under {key!r}; registered ports: {known}"
            ) from None

    def findPortKeys(self, role: PortRole) -> list[str]:
        """Returns the keys of every port in a role, in registration order."""
        return [k for k, spec in self.specs.items() if spec.role == role]


    def ensureSensorCapacity(self, key: str, width: int) -> Tensor:
        """Allocates sensors so a port owns exactly `width` of them, and returns them in order.

        Existing sensors keep their indices; only the shortfall is allocated. This is what lets
        an encoder's feature count grow without discarding what the mesh learned about the
        features it already had.
        """
        spec = self.findPortSpec(key)
        held = self.sensor_ids.get(key, torch.empty(0, dtype=torch.int64))
        shortfall = width - int(held.numel())
        if shortfall > 0:
            fresh = self.mesh.allocNeuronIds(shortfall, NeuronKind.SENSOR, owner=spec.slot)
            self.mesh.connectSensors(fresh)
            held = torch.cat([held, fresh])
            self.sensor_ids[key] = held
            spec.width = int(held.numel())
        return held[:width]


    def setLabelsActive(self, key: str, values: Iterable[Any]) -> LabelEntry:
        """Allocates motors for any unseen label and marks exactly `values` active.

        A label's identity is its motor index, so a binding once made is never reassigned; a
        label that leaves the active set is masked rather than deallocated.
        """
        spec = self.findPortSpec(key)
        values = list(values)
        if not values:
            raise LabelSpaceError(f"decoder {key!r} was given an empty label set")

        keys = [toLabelKey(v) for v in values]
        if len(set(keys)) != len(keys):
            raise LabelSpaceError(f"decoder {key!r} was given a duplicated label")

        entry = self.labels.get(key)
        if entry is None:
            entry = LabelEntry(
                value_type=calcLabelType(values), values=[], motor_ids=[], active=[]
            )
            self.labels[key] = entry
            self._label_index[key] = {}

        index = self._label_index[key]
        unseen = [v for v, k in zip(values, keys) if k not in index]
        if unseen and calcLabelType(unseen) != entry.value_type:
            raise LabelSpaceError(
                f"decoder {key!r} holds {entry.value_type} labels but was given "
                f"{calcLabelType(unseen)}; a mixed table cannot round-trip through JSON, which "
                "records one value type per decoder"
            )
        if unseen:
            motors = self.mesh.allocNeuronIds(len(unseen), NeuronKind.MOTOR, owner=spec.slot)
            self.mesh.connectMotors(motors)
            for value, motor in zip(unseen, motors.tolist()):
                index[toLabelKey(value)] = len(entry.values)
                entry.values.append(value)
                entry.motor_ids.append(int(motor))
                entry.active.append(False)

        wanted = set(keys)
        for position, value in enumerate(entry.values):
            entry.active[position] = toLabelKey(value) in wanted

        motor_ids = torch.tensor(entry.motor_ids, dtype=torch.int64)
        flags = torch.tensor(entry.active, dtype=torch.bool)
        self.mesh.active[motor_ids] = flags
        return entry

    def findLabels(self, key: str) -> LabelEntry:
        """Returns a decoder's label table, or raises if it has none yet."""
        entry = self.labels.get(key)
        if entry is None:
            raise LabelSpaceError(f"decoder {key!r} has no labels; call setLabelsActive first")
        return entry

    def findMotorId(self, key: str, value: Any) -> int:
        """Returns the motor bound to a label."""
        entry = self.findLabels(key)
        position = self._label_index[key].get(toLabelKey(value))
        if position is None:
            raise LabelSpaceError(f"decoder {key!r} has no label {value!r}")
        return entry.motor_ids[position]

    def findActiveMotorIds(self, key: str) -> Tensor:
        """Returns the motors of a decoder's currently active labels, in label order."""
        entry = self.findLabels(key)
        return torch.tensor(
            [m for m, on in zip(entry.motor_ids, entry.active) if on], dtype=torch.int64
        )


    def toPortSpecs(self) -> list[PortSpec]:
        """Returns every port's spec, ordered by slot, for writing into a checkpoint config."""
        return [self.specs[k] for k in sorted(self.specs, key=lambda k: self.specs[k].slot)]

    def toLabelTable(self) -> dict[str, LabelEntry]:
        """Returns every decoder's label table, for writing beside the checkpoint config."""
        return dict(self.labels)

    def loadLabelTable(self, table: dict[str, LabelEntry]) -> None:
        """Restores label tables from a checkpoint and rebuilds their lookup index."""
        self.labels = dict(table)
        self._label_index = {
            key: {toLabelKey(v): i for i, v in enumerate(entry.values)}
            for key, entry in table.items()
        }

    def loadSensorIds(self) -> None:
        """Rebuilds each encoder's sensor list from neuron ownership after a checkpoint load."""
        for key, spec in self.specs.items():
            if spec.role != PortRole.ENCODER:
                continue
            self.sensor_ids[key] = self.mesh.findPortNeuronIds(spec.slot)
