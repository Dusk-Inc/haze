"""Enumerations naming neuron populations and the roles a port can hold."""

from enum import IntEnum, StrEnum


class NeuronKind(IntEnum):
    """Names the four neuron populations, valued to match the `kind` buffer's encoding."""

    SENSOR = 0
    NEXUS = 1
    TERMINUS = 2
    MOTOR = 3
    FREE = 255


class MeshRegion(StrEnum):
    """Names the two interneuron meshes signal crosses between sensors and motors."""

    NEXUS = "nexus"
    TERMINUS = "terminus"


class PortRole(StrEnum):
    """Names whether a port feeds sensors or reads motors."""

    ENCODER = "encoder"
    DECODER = "decoder"


class LabelType(StrEnum):
    """Names how a decoder label round-trips through JSON, which erases Python types."""

    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    VECTOR = "vector"


INTERNEURON_KINDS = (NeuronKind.NEXUS, NeuronKind.TERMINUS)

MESH_REGION_KINDS = {
    MeshRegion.NEXUS: NeuronKind.NEXUS,
    MeshRegion.TERMINUS: NeuronKind.TERMINUS,
}

BLOCK_ORDER = (
    NeuronKind.SENSOR,
    NeuronKind.MOTOR,
    NeuronKind.NEXUS,
    NeuronKind.TERMINUS,
)
"""Capacity-block order.

Populations that grow on demand but stay small precede those that grow steadily, so that mesh
growth never shifts the motor block and motor readout stays a contiguous slice.
"""

UNOWNED = -1
"""Owner value for a neuron belonging to the mesh itself rather than to any port."""
