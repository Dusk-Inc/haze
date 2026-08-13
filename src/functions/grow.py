"""Growth: deciding when the mesh is too small for its task, and enlarging it when it is."""

from typing import Sequence

from ..models import GrowthPlan, HazeHyper, MeshCounts
from ..tokens import NeuronKind


def calcErrorRate(rewards: Sequence[float]) -> float:
    """Returns the share of recent observations that went unrewarded."""
    if not rewards:
        return 0.0
    return 1.0 - sum(float(r) for r in rewards) / len(rewards)


def calcUncertaintyRate(confidences: Sequence[float]) -> float:
    """Returns how unsure the mesh has been recently, as one minus its mean confidence."""
    if not confidences:
        return 0.0
    return 1.0 - sum(float(c) for c in confidences) / len(confidences)


def calcGrowthAmount(population: int, error: float, uncertainty: float, hyper: HazeHyper) -> int:
    """Returns how many interneurons to add to a population of the given size.

    Proportional to the population rather than a fixed count, so growth stays meaningful as the
    mesh gets larger instead of becoming a rounding error, and scaled by error and uncertainty
    together: a mesh that is wrong *and* unsure grows more than one that is wrong but confident,
    because the confident one has a working rule and the wrong one, which more neurons do not
    fix. See specs/growth.md.
    """
    if population <= 0:
        return 0
    scale = error * (1.0 + uncertainty)
    return max(1, int(round(hyper.growth_rate * population * scale)))


def calcGrowthPlan(
    rewards: Sequence[float],
    confidences: Sequence[float],
    counts: MeshCounts,
    hyper: HazeHyper,
) -> GrowthPlan:
    """Returns how much to grow each mesh, given recent reward and confidence.

    Both windows must be full before anything is planned, which is what keeps one unlucky
    observation from restructuring the network.
    """
    window = hyper.audit_window
    if len(rewards) < window or len(confidences) < window:
        return GrowthPlan()

    error = calcErrorRate(rewards[-window:])
    uncertainty = calcUncertaintyRate(confidences[-window:])
    if error <= hyper.growth_threshold:
        return GrowthPlan(error_rate=error, confidence_rate=uncertainty)

    return GrowthPlan(
        nexus=calcGrowthAmount(counts.nexus, error, uncertainty, hyper),
        terminus=calcGrowthAmount(counts.terminus, error, uncertainty, hyper),
        error_rate=error,
        confidence_rate=uncertainty,
        triggered=True,
    )


def applyGrowth(mesh, plan: GrowthPlan) -> int:
    """Adds the planned interneurons and wires them in, returning how many were added.

    Each new interneuron is wired in both directions and given its share of the sensor or motor
    connections, because one that is reachable but not readable — or readable but not reachable —
    contributes nothing and is exactly what `ensureNoOrphans` would have to rescue a moment
    later. Growth writes into slack and never renumbers a live neuron, so every edge, label
    binding, and port claim the mesh has already learned survives it. See specs/growth.md.
    """
    if plan.isEmpty:
        return 0

    added = 0
    if plan.nexus:
        fresh = mesh.allocNeuronIds(plan.nexus, NeuronKind.NEXUS)
        existing = mesh.findNeuronIds(NeuronKind.NEXUS)
        sensors = mesh.findNeuronIds(NeuronKind.SENSOR)
        terminus = mesh.findNeuronIds(NeuronKind.TERMINUS)
        mesh.connectNeurons(fresh, existing)
        mesh.connectNeurons(existing, fresh)
        if sensors.numel():
            mesh.connectNeurons(sensors, fresh)
        if terminus.numel():
            mesh.connectNeurons(fresh, terminus)
        added += int(fresh.numel())

    if plan.terminus:
        fresh = mesh.allocNeuronIds(plan.terminus, NeuronKind.TERMINUS)
        existing = mesh.findNeuronIds(NeuronKind.TERMINUS)
        nexus = mesh.findNeuronIds(NeuronKind.NEXUS)
        motors = mesh.findNeuronIds(NeuronKind.MOTOR)
        mesh.connectNeurons(fresh, existing)
        mesh.connectNeurons(existing, fresh)
        if nexus.numel():
            mesh.connectNeurons(nexus, fresh)
        if motors.numel():
            mesh.connectNeurons(fresh, motors)
        added += int(fresh.numel())

    mesh.ensureNoOrphans()
    return added
