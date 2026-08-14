"""Healing: lifting a mute neuron back to conducting without disturbing what it learned."""

import pytest
import torch

from haze import makeHaze
from haze.functions.heal import (
    applyConductanceHealing,
    calcHealingFactors,
    calcOutgoingPeak,
)
from haze.models import HazeHyper
from haze.modules.decoders import ArgMax
from haze.modules.encoders import NumericEncoder


def makeMesh(**hyper):
    """Returns a small wired mesh with healing available."""
    model = makeHaze(nexus_size=32, terminus_size=16, seed=1, growth_threshold=0.99, **hyper)
    model.registerEncoder("bits", NumericEncoder())
    model.registerDecoder("bit", ArgMax(labels=[0, 1]))
    return model


def findOutgoing(model, neuron):
    """Returns the live edge indices leaving a neuron."""
    live = model.mesh.counts.edges
    which = (model.mesh.src[:live] == neuron) & model.mesh.alive_e[:live]
    return torch.nonzero(which).flatten()


def muteNeuron(model, neuron, level=0.15):
    """Drives every outgoing edge of a neuron below the conduction floor."""
    edges = findOutgoing(model, neuron)
    model.mesh.strength[edges] = level
    return edges


class TestOutgoingPeakDomain:
    """Domain: the peak reads the strongest live edge leaving each neuron."""

    def testPeakIsTheStrongestOutgoingEdge(self):
        """A neuron's reading equals the largest magnitude among its live out-edges."""
        model = makeMesh()
        neuron = int(model.mesh.src[0])
        edges = findOutgoing(model, neuron)
        expected = float(model.mesh.strength[edges].abs().max())
        assert float(calcOutgoingPeak(model.mesh)[neuron]) == pytest.approx(expected)

    def testAMutedNeuronReadsBelowTheFloor(self):
        """Driving every out-edge down puts the neuron's peak under the conduction floor."""
        model = makeMesh()
        neuron = int(model.mesh.src[0])
        muteNeuron(model, neuron)
        peak = float(calcOutgoingPeak(model.mesh)[neuron])
        assert peak < model.config.hyper.calcConductionFloor()

    def testAMotorReadsZero(self):
        """A neuron with no outgoing edges has no peak to report."""
        model = makeMesh()
        motor = int(model.ports.findActiveMotorIds("bit")[0])
        assert float(calcOutgoingPeak(model.mesh)[motor]) == 0.0


class TestHealingFactorsDomain:
    """Domain: only a neuron that cannot conduct is given a factor above one."""

    def testAConductingNeuronIsLeftAlone(self):
        """A peak already above the floor takes a factor of exactly one."""
        hyper = HazeHyper()
        peak = torch.tensor([hyper.strength_upper])
        assert float(calcHealingFactors(peak, hyper)) == 1.0

    def testAMuteNeuronIsLiftedPastTheFloor(self):
        """The factor takes a mute peak to the floor plus its margin, and no further."""
        hyper = HazeHyper()
        peak = torch.tensor([0.15])
        lifted = 0.15 * float(calcHealingFactors(peak, hyper))
        assert lifted == pytest.approx(
            hyper.calcConductionFloor() * hyper.heal_margin, rel=1e-5
        )

    def testAFullyDeadNeuronIsNotScaled(self):
        """Scaling nothing is still nothing, so a zero peak takes a factor of one."""
        assert float(calcHealingFactors(torch.zeros(1), HazeHyper())) == 1.0


class TestHealingDomain:
    """Domain: healing restores conduction and preserves the learned ordering."""

    def testHealingRestoresConduction(self):
        """A muted neuron conducts again after healing."""
        model = makeMesh()
        neuron = int(model.mesh.src[0])
        muteNeuron(model, neuron)
        applyConductanceHealing(model.mesh, model.config.hyper)
        peak = float(calcOutgoingPeak(model.mesh)[neuron])
        assert peak >= model.config.hyper.calcConductionFloor()

    def testHealingPreservesRelativeOrder(self):
        """Every edge keeps its rank among its siblings, which is what learning decided."""
        model = makeMesh()
        neuron = int(model.mesh.src[0])
        edges = findOutgoing(model, neuron)
        model.mesh.strength[edges] = torch.linspace(0.05, 0.2, edges.numel())
        before = torch.argsort(model.mesh.strength[edges])
        applyConductanceHealing(model.mesh, model.config.hyper)
        assert torch.equal(torch.argsort(model.mesh.strength[edges]), before)

    def testHealingReportsEachNeuronOnce(self):
        """The count is neurons lifted, not edges, so muting one adds exactly one to it."""
        model = makeMesh()
        healable = int((calcHealingFactors(
            calcOutgoingPeak(model.mesh), model.config.hyper
        ) > 1.0).sum())
        target = next(
            int(n) for n in model.mesh.src[: model.mesh.counts.edges].unique()
            if float(calcOutgoingPeak(model.mesh)[int(n)])
            >= model.config.hyper.calcConductionFloor()
        )
        edges = muteNeuron(model, target)
        assert edges.numel() > 1
        assert applyConductanceHealing(model.mesh, model.config.hyper) == healable + 1

    def testAConductingMeshIsUntouched(self):
        """Healing is silent where nothing is wrong.

        Built by lifting every edge above the floor first, because a *fresh* mesh is not such a
        mesh: initial strengths are drawn from a range whose bottom quarter sits under the
        conduction floor, so a neuron whose only out-edge falls there is born unable to pass
        signal. See specs/propagation.md.
        """
        model = makeMesh()
        live = model.mesh.counts.edges
        model.mesh.strength[:live] = model.config.hyper.strength_upper
        before = model.mesh.strength[:live].clone()
        assert applyConductanceHealing(model.mesh, model.config.hyper) == 0
        assert torch.equal(model.mesh.strength[:live], before)

    def testHealingKeepsTheLogStrengthConsistent(self):
        """The propagation path reads log strength, so it must move with the strength."""
        model = makeMesh()
        muteNeuron(model, int(model.mesh.src[0]))
        applyConductanceHealing(model.mesh, model.config.hyper)
        live = model.mesh.counts.edges
        expected = model.mesh.strength[:live].abs().clamp_min(1e-6).log()
        assert torch.allclose(model.mesh.log_str[:live], expected, atol=1e-5)


class TestHealingBoundary:
    """Boundary: the rails, an empty mesh, and a sign that must not flip."""

    def testHealingNeverExceedsTheUpperRail(self):
        """A lifted edge is still a legal strength."""
        model = makeMesh()
        muteNeuron(model, int(model.mesh.src[0]), level=0.02)
        applyConductanceHealing(model.mesh, model.config.hyper)
        live = model.mesh.counts.edges
        assert float(model.mesh.strength[:live].abs().max()) <= model.config.hyper.strength_upper

    def testHealingPreservesInhibition(self):
        """An inhibitory edge is lifted in magnitude and stays inhibitory."""
        model = makeMesh()
        neuron = int(model.mesh.src[0])
        edges = findOutgoing(model, neuron)
        model.mesh.strength[edges] = -0.15
        applyConductanceHealing(model.mesh, model.config.hyper)
        assert bool((model.mesh.strength[edges] < 0).all())

    def testAnEmptyMeshHealsNothing(self):
        """A mesh with no edges returns zero rather than raising."""
        model = makeMesh()
        model.mesh.counts.edges = 0
        assert applyConductanceHealing(model.mesh, model.config.hyper) == 0


class TestHealingIntegration:
    """Domain: the flag routes healing into the ordinary learn path and reports it."""

    def testDisabledByDefault(self):
        """A default configuration heals nothing, so today's behaviour is unchanged."""
        assert HazeHyper().conductance_healing is False

    def testLearnReportsWhatItHealed(self):
        """A learn step over a muted mesh reports the neurons it lifted."""
        model = makeMesh(conductance_healing=True)
        model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
        muteNeuron(model, int(model.mesh.src[0]))
        assert model.learn(1.0).healed >= 1

    def testLearnHealsNothingWhenDisabled(self):
        """With the flag off the count stays zero even on a mute mesh."""
        model = makeMesh()
        model({"bits": [1, 0, 1, 0, 1, 0, 1, 0]})
        muteNeuron(model, int(model.mesh.src[0]))
        assert model.learn(1.0).healed == 0
