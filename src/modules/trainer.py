"""The trainer: the loop that observes, scores, learns, and restructures the mesh as it goes."""

from typing import Any, Callable, Iterable, Mapping

from ..errors import SignalDidNotReachMotorsError
from ..functions.grow import applyGrowth
from ..functions.prune import applyPrune
from ..models import TrainReport
from .auditor import Auditor

Scorer = Callable[[Any, Mapping[str, Any]], float]


class Trainer:
    """Drives a Haze model through observations, and is where recovery and growth actually run.

    This exists because driving the model by hand is a trap, and not a hypothetical one. A mesh
    can learn its own path to the motors below the edge gate, at which point `predict` raises
    rather than answering. The obvious handling — score it as a wrong answer and move on — leaves
    the mesh permanently unable to answer anything, because nothing has reopened the path. Doing
    exactly that in a measurement harness written against this library cost one seed in six its
    entire result: 0.47 where recovery gives 0.98. The recovery loop is not an optimization, and
    a caller should not have to know it exists.
    """

    def __init__(self, model, auditor: Auditor | None = None) -> None:
        """Wraps a model with an auditor and the step counters that pace restructuring."""
        self.model = model
        self.auditor = auditor or Auditor(model.config.hyper)
        self.since_audit = 0

    def flowRecoverSignal(self, inputs: Mapping[str, Any]) -> Any:
        """Observes and answers, reopening a path with reverse learning if `relearn_limit` allows.

        At the default limit of zero this is a plain observe-and-answer that raises when the mesh
        goes silent, and the caller skips the observation. That default is measured: reverse
        learning costs more than it saves. Raising the limit turns the recovery loop on for a
        mesh that genuinely cannot reach its motors. See specs/learning.md.

        The failure is raised from inside the loop rather than after it, so a limit of zero
        perturbs nothing at all. Pushing once and *then* giving up would apply the whole cost of
        recovery while collecting none of its benefit.
        """
        attempts = self.model.config.hyper.relearn_limit
        for attempt in range(attempts + 1):
            try:
                self.model.observe(inputs)
                return self.model.predict()
            except SignalDidNotReachMotorsError:
                if attempt == attempts or not self.model.learning_enabled:
                    raise
                self.model.learn(0.0, reverse=True)
        raise AssertionError("unreachable: the loop either returns or raises")

    def flowTrainStep(self, inputs: Mapping[str, Any], score: Scorer) -> float | None:
        """Runs one observation — answer, score, learn, audit — returning its reward, or None.

        None means the mesh could not answer at all. The observation is skipped rather than
        scored, because there is no answer to attach a reward to, but the auditor is still told:
        a failure to answer is exactly the sustained-error signal growth exists to respond to.
        """
        try:
            answer = self.flowRecoverSignal(inputs)
        except SignalDidNotReachMotorsError:
            self.auditor.onOutcome(0.0, 0.0)
            return None

        reward = float(score(answer, inputs))
        confidence = self.model.calcConfidenceAggregate()
        self.model.learn(reward)
        self.auditor.onOutcome(reward, confidence)
        return reward

    def flowRestructure(self, report: TrainReport) -> None:
        """Rebuilds the mesh only when the audit says what it has is not working.

        Both halves are gated on the same verdict, pruning included. Running pruning on the
        audit's cadence instead — on the reasoning that a dead edge is dead however well the mesh
        is doing — was measured and costs real accuracy: held-out copy fell from 0.95 to 0.85 and
        one seed from 0.95 to 0.46. The cause is not the removal but the repair. Pruning strands
        neurons, `ensureNoOrphans` rewires them at fresh random strengths, and doing that to a
        mesh that has already converged injects noise into a working solution.

        So restructuring is a response to failure and not routine maintenance, which is what the
        self-organizing claim actually says: the mesh rebuilds itself when what it has is not
        working, and is left alone when it is. See specs/growth.md.
        """
        if not self.auditor.isWindowFull():
            return

        plan = self.auditor.calcGrowthPlan(counts=self.model.mesh.counts)
        if not plan.triggered:
            return

        report.grew += applyGrowth(self.model.mesh, plan)
        report.pruned += applyPrune(self.model.mesh, self.model.config.hyper).edges_removed
        self.auditor.resetWindows()

    def flowTraining(
        self, samples: Iterable[Mapping[str, Any]], score: Scorer
    ) -> TrainReport:
        """Trains over a stream of observations and returns what the run did."""
        report = TrainReport()
        for inputs in samples:
            reward = self.flowTrainStep(inputs, score)
            if reward is None:
                report.lost += 1
            else:
                report.rewards.append(reward)
            report.steps += 1
            self.since_audit += 1
            if self.since_audit >= self.model.config.hyper.audit_window:
                self.since_audit = 0
                self.flowRestructure(report)
        return report
