"""The auditor: the rolling read of recent performance that decides when the mesh restructures."""

from collections import deque

from ..functions.grow import calcErrorRate, calcGrowthPlan, calcUncertaintyRate
from ..models import AuditResult, GrowthPlan, HazeHyper, MeshCounts


class Auditor:
    """Watches recent reward and confidence, and decides whether the mesh should grow.

    Kept as its own object rather than as fields on the model because it is the one piece of
    Haze that is pure policy: it reads two windows and returns a plan, touching no tensor and
    changing nothing. That is what makes an alternative policy a drop-in — it satisfies
    `IGrowthPolicy` — and what lets the decision be tested without building a mesh.

    The windows are bounded deques rather than growing lists. The prior engine accumulated every
    reward it had ever seen and averaged the tail, so the audit got steadily more expensive over
    a long run and the memory never came back.
    """

    def __init__(self, hyper: HazeHyper) -> None:
        """Allocates the two rolling windows at the configured audit width."""
        self.hyper = hyper
        self.rewards: deque[float] = deque(maxlen=hyper.audit_window)
        self.confidences: deque[float] = deque(maxlen=hyper.audit_window)

    def onOutcome(self, reward: float, confidence: float) -> None:
        """Records one scored observation."""
        self.rewards.append(float(reward))
        self.confidences.append(float(confidence))

    def calcAudit(self) -> AuditResult:
        """Returns the current read of recent performance, without deciding anything."""
        return AuditResult(
            error_rate=calcErrorRate(list(self.rewards)),
            confidence_rate=calcUncertaintyRate(list(self.confidences)),
            window_full=self.isWindowFull(),
        )

    def isWindowFull(self) -> bool:
        """Returns whether enough observations have been seen to judge anything."""
        return (
            len(self.rewards) >= self.hyper.audit_window
            and len(self.confidences) >= self.hyper.audit_window
        )

    def calcGrowthPlan(self, rewards=None, confidences=None, counts=None) -> GrowthPlan:
        """Returns a growth plan from the recorded windows, or from ones passed in.

        Accepts the windows as arguments so it satisfies `IGrowthPolicy`, whose callers may hold
        their own history, while defaulting to what it has recorded itself.
        """
        return calcGrowthPlan(
            list(self.rewards) if rewards is None else list(rewards),
            list(self.confidences) if confidences is None else list(confidences),
            counts if counts is not None else MeshCounts(),
            self.hyper,
        )

    def resetWindows(self) -> None:
        """Clears both windows, so a restructured mesh is judged on what it does next.

        Without this the observations that triggered growth stay in the window and trigger it
        again on the very next step, before the new neurons have carried a single signal.
        """
        self.rewards.clear()
        self.confidences.clear()
