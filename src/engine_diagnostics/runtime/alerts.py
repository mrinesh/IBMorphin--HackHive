"""Persistent alert state over accepted window decisions."""

from dataclasses import dataclass

NORMAL = "normal"
HOLD = {"uncertain"}  # decisions that neither confirm nor clear an alert


@dataclass
class AlertState:
    """Raise after `confirm` consecutive identical faults; clear after `clear` normals.

    Uncertain decisions and windows with a non-ok quality hold the current state
    and reset no counters, so a brief data problem neither raises nor clears.
    """

    confirm: int = 2
    clear: int = 3
    active: str | None = None
    candidate: str | None = None
    candidate_count: int = 0
    normal_count: int = 0

    def update(self, accepted: str, quality: str = "ok") -> str | None:
        if quality != "ok" or accepted in HOLD:
            return self.active
        if accepted == NORMAL:
            self.candidate, self.candidate_count = None, 0
            self.normal_count += 1
            if self.active and self.normal_count >= self.clear:
                self.active = None
            return self.active
        self.normal_count = 0
        if accepted == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate, self.candidate_count = accepted, 1
        if self.candidate_count >= self.confirm:
            self.active = accepted
        return self.active

    def reset(self) -> None:
        self.active, self.candidate = None, None
        self.candidate_count = self.normal_count = 0


@dataclass
class MultiAlertState:
    """One independent AlertState per fault, so faults raise and clear separately.

    Each fault is fed "fault", "normal", or "uncertain" for itself only; another
    fault's activity never confirms or clears it. Non-ok quality holds all.
    """

    faults: tuple[str, ...]
    confirm: dict[str, int]
    clear: int = 3

    def __post_init__(self):
        self.states = {f: AlertState(confirm=self.confirm[f], clear=self.clear)
                       for f in self.faults}

    def update(self, decisions: dict[str, str], quality: str = "ok") -> list[str]:
        for fault, state in self.states.items():
            decision = decisions.get(fault, "uncertain")
            state.update(fault if decision == "fault" else decision, quality)
        return self.active

    @property
    def active(self) -> list[str]:
        return sorted(f for f, s in self.states.items() if s.active)

    def reset(self) -> None:
        for state in self.states.values():
            state.reset()
