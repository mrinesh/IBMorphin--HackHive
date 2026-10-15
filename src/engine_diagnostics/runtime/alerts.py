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
