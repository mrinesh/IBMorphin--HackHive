"""Per-fault cylinder localization over successive replay decisions.

For each fault the detector reports in a window, the localizer's position
probabilities for that window are folded into an exponential moving average
(EMA_ALPHA), so one ambiguous window neither names nor changes a cylinder.
A cylinder is reported once at least MIN_WINDOWS windows contributed and the
leading average reaches the localizer's min_confidence for that fault. The
evidence for a fault is cleared when its decision returns to normal with no
alert active, or when its alert clears.

Phase references (pipeline.cylinders):
  cam_sync   positions are firing-order positions, so the result names a
             cylinder (firing_order[position]).
  relative   no reference stream: SlotAligner maps each window's positions to
             run-stable slots A-D by matching stroke onsets shared with the
             previous window (decision windows overlap). The result names a
             slot, never a cylinder; the slot epoch restarts (and evidence is
             cleared) when strokes cannot be matched, e.g. after a data gap.
Windows without a signature (no strokes, or a cam stream that lost every pulse
in the window) hold the evidence unchanged.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine_diagnostics.pipeline.cylinders import PHASE_CAM, PHASE_RELATIVE, CylinderSignature

EMA_ALPHA = 0.35
MIN_WINDOWS = 2
MATCH_FRACTION = 0.25  # stroke onsets within this fraction of a firing period match
SLOT_NAMES = ("A", "B", "C", "D")
LOCALIZED, LOCATING, RELATIVE_ONLY = "localized", "locating", "slot_only"


class SlotAligner:
    """Run-stable slot numbering for relative (cam-less) signatures."""

    def __init__(self, positions: int = 4):
        self.positions = positions
        self.reset()

    def reset(self) -> None:
        self.times = np.empty(0)
        self.slots = np.empty(0, dtype=int)
        self.epoch = 0

    def align(self, signature: CylinderSignature) -> tuple[int, bool]:
        """Offset with run slot = (window position + offset) mod positions.

        Returns (offset, restarted); restarted is True when no stroke matched
        the previous window, so a new slot epoch began at offset 0.
        """
        times, positions = signature.stroke_times_s, signature.stroke_positions
        offset, restarted = 0, True
        if self.times.size and times.size > 1:
            period = float(np.median(np.diff(times)))
            nearest = np.abs(times[:, None] - self.times[None, :])
            j = nearest.argmin(axis=1)
            matched = nearest[np.arange(times.size), j] < MATCH_FRACTION * period
            if matched.any():
                votes = np.mod(self.slots[j[matched]] - positions[matched], self.positions)
                offset = int(np.bincount(votes, minlength=self.positions).argmax())
                restarted = False
        if restarted and self.times.size:
            self.epoch += 1
        self.times = times
        self.slots = np.mod(positions + offset, self.positions)
        return offset, restarted


@dataclass
class FaultEvidence:
    average: np.ndarray | None = None
    windows: int = 0
    window_probabilities: np.ndarray | None = None

    def clear(self) -> None:
        self.average, self.windows, self.window_probabilities = None, 0, None


@dataclass
class LocalizationState:
    """Cylinder evidence per fault for one replayed run."""

    localizer: Any
    firing_order: list[int]
    evidence: dict[str, FaultEvidence] = field(default_factory=dict)

    def __post_init__(self):
        if list(self.localizer.firing_order) != list(self.firing_order):
            raise ValueError(f"localizer firing order {self.localizer.firing_order} differs "
                             f"from the configuration {self.firing_order}")
        self.aligner = SlotAligner(len(self.firing_order))
        self.reset()

    def reset(self) -> None:
        self.evidence = {f: FaultEvidence() for f in self.localizer.faults}
        self.aligner.reset()
        self.phase: str | None = None

    def update(self, fault_decisions: dict[str, str], active_alerts: list[str],
               signature: CylinderSignature | None, quality: str) -> dict[str, dict[str, Any]]:
        """Fold one decision window in; return the localization of each active alert.

        fault_decisions maps fault -> "fault" / "uncertain" / "normal" for this
        window. Only "fault" windows with an ok quality add evidence.
        """
        usable = signature is not None and quality == "ok"
        offset = 0
        if usable:
            self.phase = signature.phase
            if signature.phase == PHASE_RELATIVE:
                offset, restarted = self.aligner.align(signature)
                if restarted:
                    for evidence in self.evidence.values():
                        evidence.clear()
        for fault, evidence in self.evidence.items():
            decision = fault_decisions.get(fault, "normal")
            if decision == "normal" and fault not in active_alerts and quality == "ok":
                evidence.clear()
                continue
            if decision != "fault" or not usable:
                continue
            p = self.localizer.position_probabilities(fault, signature.contrast)
            p = np.roll(p, offset)  # window position -> run slot (identity for cam_sync)
            evidence.window_probabilities = p
            evidence.average = p if evidence.average is None \
                else (1 - EMA_ALPHA) * evidence.average + EMA_ALPHA * p
            evidence.windows += 1
        return {fault: self.report(fault) for fault in active_alerts if fault in self.evidence}

    def report(self, fault: str) -> dict[str, Any]:
        evidence = self.evidence[fault]
        cam = self.phase == PHASE_CAM
        out: dict[str, Any] = {"status": LOCATING, "phase": self.phase, "cylinder": None,
                               "slot": None, "confidence": None, "windows": evidence.windows,
                               "probabilities": None}
        if evidence.average is None:
            return out
        average = evidence.average
        best = int(average.argmax())
        labels = self.firing_order if cam else SLOT_NAMES[:len(average)]
        out["probabilities"] = {str(k): round(float(v), 4)
                                for k, v in zip(labels, average, strict=True)}
        out["confidence"] = float(average[best])
        confident = (evidence.windows >= MIN_WINDOWS
                     and average[best] >= self.localizer.min_confidence[fault])
        if confident and cam:
            out.update(status=LOCALIZED, cylinder=int(self.firing_order[best]))
        elif confident:
            out.update(status=RELATIVE_ONLY, slot=SLOT_NAMES[best])
        return out
