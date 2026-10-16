"""Packetized replay of a recorded run through the shared features, model, and alerts.

The engine only sees packets that have already "arrived". A decision at time t
uses waveform samples [t - window_s, t) and telemetry strictly before t, the
same definitions as offline extraction; future samples in the current packet
are excluded by absolute sample index. Cam phase pulses (contract 1.1) arrive
with the packet that contains them and are buffered like the waveforms.

With a cylinder localizer, each decision also carries `localization`: for
every active alert, the affected cylinder (or a relative slot without a cam
reference) accumulated by runtime.localization.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from engine_diagnostics.data.run_io import RunBundle
from engine_diagnostics.pipeline.cylinders import window_pulses
from engine_diagnostics.pipeline.features import (
    MODALITIES,
    all_feature_names,
    extract_window,
    telemetry_context,
    window_ends,
)
from engine_diagnostics.runtime.alerts import AlertState, MultiAlertState
from engine_diagnostics.runtime.localization import LocalizationState

TELEMETRY_HISTORY_S = 5.0


@dataclass
class Packet:
    t_start_s: float
    t_end_s: float
    waves: dict[str, np.ndarray]
    telemetry: pd.DataFrame
    cam_sync_s: np.ndarray | None = None  # pulses in [t_start, t_end); None: no stream


def packets(bundle: RunBundle, packet_s: float) -> Iterator[Packet]:
    """Split a run into consecutive packets on the master clock (t0 assumed 0)."""
    duration = bundle.metadata["duration_s"]
    count = int(np.ceil(duration / packet_s - 1e-9))
    for k in range(count):
        start, end = k * packet_s, min((k + 1) * packet_s, duration)
        waves = {}
        for m in MODALITIES:
            fs = getattr(bundle, f"{m}_fs_hz")
            waves[m] = getattr(bundle, m)[round(start * fs):round(end * fs)]
        rows = bundle.telemetry[(bundle.telemetry["t_s"] >= start)
                                & (bundle.telemetry["t_s"] < end)]
        yield Packet(start, end, waves, rows, window_pulses(bundle.cam_sync_s, start, end))


class ReplayEngine:
    def __init__(self, model, config: dict[str, Any], metadata: dict[str, Any],
                 localizer=None):
        if any(metadata["streams"][m]["t0_s"] != 0 for m in MODALITIES):
            raise ValueError("replay currently requires t0_s = 0 for every stream")
        self.model = model
        self.has_cam = "cam_sync" in metadata["streams"]
        self.localization = (LocalizationState(localizer, config["engine"]["firing_order"])
                             if localizer is not None else None)
        # The window vector holds every extracted feature; each model uses its own names.
        extracted = all_feature_names()
        self.columns = [extracted.index(n) for n in model.feature_names]
        self.config = config
        self.rates = {m: metadata["streams"][m]["fs_hz"] for m in MODALITIES}
        self.scales = {m: metadata["streams"][m]["full_scale"] for m in MODALITIES}
        self.decision_times = window_ends(metadata["duration_s"], config)
        # A model may carry alert persistence tuned on its tuning partition. A
        # multi-fault model gets one independent alert per fault.
        self.multi = hasattr(model, "fault_classes")
        clear = getattr(model, "clear_updates", config["runtime"]["clear_fault_updates"])
        if self.multi:
            self.alerts = MultiAlertState(tuple(model.fault_classes), model.confirm_updates,
                                          clear)
        else:
            self.alerts = AlertState(
                confirm=getattr(model, "confirm_updates",
                                config["runtime"]["confirm_fault_updates"]),
                clear=clear)
        self.reset()

    def reset(self) -> None:
        """Clear buffers and alert state (used on restart or seek)."""
        self.buffers = {m: np.zeros(0, dtype=np.float32) for m in MODALITIES}
        self.buffer_start = dict.fromkeys(MODALITIES, 0)  # absolute index of buffer[0]
        self.received = dict.fromkeys(MODALITIES, 0)  # samples received so far
        self.telemetry = pd.DataFrame(columns=["t_s", "signal", "value", "unit", "valid"])
        self.cam_sync = np.zeros(0)
        self.next_decision = 0
        self.alerts.reset()
        if self.localization is not None:
            self.localization.reset()

    def push(self, packet: Packet) -> list[dict[str, Any]]:
        """Ingest one packet; return decisions whose windows are now complete."""
        window_s = self.config["analysis"]["window_s"]
        for m in MODALITIES:
            self.buffers[m] = np.concatenate([self.buffers[m], packet.waves[m]])
            self.received[m] += packet.waves[m].size
            keep = round((window_s + self.config["analysis"]["step_s"]) * self.rates[m])
            drop = max(0, self.buffers[m].size - keep)
            self.buffers[m] = self.buffers[m][drop:]
            self.buffer_start[m] += drop
        if packet.cam_sync_s is not None and packet.cam_sync_s.size:
            keep_after = packet.t_end_s - window_s - self.config["analysis"]["step_s"]
            self.cam_sync = np.concatenate([self.cam_sync[self.cam_sync >= keep_after],
                                            packet.cam_sync_s])
        if not packet.telemetry.empty:
            frames = [f for f in (self.telemetry, packet.telemetry) if not f.empty]
            self.telemetry = pd.concat(frames, ignore_index=True)
            recent = self.telemetry["t_s"] >= packet.t_end_s - TELEMETRY_HISTORY_S
            self.telemetry = self.telemetry[recent].reset_index(drop=True)

        decisions = []
        while (self.next_decision < len(self.decision_times)
               and self._complete(self.decision_times[self.next_decision])):
            decisions.append(self._decide(self.decision_times[self.next_decision]))
            self.next_decision += 1
        return decisions

    def _complete(self, t: float) -> bool:
        return all(self.received[m] >= round(t * self.rates[m]) for m in MODALITIES)

    def _decide(self, t: float) -> dict[str, Any]:
        started = time.perf_counter()
        window_s = self.config["analysis"]["window_s"]
        waves = {}
        for m in MODALITIES:
            first = round((t - window_s) * self.rates[m]) - self.buffer_start[m]
            last = round(t * self.rates[m]) - self.buffer_start[m]
            waves[m] = self.buffers[m][max(first, 0):last]
        context, fresh = telemetry_context(self.telemetry, t, self.config)
        fs = self.rates["vibration"]
        window = extract_window(
            waves, self.rates, context, fresh, self.scales, t, self.config,
            cam_sync_s=window_pulses(self.cam_sync if self.has_cam else None, t - window_s, t),
            vibration_start_s=round((t - window_s) * fs) / fs)
        features = window.features[self.columns]
        if window.quality == "ok":
            result = self.model.predict(features)[0]
        else:
            result = {"predicted": None, "margin": float("nan"),
                      "anomaly_score": float("nan"), "anomalous": False,
                      "accepted": "uncertain"}
        if self.multi:  # active alerts as a list; `alert` joins them with '+'
            decisions = result.get("fault_decisions", {})
            alerts = self.alerts.update(decisions, window.quality)
            extra = {"alerts": alerts, "alert": "+".join(alerts) or None}
        else:
            accepted = result["accepted"]
            alert = self.alerts.update(accepted, window.quality)
            # One accepted class per window: that fault is "fault", the rest "normal".
            decisions = {f: "fault" if accepted == f
                         else "uncertain" if accepted == "uncertain" else "normal"
                         for f in (self.localization.localizer.faults
                                   if self.localization is not None else ())}
            alerts = [alert] if alert else []
            extra = {"alert": alert}
        if self.localization is not None:
            extra["localization"] = self.localization.update(decisions, alerts,
                                                             window.cylinders, window.quality)
        return {
            "t_end_s": float(t), "quality": window.quality, **result, **extra,
            "processing_ms": 1000 * (time.perf_counter() - started),
            "model_version": self.model.model_version,
            "feature_version": self.model.feature_version,
            "features": features, "context": context, "cylinders": window.cylinders,
        }


def replay_run(bundle: RunBundle, model, config: dict[str, Any], speed: float | None = None,
               on_decision=None, localizer=None) -> list[dict[str, Any]]:
    """Replay a whole run; `speed` paces packets in real time (1.0 = 1x), None = fast."""
    engine = ReplayEngine(model, config, bundle.metadata, localizer=localizer)
    packet_s = config["runtime"]["packet_s"]
    records = []
    clock = time.perf_counter()
    for packet in packets(bundle, packet_s):
        if speed:
            delay = clock + packet.t_end_s / speed - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        for decision in engine.push(packet):
            records.append(decision)
            if on_decision:
                on_decision(decision)
    return records


PREDICTION_LOG_COLUMNS = [
    "t_end_s", "quality", "predicted", "margin", "anomaly_score", "anomalous", "accepted",
    "alert", "processing_ms", "model_version", "feature_version",
]


def localization_summary(record: dict[str, Any]) -> str | None:
    """Compact per-alert location, e.g. 'misfire:cyl3 knock_like:locating'."""
    found = record.get("localization")
    if not found:
        return None
    parts = []
    for fault, where in sorted(found.items()):
        label = (f"cyl{where['cylinder']}" if where["cylinder"] is not None
                 else f"slot{where['slot']}" if where["slot"] is not None else where["status"])
        parts.append(f"{fault}:{label}")
    return " ".join(parts)


def prediction_log(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {k: r[k] for k in PREDICTION_LOG_COLUMNS}
        if "localization" in r:
            row["localization"] = localization_summary(r)
        rows.append(row)
    return pd.DataFrame(rows)
