"""
live_connector_base.py — Abstract base class for all live simulator connectors.

Subclass LiveConnector to implement a specific backend (CDA, Aimsun HIL, mock, etc.).
Import from here, not from live_bridge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LiveConnector(ABC):
    """
    Abstract base class for all live simulator connectors.

    Snapshot dicts returned by reset_episode() and step() must conform to the
    structured 'ego' format expected by HighwayShadowBridge.sync_shadow_state():

        {
            "ego": {
                "lane_idx":  int,    # 0-based
                "pos_m":     float,
                "speed_mps": float,
                "acc_mps2":  float,  # measured, not commanded
                "crashed":   bool,
            },
            "time_s":     float,
            "terminated": bool,
            "truncated":  bool,
            "neighbors":  [...],     # or "surrounding": {...}
            "cda":        {...},     # optional passthrough
        }
    """

    @abstractmethod
    def reset_episode(self, seed: int | None = None) -> dict[str, Any]:
        """Start a new episode. Return the initial snapshot."""

    @abstractmethod
    def step(self, command: dict[str, Any]) -> dict[str, Any]:
        """
        Apply command to the external runtime and return the resulting snapshot.

        command keys (from HighwayShadowBridge.decode_action_command):
            accel_mps2       float   — longitudinal acceleration (m/s²)
            desired_lane_id  int     — target lane (0-based)
            target_lane_index tuple  — ("0", "1", desired_lane_id)
            dt               float   — timestep (seconds)
            accel_norm       float   — raw normalised accel in [-1, 1]
            intent           float   — raw normalised intent in [-1, 1]
        """

    def close(self) -> None:
        """Release resources. Override if cleanup is needed."""
