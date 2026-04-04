"""
Unit tests for HighwayShadowBridge.

Focus areas:
  1. _synthesize_surrounding_neighbors — smart lane-gating logic using CDA
     totalLane, leftLCDir, rightLCDir.
  2. _make_shadow_vehicle — total_lanes kwarg overrides env config.
  3. decode_action_command — accel scaling, deadzone, dt.

_make_shadow_vehicle is patched in neighbor tests so highway-env Vehicle
construction is never exercised.  Vehicle is only patched in its own tests.

Run from the repo root:
    python -m pytest src/sac/shadow_reader/test_live_bridge.py -v
"""

from __future__ import annotations

import math
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shadow_reader.live_bridge import HighwayShadowBridge  # noqa: E402


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _make_env(lanes_count: int = 4) -> MagicMock:
    env = MagicMock()
    _cfg = {"lanes_count": lanes_count, "policy_frequency": 15}
    env.config.get = lambda k, d=None: _cfg.get(k, d)
    env.last_action_command = None
    return env


def _make_bridge(lanes_count: int = 4) -> HighwayShadowBridge:
    return HighwayShadowBridge(_make_env(lanes_count))


def _snap(
    *,
    ego_lane: int = 1,
    ego_pos: float = 500.0,
    cda: dict | None = None,
    surrounding: dict | None = None,
) -> dict:
    return {
        "ego": {
            "lane_idx": ego_lane,
            "pos_m":    ego_pos,
            "speed_mps": 15.0,
            "acc_mps2":  0.0,
            "crashed":   False,
        },
        "cda":         cda or {},
        "surrounding": surrounding or {},
        "terminated":  False,
        "truncated":   False,
        "time_s":      1.0,
    }


def _nbr(speed: float = 20.0, gap: float = 15.0) -> dict:
    return {"speed_mps": speed, "gap_m": gap}


def _all_neighbors(speed: float = 20.0, gap: float = 15.0) -> dict:
    """Return a surrounding dict with all five positions populated."""
    return {k: _nbr(speed, gap) for k in ("front", "llead", "llag", "rlead", "rlag")}


def _cda(
    total_lanes: int = 4,
    left_lc_dir: int = 2,
    right_lc_dir: int = 2,
) -> dict:
    """Minimal CDA dict for neighbor tests."""
    return {
        "totalLane":  total_lanes,
        "leftLCDir":  left_lc_dir,
        "rightLCDir": right_lc_dir,
    }


def _run_synthesize(bridge: HighwayShadowBridge, normalized: dict):
    """
    Call _synthesize_surrounding_neighbors with _make_shadow_vehicle patched to
    a sentinel.  Returns (neighbor_list, mock_make_shadow_vehicle).
    """
    sentinel = MagicMock()
    with patch.object(bridge, "_make_shadow_vehicle", return_value=sentinel) as mock_make:
        neighbors = bridge._synthesize_surrounding_neighbors(normalized)
    return neighbors, mock_make


def _called_lane_idxs(mock_make) -> list[int]:
    """Extract the lane_idx positional arg from every _make_shadow_vehicle call."""
    return [c.args[0] for c in mock_make.call_args_list]


# ── TestSynthesizeNeighbors ──────────────────────────────────────────────────────

class TestSynthesizeNeighbors(unittest.TestCase):
    """
    _synthesize_surrounding_neighbors must use CDA data to decide which
    neighbor slots are valid before inspecting gap / speed values.
    """

    # ── Front ──────────────────────────────────────────────────────────────────

    def test_front_placed_regardless_of_lc_dir(self):
        """
        Front (same-lane lead) is not gated by any lc_dir.
        Even with left and right lc_dir = 0, a valid front gap must produce
        one neighbor.
        """
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=1,
            cda=_cda(left_lc_dir=0, right_lc_dir=0),
            surrounding={"front": _nbr()},
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        self.assertEqual(len(neighbors), 1)
        self.assertIn(1, _called_lane_idxs(mock_make))  # ego lane

    # ── Left lane gating ───────────────────────────────────────────────────────

    def test_left_skipped_when_lc_dir_zero(self):
        """leftLCDir=0 means no left lane — llead and llag must not be placed."""
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=2,
            cda=_cda(left_lc_dir=0, right_lc_dir=0),
            surrounding=_all_neighbors(),
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        lane_idxs = _called_lane_idxs(mock_make)
        self.assertNotIn(1, lane_idxs)  # left lane = ego_lane - 1

    def test_left_placed_when_lc_dir_nonzero(self):
        """leftLCDir=2 (through lane) and ego not at leftmost → llead and llag placed."""
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=2,
            cda=_cda(left_lc_dir=2, right_lc_dir=0),
            surrounding=_all_neighbors(),
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        lane_idxs = _called_lane_idxs(mock_make)
        self.assertEqual(lane_idxs.count(1), 2)  # llead + llag both in left lane

    def test_ego_at_leftmost_lane_no_left_neighbor(self):
        """
        Even if leftLCDir is non-zero, ego at lane_idx=0 must not produce a
        left neighbor (lane -1 does not exist).
        """
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=0,
            cda=_cda(left_lc_dir=2, right_lc_dir=0),
            surrounding=_all_neighbors(),
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        self.assertNotIn(-1, _called_lane_idxs(mock_make))

    # ── Right lane gating ──────────────────────────────────────────────────────

    def test_right_skipped_when_lc_dir_zero(self):
        """rightLCDir=0 means no right lane — rlead and rlag must not be placed."""
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=1,
            cda=_cda(left_lc_dir=0, right_lc_dir=0),
            surrounding=_all_neighbors(),
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        self.assertNotIn(2, _called_lane_idxs(mock_make))  # right lane = ego_lane + 1

    def test_right_placed_when_lc_dir_nonzero(self):
        """rightLCDir=2 and ego not at rightmost → rlead and rlag placed."""
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=1,
            cda=_cda(left_lc_dir=0, right_lc_dir=2),
            surrounding=_all_neighbors(),
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        lane_idxs = _called_lane_idxs(mock_make)
        self.assertEqual(lane_idxs.count(2), 2)  # rlead + rlag both in right lane

    def test_ego_at_rightmost_no_right_neighbor(self):
        """
        Even if rightLCDir is non-zero, ego already at the rightmost lane
        (lane_idx == total_lanes - 1) must not produce a right neighbor.
        """
        bridge = _make_bridge(lanes_count=4)
        snap = _snap(
            ego_lane=3,           # rightmost in a 4-lane road (0-based)
            cda=_cda(total_lanes=4, left_lc_dir=0, right_lc_dir=2),
            surrounding=_all_neighbors(),
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        self.assertNotIn(4, _called_lane_idxs(mock_make))

    # ── totalLane overrides env config ─────────────────────────────────────────

    def test_total_lanes_from_cda_overrides_env_config(self):
        """
        env config says 4 lanes; CDA reports totalLane=2.
        Ego at lane_idx=1 (rightmost in a 2-lane road) with rightLCDir=2 must
        still produce no right neighbor because lane_idx < total_lanes - 1 is
        1 < 1 = False.
        """
        bridge = _make_bridge(lanes_count=4)     # env thinks 4 lanes
        snap = _snap(
            ego_lane=1,
            cda=_cda(total_lanes=2, left_lc_dir=0, right_lc_dir=2),
            surrounding=_all_neighbors(),
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        self.assertNotIn(2, _called_lane_idxs(mock_make))

    def test_total_lanes_kwarg_forwarded_to_make_shadow_vehicle(self):
        """
        The resolved total_lanes value must be forwarded to _make_shadow_vehicle
        so its bounds check uses the same source of truth.
        """
        bridge = _make_bridge(lanes_count=4)
        snap = _snap(
            ego_lane=1,
            cda=_cda(total_lanes=3, left_lc_dir=2, right_lc_dir=2),
            surrounding=_all_neighbors(),
        )
        _, mock_make = _run_synthesize(bridge, snap)

        for c in mock_make.call_args_list:
            self.assertEqual(c.kwargs.get("total_lanes"), 3)

    # ── Gap / speed filtering ──────────────────────────────────────────────────

    def test_zero_gap_not_placed(self):
        """gap_m=0.0 must be rejected — a vehicle at zero gap is degenerate."""
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=1,
            cda=_cda(),
            surrounding={"front": _nbr(gap=0.0)},
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        mock_make.assert_not_called()

    def test_negative_gap_not_placed(self):
        """Negative gap (behind ego in same lane direction) must be rejected."""
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=1,
            cda=_cda(),
            surrounding={"front": _nbr(gap=-5.0)},
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        mock_make.assert_not_called()

    def test_nonfinite_gap_not_placed(self):
        """inf and nan gap values must be rejected."""
        bridge = _make_bridge()
        for bad_gap in (math.inf, math.nan):
            with self.subTest(gap=bad_gap):
                snap = _snap(
                    ego_lane=1,
                    cda=_cda(),
                    surrounding={"front": _nbr(gap=bad_gap)},
                )
                neighbors, mock_make = _run_synthesize(bridge, snap)
                mock_make.assert_not_called()

    def test_missing_surrounding_key_not_placed(self):
        """A position absent from surrounding must not attempt vehicle creation."""
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=1,
            cda=_cda(left_lc_dir=2, right_lc_dir=2),
            surrounding={},        # no keys at all
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        mock_make.assert_not_called()

    def test_no_rear_neighbor_placed(self):
        """
        CDA exposes no rear getter.  Even if a caller populates surrounding
        with a 'rear' key the bridge must not place a rear vehicle — rear is
        not in the spec list.
        """
        bridge = _make_bridge()
        snap = _snap(
            ego_lane=1,
            cda=_cda(left_lc_dir=0, right_lc_dir=0),
            surrounding={"rear": _nbr()},   # only rear populated
        )
        neighbors, mock_make = _run_synthesize(bridge, snap)

        mock_make.assert_not_called()


# ── TestMakeShadowVehicle ────────────────────────────────────────────────────────

class TestMakeShadowVehicle(unittest.TestCase):
    """
    _make_shadow_vehicle must use the total_lanes kwarg when provided,
    and fall back to env.config["lanes_count"] when it is not.
    """

    def _call(self, bridge, lane_idx, total_lanes=None):
        kwargs = {} if total_lanes is None else {"total_lanes": total_lanes}
        with patch("shadow_reader.live_bridge.Vehicle") as MockVehicle:
            MockVehicle.return_value = MagicMock()
            result = bridge._make_shadow_vehicle(lane_idx, 500.0, 15.0, **kwargs)
        return result

    def test_out_of_bounds_with_total_lanes_kwarg_returns_none(self):
        """lane_idx=2 with total_lanes=2 (valid range 0-1) must return None."""
        bridge = _make_bridge(lanes_count=4)   # env config says 4 — ignored
        result = self._call(bridge, lane_idx=2, total_lanes=2)
        self.assertIsNone(result)

    def test_valid_lane_with_total_lanes_kwarg_creates_vehicle(self):
        """lane_idx=1 with total_lanes=2 is within bounds — must create a vehicle."""
        bridge = _make_bridge(lanes_count=4)
        result = self._call(bridge, lane_idx=1, total_lanes=2)
        self.assertIsNotNone(result)

    def test_falls_back_to_env_config_when_no_kwarg(self):
        """Without the kwarg, env.config lanes_count=2 must gate lane_idx=2 → None."""
        bridge = _make_bridge(lanes_count=2)
        result = self._call(bridge, lane_idx=2)    # no total_lanes kwarg
        self.assertIsNone(result)


# ── TestDecodeActionCommand ──────────────────────────────────────────────────────

class TestDecodeActionCommand(unittest.TestCase):
    """Basic sanity checks on action → command dict translation."""

    def _make_bridge_with_vehicle(self, *, lane_id: int = 1, n_lanes: int = 4):
        env = _make_env(lanes_count=n_lanes)
        env.vehicle.lane_index = ("0", "1", lane_id)
        env.road.network.graph = {
            "0": {"1": {i: MagicMock() for i in range(n_lanes)}}
        }
        # vehicle.road must point to the same mock as env.road so that
        # decode_action_command's graph lookup resolves correctly.
        env.vehicle.road = env.road
        env.last_action_command = None
        return HighwayShadowBridge(env)

    def test_accel_scaling(self):
        """accel_norm=1.0 must map to accel_mps2=3.0 (max accel)."""
        bridge = self._make_bridge_with_vehicle()
        cmd = bridge.decode_action_command([1.0, 0.0])
        self.assertAlmostEqual(cmd["accel_mps2"], 3.0)

    def test_decel_scaling(self):
        """accel_norm=-1.0 must map to accel_mps2=-3.0."""
        bridge = self._make_bridge_with_vehicle()
        cmd = bridge.decode_action_command([-1.0, 0.0])
        self.assertAlmostEqual(cmd["accel_mps2"], -3.0)

    def test_keep_lane_within_deadzone(self):
        """
        |intent| < 0.2 → stay in current lane.
        desired_lane_id must equal the ego's current lane_id.
        """
        bridge = self._make_bridge_with_vehicle(lane_id=2)
        cmd = bridge.decode_action_command([0.0, 0.1])   # intent inside deadzone
        self.assertEqual(cmd["desired_lane_id"], 2)

    def test_dt_from_policy_frequency(self):
        """dt in the command must equal 1 / policy_frequency."""
        bridge = self._make_bridge_with_vehicle()
        cmd = bridge.decode_action_command([0.0, 0.0])
        self.assertAlmostEqual(cmd["dt"], 1.0 / 15.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
