"""
Integration tests: full live path
    AapiDirectConnector → HighwayShadowBridge → LaneChangingEnv

Verifies the complete pipeline using a local UDP loopback server (_FakeAimsun).
No real Aimsun or DLL required — a background thread responds to START2 packets
with plausible START1 replies so the connector can complete reset and step cycles.

Tests:
  1. reset() returns valid obs and info with expected keys.
  2. step() produces a finite float reward.
  3. All expected reward-term keys appear in info["reward_terms"].
  4. info["cda"] is passed through from the connector snapshot.
  5. Episode state is fully clean after repeated reset().
  6. Max-step truncation fires when elapsed_steps reaches duration * policy_frequency.
  7. connector-reported truncated=True propagates to the env return value.
  8. Ego lane reported by the server is reflected in the shadow state.

Run from the repo root:
    python -m pytest src/sac/test_live_integration.py -v
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from cda_live.aapi_connector import (   # noqa: E402
    AapiDirectConnector,
    _pack, _START1, _START2, _MSGEND,
)
from lanechange_env import LaneChangingEnv  # noqa: E402


# ── Scenario constants ────────────────────────────────────────────────────────

_LANES         = 4
_EGO_ID        = 1
_EGO_LANE_WIRE = 2        # 1-based wire convention (reported by FakeAimsun)
_EGO_LANE_IDX  = 1        # 0-based  (= _EGO_LANE_WIRE - 1)
_EGO_SPEED_MPS = 15.0     # initial_speed_mps in connector config
_EGO_POS_M     = 100.0    # initial_pos_m in connector config

_EXPECTED_REWARD_TERMS = frozenset({
    "collision", "jerk", "speed", "dist",
    "speed_limit", "lane_success", "lane_changing",
})


# ── Port helper ───────────────────────────────────────────────────────────────

def _free_port() -> int:
    """Return a free local UDP port. Socket is closed immediately after binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Fake Aimsun server ────────────────────────────────────────────────────────

def _make_start1(ego_id: int, sim_tick: int) -> bytes:
    """
    Build a plausible START1 VehMessage reply using the same _pack helper
    the connector uses for START2, then swap the START2 header for START1.

    Surrounding vehicle fields are non-zero so HighwayShadowBridge creates
    shadow neighbours (required for finite reward terms).
    """
    import time as _t
    t = _t.localtime()
    pkt = _pack({
        "simID":          0,
        "ID":             ego_id,
        "targetCAVID":    ego_id,
        "leaderID":       ego_id,
        "year":           t.tm_year,
        "month":          t.tm_mon,
        "day":            t.tm_mday,
        "hour":           t.tm_hour,
        "minute":         t.tm_min,
        "second":         t.tm_sec,
        "ms":             0,
        "simTime":        sim_tick,
        "speed":          int(_EGO_SPEED_MPS * 1000),
        "linkID":         1,
        "linkPos":        int(_EGO_POS_M * 1000),
        "laneID":         _EGO_LANE_WIRE,
        "currentLane":    _EGO_LANE_WIRE,   # 1-based → lane_idx = _EGO_LANE_IDX
        "totalLane":      _LANES,
        # surrounding vehicles — positive gap/speed so bridge synthesises neighbours
        "leftLeadSpeed":  2200,   # 22.0 m/s  (raw scale 0.01)
        "leftLeadGap":    250,    # 25.0 m    (raw scale 0.1)
        "leftLagSpeed":   1900,   # 19.0 m/s
        "leftLagGap":     200,    # 20.0 m
        "rightLeadSpeed": 2300,   # 23.0 m/s
        "rightLeadGap":   350,    # 35.0 m
        "rightLagSpeed":  1800,   # 18.0 m/s
        "rightLagGap":    150,    # 15.0 m
        "leadSpeed":      2200,   # 22.0 m/s  (same-lane lead)
        "leadGap":        300,    # 30.0 m
    })
    return _START1 + pkt[len(_START2):-len(_MSGEND)] + _MSGEND


class _FakeAimsun(threading.Thread):
    """
    Minimal loopback Aimsun server for offline integration testing.

    Listens on a free port for START2 packets (ego state from the connector),
    immediately replies with a plausible START1 (surrounding state).
    Each reply increments simTime so the connector reader thread sees a new
    frame on every send cycle, preventing stale-snapshot blocking.
    """

    def __init__(self, connector_local_port: int, ego_id: int = _EGO_ID):
        super().__init__(daemon=True, name="FakeAimsun")
        self._target_port = connector_local_port
        self._ego_id      = ego_id
        self._sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.settimeout(0.05)
        self.port: int    = self._sock.getsockname()[1]
        self._stopped     = False
        self._sim_tick    = 10    # start at t = 1.0 s (simTime in 0.1s units)

    def run(self) -> None:
        while not self._stopped:
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.startswith(_START2):
                reply = _make_start1(self._ego_id, self._sim_tick)
                try:
                    self._sock.sendto(reply, ("127.0.0.1", self._target_port))
                except OSError:
                    break
                self._sim_tick += 1

    def stop(self) -> None:
        self._stopped = True
        try:
            self._sock.close()
        except OSError:
            pass


# ── Object factories ──────────────────────────────────────────────────────────

def _make_connector_and_server(
    ego_id: int = _EGO_ID,
) -> tuple[AapiDirectConnector, _FakeAimsun]:
    local_port = _free_port()
    server = _FakeAimsun(connector_local_port=local_port, ego_id=ego_id)
    server.start()
    connector = AapiDirectConnector({
        "remote_ip":         "127.0.0.1",
        "remote_port":       server.port,
        "local_port":        local_port,
        "ego_id":            ego_id,
        "link_id":           1,
        "initial_pos_m":     _EGO_POS_M,
        "initial_speed_mps": _EGO_SPEED_MPS,
        "timeout_s":         5.0,
        "poll_interval_s":   0.001,
    })
    return connector, server


def _make_env(connector: AapiDirectConnector) -> LaneChangingEnv:
    return LaneChangingEnv(config={
        "backend":          "aimsun_live",
        "live_connector":   connector,
        "lanes_count":      _LANES,
        "lane_width":       4.0,
        "road_length":      1000.0,
        "policy_frequency": 15,
        "duration":         40,    # 40 s × 15 Hz = 600 max steps
    })


def _truncated_snapshot() -> dict:
    """Minimal structured snapshot with truncated=True for connector override tests."""
    return {
        "ego": {
            "lane_idx":  _EGO_LANE_IDX,
            "pos_m":     _EGO_POS_M,
            "speed_mps": _EGO_SPEED_MPS,
            "acc_mps2":  0.0,
            "crashed":   False,
        },
        "terminated": False,
        "truncated":  True,
        "time_s":     2.0,
        "cda":        {"totalLane": _LANES, "leftLCDir": 2, "rightLCDir": 2},
        "surrounding": {},
    }


# ── Test fixture ──────────────────────────────────────────────────────────────

class TestLiveIntegration(unittest.TestCase):
    """
    Each test gets a fresh fake server, connector, and env so state never leaks
    across test cases.  tearDown stops the reader thread and fake server cleanly.
    """

    def setUp(self):
        self.connector, self.server = _make_connector_and_server()
        self.env = _make_env(self.connector)

    def tearDown(self):
        self.connector.close()
        self.server.stop()

    # ── reset() ───────────────────────────────────────────────────────────────

    def test_reset_obs_is_ndarray(self):
        """obs returned by reset() must be a numpy array."""
        obs, _ = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)

    def test_reset_info_backend_key(self):
        """info['backend'] must be 'aimsun_live'."""
        _, info = self.env.reset()
        self.assertEqual(info["backend"], "aimsun_live")

    def test_reset_info_snapshot_key(self):
        """info must carry the full normalised snapshot dict with an 'ego' sub-dict."""
        _, info = self.env.reset()
        self.assertIn("snapshot", info)
        self.assertIn("ego", info["snapshot"])

    def test_reset_info_cda_passthrough(self):
        """CDA data from the connector must appear in info['cda']."""
        _, info = self.env.reset()
        self.assertIn("cda", info)
        self.assertEqual(info["cda"]["totalLane"], _LANES)

    def test_reset_episode_counters_zero(self):
        """elapsed_steps and steps_in_target_lane must be 0 after reset."""
        self.env.reset()
        self.assertEqual(self.env.elapsed_steps, 0)
        self.assertEqual(self.env.steps_in_target_lane, 0)

    def test_reset_lane_changing_false(self):
        """lane_changing flag must be False immediately after reset."""
        self.env.reset()
        self.assertFalse(self.env.lane_changing)

    def test_reset_ego_lane_matches_server(self):
        """
        vehicle.lane_index[2] must equal the 0-based lane derived from the
        server's currentLane field (_EGO_LANE_WIRE → _EGO_LANE_IDX).
        """
        self.env.reset()
        _, _, lane_id = self.env.vehicle.lane_index
        self.assertEqual(lane_id, _EGO_LANE_IDX)

    def test_reset_ego_speed_is_initial(self):
        """vehicle.speed must equal the connector's initial_speed_mps after reset."""
        self.env.reset()
        self.assertAlmostEqual(self.env.vehicle.speed, _EGO_SPEED_MPS)

    # ── step() ────────────────────────────────────────────────────────────────

    def test_step_reward_is_finite_float(self):
        """step() must return a finite float reward."""
        self.env.reset()
        _, reward, _, _, _ = self.env.step([0.0, 0.0])
        self.assertIsInstance(reward, float)
        self.assertTrue(np.isfinite(reward))

    def test_step_all_reward_terms_present(self):
        """
        info['reward_terms'] must contain exactly the expected term names
        (raw_jerk is extra bookkeeping; excluded from the set check).
        """
        self.env.reset()
        _, _, _, _, info = self.env.step([0.0, 0.0])
        self.assertIn("reward_terms", info)
        actual = set(info["reward_terms"]) - {"raw_jerk"}
        self.assertEqual(_EXPECTED_REWARD_TERMS, actual)

    def test_step_all_reward_terms_finite(self):
        """Every value in info['reward_terms'] must be a finite number."""
        self.env.reset()
        _, _, _, _, info = self.env.step([0.5, 0.3])
        for term, val in info["reward_terms"].items():
            with self.subTest(term=term):
                self.assertTrue(np.isfinite(val), f"{term} = {val}")

    def test_step_info_cda_passthrough(self):
        """CDA data must flow from the connector snapshot into step info['cda']."""
        self.env.reset()
        _, _, _, _, info = self.env.step([0.0, 0.0])
        self.assertIn("cda", info)
        self.assertIn("totalLane", info["cda"])

    def test_step_info_lane_change_state_keys(self):
        """info must contain lane_change_state with all three expected sub-keys."""
        self.env.reset()
        _, _, _, _, info = self.env.step([0.0, 0.0])
        lcs = info.get("lane_change_state", {})
        for key in ("lane_changing", "steps_in_target_lane", "target_lane_index"):
            with self.subTest(key=key):
                self.assertIn(key, lcs)

    def test_step_increments_elapsed_steps(self):
        """elapsed_steps must increment by 1 per call to step()."""
        self.env.reset()
        for expected in range(1, 4):
            self.env.step([0.0, 0.0])
            with self.subTest(expected=expected):
                self.assertEqual(self.env.elapsed_steps, expected)

    def test_step_obs_shape_stable(self):
        """obs shape must be the same across reset and multiple step() calls."""
        obs0, _ = self.env.reset()
        shape = obs0.shape
        for i in range(3):
            obs, _, _, _, _ = self.env.step([0.0, 0.0])
            with self.subTest(step=i):
                self.assertEqual(obs.shape, shape)

    # ── terminated / truncated ────────────────────────────────────────────────

    def test_max_step_truncation(self):
        """
        When elapsed_steps hits duration * policy_frequency, truncated must be True.
        """
        self.env.reset()
        max_steps = int(
            self.env.config["duration"] * self.env.config.get("policy_frequency", 15)
        )
        self.env.elapsed_steps = max_steps - 1
        _, _, _, truncated, _ = self.env.step([0.0, 0.0])
        self.assertTrue(truncated, f"expected truncated=True at elapsed_steps={max_steps}")

    def test_below_max_steps_not_truncated(self):
        """A step well below the max-step limit must not set truncated=True."""
        self.env.reset()
        _, _, _, truncated, _ = self.env.step([0.0, 0.0])
        self.assertFalse(truncated)

    def test_connector_truncated_propagates(self):
        """
        If the connector returns truncated=True in the snapshot, env.step()
        must propagate it regardless of elapsed_steps.
        """
        self.env.reset()
        snap = _truncated_snapshot()
        self.connector.step = lambda cmd: snap
        _, _, _, truncated, _ = self.env.step([0.0, 0.0])
        self.assertTrue(truncated)

    # ── repeated episode ──────────────────────────────────────────────────────

    def test_second_reset_clears_elapsed_steps(self):
        """elapsed_steps must be 0 after a second reset(), regardless of prior steps."""
        self.env.reset()
        for _ in range(5):
            self.env.step([0.0, 0.0])
        self.assertGreater(self.env.elapsed_steps, 0)
        self.env.reset()
        self.assertEqual(self.env.elapsed_steps, 0)

    def test_second_reset_clears_steps_in_target_lane(self):
        """steps_in_target_lane must be 0 after re-reset even if non-zero mid-episode."""
        self.env.reset()
        self.env.steps_in_target_lane = 15
        self.env.reset()
        self.assertEqual(self.env.steps_in_target_lane, 0)

    def test_second_reset_lane_changing_cleared(self):
        """lane_changing must be False after a second reset."""
        self.env.reset()
        self.env.lane_changing = True
        self.env.reset()
        self.assertFalse(self.env.lane_changing)

    def test_second_reset_obs_and_info_valid(self):
        """A second reset() must return a valid obs array and info dict."""
        self.env.reset()
        for _ in range(3):
            self.env.step([0.0, 0.0])
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(info["backend"], "aimsun_live")
        self.assertIn("cda", info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
