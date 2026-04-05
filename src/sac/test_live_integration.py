"""
Integration tests: full live path
    CdaLiveConnector → HighwayShadowBridge → LaneChangingEnv

Verifies the complete pipeline using a mock DLL (no real Aimsun required):
  1. reset() returns valid obs and info with expected keys.
  2. step() produces a finite float reward.
  3. All expected reward-term keys appear in info["reward_terms"].
  4. info["cda"] is passed through from the connector snapshot.
  5. Episode state is fully clean after repeated reset().
  6. Max-step truncation fires when elapsed_steps reaches duration * policy_frequency.
  7. connector-reported truncated=True propagates to the env return value.
  8. Ego lane and speed reported by the DLL are reflected in the shadow state.

No real DLL is required — ctypes.CDLL is replaced by MagicMock.

Run from the repo root:
    python -m pytest src/sac/test_live_integration.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Make src/sac/ importable regardless of working directory.
sys.path.insert(0, os.path.dirname(__file__))

from cda_live.cda_live_connector import CdaLiveConnector  # noqa: E402
from lanechange_env import LaneChangingEnv               # noqa: E402


# ── Scenario constants ────────────────────────────────────────────────────────

_LANES         = 4
_EGO_LANE_WIRE = 2        # 1-based DLL convention
_EGO_LANE_IDX  = 1        # 0-based Python convention  (_EGO_LANE_WIRE - 1)
_EGO_SPEED_MPS = 20.0
_EGO_POS_M     = 150.0

_EXPECTED_REWARD_TERMS = frozenset({
    "collision", "jerk", "speed", "dist",
    "speed_limit", "lane_success", "lane_changing",
})


# ── DLL mock factory ──────────────────────────────────────────────────────────

def _make_dll() -> MagicMock:
    """
    Return a MagicMock configured to behave like the CDA gateway DLL.

    CDA_GetLatestVirVehTime returns a constant 1.0.  The reader thread fires
    once on startup, then once more after each reset_episode() (which resets
    _last_vir_time → 0.0, making 1.0 look like a fresh frame again).
    """
    dll = MagicMock()

    # lifecycle — all void
    for name in ("CDA_Init", "CDA_Close", "CDA_SendMessage",
                 "CDA_SetTestRouteID", "CDA_SetDebugMode"):
        getattr(dll, name).return_value = None

    # freshness ticker + ego kinematics
    dll.CDA_GetLatestVirVehTime.return_value = 1.0
    dll.CDA_GetVirVehSpeed.return_value      = _EGO_SPEED_MPS
    dll.CDA_GetVirVehPos.return_value        = _EGO_POS_M
    dll.CDA_GetElapsedSec.return_value       = 1.0
    dll.CDA_GetCurVehLane.return_value       = _EGO_LANE_WIRE

    # integer CDA fields
    dll.CDA_GetSignalState.return_value  = 0
    dll.CDA_GetTotalLane.return_value    = _LANES
    dll.CDA_GetLeftLCDir.return_value    = 2    # through lane → left adjacent exists
    dll.CDA_GetRightLCDir.return_value   = 2    # through lane → right adjacent exists
    dll.CDA_GetTurnDir.return_value      = 0
    dll.CDA_GetLaneSpec.return_value     = 0

    # float CDA fields (signal / trajectory planning — unused by reward, set to 0)
    for name in (
        "CDA_GetSigEndTime",
        "CDA_GetGreenStart1", "CDA_GetGreenEnd1",
        "CDA_GetGreenStart2", "CDA_GetGreenEnd2",
        "CDA_GetRedStart1",   "CDA_GetRedStart2",
        "CDA_GetRefAcc",      "CDA_GetTPEndLoc",
        "CDA_GetLatestTPTime", "CDA_GetDist2Turn",
    ):
        getattr(dll, name).return_value = 0.0

    # surrounding vehicles — plausible non-zero gaps so neighbors are synthesized
    dll.CDA_GetLeadSpeed.return_value      = 22.0
    dll.CDA_GetLeadGap.return_value        = 30.0
    dll.CDA_GetLeftLeadSpeed.return_value  = 21.0
    dll.CDA_GetLeftLeadGap.return_value    = 25.0
    dll.CDA_GetLeftLagSpeed.return_value   = 19.0
    dll.CDA_GetLeftLagGap.return_value     = 20.0
    dll.CDA_GetRightLeadSpeed.return_value = 23.0
    dll.CDA_GetRightLeadGap.return_value   = 35.0
    dll.CDA_GetRightLagSpeed.return_value  = 18.0
    dll.CDA_GetRightLagGap.return_value    = 15.0

    return dll


# ── Object factories ──────────────────────────────────────────────────────────

def _make_connector(dll: MagicMock) -> CdaLiveConnector:
    config = {
        "dll_path":        "fake.dll",   # never opened; CDLL is patched
        "remote_ip":       "127.0.0.1",
        "remote_port":     5555,
        "local_port":      5556,
        "ego_id":          1,
        "timeout_s":       2.0,
        "poll_interval_s": 0.0,          # no sleep — reader fires ASAP
    }
    with patch("ctypes.CDLL", return_value=dll):
        return CdaLiveConnector(config)


def _make_env(connector: CdaLiveConnector) -> LaneChangingEnv:
    return LaneChangingEnv(config={
        "backend":          "aimsun_live",
        "live_connector":   connector,
        "lanes_count":      _LANES,
        "lane_width":       4.0,
        "road_length":      1000.0,
        "policy_frequency": 15,
        "duration":         40,          # 40 s × 15 Hz = 600 max steps
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
    Each test gets a fresh DLL mock, connector, and env so state never leaks
    across test cases.  tearDown stops the connector's reader thread cleanly.
    """

    def setUp(self):
        self.dll       = _make_dll()
        self.connector = _make_connector(self.dll)
        self.env       = _make_env(self.connector)

    def tearDown(self):
        self.connector.close()

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
        """info must carry the full normalized snapshot dict with an 'ego' sub-dict."""
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

    def test_reset_ego_lane_matches_dll(self):
        """
        vehicle.lane_index[2] must equal the 0-based lane derived from
        CDA_GetCurVehLane (_EGO_LANE_WIRE → _EGO_LANE_IDX).
        """
        self.env.reset()
        _, _, lane_id = self.env.vehicle.lane_index
        self.assertEqual(lane_id, _EGO_LANE_IDX)

    def test_reset_ego_speed_matches_dll(self):
        """vehicle.speed must reflect CDA_GetVirVehSpeed after reset."""
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

        We drive elapsed_steps to (max_steps - 1) manually, then call step() once
        more to trigger the threshold.
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
        self.env.steps_in_target_lane = 15   # simulate partial lane-change progress
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
