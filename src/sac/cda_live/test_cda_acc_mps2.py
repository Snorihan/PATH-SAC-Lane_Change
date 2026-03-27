"""
Unit tests: CdaLiveConnector.acc_mps2 computation.

These tests replace ctypes.CDLL with a MagicMock so no real DLL is needed.
They verify the invariant:

    acc_mps2 = (speed_curr - speed_prev) / dt

across reset_episode() and step() calls.

Run from the repo root:
    python -m pytest src/sac/cda_live/test_cda_acc_mps2.py -v
"""

from __future__ import annotations

import itertools
import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure src/sac/ is on the path so package imports resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cda_live.cda_live_connector import CdaLiveConnector  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────────

_NOMINAL_DT = 1.0 / 15.0  # 15 Hz — the standard policy frequency


def _make_connector(dll_mock: MagicMock) -> CdaLiveConnector:
    """Instantiate CdaLiveConnector with a fully-mocked DLL."""
    config = {
        "dll_path":        "fake.dll",  # never opened; ctypes.CDLL is patched
        "remote_ip":       "127.0.0.1",
        "remote_port":     5555,
        "local_port":      5556,
        "ego_id":          1,
        "timeout_s":       2.0,
        "poll_interval_s": 0.0,        # no sleep — tests run instantly
    }
    with patch("ctypes.CDLL", return_value=dll_mock):
        return CdaLiveConnector(config)


def _prime_dll(
    dll: MagicMock,
    *,
    vir_time: float,
    speed: float,
    pos: float = 100.0,
    elapsed: float = 1.0,
) -> None:
    """
    Configure the dll mock to return the given values on the next poll.

    All CDA signal / TP getters are set to 0.0 / 0 so _build_snapshot
    does not raise.  Override individual attributes after calling this
    if a specific value matters.
    """
    dll.CDA_GetLatestVirVehTime.return_value = vir_time
    dll.CDA_GetVirVehSpeed.return_value      = speed
    dll.CDA_GetVirVehPos.return_value        = pos
    dll.CDA_GetElapsedSec.return_value       = elapsed
    dll.CDA_GetSignalState.return_value      = 0
    for name in (
        "CDA_GetSigEndTime",
        "CDA_GetGreenStart1", "CDA_GetGreenEnd1",
        "CDA_GetGreenStart2", "CDA_GetGreenEnd2",
        "CDA_GetRedStart1",   "CDA_GetRedStart2",
        "CDA_GetRefAcc",      "CDA_GetTPEndLoc",
        "CDA_GetLatestTPTime",
    ):
        getattr(dll, name).return_value = 0.0


def _command(*, lane: int = 0, dt: float = _NOMINAL_DT) -> dict:
    return {"desired_lane_id": lane, "dt": dt, "accel_mps2": 0.0}


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAccMps2(unittest.TestCase):

    # ── reset_episode ─────────────────────────────────────────────────────

    def test_reset_acc_is_zero(self):
        """
        acc_mps2 in the reset snapshot must be 0.0.

        There is no previous speed to subtract from, so the connector
        initialises _prev_speed=0 and the first computed delta is zero.
        """
        dll = MagicMock()
        connector = _make_connector(dll)
        _prime_dll(dll, vir_time=1.0, speed=12.0)

        snap = connector.reset_episode()

        self.assertEqual(snap["ego"]["acc_mps2"], 0.0)

    def test_reset_nonzero_speed_still_gives_zero_acc(self):
        """Any initial speed must yield acc_mps2=0.0; speed is not 'previous'."""
        dll = MagicMock()
        connector = _make_connector(dll)
        _prime_dll(dll, vir_time=1.0, speed=30.0)

        snap = connector.reset_episode()

        self.assertEqual(snap["ego"]["acc_mps2"], 0.0)

    # ── first step after reset ─────────────────────────────────────────────

    def test_first_step_acc(self):
        """
        After reset at speed V0, a step arriving at V1 must produce:
            acc = (V1 - V0) / AIMSUN_DT
        The reader thread owns dt; the command's dt field is ignored.
        """
        dll = MagicMock()
        connector = _make_connector(dll)
        v0, v1 = 10.0, 13.0

        _prime_dll(dll, vir_time=1.0, speed=v0)
        connector.reset_episode()

        _prime_dll(dll, vir_time=2.0, speed=v1)
        _wait_for_vir_time(connector, 2.0)
        snap = connector.step(_command())

        self.assertAlmostEqual(snap["ego"]["acc_mps2"], (v1 - v0) / _AIMSUN_DT)

    def test_first_step_deceleration(self):
        """Deceleration (V1 < V0) must produce a negative acc_mps2."""
        dll = MagicMock()
        connector = _make_connector(dll)
        v0, v1 = 20.0, 17.0

        _prime_dll(dll, vir_time=1.0, speed=v0)
        connector.reset_episode()

        _prime_dll(dll, vir_time=2.0, speed=v1)
        _wait_for_vir_time(connector, 2.0)
        snap = connector.step(_command())

        self.assertAlmostEqual(snap["ego"]["acc_mps2"], (v1 - v0) / _AIMSUN_DT)

    def test_first_step_zero_delta(self):
        """Constant speed must yield acc_mps2=0.0."""
        dll = MagicMock()
        connector = _make_connector(dll)
        v = 15.0

        _prime_dll(dll, vir_time=1.0, speed=v)
        connector.reset_episode()

        _prime_dll(dll, vir_time=2.0, speed=v)
        snap = connector.step(_command())

        self.assertEqual(snap["ego"]["acc_mps2"], 0.0)

    # ── cursor advances across steps ──────────────────────────────────────

    def test_second_step_uses_first_step_speed_as_prev(self):
        """
        _prev_speed must advance each step.

        step1: acc = (V1 - V0) / AIMSUN_DT
        step2: acc = (V2 - V1) / AIMSUN_DT   ← V1 is the new prev, not V0
        """
        dll = MagicMock()
        connector = _make_connector(dll)
        v0, v1, v2 = 10.0, 13.0, 11.5

        _prime_dll(dll, vir_time=1.0, speed=v0)
        connector.reset_episode()

        _prime_dll(dll, vir_time=2.0, speed=v1)
        _wait_for_vir_time(connector, 2.0)
        connector.step(_command())

        _prime_dll(dll, vir_time=3.0, speed=v2)
        _wait_for_vir_time(connector, 3.0)
        snap = connector.step(_command())

        self.assertAlmostEqual(snap["ego"]["acc_mps2"], (v2 - v1) / _AIMSUN_DT)

    def test_three_steps_cursor_chain(self):
        """Prev-speed cursor must chain correctly across three consecutive steps."""
        dll = MagicMock()
        connector = _make_connector(dll)
        speeds = [5.0, 8.0, 6.0, 9.5]

        _prime_dll(dll, vir_time=1.0, speed=speeds[0])
        connector.reset_episode()

        for i, (v_curr, vir_t) in enumerate(zip(speeds[1:], [2.0, 3.0, 4.0]), start=1):
            _prime_dll(dll, vir_time=vir_t, speed=v_curr)
            _wait_for_vir_time(connector, vir_t)
            snap = connector.step(_command())
            expected = (speeds[i] - speeds[i - 1]) / _AIMSUN_DT
            with self.subTest(step=i):
                self.assertAlmostEqual(snap["ego"]["acc_mps2"], expected)

    # ── reset clears prev_speed ────────────────────────────────────────────

    def test_second_reset_clears_prev_speed(self):
        """
        A second reset_episode must set _prev_speed=0 so acc_mps2 is 0.0
        again, regardless of what happened in the first episode.
        """
        dll = MagicMock()
        connector = _make_connector(dll)

        # First episode — accumulate non-zero prev_speed
        _prime_dll(dll, vir_time=1.0, speed=10.0)
        connector.reset_episode()
        _prime_dll(dll, vir_time=2.0, speed=20.0)
        connector.step(_command())

        # Second episode — VirVehTime must transition through 0 to nonzero.
        # side_effect: first poll sees 0 (stale), second poll sees nonzero (fresh).
        dll.CDA_GetLatestVirVehTime.side_effect = itertools.chain([0.0], itertools.repeat(5.0))
        dll.CDA_GetVirVehSpeed.return_value = 8.0

        snap = connector.reset_episode()

        self.assertEqual(snap["ego"]["acc_mps2"], 0.0)

    # ── custom dt ─────────────────────────────────────────────────────────

    def test_aimsun_dt_always_used(self):
        """
        acc_mps2 must always use AIMSUN_DT (0.1 s) regardless of any dt
        value in the command.  dt is owned by the reader thread.
        """
        dll = MagicMock()
        connector = _make_connector(dll)

        _prime_dll(dll, vir_time=1.0, speed=10.0)
        connector.reset_episode()

        _prime_dll(dll, vir_time=2.0, speed=11.0)
        _wait_for_vir_time(connector, 2.0)
        snap = connector.step(_command(dt=1.0 / 15.0))  # command dt ignored

        self.assertAlmostEqual(snap["ego"]["acc_mps2"], (11.0 - 10.0) / _AIMSUN_DT)

    def test_acc_correct_when_command_omits_dt(self):
        """
        If the command has no dt key, acc_mps2 must still use AIMSUN_DT.
        """
        dll = MagicMock()
        connector = _make_connector(dll)

        _prime_dll(dll, vir_time=1.0, speed=10.0)
        connector.reset_episode()

        _prime_dll(dll, vir_time=2.0, speed=12.5)
        _wait_for_vir_time(connector, 2.0)
        snap = connector.step({"desired_lane_id": 0, "accel_mps2": 0.0})  # no "dt"

        self.assertAlmostEqual(snap["ego"]["acc_mps2"], (12.5 - 10.0) / _AIMSUN_DT)


# ── Multi-vehicle helpers ───────────────────────────────────────────────────────

_AIMSUN_DT = CdaLiveConnector.AIMSUN_DT  # 0.1 s — the rate the reader thread uses


def _make_worker_connector(dll_mock: MagicMock, worker_id: int) -> CdaLiveConnector:
    """Instantiate a connector as an A3C worker would: unique ego_id and local_port."""
    config = {
        "dll_path":        "fake.dll",
        "remote_ip":       "127.0.0.1",
        "remote_port":     5555,
        "local_port":      5556 + worker_id,
        "ego_id":          worker_id + 1,
        "timeout_s":       2.0,
        "poll_interval_s": 0.0,
    }
    with patch("ctypes.CDLL", return_value=dll_mock):
        return CdaLiveConnector(config)


def _wait_for_vir_time(connector: CdaLiveConnector, vir_time: float, timeout: float = 2.0) -> None:
    """
    Block until the reader thread has ingested the frame at vir_time.
    _last_vir_time is set by the reader thread after each new frame; polling
    it from the test thread is safe in CPython (float assignment is atomic).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connector._last_vir_time == vir_time:
            return
        time.sleep(0.001)
    raise TimeoutError(f"reader thread did not reach vir_time={vir_time} within {timeout}s")


# ── Multi-vehicle tests ─────────────────────────────────────────────────────────

class TestMultiVehicle(unittest.TestCase):
    """
    Verify that concurrent A3C workers each maintain an isolated _prev_speed
    cursor.  Two connectors share no state even though they run in the same
    process; each computes acc_mps2 = (v_curr - v_prev) / AIMSUN_DT from its
    own speed history.
    """

    def test_two_workers_independent_acc(self):
        """
        Worker 0 and Worker 1 start at different speeds.
        Each must compute acc from its own prev_speed, not the other's.
        """
        dll0, dll1 = MagicMock(), MagicMock()
        w0 = _make_worker_connector(dll0, worker_id=0)
        w1 = _make_worker_connector(dll1, worker_id=1)

        v0_reset, v1_reset = 10.0, 20.0
        _prime_dll(dll0, vir_time=1.0, speed=v0_reset)
        _prime_dll(dll1, vir_time=1.0, speed=v1_reset)
        w0.reset_episode()
        w1.reset_episode()

        v0_step, v1_step = 13.0, 17.0
        _prime_dll(dll0, vir_time=2.0, speed=v0_step)
        _prime_dll(dll1, vir_time=2.0, speed=v1_step)
        _wait_for_vir_time(w0, 2.0)
        _wait_for_vir_time(w1, 2.0)

        snap0 = w0.step(_command())
        snap1 = w1.step(_command())

        self.assertAlmostEqual(snap0["ego"]["acc_mps2"], (v0_step - v0_reset) / _AIMSUN_DT)
        self.assertAlmostEqual(snap1["ego"]["acc_mps2"], (v1_step - v1_reset) / _AIMSUN_DT)

    def test_worker_reset_does_not_affect_sibling(self):
        """
        Resetting Worker 0 mid-episode must not touch Worker 1's _prev_speed.
        Worker 1's acc chain must continue uninterrupted.
        """
        dll0, dll1 = MagicMock(), MagicMock()
        w0 = _make_worker_connector(dll0, worker_id=0)
        w1 = _make_worker_connector(dll1, worker_id=1)

        _prime_dll(dll0, vir_time=1.0, speed=10.0)
        _prime_dll(dll1, vir_time=1.0, speed=5.0)
        w0.reset_episode()
        w1.reset_episode()

        # Advance worker 1 one step so it has a meaningful prev_speed.
        _prime_dll(dll1, vir_time=2.0, speed=8.0)
        _wait_for_vir_time(w1, 2.0)
        w1.step(_command())

        # Reset worker 0 — only its _prev_speed must clear.
        _prime_dll(dll0, vir_time=1.0, speed=50.0)   # new episode for w0
        w0.reset_episode()

        # Worker 1 takes another step — must use 8.0 as prev, not 0.0.
        _prime_dll(dll1, vir_time=3.0, speed=11.0)
        _wait_for_vir_time(w1, 3.0)
        snap1 = w1.step(_command())

        self.assertAlmostEqual(snap1["ego"]["acc_mps2"], (11.0 - 8.0) / _AIMSUN_DT)

    def test_n_workers_each_correct_acc(self):
        """
        N workers each follow a unique speed trajectory.
        Every worker's acc_mps2 must equal (v_step - v_reset) / AIMSUN_DT
        using only that worker's own speed history.
        """
        n = 4
        resets = [5.0 * (i + 1) for i in range(n)]   # 5, 10, 15, 20
        steps  = [r + 3.0 * (i + 1) for i, r in enumerate(resets)]  # 8, 13, 20, 29

        dlls       = [MagicMock() for _ in range(n)]
        connectors = [_make_worker_connector(dlls[i], worker_id=i) for i in range(n)]

        for i in range(n):
            _prime_dll(dlls[i], vir_time=1.0, speed=resets[i])
        for c in connectors:
            c.reset_episode()

        for i in range(n):
            _prime_dll(dlls[i], vir_time=2.0, speed=steps[i])
        for c in connectors:
            _wait_for_vir_time(c, 2.0)

        for i, c in enumerate(connectors):
            snap = c.step(_command())
            expected = (steps[i] - resets[i]) / _AIMSUN_DT
            with self.subTest(worker=i):
                self.assertAlmostEqual(snap["ego"]["acc_mps2"], expected)


# ── Surrounding-vehicle helpers ─────────────────────────────────────────────────

def _prime_surrounding(
    dll: MagicMock,
    *,
    front_speed: float = 0.0,  front_gap: float = 50.0,
    llead_speed: float = 0.0,  llead_gap: float = 50.0,
    llag_speed:  float = 0.0,  llag_gap:  float = 50.0,
    rlead_speed: float = 0.0,  rlead_gap: float = 50.0,
    rlag_speed:  float = 0.0,  rlag_gap:  float = 50.0,
    left_lc_dir:  int = 2,
    right_lc_dir: int = 2,
) -> None:
    """Set all surrounding-vehicle DLL getters on an already-primed dll mock."""
    dll.CDA_GetLeadSpeed.return_value     = front_speed
    dll.CDA_GetLeadGap.return_value       = front_gap
    dll.CDA_GetLeftLeadSpeed.return_value = llead_speed
    dll.CDA_GetLeftLeadGap.return_value   = llead_gap
    dll.CDA_GetLeftLagSpeed.return_value  = llag_speed
    dll.CDA_GetLeftLagGap.return_value    = llag_gap
    dll.CDA_GetRightLeadSpeed.return_value = rlead_speed
    dll.CDA_GetRightLeadGap.return_value   = rlead_gap
    dll.CDA_GetRightLagSpeed.return_value  = rlag_speed
    dll.CDA_GetRightLagGap.return_value    = rlag_gap
    dll.CDA_GetLeftLCDir.return_value      = left_lc_dir
    dll.CDA_GetRightLCDir.return_value     = right_lc_dir


def _reset_and_step_surrounding(dll: MagicMock) -> dict:
    """
    Create a single-ego connector, prime surrounding values, wait for the
    reader thread, and return the snapshot from step().
    """
    connector = _make_connector(dll)
    _prime_dll(dll, vir_time=1.0, speed=10.0)
    connector.reset_episode()

    _prime_dll(dll, vir_time=2.0, speed=10.0)
    _wait_for_vir_time(connector, 2.0)
    return connector.step(_command())


# ── Surrounding-vehicle tests ────────────────────────────────────────────────────

class TestSurroundingVehicles(unittest.TestCase):
    """
    Verify that the connector correctly routes each surrounding-vehicle DLL
    getter to the expected key in snapshot["surrounding"], and that lc_dir
    values land in snapshot["cda"] instead.
    """

    def test_front_lead_passthrough(self):
        """Front (same-lane lead) speed and gap appear in surrounding["front"]."""
        dll = MagicMock()
        _prime_surrounding(dll, front_speed=25.0, front_gap=18.5)
        snap = _reset_and_step_surrounding(dll)

        front = snap["surrounding"]["front"]
        self.assertAlmostEqual(front["speed_mps"], 25.0)
        self.assertAlmostEqual(front["gap_m"],     18.5)

    def test_left_lead_passthrough(self):
        """Left-lead speed and gap appear in surrounding["llead"]."""
        dll = MagicMock()
        _prime_surrounding(dll, llead_speed=22.0, llead_gap=12.0)
        snap = _reset_and_step_surrounding(dll)

        llead = snap["surrounding"]["llead"]
        self.assertAlmostEqual(llead["speed_mps"], 22.0)
        self.assertAlmostEqual(llead["gap_m"],     12.0)

    def test_left_lag_passthrough(self):
        """Left-lag speed and gap appear in surrounding["llag"]."""
        dll = MagicMock()
        _prime_surrounding(dll, llag_speed=18.0, llag_gap=7.5)
        snap = _reset_and_step_surrounding(dll)

        llag = snap["surrounding"]["llag"]
        self.assertAlmostEqual(llag["speed_mps"], 18.0)
        self.assertAlmostEqual(llag["gap_m"],     7.5)

    def test_right_lead_passthrough(self):
        """Right-lead speed and gap appear in surrounding["rlead"]."""
        dll = MagicMock()
        _prime_surrounding(dll, rlead_speed=30.0, rlead_gap=9.0)
        snap = _reset_and_step_surrounding(dll)

        rlead = snap["surrounding"]["rlead"]
        self.assertAlmostEqual(rlead["speed_mps"], 30.0)
        self.assertAlmostEqual(rlead["gap_m"],     9.0)

    def test_right_lag_passthrough(self):
        """Right-lag speed and gap appear in surrounding["rlag"]."""
        dll = MagicMock()
        _prime_surrounding(dll, rlag_speed=14.0, rlag_gap=20.0)
        snap = _reset_and_step_surrounding(dll)

        rlag = snap["surrounding"]["rlag"]
        self.assertAlmostEqual(rlag["speed_mps"], 14.0)
        self.assertAlmostEqual(rlag["gap_m"],     20.0)

    def test_lc_dir_in_cda_not_surrounding(self):
        """
        leftLCDir and rightLCDir must be in snapshot["cda"], not in
        snapshot["surrounding"].  The bridge has no slot for them in the
        neighbor synthesis path.
        """
        dll = MagicMock()
        _prime_surrounding(dll, left_lc_dir=1, right_lc_dir=3)
        snap = _reset_and_step_surrounding(dll)

        self.assertEqual(snap["cda"]["leftLCDir"],  1)
        self.assertEqual(snap["cda"]["rightLCDir"], 3)
        self.assertNotIn("leftLCDir",  snap["surrounding"])
        self.assertNotIn("rightLCDir", snap["surrounding"])

    def test_all_positions_distinct_values(self):
        """
        All five surrounding positions are populated simultaneously with
        distinct values; each must arrive at the correct key.
        """
        dll = MagicMock()
        _prime_surrounding(
            dll,
            front_speed=30.0, front_gap=15.0,
            llead_speed=28.0, llead_gap=12.0,
            llag_speed=25.0,  llag_gap=8.0,
            rlead_speed=27.0, rlead_gap=11.0,
            rlag_speed=24.0,  rlag_gap=6.0,
        )
        snap = _reset_and_step_surrounding(dll)
        s = snap["surrounding"]

        expected = {
            "front": (30.0, 15.0),
            "llead": (28.0, 12.0),
            "llag":  (25.0,  8.0),
            "rlead": (27.0, 11.0),
            "rlag":  (24.0,  6.0),
        }
        for key, (exp_speed, exp_gap) in expected.items():
            with self.subTest(position=key):
                self.assertAlmostEqual(s[key]["speed_mps"], exp_speed)
                self.assertAlmostEqual(s[key]["gap_m"],     exp_gap)

    def test_surrounding_keys_present(self):
        """
        snapshot["surrounding"] must contain exactly the five keys the bridge
        expects: front, llead, llag, rlead, rlag.
        """
        dll = MagicMock()
        snap = _reset_and_step_surrounding(dll)

        self.assertEqual(
            set(snap["surrounding"].keys()),
            {"front", "llead", "llag", "rlead", "rlag"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
