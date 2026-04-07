"""
Unit tests: AapiDirectConnector kinematics and acc_mps2 computation.

No real Aimsun or DLL required.  A _FakeAimsun loopback thread responds to
every START2 packet with a minimal START1 reply so reset_episode() can unblock.

The acc_mps2 in each snapshot is computed entirely from the connector's own
Python-side kinematics:

    acc_mps2 = (speed_after_step - speed_before_step) / AIMSUN_DT

where AIMSUN_DT = 0.1 s (fixed), and speeds are updated by:

    new_speed = clip(prev_speed + accel_mps2_command * dt, 0, max_speed)

Tests verify:
  - reset_episode() always reports acc_mps2 = 0.0
  - first step computes correct acc from (init_speed → new_speed) delta
  - _prev_speed cursor advances correctly across multiple steps
  - second reset_episode() clears _prev_speed so next acc is 0.0
  - multiple concurrent connectors maintain independent _prev_speed state

Run from the repo root:
    python -m pytest src/sac/cda_live/test_cda_acc_mps2.py -v
"""
from __future__ import annotations

import os
import socket
import struct
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cda_live.aapi_connector import (   # noqa: E402
    AapiDirectConnector,
    _pack, _START1, _START2, _MSGEND,
)

# Fixed divisor used inside _reader_loop for acc calculation.
_AIMSUN_DT = AapiDirectConnector.AIMSUN_DT   # 0.1 s
_NOMINAL_DT = 1.0 / 15.0                      # 15 Hz policy step


# ── Fake Aimsun loopback ──────────────────────────────────────────────────────

class _FakeAimsun(threading.Thread):
    """
    Minimal UDP loopback server.  Each START2 packet received triggers one
    START1 reply with a monotonically increasing simTime so the reader thread
    always sees a 'new' frame.
    """

    def __init__(self, connector_local_port: int, ego_id: int = 1) -> None:
        super().__init__(daemon=True, name="FakeAimsun")
        self._target_port = connector_local_port
        self._ego_id      = ego_id
        self._sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.settimeout(0.05)
        self.port: int    = self._sock.getsockname()[1]
        self._stopped     = False
        self._sim_tick    = 10   # start at t=1.0 s (units: 0.1 s)

    def run(self) -> None:
        while not self._stopped:
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data.startswith(_START2):
                continue
            reply = self._make_start1(self._sim_tick)
            try:
                self._sock.sendto(reply, ("127.0.0.1", self._target_port))
            except OSError:
                break
            self._sim_tick += 1

    def _make_start1(self, sim_tick: int) -> bytes:
        import time as _t
        t = _t.localtime()
        pkt = _pack({
            "simID":       0,
            "ID":          self._ego_id,
            "targetCAVID": self._ego_id,
            "leaderID":    self._ego_id,
            "year":        t.tm_year,
            "month":       t.tm_mon,
            "day":         t.tm_mday,
            "hour":        t.tm_hour,
            "minute":      t.tm_min,
            "second":      t.tm_sec,
            "ms":          0,
            "simTime":     sim_tick,
            "speed":       15000,   # 15.0 m/s × 1000
            "linkID":      1,
            "linkPos":     100000,  # 100.0 m × 1000
            "laneID":      1,
            "currentLane": 1,
            "totalLane":   4,
            "leadSpeed":   2200,    # 22.0 m/s (scale 0.01)
            "leadGap":     300,     # 30.0 m (scale 0.1)
        })
        # Swap START2 header → START1
        return _START1 + pkt[len(_START2):-len(_MSGEND)] + _MSGEND

    def stop(self) -> None:
        self._stopped = True
        try:
            self._sock.close()
        except OSError:
            pass


# ── Port helper ───────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Connector factory ─────────────────────────────────────────────────────────

def _make(
    *,
    ego_id: int = 1,
    initial_speed: float = 15.0,
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
        "next_link_id":      2,
        "initial_pos_m":     100.0,
        "initial_speed_mps": initial_speed,
        "max_speed_mps":     40.0,
        "timeout_s":         5.0,
        "poll_interval_s":   0.001,
    })
    return connector, server


def _command(accel: float = 0.0, dt: float = _NOMINAL_DT) -> dict:
    return {"accel_mps2": accel, "dt": dt, "desired_lane_id": 0}


# ── Tests: reset ──────────────────────────────────────────────────────────────

class TestResetAcc(unittest.TestCase):

    def test_reset_acc_is_zero(self):
        """acc_mps2 in the reset snapshot must always be 0.0."""
        c, srv = _make()
        try:
            snap = c.reset_episode()
            self.assertEqual(snap["ego"]["acc_mps2"], 0.0)
        finally:
            c.close(); srv.stop()

    def test_reset_nonzero_initial_speed_still_zero_acc(self):
        """Any initial speed must still produce acc_mps2 = 0.0 at reset."""
        c, srv = _make(initial_speed=28.0)
        try:
            snap = c.reset_episode()
            self.assertEqual(snap["ego"]["acc_mps2"], 0.0)
        finally:
            c.close(); srv.stop()


# ── Tests: first step ─────────────────────────────────────────────────────────

class TestFirstStepAcc(unittest.TestCase):

    def test_accel_command_increases_speed(self):
        """
        After reset at V0, step(accel=a, dt=dt) must yield:
            acc_mps2 = (clip(V0 + a*dt, 0, 40) - V0) / AIMSUN_DT
        """
        v0 = 15.0
        a  = 2.0
        dt = _NOMINAL_DT
        c, srv = _make(initial_speed=v0)
        try:
            c.reset_episode()
            snap = c.step(_command(accel=a, dt=dt))
            v1_expected = min(40.0, v0 + a * dt)
            acc_expected = (v1_expected - v0) / _AIMSUN_DT
            self.assertAlmostEqual(snap["ego"]["acc_mps2"], acc_expected, places=6)
        finally:
            c.close(); srv.stop()

    def test_decel_command_decreases_speed(self):
        """Negative accel must produce a negative acc_mps2."""
        v0 = 20.0
        a  = -3.0
        dt = _NOMINAL_DT
        c, srv = _make(initial_speed=v0)
        try:
            c.reset_episode()
            snap = c.step(_command(accel=a, dt=dt))
            v1_expected = max(0.0, v0 + a * dt)
            acc_expected = (v1_expected - v0) / _AIMSUN_DT
            self.assertAlmostEqual(snap["ego"]["acc_mps2"], acc_expected, places=6)
        finally:
            c.close(); srv.stop()

    def test_zero_accel_zero_acc(self):
        """Constant speed (accel=0) must give acc_mps2 = 0.0."""
        c, srv = _make(initial_speed=15.0)
        try:
            c.reset_episode()
            snap = c.step(_command(accel=0.0))
            self.assertAlmostEqual(snap["ego"]["acc_mps2"], 0.0, places=9)
        finally:
            c.close(); srv.stop()

    def test_speed_clipped_at_zero(self):
        """
        Large deceleration must clip speed at 0.  acc must use the clipped speed.
        """
        v0 = 2.0
        c, srv = _make(initial_speed=v0)
        try:
            c.reset_episode()
            # a*dt = -100 → would go negative; clipped to 0
            snap = c.step(_command(accel=-100.0, dt=_NOMINAL_DT))
            acc_expected = (0.0 - v0) / _AIMSUN_DT
            self.assertAlmostEqual(snap["ego"]["acc_mps2"], acc_expected, places=6)
        finally:
            c.close(); srv.stop()

    def test_speed_clipped_at_max(self):
        """Large positive accel must clip speed at max_speed_mps (40.0)."""
        v0 = 38.0
        c, srv = _make(initial_speed=v0)
        try:
            c.reset_episode()
            snap = c.step(_command(accel=100.0, dt=_NOMINAL_DT))
            acc_expected = (40.0 - v0) / _AIMSUN_DT
            self.assertAlmostEqual(snap["ego"]["acc_mps2"], acc_expected, places=6)
        finally:
            c.close(); srv.stop()


# ── Tests: cursor chains across steps ────────────────────────────────────────

class TestAccCursorChain(unittest.TestCase):

    def test_second_step_uses_first_step_speed_as_prev(self):
        """
        _prev_speed must advance each step:
            step1: acc = (V1 - V0) / AIMSUN_DT
            step2: acc = (V2 - V1) / AIMSUN_DT   ← V1 is the new prev
        """
        v0 = 10.0
        a1, a2 = 1.0, -0.5
        dt = _NOMINAL_DT
        c, srv = _make(initial_speed=v0)
        try:
            c.reset_episode()
            c.step(_command(accel=a1, dt=dt))
            v1 = min(40.0, v0 + a1 * dt)
            snap = c.step(_command(accel=a2, dt=dt))
            v2 = max(0.0, min(40.0, v1 + a2 * dt))
            expected = (v2 - v1) / _AIMSUN_DT
            self.assertAlmostEqual(snap["ego"]["acc_mps2"], expected, places=6)
        finally:
            c.close(); srv.stop()

    def test_three_step_cursor_chain(self):
        """_prev_speed cursor must chain correctly across three steps."""
        v0 = 5.0
        accels = [2.0, -1.0, 0.5]
        dt = _NOMINAL_DT
        c, srv = _make(initial_speed=v0)
        try:
            c.reset_episode()
            speeds = [v0]
            for a in accels:
                v_next = max(0.0, min(40.0, speeds[-1] + a * dt))
                speeds.append(v_next)

            for i, a in enumerate(accels):
                snap = c.step(_command(accel=a, dt=dt))
                expected = (speeds[i + 1] - speeds[i]) / _AIMSUN_DT
                with self.subTest(step=i + 1):
                    self.assertAlmostEqual(snap["ego"]["acc_mps2"], expected, places=6)
        finally:
            c.close(); srv.stop()


# ── Tests: reset clears prev_speed ───────────────────────────────────────────

class TestSecondReset(unittest.TestCase):

    def test_second_reset_acc_is_zero(self):
        """
        After stepping (accumulating non-zero _prev_speed), a second
        reset_episode() must clear _prev_speed so acc_mps2 is 0.0 again.
        """
        c, srv = _make(initial_speed=10.0)
        try:
            c.reset_episode()
            c.step(_command(accel=3.0))
            c.step(_command(accel=3.0))  # _prev_speed is now well above init

            snap = c.reset_episode()
            self.assertEqual(snap["ego"]["acc_mps2"], 0.0)
        finally:
            c.close(); srv.stop()

    def test_second_reset_restores_initial_speed(self):
        """After reset, vehicle.speed must equal initial_speed_mps."""
        v0 = 12.0
        c, srv = _make(initial_speed=v0)
        try:
            c.reset_episode()
            c.step(_command(accel=5.0, dt=1.0))   # large delta to drift speed
            snap = c.reset_episode()
            self.assertAlmostEqual(snap["ego"]["speed_mps"], v0, places=6)
        finally:
            c.close(); srv.stop()


# ── Tests: multiple independent connectors ────────────────────────────────────

class TestMultiConnector(unittest.TestCase):
    """
    Two connectors with different initial speeds must each track their own
    kinematics independently — stepping one must not affect the other's acc.
    """

    def test_two_connectors_independent_acc(self):
        v0a, v0b = 10.0, 20.0
        a = 2.0
        dt = _NOMINAL_DT

        ca, srv_a = _make(ego_id=1, initial_speed=v0a)
        cb, srv_b = _make(ego_id=2, initial_speed=v0b)
        try:
            ca.reset_episode()
            cb.reset_episode()

            snap_a = ca.step(_command(accel=a, dt=dt))
            snap_b = cb.step(_command(accel=a, dt=dt))

            va1 = min(40.0, v0a + a * dt)
            vb1 = min(40.0, v0b + a * dt)

            self.assertAlmostEqual(
                snap_a["ego"]["acc_mps2"], (va1 - v0a) / _AIMSUN_DT, places=6
            )
            self.assertAlmostEqual(
                snap_b["ego"]["acc_mps2"], (vb1 - v0b) / _AIMSUN_DT, places=6
            )
        finally:
            ca.close(); srv_a.stop()
            cb.close(); srv_b.stop()

    def test_reset_one_does_not_affect_other(self):
        """
        Resetting connector A mid-episode must not perturb connector B's
        _prev_speed — B's acc must continue its own chain uninterrupted.
        """
        v0a, v0b = 10.0, 5.0
        a = 1.5
        dt = _NOMINAL_DT

        ca, srv_a = _make(ego_id=1, initial_speed=v0a)
        cb, srv_b = _make(ego_id=2, initial_speed=v0b)
        try:
            ca.reset_episode()
            cb.reset_episode()

            # Advance B one step so its _prev_speed is v0b + a*dt
            cb.step(_command(accel=a, dt=dt))
            vb1 = min(40.0, v0b + a * dt)

            # Reset A — must not touch B
            ca.reset_episode()

            # B's second step must use vb1 as prev
            snap_b = cb.step(_command(accel=a, dt=dt))
            vb2 = min(40.0, vb1 + a * dt)
            self.assertAlmostEqual(
                snap_b["ego"]["acc_mps2"], (vb2 - vb1) / _AIMSUN_DT, places=6
            )
        finally:
            ca.close(); srv_a.stop()
            cb.close(); srv_b.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
