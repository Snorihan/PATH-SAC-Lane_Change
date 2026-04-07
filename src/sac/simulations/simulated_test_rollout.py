"""
simulated_test_rollout.py — smoke-test the full pipeline without a real Aimsun.

Spins up an in-process _FakeAimsun loopback that replies to every START2 packet
with a plausible START1, so the full connector → bridge → env → reward path
can be exercised on any machine.

Usage (from src/sac/):
  python simulations/simulated_test_rollout.py

Edit the CONFIG block below as needed.
"""

import sys
import os
import socket
import threading
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # src/sac on path

import lanechange_env  # noqa: F401 — registers lane-changing-v0
from cda_live.aapi_connector import AapiDirectConnector, _pack, _START1, _START2, _MSGEND
from lanechange_env import LaneChangingEnv

# ── CONFIG ────────────────────────────────────────────────────────────────────

EGO_ID            = 1
INITIAL_POS_M     = 100.0
INITIAL_SPEED_MPS = 15.0
N_STEPS           = 200
POLICY_HZ         = 15
DURATION_S        = 40
LANES             = 4
SPEED_LIMIT       = 30.0

# ─────────────────────────────────────────────────────────────────────────────


class _FakeAimsun(threading.Thread):
    """
    In-process loopback Aimsun server.
    Listens for START2 packets and immediately replies with a plausible START1.
    Surrounding vehicles are fixed but non-zero so all reward terms fire.
    """

    def __init__(self, connector_local_port: int, ego_id: int) -> None:
        super().__init__(daemon=True, name="FakeAimsun")
        self._target_port = connector_local_port
        self._ego_id      = ego_id
        self._sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.settimeout(0.05)
        self.port: int    = self._sock.getsockname()[1]
        self._stopped     = False
        self._sim_tick    = 10   # start at t=1.0 s (simTime units: 0.1 s)

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
            reply = self._make_reply()
            try:
                self._sock.sendto(reply, ("127.0.0.1", self._target_port))
            except OSError:
                break
            self._sim_tick += 1

    def _make_reply(self) -> bytes:
        t = time.localtime()
        pkt = _pack({
            "simID":          0,
            "ID":             self._ego_id,
            "targetCAVID":    self._ego_id,
            "leaderID":       self._ego_id,
            "year":           t.tm_year,
            "month":          t.tm_mon,
            "day":            t.tm_mday,
            "hour":           t.tm_hour,
            "minute":         t.tm_min,
            "second":         t.tm_sec,
            "ms":             0,
            "simTime":        self._sim_tick,
            "speed":          int(INITIAL_SPEED_MPS * 1000),
            "linkID":         1,
            "linkPos":        int(INITIAL_POS_M * 1000),
            "laneID":         1,
            "currentLane":    1,
            "totalLane":      LANES,
            # plausible surrounding vehicles so all reward terms fire
            "leadSpeed":      2200,   # 22.0 m/s  (raw scale ×0.01)
            "leadGap":        300,    # 30.0 m    (raw scale ×0.1)
            "leftLeadSpeed":  2100,
            "leftLeadGap":    250,
            "leftLagSpeed":   1900,
            "leftLagGap":     200,
            "rightLeadSpeed": 2300,
            "rightLeadGap":   350,
            "rightLagSpeed":  1800,
            "rightLagGap":    150,
        })
        # swap START2 header → START1
        return _START1 + pkt[len(_START2):-len(_MSGEND)] + _MSGEND

    def stop(self) -> None:
        self._stopped = True
        try:
            self._sock.close()
        except OSError:
            pass


def _sample_action(env: LaneChangingEnv) -> np.ndarray:
    front_v, _, gap_front, _ = env._vehicle_in_front_rear(
        env.vehicle.lane_index, max_range=200.0
    )
    accel  = 0.05
    intent = 0.0
    if front_v is not None and gap_front < 25.0:
        if float(env.vehicle.speed - front_v.speed) > 1.0:
            intent = 1.0
    return np.array([accel, intent], dtype=np.float32)


def _pretty(x) -> str:
    if x is None:
        return "None"
    try:
        f = float(x)
        return "inf" if f == float("inf") else "-inf" if f == float("-inf") else f"{f:+.3f}"
    except Exception:
        return str(x)


def main():
    # Pick a free local port for the connector to listen on.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
        _s.bind(("127.0.0.1", 0))
        local_port = _s.getsockname()[1]

    fake = _FakeAimsun(connector_local_port=local_port, ego_id=EGO_ID)
    fake.start()
    print(f"[sim_rollout] FakeAimsun listening on port {fake.port}, "
          f"connector on port {local_port}")

    connector = AapiDirectConnector({
        "remote_ip":         "127.0.0.1",
        "remote_port":       fake.port,
        "local_port":        local_port,
        "ego_id":            EGO_ID,
        "link_id":           1,
        "initial_pos_m":     INITIAL_POS_M,
        "initial_speed_mps": INITIAL_SPEED_MPS,
        "timeout_s":         5.0,
        "poll_interval_s":   0.001,
    })

    env = LaneChangingEnv(config={
        "backend":          "aimsun_live",
        "live_connector":   connector,
        "lanes_count":      LANES,
        "lane_width":       4.0,
        "road_length":      1000.0,
        "policy_frequency": POLICY_HZ,
        "duration":         DURATION_S,
        "speed_limit":      SPEED_LIMIT,
    })

    try:
        obs, info = env.reset()
        print(f"[sim_rollout] Reset OK  lane={env.vehicle.lane_index}  "
              f"speed={env.vehicle.speed:.1f} m/s  target={env.target_lane_index}")

        total_reward = 0.0
        for step in range(N_STEPS):
            action = _sample_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            rt       = info.get("reward_terms", {})
            lcs      = info.get("lane_change_state", {})
            snap_ego = info.get("snapshot", {}).get("ego", {})

            print(
                f"step {step:03d} | "
                f"lane={env.vehicle.lane_index[2]}  "
                f"spd={_pretty(env.vehicle.speed)}  "
                f"pos={_pretty(snap_ego.get('pos_m'))}  "
                f"lc={lcs.get('lane_changing', '?')}  "
                f"| r={_pretty(reward)}  "
                f"(col={_pretty(rt.get('collision'))}  "
                f"jrk={_pretty(rt.get('jerk'))}  "
                f"spd={_pretty(rt.get('speed'))}  "
                f"dst={_pretty(rt.get('dist'))}  "
                f"lim={_pretty(rt.get('speed_limit'))}  "
                f"suc={_pretty(rt.get('lane_success'))}  "
                f"chg={_pretty(rt.get('lane_changing'))})"
            )

            if terminated:
                print(f"[sim_rollout] TERMINATED at step {step}")
                break
            if truncated:
                print(f"[sim_rollout] TRUNCATED at step {step}")
                break

        print(f"\n[sim_rollout] Done.  total_reward={total_reward:.2f}  "
              f"steps={env.elapsed_steps}")

    finally:
        env.close()
        fake.stop()
        print("[sim_rollout] Closed.")


if __name__ == "__main__":
    main()
