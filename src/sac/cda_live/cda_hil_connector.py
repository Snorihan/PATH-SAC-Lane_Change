"""
cda_hil_connector.py — Python connector that speaks the CDA_Interface protocol
to the HIL Tool, mirroring exactly what test.cpp (CDA_Interface.cpp) does.

Architecture:
  [Aimsun] ──→ [HIL Tool :7999] ──→ [live_rollout.py :7998]

Protocol (Variable_Definition.h / CDA_Interface.cpp):
  Outbound  Python → HIL Tool port 7999:
      "START2" + bitsery(TestVehData) + "MSGEND"
      Python sends ego state each step so the HIL Tool knows vehicle position,
      speed, and lane (displayed on the dashboard, fed back to Aimsun).

  Inbound   HIL Tool → Python port 7998:
      raw bitsery(Server2TestVehData)   (no START/MSGEND framing)
      HIL Tool sends surrounding-vehicle state, signal control, and TP data
      for each registered test vehicle.

Bitsery serialises each field as little-endian raw binary with no padding,
equivalent to Python struct.pack with '<' prefix.  Struct layout is taken
directly from Variable_Definition.h serialize() template functions.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any

from .live_connector_abstract import LiveConnector


# ── Wire constants ─────────────────────────────────────────────────────────────

_START2 = b"START2"   # Python → HIL Tool frame header
_MSGEND = b"MSGEND"   # frame footer


# ── TestVehData (outbound: Python → HIL Tool) ──────────────────────────────────
# Serialisation order matches Variable_Definition.h serialize(S&, TestVehData&).
#
#   year(h)  month(b)  day(b)   hour(b)  minute(b)  second(b)  ms(h)
#   ID(h)  leaderID(h)  stringPos(b)  route(b)
#   v(h)   vLat(h)   pos(i)   lane(b)
#
# v in 0.001 m/s (int16), vLat in 0.001 m/s (int16), pos in 0.001 m (int32)

_TVD_FMT  = "<hbbbbbhhhbbhhib"
_TVD_SIZE = struct.calcsize(_TVD_FMT)   # 24 bytes

_TVD_FIELDS = [
    "year", "month", "day", "hour", "minute", "second", "ms",
    "ID", "leaderID", "stringPos", "route",
    "v", "vLat", "pos", "lane",
]


# ── Server2TestVehData (inbound: HIL Tool → Python) ────────────────────────────
# Serialisation order matches Variable_Definition.h serialize(S&, Server2TestVehData&).
#
#   year(h) month(b) day(b) hour(b) minute(b) second(b) ms(h)
#   testVehID(h) simID(h) targetVehID(h)
#   v(i) pos(i) refAcc(i)
#   signalState(b) endTime(i)
#   g1(i) g2(i) ge1(i) ge2(i) r1(i) r2(i)
#   leftLCDir(b)  rightLCDir(b)
#   leftLeadSpeed(h)  leftLeadGap(i)
#   leftLagSpeed(h)   leftLagGap(i)
#   rightLeadSpeed(h) rightLeadGap(i)
#   rightLagSpeed(h)  rightLagGap(i)
#   leadSpeed(h)  leadGap(i)
#   dist2Turn(i)  turnDir(b)  laneSpec(h)  currentLane(b)  totalLane(b)
#
# Speed fields in 0.01 m/s (int16), gap fields in 0.1 m (int32),
# v/pos in 0.001 m/s / 0.001 m (int32, leading virtual car),
# refAcc in 0.001 m/s² (int32)

_S2V_FMT  = "<hbbbbbhhhhiiibiiiiiiibbhihihihihiibhbb"
_S2V_SIZE = struct.calcsize(_S2V_FMT)   # 97 bytes

_S2V_FIELDS = [
    "year", "month", "day", "hour", "minute", "second", "ms",
    "testVehID", "simID", "targetVehID",
    "v", "pos", "refAcc",
    "signalState", "endTime",
    "g1", "g2", "ge1", "ge2", "r1", "r2",
    "leftLCDir", "rightLCDir",
    "leftLeadSpeed", "leftLeadGap",
    "leftLagSpeed",  "leftLagGap",
    "rightLeadSpeed","rightLeadGap",
    "rightLagSpeed", "rightLagGap",
    "leadSpeed",     "leadGap",
    "dist2Turn", "turnDir", "laneSpec", "currentLane", "totalLane",
]
_S2V_IDX = {name: i for i, name in enumerate(_S2V_FIELDS)}


# ── Wire helpers ───────────────────────────────────────────────────────────────

def _pack_testveh(f: dict) -> bytes:
    """Build a START2-framed TestVehData UDP packet from a field dict."""
    g = f.get
    # int16 speed clamp: 32767 * 0.001 = 32.767 m/s max representable
    v_raw = max(-32768, min(32767, g("v", 0)))
    return _START2 + struct.pack(
        _TVD_FMT,
        g("year",      0), g("month",  0), g("day",  0),
        g("hour",      0), g("minute", 0), g("second", 0),
        g("ms",        0),
        g("ID",        1), g("leaderID", 1),
        g("stringPos", 1), g("route",    1),
        v_raw, g("vLat", 0),
        g("pos",  0),
        g("lane", 1),
    ) + _MSGEND


def _unpack_s2v(data: bytes) -> dict | None:
    """Decode a raw Server2TestVehData packet. Returns None on parse error."""
    if len(data) < _S2V_SIZE:
        return None
    try:
        vals = struct.unpack_from(_S2V_FMT, data)
    except struct.error:
        return None
    return {name: vals[_S2V_IDX[name]] for name in _S2V_FIELDS}


# ── Connector class ────────────────────────────────────────────────────────────

class HilConnector(LiveConnector):
    """
    Drop-in replacement for AapiDirectConnector.
    Talks to the HIL Tool (not Aimsun directly), mirroring test.cpp / CDA_Interface.

    The HIL Tool handles all virtual vehicle data, signal control, and dashboard
    display; this connector just drives the "smart vehicle" ego.

    Config keys
    -----------
    hil_ip             str    HIL Tool machine IP.           Default "127.0.0.1"
    hil_port           int    HIL Tool receive port.         Default 7999
    local_port         int    Port HIL Tool sends back to.   Default 7998
    ego_id             int    Test vehicle ID.               REQUIRED
    initial_pos_m      float  Starting position (m).         Default 100.0
    initial_speed_mps  float  Starting speed (m/s).          Default 15.0
    initial_lane       int    Starting lane (1-based wire).  Default 1
    max_speed_mps      float  Speed ceiling (m/s).           Default 30.0
    timeout_s          float  reset() wait timeout (s).      Default 10.0
    poll_interval_s    float  Reader thread recv timeout.    Default 0.001
    """

    HIL_DT: float = 0.1   # Nominal HIL Tool update period (seconds)

    def __init__(self, config: dict) -> None:
        self._hil_ip     = str(config.get("hil_ip",           "127.0.0.1"))
        self._hil_port   = int(config.get("hil_port",          7999))
        self._local_port = int(config.get("local_port",        7998))
        self._ego_id     = int(config["ego_id"])                        # required
        self._max_speed  = float(config.get("max_speed_mps",   30.0))
        self._timeout_s  = float(config.get("timeout_s",       10.0))
        self._poll_s     = float(config.get("poll_interval_s", 0.001))

        self._init_pos_m       = float(config.get("initial_pos_m",     100.0))
        self._init_speed       = float(config.get("initial_speed_mps",  15.0))
        self._init_lane        = int(config.get("initial_lane",           1))   # 1-based wire
        self._teleport_on_reset = bool(config.get("teleport_on_reset",  True))

        # Ego kinematics — Python-maintained, sent to HIL Tool each step.
        self._pos_m      = self._init_pos_m
        self._speed_mps  = self._init_speed
        self._lane_wire  = self._init_lane
        self._prev_speed = self._init_speed

        # Stale-frame tracking: counts consecutive step() calls where the
        # HIL Tool timestamp in the cached snapshot has not changed.
        # Non-zero means HIL Tool stopped sending (sim paused / vehicle lost).
        self._stale_steps  = 0
        self._step_time_s  = -1.0

        # Shared state between reader thread and policy thread.
        self._lock                = threading.Lock()
        self._cached_snapshot: dict | None = None
        self._first_frame_event   = threading.Event()
        self._last_time_s: float  = -1.0
        self._closed               = False

        # UDP sockets.
        self._send_addr = (self._hil_ip, self._hil_port)
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.bind(("0.0.0.0", self._local_port))
        self._recv_sock.settimeout(self._poll_s)

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="HilReader", daemon=True
        )
        self._reader_thread.start()

    # ── LiveConnector interface ────────────────────────────────────────────────

    def reset_episode(self, seed=None) -> dict[str, Any]:
        with self._lock:
            if self._teleport_on_reset:
                self._pos_m      = self._init_pos_m
                self._speed_mps  = self._init_speed
                self._lane_wire  = self._init_lane
                self._prev_speed = self._init_speed
            self._cached_snapshot = None
            self._last_time_s     = -1.0
            self._stale_steps     = 0
            self._step_time_s     = -1.0
        self._first_frame_event.clear()

        # Bootstrap: send ego state so HIL Tool starts routing messages to us.
        self._send_ego_state()

        if not self._first_frame_event.wait(timeout=self._timeout_s):
            raise TimeoutError(
                f"HilConnector: no Server2TestVehData frame within {self._timeout_s}s. "
                f"HIL Tool did not send data for ego_id={self._ego_id}. "
                "Check: (1) HIL Tool is running and the Aimsun simulation is active, "
                f"(2) ego_id={self._ego_id} is registered in the HIL Tool config, "
                f"(3) hil_port={self._hil_port} / local_port={self._local_port} are correct."
            )

        with self._lock:
            snapshot = dict(self._cached_snapshot)
            # Sync lane to whatever the HIL Tool reports so _lane_wire
            # always reflects the car's actual lane, not the hardcoded init value.
            reported_lane_idx = int(snapshot["ego"]["lane_idx"])   # 0-based
            self._lane_wire = reported_lane_idx + 1                # → 1-based wire
        snapshot["ego"] = dict(snapshot["ego"])
        snapshot["ego"]["acc_mps2"] = 0.0
        return snapshot

    def step(self, command: dict[str, Any]) -> dict[str, Any]:
        dt          = float(command.get("dt",           self.HIL_DT))
        accel_mps2  = float(command.get("accel_mps2",  0.0))
        desired_lane = int(command.get("desired_lane_id", self._lane_wire - 1))

        with self._lock:
            prev      = self._speed_mps
            new_speed = float(max(0.0, min(self._max_speed, prev + accel_mps2 * dt)))
            new_pos   = self._pos_m + prev * dt
            new_lane  = desired_lane + 1   # 0-based env → 1-based wire

            self._prev_speed = prev
            self._speed_mps  = new_speed
            self._pos_m      = new_pos
            self._lane_wire  = new_lane

        acc = (new_speed - prev) / dt if dt > 1e-9 else accel_mps2

        self._send_ego_state()

        with self._lock:
            snap = dict(self._cached_snapshot) if self._cached_snapshot else {}
        if snap and "ego" in snap:
            snap["ego"] = dict(snap["ego"])
            snap["ego"]["acc_mps2"]  = acc
            snap["ego"]["speed_mps"] = new_speed
            snap["ego"]["pos_m"]     = new_pos

        # Stale detection: if the HIL Tool's timestamp hasn't advanced, the
        # connection stalled (sim paused or vehicle de-registered).
        cur_time_s = snap.get("time_s", -1.0)
        if cur_time_s == self._step_time_s:
            self._stale_steps += 1
        else:
            self._stale_steps  = 0
            self._step_time_s  = cur_time_s
        snap["aimsun_stale_steps"] = self._stale_steps

        return snap

    def get_total_lanes(self) -> int:
        """Block until the first HIL frame arrives and return totalLane.
        Call this before constructing the env so lanes_count is correct.
        reset_episode() will still work — it clears the event and waits for a fresh frame."""
        self._send_ego_state()
        if not self._first_frame_event.wait(timeout=self._timeout_s):
            return 4  # fallback if HIL Tool doesn't respond in time
        with self._lock:
            snap = self._cached_snapshot
        return int((snap.get("cda") or {}).get("totalLane") or 4)

    def idle(self, hz: float = 10.0) -> None:
        """Keep sending ego state so the HIL dashboard stays alive after training ends.
        Integrates position at the last known speed (constant velocity) so the
        vehicle keeps moving on the dashboard rather than freezing in place.
        Blocks until KeyboardInterrupt (Ctrl+C).
        hz: send rate — match HIL Tool update rate (default 10 Hz)."""
        interval = 1.0 / max(hz, 0.1)
        print(f"[HilConnector] idle — tracking ego at {hz} Hz. Press Ctrl+C to exit.")
        try:
            while True:
                with self._lock:
                    self._pos_m += self._speed_mps * interval
                self._send_ego_state()
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._reader_thread.join(timeout=2.0)
        try:
            self._send_sock.close()
            self._recv_sock.close()
        except OSError:
            pass

    # ── Outbound ──────────────────────────────────────────────────────────────

    def _send_ego_state(self) -> None:
        with self._lock:
            pos_raw   = int(self._pos_m     * 1000)   # 0.001 m  → int32
            speed_raw = int(self._speed_mps * 1000)   # 0.001 m/s → int16 (clamped in _pack)
            lane      = self._lane_wire

        t  = time.localtime()
        ms = int(time.time() * 1000) % 1000

        pkt = _pack_testveh({
            "year":      t.tm_year,
            "month":     t.tm_mon,
            "day":       t.tm_mday,
            "hour":      t.tm_hour,
            "minute":    t.tm_min,
            "second":    t.tm_sec,
            "ms":        ms,
            "ID":        self._ego_id,
            "leaderID":  self._ego_id,
            "stringPos": 1,
            "route":     1,
            "v":         speed_raw,
            "vLat":      0,
            "pos":       pos_raw,
            "lane":      lane,
        })
        try:
            self._send_sock.sendto(pkt, self._send_addr)
        except OSError:
            pass

    # ── Reader thread ──────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        while not self._closed:
            try:
                data, _ = self._recv_sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            msg = _unpack_s2v(data)
            if msg is None:
                continue
            if msg["targetVehID"] != self._ego_id:
                continue   # packet for a different test vehicle

            with self._lock:
                ego_pos   = self._pos_m
                ego_speed = self._speed_mps
                lane_wire = self._lane_wire

            snapshot = self._build_snapshot(msg, ego_pos, ego_speed, 0.0, lane_wire)

            time_s = (msg["hour"] * 3600 + msg["minute"] * 60
                      + msg["second"] + msg["ms"] / 1000.0)
            with self._lock:
                self._cached_snapshot = snapshot
                self._last_time_s     = time_s
            self._first_frame_event.set()

    # ── Snapshot builder ───────────────────────────────────────────────────────

    def _build_snapshot(
        self,
        msg: dict,
        ego_pos: float,
        ego_speed: float,
        ego_acc: float,
        lane_wire: int,
    ) -> dict[str, Any]:
        # currentLane from HIL Tool is 1-based (rightmost = 1); env uses 0-based.
        cur_lane_wire = int(msg["currentLane"])
        lane_idx = (cur_lane_wire - 1) if cur_lane_wire > 0 else max(0, lane_wire - 1)

        total_lane = max(1, int(msg["totalLane"])) if msg["totalLane"] > 0 else 4

        # Speed: int16 raw in 0.01 m/s units (same as VehMessage in aapi_connector)
        def spd(raw: int) -> float | None:
            return raw * 0.01 if raw > 0 else None

        # Gap: int32 raw in 0.1 m units; HIL Tool sends 0 or negative as "no vehicle" sentinel.
        def gap(raw: int) -> float | None:
            return None if raw <= 0 else raw * 0.1

        time_s = (msg["hour"] * 3600 + msg["minute"] * 60
                  + msg["second"] + msg["ms"] / 1000.0)

        left_lc  = int(msg["leftLCDir"])  if msg["leftLCDir"]  > 0 else 2  # default: through
        right_lc = int(msg["rightLCDir"]) if msg["rightLCDir"] > 0 else 2

        return {
            "ego": {
                "lane_idx":  lane_idx,
                "pos_m":     ego_pos,
                "speed_mps": ego_speed,
                "acc_mps2":  ego_acc,
                "crashed":   False,
            },
            "surrounding": {
                "front": {
                    "speed_mps": spd(msg["leadSpeed"]),
                    "gap_m":     gap(msg["leadGap"]),
                },
                "llead": {
                    "speed_mps": spd(msg["leftLeadSpeed"]),
                    "gap_m":     gap(msg["leftLeadGap"]),
                },
                "llag": {
                    "speed_mps": spd(msg["leftLagSpeed"]),
                    "gap_m":     gap(msg["leftLagGap"]),
                },
                "rlead": {
                    "speed_mps": spd(msg["rightLeadSpeed"]),
                    "gap_m":     gap(msg["rightLeadGap"]),
                },
                "rlag": {
                    "speed_mps": spd(msg["rightLagSpeed"]),
                    "gap_m":     gap(msg["rightLagGap"]),
                },
            },
            "cda": {
                "totalLane":  total_lane,
                "leftLCDir":  left_lc,
                "rightLCDir": right_lc,
            },
            "terminated": False,
            "truncated":  False,
            "time_s":     time_s,
        }
