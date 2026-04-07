"""
aapi_connector.py — Pure-Python connector that speaks the AAPI VehMessage protocol
directly via UDP sockets, bypassing CDA_Interface_Wrapper.dll entirely.

Protocol (VariableDef.h / AAPI.cxx):
  Outbound  Python → Aimsun port 8003:
      "START2" + bitsery(VehMessage) + "MSGEND"
      Python sends ego state each step so Aimsun places the placeholder vehicle.

  Inbound   Aimsun → Python port 7999:
      "START1" + bitsery(VehMessage) + "MSGEND"
      Aimsun replies with surrounding vehicle gaps/speeds for the ego.

Ego kinematics:
    Python integrates position and speed locally (pos += speed*dt, speed += accel*dt).
    Aimsun uses the sent position to place the placeholder and compute surrounding state.
    Lane is taken from Aimsun's reply (currentLane field) once a frame arrives;
    until then the last commanded wire lane is used.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any

from .live_connector_abstract import LiveConnector


# ── Wire constants ─────────────────────────────────────────────────────────────

_START1 = b"START1"   # Aimsun → Python  (virtual car / surrounding info)
_START2 = b"START2"   # Python → Aimsun  (ego state)
_MSGEND = b"MSGEND"
_HDR    = 6
_END    = 6

# bitsery serialisation order for VehMessage (little-endian, no padding).
# Matches VariableDef.h serialize(S&, VehMessage&).
#
#   simID(h) ID(h) targetCAVID(h) leaderID(h)
#   year(h) month(b) day(b) hour(b) minute(b) second(b) ms(h)
#   simTime(i)
#   speed(i) linkID(h) linkPos(i) laneID(b) nodeID(h) nextLinkID(h) stringPos(b)
#   leftLeadSpeed(h) leftLeadGap(i)
#   leftLagSpeed(h)  leftLagGap(i)
#   rightLeadSpeed(h) rightLeadGap(i)
#   rightLagSpeed(h)  rightLagGap(i)
#   leadSpeed(h) leadGap(i)
#   currentLane(b) totalLane(b)
_FMT  = "<hhhhhbbbbbhiihibhhbhihihihihibb"
_SIZE = struct.calcsize(_FMT)   # 69 bytes

_FIELDS = [
    "simID", "ID", "targetCAVID", "leaderID",
    "year", "month", "day", "hour", "minute", "second", "ms",
    "simTime",
    "speed", "linkID", "linkPos", "laneID", "nodeID", "nextLinkID", "stringPos",
    "leftLeadSpeed", "leftLeadGap",
    "leftLagSpeed",  "leftLagGap",
    "rightLeadSpeed","rightLeadGap",
    "rightLagSpeed", "rightLagGap",
    "leadSpeed",     "leadGap",
    "currentLane",   "totalLane",
]
_IDX = {name: i for i, name in enumerate(_FIELDS)}


def _pack(fields: dict) -> bytes:
    """Wrap a VehMessage dict in a START2 UDP packet."""
    g = fields.get
    vals = (
        g("simID",          0),
        g("ID",            -1),
        g("targetCAVID",   -1),
        g("leaderID",      -1),
        g("year",          -1),
        g("month",         -1),
        g("day",           -1),
        g("hour",          -1),
        g("minute",        -1),
        g("second",        -1),
        g("ms",             0),
        g("simTime",       -1),
        g("speed",         -1),   # int32, units: 0.001 m/s
        g("linkID",        -1),
        g("linkPos",       -1),   # int32, units: 0.001 m
        g("laneID",        -1),
        g("nodeID",        -1),
        g("nextLinkID",    -1),
        g("stringPos",      1),
        g("leftLeadSpeed", -1),
        g("leftLeadGap",   -1),
        g("leftLagSpeed",  -1),
        g("leftLagGap",    -1),
        g("rightLeadSpeed",-1),
        g("rightLeadGap",  -1),
        g("rightLagSpeed", -1),
        g("rightLagGap",   -1),
        g("leadSpeed",     -1),
        g("leadGap",       -1),
        g("currentLane",   -1),
        g("totalLane",     -1),
    )
    return _START2 + struct.pack(_FMT, *vals) + _MSGEND


def _unpack(data: bytes) -> dict | None:
    """Decode a START1 UDP packet into a VehMessage dict. Returns None on error."""
    if not data.startswith(_START1):
        return None
    payload = data[_HDR:-_END] if data.endswith(_MSGEND) else data[_HDR:]
    if len(payload) < _SIZE:
        return None
    try:
        vals = struct.unpack_from(_FMT, payload)
    except struct.error:
        return None
    return {name: vals[_IDX[name]] for name in _FIELDS}


class AapiDirectConnector(LiveConnector):
    """
    Drop-in replacement for CdaLiveConnector.
    Speaks VehMessage (START1/START2) directly — no DLL required.

    Config keys
    -----------
    remote_ip          str    Aimsun machine IP.              Default "127.0.0.1"
    remote_port        int    Aimsun receive port.            Default 8003
    local_port         int    Port Aimsun sends back to.      Default 7999
    ego_id             int    targetCAVID in outbound msgs.   Default 1
    link_id            int    Aimsun section ID for the ego.  REQUIRED — find in
                               Aimsun scenario (section properties panel).
    initial_pos_m      float  Starting position on section.   Default 100.0
    initial_speed_mps  float  Starting speed m/s.             Default 15.0
    initial_lane       int    Starting lane (1-based wire).   Default 1
    max_speed_mps      float  Speed ceiling m/s.              Default 40.0
    timeout_s          float  reset() wait timeout.           Default 10.0
    poll_interval_s    float  Reader thread recv timeout.     Default 0.001
    """

    AIMSUN_DT: float = 0.1   # Aimsun step period (seconds)

    def __init__(self, config: dict) -> None:
        self._remote_ip   = str(config.get("remote_ip",   "127.0.0.1"))
        self._remote_port = int(config.get("remote_port", 8003))
        self._local_port  = int(config.get("local_port",  7999))
        self._ego_id      = int(config.get("ego_id",      1))
        self._link_id      = int(config["link_id"])           # required
        self._next_link_id = int(config["next_link_id"])    # required — section downstream of link_id
        self._max_speed   = float(config.get("max_speed_mps",      40.0))
        self._timeout_s   = float(config.get("timeout_s",          10.0))
        self._poll_s      = float(config.get("poll_interval_s",    0.001))

        self._init_pos_m  = float(config.get("initial_pos_m",      100.0))
        self._init_speed  = float(config.get("initial_speed_mps",  15.0))
        self._init_lane   = int(config.get("initial_lane",          1))   # 1-based wire

        # Ego kinematics — Python-maintained, sent to Aimsun each step.
        self._pos_m       = self._init_pos_m
        self._speed_mps   = self._init_speed
        self._lane_wire   = self._init_lane
        self._prev_speed  = self._init_speed

        # Shared state between reader thread and policy thread.
        self._lock                = threading.Lock()
        self._cached_snapshot: dict | None = None
        self._first_frame_event   = threading.Event()
        self._last_sim_time: float = -1.0
        self._closed               = False

        # UDP sockets.
        self._send_addr = (self._remote_ip, self._remote_port)
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.bind(("0.0.0.0", self._local_port))
        self._recv_sock.settimeout(self._poll_s)

        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="AapiReader", daemon=True
        )
        self._reader_thread.start()

    # ── LiveConnector interface ────────────────────────────────────────────────

    def reset_episode(self, seed=None) -> dict[str, Any]:
        with self._lock:
            self._pos_m       = self._init_pos_m
            self._speed_mps   = self._init_speed
            self._lane_wire   = self._init_lane
            self._prev_speed  = self._init_speed
            self._cached_snapshot = None
            self._last_sim_time   = -1.0
        self._first_frame_event.clear()

        # Bootstrap: send ego state so Aimsun creates the placeholder vehicle.
        self._send_ego_state()

        if not self._first_frame_event.wait(timeout=self._timeout_s):
            raise TimeoutError(
                f"AapiDirectConnector: no START1 frame within {self._timeout_s}s. "
                "Check: (1) Aimsun simulation is running, "
                f"(2) a vehicle of test type is in the scenario, "
                f"(3) link_id={self._link_id} is correct."
            )

        with self._lock:
            snapshot = dict(self._cached_snapshot)
        snapshot["ego"]["acc_mps2"] = 0.0
        return snapshot

    def step(self, command: dict[str, Any]) -> dict[str, Any]:
        dt          = float(command.get("dt",           self.AIMSUN_DT))
        accel_mps2  = float(command.get("accel_mps2",  0.0))
        desired_lane = int(command.get("desired_lane_id", self._lane_wire - 1))

        with self._lock:
            prev = self._speed_mps
            new_speed = float(max(0.0, min(self._max_speed, prev + accel_mps2 * dt)))
            new_pos   = self._pos_m + prev * dt
            new_lane  = desired_lane + 1   # 0-based → 1-based wire

            self._prev_speed = prev
            self._speed_mps  = new_speed
            self._pos_m      = new_pos
            self._lane_wire  = new_lane

        acc = (new_speed - prev) / self.AIMSUN_DT

        self._send_ego_state()

        with self._lock:
            snap = dict(self._cached_snapshot) if self._cached_snapshot else {}
        if snap and "ego" in snap:
            snap["ego"] = dict(snap["ego"])
            snap["ego"]["acc_mps2"]  = acc
            snap["ego"]["speed_mps"] = new_speed
            snap["ego"]["pos_m"]     = new_pos
        return snap

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
            pos_raw   = int(self._pos_m     * 1000)   # 0.001 m
            speed_raw = int(self._speed_mps * 1000)   # 0.001 m/s
            lane      = self._lane_wire

        t = time.localtime()
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
            "simTime":     -1,
            "speed":       speed_raw,
            "linkID":      self._link_id,
            "linkPos":     pos_raw,
            "laneID":      lane,
            "nodeID":      -1,
            "nextLinkID":  self._next_link_id,
            "stringPos":   1,
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

            if not data.startswith(_START1):
                continue

            msg = _unpack(data)
            if msg is None:
                continue
            if msg["targetCAVID"] != self._ego_id:
                continue   # packet for a different test vehicle

            with self._lock:
                ego_pos   = self._pos_m
                ego_speed = self._speed_mps
                lane_wire = self._lane_wire

            snapshot = self._build_snapshot(msg, ego_pos, ego_speed, 0.0, lane_wire)

            sim_time = msg["simTime"] * 0.1
            with self._lock:
                self._cached_snapshot = snapshot
                if sim_time != self._last_sim_time:
                    self._last_sim_time = sim_time
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

        # Aimsun's currentLane is authoritative once we have a real frame.
        cur_lane_wire = int(msg["currentLane"])
        lane_idx = (cur_lane_wire - 1) if cur_lane_wire > 0 else max(0, lane_wire - 1)

        total_lane = max(1, int(msg["totalLane"])) if msg["totalLane"] > 0 else 4

        def spd(raw: int) -> float | None:
            return raw * 0.01 if raw > 0 else None   # 0.01 m/s scale

        def gap(raw: int) -> float | None:
            return raw * 0.1  if raw > 0 else None   # 0.1 m scale

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
                "leftLCDir":  2,   # not in VehMessage; default = through
                "rightLCDir": 2,
            },
            "terminated": False,
            "truncated":  False,
            "time_s":     msg["simTime"] * 0.1,
        }
