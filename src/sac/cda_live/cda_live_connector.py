"""
CdaLiveConnector — Thin transport adapter between the Python RL training loop
and the CDA DLL gateway.

Architecture:
    SAC policy
        ↓  action [accel_norm, intent]
    CdaLiveConnector          ← this file; owns the control loop
        ↓  CDA_SendMessage()
    CDA DLL                   ← thin gateway: send / receive / cache only
        ↓↑  V2X / UDP
    CDA / Aimsun runtime      ← authoritative simulator

DLL ownership model:
    CDA_Init() initialises implicit global state inside the DLL — it returns
    void, not a handle. All subsequent calls use that global state. Call
    CDA_Init once per process. CDA_Close tears it down.

Freshness detection:
    CDA_GetLatestVirVehTime(egoID) returns the timestamp of the last inbound
    UDP packet. We poll until that timestamp advances past our last-seen value
    (or becomes non-zero on reset). 0.0 is NOT treated as an error; it simply
    means no packet has arrived yet.

Lane index convention:
    DLL / wire  →  1-based
    Python      →  0-based
    Outbound:  _to_wire(lane_python)  = lane_python + 1
    Inbound:   _to_python(lane_wire)  = lane_wire   - 1

Neighbor data:
    CDA_GetCurVehLane returns the ego's current wire lane (1-based); converted
    to 0-based lane_idx, replacing the command-carried estimate.
    Left/right lead/lag speed and gap are exposed under "surrounding" using
    the same key names expected by HighwayShadowBridge.
"""

from __future__ import annotations

import ctypes
import threading
import time
from typing import Any

from .live_connector_abstract import LiveConnector


class CdaLiveConnector(LiveConnector):

    # Aimsun simulation step period — used for acc_mps2 finite-difference in
    # the reader thread.  Must match the Aimsun "step" setting (default 0.1 s).
    AIMSUN_DT: float = 0.1

    def __init__(self, config: dict) -> None:
        """
        Args (all read from *config* dict):
            dll_path          str    Absolute path to the CDA gateway DLL. Required.
            remote_ip         str    IP of the CDA / Aimsun runtime. Default "127.0.0.1".
            remote_port       int    Outbound UDP port (CDA_SendMessage).  Default 5555.
            local_port        int    Inbound UDP port  (CDA_Get*).         Default 5556.
            ego_id            int    Vehicle ID forwarded to all getters.  Default 1.
            timeout_s         float  Max seconds to wait for a fresh frame on reset. Default 5.0.
            poll_interval_s   float  Sleep between DLL polls in reader thread. Default 0.001.
            route_id          int    Forwarded to CDA_SetTestRouteID in reset_episode.
            debug_mode        bool   Forwarded to CDA_SetDebugMode in reset_episode.
        """
        self._ego_id     = int(config.get("ego_id", 1))
        self._timeout_s  = float(config.get("timeout_s", 5.0))
        self._poll_s     = float(config.get("poll_interval_s", 0.001))
        self._route_id   = config.get("route_id")
        self._debug_mode = config.get("debug_mode")

        # ── Shared state between reader thread and policy thread ───────────
        self._lock                = threading.Lock()
        self._cached_snapshot: dict[str, Any] | None = None
        self._first_frame_event   = threading.Event()  # set when first frame arrives
        self._last_vir_time: float = 0.0               # reader-thread-only
        self._prev_speed: float    = 0.0               # reader-thread-only
        self._closed               = False

        # ── Load DLL and bind signatures ──────────────────────────────────
        self._dll = ctypes.CDLL(config["dll_path"])
        self._bind_dll_signatures()

        remote_ip   = str(config.get("remote_ip", "127.0.0.1"))
        remote_port = int(config.get("remote_port", 5555))
        local_port  = int(config.get("local_port", 5556))
        self._dll.CDA_Init(remote_ip.encode(), ctypes.c_int(remote_port), ctypes.c_int(local_port))

        # ── Start background reader thread ────────────────────────────────
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="CdaDllReader", daemon=True
        )
        self._reader_thread.start()

    # ── LiveConnector interface ────────────────────────────────────────────

    def reset_episode(self, seed: int | None = None) -> dict[str, Any]:
        """
        Apply session-level DLL settings, then block until the first inbound
        UDP frame arrives (reader thread sets _first_frame_event).
        Returns the initial NormalizedSnapshot.
        """
        if self._route_id is not None:
            self._dll.CDA_SetTestRouteID(ctypes.c_int(int(self._route_id)))
        if self._debug_mode is not None:
            self._dll.CDA_SetDebugMode(ctypes.c_bool(bool(self._debug_mode)))

        # Signal the reader thread to reset its freshness cursor so it treats
        # the next packet as the start of a new episode.
        with self._lock:
            self._last_vir_time = 0.0
            self._prev_speed    = 0.0
            self._cached_snapshot = None
        self._first_frame_event.clear()

        # Block until the reader thread delivers the first frame.
        if not self._first_frame_event.wait(timeout=self._timeout_s):
            raise TimeoutError(
                f"CdaLiveConnector: no frame within {self._timeout_s}s during reset"
            )

        with self._lock:
            snapshot = dict(self._cached_snapshot)

        # acc_mps2 on the very first frame is meaningless (prev_speed was 0).
        snapshot["ego"]["acc_mps2"] = 0.0
        return snapshot

    def step(self, command: dict[str, Any]) -> dict[str, Any]:
        """
        1. Send ego command outbound via CDA_SendMessage.
        2. Return the latest snapshot from the reader thread (non-blocking).

        The reader thread runs at the Aimsun rate (10 Hz). The policy loop
        runs at 15 Hz.  On steps where Aimsun hasn't updated yet the previous
        frame is returned as-is — a stale hold, standard for rate-mismatched RL.
        """
        self._send_command(command)

        with self._lock:
            snapshot = dict(self._cached_snapshot) if self._cached_snapshot else {}

        return snapshot

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._reader_thread.join(timeout=2.0)
        self._dll.CDA_Close()

    # ── Outbound ──────────────────────────────────────────────────────────

    def _send_command(self, command: dict[str, Any]) -> None:
        """
        Transmit ego state to the CDA runtime via CDA_SendMessage.
        
        This is the outbound SAC to CDA broadcast to aimsun 

        vehPos and vehSpeed come from the *previous* snapshot (not from the
        command dict). desired_lane_id drives the outbound wire lane.
        """
        with self._lock:
            cached = self._cached_snapshot
        if cached is not None:
            ego       = cached["ego"]
            veh_pos   = float(ego["pos_m"])
            veh_speed = float(ego["speed_mps"])
        else:
            veh_pos   = 0.0
            veh_speed = 0.0

        desired_lane = int(command.get("desired_lane_id", 0))
        veh_lane_wire = self._to_wire(desired_lane)

        self._dll.CDA_SendMessage(
            ctypes.c_int(self._ego_id),
            ctypes.c_double(veh_pos),
            ctypes.c_double(veh_speed),
            ctypes.c_int(veh_lane_wire),
        )

    # ── Background reader thread ───────────────────────────────────────────

    def _reader_loop(self) -> None:
        """
        Runs in a daemon thread.  Polls the DLL at ~poll_interval_s and writes
        the latest snapshot to _cached_snapshot under _lock whenever Aimsun
        sends a new frame (LatestVirVehTime changes).

        acc_mps2 is computed with AIMSUN_DT (0.1 s) so the finite-difference
        reflects the actual Aimsun tick period, not the policy step period.
        """
        eid = ctypes.c_int(self._ego_id)
        while not self._closed:
            vir_time = float(self._dll.CDA_GetLatestVirVehTime(eid))
            if vir_time != 0.0 and vir_time != self._last_vir_time:
                self._last_vir_time = vir_time
                snapshot = self._build_snapshot(eid=eid, dt=self.AIMSUN_DT)
                with self._lock:
                    self._cached_snapshot = snapshot
                self._first_frame_event.set()
            time.sleep(self._poll_s)

    def _build_snapshot(
        # This is for inbound aimsun to python
        self,
        *,
        eid: ctypes.c_int,
        dt: float,
    ) -> dict[str, Any]:
        """
        Read all DLL getters and assemble the NormalizedSnapshot dict.
        Called only from the reader thread — _prev_speed is thread-local to it.

        acc_mps2 uses AIMSUN_DT so the finite-difference is correct.

        surrounding keys match what HighwayShadowBridge._synthesize_surrounding_neighbors
        expects: "front", "llead", "llag", "rlead", "rlag", each with
        "speed_mps" and "gap_m".  lc_dir values go into "cda" instead.
        """
        speed_mps = float(self._dll.CDA_GetVirVehSpeed(eid))
        pos_m     = float(self._dll.CDA_GetVirVehPos(eid))

        acc_mps2         = (speed_mps - self._prev_speed) / dt
        self._prev_speed = speed_mps

        time_s = float(self._dll.CDA_GetElapsedSec())

        cur_lane_wire = int(self._dll.CDA_GetCurVehLane(eid))
        lane_idx = self._to_python(cur_lane_wire) if cur_lane_wire > 0 else 0

        cda = {
            "signalState":  int(self._dll.CDA_GetSignalState(eid)),
            "sigEndTime":   float(self._dll.CDA_GetSigEndTime(eid)),
            "greenStart1":  float(self._dll.CDA_GetGreenStart1(eid)),
            "greenEnd1":    float(self._dll.CDA_GetGreenEnd1(eid)),
            "greenStart2":  float(self._dll.CDA_GetGreenStart2(eid)),
            "greenEnd2":    float(self._dll.CDA_GetGreenEnd2(eid)),
            "redStart1":    float(self._dll.CDA_GetRedStart1(eid)),
            "redStart2":    float(self._dll.CDA_GetRedStart2(eid)),
            "refAcc":       float(self._dll.CDA_GetRefAcc(eid)),
            "tpEndLoc":     float(self._dll.CDA_GetTPEndLoc(eid)),
            "latestTPTime": float(self._dll.CDA_GetLatestTPTime(eid)),
            "dist2Turn":    float(self._dll.CDA_GetDist2Turn(eid)),
            "turnDir":      int(self._dll.CDA_GetTurnDir(eid)),
            "laneSpec":     int(self._dll.CDA_GetLaneSpec(eid)),
            "totalLane":    int(self._dll.CDA_GetTotalLane(eid)),
            "leftLCDir":    int(self._dll.CDA_GetLeftLCDir(eid)),
            "rightLCDir":   int(self._dll.CDA_GetRightLCDir(eid)),
        }

        # Keys match HighwayShadowBridge._synthesize_surrounding_neighbors:
        #   "front"/llead/llag/rlead/rlag → {"speed_mps", "gap_m"}
        surrounding = {
            "front": {
                "speed_mps": float(self._dll.CDA_GetLeadSpeed(eid)),
                "gap_m":     float(self._dll.CDA_GetLeadGap(eid)),
            },
            "llead": {
                "speed_mps": float(self._dll.CDA_GetLeftLeadSpeed(eid)),
                "gap_m":     float(self._dll.CDA_GetLeftLeadGap(eid)),
            },
            "llag": {
                "speed_mps": float(self._dll.CDA_GetLeftLagSpeed(eid)),
                "gap_m":     float(self._dll.CDA_GetLeftLagGap(eid)),
            },
            "rlead": {
                "speed_mps": float(self._dll.CDA_GetRightLeadSpeed(eid)),
                "gap_m":     float(self._dll.CDA_GetRightLeadGap(eid)),
            },
            "rlag": {
                "speed_mps": float(self._dll.CDA_GetRightLagSpeed(eid)),
                "gap_m":     float(self._dll.CDA_GetRightLagGap(eid)),
            },
        }

        return {
            "ego": {
                "lane_idx":  lane_idx,
                "pos_m":     pos_m,
                "speed_mps": speed_mps,
                "acc_mps2":  acc_mps2,
                "crashed":   False,   # not available from wire
            },
            "terminated":  False,     # not available from wire
            "truncated":   False,
            "time_s":      time_s,
            "cda":         cda,
            "surrounding": surrounding,
        }

    # ── Lane index helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_wire(lane_python: int) -> int:
        """0-based Python lane → 1-based wire lane."""
        return lane_python + 1

    @staticmethod
    def _to_python(lane_wire: int) -> int:
        """1-based wire lane → 0-based Python lane."""
        return lane_wire - 1

    # ── DLL signature binding ──────────────────────────────────────────────

    def _bind_dll_signatures(self) -> None:
        """
        Declare argtypes / restype for every CDA DLL export.
        Must be called before any DLL function is invoked.
        """
        dll = self._dll
        _i  = ctypes.c_int
        _d  = ctypes.c_double

        # Lifecycle
        dll.CDA_Init.argtypes      = [ctypes.c_char_p, _i, _i]
        dll.CDA_Init.restype       = None

        dll.CDA_Close.argtypes     = []
        dll.CDA_Close.restype      = None

        # Outbound
        dll.CDA_SendMessage.argtypes = [_i, _d, _d, _i]
        dll.CDA_SendMessage.restype  = None

        # Configuration setters
        dll.CDA_SetTestRouteID.argtypes = [_i]
        dll.CDA_SetTestRouteID.restype  = None

        dll.CDA_SetDebugMode.argtypes = [ctypes.c_bool]
        dll.CDA_SetDebugMode.restype  = None

        # Ego vehicle getters  (egoID: int → double)
        for name in (
            "CDA_GetLatestVirVehTime",
            "CDA_GetVirVehSpeed",
            "CDA_GetVirVehPos",
            "CDA_GetLatestSigTime",
            "CDA_GetIntersectionLoc",
            "CDA_GetSigEndTime",
            "CDA_GetGreenStart1",
            "CDA_GetGreenEnd1",
            "CDA_GetGreenStart2",
            "CDA_GetGreenEnd2",
            "CDA_GetRedStart1",
            "CDA_GetRedStart2",
            "CDA_GetLatestTPTime",
            "CDA_GetRefAcc",
            "CDA_GetTPEndLoc",
            # Neighbor / surrounding getters (egoID: int → double)
            "CDA_GetLeftLeadSpeed",
            "CDA_GetLeftLeadGap",
            "CDA_GetLeftLagSpeed",
            "CDA_GetLeftLagGap",
            "CDA_GetRightLeadSpeed",
            "CDA_GetRightLeadGap",
            "CDA_GetRightLagSpeed",
            "CDA_GetRightLagGap",
            "CDA_GetLeadSpeed",
            "CDA_GetLeadGap",
            "CDA_GetDist2Turn",
        ):
            fn           = getattr(dll, name)
            fn.argtypes  = [_i]
            fn.restype   = _d

        # Integer getters (egoID: int → int)
        for name in (
            "CDA_GetSignalState",
            "CDA_GetLeftLCDir",
            "CDA_GetRightLCDir",
            "CDA_GetTurnDir",
            "CDA_GetLaneSpec",
            "CDA_GetCurVehLane",
            "CDA_GetTotalLane",
        ):
            fn           = getattr(dll, name)
            fn.argtypes  = [_i]
            fn.restype   = _i

        # Elapsed time: no egoID, returns float (not double)
        dll.CDA_GetElapsedSec.argtypes = []
        dll.CDA_GetElapsedSec.restype  = ctypes.c_float
