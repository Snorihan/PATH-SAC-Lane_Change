"""
dll_probe.py — Poll the CDA DLL directly to verify Aimsun is sending frames
and that the ego vehicle ID is correct.

Run from src/sac/ (does NOT conflict with live_rollout.py on the port):
    python simulations/dll_probe.py

What to look for:
  vir_time keeps incrementing  [NEW FRAME]   → Aimsun running, DLL receiving OK
  vir_time=1.0000              [stale    ]   → one packet arrived then stopped;
                                               Aimsun paused or CDA plugin crashed
  vir_time=0.0000              [stale    ]   → no packet ever; wrong port/IP or
                                               CDA plugin not started
  speed/pos stuck at ~-0.001  even on NEW FRAME → frames arriving but EGO_ID
                                               doesn't match Aimsun vehicle ID
"""

import ctypes
import time

# ── CONFIG — must match live_rollout.py ──────────────────────────────────────
DLL_PATH    = r"C:\Users\janus\Desktop\BerkeleyFileMasterDirectory\PATH\Vehicle_Test_Interface\out\build\Release\Debug\CDA_Interface_Wrapper.dll"
REMOTE_IP   = "127.0.0.1"
REMOTE_PORT = 5555
LOCAL_PORT  = 5556
EGO_ID      = 1
POLL_S      = 0.5   # seconds between polls
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print(f"[dll_probe] Loading DLL: {DLL_PATH}")
    dll = ctypes.CDLL(DLL_PATH)

    dll.CDA_Init.argtypes               = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    dll.CDA_Init.restype                = None
    dll.CDA_Close.argtypes              = []
    dll.CDA_Close.restype               = None
    dll.CDA_GetLatestVirVehTime.argtypes = [ctypes.c_int]
    dll.CDA_GetLatestVirVehTime.restype  = ctypes.c_double
    dll.CDA_GetVirVehSpeed.argtypes      = [ctypes.c_int]
    dll.CDA_GetVirVehSpeed.restype       = ctypes.c_double
    dll.CDA_GetVirVehPos.argtypes        = [ctypes.c_int]
    dll.CDA_GetVirVehPos.restype         = ctypes.c_double
    dll.CDA_GetCurVehLane.argtypes       = [ctypes.c_int]
    dll.CDA_GetCurVehLane.restype        = ctypes.c_int
    dll.CDA_GetElapsedSec.argtypes       = []
    dll.CDA_GetElapsedSec.restype        = ctypes.c_float

    dll.CDA_Init(
        REMOTE_IP.encode(),
        ctypes.c_int(REMOTE_PORT),
        ctypes.c_int(LOCAL_PORT),
    )
    print(f"[dll_probe] CDA_Init done  (ego_id={EGO_ID})  — polling every {POLL_S}s.  Ctrl-C to stop.\n")
    print(f"{'vir_time':>10}  {'status':12}  {'elapsed_s':>10}  {'lane_wire':>9}  {'speed_mps':>10}  {'pos_m':>10}")
    print("-" * 70)

    eid    = ctypes.c_int(EGO_ID)
    prev_t = None
    try:
        while True:
            vir_time  = dll.CDA_GetLatestVirVehTime(eid)
            elapsed   = dll.CDA_GetElapsedSec()
            lane_wire = dll.CDA_GetCurVehLane(eid)
            speed     = dll.CDA_GetVirVehSpeed(eid)
            pos       = dll.CDA_GetVirVehPos(eid)

            status = "NEW FRAME" if vir_time != prev_t else "stale    "
            print(
                f"{vir_time:10.4f}  [{status}]  "
                f"{elapsed:10.3f}  {lane_wire:9d}  {speed:10.4f}  {pos:10.4f}"
            )
            prev_t = vir_time
            time.sleep(POLL_S)

    except KeyboardInterrupt:
        print("\n[dll_probe] Stopped.")
    finally:
        dll.CDA_Close()
        print("[dll_probe] CDA_Close called.")


if __name__ == "__main__":
    main()
