"""
udp_sniff.py — Verify Aimsun/CDA is broadcasting UDP packets and decode them.

Run this BEFORE live_rollout.py (both can't bind the same port at once):
    python simulations/udp_sniff.py

Aimsun sends two packet types:
  START1 + bitsery(VehMessage) + MSGEND  — virtual car state (the one we care about)
  START7 + bitsery(HILMonitoringData) + MSGEND  — HIL heartbeat (ignore)

Key field to check:
  targetCAVID — must match EGO_ID in live_rollout.py (currently 1).
"""

import datetime
import socket
import struct

LOCAL_PORT = 7999

START1 = b"START1"   # VehMessage (virtual car) ← what we need
START4 = b"START4"   # Signal controller data (ignore)
START7 = b"START7"   # HILMonitoringData (heartbeat)
MSGEND = b"MSGEND"
HDR_LEN = 6
END_LEN = 6

# VehMessage bitsery layout (little-endian, after stripping START1 + MSGEND):
# Off  Size  Field
#   0    2   simID        (int16)
#   2    2   ID           (int16)
#   4    2   targetCAVID  (int16)  ← must match EGO_ID
#   6    2   leaderID     (int16)
#   8    2   year         (int16)
#  10    1   month        (int8)
#  11    1   day          (int8)
#  12    1   hour         (int8)
#  13    1   minute       (int8)
#  14    1   second       (int8)
#  15    2   ms           (int16)
#  17    4   simTime      (int32, 0.1 s from sim start)
#  21    4   speed        (int32, 0.001 m/s)
#  25    2   linkID       (int16)
#  27    4   linkPos      (int32, 0.001 m)
#  31    1   laneID       (int8)
MIN_VEHMSZ = 32


def decode_veh(payload: bytes) -> dict | None:
    if len(payload) < MIN_VEHMSZ:
        return None
    try:
        sim_id, veh_id, target_cav_id, leader_id = struct.unpack_from("<4h", payload, 0)
        hour, minute, second = struct.unpack_from("3b", payload, 12)
        ms, = struct.unpack_from("<h", payload, 15)
        sim_time, = struct.unpack_from("<i", payload, 17)
        speed, = struct.unpack_from("<i", payload, 21)
        link_id, = struct.unpack_from("<h", payload, 25)
        link_pos, = struct.unpack_from("<i", payload, 27)
        lane_id, = struct.unpack_from("b", payload, 31)
        return {
            "simID":       sim_id,
            "ID":          veh_id,
            "targetCAVID": target_cav_id,
            "leaderID":    leader_id,
            "speed_mps":   speed * 0.001,
            "linkPos_m":   link_pos * 0.001,
            "laneID":      lane_id,
            "linkID":      link_id,
            "sim_time_s":  sim_time * 0.1,
            "wall_time":   f"{hour:02d}:{minute:02d}:{second:02d}.{ms:03d}",
        }
    except struct.error:
        return None


def strip(data: bytes, header: bytes) -> bytes | None:
    """Strip known header and MSGEND; return payload or None if format wrong."""
    if not data.startswith(header):
        return None
    if data.endswith(MSGEND):
        return data[HDR_LEN:-END_LEN]
    return data[HDR_LEN:]  # tolerate missing ending


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", LOCAL_PORT))
sock.settimeout(2.0)
print(f"Listening on UDP :{LOCAL_PORT} — Ctrl-C to stop\n")
print(f"{'time':15}  {'bytes':>5}  {'type':8}  {'targetCAVID':>11}  {'ID':>6}  "
      f"{'spd m/s':>8}  {'pos m':>8}  {'lane':>4}  {'link':>6}  sim_t")
print("-" * 95)

while True:
    try:
        data, addr = sock.recvfrom(4096)
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")

        if data.startswith(START1):
            payload = strip(data, START1)
            d = decode_veh(payload) if payload else None
            if d:
                print(
                    f"{ts:15}  {len(data):5d}  VehMsg    "
                    f"{d['targetCAVID']:>11}  {d['ID']:>6}  "
                    f"{d['speed_mps']:>8.2f}  {d['linkPos_m']:>8.1f}  "
                    f"{d['laneID']:>4}  {d['linkID']:>6}  {d['sim_time_s']:.1f}s"
                )
            else:
                print(f"{ts:15}  {len(data):5d}  VehMsg    (too short to decode)")

        elif data.startswith(START4):
            pass  # signal controller state — not relevant, suppress

        elif data.startswith(START7):
            pass  # HIL heartbeat — suppress

        else:
            print(f"{ts:15}  {len(data):5d}  UNKNOWN   {data[:12].hex()}")

    except socket.timeout:
        print("  (no packet in 2 s)")
    except KeyboardInterrupt:
        print("\nStopped.")
        break

sock.close()
