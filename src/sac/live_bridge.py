from __future__ import annotations

from typing import Any

import numpy as np
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


class LiveConnector:
    """Minimal interface for a live simulator bridge."""

    def reset_episode(self, seed: int | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, command: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HighwayShadowBridge:
    """Translate live backend snapshots into HighwayEnv shadow state."""

    def __init__(self, env: Any):
        self.env = env

    def require_connector(self) -> LiveConnector:
        connector = getattr(self.env, "connector", None)
        if connector is None:
            raise RuntimeError(
                "backend='aimsun_live' requires config['live_connector'] or "
                "config['live_connector_factory']"
            )
        return connector

    def decode_action_command(self, action) -> dict[str, Any]:
        env = self.env
        accel_norm, intent = float(action[0]), float(action[1])
        vehicle: ControlledVehicle = env.vehicle
        if vehicle is None:
            return {
                "raw_action": np.asarray(action, dtype=np.float32),
                "accel_norm": accel_norm,
                "intent": intent,
                "accel_mps2": 0.0,
                "desired_lane_id": 0,
                "target_lane_index": ("0", "1", 0),
            }

        deadzone = 0.2
        from_n, to_n, lane_id = vehicle.lane_index
        try:
            lanes_on_segment = len(vehicle.road.network.graph[from_n][to_n])
        except Exception:
            lanes_on_segment = int(env.config.get("lanes_count", 4))

        if abs(intent) < deadzone:
            desired_lane_id = int(lane_id)
        else:
            target = getattr(env, "target_lane_index", vehicle.lane_index)
            desired_lane_id = int(target[2]) if isinstance(target, tuple) else int(target)

        desired_lane_id = int(np.clip(desired_lane_id, 0, lanes_on_segment - 1))
        dt = 1.0 / float(env.config.get("policy_frequency", 15))
        max_accel = 3.0
        accel_mps2 = float(np.clip(accel_norm, -1.0, 1.0) * max_accel)
        return {
            "raw_action": np.asarray(action, dtype=np.float32),
            "accel_norm": accel_norm,
            "intent": intent,
            "accel_mps2": accel_mps2,
            "dt": dt,
            "desired_lane_id": desired_lane_id,
            "target_lane_index": (from_n, to_n, desired_lane_id),
        }

    def sync_shadow_state(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        env = self.env
        normalized = self._normalize_snapshot(snapshot)
        ego = normalized["ego"]

        lane_index = ("0", "1", int(ego["lane_idx"]))
        lane = env.road.network.get_lane(lane_index)
        lane_length = float(getattr(lane, "length", 1e9))
        pos_m = float(np.clip(float(ego["pos_m"]), 0.0, lane_length - 1e-3))
        speed_mps = float(ego["speed_mps"])
        acc_mps2 = float(ego.get("acc_mps2", 0.0))

        env.vehicle.position = lane.position(pos_m, 0.0)
        env.vehicle.speed = speed_mps
        env.vehicle.lane_index = lane_index
        env.vehicle.lane = lane
        env.vehicle.target_lane_index = lane_index
        env.vehicle.target_speed = speed_mps
        env.vehicle.action = {"acceleration": acc_mps2}
        env.vehicle.crashed = bool(ego.get("crashed", False))

        shadow_vehicles = [env.vehicle]
        explicit_neighbors = normalized.get("neighbors", []) or []
        if explicit_neighbors:
            for neighbor in explicit_neighbors:
                neighbor_lane_idx = int(neighbor.get("lane_idx", max(0, int(neighbor.get("lane", 1)) - 1)))
                neighbor_pos_m = float(neighbor.get("pos_m", pos_m))
                neighbor_speed = float(neighbor.get("speed_mps", neighbor.get("speed", 0.0)))
                vehicle = self._make_shadow_vehicle(neighbor_lane_idx, neighbor_pos_m, neighbor_speed)
                if vehicle is not None:
                    shadow_vehicles.append(vehicle)
        else:
            shadow_vehicles.extend(self._synthesize_surrounding_neighbors(normalized))

        env.road.vehicles = shadow_vehicles

        snapshot_time = normalized.get("time_s")
        if snapshot_time is None:
            snapshot_time = float(getattr(env, "time", 0.0)) + 1.0 / float(env.config.get("policy_frequency", 15))
        env.time = float(snapshot_time)
        env.last_snapshot = normalized
        return normalized

    def _extract_snapshot_time(self, snapshot: dict[str, Any]) -> float | None:
        if snapshot.get("time_s") is not None:
            return float(snapshot["time_s"])
        if snapshot.get("sim_time_s") is not None:
            return float(snapshot["sim_time_s"])
        if all(key in snapshot for key in ("hour", "min", "sec", "ms")):
            return (
                int(snapshot["hour"]) * 3600
                + int(snapshot["min"]) * 60
                + int(snapshot["sec"])
                + int(snapshot["ms"]) * 0.001
            )
        return None

    def _normalize_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        env = self.env
        default_acc = float((env.last_action_command or {}).get("accel_mps2", 0.0))
        if "ego" in snapshot:
            normalized = dict(snapshot)
            normalized["ego"] = dict(snapshot["ego"])
            normalized["ego"].setdefault("acc_mps2", default_acc)
            normalized.setdefault("time_s", self._extract_snapshot_time(snapshot))
            normalized.setdefault("terminated", False)
            normalized.setdefault("truncated", False)
            return normalized

        lane_idx = max(0, int(snapshot.get("lane", 1)) - 1)
        normalized = {
            "sim_step": int(snapshot.get("simStep", -1)),
            "time_s": self._extract_snapshot_time(snapshot),
            "ego": {
                "veh_id": int(snapshot.get("vehID", -1)),
                "lane_idx": lane_idx,
                "pos_m": float(snapshot.get("pos_m", 0.0)),
                "speed_mps": float(snapshot.get("speed_mps", snapshot.get("speed", 0.0))),
                "acc_mps2": default_acc,
                "crashed": bool(snapshot.get("crashed", False)),
            },
            "surrounding": {
                "front": {
                    "speed_mps": snapshot.get("frontV"),
                    "gap_m": snapshot.get("frontGap"),
                },
                "rear": {
                    "speed_mps": snapshot.get("rearV"),
                    "gap_m": snapshot.get("rearGap"),
                },
                "llead": {
                    "speed_mps": snapshot.get("lleadV"),
                    "gap_m": snapshot.get("lleadGap"),
                },
                "llag": {
                    "speed_mps": snapshot.get("llagV"),
                    "gap_m": snapshot.get("llagGap"),
                },
                "rlead": {
                    "speed_mps": snapshot.get("rleadV"),
                    "gap_m": snapshot.get("rleadGap"),
                },
                "rlag": {
                    "speed_mps": snapshot.get("rlagV"),
                    "gap_m": snapshot.get("rlagGap"),
                },
            },
            "terminated": bool(snapshot.get("terminated", False)),
            "truncated": bool(snapshot.get("truncated", False)),
        }
        if "neighbors" in snapshot:
            normalized["neighbors"] = snapshot["neighbors"]
        if "cda" in snapshot:
            normalized["cda"] = snapshot["cda"]
        return normalized

    def _make_shadow_vehicle(self, lane_idx: int, pos_m: float, speed_mps: float) -> Vehicle | None:
        env = self.env
        lanes = int(env.config.get("lanes_count", 4))
        if lane_idx < 0 or lane_idx >= lanes:
            return None

        lane_index = ("0", "1", int(lane_idx))
        lane = env.road.network.get_lane(lane_index)
        lane_length = float(getattr(lane, "length", 1e9))
        clipped_pos = float(np.clip(pos_m, 0.0, lane_length - 1e-3))
        heading = float(lane.heading_at(clipped_pos)) if hasattr(lane, "heading_at") else 0.0

        vehicle = Vehicle(env.road, lane.position(clipped_pos, 0.0), heading=heading, speed=float(speed_mps))
        vehicle.lane_index = lane_index
        vehicle.lane = lane
        return vehicle

    def _synthesize_surrounding_neighbors(self, normalized: dict[str, Any]) -> list[Vehicle]:
        ego = normalized["ego"]
        lane_idx = int(ego["lane_idx"])
        pos_m = float(ego["pos_m"])
        surrounding = normalized.get("surrounding", {}) or {}

        specs = [
            ("front", lane_idx, +1.0),
            ("rear", lane_idx, -1.0),
            ("llead", lane_idx - 1, +1.0),
            ("llag", lane_idx - 1, -1.0),
            ("rlead", lane_idx + 1, +1.0),
            ("rlag", lane_idx + 1, -1.0),
        ]

        neighbors: list[Vehicle] = []
        for key, neighbor_lane_idx, direction in specs:
            data = surrounding.get(key) or {}
            gap_m = data.get("gap_m")
            speed_mps = data.get("speed_mps")
            if gap_m is None or speed_mps is None:
                continue

            gap_m = float(gap_m)
            speed_mps = float(speed_mps)
            if not np.isfinite(gap_m) or not np.isfinite(speed_mps) or gap_m <= 0.0:
                continue

            vehicle = self._make_shadow_vehicle(
                neighbor_lane_idx,
                pos_m + direction * gap_m,
                speed_mps,
            )
            if vehicle is not None:
                neighbors.append(vehicle)

        return neighbors
