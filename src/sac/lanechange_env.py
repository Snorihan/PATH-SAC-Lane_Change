import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import highway_env
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable
from shadow_reader.live_bridge import HighwayShadowBridge, LiveConnector
import numpy as np
import pprint
import wrappers
from highway_env.envs.common.action import ActionType
import datalogger_reader


# ── Reward term descriptor ────────────────────────────────────────────────────

@dataclass
class RewardTerm:
    name: str
    fn: Callable[[], float]  # zero-arg callable, returns raw signal
    weight: float
    sign: float = 1.0        # +1 to add, -1 to subtract


# ── Surrounding vehicle snapshot ──────────────────────────────────────────────

@dataclass
class SurroundingVehicles:
    """
    Mirrors the Aimsun 29-col trajectory format:
      front (same lane), left lead/lag, right lead/lag.
    All gaps are positive distances in metres; None means no vehicle in range.
    """
    front_v:    object  = None;  front_gap:  float = float("inf")
    rear_v:     object  = None;  rear_gap:   float = float("inf")
    llead_v:    object  = None;  llead_gap:  float = float("inf")
    llag_v:     object  = None;  llag_gap:   float = float("inf")
    rlead_v:    object  = None;  rlead_gap:  float = float("inf")
    rlag_v:     object  = None;  rlag_gap:   float = float("inf")


# ── Scenario source abstraction ───────────────────────────────────────────────

class ScenarioSource:
    """Interface for supplying initial ego states.
    Swap in AimsunLiveSource tomorrow without touching reset()."""

    def sample_initial_state(self, _rng) -> dict:
        """Return dict with keys: lane_idx, pos_m, speed, simStep, vehID."""
        raise NotImplementedError


class CsvScenarioSource(ScenarioSource):
    def __init__(self, path: str):
        self.rows = datalogger_reader.load_rows(path)

    def sample_initial_state(self, rng) -> dict:
        row = rng.choice(self.rows)
        return {
            "lane_idx": max(0, int(row["lane"]) - 1),  # Aimsun 1-based → 0-based
            "pos_m":    float(row["pos_m"]),
            "speed":    float(row["speed_mps"]),
            "simStep":  int(row.get("simStep", -1)),
            "vehID":    int(row.get("vehID", -1)),
        }


# ── Action type ───────────────────────────────────────────────────────────────

# LiveConnector is the canonical interface defined in live_bridge.py and
# imported above. Do not redefine it here.

class IntentContinuousAction(ActionType):
    vehicle_class = ControlledVehicle
    """
    Action = [accel_norm, intent]
    accel_norm in [-1,1] -> m/s^2
    intent in [-1,1] -> lane selection logic in _apply_action()
    """
    def space(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

    def act(self, action):
        self.env._send_action(action)   


# ── Env registration (guarded) ────────────────────────────────────────────────

if "lane-changing-v0" not in registry:
    register(
        id="lane-changing-v0",
        entry_point="lanechange_env:LaneChangingEnv",
    )


# ── Environment ───────────────────────────────────────────────────────────────

class LaneChangingEnv(HighwayEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(rewards = {
            # Safety
            "collision_penalty": -500.0,
            "ttc_weight": 80.0,
            "gap_weight": 40.0,
            "closing_speed_weight": 15.0,

            # Lane change task
            "lane_success": 80.0,
            "lane_progress_weight": 20.0,
            "lane_start_bonus": 5.0,
            "lane_keeping_penalty_when_requested": 0.2,
            "wrong_lane_penalty": 5.0,

            # Comfort
            "jerk_weight": 2.0,
            "lat_accel_weight": 1.0,

            # Rules / efficiency
            "speed_limit_weight": 1.0,
            "time_penalty": 0.05,
        })
        return config

    def __init__(self, config=None, render_mode=None):
        super().__init__(config=config, render_mode=render_mode)
        self.backend = self.config.get("backend", "csv")

        dataset_path = self.config.get("scenario_csv", None)
        self.source: ScenarioSource | None = (
            CsvScenarioSource(dataset_path) if dataset_path else None
        )

        connector_factory = self.config.get("live_connector_factory", None)
        connector = self.config.get("live_connector", None)
        if connector is None and callable(connector_factory):
            connector = connector_factory()

        self.connector: LiveConnector | None = connector
        self.last_snapshot: dict[str, Any] | None = None
        self.last_action_command: dict[str, Any] | None = None
        self.live_bridge = HighwayShadowBridge(self)

    def define_spaces(self) -> None:
        super().define_spaces()
        self.action_type  = IntentContinuousAction(self)
        self.action_space = self.action_type.space()

    def _create_road(self) -> None:
        net = RoadNetwork()
        lanes       = int(self.config.get("lanes_count",  4))
        lane_width  = float(self.config.get("lane_width", 4.0))
        road_length = float(self.config.get("road_length", 1000.0))
        from_n, to_n = "0", "1"

        for lane_id in range(lanes):
            origin = [0.0, lane_id * lane_width]
            end    = [road_length, lane_id * lane_width]
            left_type  = LineType.CONTINUOUS if lane_id == 0         else LineType.STRIPED
            right_type = LineType.CONTINUOUS if lane_id == lanes - 1 else LineType.STRIPED
            net.add_lane(from_n, to_n,
                StraightLane(origin, end, width=lane_width,
                             line_types=[left_type, right_type]))

        self.road = Road(network=net, np_random=self.np_random)

    # ── Action routing ────────────────────────────────────────────────────────

    def _observe_env(self):
        return self.observation_type.observe()

    def _reset_episode_state(self) -> None:
        self.lane_changing = False
        self.prev_acceleration = 0.0
        self.prev_lat = 0.0
        self.steps_in_target_lane = 0
        self.elapsed_steps = 0
        self.last_jerk_value = 0.0
        self.prev_reward_time = float(getattr(self, "time", 0.0))
        self.started_lane_change = False

        self.start_lane_index = self.vehicle.lane_index
        self.ultimate_target_lane_index = self.find_target_lane(self.start_lane_index)
        self.target_lane_index = self.ultimate_target_lane_index

    def _send_action(self, action):
        """Hook: swap in connector.step() for the live backend."""
        if self.backend == "aimsun_live":
            self.last_action_command = self.live_bridge.decode_action_command(action)
            return
        self._apply_action(action)

    def _apply_action(self, action):
        command = self.live_bridge.decode_action_command(action)
        vehicle: ControlledVehicle = self.vehicle
        if vehicle is None:
            return

        vehicle.target_lane_index = command["target_lane_index"]
        vehicle.action = {"acceleration": command["accel_mps2"]}
        vehicle.target_speed = float(np.clip(vehicle.speed + command["accel_mps2"] * command["dt"], 0.0, 40.0))
        self.last_action_command = command

    # -- reward -----------------------------------------------------------

    def _reward(self, _action):
        w = self.config["rewards"]
        veh_action = getattr(self.vehicle, "action", {}) or {}
        curr_acc   = float(veh_action.get("acceleration", 0.0))

        terms = [
            # Safety: sparse collision penalty (paper eq.1)
            RewardTerm(
                name   = "collision",
                fn     = lambda: 1.0 if self.vehicle.crashed else 0.0,
                weight = w["collision_penalty"],
                sign   = +1.0,
            ),
            # Safety: dense TTC penalty — 1/TTC for each vehicle within threshold (paper eq.3)
            RewardTerm(
                name   = "ttc",
                fn     = self._r_fn_ttc,
                weight = w["ttc_weight"],
                sign   = -1.0,
            ),
            # Comfort: penalize jerk; small reward for smooth driving
            RewardTerm(
                name   = "jerk",
                fn     = lambda: max(0.0, (self._reward_funct_jerk(self.time, curr_acc)) - 0.5),
                weight = w["jerk_weight"],
                sign   = -1.0,
            ),
            # Rules: reward speed up to limit, penalize exceeding it
            RewardTerm(
                name   = "speed_limit",
                fn     = self._r_fn_matching_speed_limit,
                weight = w["speed_limit_weight"],
                sign   = +1.0,
            ),
            # Progress: lateral progress toward target lane + one-time start bonus
            RewardTerm(
                name   = "lane_start",
                fn     = self._r_fn_lane_start,
                weight = w["lane_start_bonus"],
                sign   = +1.0,
            ),
            RewardTerm(
                name   = "lc_progress",
                fn     = self._r_fn_lc_progress,
                weight = w["lane_progress_weight"],
                sign   = +1.0,
            ),
            # Progress: sparse reward for reaching target lane
            RewardTerm(
                name   = "lane_success",
                fn     = lambda: self._r_fn_lane_change_success(self.vehicle.lane_index),
                weight = w["lane_success"],
                sign   = +1.0,
            ),
            # Efficiency: small per-step penalty to discourage stalling
            RewardTerm(
                name   = "time_penalty",
                fn     = lambda: 1.0,
                weight = w["time_penalty"],
                sign   = -1.0,
            ),
        ]

        reward = 0.0
        self.reward_dict = {}
        for term in terms:
            value = term.sign * term.fn() * term.weight
            reward += value
            self.reward_dict[term.name] = value

        self.reward_dict["raw_jerk"] = self.last_jerk_value
        return reward

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        if self.backend != "aimsun_live":
            obs, reward, terminated, truncated, info = super().step(action)
            self.elapsed_steps += 1

            lane = self.road.network.get_lane(self.vehicle.lane_index)
            _, lat = lane.local_coordinates(self.vehicle.position)
            lat_threshold = float(self.config.get("lane_change_lat_threshold", 0.5))
            self.lane_changing = (abs(lat) > lat_threshold)

            terminated = terminated or self._check_lane_change_termination()

            info["reward_terms"] = dict(self.reward_dict)
            info["lane_change_state"] = {
                "lane_changing": self.lane_changing,
                "steps_in_target_lane": self.steps_in_target_lane,
                "target_lane_index": self.target_lane_index,
            }
            return obs, reward, terminated, truncated, info

        command = self.live_bridge.decode_action_command(action)
        self.last_action_command = command
        snapshot = self.live_bridge.require_connector().step(command)
        normalized = self.live_bridge.sync_shadow_state(snapshot)
        self.elapsed_steps += 1

        lane = self.road.network.get_lane(self.vehicle.lane_index)
        _, lat = lane.local_coordinates(self.vehicle.position)
        lat_threshold = float(self.config.get("lane_change_lat_threshold", 0.5))
        self.lane_changing = (abs(lat) > lat_threshold)

        reward = self._reward(action)
        terminated = bool(normalized.get("terminated", False) or self.vehicle.crashed)
        max_steps = int(self.config.get("duration", 40) * self.config.get("policy_frequency", 15))
        truncated = bool(normalized.get("truncated", False)) or (self.elapsed_steps >= max_steps)
        terminated = terminated or self._check_lane_change_termination()

        obs = self._observe_env()
        info = {
            "backend": "aimsun_live",
            "reward_terms": dict(self.reward_dict),
            "lane_change_state": {
                "lane_changing": self.lane_changing,
                "steps_in_target_lane": self.steps_in_target_lane,
                "target_lane_index": self.target_lane_index,
            },
            "snapshot": normalized,
        }
        if "cda" in normalized:
            info["cda"] = normalized["cda"]
        return obs, reward, terminated, truncated, info

    def _check_lane_change_termination(self) -> bool:
        if self._same_lane(self.vehicle.lane_index, self.target_lane_index):
            self.steps_in_target_lane += 1
        else:
            self.steps_in_target_lane = 0
        return self.steps_in_target_lane >= self.config.get("duration_after_lane_change", 40)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.prev_lane_progress = 0.0
        self.got_lane_success = False

        if getattr(self, "backend", "csv") == "aimsun_live":
            snapshot = self.live_bridge.require_connector().reset_episode(seed=kwargs.get("seed"))
            normalized = self.live_bridge.sync_shadow_state(snapshot)
            self._reset_episode_state()
            obs = self._observe_env()
            info["backend"] = "aimsun_live"
            info["snapshot"] = normalized
            if "cda" in normalized:
                info["cda"] = normalized["cda"]
            return obs, info

        scenario = None
        source = getattr(self, "source", None)
        if source is not None:
            scenario = source.sample_initial_state(self.np_random)

        self._apply_initial_state(scenario, info)
        self._reset_episode_state()
        obs = self._observe_env()

        return obs, info

    def _apply_initial_state(self, scenario: dict | None, info: dict) -> None:
        """Place ego vehicle from a scenario dict. No-op if scenario is None."""
        if scenario is None:
            info["scenario"] = None
            return

        from_n, to_n = "0", "1"
        lane_idx   = int(scenario["lane_idx"])
        lane_index = (from_n, to_n, lane_idx)
        lane       = self.road.network.get_lane(lane_index)

        pos_m = float(np.clip(
            float(scenario["pos_m"]),
            0.0,
            float(getattr(lane, "length", 1e9)) - 1e-3,
        ))
        speed = float(scenario["speed"])

        self.vehicle.position          = lane.position(pos_m, 0.0)
        self.vehicle.speed             = speed
        self.vehicle.lane_index        = lane_index
        self.vehicle.lane              = lane
        self.vehicle.target_lane_index = lane_index
        self.vehicle.target_speed      = speed

        info["scenario"] = {
            "simStep":  int(scenario.get("simStep", -1)),
            "vehID":    int(scenario.get("vehID",   -1)),
            "lane_idx": lane_idx,
            "pos_m":    pos_m,
            "speed":    speed,
        }

    # ── Surrounding vehicle queries ───────────────────────────────────────────

    def _get_surrounding_vehicles(self, max_range: float = 200.0) -> SurroundingVehicles:
        """
        Query front/rear in current lane plus lead/lag in left and right lanes.
        Mirrors the Aimsun lleadV/Gap, llagV/Gap, rleadV/Gap, rlagV/Gap columns.
        """
        sv = SurroundingVehicles()
        from_n, to_n, lane_id = self.vehicle.lane_index

        try:
            n_lanes = len(self.road.network.graph[from_n][to_n])
        except Exception:
            n_lanes = int(self.config.get("lanes_count", 4))

        # current lane: front + rear
        f, r, fg, rg = self._vehicle_in_lane(self.vehicle.lane_index, max_range)
        sv.front_v, sv.front_gap = f, fg
        sv.rear_v,  sv.rear_gap  = r, abs(rg)

        # left lane (lane_id - 1)
        if lane_id - 1 >= 0:
            left_index = (from_n, to_n, lane_id - 1)
            lf, lr, lfg, lrg = self._vehicle_in_lane(left_index, max_range)
            sv.llead_v, sv.llead_gap = lf, lfg
            sv.llag_v,  sv.llag_gap  = lr, abs(lrg)

        # right lane (lane_id + 1)
        if lane_id + 1 < n_lanes:
            right_index = (from_n, to_n, lane_id + 1)
            rf, rr, rfg, rrg = self._vehicle_in_lane(right_index, max_range)
            sv.rlead_v, sv.rlead_gap = rf, rfg
            sv.rlag_v,  sv.rlag_gap  = rr, abs(rrg)

        return sv

    def _vehicle_in_lane(self, lane_index, max_range: float = 200.0):
        """Return (front_v, rear_v, gap_front, gap_rear) for a given lane index."""
        ego  = self.vehicle
        lane = self.road.network.get_lane(lane_index)
        ego_s, _ = lane.local_coordinates(ego.position)

        gap_front, gap_rear = float(np.inf), -float(np.inf)
        front_vehicle, rear_vehicle = None, None

        for v in self.road.vehicles:
            if v is ego or v.lane_index != lane_index:
                continue
            veh_s, _ = self.road.network.get_lane(v.lane_index).local_coordinates(v.position)
            gap = veh_s - ego_s
            if 0.0 < gap < gap_front and gap <= max_range:
                gap_front, front_vehicle = gap, v
            if 0.0 > gap > gap_rear and -gap <= max_range:
                gap_rear, rear_vehicle = gap, v

        return front_vehicle, rear_vehicle, gap_front, gap_rear

    def _vehicle_in_front_rear(self, lane_index, max_range: float = 200.0):
        """Convenience wrapper kept for backward compatibility."""
        f, r, fg, rg = self._vehicle_in_lane(lane_index, max_range)
        return f, r, fg, rg

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _same_lane(self, a, b) -> bool:
        return tuple(a) == tuple(b)

    def find_target_lane(self, start_lane_index):
        """Temporary: pick adjacent lane. A separate network will handle this later."""
        from_n, to_n, lane_id = start_lane_index
        lanes  = int(self.config.get("lanes_count", 4))
        target = lane_id + 1 if lane_id + 1 < lanes else lane_id - 1
        return (from_n, to_n, int(np.clip(target, 0, lanes - 1)))

    # ── Reward functions ──────────────────────────────────────────────────────

    def _reward_funct_jerk(self, curr_time, curr_acc):
        time_dt = curr_time - self.prev_reward_time
        if time_dt <= 1e-6:
            self.last_jerk_value = 0.0
            return 0.0
        jerk = (curr_acc - self.prev_acceleration) / time_dt
        self.last_jerk_value   = jerk
        self.prev_reward_time  = curr_time
        self.prev_acceleration = curr_acc
        return jerk

    def _r_fn_ttc(self, ttc_threshold: float = 3.0) -> float:
        """Dense safety penalty: bounded risk of 1/TTC for each neighbor within threshold."""
        sv      = self._get_surrounding_vehicles()
        ego_spd = float(self.vehicle.speed)
        r       = 0.0
        # (gap_m, vehicle, ego_is_approaching)
        pairs = [
            (sv.front_gap,  sv.front_v,  True),
            (sv.rear_gap,   sv.rear_v,   False),
            (sv.llead_gap,  sv.llead_v,  True),
            (sv.llag_gap,   sv.llag_v,   False),
            (sv.rlead_gap,  sv.rlead_v,  True),
            (sv.rlag_gap,   sv.rlag_v,   False),
        ]
        for gap, veh, ego_closing in pairs:
            if veh is None or gap >= float("inf"):
                continue
            v_spd   = float(veh.speed)
            closing = (ego_spd - v_spd) if ego_closing else (v_spd - ego_spd)
            if closing <= 0.0:
                continue
            ttc = gap / closing
            if ttc < ttc_threshold:
                risk = ((ttc_threshold - ttc) / ttc_threshold) ** 2
                r = max(r, risk)
        return r

    def _r_fn_matching_speed_limit(self):
        speed_limit = self.config.get("speed_limit", None)
        if speed_limit is None:
            return 0.0
        ego_speed   = float(self.vehicle.speed)
        speed_limit = float(speed_limit)
        if ego_speed <= speed_limit:
            return ego_speed / max(speed_limit, 1e-6)
        return -(ego_speed - speed_limit)

    def _r_fn_lane_change_success(self, curr_lane_index):
        if self._same_lane(curr_lane_index, self.target_lane_index) and not self.got_lane_success:
            self.got_lane_success = True
            return 1
        else:
            return 0

    def _r_fn_lc_progress(self) -> float:
        _, _, curr_id = self.vehicle.lane_index
        _, _, target_id = self.target_lane_index

        if curr_id == target_id:
            self.prev_lane_progress = 1.0
            return 0.0

        lane = self.road.network.get_lane(self.vehicle.lane_index)
        _, lat = lane.local_coordinates(self.vehicle.position)
        width = float(getattr(lane, "width", 4.0))

        direction = float(np.sign(target_id - curr_id))
        progress_now = float(np.clip(lat * direction / max(width * 0.5, 1e-6), 0.0, 1.0))

        delta_progress = max(0.0, progress_now - self.prev_lane_progress)
        self.prev_lane_progress = progress_now

        return delta_progress

    def _r_fn_lane_start(self) -> float:
        if self.started_lane_change:
            return 0.0

        _, _, curr_id = self.vehicle.lane_index
        _, _, target_id = self.target_lane_index

        if curr_id == target_id:
            return 0.0

        lane = self.road.network.get_lane(self.vehicle.lane_index)
        _, lat = lane.local_coordinates(self.vehicle.position)
        width = float(getattr(lane, "width", 4.0))

        direction = float(np.sign(target_id - curr_id))
        progress_now = float(np.clip(lat * direction / max(width * 0.5, 1e-6), 0.0, 1.0))

        if progress_now > 0.05:
            self.started_lane_change = True
            return 1.0

        return 0.0
    # ── Legacy ────────────────────────────────────────────────────────────────

    def _sample_scenario(self):
        """Deprecated: use self.source.sample_initial_state() instead."""
        if self.source is None:
            return None
        return self.source.sample_initial_state(self.np_random)


    def close(self):
        if getattr(self, "connector", None) is not None:
            try:
                self.connector.close()
            except Exception:
                pass
        return super().close()

def check_env():
    from gymnasium.utils.env_checker import check_env
    env = gym.make("lane-changing-v0", render_mode=None)
    env = wrappers.ObsWrapper(env)
    check_env(env.unwrapped)


if __name__ == "__main__":
    check_env()
