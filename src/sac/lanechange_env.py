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
            "braking_reward_weight": 10.0,
            "blocked_merge_weight": -20.0,

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

            # Oscillation
            "oscillation_penalty_weight": 0.0,
        })
        config.update({
            # Hysteresis thresholds for intent → lane-change decision
            "lc_start_threshold":       0.6,   # |intent| must exceed this to start a LC
            "lc_stop_threshold":        0.3,   # |intent| must drop below this to cancel
            "lc_cooldown_steps":        20,    # steps blocked after a LC starts
            "oscillation_window_steps": 40,    # window for detecting direction reversals
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
        self.merge_blocked = False

        # Hysteresis / cooldown state
        self.lc_cooldown       = 0     # steps remaining in post-LC cooldown
        self.lc_active         = False # True while a lane-change maneuver is in flight
        self.last_lc_direction = 0     # direction of last initiated LC: -1=left, +1=right
        self.last_lc_step      = -999  # elapsed_steps when last LC was initiated
        self._last_raw_intent  = 0.0   # raw action[1] from last step (for osc penalty)

        # Safety shield telemetry (reset each episode)
        self.shield_fwd_interventions  = 0    # times _cap_accel_for_front_gap overrode accel
        self.shield_lc_interventions   = 0    # times _lane_change_safe blocked a LC attempt
        self.shield_min_fwd_ttc        = float("inf")  # min TTC seen before a fwd override
        self.shield_min_lc_ttc         = float("inf")  # min TTC seen before a LC override

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

        raw_intent = float(action[1])
        self._last_raw_intent = raw_intent

        start_thr      = float(self.config.get("lc_start_threshold", 0.6))
        stop_thr       = float(self.config.get("lc_stop_threshold",  0.3))
        cooldown_steps = int(self.config.get("lc_cooldown_steps",    20))

        # If vehicle completed its lane change, clear the active flag
        if self.lc_active and self._same_lane(vehicle.lane_index, vehicle.target_lane_index):
            self.lc_active = False

        # Decrement cooldown each step
        if self.lc_cooldown > 0:
            self.lc_cooldown -= 1

        # Hysteresis gate: translate continuous intent into a persistent LC decision
        if self.lc_active:
            if abs(raw_intent) < stop_thr:
                # Intent dropped below stop threshold — abort maneuver
                self.lc_active = False
                target = vehicle.lane_index
            else:
                target = command["target_lane_index"]
        elif self.lc_cooldown == 0 and abs(raw_intent) > start_thr:
            # Intent strong enough and cooldown expired — start new LC
            self.lc_active         = True
            self.lc_cooldown       = cooldown_steps
            self.last_lc_direction = int(np.sign(raw_intent))
            self.last_lc_step      = self.elapsed_steps
            target = command["target_lane_index"]
        else:
            target = vehicle.lane_index

        # TTC safety gate (always applied on top of hysteresis decision)
        intended_lc = not self._same_lane(target, vehicle.lane_index)
        if intended_lc and not self._lane_change_safe(target):
            target = vehicle.lane_index
            self.merge_blocked = True
            self.shield_lc_interventions += 1
            self.shield_min_lc_ttc = min(
                self.shield_min_lc_ttc,
                self._min_target_lane_ttc(target),
            )
        else:
            self.merge_blocked = False

        accel = self._cap_accel_for_front_gap(command["accel_mps2"])

        vehicle.target_lane_index = target
        vehicle.action = {"acceleration": accel}
        vehicle.target_speed = float(np.clip(vehicle.speed + accel * command["dt"], 0.0, 40.0))
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
            # Comfort: reward for maintaining safe following gap (0→1)
            RewardTerm(
                name   = "gap",
                fn     = self._r_fn_gap,
                weight = w["gap_weight"],
                sign   = +1.0,
            ),
            # Safety: penalize closing speed toward front vehicle — teaches braking before merge
            RewardTerm(
                name   = "closing_speed",
                fn     = self._r_fn_closing_speed,
                weight = w["closing_speed_weight"],
                sign   = -1.0,
            ),
            # Safety: reward choosing to brake when a front vehicle is within comfortable gap
            RewardTerm(
                name   = "braking_reward",
                fn     = self._r_fn_braking_reward,
                weight = w["braking_reward_weight"],
                sign   = +1.0,
            ),
            # Timing: penalize attempting a merge that the TTC gate had to block
            RewardTerm(
                name   = "blocked_merge",
                fn     = lambda: 1.0 if self.merge_blocked else 0.0,
                weight = w["blocked_merge_weight"],
                sign   = +1.0,
            ),
            # Comfort: penalize lateral speed as discomfort proxy
            RewardTerm(
                name   = "lat_accel",
                fn     = self._r_fn_lat_accel,
                weight = w["lat_accel_weight"],
                sign   = -1.0,
            ),
            # Task: per-step penalty for not initiating a requested lane change
            RewardTerm(
                name   = "lane_keeping",
                fn     = self._r_fn_lane_keeping_when_requested,
                weight = w["lane_keeping_penalty_when_requested"],
                sign   = -1.0,
            ),
            # Task: penalty for drifting in the wrong lateral direction
            RewardTerm(
                name   = "wrong_lane",
                fn     = self._r_fn_wrong_lane,
                weight = w["wrong_lane_penalty"],
                sign   = -1.0,
            ),
            # Stability: penalize rapid intent direction reversals
            RewardTerm(
                name   = "oscillation",
                fn     = self._r_fn_oscillation_penalty,
                weight = w.get("oscillation_penalty_weight", 0.0),
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
            info["shield"] = self._shield_info()
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
            "shield":   self._shield_info(),
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

        threshold = int(self.config.get("duration_after_lane_change", 40))
        if self.steps_in_target_lane < threshold:
            return False

        # Reached target lane. If continuous_targets is enabled (default for live backend),
        # pick the next adjacent target instead of terminating so the car keeps driving.
        continuous = self.config.get(
            "continuous_targets",
            self.config.get("backend") == "aimsun_live",
        )
        if continuous:
            new_target = self.find_target_lane(self.vehicle.lane_index)
            if not self._same_lane(new_target, self.vehicle.lane_index):
                self.target_lane_index = new_target
                self.ultimate_target_lane_index = new_target
                self.got_lane_success = False
                self.steps_in_target_lane = 0
                self.started_lane_change = False
                return False   # continue episode with new target
        return True  # terminate (single-lane road or continuous_targets=False)

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

    @staticmethod
    def _ttc(gap: float, other_v, ego_speed: float, ego_closing: bool) -> float:
        """Time-to-collision in seconds; inf if gap is clear or not closing."""
        if other_v is None or gap >= float("inf"):
            return float("inf")
        closing = (ego_speed - float(other_v.speed)) if ego_closing else (float(other_v.speed) - ego_speed)
        if closing <= 0.0:
            return float("inf")
        return gap / closing

    def _cap_accel_for_front_gap(self, accel_mps2: float) -> float:
        """Acceleration gate based on front TTC.
        Above gate: pass through. Below gate: cap upward accel.
        Below half-gate: override with active braking regardless of agent command.
        """
        sv = self._get_surrounding_vehicles()
        ego_speed = float(self.vehicle.speed)
        front_ttc = self._ttc(sv.front_gap, sv.front_v, ego_speed, ego_closing=True)
        gate = float(self.config.get("fwd_ttc_gate", 2.0))
        if front_ttc >= gate:
            return accel_mps2
        # Shield is intervening — log it
        self.shield_fwd_interventions += 1
        self.shield_min_fwd_ttc = min(self.shield_min_fwd_ttc, front_ttc)
        alpha = float(np.clip(front_ttc / gate, 0.0, 1.0))
        capped = min(accel_mps2, accel_mps2 * alpha) if accel_mps2 > 0.0 else accel_mps2
        # Below half-gate: force braking proportional to urgency
        half_gate = gate * 0.5
        if front_ttc < half_gate:
            max_decel = float(self.config.get("max_decel", 3.0))
            urgency = float(np.clip(1.0 - front_ttc / half_gate, 0.0, 1.0))
            return min(capped, -max_decel * urgency)
        return capped

    def _min_target_lane_ttc(self, target_lane_index) -> float:
        """Return the minimum TTC to lead/lag in target lane (used for shield telemetry)."""
        sv = self._get_surrounding_vehicles()
        ego_speed = float(self.vehicle.speed)
        _, _, cur_id = self.vehicle.lane_index
        _, _, tgt_id = target_lane_index
        if tgt_id < cur_id:
            lead_ttc = self._ttc(sv.llead_gap, sv.llead_v, ego_speed, ego_closing=True)
            lag_ttc  = self._ttc(sv.llag_gap,  sv.llag_v,  ego_speed, ego_closing=False)
        else:
            lead_ttc = self._ttc(sv.rlead_gap, sv.rlead_v, ego_speed, ego_closing=True)
            lag_ttc  = self._ttc(sv.rlag_gap,  sv.rlag_v,  ego_speed, ego_closing=False)
        return min(lead_ttc, lag_ttc)

    def _lane_change_safe(self, target_lane_index) -> bool:
        """Return True if TTC to target-lane lead and lag both exceed lc_ttc_gate."""
        gate = float(self.config.get("lc_ttc_gate", 2.0))
        return self._min_target_lane_ttc(target_lane_index) >= gate

    def _shield_info(self) -> dict:
        """Snapshot of safety-shield intervention telemetry for the current episode so far."""
        total = self.shield_fwd_interventions + self.shield_lc_interventions
        steps = max(self.elapsed_steps, 1)
        return {
            "fwd_interventions":   self.shield_fwd_interventions,
            "lc_interventions":    self.shield_lc_interventions,
            "total_interventions": total,
            "intervention_rate":   total / steps,
            "min_fwd_ttc":         self.shield_min_fwd_ttc,
            "min_lc_ttc":          self.shield_min_lc_ttc,
        }

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

    def _r_fn_gap(self) -> float:
        sv = self._get_surrounding_vehicles()
        if sv.front_v is None:
            return 1.0
        comfortable_gap = max(float(self.vehicle.speed) * 2.0, 10.0)
        return float(np.clip(sv.front_gap / comfortable_gap, 0.0, 1.0))

    def _r_fn_closing_speed(self) -> float:
        sv = self._get_surrounding_vehicles()
        if sv.front_v is None:
            return 0.0
        closing = float(self.vehicle.speed) - float(sv.front_v.speed)
        speed_limit = float(self.config.get("speed_limit", 30.0))
        return float(np.clip(closing / max(speed_limit, 1.0), 0.0, 1.0))

    def _r_fn_braking_reward(self) -> float:
        """Reward braking when front gap is tight. Agent decides whether to LC or brake."""
        sv = self._get_surrounding_vehicles()
        if sv.front_v is None:
            return 0.0
        comfortable_gap = max(float(self.vehicle.speed) * 2.0, 10.0)
        if sv.front_gap >= comfortable_gap:
            return 0.0
        veh_action = getattr(self.vehicle, "action", {}) or {}
        accel = float(veh_action.get("acceleration", 0.0))
        if accel >= 0.0:
            return 0.0
        max_decel = float(self.config.get("max_decel", 3.0))
        return float(np.clip(abs(accel) / max(max_decel, 1e-6), 0.0, 1.0))

    def _r_fn_lat_accel(self) -> float:
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        _, lat = lane.local_coordinates(self.vehicle.position)
        freq = float(self.config.get("policy_frequency", 15))
        lat_speed = abs(lat - self.prev_lat) * freq
        self.prev_lat = lat
        return min(lat_speed, 5.0)

    def _r_fn_lane_keeping_when_requested(self) -> float:
        if self._same_lane(self.vehicle.lane_index, self.target_lane_index):
            return 0.0
        return 0.0 if self.started_lane_change else 1.0

    def _r_fn_wrong_lane(self) -> float:
        _, _, curr_id = self.vehicle.lane_index
        _, _, target_id = self.target_lane_index
        if curr_id == target_id:
            return 0.0
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        _, lat = lane.local_coordinates(self.vehicle.position)
        direction = float(np.sign(target_id - curr_id))
        return 1.0 if lat * direction < -0.1 else 0.0

    def _r_fn_oscillation_penalty(self) -> float:
        """Fire when agent commands a LC direction opposite to its last LC within the window."""
        if self.last_lc_direction == 0:
            return 0.0
        start_thr = float(self.config.get("lc_start_threshold", 0.6))
        window    = int(self.config.get("oscillation_window_steps", 40))
        raw_intent = float(self._last_raw_intent)
        if abs(raw_intent) < start_thr:
            return 0.0
        current_dir   = int(np.sign(raw_intent))
        steps_since_lc = self.elapsed_steps - self.last_lc_step
        if current_dir != self.last_lc_direction and steps_since_lc < window:
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
