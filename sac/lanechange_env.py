import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import highway_env
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.kinematics import Vehicle
from enum import Enum
import numpy as np
import pprint
import wrappers
from highway_env.envs.common.action import ActionType  # highway-env import

class IntentContinuousAction(ActionType):
    vehicle_class = ControlledVehicle
    """
    Action = [accel_norm, intent]
    accel_norm in [-1,1] -> m/s^2
    intent in [-1,1] -> your lane selection logic in _apply_action()
    """
    def space(self):
        return spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def act(self, action):
        # This is the critical line: route actions to YOUR mapping
        self.env._apply_action(action)
# register model as gym env. Once registered, the id is usable for gym.make()
register(
    id = 'lane-changing-v0',
    entry_point = 'lanechange_env:LaneChangingEnv',
)

class LaneChangingEnv(HighwayEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "rewards": { # treating these almost like binary strings MSB/LSB weighings
                "collision": -500.0,
                "lane_success": 5.0,
                "jerk_weight" : 5.0,
                "dist_relative_to_nearby_vehicles_weight": 10.0,
                "speed_relative_to_nearby_vehicles_weight": 10,
                "follow_spd_lmt_weight": 1,
                "lane_changing_weight": 5.0, # kind of a gradient yknow
                # more weights to come
            }
        })
        config.update({"duration_after_lane_change": 100}) # Steps to see car behavior after lane change
        config.update({"duration_before_lane_change": 40}) # Steps to stay in lane before changing
        config.update({"lane_change_lat_threshold": 0.5})
        return config
    # Action = [acceleration, steering] for now

    def define_spaces(self) -> None:
        super().define_spaces()

        # Force our custom action type regardless of config["action"]["type"]
        self.action_type = IntentContinuousAction(self)
        self.action_space = self.action_type.space()

    def _create_road(self) -> None:
        """
        Create a long, straight, eastbound road (positive x direction)
        with `lanes_count` parallel lanes.
        """
        net = RoadNetwork()

        lanes = int(self.config.get("lanes_count", 4))
        lane_width = float(self.config.get("lane_width", 4.0))
        road_length = float(self.config.get("road_length", 1000.0))

        from_n, to_n = "0", "1"

        for lane_id in range(lanes):
            origin = [0.0, lane_id * lane_width]
            end = [road_length, lane_id * lane_width]

            # Outer boundaries solid; interior boundaries striped
            left_type = LineType.CONTINUOUS if lane_id == 0 else LineType.STRIPED
            right_type = LineType.CONTINUOUS if lane_id == lanes - 1 else LineType.STRIPED
            line_types = [left_type, right_type]

            net.add_lane(from_n, to_n, StraightLane(origin, end, width=lane_width, line_types=line_types))

        self.road = Road(network=net, np_random=self.np_random)

    def __init__(self, config = None, render_mode = None):
        super().__init__(config=config, render_mode=render_mode)
     
    # def _create_action_type(self):
    #     self.action_type = IntentContinuousAction(self)
    #     self.action_space = self.action_type.space()

    def _apply_action(self, action):
        accel_norm, intent = float(action[0]), float(action[1])
        vehicle: ControlledVehicle = self.vehicle
        if vehicle is None:
            return

        # ---- lane target ----
        deadzone = 0.2
        from_n, to_n, lane_id = vehicle.lane_index

        try:
            lanes_on_segment = len(vehicle.road.network.graph[from_n][to_n])
        except Exception:
            lanes_on_segment = int(self.config.get("lanes_count", 4))

        if abs(intent) < deadzone:
            desired_lane_id = int(lane_id)
        else:
            # track the env target lane (what you want long-term)
            target = getattr(self, "target_lane_index", vehicle.lane_index)
            desired_lane_id = int(target[2]) if isinstance(target, tuple) else int(target)

        desired_lane_id = int(np.clip(desired_lane_id, 0, lanes_on_segment - 1))
        target_lane_index = (from_n, to_n, desired_lane_id)

        # ---- speed target ----
        max_accel = 3.0
        a = np.clip(accel_norm, -1.0, 1.0) * max_accel
        dt = 1.0 / float(self.config.get("policy_frequency", 15))

        # integrate accel into a desired speed target
        desired_speed = float(np.clip(vehicle.speed + a * dt, 0.0, 40.0))

        # set targets (these survive the sim calling vehicle.act())
        vehicle.target_lane_index = target_lane_index
        vehicle.target_speed = desired_speed


    def _reward(self, action):
        w = self.config["rewards"]
        curr_time = self.time
        curr_acc = self.vehicle.action["acceleration"]
        reward = 0.0

        # Calculate components
        r_collision = w["collision"] if self.vehicle.crashed else 0.0
        r_jerk = (abs(self._reward_funct_jerk(curr_time, curr_acc)) - 0.5) * w["jerk_weight"]
        r_speed = self._r_fn_relative_speeds() * w["speed_relative_to_nearby_vehicles_weight"]
        r_dist = self._r_fn_relative_distance() * w["dist_relative_to_nearby_vehicles_weight"]
        r_limit = self._r_fn_matching_speed_limit() * w["follow_spd_lmt_weight"]
        r_success = self._r_fn_lane_change_success(self.vehicle.lane_index) * w["lane_success"]
        r_changing = self._r_fn_lane_changing() * w["lane_changing_weight"]

        if self.vehicle.crashed:
            reward += r_collision # Add negative reward

        reward -= r_jerk
        reward += r_speed
        reward += r_dist
        reward += r_limit
        reward += r_success
        reward += r_changing

        self.reward_dict = {
            "r_collision": r_collision, "r_jerk": r_jerk, "r_speed": r_speed,
            "r_dist": r_dist, "r_limit": r_limit, "r_success": r_success, "r_changing": r_changing,
            "raw_jerk": self.last_jerk_value
        }
        return reward

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        self.elapsed_steps += 1

        # ---- Update lane-changing flag ----
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        _, lat = lane.local_coordinates(self.vehicle.position)

        # threshold in "meters-ish"; we should move this to config
        lat_threshold = float(self.config.get("lane_change_lat_threshold", 0.5)) # right now 0.5 meters. Subject to change

        # geometry-based indicator
        self.lane_changing = (abs(lat) > lat_threshold)
        if hasattr(self, "reward_dict"):
            info.update(self.reward_dict)

        # ---- Termination Logic: End N steps after reaching target ----. Possible bug about a new lane change 
        if self.vehicle.lane_index == self.target_lane_index:
            self.steps_in_target_lane += 1
        else:
            self.steps_in_target_lane = 0 # Reset if we drift back out

        if self.steps_in_target_lane >= self.config.get("duration_after_lane_change", 40):
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.lane_changing = False
        self.prev_acceleration = 0.0 # this is for the reward function
        self.prev_lat = 0.0
        self.steps_in_target_lane = 0
        self.elapsed_steps = 0
        self.last_jerk_value = 0.0
        self.start_time = getattr(self, "time", 0.0) # find the start time
        self.start_lane_index = self.vehicle.lane_index # lane where the vehicle is. This is more a node. If we want spacial reasoning use .get_lane(__)
        
        # Determine the ultimate goal, but initially target the start lane (warmup)
        self.ultimate_target_lane_index = self.find_target_lane(self.start_lane_index)
        self.target_lane_index = self.ultimate_target_lane_index
        
        # self.target_lane = int(np.clip(self.target_lane, 0, self.config.get("lanes_count", 4) - 1))
        # print("ROAD: ", dir(self.road)) # self.road.neighbor_vehicles exists wow
        print("EGO TYPE:", type(self.vehicle))

        return obs, info
    
    def _reward_funct_jerk(self, curr_time, curr_acc):
        time_dt = curr_time - self.start_time

        if time_dt <= 1e-6:
            self.last_jerk_value = 0.0
            return 0.0

        acc_dt = curr_acc - self.prev_acceleration
        jerk = acc_dt / time_dt

        self.last_jerk_value = jerk
        self.start_time = curr_time
        self.prev_acceleration = curr_acc
        return jerk

    def _r_fn_relative_distance(self, min_gap: float = 20.0, max_gap: float = 100.0): # I imagine these are in meters?
        r = 0.0
        curr_lane_index = self.vehicle.lane_index 
        _, __, gap_front, gap_rear = self._vehicle_in_front_rear(curr_lane_index)
        if (gap_front < min_gap):
            r -= min_gap - gap_front # TODO: ya we got to scale it DOWN later
        if (abs(gap_rear) < min_gap):
            r -= min_gap - abs(gap_rear)

        if gap_front >= max_gap and abs(gap_rear) >= min_gap:
            r = 0.0

        return r
        
    def _r_fn_relative_speeds(self, max_range: float = 200.0):
        """
        Penalize unsafe relative speed:
        - If you're faster than the front car, that's a closing speed.
        - If the rear car is faster than you, that's a rear closing speed.
        """
        r = 0.0
        ego_speed = float(self.vehicle.speed)

        curr_lane_index = self.vehicle.lane_index
        front_v, rear_v, gap_front, gap_rear = self._vehicle_in_front_rear(curr_lane_index, max_range=max_range)

        # Front closing speed (only if a front vehicle exists within range)
        if front_v is not None and gap_front <= max_range:
            front_speed = float(front_v.speed)
            closing = ego_speed - front_speed
            if closing > 0.0:
                r -= closing

        # Rear closing speed (rear approaches you)
        if rear_v is not None and abs(gap_rear) <= max_range:
            rear_speed = float(rear_v.speed)
            rear_closing = rear_speed - ego_speed
            if rear_closing > 0.0:
                r -= rear_closing

        return float(r)

    def _r_fn_matching_speed_limit(self):
        """
        Reward/penalize based on how close ego speed is to the configured speed limit.
        Requires you to set self.config["speed_limit"] (or provide a fallback).
        """
        speed_limit = self.config.get("speed_limit", None)
        if speed_limit is None:
            return 0.0  # nothing to do if you haven't defined it

        ego_speed = float(self.vehicle.speed)
        speed_limit = float(speed_limit)

        # Positive when at/under limit; negative when over (simple, sign-consistent)
        if ego_speed <= speed_limit:
            return ego_speed / max(speed_limit, 1e-6)
        else:
            return -(ego_speed - speed_limit)

    def _r_fn_lane_change_success(self, curr_lane_index):
        return 1 if curr_lane_index == self.target_lane_index else 0

    def _r_fn_lane_changing(self):
        # two stages. One halfway through lane changing (measuring whether we accelerate). Next half of lane changing (measuring whether we deccelerate)
        if not self.lane_changing:
            return 0.0
        velocity_reward = 0.0
        acc_reward = 0.0

        # --- lane geometry ---
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        width = float(getattr(lane, "width", 4.0))
        half_width = 0.5 * width

        # lane-local coordinates
        _, lat = lane.local_coordinates(self.vehicle.position)

        # normalized progress across lane [0, 1]
        progress = min(abs(lat) / max(half_width, 1e-6), 1.0)

        # --- timing / derived signals ---
        dt = 1.0 / float(self.config.get("policy_frequency", 15))

        prev_lat = float(getattr(self, "prev_lat", lat))
        v_lat = (lat - prev_lat) / dt  # m/s lateral speed
        self.prev_lat = float(lat)     # update for next call

        # threshold replacing paper's "0.2"
        v_lat_th = 0.05 * width  # lane-width-scaled threshold (m/s)

        # --- rewards ---
        # Stage selection
        early = (progress < 0.5)

        # Longitudinal accel (m/s^2) from applied action
        a_long = float(self.vehicle.action.get("acceleration", 0.0))

        velocity_reward = 0.0
        acc_reward = 0.0

        # (1) gate: only reward these behaviors if we're actually moving laterally meaningfully
        if abs(v_lat) < v_lat_th:
            return 0.0

        # (2) early half: prefer positive accel; late half: prefer negative accel
        if early:
            acc_reward = max(a_long, 0.0)   # reward only positive accel
        else:
            acc_reward = max(-a_long, 0.0)  # reward only braking / decel

        # (3) optional: modest reward for sustained lateral motion (keeps maneuver from stalling)
        # tie it to lane width so scale is consistent across roads
        velocity_reward = min(abs(v_lat) / max(v_lat_th, 1e-6), 2.0)

        return float(velocity_reward + acc_reward)

    def _vehicle_in_front_rear(self, lane_index, max_range: float = 200.0): 
        ego = self.vehicle
        lane = self.road.network.get_lane(lane_index)
        ego_s, ego_lat = lane.local_coordinates(ego.position)

        gap_front = float(np.inf)
        gap_rear = -float(np.inf)
        front_vehicle, rear_vehicle = None, None
        for v in self.road.vehicles:
            if v == ego:
                continue 

            if v.lane_index != lane_index:
                continue
            
            veh_s, veh_lat = self.road.network.get_lane(v.lane_index).local_coordinates(v.position)
            gap = veh_s - ego_s

            if 0.0 < gap < gap_front and gap <= max_range:
                gap_front = gap
                front_vehicle = v
            if 0.0 > gap > gap_rear and -gap <= max_range:
                gap_rear = gap
                rear_vehicle = v

        return front_vehicle, rear_vehicle, gap_front, gap_rear

    # def find_target_lane(self, start_lane_index):
    #     """
    #     Return an abstract lane (LaneIndex) rather than a lane_id number.
    #     start_lane_index: (from_node, to_node, lane_id). Will be changed when I figure out the trigger network
    #     """
    #     from_n, to_n, lane_id = start_lane_index

    #     # choose target lane id (example: +1 lane to the left/right depending on your convention)
    #     target_lane_id = int(lane_id) + 1

    #     # clamp to lanes available on this road segment
    #     try:
    #         lanes_on_segment = len(self.road.network.graph[from_n][to_n])
    #     except Exception:
    #         lanes_on_segment = int(self.config.get("lanes_count", 4))

    #     target_lane_id = int(np.clip(target_lane_id, 0, lanes_on_segment - 1))

    #     return (from_n, to_n, target_lane_id)

    def find_target_lane(self, start_lane_index):
        '''Temporary to just choose the adjacent lanes. We will train another network to do so later'''
        from_n, to_n, lane_id = start_lane_index
        lanes = int(self.config.get("lanes_count", 4))
        # prefer +1, else -1
        target = lane_id + 1 if lane_id + 1 < lanes else lane_id - 1
        target = int(np.clip(target, 0, lanes - 1))
        return (from_n, to_n, target)


def check_env(): 
    from gymnasium.utils.env_checker import check_env
    env = gym.make('lane-changing-v0', render_mode = None)
    env = wrappers.ObsWrapper(env)

    check_env(env.unwrapped)
    # print((env.observation_space))
    # print("--Config list--")
    # pprint.pprint(env.unwrapped.config)


if __name__ == '__main__':
    check_env()

    
