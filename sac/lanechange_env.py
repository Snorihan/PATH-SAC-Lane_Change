import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import highway_env
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.highway_env import HighwayEnv
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from enum import Enum
import numpy as np
import pprint
import wrappers

# register model as gym env. Once registered, the id is usable for gym.make()
register(
    id = 'lane-changing-v0',
    entry_point = 'lanechange_env:LaneChangingEnv',
)

class LaneChangingEnv(HighwayEnv):

    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "rewards": {
                "collision": -1000.0,
                "lane_success": 1,
                "jerk_weight" : 0.5,
                # more weights to come
            }
        })
        return config
    # Action = [acceleration, steering] for now

    def __init__(self, config = None, render_mode = None):
        super().__init__(config=config, render_mode=render_mode)
     
    def _create_action_space(self):
        # accel = action[0] 
        # steer = action[1] 
        # we will need to scale them accordingly later. Accel is m/s^2 and steer is in degrees (after change)
        self.action_space = spaces.Box(
            low = np.array([-1.0, -1.0]), 
            high = np.array([1.0, 1.0]),
            dtype = np.float32
        )

    def _apply_action(self, action):
        """
        Map continuous action to vehicle control.
        """
        accel_norm, intent = float(action[0]), float(action[1])

        vehicle: ControlledVehicle = self.vehicle
        if vehicle is None:
            return

        # Longitudinal: normalized -> physical accel
        max_accel = 3.0
        acceleration = np.clip(accel_norm, -1.0, 1.0) * max_accel

        # Lateral: decide which lane to track (current vs target)
        deadzone = 0.2
        current_lane_index = vehicle.lane_index  # (from_node, to_node, lane_id)
        from_n, to_n, lane_id = current_lane_index

        # determine number of lanes on the current segment (more correct than config)
        try:
            lanes_on_segment = len(vehicle.road.network.graph[from_n][to_n])
        except Exception:
            lanes_on_segment = int(self.config.get("lanes_count", 4))

        # choose lane to follow
        if abs(intent) < deadzone:
            desired_lane_id = int(lane_id)  # keep lane
        else:
            desired_lane_id = int(getattr(self, "target_lane", lane_id))  # commit to goal

        desired_lane_id = int(np.clip(desired_lane_id, 0, lanes_on_segment - 1))
        target_lane_index = (from_n, to_n, desired_lane_id)

        # Use controller to compute steering angle to track the chosen lane
        steering = float(vehicle.steering_control(target_lane_index))

        # Apply to vehicle
        vehicle.action = {
            "acceleration": acceleration,
            "steering": steering
        }

    def _reward(self, action):
        w = self.configs["rewards"]
        curr_time = self.time
        curr_acc = self.vehicle.action["acceleration"]
        reward = 0.0

        if self.vehicle.crashed:
            reward -= w["collision"] # want no crashes. Maybe we make it infinity idk

        # -- Jerk --
        reward -= (abs(self.__reward_funct_jerk(curr_time, curr_acc)) - 0.5) * w["jerk_weight"]
        # according to studies, discomfort for jerk starts around 0.5 m/s^3
        # -- Encourage speed --

        # -- Encourage Lane Change completion -- 
        # print("vehicle attributes", dir(self.vehicle))
        # lane_index[0]: from_node, [1]: to_node, [2]: lane_id
        if self.start_lane != self.vehicle.lane_index[2]:
            reward += 0.5

        # add penalties and replace with dynamic rewards later
        
        return reward

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.prev_acceleration = 0.0 # this is for the reward function
        self.start_time = getattr(self, "time", 0.0) # find the start time
        self.start_lane = self.vehicle.lane_index[2] # lane where the vehicle is. Maybe this could be a source of bugs
        self.target_lane = self.find_target_lane(self.start_lane)
        # limit to the lanes on the road
        self.target_lane = int(np.clip(self.target_lane, 0, self.config.get("lanes_count", 4) - 1))
        # print("ROAD: ", dir(self.road)) # self.road.neighbor_vehicles exists wow
        return obs, info
    
    def __reward_funct_jerk(self, curr_time, curr_acc):
        time_dt = curr_time - self.start_time

        acc_dt = curr_acc - self.prev_acceleration
        jerk = acc_dt / time_dt

        self.start_time = curr_time
        self.prev_acceleration = curr_acc
        return jerk

    def __r_fn_relative_distance(self, min_gap: float = 20.0, max_gap: float = 100.0): # I imagine these are in meters?
        r = 0.0
        curr_lane_index = self.vehicle.lane_index 
        _, __, gap_front, gap_rear = self.__vehicle_in_front_rear(curr_lane_index)
        if (gap_front < min_gap):
            r -= min_gap - gap_front # TODO: ya we got to scale it DOWN later
        if (abs(gap_rear) < min_gap):
            r -= min_gap - abs(gap_rear)

        if gap_front >= max_gap and abs(gap_rear) >= min_gap:
            r = 0.0

        return r
        
    def _r_fn_relative_speeds(self, max_range: float = 200.0) -> float:
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

    def _r_fn_matching_speed_limit(self) -> float:
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

    def __vehicle_in_front_rear(self, lane_index, max_range: float = 200.0): 
        ego = self.vehicle
        lane = self.road.network.get_lane(lane_index)
        ego_s, ego_lat = lane.local_coordinates(ego.position)

        gap_front = float(np.inf)
        gap_rear = -float(np.inf)
        front_vehicle, rear_vehicle = None, None
        for v in self.road.vehicles:
            if v == ego:
                continue 

            if getattr(v, "lane", None) is not lane:
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

    def find_target_lane(self, start_lane):
        return start_lane + 1
        # TODO: Will need to customize this based off of the target lane really made in the trigger side right?

    

def check_env(): 
    from gymnasium.utils.env_checker import check_env
    env = gym.make('lane-changing-v0', render_mode = None)
    env = wrappers.ObsWrapper(env)


    check_env(env.unwrapped)
    print((env.observation_space))
    print("--Config list--")
    pprint.pprint(env.unwrapped.config)


if __name__ == '__main__':
    check_env()

    
