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
        reward = 0.0

        if self.vehicle.crashed:
            reward -= 1000000.0 # no crashes

        # encourage speed
        # reward += self.vehicle.speed / self.config["speed_limit"]

        # encourage lane change completion
        # print("vehicle attributes", dir(self.vehicle))
        # lane_index[0]: from_node, [1]: to_node, [2]: lane_id
        if self.start_lane != self.vehicle.lane_index[2]:
            reward += 0.5

        # add penalties and replace with dynamic rewards later
        
        return reward

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.start_lane = self.vehicle.lane_index[2]
        self.target_lane = self.find_target_lane(self.start_lane)
        # limit to the lanes on the road
        self.target_lane = int(np.clip(self.target_lane, 0, self.config.get("lanes_count", 4) - 1))

        return obs, info
    
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

    
