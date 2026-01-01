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

# register model as gym env. Once registered, the id is usable for gym.make()
register(
    id = 'lane-changing-v0',
    entry_point = 'lanechange_env:LaneChangingEnv',
)

class LaneChangingEnv(HighwayEnv):
    # Action = [acceleration, steering] for now

    def __init__(self, config = None):
        super().__init__(config)

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
        accel_norm, steer_norm = action

        vehicle: ControlledVehicle = self.vehicle
        if vehicle is None:
            return

        # ---- Longitudinal control ----
        # Map [-1, 1] → reasonable accel range
        max_accel = 3.0  # m/s^2. TODO: SUBJECT TO CHANGE
        acceleration = float(np.clip(accel_norm, -1, 1)) * max_accel

        # ---- Lateral control ----
        # Steering mapped to lane offset / heading
        max_steering = np.deg2rad(15)  # conservative. Subject to change
        steering = float(np.clip(steer_norm, -1, 1)) * max_steering

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
        reward += self.vehicle.speed / self.config["speed_limit"]

        # encourage lane change completion
        if self.vehicle.lane_index[2] != self.vehicle.target_lane_index[2]:
            reward += 0.5

        # add penalties and replace with dynamic rewards later
        
        return reward


def check_env(): 
    from gymnasium.utils.env_checker import check_env
    env = gym.make('lane-changing-v0', render_mode = None)
    check_env(env.unwrapped)


if __name__ == '__main__':
    check_env()