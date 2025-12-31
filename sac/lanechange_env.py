import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from enum import Enum
import numpy as np

# register model as gym env. Once registered, the id is usable for gym.make()
register(
    id = 'lane-changing-v0',
    entry_point = 'lane_changing:LaneChangingEnv',
)

class LaneChangingEnv(gym.Env):
    metadata = {}
    #TODO: Finish init, reset, step, observation, action, reward
    def __init__(self, dt = 0.1, max_steps = 200, render_mode = None):
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        pass

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        # TODO: Figure out what goes in a reset state
        pass 

    def step(self):
        pass



def check_env(): 
    from gymnasium.utils.env_checker import check_env
    env = gym.make('lane-changing-v0', render_mode = None)
    check_env(env.unwrapped)


if __name__ == '__main__':
    check_env()