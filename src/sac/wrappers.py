import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ObsWrapper(gym.ObservationWrapper):
    """
    Flattens HighwayEnv's (vehicles_count, features) observation into 1D
    and appends 6 scalars:
      [lane_delta_to_target, target_lane_norm,
       is_lane_changing, lc_direction, cooldown_remaining, lc_progress]

    The last 4 expose the hysteresis decoder's internal state so that the
    environment remains Markov from SAC's perspective (same obs → same transition).
    """

    NUM_ADDED_FEATURES = 6

    def __init__(self, env: gym.Env):
        super().__init__(env)

        base_shape = self.observation_space.shape  # e.g. (5, 5)
        base_dim = int(np.prod(base_shape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_dim + self.NUM_ADDED_FEATURES,),
            dtype=np.float32
        )

    def observation(self, obs):
        obs_flat = np.asarray(obs, dtype=np.float32).ravel()
        obs_flat = np.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.isfinite(obs_flat).all():
            raise ValueError("Non-finite obs detected")

        base_env = self.env.unwrapped
        current_lane = int(base_env.vehicle.lane_index[2])

        if hasattr(base_env, "target_lane_index"):
            target_lane = int(base_env.target_lane_index[2])
        else:
            target_lane = int(getattr(base_env, "target_lane", current_lane))

        lanes_count = int(base_env.config.get("lanes_count", 4))
        denom = max(lanes_count - 1, 1)

        lane_delta   = float(np.clip((target_lane - current_lane) / denom, -1.0, 1.0))
        target_norm  = float(np.clip(2.0 * (target_lane / denom) - 1.0, -1.0, 1.0))

        # Hysteresis decoder state — makes the MDP Markov
        is_lc        = float(getattr(base_env, "lc_active", False))
        lc_dir       = float(getattr(base_env, "last_lc_direction", 0))   # -1, 0, or +1
        cooldown_max = float(base_env.config.get("lc_cooldown_steps", 20))
        cooldown_rem = float(np.clip(
            getattr(base_env, "lc_cooldown", 0) / max(cooldown_max, 1.0), 0.0, 1.0
        ))

        # Lateral progress toward target lane (0 = at current lane, 1 = fully in target lane)
        _, _, curr_id   = base_env.vehicle.lane_index
        _, _, tgt_id    = base_env.target_lane_index if hasattr(base_env, "target_lane_index") \
                          else (None, None, curr_id)
        if curr_id == tgt_id:
            lc_progress = 1.0
        else:
            lane  = base_env.road.network.get_lane(base_env.vehicle.lane_index)
            _, lat = lane.local_coordinates(base_env.vehicle.position)
            width  = float(getattr(lane, "width", 4.0))
            direction = float(np.sign(tgt_id - curr_id))
            lc_progress = float(np.clip(lat * direction / max(width * 0.5, 1e-6), 0.0, 1.0))

        extra = np.array(
            [lane_delta, target_norm, is_lc, lc_dir, cooldown_rem, lc_progress],
            dtype=np.float32,
        )
        return np.concatenate([obs_flat, extra], axis=0).astype(np.float32, copy=False)
