"""
SAC Training and Demo Script for Pendulum-v1
--------------------------------------------
Train a SAC agent (no rendering) or run a graphical demo.
 
USAGE:
 
# Train SAC
python sac_pendulum.py --train --timesteps 200000
 
# Demo
python sac_pendulum.py --demo --model <path>
 
Author: Calin Dragos George
"""
 
import argparse
import os
import time
import gymnasium as gym
import numpy as np
import torch
 
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
 
 
# ---------------------------------------------------------
# Set seeds
# ---------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
# ---------------------------------------------------------
# Create Pendulum Environment
# ---------------------------------------------------------
def make_env(render_mode=None):
    # render_mode=None → training (no graphics)
    # render_mode="human" → demo (graphics)
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    env = Monitor(env)
    return env
 
 
# ---------------------------------------------------------
# Create SAC Model for Pendulum
# ---------------------------------------------------------
def create_sac(env, lr, log_dir):
 
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=lr,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=log_dir,
    )
 
    return model
 
 
# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train_sac(lr, timesteps, seed):
 
    set_seed(seed)
    env = make_env(render_mode=None)
 
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/SAC_Pendulum_lr{lr}_seed{seed}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
 
    print(f"\n Training SAC on Pendulum-v1")
    print(f"→ Learning Rate: {lr}")
    print(f"→ Seed: {seed}")
    print(f"→ Timesteps: {timesteps}")
    print(f"→ Logs: {log_dir}\n")
 
    model = create_sac(env, lr, log_dir)
    model.learn(total_timesteps=timesteps, progress_bar=True)
 
    model_path = os.path.join(log_dir, f"SAC_Pendulum_lr{lr}_seed{seed}.zip")
    model.save(model_path)
 
    print(f"\n Model saved to: {model_path}\n")
    env.close()
 
 
# ---------------------------------------------------------
# Demo Function (with graphics)
# ---------------------------------------------------------
def run_demo(model_path, episodes):
 
    if not os.path.exists(model_path):
        print(f"\n Model not found: {model_path}\n")
        return
 
    print(f"\n Running SAC Demo: {model_path}\n")
 
    env = make_env(render_mode="human")
    model = SAC.load(model_path)
 
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
 
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
 
        print(f"Episode {ep + 1} Reward = {ep_reward}")
 
    env.close()
 
 
# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--demo", action="store_true")
 
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
 
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
 
    args = parser.parse_args()
 
    if args.train:
        train_sac(args.lr, args.timesteps, args.seed)
 
    elif args.demo:
        if args.model is None:
            print("\n ERROR: missing --model path\n")
        else:
            run_demo(args.model, args.episodes)
 
    else:
        print("\n Please specify --train or --demo.\n")