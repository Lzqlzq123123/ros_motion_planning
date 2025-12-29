#!/usr/bin/env python3
"""
Evaluate a policy trained with train_new.py (rsl_rl OnPolicyRunner + RosGazeboEnv).

Example:
    python eval_trained_policy.py \
        --config src/rl_training/config/forklift_ppo.yaml \
        --model src/rl_training/logs/forklift_ppo/run_1/model_100.pt \
        --num-episodes 5
"""

import argparse
import os
from typing import Any, Dict

import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from rsl_rl.runners import OnPolicyRunner
from envs.ros_gazebo_env import RosGazeboEnv


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config yaml used for train_new.py")
    parser.add_argument("--model", required=True, help="Path to saved model (logs/.../model_XXX.pt)")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", default=None, help="Optional tensorboard logdir for eval metrics")
    args = parser.parse_args()

    # Load training config (contains env + runner)
    cfg = load_yaml(args.config)
    env_cfg = cfg["env"]
    runner_cfg = cfg["runner"]

    device = args.device
    env = RosGazeboEnv(env_cfg, device=device)

    # Build runner consistent with training
    runner = OnPolicyRunner(env=env, train_cfg=runner_cfg, log_dir=None, device=device)

    # Load model
    print(f"Loading model from {args.model} ...")
    runner.load(args.model, load_optimizer=False, map_location=device)

    # Inference function
    infer_fn = runner.get_inference_policy(device=device)

    writer = SummaryWriter(log_dir=args.logdir) if args.logdir else None

    rew_hist = []
    len_hist = []
    for ep in range(args.num_episodes):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        ep_len = 0
        while not done:
            with torch.no_grad():
                action = infer_fn(obs)
            obs, reward, done, extras = env.step(action.to(env.device))
            ep_rew += float(reward.item())
            ep_len += 1
        rew_hist.append(ep_rew)
        len_hist.append(ep_len)
        print(f"Episode {ep}: reward={ep_rew:.2f}, length={ep_len}")
        if writer:
            writer.add_scalar("Eval/episode_reward", ep_rew, ep)
            writer.add_scalar("Eval/episode_length", ep_len, ep)

    # Summary
    import statistics

    mean_rew = statistics.mean(rew_hist)
    std_rew = statistics.pstdev(rew_hist) if len(rew_hist) > 1 else 0.0
    mean_len = statistics.mean(len_hist)
    std_len = statistics.pstdev(len_hist) if len(len_hist) > 1 else 0.0

    print("Evaluation summary:")
    print(f"Mean reward: {mean_rew:.2f} ± {std_rew:.2f}")
    print(f"Mean length: {mean_len:.2f} ± {std_len:.2f}")

    if writer:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
