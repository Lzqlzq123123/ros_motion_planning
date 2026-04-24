import os
import argparse
import yaml
import numpy as np
import torch
import time
import sys
import os.path as osp

proj_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(proj_root, 'src'))

from rl_training.third_party.drl_td3.td3_models import TD3
from rl_training.envs.movebase_gazebo_env import MoveBaseGazeboEnv


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_eval_overrides(env_cfg, args):
    env_cfg = dict(env_cfg)

    if args.max_steps is not None:
        env_cfg['max_episode_length'] = int(args.max_steps)

    if any(v is not None for v in [args.goal_x_min, args.goal_x_max, args.goal_y_min, args.goal_y_max]):
        if None in [args.goal_x_min, args.goal_x_max, args.goal_y_min, args.goal_y_max]:
            raise ValueError('goal_range override requires goal_x_min, goal_x_max, goal_y_min, goal_y_max')
        if args.goal_x_min > args.goal_x_max:
            raise ValueError('goal_x_min must be <= goal_x_max')
        if args.goal_y_min > args.goal_y_max:
            raise ValueError('goal_y_min must be <= goal_y_max')
        env_cfg['goal_range'] = {
            'x_min': float(args.goal_x_min),
            'x_max': float(args.goal_x_max),
            'y_min': float(args.goal_y_min),
            'y_max': float(args.goal_y_max),
        }
        env_cfg['goal_mode'] = 'random'

    if args.goal_x is not None and args.goal_y is not None:
        env_cfg['goal_range'] = {
            'x_min': float(args.goal_x),
            'x_max': float(args.goal_x),
            'y_min': float(args.goal_y),
            'y_max': float(args.goal_y),
        }
        env_cfg['goal_mode'] = 'fixed'

    opponent_cfg = dict(env_cfg.get('opponent', {}))
    if args.opponent_mode is not None:
        opponent_cfg['mode'] = args.opponent_mode
    if opponent_cfg:
        env_cfg['opponent'] = opponent_cfg

    if args.opponent_goal_offset is not None:
        env_cfg['goal_offset'] = float(args.opponent_goal_offset)

    return env_cfg


def evaluate(network, env, device, eval_episodes=10, step_limit=5000):
    avg_reward = 0.0
    collision_episodes = 0
    success_episodes = 0
    timeout_episodes = 0
    print(f"Starting evaluation over {eval_episodes} episodes...")

    for i in range(eval_episodes):
        count = 0
        obs_td = env.reset()
        state = obs_td['policy'][0].cpu().numpy()
        done = False
        episode_reward = 0.0
        episode_collided = False
        episode_success = False

        while not done and count < step_limit:
            action = network.get_action(np.array(state))
            a_in = np.array([(action[0] + 1) / 2.0, action[1]])

            action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
            next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)

            state = next_obs_td['policy'][0].cpu().numpy()
            reward = float(reward_tensor[0].cpu().item())
            done = bool(done_tensor[0].cpu().item())

            episode_reward += reward
            count += 1

            if reward < -90:
                episode_collided = True
            elif reward > 90:
                episode_success = True

        avg_reward += episode_reward
        if episode_collided:
            collision_episodes += 1
        if episode_success:
            success_episodes += 1
        if not done and count >= step_limit:
            timeout_episodes += 1

        print(f"Episode {i + 1}: Reward={episode_reward:.2f}, Steps={count}, Collided={episode_collided}, Success={episode_success}")

    avg_reward /= eval_episodes
    collision_rate = collision_episodes / eval_episodes
    success_rate = success_episodes / eval_episodes
    timeout_rate = timeout_episodes / eval_episodes

    print("..............................................")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Collision Rate: {collision_rate:.4f}")
    print(f"Timeout Rate: {timeout_rate:.4f}")
    print("..............................................")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file (without _actor.pth suffix)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=None, help='Override max_episode_length from yaml')
    parser.add_argument('--goal_x', type=float, default=None, help='Fixed goal x in map frame')
    parser.add_argument('--goal_y', type=float, default=None, help='Fixed goal y in map frame')
    parser.add_argument('--goal_x_min', type=float, default=None, help='Goal range x lower bound override')
    parser.add_argument('--goal_x_max', type=float, default=None, help='Goal range x upper bound override')
    parser.add_argument('--goal_y_min', type=float, default=None, help='Goal range y lower bound override')
    parser.add_argument('--goal_y_max', type=float, default=None, help='Goal range y upper bound override')
    parser.add_argument('--opponent_mode', type=str, choices=['movebase', 'diffusion', 'rule_based'], default=None, help='Override opponent mode from yaml')
    parser.add_argument('--opponent_goal_offset', type=float, default=None, help='Override opponent goal offset from yaml')
    args = parser.parse_args()

    if args.goal_x is not None and args.goal_y is None:
        print('Invalid goal configuration: both goal_x and goal_y are required for fixed goal')
        sys.exit(2)
    if args.goal_y is not None and args.goal_x is None:
        print('Invalid goal configuration: both goal_x and goal_y are required for fixed goal')
        sys.exit(2)
    if args.goal_x is not None and any(v is not None for v in [args.goal_x_min, args.goal_x_max, args.goal_y_min, args.goal_y_max]):
        print('Invalid goal configuration: choose either fixed goal or goal_range override')
        sys.exit(2)

    cfg = load_yaml(args.config)
    try:
        env_cfg = apply_eval_overrides(cfg['env'], args)
    except ValueError as exc:
        print(f"Invalid goal configuration: {exc}")
        sys.exit(2)

    device = torch.device(args.device)
    env = MoveBaseGazeboEnv(env_cfg, device=args.device)
    time.sleep(2)

    state_dim = env_cfg.get('num_observations')
    action_dim = env_cfg.get('num_actions')
    max_action = 1
    network = TD3(state_dim, action_dim, max_action, device=device)

    model_dir = os.path.dirname(args.model_path)
    model_name = os.path.basename(args.model_path)
    print(f"Loading model: {model_name} from {model_dir}")

    try:
        network.load(model_name, model_dir)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    try:
        evaluate(network, env, device, args.episodes, args.max_steps or env_cfg.get('max_episode_length', 500))
    except KeyboardInterrupt:
        print("Evaluation interrupted.")
