import os
import time
import argparse
import csv
import yaml
import numpy as np
import torch
import subprocess
import sys
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

proj_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(proj_root, 'src'))

from rl_training.third_party.drl_td3.replay_buffer import ReplayBuffer
from rl_training.third_party.drl_td3.td3_models import TD3, Actor, Critic
from rl_training.envs.movebase_gazebo_env import MoveBaseGazeboEnv


def get_git_info():
    """Get current git commit hash and branch info."""
    git_info = {
        'commit': 'unknown',
        'branch': 'unknown',
        'is_dirty': False
    }
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=proj_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['commit'] = result.stdout.strip()[:8]

        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=proj_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()

        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=proj_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        git_info['is_dirty'] = len(result.stdout.strip()) > 0
    except Exception as e:
        print(f"[Warning] Could not get git info: {e}")

    return git_info


def save_experiment_config(log_dir, cfg, env_cfg, runner_cfg, hyperparams, git_info, resume_path=None):
    """Save complete experiment configuration to log_dir/experiment_config.yaml."""
    opponent_cfg = env_cfg.get('opponent', {})
    opponent_config = {
        'enabled': opponent_cfg.get('enabled', False),
        'mode': opponent_cfg.get('mode', 'none'),
    }
    if opponent_cfg.get('mode') == 'rule_based':
        opponent_config['behaviors'] = ['head_on', 'cross', 'follow_stop']
        opponent_config['params'] = opponent_cfg.get('rule_based', {})

    exp_config = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'git_commit': git_info['commit'],
            'git_branch': git_info['branch'],
            'git_dirty': git_info['is_dirty'],
            'resume_from': resume_path,
        },
        'environment': {
            'env_class': env_cfg.get('env_class', 'MoveBaseGazeboEnv'),
            'robot': cfg.get('robot_type', 'unknown'),
            'map': cfg.get('map', 'unknown'),
            'num_observations': env_cfg.get('num_observations'),
            'num_actions': env_cfg.get('num_actions'),
            'control_dt': env_cfg.get('control_dt', 0.1),
            'max_episode_length': env_cfg.get('max_episode_length', 500),
            'collision_dist': env_cfg.get('collision_dist', 0.35),
            'goal_reached_dist': env_cfg.get('goal_reached_dist', 0.3),
            'curriculum': env_cfg.get('curriculum', {}),
        },
        'opponent': opponent_config,
        'network': {
            'actor': {
                'layers': [env_cfg.get('num_observations', 24), 800, 600, env_cfg.get('num_actions', 2)],
                'activation': 'ReLU',
                'output_activation': 'Tanh',
            },
            'critic': {
                'type': 'TwinQ',
                'description': 'state->800->600, action->600, concat->1',
            }
        },
        'hyperparameters': hyperparams,
        'training': {
            'max_timesteps': hyperparams.get('max_timesteps'),
            'eval_freq': hyperparams.get('eval_freq'),
            'eval_episodes': hyperparams.get('eval_episodes'),
            'random_near_obstacle': hyperparams.get('random_near_obstacle'),
            'stuck_threshold': hyperparams.get('stuck_threshold'),
        },
        'runner': {
            'experiment_name': runner_cfg.get('experiment_name'),
            'run_name': runner_cfg.get('run_name'),
            'seed': runner_cfg.get('seed', 1),
        }
    }

    config_path = os.path.join(log_dir, 'experiment_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False, allow_unicode=True)

    print(f"[train] Saved experiment config to {config_path}")
    return config_path


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_goal(args):
    if args.goal_x is None or args.goal_y is None:
        return None
    goal_yaw = args.goal_yaw
    if goal_yaw is None:
        goal_yaw = float(np.arctan2(args.goal_y, args.goal_x))
    return [float(args.goal_x), float(args.goal_y), float(goal_yaw)]


def apply_train_overrides(env_cfg, args):
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


def append_episode_result(csv_path, row):
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    fieldnames = [
        'episode',
        'step',
        'episode_reward',
        'collided',
        'reached_goal',
        'timed_out',
        'episode_length',
        'avg_reward_so_far',
    ]

    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(network, env, device, eval_episodes=10, step_limit=5000):
    avg_reward = 0.0
    collision_episodes = 0
    goal_episodes = 0
    timeout_episodes = 0
    print(f"Starting evaluation over {eval_episodes} episodes...")

    for episode_idx in range(eval_episodes):
        obs_td = env.reset()
        state = obs_td['policy'][0].cpu().numpy()
        done = False
        step_count = 0
        episode_reward = 0.0
        collided = False
        reached_goal = False

        while not done and step_count < step_limit:
            action = network.get_action(np.array(state))
            a_in = np.array([(action[0] + 1) / 2.0, action[1]])

            action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
            next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)

            state = next_obs_td['policy'][0].cpu().numpy()
            reward = float(reward_tensor[0].cpu().item())
            done = bool(done_tensor[0].cpu().item())
            episode_reward += reward
            step_count += 1

            if reward < -90:
                collided = True
            elif reward > 90:
                reached_goal = True

        if not done and step_count >= step_limit:
            timeout_episodes += 1
        if collided:
            collision_episodes += 1
        if reached_goal:
            goal_episodes += 1

        print(
            f"Eval episode {episode_idx + 1}: reward={episode_reward:.2f}, "
            f"steps={step_count}, collided={collided}, reached_goal={reached_goal}"
        )

        avg_reward += episode_reward

    avg_reward /= eval_episodes
    collision_rate = collision_episodes / eval_episodes
    success_rate = goal_episodes / eval_episodes
    timeout_rate = timeout_episodes / eval_episodes

    print("..............................................")
    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Collision Rate: {collision_rate:.4f}")
    print(f"Timeout Rate: {timeout_rate:.4f}")
    print("..............................................")

    return avg_reward, collision_rate, success_rate, timeout_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/forklift_movebase.yaml", help='Path to config yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_steps', type=int, default=None, help='Override max_episode_length from yaml (None means use yaml value)')
    parser.add_argument('--max_timesteps', type=int, default=int(5e6), help='Total training timesteps')
    parser.add_argument('--eval_freq', type=int, default=int(5e3), help='Evaluation frequency (steps)')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size')
    parser.add_argument('--discount', type=float, default=0.99999, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update parameter')
    parser.add_argument('--policy_noise', type=float, default=0.2, help='Policy noise')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='Noise clip')
    parser.add_argument('--policy_freq', type=int, default=2, help='Policy update frequency')
    parser.add_argument('--expl_noise_initial', type=float, default=1.0, help='Initial exploration noise')
    parser.add_argument('--expl_noise_min', type=float, default=0.1, help='Minimum exploration noise')
    parser.add_argument('--expl_decay_steps', type=int, default=500000, help='Exploration decay steps')
    parser.add_argument('--goal_x', type=float, default=None, help='Fixed goal x in map frame')
    parser.add_argument('--goal_y', type=float, default=None, help='Fixed goal y in map frame')
    parser.add_argument('--goal_yaw', type=float, default=None, help='Fixed goal yaw in map frame')
    parser.add_argument('--goal_x_min', type=float, default=None, help='Goal range x lower bound override')
    parser.add_argument('--goal_x_max', type=float, default=None, help='Goal range x upper bound override')
    parser.add_argument('--goal_y_min', type=float, default=None, help='Goal range y lower bound override')
    parser.add_argument('--goal_y_max', type=float, default=None, help='Goal range y upper bound override')
    parser.add_argument('--opponent_mode', type=str, choices=['movebase', 'diffusion', 'rule_based'], default=None, help='Override opponent mode from yaml')
    parser.add_argument('--opponent_goal_offset', type=float, default=None, help='Override opponent goal offset from yaml')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume training from')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--experiment_name', type=str, default="forklift_movebase_with_goal", help='Experiment name')
    parser.add_argument('--run_name', type=str, default="run_1", help='Run name')
    parser.add_argument('--csv_path', type=str, default=None, help='CSV path for recording training episodes')
    args = parser.parse_args()

    if args.goal_x is not None and args.goal_y is None:
        print('Error: both goal_x and goal_y are required for fixed goal')
        sys.exit(2)
    if args.goal_y is not None and args.goal_x is None:
        print('Error: both goal_x and goal_y are required for fixed goal')
        sys.exit(2)
    if args.goal_x is not None and any(v is not None for v in [args.goal_x_min, args.goal_x_max, args.goal_y_min, args.goal_y_max]):
        print('Error: choose either fixed goal or goal_range override')
        sys.exit(2)

    # Load config
    cfg = load_yaml(args.config)
    try:
        env_cfg = apply_train_overrides(cfg['env'], args)
    except ValueError as exc:
        print(f"Invalid goal configuration: {exc}")
        sys.exit(2)
    runner_cfg = cfg['runner']

    # Set max_steps from yaml if not provided via CLI
    max_steps = args.max_steps if args.max_steps is not None else env_cfg.get('max_episode_length', 500)
    print(f"[train] Using max_steps: {max_steps}")

    # Setup device and environment
    device = torch.device(args.device)
    env = MoveBaseGazeboEnv(env_cfg, device=args.device)
    time.sleep(2)

    # Setup logging
    log_dir = osp.join(os.path.dirname(__file__), 'logs', args.experiment_name, args.run_name)
    os.makedirs(log_dir, exist_ok=True)

    git_info = get_git_info()

    hyperparams = {
        'batch_size': args.batch_size,
        'discount': args.discount,
        'tau': args.tau,
        'policy_noise': args.policy_noise,
        'noise_clip': args.noise_clip,
        'policy_freq': args.policy_freq,
        'expl_noise_initial': args.expl_noise_initial,
        'expl_noise_min': args.expl_noise_min,
        'expl_decay_steps': args.expl_decay_steps,
        'max_timesteps': args.max_timesteps,
        'eval_freq': args.eval_freq,
        'eval_episodes': args.eval_episodes,
    }

    user_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'user_config/user_config.yaml'
    )
    user_cfg = {}
    if os.path.exists(user_config_path):
        with open(user_config_path, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}

    save_experiment_config(
        log_dir=log_dir,
        cfg=user_cfg,
        env_cfg=env_cfg,
        runner_cfg=runner_cfg,
        hyperparams=hyperparams,
        git_info=git_info,
        resume_path=args.resume_path
    )

    writer = SummaryWriter(log_dir)
    print(f"[train] Tensorboard logs to {log_dir}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create model
    state_dim = env_cfg.get('num_observations')
    action_dim = env_cfg.get('num_actions')
    max_action = 1

    network = TD3(state_dim, action_dim, max_action, device=device, log_dir=log_dir)
    replay_buffer = ReplayBuffer(1e6, random_seed=args.seed)

    # Resume training if specified
    expl_noise = args.expl_noise_initial
    if args.resume_path:
        if os.path.isfile(args.resume_path + "_actor.pth"):
            print(f"Loading model from {args.resume_path}...")
            try:
                network.load(os.path.basename(args.resume_path), os.path.dirname(args.resume_path))
                print(f"Successfully loaded model from {args.resume_path}")
                expl_noise = args.expl_noise_min
            except Exception as e:
                print(f"Could not load model: {e}")
        else:
            print(f"Resume path file not found: {args.resume_path}_actor.pth")

    # Training loop
    evaluations = []
    eval_reward_history = []
    eval_col_history = []
    eval_success_history = []
    eval_timeout_history = []
    train_collision_history = []
    train_collision_total_history = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1
    episode_reward = 0
    episode_timesteps = 0
    episode_collisions = 0
    count_rand_actions = 0
    random_action = []
    train_collision_count_total = 0
    collisions_since_last_eval = 0

    if args.csv_path is None:
        args.csv_path = os.path.join(log_dir, 'training_episodes.csv')

    print(f"Starting training for {args.max_timesteps} timesteps...")
    print(f"Goal config: mode={env_cfg.get('goal_mode', 'random')}, range={env_cfg.get('goal_range')}")

    while timestep < args.max_timesteps:
        if done:
            if timestep != 0:
                writer.add_scalar('train/episode_reward', episode_reward, episode_num + 1)
                writer.add_scalar('train/episode_length', episode_timesteps, episode_num + 1)
                writer.add_scalar('train/episode_collisions', episode_collisions, episode_num + 1)
                writer.add_scalar('train/replay_size', replay_buffer.size(), episode_num + 1)
                writer.add_scalar('train/exploration_noise', expl_noise, episode_num + 1)

                if args.csv_path:
                    append_episode_result(
                        args.csv_path,
                        {
                            'episode': episode_num,
                            'step': timestep,
                            'episode_reward': episode_reward,
                            'collided': int(episode_collisions > 0),
                            'reached_goal': int(episode_reward > 90),
                            'timed_out': 0,
                            'episode_length': episode_timesteps,
                            'avg_reward_so_far': np.mean(eval_reward_history) if eval_reward_history else 0,
                        },
                    )

                network.train(
                    replay_buffer=replay_buffer,
                    iterations=episode_timesteps,
                    batch_size=args.batch_size,
                    discount=args.discount,
                    tau=args.tau,
                    policy_noise=args.policy_noise,
                    noise_clip=args.noise_clip,
                    policy_freq=args.policy_freq,
                )

            if timesteps_since_eval >= args.eval_freq:
                print(f"\n[Epoch {epoch}] Running evaluation...")
                timesteps_since_eval %= args.eval_freq
                avg_reward, col_rate, success_rate, timeout_rate = evaluate(
                    network=network,
                    env=env,
                    device=device,
                    eval_episodes=args.eval_episodes,
                    step_limit=max_steps,
                )
                evaluations.append(avg_reward)
                eval_reward_history.append(avg_reward)
                eval_col_history.append(col_rate)
                eval_success_history.append(success_rate)
                eval_timeout_history.append(timeout_rate)
                train_collision_history.append(collisions_since_last_eval)
                train_collision_total_history.append(train_collision_count_total)

                writer.add_scalar('eval/avg_reward', avg_reward, epoch)
                writer.add_scalar('eval/collision_rate', col_rate, epoch)
                writer.add_scalar('eval/success_rate', success_rate, epoch)
                writer.add_scalar('eval/timeout_rate', timeout_rate, epoch)
                writer.add_scalar('train/collisions_interval', collisions_since_last_eval, epoch)
                writer.add_scalar('train/collisions_total', train_collision_count_total, epoch)

                collisions_since_last_eval = 0
                network.save("td3_model", directory=log_dir)
                np.save(os.path.join(log_dir, "evaluations.npy"), evaluations)
                np.save(os.path.join(log_dir, "eval_reward_history.npy"), eval_reward_history)
                np.save(os.path.join(log_dir, "eval_col_history.npy"), eval_col_history)
                np.save(os.path.join(log_dir, "eval_success_history.npy"), eval_success_history)
                np.save(os.path.join(log_dir, "eval_timeout_history.npy"), eval_timeout_history)
                np.save(os.path.join(log_dir, "train_collision_history.npy"), train_collision_history)
                np.save(os.path.join(log_dir, "train_collision_total_history.npy"), train_collision_total_history)
                epoch += 1

            obs_td = env.reset()
            state = obs_td['policy'][0].cpu().numpy()
            done = False

            episode_reward = 0
            episode_timesteps = 0
            episode_collisions = 0
            episode_num += 1

        # Update exploration noise
        if expl_noise > args.expl_noise_min:
            expl_noise = expl_noise - ((args.expl_noise_initial - args.expl_noise_min) / args.expl_decay_steps)

        # Select action
        action = network.get_action(np.array(state))
        action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

        # Random action near obstacles
        if np.random.uniform(0, 1) > 0.85 and np.min(state[:20]) < 0.6 and count_rand_actions < 1:
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

        a_in = np.array([(action[0] + 1) / 2.0, action[1]])

        # Step environment
        action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
        next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)

        next_state = next_obs_td['policy'][0].cpu().numpy()
        reward = float(reward_tensor[0].cpu().item())

        done_val = bool(done_tensor[0].cpu().item())
        done = 1 if done_val else 0
        done_bool = 0 if episode_timesteps + 1 == max_steps else int(done)

        episode_reward += reward
        replay_buffer.add(state, action, reward, done_bool, next_state)

        state = next_state
        episode_timesteps += 1
        timestep += 1
        timesteps_since_eval += 1

        # Track collisions
        if reward < -90:
            train_collision_count_total += 1
            collisions_since_last_eval += 1
            episode_collisions += 1

    # Final evaluation
    print("\nFinal evaluation...")
    avg_reward, col_rate, success_rate, timeout_rate = evaluate(
        network=network,
        env=env,
        device=device,
        eval_episodes=args.eval_episodes,
        step_limit=max_steps,
    )
    evaluations.append(avg_reward)
    eval_reward_history.append(avg_reward)
    eval_col_history.append(col_rate)
    eval_success_history.append(success_rate)
    eval_timeout_history.append(timeout_rate)

    network.save("td3_model_final", directory=log_dir)
    np.save(os.path.join(log_dir, "evaluations.npy"), evaluations)
    np.save(os.path.join(log_dir, "eval_reward_history.npy"), eval_reward_history)
    np.save(os.path.join(log_dir, "eval_col_history.npy"), eval_col_history)
    np.save(os.path.join(log_dir, "eval_success_history.npy"), eval_success_history)
    np.save(os.path.join(log_dir, "eval_timeout_history.npy"), eval_timeout_history)

    writer.flush()
    writer.close()

    print(f"\nTraining completed. Results saved to {log_dir}")
