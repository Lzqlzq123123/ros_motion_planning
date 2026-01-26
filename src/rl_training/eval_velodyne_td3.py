import os
import argparse
import yaml
import numpy as np
import torch
import time
import sys
import os.path as osp

# Imports
proj_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(proj_root, 'src'))

from rl_training.third_party.drl_td3.td3_models import TD3
from rl_training.envs.movebase_gazebo_env import MoveBaseGazeboEnv

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_user_config_init_poses():
    # Path to user_config.yaml relative to this script
    # src/rl_training/eval_velodyne_td3.py -> src/user_config/user_config.yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'user_config/user_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Warning: user_config.yaml not found at {config_path}")
        return {}
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        init_poses = {}
        if 'robots_config' in config:
            for robot_conf in config['robots_config']:
                # Find keys like robot1_x_pos
                keys = list(robot_conf.keys())
                if not keys:
                    continue
                
                # Extract ID
                import re
                match = re.match(r'robot(\d+)_', keys[0])
                if match:
                    robot_id = match.group(1)
                    name = f"robot{robot_id}"
                    
                    x = float(robot_conf.get(f'{name}_x_pos', 0.0))
                    y = float(robot_conf.get(f'{name}_y_pos', 0.0))
                    yaw = float(robot_conf.get(f'{name}_yaw', 0.0))
                    
                    init_poses[name] = [x, y, yaw]
        return init_poses
    except Exception as e:
        print(f"Error parsing user_config.yaml: {e}")
        return {}

def evaluate(network, env, device, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    print(f"Starting evaluation over {eval_episodes} episodes...")
    
    for i in range(eval_episodes):
        count = 0
        obs_td = env.reset()
        state = obs_td['policy'][0].cpu().numpy()
        done = False
        episode_reward = 0
        
        while not done and count < 5000: # Safety break
            # Get action from network (deterministic for evaluation)
            action = network.get_action(np.array(state))
            a_in = np.array([(action[0] + 1) / 2.0, action[1]])
            
            # Step env
            action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
            next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)
            
            state = next_obs_td['policy'][0].cpu().numpy()
            reward = float(reward_tensor[0].cpu().item())
            done = bool(done_tensor[0].cpu().item())
            
            episode_reward += reward
            count += 1
            
            # Simple collision check logic based on reward
            if reward < -90:
                col += 1
        
        avg_reward += episode_reward
        print(f"Episode {i+1}: Reward: {episode_reward:.2f}, Steps: {count}")

    avg_reward /= eval_episodes
    avg_col = col / eval_episodes # Determine collision rate differently if needed? Or simply check if reward < -90 occurred in episode?
    # In original script, col is incremented per step with collision. 
    # If we want collision RATE (episodes with collision / total episodes), we need to track per episode.
    # But sticking to original logic if 'col' accumulates counts.
    # Ideally col should count EPISODES with collision.
    
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes: %f, Avg Collision Impact: %f"
        % (eval_episodes, avg_reward, avg_col)
    )
    print("..............................................")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file (without _actor.pth suffix)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env_cfg = cfg['env']
    
    device = torch.device(args.device)
    
    # Inject init poses from user_config.yaml
    init_poses = get_user_config_init_poses()
    if init_poses:
        print(f"Loaded init poses from user_config: {init_poses}")
        env_cfg['init_poses'] = init_poses
    
    # Environment
    env = MoveBaseGazeboEnv(env_cfg, device=args.device)
    time.sleep(2) # Wait for connections
    
    state_dim = env_cfg.get('num_observations')
    action_dim = env_cfg.get('num_actions')
    max_action = 1
    
    # Network
    # Note: log_dir is not used for loading, but TD3 init might require it if we were training to write logs. 
    # Logic in td3_models.py allows log_dir=None.
    network = TD3(state_dim, action_dim, max_action, device=device)
    
    # Load Model
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
        evaluate(network, env, device, args.episodes)
    except KeyboardInterrupt:
        print("Evaluation interrupted.")
