#!/usr/bin/env python3
"""
Minimal training test to debug episode termination
"""
import os
import sys
import yaml
import torch
import numpy as np
import os.path as osp

proj_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(proj_root, 'src'))

# Must set before importing ROS
if 'ROS_MASTER_URI' not in os.environ:
    os.environ['ROS_MASTER_URI'] = 'http://localhost:11311'
if 'ROS_IP' not in os.environ:
    os.environ['ROS_IP'] = '127.0.0.1'

import rospy
from rl_training.envs.movebase_gazebo_env import MoveBaseGazeboEnv


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_user_config_init_poses():
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'src/user_config/user_config.yaml',
    )
    
    if not os.path.exists(config_path):
        return {}
    
    try:
        user_cfg = load_yaml(config_path)
        init_poses = {}
        if 'robots_config' in user_cfg:
            for robot_conf in user_cfg['robots_config']:
                keys = list(robot_conf.keys())
                if not keys:
                    continue
                
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
        print(f"Error loading init_poses: {e}")
        return {}


def main():
    print("="*70)
    print("MINIMAL TRAINING TEST - EPISODE TERMINATION DEBUG")
    print("="*70)
    
    # Load config
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'src/rl_training/config/forklift_movebase.yaml'
    )
    cfg = load_yaml(cfg_path)
    env_cfg = cfg['env']
    
    # Load init_poses
    init_poses = get_user_config_init_poses()
    if init_poses:
        print(f"\nLoaded init_poses: {init_poses}")
        env_cfg['init_poses'] = init_poses
    else:
        print("\nWARNING: No init_poses loaded")
    
    # Create environment
    print("\nCreating environment...")
    device = torch.device('cpu')
    env_cfg['debug'] = True  # Enable debug logging
    env = MoveBaseGazeboEnv(env_cfg, device='cpu')
    
    print(f"Max Episode Length: {env.max_episode_length}")
    print(f"Num Envs: {env.num_envs}")
    print(f"episode_length_buf shape: {env.episode_length_buf.shape}")
    print(f"reset_buf shape: {env.reset_buf.shape}")
    
    # Test 3 episodes
    for episode in range(3):
        print(f"\n{'='*70}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*70}")
        
        print("Calling reset()...")
        obs_td = env.reset()
        
        print(f"After reset:")
        print(f"  episode_length_buf: {env.episode_length_buf.tolist()}")
        print(f"  reset_buf: {env.reset_buf.tolist()}")
        
        done = False
        step = 0
        max_steps = 500
        
        while not done and step < max_steps:
            # Random action
            action = np.random.uniform(-1, 1, 2)
            a_in = np.array([(action[0] + 1) / 2.0, action[1]])
            action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Step
            next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)
            
            done = bool(done_tensor[0].cpu().item())
            ep_len = int(env.episode_length_buf[0].item())
            rew = float(reward_tensor[0].cpu().item())
            step += 1
            
            if step % 50 == 1 or done or step <= 5:
                print(f"  Step {step:3d}: ep_len={ep_len:3d}, reward={rew:7.2f}, done={done}")
        
        print(f"\nEpisode {episode + 1} ended after {step} steps")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
