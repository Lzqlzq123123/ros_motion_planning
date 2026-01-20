import argparse
import os
import sys
import yaml
import torch

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add local rsl_rl to path to use custom modified version
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rsl_rl_path = os.path.join(project_root, 'third_party', 'rsl_rl')
if os.path.exists(rsl_rl_path):
    sys.path.insert(0, rsl_rl_path)
    print(f"Using local rsl_rl from: {rsl_rl_path}")

from rsl_rl.runners import OnPolicyRunner

from envs.ros_gazebo_env import RosGazeboEnv

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_user_config_init_poses():
    # Path to user_config.yaml relative to this script
    # src/rl_training/train_new.py -> src/user_config/user_config.yaml
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to training config yaml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load Config
    cfg = load_yaml(args.config)
    env_cfg = cfg['env']
    runner_cfg = cfg['runner']
    
    # Inject init poses from user_config.yaml
    init_poses = get_user_config_init_poses()
    if init_poses:
        print(f"Loaded init poses from user_config: {init_poses}")
        env_cfg['init_poses'] = init_poses

    # Create Environment
    env = RosGazeboEnv(env_cfg, device=args.device)

    # Reset once so goals/plans are initialized before training rollouts
    env.reset()

    # Create Runner
    log_dir = os.path.join(os.path.dirname(__file__), 'logs', runner_cfg['experiment_name'], runner_cfg['run_name'])
    os.makedirs(log_dir, exist_ok=True)
    
    # Disable the runner's internal resume mechanism since we handle it manually
    runner_cfg['resume'] = False
    runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=args.device)

    # Manually load checkpoint if resume_path is provided in the config
    resume_path = runner_cfg.get("resume_path")
    if resume_path and os.path.exists(resume_path):
        print(f"Loading model from: {resume_path}")
        try:
            runner.load(resume_path, load_optimizer=False)
            # Reset counters so logging starts from iteration 0 for this run
            runner.current_learning_iteration = 0
            runner.tot_timesteps = 0
        except Exception as e:
            print(f"Error loading model from {resume_path}: {e}")
            print("Starting training from scratch.")

    # Train
    runner.learn(num_learning_iterations=runner_cfg['max_iterations'])

if __name__ == '__main__':
    main()
