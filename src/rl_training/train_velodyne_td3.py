import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Imports from third_party to ensure identical models/buffer
import sys
import os.path as osp
proj_root = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(proj_root, 'src'))

from rl_training.third_party.drl_td3.replay_buffer import ReplayBuffer
from rl_training.third_party.drl_td3.td3_models import TD3, Actor, Critic

# Use local environment
from rl_training.envs.movebase_gazebo_env import MoveBaseGazeboEnv

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Helper function to align loading of config with original script params
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

cfg = load_yaml(args.config)
env_cfg = cfg['env']
runner_cfg = cfg['runner']
algo_cfg = runner_cfg.get('algorithm', {})

# Set the parameters for the implementation
device = torch.device(args.device)
seed = int(runner_cfg.get('seed', 1)) # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = env_cfg.get('max_episode_length', 500)  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = 500000  # Number of steps over which the initial exploration noise will decay over
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
save_model = True  # Weather to save the model or not
random_near_obstacle = True  # To take random actions near obstacles or not
resume_path = runner_cfg.get('resume_path', None)

# Update params from config if present (to respect forklift_movebase.yaml settings while keeping logic)
if 'max_iterations' in runner_cfg:
    # Approximate mapping: max_iterations * num_steps_per_env -> max_timesteps
    max_timesteps = int(runner_cfg['max_iterations']) * int(runner_cfg.get('num_steps_per_env', 500))

# Logging & save path setup to match project structure AND legacy structure
experiment = runner_cfg.get('experiment_name', 'td3_experiment')
run_name = runner_cfg.get('run_name', 'run_1')
log_dir = osp.join(os.path.dirname(__file__), 'logs', experiment, run_name)
os.makedirs(log_dir, exist_ok=True)


# Create the training environment
# env = GazeboEnv("multi_robot_scenario.launch", environment_dim) -> Replaced by MoveBaseGazeboEnv
env = MoveBaseGazeboEnv(env_cfg, device=args.device)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env_cfg.get('num_observations')
action_dim = env_cfg.get('num_actions')
max_action = 1

# Create the network
# Passed log_dir to TD3 to ensure Tensorboard writes to correct location
network = TD3(state_dim, action_dim, max_action, device=device, log_dir=log_dir)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, random_seed=seed)
if resume_path:
    if os.path.isfile(resume_path + "_actor.pth"):
        print(f"Loading model from {resume_path}...")
        try:
            network.load(os.path.basename(resume_path), os.path.dirname(resume_path))
            print(f"Successfully loaded model from {resume_path}")
            # If resuming a trained model, we don't want high exploration noise
            print("Resuming training: Setting exploration noise to minimum to preserve policy performance.")
            expl_noise = expl_min
        except Exception as e:
            print(f"Could not load model parameters: {e}")
    else:
        print(f"Resume path specified but file not found: {resume_path}_actor.pth")


def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        obs_td = env.reset()
        state = obs_td['policy'][0].cpu().numpy()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = np.array([(action[0] + 1) / 2.0, action[1]])
            
            # Step env
            action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
            next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)
            
            state = next_obs_td['policy'][0].cpu().numpy()
            reward = float(reward_tensor[0].cpu().item())
            done = bool(done_tensor[0].cpu().item())
            
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin the training loop
while timestep < max_timesteps:

    # On termination of episode
    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            # Save to project log_dir
            network.save("td3_model", directory=log_dir)
            np.save(os.path.join(log_dir, "evaluations.npy"), evaluations)
            epoch += 1

        # state = env.reset()
        # MoveBaseGazeboEnv manual reset handling
        obs_td = env.reset()
        state = obs_td['policy'][0].cpu().numpy()
        
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        
    # add some exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    if random_near_obstacle:
        # Adaptation for MoveBaseGazeboEnv observation structure
        # Heuristic slice for obstacle check
        lidar_end = 20 if state_dim > 20 else state_dim 
        try:
            if np.random.uniform(0, 1) > 0.85 and np.min(state[:lidar_end]) < 0.6 and count_rand_actions < 1:
                count_rand_actions = np.random.randint(8, 15)
                random_action = np.random.uniform(-1, 1, 2)
        except:
             pass

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = np.array([(action[0] + 1) / 2.0, action[1]])
    
    # next_state, reward, done, target = env.step(a_in)
    action_tensor = torch.tensor(a_in, dtype=torch.float32, device=device).unsqueeze(0)
    next_obs_td, reward_tensor, done_tensor, _ = env.step(action_tensor)
    
    next_state = next_obs_td['policy'][0].cpu().numpy()
    reward = float(reward_tensor[0].cpu().item())
    
    done_val = bool(done_tensor[0].cpu().item())
    
    done = 1 if done_val else 0 
    
    # Original logic: done_bool is 0 if episode length reached, else int(done)
    # env uses max_episode_length internally if configured, can we detect timeout?
    # MoveBaseGazeboEnv already resets on timeout? 
    # Actually MoveBaseGazeboEnv sets done=True on timeout.
    # To strictly match original, we should check max_ep:
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    
    episode_reward += reward

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("td3_model_final", directory=log_dir)
np.save(os.path.join(log_dir, "evaluations.npy"), evaluations)
