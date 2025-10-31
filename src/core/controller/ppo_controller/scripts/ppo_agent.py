#!/usr/bin/env python3
"""
PPO Agent for ROS Navigation using stable-baselines3
This script handles the PPO training and inference for local navigation
"""

import rospy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, Figure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os
import json
import time

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Float32, Bool


class ROSNavigationEnv(gym.Env):
    """
    Custom Gym environment for ROS navigation using PPO
    Interfaces with ROS topics to get state and send actions
    """
    
    def __init__(self, node_name="ppo_navigation_env"):
        super(ROSNavigationEnv, self).__init__()
        
        # Initialize ROS node only if not already initialized
        try:
            rospy.get_master().getSystemState()
        except:
            rospy.init_node(node_name, anonymous=True)
        
        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0]), 
            high=np.array([0.8, 1.0]), 
            dtype=np.float32
        )
        
        # Observation space: [laser_scan(360), pose(3), goal(3), velocity(2)]
        self.laser_dim = 360
        self.pose_dim = 3  # x, y, theta
        self.goal_dim = 3  # relative dx, dy, dtheta
        self.vel_dim = 2   # linear, angular
        obs_dim = self.laser_dim + self.pose_dim + self.goal_dim + self.vel_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # ROS Publishers and Subscribers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.state_sub = rospy.Subscriber('/ppo/state', Float32MultiArray, self.state_callback)
        self.reward_sub = rospy.Subscriber('/ppo/reward', Float32, self.reward_callback)
        self.done_sub = rospy.Subscriber('/ppo/done', Bool, self.done_callback)
        
        # State variables
        self.current_state = np.zeros(obs_dim, dtype=np.float32)
        self.current_reward = 0.0
        self.is_done = False
        self.state_received = False
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.episode_reward = 0.0
        
        rospy.loginfo("PPO Navigation Environment initialized")
    
    def state_callback(self, msg):
        """Callback for receiving state from C++ controller"""
        self.current_state = np.array(msg.data, dtype=np.float32)
        self.state_received = True
    
    def reward_callback(self, msg):
        """Callback for receiving reward from C++ controller"""
        self.current_reward = msg.data
    
    def done_callback(self, msg):
        """Callback for receiving done signal from C++ controller"""
        self.is_done = msg.data
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode - improved version based on navbot_ppo"""
        super().reset(seed=seed)
        
        # 发布重置信号给环境管理器
        reset_pub = rospy.Publisher('/ppo/env_reset', Bool, queue_size=1)
        reset_pub.publish(Bool(data=True))
        
        # 等待环境重置完成和新状态
        self.state_received = False
        reset_timeout = rospy.Time.now() + rospy.Duration(15.0)  # 增加超时时间以适应Gazebo重置
        
        while not self.state_received and rospy.Time.now() < reset_timeout:
            rospy.sleep(0.1)
        
        if not self.state_received:
            rospy.logwarn("Timeout waiting for environment reset, using zero state")
            self.current_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        else:
            rospy.loginfo("Environment reset completed, received new state")
            
        self.episode_count += 1
        self.step_count = 0
        self.episode_reward = 0.0
        self.is_done = False
        
        rospy.loginfo(f"Episode {self.episode_count} reset complete")
        return self.current_state.copy(), {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info - improved version"""
        # 发布动作到ROS
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)
        
        # 等待状态更新，使用更合理的超时和更新检测
        old_state = self.current_state.copy()
        timeout = rospy.Time.now() + rospy.Duration(3.0)  # 增加超时时间
        state_updated = False
        
        # 更好的状态更新检测
        while rospy.Time.now() < timeout and not rospy.is_shutdown():
            rospy.sleep(0.02)  # 50Hz更新率
            
            # 检查状态是否有显著变化
            state_diff = np.linalg.norm(self.current_state - old_state)
            if state_diff > 0.001:  # 状态变化阈值
                state_updated = True
                break
        
        if not state_updated:
            rospy.logwarn(f"State update timeout at step {self.step_count}, state_diff: {np.linalg.norm(self.current_state - old_state):.6f}")
        
        self.step_count += 1
        self.episode_reward += self.current_reward
        
        # 准备返回值
        next_state = self.current_state.copy()
        reward = self.current_reward
        done = self.is_done or self.step_count >= 500  # 最大步数限制
        
        # 奖励调整 - 参考navbot_ppo的奖励系统
        if done:
            if self.current_reward > 50:  # 假设大于50为成功到达
                reward = 120.0  # 成功奖励
                rospy.loginfo(f"Episode {self.episode_count} SUCCESS - Steps: {self.step_count}, Total Reward: {self.episode_reward:.2f}")
            elif self.current_reward < -50:  # 假设小于-50为碰撞
                reward = -100.0  # 碰撞惩罚
                rospy.logwarn(f"Episode {self.episode_count} COLLISION - Steps: {self.step_count}, Total Reward: {self.episode_reward:.2f}")
            else:
                reward = -10.0  # 超时惩罚
                rospy.loginfo(f"Episode {self.episode_count} TIMEOUT - Steps: {self.step_count}, Total Reward: {self.episode_reward:.2f}")
        
        info = {
            'episode': self.episode_count,
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'linear_vel': action[0],
            'angular_vel': action[1],
            'reward': reward,
            'state_updated': state_updated,
            'raw_reward': self.current_reward
        }
        
        return next_state, reward, done, False, info
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        pass
    
    def close(self):
        """Close the environment"""
        if hasattr(self, 'cmd_pub'):
            # Send zero velocity
            cmd = Twist()
            self.cmd_pub.publish(cmd)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging PPO training metrics to TensorBoard
    """
    
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log environment info
        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]
            
            # Log episode metrics
            if 'episode_reward' in info:
                self.logger.record('episode/reward', info['episode_reward'])
                
            if 'step' in info:
                self.logger.record('episode/length', info['step'])
                
            # Log action values
            if 'linear_vel' in info:
                self.logger.record('action/linear_velocity', info['linear_vel'])
            if 'angular_vel' in info:
                self.logger.record('action/angular_velocity', info['angular_vel'])
        
        # Log policy and value function losses
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            for key, value in self.model.logger.name_to_value.items():
                if 'loss' in key.lower():
                    self.logger.record(f'losses/{key}', value)
        
        return True
    
    def _on_training_start(self) -> None:
        """Log hyperparameters at the start of training"""
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "clip_range": self.model.clip_range,
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
        }
        
        metric_dict = {
            "episode/reward": 0.0,
            "episode/length": 0.0,
            "losses/policy_loss": 0.0,
            "losses/value_loss": 0.0
        }
        
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )


class PPONavigationAgent:
    """
    Main PPO agent class for navigation
    """
    
    def __init__(self, config_path=None):
        # Default configuration
        self.config = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'tensorboard_log': './ppo_navigation_tensorboard/',
            'model_save_path': './ppo_navigation_model.zip',
            'total_timesteps': 100000,
            'save_freq': 10000
        }
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        
        # Create environment
        self.env = ROSNavigationEnv()
        
        # Initialize model
        self.model = None
        self.is_training = False
        
        rospy.loginfo("PPO Navigation Agent initialized")
    
    def create_model(self):
        """Create PPO model with custom network architecture"""
        
        # Custom policy network for navigation
        policy_kwargs = dict(
            net_arch=[
                dict(pi=[256, 128, 64], vf=[256, 128, 64])
            ],
            activation_fn=nn.ReLU,
        )
        
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            tensorboard_log=self.config['tensorboard_log'],
            policy_kwargs=policy_kwargs,
            verbose=1
        )
        
        rospy.loginfo("PPO model created successfully")
    
    def train(self, total_timesteps=None):
        """Train the PPO agent"""
        if not self.model:
            self.create_model()
        
        if total_timesteps is None:
            total_timesteps = self.config['total_timesteps']
        
        self.is_training = True
        
        # Create tensorboard callback
        tensorboard_callback = TensorboardCallback(
            log_dir=self.config['tensorboard_log']
        )
        
        try:
            rospy.loginfo(f"Starting PPO training for {total_timesteps} timesteps")
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=tensorboard_callback,
                tb_log_name="PPO_navigation"
            )
            
            # Save the model
            self.model.save(self.config['model_save_path'])
            rospy.loginfo(f"Model saved to {self.config['model_save_path']}")
            
        except Exception as e:
            rospy.logerr(f"Training failed: {e}")
        finally:
            self.is_training = False
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            self.model = PPO.load(model_path, env=self.env)
            rospy.loginfo(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            return False
    
    def predict(self, observation):
        """Get action prediction from the model"""
        if not self.model:
            rospy.logwarn("Model not loaded, using random action")
            return self.env.action_space.sample()
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def evaluate(self, num_episodes=10):
        """Evaluate the trained model"""
        if not self.model:
            rospy.logerr("No model loaded for evaluation")
            return
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.predict(obs)
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if episode_length >= 1000:  # Max steps
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            rospy.loginfo(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        rospy.loginfo(f"Evaluation completed:")
        rospy.loginfo(f"Average Reward: {avg_reward:.2f}")
        rospy.loginfo(f"Average Episode Length: {avg_length:.2f}")
        
        return avg_reward, avg_length


if __name__ == "__main__":
    import argparse
    import sys
    
    # Initialize ROS node first to get parameters
    rospy.init_node('ppo_agent', anonymous=True)
    
    # Filter out ROS-specific arguments
    filtered_args = []
    for arg in sys.argv[1:]:
        if not arg.startswith('__'):
            filtered_args.append(arg)
    
    # Try to get parameters from ROS parameter server first, then command line
    mode = rospy.get_param('~mode', 'inference')
    model_path = rospy.get_param('~model_path', None)
    config_path = rospy.get_param('~config', None)
    timesteps = rospy.get_param('~timesteps', 100000)
    episodes = rospy.get_param('~episodes', 10)
    tensorboard_dir = rospy.get_param('~tensorboard_dir', None)
    
    # Also support command line arguments
    parser = argparse.ArgumentParser(description='PPO Navigation Agent')
    parser.add_argument('--mode', choices=['train', 'eval', 'inference'], 
                       default=mode, help='Mode to run the agent')
    parser.add_argument('--model_path', type=str, default=model_path, help='Path to load/save model')
    parser.add_argument('--config', type=str, default=config_path, help='Path to configuration file')
    parser.add_argument('--timesteps', type=int, default=timesteps, 
                       help='Number of timesteps for training')
    parser.add_argument('--episodes', type=int, default=episodes, 
                       help='Number of episodes for evaluation')
    parser.add_argument('--tensorboard_dir', type=str, default=tensorboard_dir,
                       help='Directory for TensorBoard logs')
    
    # Parse only the filtered arguments
    args = parser.parse_args(filtered_args)
    
    rospy.loginfo(f"PPO Agent starting in {args.mode} mode")
    
    try:
        # Create agent
        agent = PPONavigationAgent(config_path=args.config)
        
        # Set tensorboard directory if provided
        if args.tensorboard_dir:
            agent.config['tensorboard_log'] = args.tensorboard_dir
        
        if args.mode == 'train':
            if args.model_path:
                agent.config['model_save_path'] = args.model_path
            rospy.loginfo(f"Starting training for {args.timesteps} timesteps")
            agent.train(total_timesteps=args.timesteps)
            
        elif args.mode == 'eval':
            if not args.model_path:
                rospy.logerr("Model path required for evaluation")
                exit(1)
            
            if agent.load_model(args.model_path):
                agent.evaluate(num_episodes=args.episodes)
            
        elif args.mode == 'inference':
            if args.model_path and agent.load_model(args.model_path):
                rospy.loginfo("Model loaded, ready for inference")
            else:
                rospy.loginfo("No model loaded, using random actions")
            
            # Keep the node alive for inference
            rospy.loginfo("PPO agent ready for inference...")
            rospy.spin()
            
    except rospy.ROSInterruptException:
        rospy.loginfo("PPO agent shutting down")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        if 'agent' in locals():
            agent.env.close()