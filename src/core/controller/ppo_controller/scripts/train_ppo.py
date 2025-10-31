#!/usr/bin/env python3
"""
Training script for PPO navigation controller
This script sets up the training environment and starts PPO training
"""

import rospy
import os
import sys
import argparse
from ppo_agent import PPONavigationAgent

def main():
    parser = argparse.ArgumentParser(description='Train PPO Navigation Controller')
    parser.add_argument('--config', type=str, 
                       default='$(find ppo_controller)/scripts/ppo_config.json',
                       help='Path to configuration file')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps')
    parser.add_argument('--model_path', type=str, 
                       default='./ppo_navigation_model.zip',
                       help='Path to save trained model')
    parser.add_argument('--tensorboard_dir', type=str,
                       default='./ppo_navigation_tensorboard/',
                       help='TensorBoard log directory')
    parser.add_argument('--continue_training', action='store_true',
                       help='Continue training from existing model')
    
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('ppo_training', anonymous=True)
    
    try:
        # Create PPO agent
        agent = PPONavigationAgent(config_path=args.config)
        
        # Update config with command line args
        agent.config['total_timesteps'] = args.timesteps
        agent.config['model_save_path'] = args.model_path
        agent.config['tensorboard_log'] = args.tensorboard_dir
        
        if args.continue_training and os.path.exists(args.model_path):
            rospy.loginfo(f"Loading existing model from {args.model_path}")
            agent.load_model(args.model_path)
        
        # Start training
        rospy.loginfo("Starting PPO training...")
        rospy.loginfo(f"Training for {args.timesteps} timesteps")
        rospy.loginfo(f"Model will be saved to: {args.model_path}")
        rospy.loginfo(f"TensorBoard logs: {args.tensorboard_dir}")
        
        agent.train(total_timesteps=args.timesteps)
        
        rospy.loginfo("Training completed successfully!")
        
    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted by user")
    except Exception as e:
        rospy.logerr(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()