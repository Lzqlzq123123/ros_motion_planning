#!/usr/bin/env python3
"""
Evaluation script for trained PPO navigation controller
"""

import rospy
import argparse
import os
from ppo_agent import PPONavigationAgent

def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO Navigation Controller')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--config', type=str,
                       default='$(find ppo_controller)/scripts/ppo_config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        rospy.logerr(f"Model file not found: {args.model_path}")
        return
    
    # Initialize ROS node
    rospy.init_node('ppo_evaluation', anonymous=True)
    
    try:
        # Create PPO agent
        agent = PPONavigationAgent(config_path=args.config)
        
        # Load trained model
        if agent.load_model(args.model_path):
            rospy.loginfo(f"Evaluating model: {args.model_path}")
            
            # Run evaluation
            avg_reward, avg_length = agent.evaluate(num_episodes=args.episodes)
            
            rospy.loginfo("=== Evaluation Results ===")
            rospy.loginfo(f"Episodes: {args.episodes}")
            rospy.loginfo(f"Average Reward: {avg_reward:.2f}")
            rospy.loginfo(f"Average Episode Length: {avg_length:.2f}")
            
        else:
            rospy.logerr("Failed to load model for evaluation")
            
    except KeyboardInterrupt:
        rospy.loginfo("Evaluation interrupted by user")
    except Exception as e:
        rospy.logerr(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()