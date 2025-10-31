#!/usr/bin/env python3
"""
Simple PPO training script for testing
This version bypasses some of the complex launch file interactions
"""

import rospy
import subprocess
import time
import os
import signal
import sys

def run_simple_training():
    """Run a simple PPO training session"""
    
    print("=== Simple PPO Training ===")
    
    # Set ROS workspace
    workspace = "/home/galbot/ros_motion_planning"
    os.chdir(workspace)
    
    # Source the workspace
    env = os.environ.copy()
    env['ROS_PACKAGE_PATH'] = f"{workspace}/src:{env.get('ROS_PACKAGE_PATH', '')}"
    env['CMAKE_PREFIX_PATH'] = f"{workspace}/devel:{env.get('CMAKE_PREFIX_PATH', '')}"
    
    # Start roscore
    print("Starting roscore...")
    roscore_proc = subprocess.Popen(['roscore'], env=env)
    time.sleep(3)
    
    try:
        # Start Gazebo simulation
        print("Starting Gazebo simulation...")
        gazebo_proc = subprocess.Popen([
            'roslaunch', 'sim_env', 'config.launch',
            'world:=warehouse',
            'map:=warehouse',
            'robot_number:=1'
        ], env=env)
        time.sleep(10)
        
        # Start move_base with PPO
        print("Starting move_base with PPO controller...")
        move_base_proc = subprocess.Popen([
            'roslaunch', 'ppo_controller', 'test_ppo_movebase.launch'
        ], env=env)
        time.sleep(5)
        
        # Start PPO training
        print("Starting PPO training...")
        ppo_proc = subprocess.Popen([
            'python3', 
            f'{workspace}/src/core/controller/ppo_controller/scripts/ppo_agent.py',
            '--mode', 'train',
            '--timesteps', '10000',
            '--model_path', f'{workspace}/src/core/controller/ppo_controller/models/test_model.zip'
        ], env=env)
        
        print("Training started! Press Ctrl+C to stop...")
        
        # Wait for training or user interrupt
        try:
            ppo_proc.wait()
        except KeyboardInterrupt:
            print("\nStopping training...")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup processes
        print("Cleaning up...")
        for proc in [ppo_proc, move_base_proc, gazebo_proc, roscore_proc]:
            try:
                if 'proc' in locals() and proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
            except:
                pass
        
        # Kill any remaining processes
        subprocess.run(['pkill', '-f', 'gazebo'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-f', 'move_base'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-f', 'roscore'], stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    run_simple_training()