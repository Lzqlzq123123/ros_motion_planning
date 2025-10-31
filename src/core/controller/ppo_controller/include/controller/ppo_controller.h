/**
 * *********************************************************
 *
 * @file: ppo_controller.h
 * @brief: Contains the PPO Controller class using stable-baselines3
 * @author: Assistant
 * @date: 2024-10-27
 * @version: 1.0
 *
 * Copyright (c) 2024, Assistant.
 * All rights reserved.
 *
 * --------------------------------------------------------
 *
 * ********************************************************
 */
#ifndef RMP_CONTROLLER_PPO_CONTROLLER_H_
#define RMP_CONTROLLER_PPO_CONTROLLER_H_

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>

#include <tf2_ros/buffer.h>
#include <dynamic_reconfigure/server.h>
#include <angles/angles.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>

#include <costmap_2d/costmap_2d_ros.h>
#include <nav_core/base_local_planner.h>
#include <base_local_planner/latched_stop_rotate_controller.h>
#include <base_local_planner/odometry_helper_ros.h>

#include "controller/controller.h"

namespace rmp
{
namespace controller
{
/**
 * @class PPOController
 * @brief ROS Wrapper for the PPO Controller that adheres to the
 * BaseLocalPlanner interface and can be used as a plugin for move_base.
 * Uses stable-baselines3 PPO algorithm for local navigation.
 */
class PPOController : public nav_core::BaseLocalPlanner, Controller
{
public:
  /**
   * @brief Constructor for PPOController wrapper
   */
  PPOController();

  /**
   * @brief Destructor for the wrapper
   */
  ~PPOController();

  /**
   * @brief Constructs the ros wrapper
   * @param name The name to give this instance of the trajectory planner
   * @param tf A pointer to a transform listener
   * @param costmap The cost map to use for assigning costs to trajectories
   */
  void initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros);

  /**
   * @brief Given the current position, orientation, and velocity of the robot,
   * compute velocity commands to send to the base using PPO algorithm
   * @param cmd_vel Will be filled with the velocity command to be passed to the robot base
   * @return True if a valid trajectory was found, false otherwise
   */
  bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);

  /**
   * @brief Set the plan that the controller is following
   * @param orig_global_plan The plan to pass to the controller
   * @return True if the plan was updated successfully, false otherwise
   */
  bool setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan);

  /**
   * @brief Check if the goal pose has been achieved
   * @return True if achieved, false otherwise
   */
  bool isGoalReached();

  /**
   * @brief Check if the controller is initialized
   * @return True if initialized, false otherwise
   */
  bool isInitialized() { return initialized_; }

  /**
   * @brief Start training mode for the PPO agent
   * @param training_episodes Number of episodes to train
   * @param model_save_path Path to save the trained model
   */
  void startTraining(int training_episodes = 10000, const std::string& model_save_path = "");

  /**
   * @brief Load a pre-trained PPO model
   * @param model_path Path to the trained model file
   * @return True if model loaded successfully, false otherwise
   */
  bool loadModel(const std::string& model_path);

  /**
   * @brief Set training mode on/off
   * @param training True for training mode, false for inference mode
   */
  void setTrainingMode(bool training);

private:
  /**
   * @brief Initialize the PPO agent and environment
   */
  void initializePPOAgent();

  /**
   * @brief Get current state observation for PPO agent
   * @return State vector containing laser scan, pose, goal, velocity info
   */
  std::vector<float> getCurrentState();

  /**
   * @brief Execute action from PPO agent
   * @param action Action vector [linear_vel, angular_vel]
   * @param cmd_vel Output velocity command
   */
  void executeAction(const std::vector<float>& action, geometry_msgs::Twist& cmd_vel);

  /**
   * @brief Calculate reward for current state and action
   * @return Reward value
   */
  float calculateReward();

  /**
   * @brief Check if episode is done (goal reached or collision)
   * @return True if episode should terminate
   */
  bool isEpisodeDone();

  /**
   * @brief Laser scan callback
   * @param msg Laser scan message
   */
  void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);

  /**
   * @brief Reset callback for PPO episode reset
   * @param msg Bool message to trigger reset
   */
  void resetCallback(const std_msgs::Bool::ConstPtr& msg);

  /**
   * @brief Normalize laser scan data
   * @param scan Raw laser scan data
   * @return Normalized laser scan data
   */
  std::vector<float> normalizeLaserScan(const sensor_msgs::LaserScan& scan);

  /**
   * @brief Get relative goal position in robot frame
   * @return [dx, dy, dtheta] relative to robot
   */
  std::vector<float> getRelativeGoal();

  /**
   * @brief Get current robot velocity
   * @return [linear_x, angular_z] velocity
   */
  std::vector<float> getCurrentVelocity();

  /**
   * @brief Publish local plan for visualization
   */
  void publishLocalPlan(const std::vector<geometry_msgs::PoseStamped>& path);

  /**
   * @brief Publish global plan for visualization
   */
  void publishGlobalPlan(const std::vector<geometry_msgs::PoseStamped>& path);

private:
  // ROS members
  tf2_ros::Buffer* tf_;
  costmap_2d::Costmap2DROS* costmap_ros_;
  ros::NodeHandle private_nh_;
  
  // Publishers and subscribers
  ros::Publisher g_plan_pub_, l_plan_pub_;
  ros::Subscriber laser_sub_;
  
  // Current state
  geometry_msgs::PoseStamped current_pose_;
  geometry_msgs::PoseStamped goal_pose_;
  sensor_msgs::LaserScan current_scan_;
  nav_msgs::Odometry current_odom_;
  std::vector<geometry_msgs::PoseStamped> global_plan_;
  
  // PPO agent related
  bool initialized_;
  bool training_mode_;
  std::string ppo_script_path_;
  std::string model_save_path_;
  std::string tensorboard_log_dir_;
  
  // Parameters
  double max_linear_vel_;
  double max_angular_vel_;
  double min_linear_vel_;
  double min_angular_vel_;
  double goal_tolerance_;
  double obstacle_threshold_;
  int laser_scan_size_;
  
  // State dimensions
  static const int LASER_SCAN_DIM = 360;  // Number of laser scan points to use
  static const int POSE_DIM = 3;          // x, y, theta
  static const int GOAL_DIM = 3;          // relative dx, dy, dtheta
  static const int VEL_DIM = 2;           // linear, angular velocity
  static const int STATE_DIM = LASER_SCAN_DIM + POSE_DIM + GOAL_DIM + VEL_DIM;
  
  // Action dimensions
  static const int ACTION_DIM = 2;        // linear_vel, angular_vel
  
  // Stop controller
  base_local_planner::LatchedStopRotateController latchedStopRotateController_;
  
  // Odometry helper
  base_local_planner::OdometryHelperRos odom_helper_;
  std::string odom_topic_;
  
  // Training episode tracking
  int current_episode_;
  int max_episodes_;
  float episode_reward_;
  int step_count_;
  
  // Reward function weights
  double reward_goal_weight_;
  double reward_collision_weight_;
  double reward_smooth_weight_;
  double reward_progress_weight_;
  
  // Previous state for reward calculation
  geometry_msgs::PoseStamped prev_pose_;
  double prev_distance_to_goal_;
  geometry_msgs::Twist prev_cmd_;
};

}  // namespace controller
}  // namespace rmp

#endif  // RMP_CONTROLLER_PPO_CONTROLLER_H_