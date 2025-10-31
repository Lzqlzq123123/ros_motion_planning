/**
 * *********************************************************
 *
 * @file: ppo_controller.cpp
 * @brief: Contains the implementation of PPO Controller class
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

#include "controller/ppo_controller.h"
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Path.h>
#include <pluginlib/class_list_macros.h>
#include <algorithm>
#include <numeric>
#include <limits>

namespace rmp
{
namespace controller
{

PPOController::PPOController() : 
    initialized_(false),
    training_mode_(false),
    current_episode_(0),
    max_episodes_(10000),
    episode_reward_(0.0),
    step_count_(0),
    prev_distance_to_goal_(std::numeric_limits<double>::max())
{
}

PPOController::~PPOController()
{
    if (initialized_)
    {
        // Send zero velocity command
        geometry_msgs::Twist zero_cmd;
        ros::Publisher cmd_pub = private_nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        cmd_pub.publish(zero_cmd);
    }
}

void PPOController::initialize(std::string name, tf2_ros::Buffer* tf, costmap_2d::Costmap2DROS* costmap_ros)
{
    if (initialized_)
    {
        ROS_WARN("PPOController has already been initialized... doing nothing");
        return;
    }

    tf_ = tf;
    costmap_ros_ = costmap_ros;
    private_nh_ = ros::NodeHandle("~/" + name);

    // Load parameters
    private_nh_.param("max_linear_vel", max_linear_vel_, 0.8);
    private_nh_.param("max_angular_vel", max_angular_vel_, 1.0);
    private_nh_.param("min_linear_vel", min_linear_vel_, -0.5);
    private_nh_.param("min_angular_vel", min_angular_vel_, -1.0);
    private_nh_.param("goal_tolerance", goal_tolerance_, 0.2);
    private_nh_.param("obstacle_threshold", obstacle_threshold_, 0.5);
    private_nh_.param("laser_scan_size", laser_scan_size_, 360);
    
    // PPO specific parameters
    private_nh_.param("ppo_script_path", ppo_script_path_, 
                     std::string("$(find ppo_controller)/scripts/ppo_agent.py"));
    private_nh_.param("model_save_path", model_save_path_, 
                     std::string("ppo_navigation_model.zip"));
    private_nh_.param("tensorboard_log_dir", tensorboard_log_dir_, 
                     std::string("./ppo_navigation_tensorboard/"));
    private_nh_.param("training_mode", training_mode_, false);
    
    // Reward function weights
    private_nh_.param("reward_goal_weight", reward_goal_weight_, 10.0);
    private_nh_.param("reward_collision_weight", reward_collision_weight_, -10.0);
    private_nh_.param("reward_smooth_weight", reward_smooth_weight_, -0.1);
    private_nh_.param("reward_progress_weight", reward_progress_weight_, 1.0);

    // Initialize publishers
    g_plan_pub_ = private_nh_.advertise<nav_msgs::Path>("global_plan", 1);
    l_plan_pub_ = private_nh_.advertise<nav_msgs::Path>("local_plan", 1);
    
    // Publishers for PPO communication
    ros::NodeHandle nh;
    ros::Publisher state_pub = nh.advertise<std_msgs::Float32MultiArray>("/ppo/state", 1);
    ros::Publisher reward_pub = nh.advertise<std_msgs::Float32>("/ppo/reward", 1);
    ros::Publisher done_pub = nh.advertise<std_msgs::Bool>("/ppo/done", 1);
    
    // Subscribers for PPO communication  
    ros::Subscriber reset_sub = nh.subscribe("/ppo/reset", 1, &PPOController::resetCallback, this);

    // Subscribe to laser scan
    laser_sub_ = nh.subscribe("/scan", 1, &PPOController::laserScanCallback, this);

    // Initialize odometry helper
    odom_helper_.setOdomTopic(odom_topic_);

    // Initialize Controller base class
    setBaseFrame(costmap_ros_->getBaseFrameID());
    setMapFrame(costmap_ros_->getGlobalFrameID());

    // Initialize PPO agent
    initializePPOAgent();

    initialized_ = true;
    ROS_INFO("PPOController initialized successfully");
}

void PPOController::initializePPOAgent()
{
    // In training mode, the PPO agent is launched separately by the launch file
    // In inference mode, we expect the agent to be already running
    if (training_mode_) {
        ROS_INFO("PPO Controller in training mode - expecting external PPO agent");
    } else {
        ROS_INFO("PPO Controller in inference mode - expecting external PPO agent");
    }
    
    // Wait a moment for connection setup
    ros::Duration(1.0).sleep();
}

bool PPOController::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan)
{
    if (!initialized_)
    {
        ROS_ERROR("PPOController not initialized");
        return false;
    }

    global_plan_ = orig_global_plan;
    
    if (!global_plan_.empty())
    {
        goal_pose_ = global_plan_.back();
        prev_distance_to_goal_ = std::numeric_limits<double>::max();
    }

    // Publish global plan for visualization
    publishGlobalPlan(global_plan_);

    return true;
}

bool PPOController::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
{
    if (!initialized_)
    {
        ROS_ERROR("PPOController not initialized");
        return false;
    }

    // Get current robot pose
    geometry_msgs::PoseStamped robot_pose;
    if (!costmap_ros_->getRobotPose(robot_pose))
    {
        ROS_WARN("Could not get robot pose");
        return false;
    }
    current_pose_ = robot_pose;

    // Check if goal is reached
    if (isGoalReached())
    {
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
        return true;
    }

    // Get current state for PPO
    std::vector<float> state = getCurrentState();
    
    // Publish state to PPO agent
    std_msgs::Float32MultiArray state_msg;
    state_msg.data = state;
    ros::NodeHandle nh;
    ros::Publisher state_pub = nh.advertise<std_msgs::Float32MultiArray>("/ppo/state", 1);
    state_pub.publish(state_msg);

    // For now, use a simple controller while PPO agent is learning
    // In production, this would get action from PPO agent via ROS topic
    std::vector<float> action = {0.3, 0.0}; // Default forward motion
    
    // Execute action
    executeAction(action, cmd_vel);

    // Calculate and publish reward for training
    if (training_mode_)
    {
        float reward = calculateReward();
        episode_reward_ += reward;
        step_count_++;
        
        std_msgs::Float32 reward_msg;
        reward_msg.data = reward;
        ros::Publisher reward_pub = nh.advertise<std_msgs::Float32>("/ppo/reward", 1);
        reward_pub.publish(reward_msg);
        
        // Check if episode is done
        bool done = isEpisodeDone();
        std_msgs::Bool done_msg;
        done_msg.data = done;
        ros::Publisher done_pub = nh.advertise<std_msgs::Bool>("/ppo/done", 1);
        done_pub.publish(done_msg);
        
        if (done)
        {
            ROS_INFO("Episode %d finished with reward: %.2f", current_episode_, episode_reward_);
            current_episode_++;
            episode_reward_ = 0.0;
            step_count_ = 0;
        }
    }

    // Update previous state
    prev_pose_ = current_pose_;
    prev_cmd_ = cmd_vel;

    return true;
}

bool PPOController::isGoalReached()
{
    if (global_plan_.empty())
        return false;

    double dx = current_pose_.pose.position.x - goal_pose_.pose.position.x;
    double dy = current_pose_.pose.position.y - goal_pose_.pose.position.y;
    double distance = sqrt(dx * dx + dy * dy);

    return distance < goal_tolerance_;
}

std::vector<float> PPOController::getCurrentState()
{
    std::vector<float> state;
    state.reserve(STATE_DIM);

    // 1. Normalized laser scan data
    std::vector<float> laser_data = normalizeLaserScan(current_scan_);
    state.insert(state.end(), laser_data.begin(), laser_data.end());

    // 2. Current pose (normalized)
    state.push_back(current_pose_.pose.position.x / 10.0);  // Normalize by typical map size
    state.push_back(current_pose_.pose.position.y / 10.0);
    state.push_back(tf2::getYaw(current_pose_.pose.orientation) / M_PI);

    // 3. Relative goal position
    std::vector<float> goal_data = getRelativeGoal();
    state.insert(state.end(), goal_data.begin(), goal_data.end());

    // 4. Current velocity
    std::vector<float> vel_data = getCurrentVelocity();
    state.insert(state.end(), vel_data.begin(), vel_data.end());

    // Ensure state vector has correct size
    if (state.size() != STATE_DIM)
    {
        ROS_WARN("State dimension mismatch: expected %d, got %zu", STATE_DIM, state.size());
        state.resize(STATE_DIM, 0.0);
    }

    return state;
}

std::vector<float> PPOController::normalizeLaserScan(const sensor_msgs::LaserScan& scan)
{
    std::vector<float> normalized_scan;
    normalized_scan.reserve(LASER_SCAN_DIM);

    if (scan.ranges.empty())
    {
        // Return zeros if no scan data
        normalized_scan.assign(LASER_SCAN_DIM, 0.0);
        return normalized_scan;
    }

    // Downsample or interpolate to fixed size
    size_t scan_size = scan.ranges.size();
    for (int i = 0; i < LASER_SCAN_DIM; ++i)
    {
        size_t idx = (i * scan_size) / LASER_SCAN_DIM;
        float range = scan.ranges[idx];
        
        // Handle invalid readings
        if (std::isnan(range) || std::isinf(range) || range < scan.range_min || range > scan.range_max)
        {
            range = scan.range_max;
        }
        
        // Normalize to [0, 1]
        normalized_scan.push_back(std::min(range / scan.range_max, 1.0f));
    }

    return normalized_scan;
}

std::vector<float> PPOController::getRelativeGoal()
{
    std::vector<float> relative_goal(3, 0.0);

    if (global_plan_.empty())
        return relative_goal;

    // Calculate relative position to goal
    double dx = goal_pose_.pose.position.x - current_pose_.pose.position.x;
    double dy = goal_pose_.pose.position.y - current_pose_.pose.position.y;
    
    // Transform to robot frame
    double robot_yaw = tf2::getYaw(current_pose_.pose.orientation);
    double cos_yaw = cos(robot_yaw);
    double sin_yaw = sin(robot_yaw);
    
    relative_goal[0] = dx * cos_yaw + dy * sin_yaw;  // forward distance
    relative_goal[1] = -dx * sin_yaw + dy * cos_yaw; // lateral distance
    
    // Relative angle to goal
    double goal_yaw = tf2::getYaw(goal_pose_.pose.orientation);
    relative_goal[2] = regularizeAngle(goal_yaw - robot_yaw) / M_PI;

    // Normalize distances
    relative_goal[0] /= 10.0;  // Normalize by typical map size
    relative_goal[1] /= 10.0;

    return relative_goal;
}

std::vector<float> PPOController::getCurrentVelocity()
{
    std::vector<float> velocity(2, 0.0);

    nav_msgs::Odometry odom;
    odom_helper_.getOdom(odom);
    
    velocity[0] = odom.twist.twist.linear.x / max_linear_vel_;   // Normalized linear velocity
    velocity[1] = odom.twist.twist.angular.z / max_angular_vel_; // Normalized angular velocity

    return velocity;
}

void PPOController::executeAction(const std::vector<float>& action, geometry_msgs::Twist& cmd_vel)
{
    if (action.size() < ACTION_DIM)
    {
        ROS_WARN("Invalid action dimension");
        cmd_vel.linear.x = 0.0;
        cmd_vel.angular.z = 0.0;
        return;
    }

    // Clip actions to valid range
    cmd_vel.linear.x = std::max(min_linear_vel_, std::min(max_linear_vel_, (double)action[0]));
    cmd_vel.angular.z = std::max(min_angular_vel_, std::min(max_angular_vel_, (double)action[1]));

    // Additional safety check for obstacles
    if (!current_scan_.ranges.empty())
    {
        // Check front area for obstacles
        size_t front_start = current_scan_.ranges.size() * 0.4;  // 40% to 60% of scan
        size_t front_end = current_scan_.ranges.size() * 0.6;
        
        bool obstacle_detected = false;
        for (size_t i = front_start; i < front_end; ++i)
        {
            if (current_scan_.ranges[i] < obstacle_threshold_)
            {
                obstacle_detected = true;
                break;
            }
        }
        
        if (obstacle_detected && cmd_vel.linear.x > 0)
        {
            cmd_vel.linear.x *= 0.5;  // Reduce speed near obstacles
        }
    }
}

float PPOController::calculateReward()
{
    float reward = 0.0;

    if (global_plan_.empty())
        return reward;

    // 1. Goal reaching reward
    double current_distance = sqrt(
        pow(current_pose_.pose.position.x - goal_pose_.pose.position.x, 2) +
        pow(current_pose_.pose.position.y - goal_pose_.pose.position.y, 2)
    );

    if (isGoalReached())
    {
        reward += reward_goal_weight_;
    }
    else
    {
        // Progress reward
        if (prev_distance_to_goal_ < std::numeric_limits<double>::max())
        {
            double progress = prev_distance_to_goal_ - current_distance;
            reward += reward_progress_weight_ * progress;
        }
    }
    prev_distance_to_goal_ = current_distance;

    // 2. Collision penalty
    if (!current_scan_.ranges.empty())
    {
        float min_range = *std::min_element(current_scan_.ranges.begin(), current_scan_.ranges.end());
        if (min_range < obstacle_threshold_)
        {
            reward += reward_collision_weight_ * (obstacle_threshold_ - min_range) / obstacle_threshold_;
        }
    }

    // 3. Smoothness penalty
    if (step_count_ > 0)
    {
        double linear_change = abs(prev_cmd_.linear.x - 0.0);  // Assuming current command not available yet
        double angular_change = abs(prev_cmd_.angular.z - 0.0);
        reward += reward_smooth_weight_ * (linear_change + angular_change);
    }

    return reward;
}

bool PPOController::isEpisodeDone()
{
    // Episode is done if:
    // 1. Goal is reached
    if (isGoalReached())
        return true;

    // 2. Collision detected
    if (!current_scan_.ranges.empty())
    {
        float min_range = *std::min_element(current_scan_.ranges.begin(), current_scan_.ranges.end());
        if (min_range < 0.2)  // Very close collision threshold
            return true;
    }

    // 3. Max steps reached
    if (step_count_ >= 1000)
        return true;

    return false;
}

void PPOController::laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    current_scan_ = *msg;
}

void PPOController::resetCallback(const std_msgs::Bool::ConstPtr& msg)
{
    if (msg->data)
    {
        // Reset episode state
        step_count_ = 0;
        episode_reward_ = 0.0;
        prev_distance_to_goal_ = std::numeric_limits<double>::max();
        
        ROS_INFO("PPO episode reset");
    }
}

void PPOController::publishGlobalPlan(const std::vector<geometry_msgs::PoseStamped>& path)
{
    nav_msgs::Path gui_path;
    gui_path.poses.resize(path.size());
    gui_path.header.frame_id = path.empty() ? map_frame_ : path[0].header.frame_id;
    gui_path.header.stamp = ros::Time::now();
    
    for (unsigned int i = 0; i < path.size(); i++)
    {
        gui_path.poses[i] = path[i];
    }
    
    g_plan_pub_.publish(gui_path);
}

void PPOController::publishLocalPlan(const std::vector<geometry_msgs::PoseStamped>& path)
{
    nav_msgs::Path gui_path;
    gui_path.poses.resize(path.size());
    gui_path.header.frame_id = path.empty() ? base_frame_ : path[0].header.frame_id;
    gui_path.header.stamp = ros::Time::now();
    
    for (unsigned int i = 0; i < path.size(); i++)
    {
        gui_path.poses[i] = path[i];
    }
    
    l_plan_pub_.publish(gui_path);
}

void PPOController::startTraining(int training_episodes, const std::string& model_save_path)
{
    training_mode_ = true;
    max_episodes_ = training_episodes;
    current_episode_ = 0;
    
    if (!model_save_path.empty())
    {
        model_save_path_ = model_save_path;
    }
    
    ROS_INFO("Started PPO training mode for %d episodes", max_episodes_);
}

bool PPOController::loadModel(const std::string& model_path)
{
    model_save_path_ = model_path;
    training_mode_ = false;
    
    // Restart PPO agent with the new model
    initializePPOAgent();
    
    ROS_INFO("PPO model loaded from: %s", model_path.c_str());
    return true;
}

void PPOController::setTrainingMode(bool training)
{
    training_mode_ = training;
    ROS_INFO("PPO training mode: %s", training ? "ON" : "OFF");
}

}  // namespace controller
}  // namespace rmp

// Register this plugin with pluginlib
PLUGINLIB_EXPORT_CLASS(rmp::controller::PPOController, nav_core::BaseLocalPlanner)