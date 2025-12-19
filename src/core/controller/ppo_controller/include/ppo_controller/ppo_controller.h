#ifndef PPO_CONTROLLER_H
#define PPO_CONTROLLER_H

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>
#include <tf2_ros/buffer.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <mutex>

namespace ppo_controller
{
    class PPOController : public nav_core::BaseLocalPlanner
    {
    public:
        PPOController();
        ~PPOController();

        void initialize(std::string name, tf2_ros::Buffer *tf, costmap_2d::Costmap2DROS *costmap_ros);
        bool setPlan(const std::vector<geometry_msgs::PoseStamped> &plan);
        bool computeVelocityCommands(geometry_msgs::Twist &cmd_vel);
        bool isGoalReached();

    private:
        void actionCallback(const geometry_msgs::Twist::ConstPtr& msg);

        bool initialized_;
        tf2_ros::Buffer *tf_;
        costmap_2d::Costmap2DROS *costmap_ros_;
        std::vector<geometry_msgs::PoseStamped> global_plan_;
        
        ros::Subscriber sub_action_;
        geometry_msgs::Twist last_cmd_vel_;
        std::mutex action_mutex_;
    };
};

#endif
