#include <ppo_controller/ppo_controller.h>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(ppo_controller::PPOController, nav_core::BaseLocalPlanner)

namespace ppo_controller
{
    PPOController::PPOController() : initialized_(false), tf_(NULL), costmap_ros_(NULL) {}

    PPOController::~PPOController() {}

    void PPOController::initialize(std::string name, tf2_ros::Buffer *tf, costmap_2d::Costmap2DROS *costmap_ros)
    {
        if (!initialized_)
        {
            ros::NodeHandle private_nh("~/" + name);
            tf_ = tf;
            costmap_ros_ = costmap_ros;
            
            // Initialize publishers and subscribers
            // For training mode, we might listen to an external command topic
            // For inference mode, we would load the model here
            
            // Example: Subscribe to action topic from RL agent
            ros::NodeHandle nh;
            std::string action_topic = "ppo_cmd_vel"; 
            sub_action_ = private_nh.subscribe(action_topic, 1, &PPOController::actionCallback, this);

            initialized_ = true;
            ROS_INFO("PPOController initialized");
        }
    }

    bool PPOController::setPlan(const std::vector<geometry_msgs::PoseStamped> &plan)
    {
        if (!initialized_)
        {
            ROS_ERROR("PPOController has not been initialized");
            return false;
        }
        global_plan_ = plan;
        return true;
    }

    bool PPOController::computeVelocityCommands(geometry_msgs::Twist &cmd_vel)
    {
        if (!initialized_)
        {
            ROS_ERROR("PPOController has not been initialized");
            return false;
        }

        // In training mode, we just pass through the command received from the RL agent
        // In inference mode, we would compute the command using the loaded model
        
        // For now, let's assume we are receiving commands via callback
        // We need to ensure safety checks here (collision avoidance etc if needed)
        
        std::lock_guard<std::mutex> lock(action_mutex_);
        cmd_vel = last_cmd_vel_;
        
        return true;
    }

    bool PPOController::isGoalReached()
    {
        if (!initialized_)
        {
            ROS_ERROR("PPOController has not been initialized");
            return false;
        }
        
        // Implement goal check logic here using costmap_ros_->getRobotPose() and global_plan_.back()
        // For simplicity, returning false to keep control loop running
        return false;
    }

    void PPOController::actionCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(action_mutex_);
        last_cmd_vel_ = *msg;
    }
};
