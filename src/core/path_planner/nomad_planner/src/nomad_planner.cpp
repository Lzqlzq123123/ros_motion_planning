#include "nomad_planner/nomad_planner.h"

#include <pluginlib/class_list_macros.h>
#include <ros/duration.h>
#include <stdexcept>
#include <utility>

namespace nomad_planner
{
NoMaDPlanner::NoMaDPlanner() : initialized_(false), costmap_ros_(nullptr), connect_timeout_sec_(5.0)
{
}

NoMaDPlanner::NoMaDPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros) : NoMaDPlanner()
{
  initialize(std::move(name), costmap_ros);
}

void NoMaDPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
{
  if (initialized_)
  {
    ROS_WARN_NAMED("nomad_planner", "NoMaDPlanner has already been initialized, skipping reinitialization.");
    return;
  }

  if (!costmap_ros)
  {
    ROS_ERROR_NAMED("nomad_planner", "Costmap pointer is null during initialization.");
    throw std::runtime_error("NoMaDPlanner requires a valid costmap");
  }

  costmap_ros_ = costmap_ros;
  global_frame_ = costmap_ros_->getGlobalFrameID();

  nh_ = ros::NodeHandle("~" + (name.empty() ? std::string("/nomad_planner") : "/" + name));

  nh_.param<std::string>("service_name", service_name_, std::string("/nomad/make_plan"));
  nh_.param<std::string>("goal_image_name", goal_image_name_, std::string(""));
  nh_.param("connect_timeout", connect_timeout_sec_, 5.0);

  if (!ensureClientConnection())
  {
    ROS_WARN_NAMED("nomad_planner", "NoMaD service %s not available yet.", service_name_.c_str());
  }

  initialized_ = true;
  ROS_INFO_NAMED("nomad_planner", "NoMaDPlanner initialized for frame %s", global_frame_.c_str());
}

bool NoMaDPlanner::makePlan(const geometry_msgs::PoseStamped& start,
                            const geometry_msgs::PoseStamped& goal,
                            std::vector<geometry_msgs::PoseStamped>& plan)
{
  return makePlan(start, goal, 0.0, plan);
}

bool NoMaDPlanner::makePlan(const geometry_msgs::PoseStamped& start,
                            const geometry_msgs::PoseStamped& goal,
                            double /*tolerance*/,
                            std::vector<geometry_msgs::PoseStamped>& plan)
{
  plan.clear();

  if (!initialized_)
  {
    ROS_ERROR_NAMED("nomad_planner", "Planner has not been initialized.");
    return false;
  }

  if (!validatePoseFrames(start, goal))
  {
    return false;
  }

  if (!ensureClientConnection())
  {
    ROS_ERROR_NAMED("nomad_planner", "Unable to connect to NoMaD planning service: %s", service_name_.c_str());
    return false;
  }

  nomad_planner_msgs::MakePlan srv;
  srv.request.start = start;
  srv.request.goal = goal;
  srv.request.goal_image_name = goal_image_name_;

  if (!nomad_client_.call(srv))
  {
    ROS_ERROR_NAMED("nomad_planner", "Failed to call NoMaD make_plan service: %s", service_name_.c_str());
    return false;
  }

  if (!srv.response.success)
  {
    ROS_WARN_NAMED("nomad_planner", "NoMaD planning failed: %s", srv.response.message.c_str());
    return false;
  }

  translatePathToPlan(srv.response.plan, plan);

  if (plan.empty())
  {
    ROS_WARN_NAMED("nomad_planner", "NoMaD returned an empty path.");
    return false;
  }

  return true;
}

bool NoMaDPlanner::ensureClientConnection()
{
  if (!nomad_client_.isValid())
  {
    nomad_client_ = nh_.serviceClient<nomad_planner_msgs::MakePlan>(service_name_, /* persistent */ true);
  }

  if (!nomad_client_.waitForExistence(ros::Duration(connect_timeout_sec_)))
  {
    ROS_WARN_NAMED("nomad_planner",
                   "Timed out waiting for NoMaD service %s (%.2fs).",
                   service_name_.c_str(),
                   connect_timeout_sec_);
    return false;
  }

  return true;
}

bool NoMaDPlanner::validatePoseFrames(const geometry_msgs::PoseStamped& start,
                                      const geometry_msgs::PoseStamped& goal) const
{
  if (start.header.frame_id != global_frame_)
  {
    ROS_ERROR_NAMED("nomad_planner",
                    "Start pose frame (%s) does not match global frame (%s).",
                    start.header.frame_id.c_str(),
                    global_frame_.c_str());
    return false;
  }

  if (goal.header.frame_id != global_frame_)
  {
    ROS_ERROR_NAMED("nomad_planner",
                    "Goal pose frame (%s) does not match global frame (%s).",
                    goal.header.frame_id.c_str(),
                    global_frame_.c_str());
    return false;
  }

  return true;
}

void NoMaDPlanner::translatePathToPlan(const nav_msgs::Path& path,
                                       std::vector<geometry_msgs::PoseStamped>& plan) const
{
  const std::string path_frame = path.header.frame_id.empty() ? global_frame_ : path.header.frame_id;
  if (path_frame != global_frame_)
  {
    ROS_WARN_NAMED("nomad_planner",
                   "NoMaD path published in frame %s differs from global frame %s.",
                   path_frame.c_str(),
                   global_frame_.c_str());
  }

  plan.reserve(path.poses.size());

  const ros::Time stamp = ros::Time::now();
  for (const auto& pose : path.poses)
  {
    geometry_msgs::PoseStamped stamped_pose = pose;
    if (stamped_pose.header.frame_id.empty())
    {
      stamped_pose.header.frame_id = path_frame;
    }
    stamped_pose.header.frame_id = global_frame_;
    stamped_pose.header.stamp = stamp;
    plan.push_back(stamped_pose);
  }
}
}  // namespace nomad_planner

PLUGINLIB_EXPORT_CLASS(nomad_planner::NoMaDPlanner, nav_core::BaseGlobalPlanner)
