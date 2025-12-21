#ifndef NOMAD_PLANNER_H
#define NOMAD_PLANNER_H

#include <vector>

#include <ros/ros.h>
#include <nav_core/base_global_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <nomad_planner_msgs/MakePlan.h>

namespace nomad_planner
{
class NoMaDPlanner : public nav_core::BaseGlobalPlanner
{
public:
  NoMaDPlanner();
  NoMaDPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros);
  
  void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) override;
  
  bool makePlan(const geometry_msgs::PoseStamped& start,
                const geometry_msgs::PoseStamped& goal,
                std::vector<geometry_msgs::PoseStamped>& plan) override;
  bool makePlan(const geometry_msgs::PoseStamped& start,
                const geometry_msgs::PoseStamped& goal,
                double tolerance,
                std::vector<geometry_msgs::PoseStamped>& plan);

private:
  bool ensureClientConnection();
  bool validatePoseFrames(const geometry_msgs::PoseStamped& start,
                          const geometry_msgs::PoseStamped& goal) const;
  void translatePathToPlan(const nav_msgs::Path& path,
                           std::vector<geometry_msgs::PoseStamped>& plan) const;

  bool initialized_;
  ros::NodeHandle nh_;
  ros::ServiceClient nomad_client_;
  costmap_2d::Costmap2DROS* costmap_ros_;
  std::string global_frame_;
  std::string service_name_;
  std::string goal_image_name_;
  double connect_timeout_sec_;
};
}

#endif
