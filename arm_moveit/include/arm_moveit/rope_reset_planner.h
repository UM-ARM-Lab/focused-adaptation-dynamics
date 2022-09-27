#pragma once
#include <geometry_msgs/Pose.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/RobotTrajectory.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <ompl/base/Goal.h>
#include <ompl/base/PlannerStatus.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/StateSampler.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ros/ros.h>

#include <string>
#include <thread>
#include <vector>

namespace ob = ompl::base;
namespace og = ompl::geometric;

void addLinkPadding(planning_scene::PlanningScenePtr const& planning_scene);

class ArmStateSpace : public ob::RealVectorStateSpace {
 public:
  std::vector<std::string> joint_names_;
  explicit ArmStateSpace(std::vector<std::string> const& joint_names)
      : ob::RealVectorStateSpace(static_cast<unsigned int>(joint_names.size())), joint_names_(joint_names) {
    auto const dim = joint_names.size();
    for (auto i{0u}; i < dim; ++i) {
      setDimensionName(i, joint_names[i]);
    }
  }

  int getJointIndex(std::string const& joint_name) { return getDimensionIndex(joint_name); }
};

struct PlanningResult {
  std::string status;
  moveit_msgs::RobotTrajectory traj;
};

class RopeResetPlanner {
 public:
  RopeResetPlanner(std::string const& group_name = "both_arms");

  PlanningResult planWithConstraints(planning_scene::PlanningScenePtr const& planning_scene, ob::GoalPtr const& goal,
                                     ob::StateValidityCheckerFn const& state_validity_fn, double timeout);

  PlanningResult planToReset(geometry_msgs::Pose const& left_pose, geometry_msgs::Pose const& right_pose,
                             double orientation_path_tolerance, double orientation_goal_tolerance, double timeout,
                             bool debug_collisions);

  PlanningResult planToStart(geometry_msgs::Pose const& left_pose, geometry_msgs::Pose const& right_pose,
                             double max_gripper_dist, double orientation_path_tolerance,
                             double orientation_goal_tolerance, double timeout, bool debug_collisions);

  og::PathGeometric simplify(og::PathGeometric original_path, ob::GoalPtr goal);

  robot_model_loader::RobotModelLoaderPtr model_loader_;
  moveit::core::RobotModelConstPtr const model_;
  planning_scene_monitor::PlanningSceneMonitorPtr scene_monitor_;

  moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
  ros::NodeHandle nh_;
  trajectory_processing::IterativeParabolicTimeParameterization time_param_;
  std::string group_name_;
  moveit::core::JointModelGroup const* group_;
  unsigned int n_joints_;
  std::vector<std::string> joint_names_;
  std::shared_ptr<ArmStateSpace> space_;
  og::SimpleSetup ss_;
  ob::SpaceInformationPtr si_;
};
