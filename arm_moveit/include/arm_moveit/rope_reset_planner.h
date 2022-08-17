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
#include <ros/ros.h>

#include <thread>

class RopeResetPlanner {
 public:
  RopeResetPlanner();

  std::pair<ob::PlannerStatus, moveit_msgs::RobotTrajectory> planToReset(geometry_msgs::Pose const& left_reset_pose,
                                           geometry_msgs::Pose const& right_reset_pose, double timeout);

  robot_model_loader::RobotModelLoaderPtr model_loader_;
  moveit::core::RobotModelConstPtr const model_;
  planning_scene_monitor::PlanningSceneMonitorPtr scene_monitor_;

  moveit_visual_tools::MoveItVisualTools visual_tools_;
  ros::NodeHandle nh_;
  trajectory_processing::IterativeParabolicTimeParameterization time_param_;
};
