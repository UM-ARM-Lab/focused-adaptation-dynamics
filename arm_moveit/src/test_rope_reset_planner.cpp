#include <arm_moveit/rope_reset_planner.h>
#include <geometry_msgs/Pose.h>
#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "rope_reset_planner");
  geometry_msgs::Pose right_tool_grasp_pose;
  right_tool_grasp_pose.position.x = 0.7;
  right_tool_grasp_pose.position.y = -0.1;
  right_tool_grasp_pose.position.z = 1.1;
  right_tool_grasp_pose.orientation.x = 0.6863740920938239;
  right_tool_grasp_pose.orientation.y = 0.3315564820939015;
  right_tool_grasp_pose.orientation.z = 0.5828341014711744;
  right_tool_grasp_pose.orientation.w = -0.2815409651297376;

  auto left_tool_grasp_pose = geometry_msgs::Pose(right_tool_grasp_pose);
  left_tool_grasp_pose.position.x = right_tool_grasp_pose.position.z - 0.8;
  left_tool_grasp_pose.orientation.x = 0.5474187909624268;
  left_tool_grasp_pose.orientation.y = 0.547418790962427;
  left_tool_grasp_pose.orientation.z = -0.44758537431559875;
  left_tool_grasp_pose.orientation.w = 0.4475853743155988;

  RopeResetPlanner rope_reset_planner;
  auto const traj_msg = rope_reset_planner.planToReset(left_tool_grasp_pose, right_tool_grasp_pose, 1);
  return 0;
}