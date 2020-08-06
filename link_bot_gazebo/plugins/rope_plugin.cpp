#include "rope_plugin.h"

#include <std_srvs/EmptyRequest.h>

#include <link_bot_gazebo/gazebo_plugin_utils.h>
#include <boost/regex.hpp>

#include <cstdio>
#include <gazebo/common/Time.hh>
#include <gazebo/common/Timer.hh>
#include <memory>
#include <sstream>

#include "enumerate.h"

namespace gazebo
{
auto constexpr PLUGIN_NAME{ "RopePlugin" };

void RopePlugin::Load(physics::ModelPtr const parent, sdf::ElementPtr const sdf)
{
  model_ = parent;

  // Make sure the ROS node for Gazebo has already been initialized
  if (!ros::isInitialized())
  {
    auto argc = 0;
    char **argv = nullptr;
    ros::init(argc, argv, "rope_plugin", ros::init_options::NoSigintHandler);
  }

  rope_link1_ = GetLink(PLUGIN_NAME, model_, "rope_link_1");
  gripper1_ = GetLink(PLUGIN_NAME, model_, "gripper1");
  gripper2_ = GetLink(PLUGIN_NAME, model_, "gripper2");
  if (gripper1_ and rope_link1_)
  {
    rest_distance_between_gripper1_and_link_1_ =
        (gripper1_->WorldPose().Pos() - rope_link1_->WorldPose().Pos()).Length();
  }

  auto set_state_bind = [this](auto &&req, auto &&res) { return SetRopeState(req, res); };
  auto set_state_so = ros::AdvertiseServiceOptions::create<peter_msgs::SetRopeState>("set_rope_state", set_state_bind,
                                                                                     ros::VoidPtr(), &queue_);

  auto get_state_bind = [this](auto &&req, auto &&res) { return GetRopeState(req, res); };
  auto get_state_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetRopeState>("get_rope_state", get_state_bind,
                                                                                     ros::VoidPtr(), &queue_);

  auto overstretched_bind = [this](auto &&req, auto &&res) { return GetOverstretched(req, res); };
  auto overstretched_so = ros::AdvertiseServiceOptions::create<peter_msgs::GetBool>(
      "rope_overstretched", overstretched_bind, ros::VoidPtr(), &queue_);

  set_state_service_ = ros_node_.advertiseService(set_state_so);
  rope_overstretched_service_ = ros_node_.advertiseService(overstretched_so);
  get_state_service_ = ros_node_.advertiseService(get_state_so);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });

  {
    if (sdf->HasElement("overstretching_factor"))
    {
      overstretching_factor_ = sdf->GetElement("overstretching_factor")->Get<double>();
    }

    if (!sdf->HasElement("num_links"))
    {
      printf("using default num_links=%u\n", num_links_);
    }
    else
    {
      num_links_ = sdf->GetElement("num_links")->Get<unsigned int>();
    }
  }
  ROS_INFO("Rope Plugin finished initializing!");
}

bool RopePlugin::SetRopeState(peter_msgs::SetRopeStateRequest &req, peter_msgs::SetRopeStateResponse &)
{
  for (auto pair : enumerate(model_->GetJoints()))
  {
    auto const &[i, joint] = pair;
    if (i < req.joint_angles_axis1.size())
    {
      joint->SetPosition(0, req.joint_angles_axis1[i]);
      joint->SetPosition(1, req.joint_angles_axis2[i]);
    }
  }
  if (gripper1_ and gripper2_)
  {
    gripper1_->SetWorldPose({ req.gripper1.x, req.gripper1.y, req.gripper1.z, 0, 0, 0 });
    gripper2_->SetWorldPose({ req.gripper2.x, req.gripper2.y, req.gripper2.z, 0, 0, 0 });
  }
  else
  {
    ROS_ERROR_STREAM("Tried to set link to pose but couldn't find the gripper links");
    ROS_ERROR_STREAM("Available link names are");
    for (auto const l : model_->GetLinks())
    {
      ROS_ERROR_STREAM(l->GetName());
    }
  }

  return true;
}

bool RopePlugin::GetRopeState(peter_msgs::GetRopeStateRequest &, peter_msgs::GetRopeStateResponse &res)
{
  static peter_msgs::GetRopeStateResponse previous_res;
  static auto initialized = false;

  for (auto const &joint : model_->GetJoints())
  {
    res.joint_angles_axis1.push_back(joint->Position(0));
    res.joint_angles_axis2.push_back(joint->Position(1));
  }
  for (auto const &pair : enumerate(model_->GetLinks()))
  {
    auto const &[i, link] = pair;
    auto const name = link->GetName();
    boost::regex e(".*rope_link_\\d+");
    if (boost::regex_match(name, e))
    {
      geometry_msgs::Point pt;
      pt.x = link->WorldPose().Pos().X();
      pt.y = link->WorldPose().Pos().Y();
      pt.z = link->WorldPose().Pos().Z();
      res.positions.emplace_back(pt);

      geometry_msgs::Point velocity;
      if (initialized)
      {
        velocity.x = pt.x - previous_res.positions[i].x;
        velocity.y = pt.y - previous_res.positions[i].y;
        velocity.z = pt.z - previous_res.positions[i].z;
      }
      else
      {
        velocity.x = 0;
        velocity.y = 0;
        velocity.z = 0;
      }
      res.velocities.emplace_back(velocity);
    }
    else
    {
      // ROS_INFO_STREAM("skipping link with name " << name);
    }
  }
  res.model_pose.position.x = model_->WorldPose().Pos().X();
  res.model_pose.position.y = model_->WorldPose().Pos().Y();
  res.model_pose.position.z = model_->WorldPose().Pos().Z();
  res.model_pose.orientation.x = model_->WorldPose().Rot().X();
  res.model_pose.orientation.y = model_->WorldPose().Rot().Y();
  res.model_pose.orientation.z = model_->WorldPose().Rot().Z();
  res.model_pose.orientation.w = model_->WorldPose().Rot().W();

  previous_res = res;
  initialized = true;

  return true;
}

bool RopePlugin::GetOverstretched(peter_msgs::GetBoolRequest &req, peter_msgs::GetBoolResponse &res)
{
  (void)req;  // unused

  // check the distance between the position of rope_link_1 and gripper_1
  if (not gripper1_ or not rope_link1_)
  {
    return false;
  }
  auto const distance = (gripper1_->WorldPose().Pos() - rope_link1_->WorldPose().Pos()).Length();
  res.data = distance > (rest_distance_between_gripper1_and_link_1_ * overstretching_factor_);
  return true;
}

void RopePlugin::QueueThread()
{
  double constexpr timeout = 0.01;
  while (ros_node_.ok())
  {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

RopePlugin::~RopePlugin()
{
  queue_.clear();
  queue_.disable();
  ros_node_.shutdown();
  ros_queue_thread_.join();
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(RopePlugin)
}  // namespace gazebo
