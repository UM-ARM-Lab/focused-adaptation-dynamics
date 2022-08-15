#include <arm_moveit/rope_reset_planner.h>
#include <bio_ik/bio_ik.h>
#include <eigen_conversions/eigen_msg.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <tf2_eigen/tf2_eigen.h>

#include <iostream>

namespace ob = ompl::base;
namespace og = ompl::geometric;

namespace std {
template <typename T>
ostream &operator<<(ostream &os, const vector<T> &v) {
  os << "[";
  for (auto i{0u}; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) os << ", ";
  }
  os << "]\n";
  return os;
}
}  // namespace std

template <typename T, typename T2>
std::vector<double> operator+(const std::vector<T> &vec1, const std::vector<T2> &vec2) {
  if (vec1.size() != vec2.size()) {
    throw std::invalid_argument("Error: vectors must be the same size");
  }
  std::vector<double> newvec(vec1.size());
  for (unsigned int i = 0; i < vec1.size(); i++) {
    newvec[i] = vec1[i] + vec2[i];
  }
  return newvec;
}

template <typename T, typename T2>
std::vector<double> operator*(const T &scalar, const std::vector<T2> &vec) {
  std::vector<double> newvec(vec.size());
  for (unsigned int i = 0; i < vec.size(); i++) {
    newvec[i] = scalar * vec[i];
  }
  return newvec;
}

template <typename T>
std::vector<double> values_to_vector(T *state, int size) {
  std::vector<double> vec;
  for (auto i{0}; i < size; ++i) {
    vec.push_back((*state)[i]);
  }
  return vec;
}

void copy_vector_to_values(ob::RealVectorStateSpace::StateType *state, std::vector<double> vec) {
  for (auto i{0u}; i < vec.size(); ++i) {
    (*state)[i] = vec[i];
  }
}

class ArmStateSpace : public ob::RealVectorStateSpace {
 public:
  std::vector<std::string> joint_names_;
  ros::Publisher &display_robot_state_pub_;
  class StateType : public RealVectorStateSpace::StateType {
   public:
    StateType() = default;

    void setPositions(std::vector<double> const &positions) { copy_vector_to_values(this, positions); }
  };

  explicit ArmStateSpace(std::vector<std::string> const &joint_names, ros::Publisher &publisher)
      : ob::RealVectorStateSpace(joint_names.size()), joint_names_(joint_names), display_robot_state_pub_(publisher) {
    auto const dim = joint_names.size();
    for (auto i{0u}; i < dim; ++i) {
      setDimensionName(i, joint_names[i]);
    }
  }

  int getJointIndex(std::string const &joint_name) { return getDimensionIndex(joint_name); }

  void displayRobotState(std::vector<double> const &positions) {
    moveit_msgs::DisplayRobotState display_msg;
    display_msg.state.joint_state.name = joint_names_;
    display_msg.state.joint_state.position = positions;
    display_robot_state_pub_.publish(display_msg);
  }
};

bool isStateValid(const ob::SpaceInformation *si, const ob::State *state) {
  //  const auto *real_state = state->as<ArmStateSpace::StateType>();
  // FIXME: define the other constraints
  return si->satisfiesBounds(state);
}

auto rotMatDist(const Eigen::Matrix3d P, const Eigen::Matrix3d Q) {
  // http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
  Eigen::Matrix3d R = P * Q.transpose();
  auto const d = std::acos((R.trace() - 1) / 2);
  return d;
}

auto eigenVectorTotf2Vector(Eigen::Vector3d const &v) { return tf2::Vector3{v.x(), v.y(), v.z()}; }

auto eigenMatToQuaternion(Eigen::Matrix3d const &r) {
  Eigen::Quaterniond q(r);
  return tf2::Quaternion{q.x(), q.y(), q.z(), q.w()};
}

class PosesGoal : public ob::GoalSampleableRegion {
 public:
  moveit::core::RobotModelConstPtr const model_;
  moveit::core::JointModelGroup const *group_;
  moveit_visual_tools::MoveItVisualTools &visual_tools_;
  Eigen::Isometry3d left_goal_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d right_goal_pose_{Eigen::Isometry3d::Identity()};
  double const translation_tolerance_;
  double const orientation_tolerance_;

  PosesGoal(moveit::core::RobotModelConstPtr const &model, moveit::core::JointModelGroup const *group,
            moveit_visual_tools::MoveItVisualTools &visual_tools, const ob::SpaceInformationPtr &si,
            geometry_msgs::Pose const &left_pose, geometry_msgs::Pose const &right_pose, double translation_tolerance,
            double orientation_tolerance)
      : ompl::base::GoalSampleableRegion(si),
        model_(model),
        group_(group),
        visual_tools_(visual_tools),
        translation_tolerance_(translation_tolerance),
        orientation_tolerance_(orientation_tolerance) {
    tf::poseMsgToEigen(left_pose, left_goal_pose_);
    tf::poseMsgToEigen(right_pose, right_goal_pose_);
  }

  double distanceGoal(const ob::State *s) const override {
    // TODO: reduce code duplication

    // compute EE poses with FK using Moveit
    robot_state::RobotState robot_state(model_);
    robot_state.setVariablePositions(s->as<ob::RealVectorStateSpace::StateType>()->values);
    const auto &left_tool_pose = robot_state.getGlobalLinkTransform("left_tool");
    const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");

    // Compute error as sum of translation error and orientation (quaternion) error
    auto const left_orientation_error = rotMatDist(left_goal_pose_.rotation(), left_tool_pose.rotation());
    auto const right_orientation_error = rotMatDist(right_goal_pose_.rotation(), right_tool_pose.rotation());
    auto const orientation_error = left_orientation_error + right_orientation_error;
    auto const left_translation_error = (left_goal_pose_.translation() - left_tool_pose.translation()).norm();
    auto const right_translation_error = (right_goal_pose_.translation() - right_tool_pose.translation()).norm();
    auto const translation_error = left_translation_error + right_translation_error;
    return translation_error + orientation_error;
  }

  bool isSatisfied(const ob::State *s) const override {
    // compute EE poses with FK using Moveit
    robot_state::RobotState robot_state(model_);
    robot_state.setVariablePositions(s->as<ob::RealVectorStateSpace::StateType>()->values);
    const auto &left_tool_pose = robot_state.getGlobalLinkTransform("left_tool");
    const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");

    // Compute error as sum of translation error and orientation (quaternion) error
    auto const right_translation_error = (right_goal_pose_.translation() - right_tool_pose.translation()).norm();
    auto const left_orientation_error = rotMatDist(left_goal_pose_.rotation(), left_tool_pose.rotation());
    auto const right_orientation_error = rotMatDist(right_goal_pose_.rotation(), right_tool_pose.rotation());
    auto const left_translation_error = (left_goal_pose_.translation() - left_tool_pose.translation()).norm();
    return (left_translation_error < translation_tolerance_) && (right_translation_error < translation_tolerance_) &&
           (left_orientation_error < orientation_tolerance_) && (right_orientation_error < orientation_tolerance_);
  }

  void sampleGoal(ob::State *s) const override {
    bio_ik::BioIKKinematicsQueryOptions opts;
    opts.replace = true;
    opts.return_approximate_solution = false;

    auto const left_position_tf2 = eigenVectorTotf2Vector(left_goal_pose_.translation());
    auto const left_quat_tf2 = eigenMatToQuaternion(left_goal_pose_.rotation());
    opts.goals.emplace_back(std::make_unique<bio_ik::PoseGoal>("left_tool", left_position_tf2, left_quat_tf2));

    auto const right_position_tf2 = eigenVectorTotf2Vector(right_goal_pose_.translation());
    auto const right_quat_tf2 = eigenMatToQuaternion(right_goal_pose_.rotation());
    opts.goals.emplace_back(std::make_unique<bio_ik::PoseGoal>("right_tool", right_position_tf2, right_quat_tf2));

    robot_state::RobotState robot_state_ik{model_};
    robot_state_ik.update();

    // run IK to sample goals satisfying the poses constraints
    // Collision checking
    moveit::core::GroupStateValidityCallbackFn empty_constraint_fn;

    bool ok = false;
    while (not ok) {
      robot_state_ik.setToRandomPositionsNearBy(group_, robot_state_ik, 0.1);
      ok = robot_state_ik.setFromIK(group_,                         // joints to be used for IK
                                    EigenSTL::vector_Isometry3d(),  // this isn't used, goals are described in opts
                                    std::vector<std::string>(),     // names of the end-effector links
                                    0,                              // take values from YAML
                                    empty_constraint_fn,
                                    opts  // mostly empty
      );
    }

    visual_tools_.publishRobotState(robot_state_ik);

    // copy IK solution into OMPL state
    auto *real_state = s->as<ob::RealVectorStateSpace::StateType>();
    auto i{0u};
    for (auto const &joint_name : group_->getActiveJointModelNames()) {
      (*real_state)[i] = robot_state_ik.getVariablePosition(joint_name);
      ++i;
    }
  }

  [[nodiscard]] unsigned int maxSampleCount() const override { return 10u; }
};

RopeResetPlanner::RopeResetPlanner()
    : model_loader_(std::make_shared<robot_model_loader::RobotModelLoader>("hdt_michigan/robot_description")),
      model_(model_loader_->getModel()),
      scene_monitor_(std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(model_loader_)),
      visual_tools_("robot_root", "hdt_michigan/moveit_visual_markers", model_) {
  auto const scene_topic = "hdt_michigana/move_group/monitored_planning_scene";
  scene_monitor_->startSceneMonitor(scene_topic);
  auto const service_name = "hdt_michigan/get_planning_scene";
  scene_monitor_->requestPlanningSceneState(service_name);

  ros_queue_thread_ = std::thread([this] { QueueThread(); });
}

RopeResetPlanner::~RopeResetPlanner() {
  queue_.clear();
  queue_.disable();
  nh_.shutdown();
  ros_queue_thread_.join();
}

void RopeResetPlanner::QueueThread() {
  double constexpr timeout = 0.01;
  while (nh_.ok()) {
    queue_.callAvailable(ros::WallDuration(timeout));
  }
}

moveit_msgs::RobotTrajectory RopeResetPlanner::planToReset(geometry_msgs::Pose const &left_pose,
                                                           geometry_msgs::Pose const &right_pose, double timeout) {
  auto o = ros::AdvertiseOptions::create<moveit_msgs::DisplayRobotState>(
      "rope_reset_state", 10, ros::SubscriberStatusCallback(), ros::SubscriberStatusCallback(), ros::VoidConstPtr(),
      &queue_);

  auto pub = nh_.advertise(o);
  moveit_msgs::RobotTrajectory msg;
  msg.joint_trajectory.header.stamp = ros::Time::now();

  auto const *group = model_->getJointModelGroup("both_arms");
  auto const &joint_names = group->getActiveJointModelNames();
  msg.joint_trajectory.joint_names = joint_names;
  std::cout << joint_names << "\n";

  scene_monitor_->lockSceneRead();
  auto planning_scene = planning_scene::PlanningScene::clone(scene_monitor_->getPlanningScene());
  scene_monitor_->unlockSceneRead();
  auto const &start_robot_state = planning_scene->getCurrentState();

  auto const n_joints = group->getActiveVariableCount();
  auto space(std::make_shared<ArmStateSpace>(joint_names, pub));

  ob::RealVectorBounds position_bounds(n_joints);
  ob::RealVectorBounds velocity_bounds(n_joints);
  ob::RealVectorBounds acceleration_bounds(n_joints);
  auto const bounds = group->getActiveJointModelsBounds();
  auto joint_i{0u};
  for (auto const *joint_bounds : bounds) {
    auto const &joint_name = group->getActiveJointModelNames()[joint_i];
    if (joint_bounds->size() != 1) {
      std::stringstream ss;
      ss << "Joint " << joint_name << " has " << joint_bounds->size() << " bounds\n";
      throw std::runtime_error(ss.str());
    } else {
      auto const &bound = (*joint_bounds)[0];
      // std::cout << joint_name << " " << bound << std::endl;
      position_bounds.setLow(joint_i, bound.min_position_);
      position_bounds.setHigh(joint_i, bound.max_position_);
      velocity_bounds.setLow(joint_i, bound.min_velocity_);
      velocity_bounds.setHigh(joint_i, bound.max_velocity_);
      acceleration_bounds.setLow(joint_i, bound.min_acceleration_);
      acceleration_bounds.setHigh(joint_i, bound.max_acceleration_);
      ++joint_i;
    }
  }
  space->setBounds(position_bounds);

  og::SimpleSetup ss(space);
  auto const &si = ss.getSpaceInformation();

  ss.setStateValidityChecker(
      [&ss](const ob::State *state) { return isStateValid(ss.getSpaceInformation().get(), state); });

  ob::ScopedState<ArmStateSpace> start(space);
  for (auto const &joint_name : joint_names) {
    auto const i = space->getJointIndex(joint_name);
    start[i] = start_robot_state.getVariablePosition(joint_name);
  }
  std::cout << "Start State: \n";
  space->printState(start.get(), std::cout);
  ss.setStartState(start);

  std::vector<double> zeros(n_joints, 0);
  // TODO:
  //  - use a GoalSampleableRegion to sample joint positions that obey the left and right pose constraints
  //  - make a custom state sampler which ensure the right orientation constraint is satisfied
  //  - add collision checking to isStateValid
  //  - add "visiblity constraint" to isStateValid
  //  - add "visiblity constraint" to isStateValid

  auto goal = std::make_shared<PosesGoal>(model_, group, visual_tools_, si, left_pose, right_pose, 0.01, 0.1);
  ss.setGoal(goal);

  auto const &planner = std::make_shared<og::RRTConnect>(ss.getSpaceInformation());
  planner->setRange(0.1);
  ss.setPlanner(planner);
  std::cout << "Starting..." << std::endl;
  ob::PlannerStatus solved = ss.solve(timeout);

  if (solved) {
    auto &path = ss.getSolutionPath();
    std::cout << "Solution has " << path.getStateCount() << " states\n";
    for (auto const &state : path.getStates()) {
      const auto &arm_state = state->as<ArmStateSpace::StateType>();
      trajectory_msgs::JointTrajectoryPoint point_msg;
      point_msg.positions = values_to_vector(arm_state, n_joints);
      msg.joint_trajectory.points.push_back(point_msg);
    }
  } else {
    std::cout << "No solution found" << std::endl;
  }

  return msg;
}
