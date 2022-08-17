#include <arm_moveit/rope_reset_planner.h>
#include <bio_ik/bio_ik.h>
#include <eigen_conversions/eigen_msg.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include <iostream>

namespace ob = ompl::base;
namespace og = ompl::geometric;

constexpr auto LOGGER_NAME{"rope_reset_planner"};

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

void copy_vector_to_values(ob::RealVectorStateSpace::StateType *state, std::vector<double> vec) {
  for (auto i{0u}; i < vec.size(); ++i) {
    (*state)[i] = vec[i];
  }
}

class ArmStateSpace : public ob::RealVectorStateSpace {
 public:
  std::vector<std::string> joint_names_;
  class StateType : public RealVectorStateSpace::StateType {
   public:
    StateType() = default;

    void setPositions(std::vector<double> const &positions) { copy_vector_to_values(this, positions); }
  };

  explicit ArmStateSpace(std::vector<std::string> const &joint_names)
      : ob::RealVectorStateSpace(joint_names.size()), joint_names_(joint_names) {
    auto const dim = joint_names.size();
    for (auto i{0u}; i < dim; ++i) {
      setDimensionName(i, joint_names[i]);
    }
  }

  int getJointIndex(std::string const &joint_name) { return getDimensionIndex(joint_name); }
};

auto rotMatDist(const Eigen::Matrix3d P, const Eigen::Matrix3d Q) {
  // http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
  Eigen::Matrix3d R = P * Q.transpose();
  auto const d = std::acos((R.trace() - 1) / 2);
  return d;
}

// equivalent metric to the above rotMatDist
double quatDist(Eigen::Quaterniond const &q1, Eigen::Quaterniond const &q2) { return q1.angularDistance(q2); }

auto eigenVectorTotf2Vector(Eigen::Vector3d const &v) { return tf2::Vector3{v.x(), v.y(), v.z()}; }

auto eigenQuaternionTotf2Quaternion(Eigen::Quaterniond const &q) { return tf2::Quaternion{q.x(), q.y(), q.z(), q.w()}; }

auto eigenMatToQuaternion(Eigen::Matrix3d const &r) {
  Eigen::Quaterniond q(r);
  return eigenQuaternionTotf2Quaternion(q);
}

void addGripperCollisionSpheres(moveit::core::RobotState &robot_state) {
  auto addGripperCollisionSphere = [&](std::string const &side) {
    auto shape = std::make_shared<shapes::Sphere>(0.025);
    Eigen::Isometry3d identity{Eigen::Isometry3d::Identity()};
    std::vector<shapes::ShapeConstPtr> shapes{shape};
    EigenSTL::vector_Isometry3d poses{identity};
    std::vector<std::string> touch_links{"end_effector_" + side, side + "gripper_link", side + "gripper2_link",
                                         side + "_tool"};
    robot_state.attachBody(side + "_tool_aco", identity, shapes, poses, touch_links, side + "_tool");
  };

  addGripperCollisionSphere("left");
  addGripperCollisionSphere("right");
}

auto ik(moveit::core::RobotModelConstPtr const &model, moveit::core::JointModelGroup const *group, ob::State *s,
        bio_ik::BioIKKinematicsQueryOptions const &opts) {
  robot_state::RobotState robot_state_ik{model};
  addGripperCollisionSpheres(robot_state_ik);
  robot_state_ik.setToDefaultValues();

  moveit::core::GroupStateValidityCallbackFn empty_constraint_fn;

  bool ok = false;
  while (not ok) {
    robot_state_ik.setToRandomPositionsNearBy(group, robot_state_ik, 0.1);
    ok = robot_state_ik.setFromIK(group, EigenSTL::vector_Isometry3d(), std::vector<std::string>(), 0,
                                  empty_constraint_fn, opts);
  }

  return robot_state_ik;
}

void copy_robot_state_to_ompl_state(moveit::core::JointModelGroup const *group, moveit::core::RobotState robot_state,
                                    ob::State *s) {
  auto *real_state = s->as<ob::RealVectorStateSpace::StateType>();
  auto i{0u};
  for (auto const &joint_name : group->getActiveJointModelNames()) {
    (*real_state)[i] = robot_state.getVariablePosition(joint_name);
    ++i;
  }
}

auto omplStateToRobotState(ob::State const *state, moveit::core::RobotModelConstPtr const &model,
                           std::shared_ptr<ob::StateSpace> const &space) {
  auto const &real_space = std::dynamic_pointer_cast<ob::RealVectorStateSpace>(space);
  robot_state::RobotState robot_state(model);
  robot_state.setToDefaultValues();
  for (auto i{0u}; i < real_space->getDimension(); ++i) {
    auto const joint_name = real_space->getDimensionName(i);
    auto const position = (*state->as<ob::RealVectorStateSpace::StateType>())[i];
    robot_state.setVariablePosition(joint_name, position);
  }
  addGripperCollisionSpheres(robot_state);
  robot_state.update();
  return robot_state;
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
    auto const robot_state = omplStateToRobotState(s, model_, si_->getStateSpace());
    const auto &left_tool_pose = robot_state.getGlobalLinkTransform("left_tool");
    const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");

    // Compute error as sum of translation error and orientation error
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
    auto const robot_state = omplStateToRobotState(s, model_, si_->getStateSpace());
    const auto &left_tool_pose = robot_state.getGlobalLinkTransform("left_tool");
    const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");

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

    auto const robot_state_ik = ik(model_, group_, s, opts);
    visual_tools_.publishRobotState(robot_state_ik);

    copy_robot_state_to_ompl_state(group_, robot_state_ik, s);
  }

  [[nodiscard]] unsigned int maxSampleCount() const override { return 10u; }
};

class GripperOrientationStateSampler : public ob::ValidStateSampler {
 public:
  moveit::core::RobotModelConstPtr const model_;
  moveit::core::JointModelGroup const *group_;
  moveit_visual_tools::MoveItVisualTools &visual_tools_;
  Eigen::Quaterniond target_orientation_;

  GripperOrientationStateSampler(const ob::SpaceInformation *si, moveit::core::RobotModelConstPtr const &model,
                                 moveit::core::JointModelGroup const *group,
                                 moveit_visual_tools::MoveItVisualTools &visual_tools,
                                 geometry_msgs::Quaternion const &orientation)
      : ValidStateSampler(si), model_(model), group_(group), visual_tools_(visual_tools) {
    tf::quaternionMsgToEigen(orientation, target_orientation_);
    name_ = "gripper_orientation_sampler";
  }
  bool sample(ob::State *s) override {
    // sample a joint configuration which obeys the orientation constraint.
    bio_ik::BioIKKinematicsQueryOptions opts;
    opts.replace = true;
    opts.return_approximate_solution = false;

    auto const quat_tf2 = eigenQuaternionTotf2Quaternion(target_orientation_);
    opts.goals.emplace_back(std::make_unique<bio_ik::OrientationGoal>("right_tool", quat_tf2));

    auto const robot_state_ik = ik(model_, group_, s, opts);
    visual_tools_.publishRobotState(robot_state_ik);

    copy_robot_state_to_ompl_state(group_, robot_state_ik, s);

    assert(si_->isValid(s));
    return true;
  }

  // None of the RRT based planners use this... although I could probably implement it
  bool sampleNear(ob::State * /*state*/, const ob::State * /*near*/, const double /*distance*/) override {
    throw ompl::Exception("GripperOrientationStateSampler::sampleNear", "not implemented");
    return false;
  }
};

RopeResetPlanner::RopeResetPlanner()
    : model_loader_(std::make_shared<robot_model_loader::RobotModelLoader>("hdt_michigan/robot_description")),
      model_(model_loader_->getModel()),
      scene_monitor_(std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(model_loader_)),
      visual_tools_("robot_root", "hdt_michigan/moveit_visual_markers", model_) {
  auto const scene_topic = "hdt_michigan/move_group/monitored_planning_scene";
  scene_monitor_->startSceneMonitor(scene_topic);
  auto const service_name = "hdt_michigan/get_planning_scene";
  scene_monitor_->requestPlanningSceneState(service_name);
}

std::pair<ob::PlannerStatus, moveit_msgs::RobotTrajectory> RopeResetPlanner::planToReset(geometry_msgs::Pose const &left_pose,
                                                           geometry_msgs::Pose const &right_pose, double timeout) {
  auto const orientation_tolerance = 0.1;

  std::string group_name = "both_arms";
  robot_trajectory::RobotTrajectory traj(model_, group_name);

  auto const *group = model_->getJointModelGroup(group_name);
  auto const &joint_names = group->getActiveJointModelNames();
  std::cout << joint_names << "\n";

  scene_monitor_->lockSceneRead();
  auto planning_scene = planning_scene::PlanningScene::clone(scene_monitor_->getPlanningScene());
  scene_monitor_->unlockSceneRead();
  auto start_robot_state = planning_scene->getCurrentState();

  addGripperCollisionSpheres(start_robot_state);

  auto const n_joints = group->getActiveVariableCount();
  auto space(std::make_shared<ArmStateSpace>(joint_names));

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

  auto alloc = [&](const ob::SpaceInformation *_si) {
    return std::make_shared<GripperOrientationStateSampler>(_si, model_, group, visual_tools_, right_pose.orientation);
  };

  si->setValidStateSamplerAllocator(alloc);
  ss.setStateValidityChecker([&](const ob::State *s) {
    auto const robot_state = omplStateToRobotState(s, model_, space);

    const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");

    Eigen::Isometry3d right_target_pose;
    tf::poseMsgToEigen(right_pose, right_target_pose);
    auto const right_orientation_error = rotMatDist(right_target_pose.rotation(), right_tool_pose.rotation());
    auto const right_orientation_satisfied = right_orientation_error < orientation_tolerance;

    auto const collision_free = planning_scene->isStateValid(robot_state);

    return si->satisfiesBounds(s) && right_orientation_satisfied && collision_free;
  });

  ob::ScopedState<ArmStateSpace> start(space);
  for (auto const &joint_name : joint_names) {
    auto const i = space->getJointIndex(joint_name);
    start[i] = start_robot_state.getVariablePosition(joint_name);
  }
  std::cout << "Start State: \n";
  space->printState(start.get(), std::cout);
  ss.setStartState(start);

  std::vector<double> zeros(n_joints, 0);

  auto goal =
      std::make_shared<PosesGoal>(model_, group, visual_tools_, si, left_pose, right_pose, 0.01, orientation_tolerance);
  ss.setGoal(goal);

  auto const &planner = std::make_shared<og::RRTConnect>(si);
  planner->setRange(0.1);
  ss.setPlanner(planner);
  std::cout << "Starting..." << std::endl;
  ob::PlannerStatus status = ss.solve(timeout);

  if (status) {
    auto &path = ss.getSolutionPath();
    ss.simplifySolution();  // not sure if this is safe... does the resulting path still satisfy constraints?
    std::cout << "Solution has " << path.getStateCount() << " states\n";
    for (auto const &state : path.getStates()) {
      const auto &arm_state = state->as<ArmStateSpace::StateType>();
      auto const robot_state = omplStateToRobotState(state, model_, space);
      traj.addSuffixWayPoint(robot_state, 0);
    }

    if (!time_param_.computeTimeStamps(traj, 1, 1)) {
      ROS_ERROR_STREAM_NAMED(LOGGER_NAME, "Time parametrization for the solution path failed.");
    }
  } else {
    std::cout << "No solution found" << std::endl;
  }

  moveit_msgs::RobotTrajectory traj_msg;
  traj.getRobotTrajectoryMsg(traj_msg);
  traj_msg.joint_trajectory.header.stamp = ros::Time::now();
  return {status, traj_msg};
}
