#include <arm_moveit/rope_reset_planner.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <iostream>

namespace ob = ompl::base;
namespace oc = ompl::control;

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
std::vector<double> values_to_vector(T *state_or_control, int size) {
  std::vector<double> vec;
  for (auto i{0}; i < size; ++i) {
    vec.push_back((*state_or_control)[i]);
  }
  return vec;
}

void copy_vector_to_values(ob::RealVectorStateSpace::StateType *state_or_control, std::vector<double> vec) {
  for (auto i{0u}; i < vec.size(); ++i) {
    (*state_or_control)[i] = vec[i];
  }
}

class ArmStateSpace : public ob::CompoundStateSpace {
 public:
  std::vector<std::string> joint_names_;
  ros::Publisher &display_robot_state_pub_;
  class StateType : public CompoundStateSpace::StateType {
   public:
    StateType() = default;

    [[nodiscard]] auto getPositionState() { return this->operator[](0)->as<ob::RealVectorStateSpace::StateType>(); }

    [[nodiscard]] auto getPositionState() const {
      return this->operator[](0)->as<ob::RealVectorStateSpace::StateType>();
    }

    [[nodiscard]] auto getPositionValues() const {
      return this->operator[](0)->as<ob::RealVectorStateSpace::StateType>()->values;
    }

    [[nodiscard]] auto getVelocityState() { return this->operator[](1)->as<ob::RealVectorStateSpace::StateType>(); }

    [[nodiscard]] auto getVelocityState() const {
      return this->operator[](1)->as<ob::RealVectorStateSpace::StateType>();
    }

    [[nodiscard]] auto getVelocityValues() const {
      return this->operator[](1)->as<ob::RealVectorStateSpace::StateType>()->values;
    }

    [[nodiscard]] auto getAccelerationState() { return this->operator[](2)->as<ob::RealVectorStateSpace::StateType>(); }

    [[nodiscard]] auto getAccelerationState() const {
      return this->operator[](2)->as<ob::RealVectorStateSpace::StateType>();
    }

    [[nodiscard]] auto getAccelerationValues() const {
      return this->operator[](2)->as<ob::RealVectorStateSpace::StateType>()->values;
    }

    void setPositions(std::vector<double> const &positions) { copy_vector_to_values(getPositionState(), positions); }

    void setVelocities(std::vector<double> const &velocities) { copy_vector_to_values(getVelocityState(), velocities); }
    void setAccelerations(std::vector<double> const &accelerations) {
      copy_vector_to_values(getAccelerationState(), accelerations);
    }
  };

  explicit ArmStateSpace(std::vector<std::string> const &joint_names, ros::Publisher &publisher)
      : joint_names_(joint_names), display_robot_state_pub_(publisher) {
    auto const dim = joint_names.size();
    auto const &position_subspace = std::make_shared<ob::RealVectorStateSpace>(dim);
    position_subspace->setName("position");
    auto const &velocity_subspace = std::make_shared<ob::RealVectorStateSpace>(dim);
    velocity_subspace->setName("velocity");
    auto const &acceleration_subspace = std::make_shared<ob::RealVectorStateSpace>(dim);
    acceleration_subspace->setName("acceleration");

    for (auto i{0u}; i < dim; ++i) {
      position_subspace->setDimensionName(i, joint_names[i]);
      velocity_subspace->setDimensionName(i, joint_names[i]);
      acceleration_subspace->setDimensionName(i, joint_names[i]);
    }

    addSubspace(position_subspace, 1.0);
    addSubspace(velocity_subspace, 0);
    addSubspace(acceleration_subspace, 0);
    lock();
  }

  ob::RealVectorStateSpace *getPositionSpace() { return getSubspace("position")->as<ob::RealVectorStateSpace>(); }

  ob::RealVectorStateSpace *getVelocitySpace() { return getSubspace("velocity")->as<ob::RealVectorStateSpace>(); }
  ob::RealVectorStateSpace *getAccelerationSpace() {
    return getSubspace("acceleration")->as<ob::RealVectorStateSpace>();
  }

  int getJointIndex(std::string const &joint_name) {
    return getSubspace("position")->as<ob::RealVectorStateSpace>()->getDimensionIndex(joint_name);
  }

  void displayRobotState(std::vector<double> const &positions) {
    moveit_msgs::DisplayRobotState display_msg;
    display_msg.state.joint_state.name = joint_names_;
    display_msg.state.joint_state.position = positions;
    display_robot_state_pub_.publish(display_msg);
  }
};

bool isStateValid(const oc::SpaceInformation *si, const ob::State *state) {
  //  const auto *real_state = state->as<ArmStateSpace::StateType>();
  // FIXME: define the other constraints
  return si->satisfiesBounds(state);
}

void propagate(const moveit_visual_tools::MoveItVisualTools &visual_tools, const std::shared_ptr<ArmStateSpace> &space,
               const std::shared_ptr<oc::RealVectorControlSpace> &cspace, const ob::State *start,
               const oc::Control *control, const double dt, ob::State *result) {
  const auto arm_state = start->as<ArmStateSpace::StateType>();
  const auto result_arm_state = result->as<ArmStateSpace::StateType>();
  const auto real_control_ptr = control->as<oc::RealVectorControlSpace::ControlType>();

  auto result_qddot_ptr = result_arm_state->getAccelerationState();
  auto result_qdot_ptr = result_arm_state->getVelocityState();
  auto result_q_ptr = result_arm_state->getPositionState();
  auto const state_qddot_ptr = arm_state->getAccelerationState();
  auto const state_qdot_ptr = arm_state->getVelocityState();
  auto const state_q_ptr = arm_state->getPositionState();

  auto const n_joints = cspace->getDimension();
  auto const state_qddot = values_to_vector(state_qddot_ptr, n_joints);
  auto const state_qdot = values_to_vector(state_qdot_ptr, n_joints);
  auto const state_q = values_to_vector(state_q_ptr, n_joints);
  auto const real_control = values_to_vector(real_control_ptr, n_joints);

  auto const &result_qddot = real_control;
  auto const result_qdot = state_qdot + dt * state_qddot;
  auto const result_q = state_q + dt * state_qdot + 0.5 * dt * dt * state_qddot;

  copy_vector_to_values(result_qddot_ptr, result_qddot);
  copy_vector_to_values(result_qdot_ptr, result_qdot);
  copy_vector_to_values(result_q_ptr, result_q);

  space->displayRobotState(result_q);
  //  std::cout << dt << std::endl;
  //  std::cout << result_qddot;
  //  std::cout << result_qdot;
  //  std::cout << result_q;
}

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

moveit_msgs::RobotTrajectory RopeResetPlanner::planToReset(geometry_msgs::Pose const &, geometry_msgs::Pose const &,
                                                           double timeout) {
  auto o = ros::AdvertiseOptions::create<moveit_msgs::DisplayRobotState>(
      "rope_reset_state", 10, ros::SubscriberStatusCallback(), ros::SubscriberStatusCallback(), ros::VoidConstPtr(),
      &queue_);

  auto pub = nh_.advertise(o);
  moveit_msgs::RobotTrajectory msg;
  msg.joint_trajectory.header.stamp = ros::Time::now();

  auto const &group = model_->getJointModelGroup("both_arms");
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
  // TODO: get bounds from moveit
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
  space->getPositionSpace()->setBounds(position_bounds);
  space->getVelocitySpace()->setBounds(velocity_bounds);
  space->getAccelerationSpace()->setBounds(acceleration_bounds);

  auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, n_joints));

  ob::RealVectorBounds cbounds(n_joints);
  // TODO: get bounds from moveit
  cbounds.setLow(-1);
  cbounds.setHigh(1);
  cspace->setBounds(cbounds);

  oc::SimpleSetup ss(cspace);
  auto const &si = ss.getSpaceInformation();
  si->setPropagationStepSize(0.01);

  auto _propagate = [&](const ob::State *start, const oc::Control *control, const double dt, ob::State *result) {
    propagate(visual_tools_, space, cspace, start, control, dt, result);
  };

  ss.setStatePropagator(_propagate);

  ss.setStateValidityChecker(
      [&ss](const ob::State *state) { return isStateValid(ss.getSpaceInformation().get(), state); });

  ob::ScopedState<ArmStateSpace> start(space);
  for (auto const &joint_name : joint_names) {
    auto const i = space->getJointIndex(joint_name);
    start->getPositionValues()[i] = start_robot_state.getVariablePosition(joint_name);
    start->getVelocityValues()[i] = 0;
    start->getAccelerationValues()[i] = 0;
  }

  std::cout << "Start State: \n";
  space->printState(start.get(), std::cout);

  ob::ScopedState<ArmStateSpace> goal(space);
  std::vector<double> zeros(n_joints, 0);
  // FIXME: use IK to sample a bunch of goals in joint space? try to find ones near start joint config?
  goal->setPositions(zeros);
  // FIXME: ^^^^^^^^^^^^^^^^^^^^^ THIS IS TEMPORARY
  goal->setVelocities(zeros);
  goal->setAccelerations(zeros);

  ss.setStartAndGoalStates(start, goal, 0.1);

  ss.setPlanner(std::make_shared<oc::SST>(ss.getSpaceInformation()));
  ss.getSpaceInformation()->setMinMaxControlDuration(1, 10);
  std::cout << "Starting..." << std::endl;
  ob::PlannerStatus solved = ss.solve(timeout);

  if (solved) {
    auto time_from_start = ros::Duration(0);
    auto j{0u};
    auto &path = ss.getSolutionPath();
    std::cout << "Solution has " << path.getStateCount() << " states\n";
    for (auto const &state : path.getStates()) {
      const auto &arm_state = *state->as<ArmStateSpace::StateType>();
      if (j < path.getControlCount()) {
        const auto &dt = path.getControlDuration(j);
        time_from_start += ros::Duration(dt);
      }
      trajectory_msgs::JointTrajectoryPoint point_msg;
      point_msg.time_from_start = time_from_start;
      point_msg.positions = values_to_vector(arm_state.getPositionState(), n_joints);
      point_msg.velocities = values_to_vector(arm_state.getVelocityState(), n_joints);
      point_msg.accelerations = values_to_vector(arm_state.getAccelerationState(), n_joints);
      msg.joint_trajectory.points.push_back(point_msg);
      ++j;
    }
  } else {
    std::cout << "No solution found" << std::endl;
  }

  return msg;
}
