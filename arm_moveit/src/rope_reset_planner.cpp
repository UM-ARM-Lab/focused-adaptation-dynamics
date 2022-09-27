#include <arm_moveit/rope_reset_planner.h>
#include <bio_ik/bio_ik.h>
#include <eigen_conversions/eigen_msg.h>
#include <ompl/base/Constraint.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>

#include <iostream>

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
    auto shape = std::make_shared<shapes::Sphere>(0.05);
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

void addLinkPadding(planning_scene::PlanningScenePtr const &planning_scene) {
  auto &collision_env = planning_scene->getCollisionEnvNonConst();
  collision_env->setLinkPadding("drive56", 0.02);
  collision_env->setLinkPadding("drive57", 0.02);
  collision_env->setLinkPadding("torso", 0.03);
  collision_env->setLinkPadding("rightgripper_link", 0.02);
  collision_env->setLinkPadding("rightgripper2_link", 0.02);
  collision_env->setLinkPadding("leftgripper_link", 0.02);
  collision_env->setLinkPadding("leftgripper2_link", 0.02);
  collision_env->setLinkPadding("leftforearm", 0.02);
  collision_env->setLinkPadding("rightforearm", 0.02);
  collision_env->setLinkPadding("lefttube", 0.02);
  collision_env->setLinkPadding("righttube", 0.02);
  collision_env->setLinkPadding("drive6", 0.02);
  collision_env->setLinkPadding("drive46", 0.02);
  planning_scene->propogateRobotPadding();
}

std::optional<robot_state::RobotState> ik_near(moveit::core::JointModelGroup const *group,
                                               bio_ik::BioIKKinematicsQueryOptions const &opts, double rng_dist,
                                               robot_state::RobotState robot_state_ik) {
  moveit::core::GroupStateValidityCallbackFn empty_constraint_fn;

  bool ok = false;
  constexpr auto max_ik_attempts{25};
  for (auto i{0}; i < max_ik_attempts; ++i) {
    // robot_state_ik.setToRandomPositions(group);
    robot_state_ik.setToRandomPositionsNearBy(group, robot_state_ik, rng_dist);
    ok = robot_state_ik.setFromIK(group, EigenSTL::vector_Isometry3d(), std::vector<std::string>(), 0,
                                  empty_constraint_fn, opts);
    if (ok) {
      break;
    }
    ROS_DEBUG_STREAM_NAMED(LOGGER_NAME + ".ik", "sampling ik [" << i << "/" << max_ik_attempts << "]");
  }

  if (!ok) {
    return {};
  } else {
    return robot_state_ik;
  }
}

std::optional<robot_state::RobotState> ik(moveit::core::RobotModelConstPtr const &model,
                                          moveit::core::JointModelGroup const *group,
                                          bio_ik::BioIKKinematicsQueryOptions const &opts, double rng_dist) {
  robot_state::RobotState robot_state_ik_seed{model};
  addGripperCollisionSpheres(robot_state_ik_seed);
  robot_state_ik_seed.setToDefaultValues();
  return ik_near(group, opts, rng_dist, robot_state_ik_seed);
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
  moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
  Eigen::Isometry3d left_goal_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d right_goal_pose_{Eigen::Isometry3d::Identity()};
  double const translation_tolerance_;
  double const orientation_tolerance_;

  PosesGoal(moveit::core::RobotModelConstPtr const &model, moveit::core::JointModelGroup const *group,
            moveit_visual_tools::MoveItVisualToolsPtr visual_tools, const ob::SpaceInformationPtr &si,
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

    auto const joint56_err = abs(*robot_state.getJointPositions("joint56"));
    auto const joint57_err = std::max(*robot_state.getJointPositions("joint57") - (-0.05), 0.0);

    return translation_error + orientation_error + joint56_err + joint57_err;
  }

  bool isSatisfied(const ob::State *s, double *dist) const override {
    *dist = distanceGoal(s);

    // compute EE poses with FK using Moveit
    auto const robot_state = omplStateToRobotState(s, model_, si_->getStateSpace());
    const auto &left_tool_pose = robot_state.getGlobalLinkTransform("left_tool");
    const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");

    auto const right_translation_error = (right_goal_pose_.translation() - right_tool_pose.translation()).norm();
    auto const left_orientation_error = rotMatDist(left_goal_pose_.rotation(), left_tool_pose.rotation());
    auto const right_orientation_error = rotMatDist(right_goal_pose_.rotation(), right_tool_pose.rotation());
    auto const left_translation_error = (left_goal_pose_.translation() - left_tool_pose.translation()).norm();

    auto const torso_ok = *robot_state.getJointPositions("joint57") < -0.05;
    auto const torso_ok2 = abs(*robot_state.getJointPositions("joint56")) < 0.25;

    return (left_translation_error < translation_tolerance_) && (right_translation_error < translation_tolerance_) &&
           (left_orientation_error < orientation_tolerance_) && (right_orientation_error < orientation_tolerance_) &&
           torso_ok && torso_ok2;
  }

  bool isSatisfied(const ob::State * /*s*/) const override { throw std::runtime_error("Not implemented!"); }

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

    auto const rng_dist = 2;
    auto const robot_state_ik = ik(model_, group_, opts, rng_dist);

    if (!robot_state_ik) {
      // if we failed, try just the left or just the right pose constraint to give better error messages
      Eigen::IOFormat fmt(3);
      std::stringstream err_ss;

      bio_ik::BioIKKinematicsQueryOptions opts_right_only;
      opts_right_only.replace = true;
      opts_right_only.return_approximate_solution = false;
      opts_right_only.goals.emplace_back(
          std::make_unique<bio_ik::PoseGoal>("right_tool", right_position_tf2, right_quat_tf2));
      auto const robot_state_ik_right_only = ik(model_, group_, opts_right_only, rng_dist);

      bio_ik::BioIKKinematicsQueryOptions opts_left_only;
      opts_left_only.replace = true;
      opts_left_only.return_approximate_solution = false;
      opts_left_only.goals.emplace_back(
          std::make_unique<bio_ik::PoseGoal>("left_tool", left_position_tf2, left_quat_tf2));
      auto const robot_state_ik_left_only = ik(model_, group_, opts_left_only, rng_dist);

      if (!robot_state_ik_left_only && !robot_state_ik_right_only) {
        err_ss << "Failed to solve ik for GOAL, neither left nor right was solvable on their own:\n";
      } else if (!robot_state_ik_left_only && robot_state_ik_right_only) {
        err_ss << "Failed to solve ik for GOAL, left failed but right only succeeded:\n";
      } else if (robot_state_ik_left_only && !robot_state_ik_right_only) {
        err_ss << "Failed to solve ik for GOAL, right failed but left only succeeded:\n";
      } else {
        err_ss << "Failed to solve ik for GOAL. Both succeeded individually, but not together:\n";
      }
      err_ss << "left_pose:\n"
             << left_goal_pose_.matrix().format(fmt) << "\nright_pose:\n"
             << right_goal_pose_.matrix().format(fmt) << "\n";

      throw std::runtime_error(err_ss.str());
    }

    copy_robot_state_to_ompl_state(group_, *robot_state_ik, s);
  }

  [[nodiscard]] unsigned int maxSampleCount() const override { return 25u; }
};

class StrictRightGripperOrientationStateSampler : public ob::StateSampler {
 public:
  moveit::core::RobotModelConstPtr const model_;
  moveit::core::JointModelGroup const *group_;
  moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
  Eigen::Quaterniond target_orientation_;

  StrictRightGripperOrientationStateSampler(const ob::StateSpace *space, moveit::core::RobotModelConstPtr const &model,
                                            moveit::core::JointModelGroup const *group,
                                            moveit_visual_tools::MoveItVisualToolsPtr visual_tools,
                                            geometry_msgs::Quaternion const &orientation)
      : ob::StateSampler(space), model_(model), group_(group), visual_tools_(visual_tools) {
    tf::quaternionMsgToEigen(orientation, target_orientation_);
  }

  void sampleGaussian(ob::State *s, const ob::State *, double) override { sampleUniform(s); }

  void sampleUniformNear(ob::State *s, const ob::State *, double) override { sampleUniform(s); }

  void sampleUniform(ob::State *s) override {
    // sample a joint configuration which obeys the orientation constraint.
    bio_ik::BioIKKinematicsQueryOptions opts;
    opts.replace = true;
    opts.return_approximate_solution = false;

    auto const quat_tf2 = eigenQuaternionTotf2Quaternion(target_orientation_);
    opts.goals.emplace_back(std::make_unique<bio_ik::OrientationGoal>("right_tool", quat_tf2));

    auto const rng_dist = 4;
    auto const robot_state_ik = ik(model_, group_, opts, rng_dist);
    if (!robot_state_ik) {
      std::stringstream err_ss;
      err_ss << "Failed to solve ik:\n"
             << "target_orientation:\n"
             << target_orientation_.matrix() << "\n";
      throw std::runtime_error(err_ss.str());
    }

     visual_tools_->publishRobotState(*robot_state_ik);
    copy_robot_state_to_ompl_state(group_, *robot_state_ik, s);
  }
};

RopeResetPlanner::RopeResetPlanner(std::string const &group_name)
    : model_loader_(std::make_shared<robot_model_loader::RobotModelLoader>("hdt_michigan/robot_description")),
      model_(model_loader_->getModel()),
      scene_monitor_(std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(model_loader_)),
      group_name_(group_name),
      group_(model_->getJointModelGroup(group_name_)),
      n_joints_(group_->getActiveVariableCount()),
      joint_names_(group_->getActiveJointModelNames()),
      space_(std::make_shared<ArmStateSpace>(joint_names_)),
      ss_(space_),
      si_(ss_.getSpaceInformation()) {
  visual_tools_ = std::make_shared<moveit_visual_tools::MoveItVisualTools>(
      "robot_root", "hdt_michigan/moveit_visual_markers", model_);
  auto const scene_topic = "hdt_michigan/move_group/monitored_planning_scene";
  scene_monitor_->startSceneMonitor(scene_topic);
  auto const service_name = "hdt_michigan/get_planning_scene";
  scene_monitor_->requestPlanningSceneState(service_name);

  ob::RealVectorBounds position_bounds(n_joints_);
  ob::RealVectorBounds velocity_bounds(n_joints_);
  ob::RealVectorBounds acceleration_bounds(n_joints_);
  auto const bounds = group_->getActiveJointModelsBounds();
  auto joint_i{0u};
  for (auto const *joint_bounds : bounds) {
    auto const &joint_name = group_->getActiveJointModelNames()[joint_i];
    if (joint_bounds->size() != 1) {
      std::stringstream ss;
      ss << "Joint " << joint_name << " has " << joint_bounds->size() << " bounds\n";
      throw std::runtime_error(ss.str());
    } else {
      auto const &bound = (*joint_bounds)[0];
      position_bounds.setLow(joint_i, bound.min_position_);
      position_bounds.setHigh(joint_i, bound.max_position_);
      velocity_bounds.setLow(joint_i, bound.min_velocity_);
      velocity_bounds.setHigh(joint_i, bound.max_velocity_);
      acceleration_bounds.setLow(joint_i, bound.min_acceleration_);
      acceleration_bounds.setHigh(joint_i, bound.max_acceleration_);
      ++joint_i;
    }
  }
  space_->setBounds(position_bounds);

  auto const &planner = std::make_shared<og::RRTConnect>(si_);
  planner->setRange(0.2);
  ss_.setPlanner(planner);
}

PlanningResult RopeResetPlanner::planWithConstraints(planning_scene::PlanningScenePtr const &planning_scene,
                                                     ob::GoalPtr const &goal,
                                                     ob::StateValidityCheckerFn const &state_validity_fn,
                                                     double timeout) {
  robot_trajectory::RobotTrajectory traj(model_, group_name_);

  auto start_robot_state = planning_scene->getCurrentState();

  addLinkPadding(planning_scene);
  // for safety, we add collision spheres around the "tool" points and pad all the links
  addGripperCollisionSpheres(start_robot_state);

  ob::ScopedState<ArmStateSpace> start(space_);
  for (auto const &joint_name : joint_names_) {
    auto const i = space_->getJointIndex(joint_name);
    start[i] = start_robot_state.getVariablePosition(joint_name);
  }

  std::vector<double> zeros(n_joints_, 0);

  ss_.clearStartStates();
  ss_.clear();
  ss_.setStateValidityChecker(state_validity_fn);
  ss_.setStartState(start);
  ss_.setGoal(goal);
  ob::PlannerStatus status = ss_.solve(timeout);

  if (status) {
    auto &original_path = ss_.getSolutionPath();
    auto path = simplify(original_path, goal);

    ob::PlannerData pd(si_);
    ss_.getPlannerData(pd);
    for (auto const &kv : pd.properties) {
      std::cout << kv.first << ": " << kv.second << std::endl;
    }
    std::cout << "Solution has " << path.getStateCount() << " states\n";
    for (auto const &state : path.getStates()) {
      auto const robot_state = omplStateToRobotState(state, model_, space_);
      traj.addSuffixWayPoint(robot_state, 0);
    }

    if (!time_param_.computeTimeStamps(traj, 0.75, 0.75)) {
      ROS_ERROR_STREAM_NAMED(LOGGER_NAME, "Time parametrization for the solution path failed.");
    }
  } else {
    std::cout << "No solution found" << std::endl;
  }

  moveit_msgs::RobotTrajectory traj_msg;
  traj.getRobotTrajectoryMsg(traj_msg);
  traj_msg.joint_trajectory.header.stamp = ros::Time::now();
  return {status.asString(), traj_msg};
}

og::PathGeometric RopeResetPlanner::simplify(og::PathGeometric original_path, ob::GoalPtr goal) {
  // ss_.simplifySolution();
  //  NOTE: do simplification ourselves, so we can check the success result, since by default OMPL allows slightly
  //  invalid solutions
  while (true) {
    auto path = original_path;  // make a copy!
    std::size_t numStates = original_path.getStateCount();
    og::PathSimplifier psk(si_, goal);
    auto const valid = psk.simplifyMax(path);
    if (!valid) {
      ROS_INFO_STREAM_NAMED(LOGGER_NAME, "Path simplification failed!");
    } else {
      ROS_INFO_STREAM_NAMED(LOGGER_NAME,
                            "Path simplification: " << numStates << " to " << path.getStateCount() << " states");
      return path;
    }
  }
}

PlanningResult RopeResetPlanner::planToReset(geometry_msgs::Pose const &left_pose,
                                             geometry_msgs::Pose const &right_pose, double orientation_path_tolerance,
                                             double orientation_goal_tolerance, double timeout, bool debug_collisions) {
  scene_monitor_->lockSceneRead();
  auto planning_scene = planning_scene::PlanningScene::clone(scene_monitor_->getPlanningScene());
  scene_monitor_->unlockSceneRead();

  auto goal = std::make_shared<PosesGoal>(model_, group_, visual_tools_, si_, left_pose, right_pose, 0.01,
                                          orientation_goal_tolerance);

  auto state_validity_fn = [&](const ob::State *s) {
    auto const robot_state = omplStateToRobotState(s, model_, space_);
    // visual_tools_->publishRobotState(robot_state);
    // usleep(1'000'000);

    const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");

    Eigen::Isometry3d right_target_pose;
    tf::poseMsgToEigen(right_pose, right_target_pose);
    auto const right_orientation_error = rotMatDist(right_target_pose.rotation(), right_tool_pose.rotation());
    auto const right_orientation_satisfied = right_orientation_error < orientation_path_tolerance;

    auto const collision_free = [&]() {
      if (debug_collisions) {
        return planning_scene->isStateValid(robot_state, "", true);
      } else {
        return planning_scene->isStateValid(robot_state);
      }
    }();
    // visual_tools_->publishRobotState(robot_state);

    auto const joint56_ok = abs(*robot_state.getJointPositions("joint56")) < 1.2;

    ROS_DEBUG_STREAM_NAMED(LOGGER_NAME + ".isStateValid", ""
                                                              << "satisfies bounds? " << si_->satisfiesBounds(s) << "\n"
                                                              << "orientation? " << right_orientation_satisfied << "\n"
                                                              << "joint56_ok? " << joint56_ok << "\n"
                                                              << "collision free? " << collision_free << "\n");

    return si_->satisfiesBounds(s) && right_orientation_satisfied && collision_free && joint56_ok;
  };

  auto state_sampler_allocator = [&](const ob::StateSpace *space) {
    return std::make_shared<StrictRightGripperOrientationStateSampler>(space, model_, group_, visual_tools_,
                                                                       right_pose.orientation);
  };
  space_->setStateSamplerAllocator(state_sampler_allocator);

  return planWithConstraints(planning_scene, goal, state_validity_fn, timeout);
}

PlanningResult RopeResetPlanner::planToStart(geometry_msgs::Pose const &left_pose,
                                             geometry_msgs::Pose const &right_pose, double max_gripper_dist,
                                             double orientation_path_tolerance, double orientation_goal_tolerance,
                                             double timeout, bool debug_collisions) {
  scene_monitor_->lockSceneRead();
  auto planning_scene = planning_scene::PlanningScene::clone(scene_monitor_->getPlanningScene());
  scene_monitor_->unlockSceneRead();

  auto start_robot_state = planning_scene->getCurrentState();
  const auto &start_left_tool_pose = start_robot_state.getGlobalLinkTransform("left_tool");
  Eigen::Isometry3d left_target_pose;
  tf::poseMsgToEigen(left_pose, left_target_pose);
  auto const initial_left_orientation_error = rotMatDist(left_target_pose.rotation(), start_left_tool_pose.rotation());

  auto goal = std::make_shared<PosesGoal>(model_, group_, visual_tools_, si_, left_pose, right_pose, 0.01,
                                          orientation_goal_tolerance);

  auto state_validity_fn =
      [&](const ob::State *s) {
        auto const robot_state = omplStateToRobotState(s, model_, space_);

        const auto &left_tool_pose = robot_state.getGlobalLinkTransform("left_tool");
        auto const left_orientation_error = rotMatDist(left_target_pose.rotation(), left_tool_pose.rotation());
        auto const left_orientation_satisfied = left_orientation_error < initial_left_orientation_error + 0.5;

        const auto &right_tool_pose = robot_state.getGlobalLinkTransform("right_tool");
        Eigen::Isometry3d right_target_pose;
        tf::poseMsgToEigen(right_pose, right_target_pose);
        auto const right_orientation_error = rotMatDist(right_target_pose.rotation(), right_tool_pose.rotation());
        auto const right_orientation_satisfied = right_orientation_error < orientation_path_tolerance;

        auto const collision_free = [&]() {
          if (debug_collisions) {
            return planning_scene->isStateValid(robot_state, "", true);
          } else {
            return planning_scene->isStateValid(robot_state);
          }
        }();

        auto const grippers_dist = (right_tool_pose.translation() - left_tool_pose.translation()).norm();
        auto const grippers_close = grippers_dist < max_gripper_dist;

        auto const joint56_ok = abs(*robot_state.getJointPositions("joint56")) < 1;
        auto const joint57_ok = *robot_state.getJointPositions("joint57") < 0.15;

        ROS_DEBUG_STREAM_THROTTLE_NAMED(1, LOGGER_NAME + ".isStateValid",
                                        ""
                                            << "satisfies bounds? " << si_->satisfiesBounds(s) << "\n"
                                            << "left orientation? " << left_orientation_satisfied << "\n"
                                            << "right orientation? " << right_orientation_satisfied << "\n"
                                            << "collision free? " << collision_free << "\n"
                                            << "joint56 ok? " << joint56_ok << "\n"
                                            << "joint57 ok? " << joint57_ok << "\n"
                                            << "grippers close? " << grippers_close);
        return si_->satisfiesBounds(s) && left_orientation_satisfied && right_orientation_satisfied && collision_free &&
               grippers_close && joint56_ok && joint57_ok;
      };

  auto state_sampler_allocator = [&](const ob::StateSpace *space) {
    return std::make_shared<StrictRightGripperOrientationStateSampler>(space, model_, group_, visual_tools_,
                                                                       right_pose.orientation);
  };
  space_->setStateSamplerAllocator(state_sampler_allocator);

  return planWithConstraints(planning_scene, goal, state_validity_fn, timeout);
}
