#include <arm_moveit/rope_reset_planner.h>
#include <geometry_msgs/Pose.h>
#include <ros/ros.h>

#include <chrono>

int main(int argc, char** argv) {
  ros::init(argc, argv, "rope_reset_planner");

  ros::NodeHandle nh("test_rope_reset_planner");

  auto model_loader = std::make_shared<robot_model_loader::RobotModelLoader>("hdt_michigan/robot_description");
  moveit::core::RobotModelConstPtr model(model_loader->getModel());
  auto scene_monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(model_loader);

  scene_monitor->lockSceneRead();
  auto planning_scene = planning_scene::PlanningScene::clone(scene_monitor->getPlanningScene());
  scene_monitor->unlockSceneRead();

  auto robot_state = planning_scene->getCurrentState();

  collision_detection::CollisionRequest collisionRequest;
  collisionRequest.contacts = true;
  collisionRequest.max_contacts = 1;
  collisionRequest.max_contacts_per_pair = 1;
  collision_detection::CollisionResult collisionResult;
  planning_scene->checkCollision(collisionRequest, collisionResult, robot_state);


  auto test = [&]() {
    for (auto i{0}; i < 100; ++i) {

      // planning_scene->isStateValid(robot_state);
      planning_scene->checkCollision(collisionRequest, collisionResult, robot_state);

    }
  };

  test();
  auto const t0 = std::chrono::steady_clock::now();
  addLinkPadding(planning_scene);
  std::chrono::duration<double, std::milli> const dt = std::chrono::steady_clock::now() - t0;
  std::cout << dt.count() << "ms" << std::endl;
  std::cout << "---------\n";
  test();
  return 0;
}