cmake_minimum_required(VERSION 3.0.2)
project(arm_moveit)

set(CMAKE_CXX_STANDARD 17)
set(CATKIN_DEPENDS
        jsk_recognition_msgs
        moveit_commander
        moveit_core
        moveit_visual_tools
        pybind11_catkin
        roscpp
        eigen_conversions
        bio_ik
        tf2_eigen
        )

find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPENDS})
find_package(ompl REQUIRED)

catkin_python_setup()

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES arm_moveit robot_points_generator rope_reset_planner
        CATKIN_DEPENDS ${CATKIN_DEPENDS}
)

set(FLAGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type -Werror=missing-field-initializers -Wno-unknown-pragmas)

add_library(robot_points_generator src/robot_points_generator.cpp)
target_include_directories(robot_points_generator PUBLIC include)
target_include_directories(robot_points_generator SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(robot_points_generator PUBLIC ${catkin_LIBRARIES})
target_compile_options(robot_points_generator PUBLIC ${FLAGS})

add_executable(motion_planning_tutorial src/motion_planning_tutorial.cpp)
target_link_libraries(motion_planning_tutorial PUBLIC ${catkin_LIBRARIES} robot_points_generator)

add_executable(robot_points_generator_node src/robot_points_generator_node.cpp)
target_link_libraries(robot_points_generator_node PUBLIC ${catkin_LIBRARIES} robot_points_generator)

pybind_add_module(pyrobot_points_generator src/bindings.cpp)
target_link_libraries(pyrobot_points_generator PUBLIC robot_points_generator)

add_library(rope_reset_planner src/rope_reset_planner.cpp)
target_include_directories(rope_reset_planner PUBLIC include)
target_include_directories(rope_reset_planner SYSTEM PUBLIC ${catkin_INCLUDE_DIRS} ${OMPL_INCLUDE_DIRS})
target_link_libraries(rope_reset_planner PUBLIC ${catkin_LIBRARIES} ${OMPL_LIBRARIES})
target_compile_options(rope_reset_planner PUBLIC ${FLAGS})

pybind_add_module(pyrope_reset_planner src/rope_reset_planner_bindings.cpp)
target_link_libraries(pyrope_reset_planner PUBLIC rope_reset_planner)

add_executable(test_rope_reset_planner src/test_rope_reset_planner.cpp)
target_link_libraries(test_rope_reset_planner PUBLIC rope_reset_planner)
