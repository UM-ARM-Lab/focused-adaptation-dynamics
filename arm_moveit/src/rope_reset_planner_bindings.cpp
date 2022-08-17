#include <arm_moveit/rope_reset_planner.h>
#include <moveit/python/pybind_rosmsg_typecasters.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pyrope_reset_planner, m) {
  py::class_<PlanningResult>(m, "PlanningResult")
      .def(py::init<>())
      .def_readwrite("traj", &PlanningResult::traj)
      .def_readwrite("status", &PlanningResult::status)
      //
      ;
  py::class_<RopeResetPlanner>(m, "RopeResetPlanner")
      .def(py::init<>())
      .def("plan_to_reset", &RopeResetPlanner::planToReset, py::arg("left_pose"), py::arg("right_pose"),
           py::arg("orientation_path_tolerance"), py::arg("orientation_goal_tolerance"), py::arg("timeout"))
      .def("plan_to_start", &RopeResetPlanner::planToStart, py::arg("left_pose"), py::arg("right_pose"),
           py::arg("max_gripper_distance"), py::arg("orientation_path_tolerance"), py::arg("orientation_goal_tolerance"), py::arg("timeout"))
      //
      ;
}
