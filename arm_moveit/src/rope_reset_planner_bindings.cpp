#include <arm_moveit/rope_reset_planner.h>
#include <moveit/python/pybind_rosmsg_typecasters.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(pyrope_reset_planner, m) {
  py::class_<RopeResetPlanner>(m, "RopeResetPlanner")
      .def(py::init<>())
      .def("plan_to_reset", &RopeResetPlanner::planToReset, py::arg("left_pose"), py::arg("right_pose"), py::arg("timeout"))
      //
      ;
}
