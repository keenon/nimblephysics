#include <Eigen/Dense>
#include <dart/math/RelativeFilter.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void RelativeFilter(py::module& m)
{
  ::py::class_<
      dart::math::RelativeFilter,
      std::shared_ptr<dart::math::RelativeFilter>>(m, "RelativeFilter")
      .def(
          ::py::init<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>(),
          ::py::arg("acc_std") = Eigen::Vector3d::Constant(0.05),
          ::py::arg("gyro_std") = Eigen::Vector3d::Constant(0.05),
          ::py::arg("mag_std") = Eigen::Vector3d::Constant(0.05))
      .def(
          "get_q_pc",
          &dart::math::RelativeFilter::get_q_pc,
          "Get the quaternion representing the relative rotation between "
          "parent and child.")
      .def(
          "get_R_pc",
          &dart::math::RelativeFilter::get_R_pc,
          "Get the rotation matrix representing the relative rotation between "
          "parent and child.")
      .def(
          "update",
          &dart::math::RelativeFilter::update,
          ::py::arg("gyro_p"),
          ::py::arg("gyro_c"),
          ::py::arg("acc_jc_p"),
          ::py::arg("acc_jc_c"),
          ::py::arg("mag_p"),
          ::py::arg("mag_c"),
          ::py::arg("dt"),
          "Update the filter with new sensor readings and timestep.")
      .def(
          "set_qs",
          &dart::math::RelativeFilter::set_qs,
          ::py::arg("q_wp"),
          ::py::arg("q_wc"),
          "Set the quaternions for parent and child.")
      .def(
          "get_h",
          [](const dart::math::RelativeFilter& self,
             const Eigen::Matrix3d& R_wp,
             const Eigen::Matrix3d& R_wc,
             const Eigen::Vector3d& acc_jc_p,
             const Eigen::Vector3d& acc_jc_c,
             const Eigen::Vector3d& mag_jc_p,
             const Eigen::Vector3d& mag_jc_c,
             const Eigen::Vector6d& perturbation = Eigen::Vector6d::Zero()) {
            return self.get_h(
                R_wp,
                R_wc,
                acc_jc_p,
                acc_jc_c,
                mag_jc_p,
                mag_jc_c,
                perturbation);
          },
          ::py::arg("R_wp"),
          ::py::arg("R_wc"),
          ::py::arg("acc_jc_p"),
          ::py::arg("acc_jc_c"),
          ::py::arg("mag_jc_p"),
          ::py::arg("mag_jc_c"),
          ::py::arg("perturbation") = Eigen::Vector6d::Zero(),
          "Compute the measurement function h with optional perturbations.")
      .def(
          "get_H_jacobian",
          [](const dart::math::RelativeFilter& self,
             const Eigen::Matrix3d& R_wp,
             const Eigen::Matrix3d& R_wc,
             const Eigen::Vector3d& acc_jc_p,
             const Eigen::Vector3d& acc_jc_c,
             const Eigen::Vector3d& mag_jc_p,
             const Eigen::Vector3d& mag_jc_c) {
            return self.get_H_jacobian(
                R_wp, R_wc, acc_jc_p, acc_jc_c, mag_jc_p, mag_jc_c);
          },
          ::py::arg("R_wp"),
          ::py::arg("R_wc"),
          ::py::arg("acc_jc_p"),
          ::py::arg("acc_jc_c"),
          ::py::arg("mag_jc_p"),
          ::py::arg("mag_jc_c"),
          "Compute the Jacobian of the measurement function h.")
      .def(
          "get_M_jacobian",
          &dart::math::RelativeFilter::get_M_jacobian,
          ::py::arg("R_wp"),
          ::py::arg("R_wc"),
          ::py::arg("update") = Eigen::Vector6d::Zero(),
          "Compute the Jacobian of the measurement function for sensor noise.")
      .def_readonly(
          "Q",
          &dart::math::RelativeFilter::Q,
          "Covariance matrix for gyro sensor noise.")
      .def_readonly(
          "R",
          &dart::math::RelativeFilter::R,
          "Covariance matrix for accelerometer and magnetometer sensor noise.")
      .def_static(
          "skew_symmetric",
          &dart::math::RelativeFilter::skew_symmetric,
          ::py::arg("v"),
          "Compute the skew-symmetric matrix for a given 3D vector.");
}

} // namespace python
} // namespace dart
