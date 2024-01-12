#include <Eigen/Dense>
#include <dart/math/PolynomialFitter.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void PolynomialFitter(py::module& m)
{
  ::py::class_<
      dart::math::PolynomialFitter,
      std::shared_ptr<dart::math::PolynomialFitter>>(m, "PolynomialFitter")
      .def(
          ::py::init<Eigen::VectorXs, int>(),
          ::py::arg("timesteps"),
          ::py::arg("order"))
      .def(
          "calcCoeffs",
          &dart::math::PolynomialFitter::calcCoeffs,
          ::py::arg("values"))
      .def(
          "projectPosVelAccAtTime",
          &dart::math::PolynomialFitter::projectPosVelAccAtTime,
          ::py::arg("timestep"),
          ::py::arg("pastValues"));
}

} // namespace python
} // namespace dart
