#include <Eigen/Dense>
#include <dart/biomechanics/MarkerLabeller.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/cast.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MarkerLabeller(py::module& m)
{
  (void)m;
  ::py::class_<dart::biomechanics::MarkerTrace>(m, "MarkerTrace")
      .def_readonly("minTime", &dart::biomechanics::MarkerTrace::mMinTime)
      .def_readonly("maxTime", &dart::biomechanics::MarkerTrace::mMaxTime)
      .def_readonly("times", &dart::biomechanics::MarkerTrace::mTimes)
      .def_readonly("points", &dart::biomechanics::MarkerTrace::mPoints)
      .def_readonly(
          "markerLabel", &dart::biomechanics::MarkerTrace::mMarkerLabel)
      .def_readonly(
          "bodyMarkerOffsets",
          &dart::biomechanics::MarkerTrace::mBodyMarkerOffsets)
      .def_readonly(
          "bodyMarkerOffsetVariance",
          &dart::biomechanics::MarkerTrace::mBodyMarkerOffsetVariance)
      .def_readonly(
          "bodyRootJointDistVariance",
          &dart::biomechanics::MarkerTrace::mBodyRootJointDistVariance)
      .def_readonly(
          "bodyClosestPointDistance",
          &dart::biomechanics::MarkerTrace::mBodyClosestPointDistance);

  ::py::class_<dart::biomechanics::LabelledMarkers>(m, "LabelledMarkers")
      .def_readwrite(
          "markerObservations",
          &dart::biomechanics::LabelledMarkers::markerObservations)
      .def_readwrite(
          "markerOffsets", &dart::biomechanics::LabelledMarkers::markerOffsets)
      .def_readwrite(
          "jointCenterGuesses",
          &dart::biomechanics::LabelledMarkers::jointCenterGuesses)
      .def_readwrite("traces", &dart::biomechanics::LabelledMarkers::traces);

  ::py::class_<dart::biomechanics::MarkerLabeller>(m, "MarkerLabeller")
      .def(
          "guessJointLocations",
          &dart::biomechanics::MarkerLabeller::guessJointLocations,
          ::py::arg("pointClouds"))
      .def(
          "setSkeleton",
          &dart::biomechanics::MarkerLabeller::setSkeleton,
          ::py::arg("skeleton"))
      .def(
          "labelPointClouds",
          &dart::biomechanics::MarkerLabeller::labelPointClouds,
          ::py::arg("pointClouds"),
          ::py::arg("mergeMarkersThreshold") = 0.01)
      .def(
          "matchUpJointToSkeletonJoint",
          &dart::biomechanics::MarkerLabeller::matchUpJointToSkeletonJoint,
          ::py::arg("jointName"),
          ::py::arg("skeletonJointName"))
      .def(
          "evaluate",
          &dart::biomechanics::MarkerLabeller::evaluate,
          ::py::arg("markerOffsets"),
          ::py::arg("labeledPointClouds"));

  ::py::class_<
      dart::biomechanics::MarkerLabellerMock,
      dart::biomechanics::MarkerLabeller>(m, "MarkerLabellerMock")
      .def(::py::init<>())
      .def(
          "setMockJointLocations",
          &dart::biomechanics::MarkerLabellerMock::setMockJointLocations,
          ::py::arg("jointsOverTime"));

  ::py::class_<
      dart::biomechanics::NeuralMarkerLabeller,
      dart::biomechanics::MarkerLabeller>(m, "NeuralMarkerLabeller")
      .def(
          ::py::init<
              std::function<std::vector<std::map<std::string, Eigen::Vector3s>>(
                  const std::vector<std::vector<Eigen::Vector3s>>&)>>(),
          ::py::arg("jointCenterPredictor"));
}

} // namespace python
} // namespace dart
