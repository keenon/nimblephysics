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
      .def_static(
          "createRawTraces",
          &dart::biomechanics::MarkerTrace::createRawTraces,
          ::py::arg("pointClouds"),
          ::py::arg("mergeDistance") = 0.01,
          ::py::arg("mergeFrames") = 5)
      .def(
          "appendPoint",
          &dart::biomechanics::MarkerTrace::appendPoint,
          ::py::arg("time"),
          ::py::arg("point"),
          "Add a point to the end of the marker trace")
      .def(
          "pointToAppendDistance",
          &dart::biomechanics::MarkerTrace::pointToAppendDistance,
          ::py::arg("time"),
          ::py::arg("point"),
          ::py::arg("extrapolate"),
          "This gives the distance from the last point (or an extrapolation at "
          "this timestep of the last point, of order up to 2)")
      .def(
          "computeBodyMarkerStats",
          &dart::biomechanics::MarkerTrace::computeBodyMarkerStats,
          ::py::arg("skel"),
          ::py::arg("posesOverTime"),
          ::py::arg("scalesOverTime"),
          "Each possible combination of (trace, body) can create a marker. So "
          "we can compute some summary statistics for each body we could "
          "assign this trace to.")
      .def(
          "computeBodyMarkerLoss",
          &dart::biomechanics::MarkerTrace::computeBodyMarkerLoss,
          ::py::arg("bodyName"),
          "Each possible combination of (trace, body) can create a marker. "
          "This returns a score for a given body, for how \"good\" of a marker "
          "that body would create when combined with this trace. Lower is "
          "better.")
      .def(
          "getBestMarker",
          &dart::biomechanics::MarkerTrace::getBestMarker,
          "This finds the best body to pair this trace with (using the stats "
          "from computeBodyMarkerStats()) and returns the best marker")
      .def(
          "overlap",
          &dart::biomechanics::MarkerTrace::overlap,
          ::py::arg("toAppend"),
          "Returns true if these traces overlap in time")
      .def(
          "concat",
          &dart::biomechanics::MarkerTrace::concat,
          ::py::arg("toAppend"),
          "This merges two MarkerTrace's together, to create a new trace "
          "object")
      .def(
          "firstTimestep",
          &dart::biomechanics::MarkerTrace::firstTimestep,
          "This returns when this MarkerTrace begins (inclusive)")
      .def(
          "lastTimestep",
          &dart::biomechanics::MarkerTrace::lastTimestep,
          "This returns when this MarkerTrace ends (inclusive)")
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
