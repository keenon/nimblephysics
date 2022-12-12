#include "dart/biomechanics/SubjectOnDisk.hpp"

#include <memory>

#include <dart/biomechanics/OpenSimParser.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/MeshShape.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dart/math/MathTypes.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void SubjectOnDisk(py::module& m)
{
  ::py::class_<
      dart::biomechanics::SubjectOnDisk,
      std::shared_ptr<dart::biomechanics::SubjectOnDisk>>(m, "SubjectOnDisk")
      .def(
          ::py::init<std::string, bool>(),
          ::py::arg("path"),
          ::py::arg("printDebuggingDetails") = false)
      .def(
          "readSkel",
          &dart::biomechanics::SubjectOnDisk::readSkel,
          ::py::arg("geometryFolder") = "")
      .def(
          "readFrames",
          &dart::biomechanics::SubjectOnDisk::readFrames,
          ::py::arg("trial"),
          ::py::arg("startFrame"),
          ::py::arg("numFramesToRead") = 1)
      .def_static(
          "writeSubject",
          &dart::biomechanics::SubjectOnDisk::writeSubject,
          ::py::arg("outputPath"),
          // The OpenSim file XML gets copied into our binary bundle, along with
          // any necessary Geometry files
          ::py::arg("openSimFilePath"),
          // The per-trial motion data
          ::py::arg("trialTimesteps"),
          ::py::arg("trialPoses"),
          ::py::arg("trialVels"),
          ::py::arg("trialAccs"),
          ::py::arg("probablyMissingGRF"),
          ::py::arg("trialTaus"),
          // These are generalized 6-dof wrenches applied to arbitrary bodies
          // (generally by foot-ground contact, though other things too)
          ::py::arg("groundForceBodies"),
          ::py::arg("trialGroundBodyWrenches"),
          ::py::arg("trialGroundBodyCopTorqueForce"),
          // We include this to allow the binary format to store/load a bunch of
          // new types of values while remaining backwards compatible.
          ::py::arg("customValueNames"),
          ::py::arg("customValues"),
          // The provenance info, optional, for investigating where training
          // data came from after its been aggregated
          ::py::arg("trialNames") = std::vector<std::string>(),
          ::py::arg("sourceHref") = "",
          ::py::arg("notes") = "")
      .def("getNumDofs", &dart::biomechanics::SubjectOnDisk::getNumDofs)
      .def("getNumTrials", &dart::biomechanics::SubjectOnDisk::getNumTrials)
      .def(
          "getTrialLength",
          &dart::biomechanics::SubjectOnDisk::getTrialLength,
          ::py::arg("trial"))
      .def(
          "getProbablyMissingGRF",
          &dart::biomechanics::SubjectOnDisk::getProbablyMissingGRF,
          ::py::arg("trial"))
      .def(
          "getTrialName",
          &dart::biomechanics::SubjectOnDisk::getTrialName,
          ::py::arg("trial"))
      .def("getHref", &dart::biomechanics::SubjectOnDisk::getHref)
      .def("getNotes", &dart::biomechanics::SubjectOnDisk::getNotes)
      .def(
          "getContactBodies",
          &dart::biomechanics::SubjectOnDisk::getGroundContactBodies)
      .def(
          "getCustomValues",
          &dart::biomechanics::SubjectOnDisk::getCustomValues)
      .def(
          "getCustomValueDim",
          &dart::biomechanics::SubjectOnDisk::getCustomValueDim,
          ::py::arg("customValue"));

  ::py::class_<
      dart::biomechanics::Frame,
      std::shared_ptr<dart::biomechanics::Frame>>(m, "Frame")
      .def_readwrite("trial", &dart::biomechanics::Frame::trial)
      .def_readwrite("t", &dart::biomechanics::Frame::t)
      .def_readwrite(
          "probablyMissingGRF", &dart::biomechanics::Frame::probablyMissingGRF)
      .def_readwrite("pos", &dart::biomechanics::Frame::pos)
      .def_readwrite("vel", &dart::biomechanics::Frame::vel)
      .def_readwrite("acc", &dart::biomechanics::Frame::acc)
      .def_readwrite("tau", &dart::biomechanics::Frame::tau)
      .def_readwrite(
          "groundContactWrenches",
          &dart::biomechanics::Frame::groundContactWrenches)
      .def_readwrite(
          "groundContactCenterOfPressure",
          &dart::biomechanics::Frame::groundContactCenterOfPressure)
      .def_readwrite(
          "groundContactTorque",
          &dart::biomechanics::Frame::groundContactTorque)
      .def_readwrite(
          "groundContactForce", &dart::biomechanics::Frame::groundContactForce)
      .def_readwrite("dt", &dart::biomechanics::Frame::dt)
      .def_readwrite("customValues", &dart::biomechanics::Frame::customValues);
}

} // namespace python
} // namespace dart