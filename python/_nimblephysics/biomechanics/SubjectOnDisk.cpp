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
  auto frame
      = ::py::class_<
            dart::biomechanics::Frame,
            std::shared_ptr<dart::biomechanics::Frame>>(m, "Frame")
            .def_readwrite(
                "trial",
                &dart::biomechanics::Frame::trial,
                "The index of the trial in the containing SubjectOnDisk.")
            .def_readwrite(
                "t",
                &dart::biomechanics::Frame::t,
                "The frame number in this trial.")
            .def_readwrite(
                "probablyMissingGRF",
                &dart::biomechanics::Frame::probablyMissingGRF,
                R"doc(
            This is true if this frame probably has unmeasured forces acting on the body. For example, if a subject
            steps off of the available force plates during this frame, this will probably be true.

            WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on this frame!!
          )doc")
            .def_readwrite(
                "pos",
                &dart::biomechanics::Frame::pos,
                "The joint positions on this frame.")
            .def_readwrite(
                "vel",
                &dart::biomechanics::Frame::vel,
                "The joint velocities on this frame.")
            .def_readwrite(
                "acc",
                &dart::biomechanics::Frame::acc,
                "The joint accelerations on this frame.")
            .def_readwrite(
                "tau",
                &dart::biomechanics::Frame::tau,
                "The joint control forces on this frame.")
            .def_readwrite(
                "groundContactWrenches",
                &dart::biomechanics::Frame::groundContactWrenches,
                R"doc(
This is a list of pairs of (:code:`body_name`, :code:`body_wrench`), where :code:`body_wrench` is a 6 vector (first 3 are torque, last 3 are force). 
:code:`body_wrench` is expressed in the local frame of the body at :code:`body_name`, and assumes that the skeleton is set to positions `pos`.

Here's an example usage
.. code-block::
    for name, wrench in frame.groundContactWrenches:
        body: nimble.dynamics.BodyNode = skel.getBodyNode(name)
        torque_local = wrench[:3]
        force_local = wrench[3:]
        # For example, to rotate the force to the world frame
        R_wb = body.getWorldTransform().rotation()
        force_world = R_wb @ force_local

Note that these are specified in the local body frame, acting on the body at its origin, so transforming them to the world frame requires a transformation!
)doc")
            .def_readwrite(
                "groundContactCenterOfPressure",
                &dart::biomechanics::Frame::groundContactCenterOfPressure,
                R"doc(
            This is a list of pairs of (:code:`body name`, :code:`CoP`), where :code:`CoP` is a 3 vector representing the center of pressure for a contact measured on the force plate. :code:`CoP` is 
            expressed in the world frame.
        )doc")
            .def_readwrite(
                "groundContactTorque",
                &dart::biomechanics::Frame::groundContactTorque,
                R"doc(
            This is a list of pairs of (:code:`body name`, :code:`tau`), where :code:`tau` is a 3 vector representing the ground-reaction torque from a contact, measured on the force plate. :code:`tau` is 
            expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            .def_readwrite(
                "groundContactForce",
                &dart::biomechanics::Frame::groundContactForce,
                R"doc(
            This is a list of pairs of (:code:`body name`, :code:`f`), where :code:`f` is a 3 vector representing the ground-reaction force from a contact, measured on the force plate. :code:`f` is 
            expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            .def_readwrite(
                "dt",
                &dart::biomechanics::Frame::dt,
                "This is the size of the simulation timestep at this frame.")
            .def_readwrite(
                "customValues",
                &dart::biomechanics::Frame::customValues,
                "This is list of :code:`Pair[str, np.ndarray]` of unspecified "
                "values. The idea here is to allow the format to be easily "
                "extensible with unusual data (for example, exoskeleton "
                "torques) "
                "without bloating ordinary training files.");
  frame.doc() = R"doc(
        This is for doing ML and large-scale data analysis. This is a single frame of data, returned in a list by :code:`SubjectOnDisk.readFrames()`, which contains everything needed to reconstruct all the dynamics of a snapshot in time.
      )doc";

  py::enum_<dart::biomechanics::MissingGRFReason>(m, "MissingGRFReason")
      .value(
          "notMissingGRF", dart::biomechanics::MissingGRFReason::notMissingGRF)
      .value(
          "measuredGrfZeroWhenAccelerationNonZero",
          dart::biomechanics::MissingGRFReason::
              measuredGrfZeroWhenAccelerationNonZero)
      .value(
          "unmeasuredExternalForceDetected",
          dart::biomechanics::MissingGRFReason::unmeasuredExternalForceDetected)
      .value(
          "torqueDiscrepancy",
          dart::biomechanics::MissingGRFReason::torqueDiscrepancy)
      .value(
          "forceDiscrepancy",
          dart::biomechanics::MissingGRFReason::forceDiscrepancy)
      .value(
          "notOverForcePlate",
          dart::biomechanics::MissingGRFReason::notOverForcePlate)
      .value(
          "missingImpact", dart::biomechanics::MissingGRFReason::missingImpact)
      .value("missingBlip", dart::biomechanics::MissingGRFReason::missingBlip)
      .value("shiftGRF", dart::biomechanics::MissingGRFReason::shiftGRF)
      .export_values();

  auto subjectOnDisk
      = ::py::class_<
            dart::biomechanics::SubjectOnDisk,
            std::shared_ptr<dart::biomechanics::SubjectOnDisk>>(
            m, "SubjectOnDisk")
            .def(
                ::py::init<std::string, bool>(),
                ::py::arg("path"),
                ::py::arg("printDebuggingDetails") = false)
            .def(
                "readSkel",
                &dart::biomechanics::SubjectOnDisk::readSkel,
                ::py::arg("geometryFolder") = "",
                "This will read the skeleton from the binary, and optionally "
                "use the passed in :code:`geometryFolder` to load meshes. We "
                "do not bundle meshes with :code:`SubjectOnDisk` files, to "
                "save space. If you do not pass in :code:`geometryFolder`, "
                "expect to get warnings about being unable to load meshes, and "
                "expect that your skeleton will not display if you attempt to "
                "visualize it.")
            .def(
                "readFrames",
                &dart::biomechanics::SubjectOnDisk::readFrames,
                ::py::arg("trial"),
                ::py::arg("startFrame"),
                ::py::arg("numFramesToRead") = 1,
                "This will read from disk and allocate a number of "
                ":code:`Frame` "
                "objects. These Frame objects are assumed to be short-lived, "
                "to save working memory. For example, you might "
                ":code:`readFrames()` to construct a training batch, then "
                "immediately allow the frames to go out of scope and be "
                "released after the batch backpropagates gradient and loss."
                " On OOB access, prints an error and returns an empty vector.")
            .def_static(
                "writeSubject",
                &dart::biomechanics::SubjectOnDisk::writeSubject,
                ::py::arg("outputPath"),
                // The OpenSim file XML gets copied into our binary bundle,
                // along with any necessary Geometry files
                ::py::arg("openSimFilePath"),
                // The per-trial motion data
                ::py::arg("trialTimesteps"),
                ::py::arg("trialPoses"),
                ::py::arg("trialVels"),
                ::py::arg("trialAccs"),
                ::py::arg("probablyMissingGRF"),
                ::py::arg("missingGRFReason"),
                ::py::arg("trialTaus"),
                // These are generalized 6-dof wrenches applied to arbitrary
                // bodies (generally by foot-ground contact, though other things
                // too)
                ::py::arg("groundForceBodies"),
                ::py::arg("trialGroundBodyWrenches"),
                ::py::arg("trialGroundBodyCopTorqueForce"),
                // We include this to allow the binary format to store/load a
                // bunch of new types of values while remaining backwards
                // compatible.
                ::py::arg("customValueNames"),
                ::py::arg("customValues"),
                // The provenance info, optional, for investigating where
                // training data came from after its been aggregated
                ::py::arg("trialNames") = std::vector<std::string>(),
                ::py::arg("sourceHref") = "",
                ::py::arg("notes") = "",
                "This writes a subject out to disk in a compressed and "
                "random-seekable binary format.")
            .def(
                "getNumDofs",
                &dart::biomechanics::SubjectOnDisk::getNumDofs,
                "This returns the number of DOFs for the model on this Subject")
            .def(
                "getNumTrials",
                &dart::biomechanics::SubjectOnDisk::getNumTrials,
                "This returns the number of trials that are in this file.")
            .def(
                "getTrialLength",
                &dart::biomechanics::SubjectOnDisk::getTrialLength,
                ::py::arg("trial"),
                "This returns the length of the trial requested")
            .def(
                "getProbablyMissingGRF",
                &dart::biomechanics::SubjectOnDisk::getProbablyMissingGRF,
                ::py::arg("trial"),
                R"doc(
            This returns an array of boolean values, one per frame in the specified trial.
            Each frame is :code:`True` if this frame probably has unmeasured forces acting on the body. For example, if a subject
            steps off of the available force plates during this frame, this will probably be true.

            WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on the corresponding frame!!

            This method is provided to give a cheaper way to filter out frames we want to ignore for training, without having to call
            the more expensive :code:`loadFrames()`
          )doc")
            .def(
                "getMissingGRFReason",
                &dart::biomechanics::SubjectOnDisk::getMissingGRFReason,
                ::py::arg("trial"),
                R"doc(
            This returns an array of enum values, one per frame in the specified trial,
            each corresponding to the reason why a frame was marked as `probablyMissingGRF`.
          )doc")
            .def(
                "getTrialName",
                &dart::biomechanics::SubjectOnDisk::getTrialName,
                ::py::arg("trial"),
                "This returns the human readable name of the specified trial, "
                "given by the person who uploaded the data to AddBiomechanics. "
                "This isn't necessary for training, but may be useful for "
                "analyzing the data.")
            .def(
                "getHref",
                &dart::biomechanics::SubjectOnDisk::getHref,
                "The AddBiomechanics link for this subject's data.")
            .def(
                "getNotes",
                &dart::biomechanics::SubjectOnDisk::getNotes,
                "The notes (if any) added by the person who uploaded this data "
                "to AddBiomechanics.")
            .def(
                "getContactBodies",
                &dart::biomechanics::SubjectOnDisk::getGroundContactBodies,
                "A list of the :code:`body_name`'s for each body that was "
                "assumed to be able to take ground-reaction-force from force "
                "plates.")
            .def(
                "getCustomValues",
                &dart::biomechanics::SubjectOnDisk::getCustomValues,
                "A list of all the different types of custom values that this "
                "SubjectOnDisk contains. These are unspecified, and are "
                "intended to allow an easy extension of the format to unusual "
                "types of data (like exoskeleton torques or unusual physical "
                "sensors) that may be present on some subjects but not others.")
            .def(
                "getCustomValueDim",
                &dart::biomechanics::SubjectOnDisk::getCustomValueDim,
                ::py::arg("valueName"),
                "This returns the dimension of the custom value specified by "
                ":code:`valueName`");
  subjectOnDisk.doc() = R"doc(
        This is for doing ML and large-scale data analysis. The idea here is to
        create a lazy-loadable view of a subject, where everything remains on disk
        until asked for. That way we can instantiate thousands of these in memory,
        and not worry about OOM'ing a machine.
      )doc";
}
} // namespace python
} // namespace dart