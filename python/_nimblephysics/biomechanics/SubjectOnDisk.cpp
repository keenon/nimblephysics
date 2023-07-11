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
                "residual",
                &dart::biomechanics::Frame::residual,
                "The norm of the root residual force on this trial.")
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
                py::return_value_policy::reference_internal,
                "The joint positions on this frame.")
            .def_readwrite(
                "vel",
                &dart::biomechanics::Frame::vel,
                py::return_value_policy::reference_internal,
                "The joint velocities on this frame.")
            .def_readwrite(
                "acc",
                &dart::biomechanics::Frame::acc,
                py::return_value_policy::reference_internal,
                "The joint accelerations on this frame.")
            .def_readwrite(
                "tau",
                &dart::biomechanics::Frame::tau,
                py::return_value_policy::reference_internal,
                "The joint control forces on this frame.")
            .def_readwrite(
                "contact",
                &dart::biomechanics::Frame::contact,
                py::return_value_policy::reference_internal,
                "A vector of [0,1] booleans for if a body is in contact with "
                "the ground.")
            .def_readwrite(
                "comPos",
                &dart::biomechanics::Frame::comPos,
                py::return_value_policy::reference_internal,
                "The position of the COM, in world space")
            .def_readwrite(
                "comVel",
                &dart::biomechanics::Frame::comVel,
                py::return_value_policy::reference_internal,
                "The velocity of the COM, in world space")
            .def_readwrite(
                "comAcc",
                &dart::biomechanics::Frame::comAcc,
                py::return_value_policy::reference_internal,
                "The acceleration of the COM, in world space")
            .def_readwrite(
                "posObserved",
                &dart::biomechanics::Frame::posObserved,
                py::return_value_policy::reference_internal,
                "A boolean mask of [0,1]s for each DOF, with a 1 indicating "
                "that this DOF was observed on this frame")
            .def_readwrite(
                "velFiniteDifferenced",
                &dart::biomechanics::Frame::velFiniteDifferenced,
                py::return_value_policy::reference_internal,
                "A boolean mask of [0,1]s for each DOF, with a 1 indicating "
                "that this DOF got its velocity through finite differencing, "
                "and therefore may be somewhat unreliable")
            .def_readwrite(
                "accFiniteDifferenced",
                &dart::biomechanics::Frame::accFiniteDifferenced,
                py::return_value_policy::reference_internal,
                "A boolean mask of [0,1]s for each DOF, with a 1 indicating "
                "that this DOF got its acceleration through finite "
                "differencing, and therefore may be somewhat unreliable")
            .def_readwrite(
                "groundContactWrenches",
                &dart::biomechanics::Frame::groundContactWrenches,
                py::return_value_policy::reference_internal,
                R"doc(
This is a vector of concatenated contact body wrenches :code:`body_wrench`, where :code:`body_wrench` is a 6 vector (first 3 are torque, last 3 are force). 
:code:`body_wrench` is expressed in the local frame of the body at :code:`body_name`, and assumes that the skeleton is set to positions `pos`.

Here's an example usage
.. code-block::
    for i, bodyName in enumerate(subject.getContactBodies()):
        body: nimble.dynamics.BodyNode = skel.getBodyNode(bodyName)
        torque_local = wrench[i*6:i*6+3]
        force_local = wrench[i*6+3:i*6+6]
        # For example, to rotate the force to the world frame
        R_wb = body.getWorldTransform().rotation()
        force_world = R_wb @ force_local

Note that these are specified in the local body frame, acting on the body at its origin, so transforming them to the world frame requires a transformation!
)doc")
            .def_readwrite(
                "groundContactCenterOfPressure",
                &dart::biomechanics::Frame::groundContactCenterOfPressure,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`CoP` values for each contact body, where :code:`CoP` is a 3 vector representing the center of pressure for a contact measured on the force plate. :code:`CoP` is 
            expressed in the world frame.
        )doc")
            .def_readwrite(
                "groundContactTorque",
                &dart::biomechanics::Frame::groundContactTorque,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`tau` values for each contact body, where :code:`tau` is a 3 vector representing the ground-reaction torque from a contact, measured on the force plate. :code:`tau` is 
            expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            .def_readwrite(
                "groundContactForce",
                &dart::biomechanics::Frame::groundContactForce,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`f` values for each contact body, where :code:`f` is a 3 vector representing the ground-reaction force from a contact, measured on the force plate. :code:`f` is 
            expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            .def_readwrite(
                "markerObservations",
                &dart::biomechanics::Frame::markerObservations,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[str, np.ndarray]` of the marker "
                "observations at this frame. Markers that were not observed "
                "will not be present in this list. For the full specification "
                "of the markerset, load the model from the "
                ":code:`SubjectOnDisk`")
            .def_readwrite(
                "accObservations",
                &dart::biomechanics::Frame::accObservations,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[str, np.ndarray]` of the "
                "accelerometers observations at this frame. Accelerometers "
                "that were not observed (perhaps due to time offsets in "
                "uploaded data) will not be present in this list. For the full "
                "specification of the accelerometer set, load the model from "
                "the :code:`SubjectOnDisk`")
            .def_readwrite(
                "gyroObservations",
                &dart::biomechanics::Frame::gyroObservations,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[str, np.ndarray]` of the "
                "gyroscope observations at this frame. Gyroscopes that were "
                "not observed (perhaps due to time offsets in uploaded data) "
                "will not be present in this list. For the full specification "
                "of the gyroscope set, load the model from the "
                ":code:`SubjectOnDisk`")
            .def_readwrite(
                "rawForcePlateCenterOfPressures",
                &dart::biomechanics::Frame::rawForcePlateCenterOfPressures,
                py::return_value_policy::reference_internal,
                "This is list of :code:`np.ndarray` of the original center of "
                "pressure readings on each force plate, without any processing "
                "by AddBiomechanics. These are the original inputs that were "
                "used to create this SubjectOnDisk.")
            .def_readwrite(
                "rawForcePlateTorques",
                &dart::biomechanics::Frame::rawForcePlateTorques,
                py::return_value_policy::reference_internal,
                "This is list of :code:`np.ndarray` of the original torque "
                "readings on each force plate, without any processing "
                "by AddBiomechanics. These are the original inputs that were "
                "used to create this SubjectOnDisk.")
            .def_readwrite(
                "rawForcePlateForces",
                &dart::biomechanics::Frame::rawForcePlateForces,
                py::return_value_policy::reference_internal,
                "This is list of :code:`np.ndarray` of the original force "
                "readings on each force plate, without any processing "
                "by AddBiomechanics. These are the original inputs that were "
                "used to create this SubjectOnDisk.")
            .def_readwrite(
                "emgSignals",
                &dart::biomechanics::Frame::emgSignals,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[str, np.ndarray]` of the "
                "EMG signals at this frame. EMG signals are generally "
                "preserved at a higher sampling frequency than the motion "
                "capture, so the `np.ndarray` vector will be a number of "
                "samples that were captured during this single motion capture "
                "frame. For example, if EMG is at 1000Hz and mocap is at "
                "100Hz, the `np.ndarray` vector will be of length 10.")
            .def_readwrite(
                "customValues",
                &dart::biomechanics::Frame::customValues,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[str, np.ndarray]` of unspecified "
                "values. The idea here is to allow the format to be easily "
                "extensible with unusual data (for example, exoskeleton "
                "torques) "
                "without bloating ordinary training files.");
  frame.doc() = R"doc(
        This is for doing ML and large-scale data analysis. This is a single frame of data, returned in a list by :code:`SubjectOnDisk.readFrames()`, which contains everything needed to reconstruct all the dynamics of a snapshot in time.
      )doc";

  auto subjectOnDisk
      = ::py::class_<
            dart::biomechanics::SubjectOnDisk,
            std::shared_ptr<dart::biomechanics::SubjectOnDisk>>(
            m, "SubjectOnDisk")
            .def(::py::init<std::string>(), ::py::arg("path"))
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
                "readRawOsimFileText",
                &dart::biomechanics::SubjectOnDisk::readRawOsimFileText,
                "This will read the raw OpenSim file XML out of the "
                "SubjectOnDisk, and return it as a string.")
            .def(
                "readFrames",
                &dart::biomechanics::SubjectOnDisk::readFrames,
                ::py::arg("trial"),
                ::py::arg("startFrame"),
                ::py::arg("numFramesToRead") = 1,
                ::py::arg("stride") = 1,
                ::py::arg("contactThreshold") = 1.0,
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
                ::py::arg("dofPositionsObserved"),
                ::py::arg("dofVelocitiesFiniteDifferenced"),
                ::py::arg("dofAccelerationsFiniteDifferenced"),
                ::py::arg("trialTaus"),
                ::py::arg("trialComPoses"),
                ::py::arg("trialComVels"),
                ::py::arg("trialComAccs"),
                ::py::arg("trialResidualNorms"),
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
                // These are the markers, gyros and accelerometers, and EMGs
                ::py::arg("markerObservations"),
                ::py::arg("accObservations"),
                ::py::arg("gyroObservations"),
                ::py::arg("emgObservations"),
                // The raw original force plate data
                ::py::arg("forcePlates"),
                // This is the subject info
                ::py::arg("biologicalSex"),
                ::py::arg("heightM"),
                ::py::arg("massKg"),
                ::py::arg("ageYears"),
                // The provenance info, optional, for investigating where
                // training data came from after its been aggregated
                ::py::arg("trialNames") = std::vector<std::string>(),
                ::py::arg("subjectTags") = std::vector<std::string>(),
                ::py::arg("trialTags")
                = std::vector<std::vector<std::string>>(),
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
                "getTrialTimestep",
                &dart::biomechanics::SubjectOnDisk::getTrialTimestep,
                ::py::arg("trial"),
                "This returns the timestep size for the trial requested, in "
                "seconds per frame")
            .def(
                "getDofPositionsObserved",
                &dart::biomechanics::SubjectOnDisk::getDofPositionsObserved,
                ::py::arg("trial"),
                "This returns the vector of booleans indicating which DOFs "
                "have their positions observed during this trial")
            .def(
                "getDofVelocitiesFiniteDifferenced",
                &dart::biomechanics::SubjectOnDisk::
                    getDofVelocitiesFiniteDifferenced,
                ::py::arg("trial"),
                "This returns the vector of booleans indicating which DOFs "
                "have their velocities from finite-differencing during this "
                "trial (as opposed to observed directly through a gyroscope or "
                "IMU)")
            .def(
                "getDofAccelerationsFiniteDifferenced",
                &dart::biomechanics::SubjectOnDisk::
                    getDofAccelerationsFiniteDifferenced,
                ::py::arg("trial"),
                "This returns the vector of booleans indicating which DOFs "
                "have their accelerations from finite-differencing during this "
                "trial (as opposed to observed directly through a "
                "accelerometer or IMU)")
            .def(
                "getTrialResidualNorms",
                &dart::biomechanics::SubjectOnDisk::getTrialResidualNorms,
                ::py::arg("trial"),
                "This returns the vector of scalars indicating the norm of the "
                "root residual forces + torques on each timestep of a given "
                "trial")
            .def(
                "getTrialMaxJointVelocity",
                &dart::biomechanics::SubjectOnDisk::getTrialMaxJointVelocity,
                ::py::arg("trial"),
                "This returns the vector of scalars indicating the maximum "
                "absolute velocity of all DOFs on each timestep of a given "
                "trial")
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
                "getBiologicalSex",
                &dart::biomechanics::SubjectOnDisk::getBiologicalSex,
                "This returns a string, one of \"male\", \"female\", or "
                "\"unknown\".")
            .def(
                "getHeightM",
                &dart::biomechanics::SubjectOnDisk::getHeightM,
                "This returns the height in meters, or 0.0 if unknown.")
            .def(
                "getMassKg",
                &dart::biomechanics::SubjectOnDisk::getMassKg,
                "This returns the mass in kilograms, or 0.0 if unknown.")
            .def(
                "getAgeYears",
                &dart::biomechanics::SubjectOnDisk::getAgeYears,
                "This returns the age of the subject, or 0 if unknown.")
            .def(
                "getSubjectTags",
                &dart::biomechanics::SubjectOnDisk::getSubjectTags,
                "This returns the list of tags attached to this subject, which "
                "are arbitrary strings from the AddBiomechanics platform.")
            .def(
                "getTrialTags",
                &dart::biomechanics::SubjectOnDisk::getTrialTags,
                ::py::arg("trial"),
                "This returns the list of tags attached to a given trial "
                "index, which are arbitrary strings from the AddBiomechanics "
                "platform.")
            .def(
                "getNumForcePlates",
                &dart::biomechanics::SubjectOnDisk::getNumForcePlates,
                ::py::arg("trial"),
                "The number of force plates in the source data.")
            .def(
                "getForcePlateCorners",
                &dart::biomechanics::SubjectOnDisk::getForcePlateCorners,
                ::py::arg("trial"),
                ::py::arg("forcePlate"),
                "Get an array of force plate corners (as 3D vectors) for the "
                "given force plate in the given trial. Empty array on "
                "out-of-bounds access.")
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