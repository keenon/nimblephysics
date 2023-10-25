#include "dart/biomechanics/SubjectOnDisk.hpp"

#include <memory>

#include <dart/biomechanics/OpenSimParser.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/MeshShape.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <dart/simulation/World.hpp>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dart/math/MathTypes.hpp"

namespace py = pybind11;

namespace dart {
namespace python {

void SubjectOnDisk(py::module& m)
{
  auto processingPassType
      = ::py::enum_<dart::biomechanics::ProcessingPassType>(
            m, "ProcessingPassType")
            .value(
                "KINEMATICS",
                dart::biomechanics::ProcessingPassType::kinematics,
                "This is the pass where we solve for kinematics.")
            .value(
                "DYNAMICS",
                dart::biomechanics::ProcessingPassType::dynamics,
                "This is the pass where we solve for dynamics.")
            .value(
                "LOW_PASS_FILTER",
                dart::biomechanics::ProcessingPassType::lowPassFilter,
                "This is the pass where we apply a low-pass filter to the "
                "kinematics and dynamics.");

  auto framePass
      = ::py::class_<dart::biomechanics::FramePass>(m, "FramePass")
            //   ProcessingPassType type;
            .def_readwrite(
                "type",
                &dart::biomechanics::FramePass::type,
                "The type of processing pass that this data came from. Options "
                "include KINEMATICS (for movement only), DYNAMICS (for "
                "movement and physics), and LOW_PASS_FILTER (to apply a simple "
                "Butterworth to the observed data from the previous pass).")
            //   s_t markerRMS;
            .def_readwrite(
                "markerRMS",
                &dart::biomechanics::FramePass::markerRMS,
                "A scalar indicating the RMS marker error (discrepancy between "
                "the model and the experimentally observed marker locations) "
                "on this frame, in meters, with these joint positions.")
            //   s_t markerMax;
            .def_readwrite(
                "markerMax",
                &dart::biomechanics::FramePass::markerMax,
                "A scalar indicating the maximum marker error (discrepancy "
                "between the model and the experimentally observed marker "
                "locations) on this frame, in meters, with these joint "
                "positions.")
            //   s_t linearResidual;
            .def_readwrite(
                "linearResidual",
                &dart::biomechanics::FramePass::linearResidual,
                "A scalar giving how much linear force, in Newtons, would need "
                "to be applied at the root of the skeleton in order to enable "
                "the skeleton's observed accelerations (given positions and "
                "velocities) on this frame.")
            //   s_t angularResidual;
            .def_readwrite(
                "angularResidual",
                &dart::biomechanics::FramePass::angularResidual,
                "A scalar giving how much angular torque, in Newton-meters, "
                "would need "
                "to be applied at the root of the skeleton in order to enable "
                "the skeleton's observed accelerations (given positions and "
                "velocities) on this frame.")
            //   Eigen::VectorXd pos;
            .def_readwrite(
                "pos",
                &dart::biomechanics::FramePass::pos,
                py::return_value_policy::reference_internal,
                "The joint positions on this frame.")
            //   Eigen::VectorXd vel;
            .def_readwrite(
                "vel",
                &dart::biomechanics::FramePass::vel,
                py::return_value_policy::reference_internal,
                "The joint velocities on this frame.")
            //   Eigen::VectorXd acc;
            .def_readwrite(
                "acc",
                &dart::biomechanics::FramePass::acc,
                py::return_value_policy::reference_internal,
                "The joint accelerations on this frame.")
            //   Eigen::VectorXd tau;
            .def_readwrite(
                "tau",
                &dart::biomechanics::FramePass::tau,
                py::return_value_policy::reference_internal,
                "The joint control forces on this frame.")
            //   // These are boolean values (0 or 1) for each contact body
            //   indicating whether
            //   // or not it's in contact
            //   Eigen::VectorXi contact;
            .def_readwrite(
                "contact",
                &dart::biomechanics::FramePass::contact,
                py::return_value_policy::reference_internal,
                "A vector of [0,1] booleans for if a body is in contact with "
                "the ground.")
            //   // These are each 6-vector of contact body wrenches, all
            //   concatenated together Eigen::VectorXd groundContactWrenches;
            .def_readwrite(
                "groundContactWrenches",
                &dart::biomechanics::FramePass::groundContactWrenches,
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
            //   // These are each 3-vector for each contact body, concatenated
            //   together Eigen::VectorXd groundContactCenterOfPressure;
            .def_readwrite(
                "groundContactCenterOfPressure",
                &dart::biomechanics::FramePass::groundContactCenterOfPressure,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`CoP` values for each contact body, where :code:`CoP` is a 3 vector representing the center of pressure for a contact measured on the force plate. :code:`CoP` is 
            expressed in the world frame.
        )doc")
            //   Eigen::VectorXd groundContactTorque;
            .def_readwrite(
                "groundContactTorque",
                &dart::biomechanics::FramePass::groundContactTorque,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`tau` values for each contact body, where :code:`tau` is a 3 vector representing the ground-reaction torque from a contact, measured on the force plate. :code:`tau` is 
            expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            //   Eigen::VectorXd groundContactForce;
            .def_readwrite(
                "groundContactForce",
                &dart::biomechanics::FramePass::groundContactForce,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`f` values for each contact body, where :code:`f` is a 3 vector representing the ground-reaction force from a contact, measured on the force plate. :code:`f` is 
            expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            //   // These are each 3-vector for each contact body, concatenated
            //   together Eigen::VectorXd groundContactCenterOfPressure;
            .def_readwrite(
                "groundContactCenterOfPressureInRootFrame",
                &dart::biomechanics::FramePass::
                    groundContactCenterOfPressureInRootFrame,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`CoP` values for each contact body, where :code:`CoP` is a 3 vector representing the center of pressure for a contact measured on the force plate. :code:`CoP` is 
            expressed in the root frame, which is a frame that is rigidly attached to the root body of the skeleton (probably the pelvis).
        )doc")
            //   Eigen::VectorXd groundContactTorque;
            .def_readwrite(
                "groundContactTorqueInRootFrame",
                &dart::biomechanics::FramePass::groundContactTorqueInRootFrame,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`tau` values for each contact body, where :code:`tau` is a 3 vector representing the ground-reaction torque from a contact, measured on the force plate. :code:`tau` is 
            expressed in the root frame, which is a frame that is rigidly attached to the root body of the skeleton (probably the pelvis), and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            //   Eigen::VectorXd groundContactForce;
            .def_readwrite(
                "groundContactForceInRootFrame",
                &dart::biomechanics::FramePass::groundContactForceInRootFrame,
                py::return_value_policy::reference_internal,
                R"doc(
            This is a vector of all the concatenated :code:`f` values for each contact body, where :code:`f` is a 3 vector representing the ground-reaction force from a contact, measured on the force plate. :code:`f` is 
            expressed in the root frame, which is a frame that is rigidly attached to the root body of the skeleton (probably the pelvis), and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
          )doc")
            //   // These are the center of mass kinematics
            //   Eigen::Vector3s comPos;
            .def_readwrite(
                "comPos",
                &dart::biomechanics::FramePass::comPos,
                py::return_value_policy::reference_internal,
                "The position of the COM, in world space")
            //   Eigen::Vector3s comVel;
            .def_readwrite(
                "comVel",
                &dart::biomechanics::FramePass::comVel,
                py::return_value_policy::reference_internal,
                "The velocity of the COM, in world space")
            //   Eigen::Vector3s comAcc;
            .def_readwrite(
                "comAcc",
                &dart::biomechanics::FramePass::comAcc,
                py::return_value_policy::reference_internal,
                "The acceleration of the COM, in world space")
            //   // These are masks for which DOFs are observed
            //   Eigen::VectorXi posObserved;
            .def_readwrite(
                "posObserved",
                &dart::biomechanics::FramePass::posObserved,
                py::return_value_policy::reference_internal,
                "A boolean mask of [0,1]s for each DOF, with a 1 indicating "
                "that this DOF was observed on this frame")
            //   // These are masks for which DOFs have been finite differenced
            //   (if they
            //   // haven't been finite differenced, they're from real sensors
            //   and therefore
            //   // more trustworthy)
            //   Eigen::VectorXi velFiniteDifferenced;
            .def_readwrite(
                "velFiniteDifferenced",
                &dart::biomechanics::FramePass::velFiniteDifferenced,
                py::return_value_policy::reference_internal,
                "A boolean mask of [0,1]s for each DOF, with a 1 indicating "
                "that this DOF got its velocity through finite differencing, "
                "and therefore may be somewhat unreliable")
            //   Eigen::VectorXi accFiniteDifferenced;
            .def_readwrite(
                "accFiniteDifferenced",
                &dart::biomechanics::FramePass::accFiniteDifferenced,
                py::return_value_policy::reference_internal,
                "A boolean mask of [0,1]s for each DOF, with a 1 indicating "
                "that this DOF got its acceleration through finite "
                "differencing, and therefore may be somewhat unreliable")
            //   Eigen::Vector3s rootLinearVelInRootFrame;
            .def_readwrite(
                "rootLinearVelInRootFrame",
                &dart::biomechanics::FramePass::rootLinearVelInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the linear velocity, in meters per second, of the "
                "root body of the skeleton (probably the pelvis) expressed in "
                "its own coordinate frame.")
            //   Eigen::Vector3s rootAngularVelInRootFrame;
            .def_readwrite(
                "rootAngularVelInRootFrame",
                &dart::biomechanics::FramePass::rootAngularVelInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the angular velocity, in an angle-axis representation "
                "where the norm of this 3-vector is given in radians per "
                "second, of the root body of the skeleton (probably the "
                "pelvis) expressed in its own coordinate frame.")
            //   Eigen::Vector3s rootLinearAccInRootFrame;
            .def_readwrite(
                "rootLinearAccInRootFrame",
                &dart::biomechanics::FramePass::rootLinearAccInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the linear acceleration, in meters per second "
                "squared, of the "
                "root body of the skeleton (probably the pelvis) expressed in "
                "its own coordinate frame.")
            //   Eigen::Vector3s rootAngularAccInRootFrame;
            .def_readwrite(
                "rootAngularAccInRootFrame",
                &dart::biomechanics::FramePass::rootAngularAccInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the angular velocity, in an angle-axis representation "
                "where the norm of this 3-vector is given in radians per "
                "second squared, of the root body of the skeleton (probably "
                "the pelvis) expressed in its own coordinate frame.")
            //   Eigen::VectorXs rootPosHistoryInRootFrame;
            .def_readwrite(
                "rootPosHistoryInRootFrame",
                &dart::biomechanics::FramePass::rootPosHistoryInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the recent history of the positions of the root body"
                " of the skeleton (probably the pelvis) expressed in "
                "its own coordinate frame. These are concatenated 3-vectors. "
                "The [0:3] of the vector is the most recent, and they get "
                "older from there. Vectors  ")
            //   Eigen::VectorXs rootEulerHistoryInRootFrame;
            .def_readwrite(
                "rootEulerHistoryInRootFrame",
                &dart::biomechanics::FramePass::rootEulerHistoryInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the recent history of the angles (expressed as euler "
                "angles) of the root body"
                " of the skeleton (probably the pelvis) expressed in "
                "its own coordinate frame.")
            // Eigen::Vector3s comAccInRootFrame;
            .def_readwrite(
                "comAccInRootFrame",
                &dart::biomechanics::FramePass::comAccInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the acceleration of the center of mass of the "
                "subject, expressed in the root body frame (which probably "
                "means expressed in pelvis coordinates, though some skeletons "
                "may use a different body as the root, for instance the "
                "torso).")
            // Eigen::VectorXd groundContactWrenchesInRootFrame;
            .def_readwrite(
                "groundContactWrenchesInRootFrame",
                &dart::biomechanics::FramePass::
                    groundContactWrenchesInRootFrame,
                py::return_value_policy::reference_internal,
                "These are the wrenches (each vectors of length 6, composed of "
                "first 3 = "
                "torque, last 3 = force) expressed in the root body frame, and "
                "concatenated together. The "
                "root body is probably the pelvis, but for some skeletons they "
                "may use another body as the root, like the torso.")
            // Eigen::VectorXd groundContactWrenchesInRootFrame;
            .def_readwrite(
                "residualWrenchInRootFrame",
                &dart::biomechanics::FramePass::residualWrenchInRootFrame,
                py::return_value_policy::reference_internal,
                "This is the 'residual' force wrench (or 'modelling error' "
                "force, the "
                "force necessary to make Newton's laws match up with our "
                "model, even though it's imaginary) expressed in the root body "
                "frame. This is a wrench (vector of length 6, composed of "
                "first 3 = "
                "torque, last 3 = force). The "
                "root body is probably the pelvis, but for some skeletons they "
                "may use another body as the root, like the torso.")
            .def_readwrite(
                "jointCenters",
                &dart::biomechanics::FramePass::jointCenters,
                py::return_value_policy::reference_internal,
                "These are the joint center locations, concatenated together, "
                "given in the world frame.")
            .def_readwrite(
                "jointCentersInRootFrame",
                &dart::biomechanics::FramePass::jointCentersInRootFrame,
                py::return_value_policy::reference_internal,
                "These are the joint center locations, concatenated together, "
                "given in the root frame. The "
                "root body is probably the pelvis, but for some skeletons they "
                "may use another body as the root, like the torso.");

  framePass.doc() = R"doc(
        This is for doing ML and large-scale data analysis. This is a single processing pass on a single frame of data, returned from a list within a :code:`nimblephysics.biomechanics.Frame` (which can be got with :code:`SubjectOnDisk.readFrames()`), which contains the full reconstruction of your subject at this instant created by this processing pass. Earlier processing passes are likely to have more discrepancies with the original data, bet later processing passes require more types of sensor signals that may not always be available.
      )doc";

  auto frame
      = ::py::class_<
            dart::biomechanics::Frame,
            std::shared_ptr<dart::biomechanics::Frame>>(m, "Frame")
            // int trial;
            .def_readwrite(
                "trial",
                &dart::biomechanics::Frame::trial,
                "The index of the trial in the containing SubjectOnDisk.")
            // int t;
            .def_readwrite(
                "t",
                &dart::biomechanics::Frame::t,
                "The frame number in this trial.")
            // bool probablyMissingGRF;
            // MissingGRFReason missingGRFReason;
            .def_readwrite(
                "missingGRFReason",
                &dart::biomechanics::Frame::missingGRFReason,
                R"doc(
            This is the reason that this frame is missing GRF, or else is the flag notMissingGRF to indicate that this frame has physics.

            WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on this frame!!
          )doc")
            // // Each processing pass has its own set of kinematics and
            // dynamics, as the
            // // model and trajectories are adjusted
            // std::vector<FramePass> processingPasses;
            .def_readwrite(
                "processingPasses",
                &dart::biomechanics::Frame::processingPasses,
                R"doc(
            The processing passes that were done on this Frame. For example, if we solved for kinematics, then dynamics, 
            then low pass filtered, this will have 3 entries.
                )doc")
            // // We include this to allow the binary format to store/load a
            // bunch of new
            // // types of values while remaining backwards compatible.
            // std::vector<std::pair<std::string, Eigen::VectorXd>>
            // customValues;
            .def_readwrite(
                "customValues",
                &dart::biomechanics::Frame::customValues,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[str, np.ndarray]` of unspecified "
                "values. The idea here is to allow the format to be easily "
                "extensible with unusual data (for example, exoskeleton "
                "torques) "
                "without bloating ordinary training files.")
            // // These are the marker observations
            // std::vector<std::pair<std::string, Eigen::Vector3s>>
            // markerObservations;
            .def_readwrite(
                "markerObservations",
                &dart::biomechanics::Frame::markerObservations,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[str, np.ndarray]` of the marker "
                "observations at this frame. Markers that were not observed "
                "will not be present in this list. For the full specification "
                "of the markerset, load the model from the "
                ":code:`SubjectOnDisk`")
            // // These are the accelerometer observations
            // std::vector<std::pair<std::string, Eigen::Vector3s>>
            // accObservations;
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
            // // These are the gyroscope observations
            // std::vector<std::pair<std::string, Eigen::Vector3s>>
            // gyroObservations;
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
            // // These are the raw values recorded from the force plates,
            // without any
            // // post-processing or aggregation applied
            // std::vector<Eigen::Vector3s> rawForcePlateCenterOfPressures;
            .def_readwrite(
                "rawForcePlateCenterOfPressures",
                &dart::biomechanics::Frame::rawForcePlateCenterOfPressures,
                py::return_value_policy::reference_internal,
                "This is list of :code:`np.ndarray` of the original center of "
                "pressure readings on each force plate, without any processing "
                "by AddBiomechanics. These are the original inputs that were "
                "used to create this SubjectOnDisk.")
            // std::vector<Eigen::Vector3s> rawForcePlateTorques;
            .def_readwrite(
                "rawForcePlateTorques",
                &dart::biomechanics::Frame::rawForcePlateTorques,
                py::return_value_policy::reference_internal,
                "This is list of :code:`np.ndarray` of the original torque "
                "readings on each force plate, without any processing "
                "by AddBiomechanics. These are the original inputs that were "
                "used to create this SubjectOnDisk.")
            // std::vector<Eigen::Vector3s> rawForcePlateForces;
            .def_readwrite(
                "rawForcePlateForces",
                &dart::biomechanics::Frame::rawForcePlateForces,
                py::return_value_policy::reference_internal,
                "This is list of :code:`np.ndarray` of the original force "
                "readings on each force plate, without any processing "
                "by AddBiomechanics. These are the original inputs that were "
                "used to create this SubjectOnDisk.")
            // // These are the EMG signals, where the signal is represented as
            // a vector that
            // // can vary in length depending on how much faster the EMG
            // sampling was than
            // // the motion capture sampling.
            // std::vector<std::pair<std::string, Eigen::VectorXs>> emgSignals;
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
            // // These are the torques from the exo, along with the index of
            // the DOFs
            // // they are applied to
            // std::vector<std::pair<int, s_t>> exoTorques;
            .def_readwrite(
                "exoTorques",
                &dart::biomechanics::Frame::exoTorques,
                py::return_value_policy::reference_internal,
                "This is list of :code:`Pair[int, np.ndarray]` of the "
                "DOF indices that are actuated by exoskeletons, and the "
                "torques on those DOFs.");
  frame.doc() = R"doc(
        This is for doing ML and large-scale data analysis. This is a single frame of data, returned in a list by :code:`SubjectOnDisk.readFrames()`, which contains everything needed to reconstruct all the dynamics of a snapshot in time.
      )doc";

  auto subjectOnDiskTrialPass
      = ::py::class_<
            dart::biomechanics::SubjectOnDiskTrialPass,
            std::shared_ptr<dart::biomechanics::SubjectOnDiskTrialPass>>(
            m, "SubjectOnDiskTrialPass")
            .def(::py::init<>())
            .def(
                "copyValuesFrom",
                &dart::biomechanics::SubjectOnDiskTrialPass::copyValuesFrom,
                ::py::arg("other"))
            .def(
                "setType",
                &dart::biomechanics::SubjectOnDiskTrialPass::setType,
                ::py::arg("type"))
            .def(
                "setDofPositionsObserved",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setDofPositionsObserved,
                ::py::arg("dofPositionsObserved"))
            .def(
                "setDofVelocitiesFiniteDifferenced",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setDofVelocitiesFiniteDifferenced,
                ::py::arg("dofVelocitiesFiniteDifferenced"))
            .def(
                "setDofAccelerationFiniteDifferenced",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setDofAccelerationFiniteDifferenced,
                ::py::arg("dofAccelerationFiniteDifference"))
            .def(
                "setMarkerRMS",
                &dart::biomechanics::SubjectOnDiskTrialPass::setMarkerRMS,
                ::py::arg("markerRMS"))
            .def(
                "getMarkerRMS",
                &dart::biomechanics::SubjectOnDiskTrialPass::getMarkerRMS)
            .def(
                "setMarkerMax",
                &dart::biomechanics::SubjectOnDiskTrialPass::setMarkerMax,
                ::py::arg("markerMax"))
            .def(
                "getMarkerMax",
                &dart::biomechanics::SubjectOnDiskTrialPass::getMarkerMax)
            .def(
                "setLowpassCutoffFrequency",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setLowpassCutoffFrequency,
                ::py::arg("freq"))
            .def(
                "setLowpassFilterOrder",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setLowpassFilterOrder,
                ::py::arg("order"))
            .def(
                "setForcePlateCutoffs",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setForcePlateCutoffs,
                ::py::arg("cutoffs"))
            .def(
                "computeValues",
                &dart::biomechanics::SubjectOnDiskTrialPass::computeValues,
                ::py::arg("skel"),
                ::py::arg("timestep"),
                ::py::arg("poses"),
                ::py::arg("footBodyNames"),
                ::py::arg("forces"),
                ::py::arg("moments"),
                ::py::arg("cops"),
                ::py::arg("rootHistoryLen") = 5,
                ::py::arg("rootHistoryStride") = 1)
            .def(
                "computeValuesFromForcePlates",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    computeValuesFromForcePlates,
                ::py::arg("skel"),
                ::py::arg("timestep"),
                ::py::arg("poses"),
                ::py::arg("footBodyNames"),
                ::py::arg("forcePlates"),
                ::py::arg("rootHistoryLen") = 5,
                ::py::arg("rootHistoryStride") = 1,
                ::py::arg("explicitVels") = Eigen::MatrixXs::Zero(0, 0),
                ::py::arg("explicitAccs") = Eigen::MatrixXs::Zero(0, 0))
            .def(
                "setLinearResidual",
                &dart::biomechanics::SubjectOnDiskTrialPass::setLinearResidual,
                ::py::arg("linearResidual"))
            .def(
                "getLinearResidual",
                &dart::biomechanics::SubjectOnDiskTrialPass::getLinearResidual)
            .def(
                "setAngularResidual",
                &dart::biomechanics::SubjectOnDiskTrialPass::setAngularResidual,
                ::py::arg("angularResidual"))
            .def(
                "getAngularResidual",
                &dart::biomechanics::SubjectOnDiskTrialPass::getAngularResidual)
            .def(
                "setPoses",
                &dart::biomechanics::SubjectOnDiskTrialPass::setPoses,
                ::py::arg("poses"))
            .def(
                "getPoses",
                &dart::biomechanics::SubjectOnDiskTrialPass::getPoses)
            .def(
                "setVels",
                &dart::biomechanics::SubjectOnDiskTrialPass::setVels,
                ::py::arg("vels"))
            .def(
                "getVels", &dart::biomechanics::SubjectOnDiskTrialPass::getVels)
            .def(
                "setAccs",
                &dart::biomechanics::SubjectOnDiskTrialPass::setAccs,
                ::py::arg("accs"))
            .def(
                "getAccs", &dart::biomechanics::SubjectOnDiskTrialPass::getAccs)
            .def(
                "setTaus",
                &dart::biomechanics::SubjectOnDiskTrialPass::setTaus,
                ::py::arg("taus"))
            .def(
                "getTaus", &dart::biomechanics::SubjectOnDiskTrialPass::getTaus)
            .def(
                "setGroundBodyWrenches",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setGroundBodyWrenches,
                ::py::arg("wrenches"))
            .def(
                "getGroundBodyWrenches",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getGroundBodyWrenches)
            .def(
                "setGroundBodyCopTorqueForce",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setGroundBodyCopTorqueForce,
                ::py::arg("copTorqueForces"))
            .def(
                "getGroundBodyCopTorqueForce",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getGroundBodyCopTorqueForce)
            .def(
                "setComPoses",
                &dart::biomechanics::SubjectOnDiskTrialPass::setComPoses,
                ::py::arg("poses"))
            .def(
                "getComPoses",
                &dart::biomechanics::SubjectOnDiskTrialPass::getComPoses)
            .def(
                "setComVels",
                &dart::biomechanics::SubjectOnDiskTrialPass::setComVels,
                ::py::arg("vels"))
            .def(
                "getComVels",
                &dart::biomechanics::SubjectOnDiskTrialPass::getComVels)
            .def(
                "setComAccs",
                &dart::biomechanics::SubjectOnDiskTrialPass::setComAccs,
                ::py::arg("accs"))
            .def(
                "getComAccs",
                &dart::biomechanics::SubjectOnDiskTrialPass::getComAccs)
            .def(
                "setComAccsInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setComAccsInRootFrame,
                ::py::arg("accs"))
            .def(
                "getComAccsInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getComAccsInRootFrame)
            .def(
                "setResidualWrenchInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setResidualWrenchInRootFrame,
                ::py::arg("wrenches"))
            .def(
                "getResidualWrenchInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getResidualWrenchInRootFrame)
            .def(
                "setGroundBodyWrenchesInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setGroundBodyWrenchesInRootFrame,
                ::py::arg("wrenches"))
            .def(
                "getGroundBodyWrenchesInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getGroundBodyWrenchesInRootFrame)
            .def(
                "setGroundBodyCopTorqueForceInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setGroundBodyCopTorqueForceInRootFrame,
                ::py::arg("copTorqueForces"))
            .def(
                "getGroundBodyCopTorqueForceInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getGroundBodyCopTorqueForceInRootFrame)
            .def(
                "setJointCenters",
                &dart::biomechanics::SubjectOnDiskTrialPass::setJointCenters,
                ::py::arg("centers"))
            .def(
                "getJointCenters",
                &dart::biomechanics::SubjectOnDiskTrialPass::getJointCenters)
            .def(
                "setJointCentersInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setJointCentersInRootFrame,
                ::py::arg("centers"))
            .def(
                "getJointCentersInRootFrame",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getJointCentersInRootFrame)
            .def(
                "setResamplingMatrix",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    setResamplingMatrix,
                ::py::arg("resamplingMatrix"))
            .def(
                "getResamplingMatrix",
                &dart::biomechanics::SubjectOnDiskTrialPass::
                    getResamplingMatrix);

  auto subjectOnDiskTrial
      = ::py::class_<
            dart::biomechanics::SubjectOnDiskTrial,
            std::shared_ptr<dart::biomechanics::SubjectOnDiskTrial>>(
            m, "SubjectOnDiskTrial")
            .def(::py::init<>())
            .def(
                "setName",
                &dart::biomechanics::SubjectOnDiskTrial::setName,
                ::py::arg("name"))
            .def(
                "setTimestep",
                &dart::biomechanics::SubjectOnDiskTrial::setTimestep,
                ::py::arg("timestep"))
            .def(
                "getTimestep",
                &dart::biomechanics::SubjectOnDiskTrial::getTimestep)
            .def(
                "setTrialTags",
                &dart::biomechanics::SubjectOnDiskTrial::setTrialTags,
                ::py::arg("trialTags"))
            .def(
                "setOriginalTrialName",
                &dart::biomechanics::SubjectOnDiskTrial::setOriginalTrialName,
                ::py::arg("name"))
            .def(
                "setSplitIndex",
                &dart::biomechanics::SubjectOnDiskTrial::setSplitIndex,
                ::py::arg("split"))
            .def(
                "setMissingGRFReason",
                &dart::biomechanics::SubjectOnDiskTrial::setMissingGRFReason,
                ::py::arg("missingGRFReason"))
            .def(
                "getMissingGRFReason",
                &dart::biomechanics::SubjectOnDiskTrial::getMissingGRFReason)
            .def(
                "setCustomValues",
                &dart::biomechanics::SubjectOnDiskTrial::setCustomValues,
                ::py::arg("customValues"))
            .def(
                "setMarkerNamesGuessed",
                &dart::biomechanics::SubjectOnDiskTrial::setMarkerNamesGuessed,
                ::py::arg("markersGuessed"))
            .def(
                "setMarkerObservations",
                &dart::biomechanics::SubjectOnDiskTrial::setMarkerObservations,
                ::py::arg("markerObservations"))
            .def(
                "setAccObservations",
                &dart::biomechanics::SubjectOnDiskTrial::setAccObservations,
                ::py::arg("accObservations"))
            .def(
                "setGyroObservations",
                &dart::biomechanics::SubjectOnDiskTrial::setGyroObservations,
                ::py::arg("gyroObservations"))
            .def(
                "setEmgObservations",
                &dart::biomechanics::SubjectOnDiskTrial::setEmgObservations,
                ::py::arg("emgObservations"))
            .def(
                "setExoTorques",
                &dart::biomechanics::SubjectOnDiskTrial::setExoTorques,
                ::py::arg("exoTorques"))
            .def(
                "setForcePlates",
                &dart::biomechanics::SubjectOnDiskTrial::setForcePlates,
                ::py::arg("forcePlates"))
            .def(
                "getForcePlates",
                &dart::biomechanics::SubjectOnDiskTrial::getForcePlates)
            .def(
                "addPass",
                &dart::biomechanics::SubjectOnDiskTrial::addPass,
                "This creates a new :code:`SubjectOnDiskTrialPass` for this "
                "trial, and returns it. That object can store results from IK "
                "and ID, as well as other results from the processing "
                "pipeline.")
            .def(
                "getPasses",
                &dart::biomechanics::SubjectOnDiskTrial::getPasses);

  auto subjectOnDiskPassHead
      = ::py::class_<
            dart::biomechanics::SubjectOnDiskPassHeader,
            std::shared_ptr<dart::biomechanics::SubjectOnDiskPassHeader>>(
            m, "SubjectOnDiskPassHeader")
            .def(::py::init<>())
            //   SubjectOnDiskPassHeader&
            //   setProcessingPassType(ProcessingPassType type);
            .def(
                "setProcessingPassType",
                &dart::biomechanics::SubjectOnDiskPassHeader::
                    setProcessingPassType,
                ::py::arg("type"))
            .def(
                "getProcessingPassType",
                &dart::biomechanics::SubjectOnDiskPassHeader::
                    getProcessingPassType)
            //   SubjectOnDiskPassHeader& setOpenSimFileText(
            //       const std::string& openSimFileText);
            .def(
                "setOpenSimFileText",
                &dart::biomechanics::SubjectOnDiskPassHeader::
                    setOpenSimFileText,
                ::py::arg("openSimFileText"))
            .def(
                "getOpenSimFileText",
                &dart::biomechanics::SubjectOnDiskPassHeader::
                    getOpenSimFileText);

  auto subjectOnDiskHeader
      = ::py::class_<
            dart::biomechanics::SubjectOnDiskHeader,
            std::shared_ptr<dart::biomechanics::SubjectOnDiskHeader>>(
            m, "SubjectOnDiskHeader")
            .def(::py::init<>())
            .def(
                "setNumDofs",
                &dart::biomechanics::SubjectOnDiskHeader::setNumDofs,
                ::py::arg("dofs"))
            .def(
                "setNumJoints",
                &dart::biomechanics::SubjectOnDiskHeader::setNumJoints,
                ::py::arg("joints"))
            .def(
                "setGroundForceBodies",
                &dart::biomechanics::SubjectOnDiskHeader::setGroundForceBodies,
                ::py::arg("groundForceBodies"))
            .def(
                "setCustomValueNames",
                &dart::biomechanics::SubjectOnDiskHeader::setCustomValueNames,
                ::py::arg("customValueNames"))
            .def(
                "setBiologicalSex",
                &dart::biomechanics::SubjectOnDiskHeader::setBiologicalSex,
                ::py::arg("biologicalSex"))
            .def(
                "setHeightM",
                &dart::biomechanics::SubjectOnDiskHeader::setHeightM,
                ::py::arg("heightM"))
            .def(
                "setMassKg",
                &dart::biomechanics::SubjectOnDiskHeader::setMassKg,
                ::py::arg("massKg"))
            .def(
                "setAgeYears",
                &dart::biomechanics::SubjectOnDiskHeader::setAgeYears,
                ::py::arg("ageYears"))
            .def(
                "setSubjectTags",
                &dart::biomechanics::SubjectOnDiskHeader::setSubjectTags,
                ::py::arg("subjectTags"))
            .def(
                "setHref",
                &dart::biomechanics::SubjectOnDiskHeader::setHref,
                ::py::arg("sourceHref"))
            .def(
                "setNotes",
                &dart::biomechanics::SubjectOnDiskHeader::setNotes,
                ::py::arg("notes"))
            .def(
                "addProcessingPass",
                &dart::biomechanics::SubjectOnDiskHeader::addProcessingPass)
            .def(
                "getProcessingPasses",
                &dart::biomechanics::SubjectOnDiskHeader::getProcessingPasses)
            .def("addTrial", &dart::biomechanics::SubjectOnDiskHeader::addTrial)
            .def(
                "getTrials",
                &dart::biomechanics::SubjectOnDiskHeader::getTrials)
            .def(
                "setTrials",
                &dart::biomechanics::SubjectOnDiskHeader::setTrials,
                ::py::arg("trials"))
            .def(
                "recomputeColumnNames",
                &dart::biomechanics::SubjectOnDiskHeader::recomputeColumnNames);

  auto subjectOnDisk
      = ::py::class_<
            dart::biomechanics::SubjectOnDisk,
            std::shared_ptr<dart::biomechanics::SubjectOnDisk>>(
            m, "SubjectOnDisk")
            //   SubjectOnDisk(const std::string& path);
            .def(::py::init<std::string>(), ::py::arg("path"))
            //   /// This will write a B3D file to disk
            //   static void writeB3D(const std::string& path,
            //   SubjectOnDiskHeader& header);
            .def_static(
                "writeB3D",
                &dart::biomechanics::SubjectOnDisk::writeB3D,
                ::py::arg("path"),
                ::py::arg("header"))
            //   /// This will read the skeleton from the binary, and optionally
            //   use the passed
            //   /// in Geometry folder.
            //   std::shared_ptr<dynamics::Skeleton> readSkel(
            //       int processingPass, std::string geometryFolder = "");
            .def(
                "loadAllFrames",
                &dart::biomechanics::SubjectOnDisk::loadAllFrames,
                "This loads all the frames of data, and fills in the "
                "processing pass data matrices in the proto header classes.")
            .def(
                "getHeaderProto",
                &dart::biomechanics::SubjectOnDisk::getHeaderProto,
                "This returns the raw proto header for this subject, which can "
                "be used to write out a new B3D file")
            .def(
                "readForcePlates",
                &dart::biomechanics::SubjectOnDisk::readForcePlates,
                "This reads all the raw sensor data for this trial, and "
                "constructs force plates.")
            .def(
                "readSkel",
                &dart::biomechanics::SubjectOnDisk::readSkel,
                ::py::arg("processingPass"),
                ::py::arg("geometryFolder") = "",
                "This will read the skeleton from the binary, and optionally "
                "use the passed in :code:`geometryFolder` to load meshes. We "
                "do not bundle meshes with :code:`SubjectOnDisk` files, to "
                "save space. If you do not pass in :code:`geometryFolder`, "
                "expect to get warnings about being unable to load meshes, and "
                "expect that your skeleton will not display if you attempt to "
                "visualize it.")
            //   /// This will read the raw OpenSim XML file text out of the
            //   binary, and return
            //   /// it as a string
            //   std::string getOpensimFileText(int processingPass);
            .def(
                "getOpensimFileText",
                &dart::biomechanics::SubjectOnDisk::getOpensimFileText,
                ::py::arg("processingPass"),
                "This will read the raw OpenSim file XML out of the "
                "SubjectOnDisk, and return it as a string.")
            //   /// This will read from disk and allocate a number of Frame
            //   objects.
            //   /// These Frame objects are assumed to be
            //   /// short-lived, to save working memory.
            //   ///
            //   /// On OOB access, prints an error and returns an empty vector.
            //   std::vector<std::shared_ptr<Frame>> readFrames(
            //       int trial,
            //       int startFrame,
            //       int numFramesToRead = 1,
            //       bool includeSensorData = true,
            //       bool includeProcessingPasses = true,
            //       int stride = 1,
            //       s_t contactThreshold = 1.0);
            .def(
                "getLowpassCutoffFrequency",
                &dart::biomechanics::SubjectOnDisk::getLowpassCutoffFrequency,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "If we're doing a lowpass filter on this pass, then what was "
                "the cutoff frequency of that (Butterworth) filter?")
            .def(
                "getLowpassFilterOrder",
                &dart::biomechanics::SubjectOnDisk::getLowpassFilterOrder,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "If we're doing a lowpass filter on this pass, then what was "
                "the order of that (Butterworth) filter?")
            .def(
                "getForceplateCutoffs",
                &dart::biomechanics::SubjectOnDisk::getForceplateCutoffs,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "If we reprocessed the force plates with a cutoff, then these "
                "are the cutoff values we used.")
            .def(
                "readFrames",
                &dart::biomechanics::SubjectOnDisk::readFrames,
                ::py::arg("trial"),
                ::py::arg("startFrame"),
                ::py::arg("numFramesToRead") = 1,
                ::py::arg("includeSensorData") = true,
                ::py::arg("includeProcessingPasses") = true,
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
            //   /// This returns the number of trials on the subject
            //   int getNumTrials();
            .def(
                "getNumTrials",
                &dart::biomechanics::SubjectOnDisk::getNumTrials,
                "This returns the number of trials that are in this file.")
            //   /// This returns the length of the trial
            //   int getTrialLength(int trial);
            .def(
                "getTrialLength",
                &dart::biomechanics::SubjectOnDisk::getTrialLength,
                ::py::arg("trial"),
                "This returns the length of the trial requested")
            .def(
                "getTrialOriginalName",
                &dart::biomechanics::SubjectOnDisk::getTrialOriginalName,
                ::py::arg("trial"),
                "This returns the original name of the trial before it was "
                "(potentially) split into multiple pieces")
            .def(
                "getTrialSplitIndex",
                &dart::biomechanics::SubjectOnDisk::getTrialSplitIndex,
                ::py::arg("trial"),
                "This returns the index of the split, if this trial was the "
                "result of splitting an original trial into multiple pieces")
            //   /// This returns the number of processing passes in the trial
            //   int getTrialNumProcessingPasses(int trial);
            .def(
                "getTrialNumProcessingPasses",
                &dart::biomechanics::SubjectOnDisk::getTrialNumProcessingPasses,
                ::py::arg("trial"),
                "This returns the number of processing passes that "
                "successfully completed on this trial")
            //   /// This returns the timestep size for the trial
            //   s_t getTrialTimestep(int trial);
            .def(
                "getTrialTimestep",
                &dart::biomechanics::SubjectOnDisk::getTrialTimestep,
                ::py::arg("trial"),
                "This returns the timestep size for the trial requested, in "
                "seconds per frame")
            //   /// This returns the number of DOFs for the model on this
            //   Subject int getNumDofs();
            .def(
                "getNumDofs",
                &dart::biomechanics::SubjectOnDisk::getNumDofs,
                "This returns the number of DOFs for the model on this Subject")
            .def(
                "getNumJoints",
                &dart::biomechanics::SubjectOnDisk::getNumJoints,
                "This returns the number of joints for the model on this "
                "Subject")
            //   /// This returns the vector of enums of type
            //   'MissingGRFReason', which labels
            //   /// why each time step was identified as 'probablyMissingGRF'.
            //   std::vector<MissingGRFReason> getMissingGRFReason(int trial);
            .def(
                "getMissingGRF",
                &dart::biomechanics::SubjectOnDisk::getMissingGRF,
                ::py::arg("trial"),
                R"doc(
            This returns an array of enum values, one per frame in the specified trial,
            each describing whether physics data can be trusted for the corresponding frame of that trial.
            
            Each frame is either `MissingGRFReason.notMissingGRF`, in which case the physics data is probably trustworthy, or
            some other value indicating why the processing system heuristics believe that there is likely to be unmeasured 
            external force acting on the body at this time.

            WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on the corresponding frame!!

            This method is provided to give a cheaper way to filter out frames we want to ignore for training, without having to call
            the more expensive :code:`loadFrames()` and examine frames individually.
          )doc")
            //   int getNumProcessingPasses();
            .def(
                "getNumProcessingPasses",
                &dart::biomechanics::SubjectOnDisk::getNumProcessingPasses,
                "This returns the number of processing passes that were "
                "successfully completed on this subject. IMPORTANT: Just "
                "because a processing pass was done for the subject does not "
                "mean that every trial will have successfully completed that "
                "processing pass. For example, some trials may lack force "
                "plate data, and thus will not have a dynamics pass that "
                "requires force plate data.")
            //   ProcessingPassType getProcessingPassType(int processingPass);
            .def(
                "getProcessingPassType",
                &dart::biomechanics::SubjectOnDisk::getProcessingPassType,
                ::py::arg("processingPass"),
                "This returns the type of processing pass at a given index, up "
                "to the number of processing passes that were done")
            //   std::vector<bool> getDofPositionsObserved(int trial, int
            //   processingPass);
            .def(
                "getDofPositionsObserved",
                &dart::biomechanics::SubjectOnDisk::getDofPositionsObserved,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of booleans indicating which DOFs "
                "have their positions observed during this trial")
            //   std::vector<bool> getDofVelocitiesFiniteDifferenced(
            //       int trial, int processingPass);
            .def(
                "getDofVelocitiesFiniteDifferenced",
                &dart::biomechanics::SubjectOnDisk::
                    getDofVelocitiesFiniteDifferenced,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of booleans indicating which DOFs "
                "have their velocities from finite-differencing during this "
                "trial (as opposed to observed directly through a gyroscope or "
                "IMU)")
            //   std::vector<bool> getDofAccelerationsFiniteDifferenced(
            //       int trial, int processingPass);
            .def(
                "getDofAccelerationsFiniteDifferenced",
                &dart::biomechanics::SubjectOnDisk::
                    getDofAccelerationsFiniteDifferenced,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of booleans indicating which DOFs "
                "have their accelerations from finite-differencing during this "
                "trial (as opposed to observed directly through a "
                "accelerometer or IMU)")
            //   std::vector<s_t> getTrialLinearResidualNorms(int trial, int
            .def(
                "getTrialLinearResidualNorms",
                &dart::biomechanics::SubjectOnDisk::getTrialLinearResidualNorms,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of scalars indicating the norm of the "
                "root residual forces on each timestep of a given "
                "trial")
            //   processingPass); std::vector<s_t>
            //   getTrialAngularResidualNorms(int trial,
            .def(
                "getTrialAngularResidualNorms",
                &dart::biomechanics::SubjectOnDisk::
                    getTrialAngularResidualNorms,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of scalars indicating the norm of the "
                "root residual torques on each timestep of a given "
                "trial")
            //   int processingPass); std::vector<s_t> getTrialMarkerRMSs(int
            //   trial, int
            .def(
                "getTrialMarkerRMSs",
                &dart::biomechanics::SubjectOnDisk::getTrialMarkerRMSs,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of scalars indicating the RMS marker "
                "error on each timestep of a given trial")
            //   processingPass); std::vector<s_t> getTrialMarkerMaxs(int trial,
            //   int
            .def(
                "getTrialMarkerMaxs",
                &dart::biomechanics::SubjectOnDisk::getTrialMarkerMaxs,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of scalars indicating the max marker "
                "error on each timestep of a given trial")
            //   /// This returns the maximum absolute velocity of any DOF at
            //   each timestep for
            //   /// a given trial
            //   std::vector<s_t> getTrialMaxJointVelocity(int trial, int
            //   processingPass);
            .def(
                "getTrialMaxJointVelocity",
                &dart::biomechanics::SubjectOnDisk::getTrialMaxJointVelocity,
                ::py::arg("trial"),
                ::py::arg("processingPass"),
                "This returns the vector of scalars indicating the maximum "
                "absolute velocity of all DOFs on each timestep of a given "
                "trial")
            //   /// This returns the list of contact body names for this
            //   Subject std::vector<std::string> getGroundForceBodies();
            .def(
                "getGroundForceBodies",
                &dart::biomechanics::SubjectOnDisk::getGroundForceBodies,
                "A list of the :code:`body_name`'s for each body that was "
                "assumed to be able to take ground-reaction-force from force "
                "plates.")
            //   /// This returns the list of custom value names stored in this
            //   subject std::vector<std::string> getCustomValues();
            .def(
                "getCustomValues",
                &dart::biomechanics::SubjectOnDisk::getCustomValues,
                "A list of all the different types of custom values that this "
                "SubjectOnDisk contains. These are unspecified, and are "
                "intended to allow an easy extension of the format to unusual "
                "types of data (like exoskeleton torques or unusual physical "
                "sensors) that may be present on some subjects but not others.")
            //   /// This returns the dimension of the custom value specified by
            //   `valueName` int getCustomValueDim(std::string valueName);
            .def(
                "getCustomValueDim",
                &dart::biomechanics::SubjectOnDisk::getCustomValueDim,
                ::py::arg("valueName"),
                "This returns the dimension of the custom value specified by "
                ":code:`valueName`")
            //   /// The name of the trial, if provided, or else an empty string
            //   std::string getTrialName(int trial);
            .def(
                "getTrialName",
                &dart::biomechanics::SubjectOnDisk::getTrialName,
                ::py::arg("trial"),
                "This returns the human readable name of the specified trial, "
                "given by the person who uploaded the data to AddBiomechanics. "
                "This isn't necessary for training, but may be useful for "
                "analyzing the data.")
            //   std::string getBiologicalSex();
            .def(
                "getBiologicalSex",
                &dart::biomechanics::SubjectOnDisk::getBiologicalSex,
                "This returns a string, one of \"male\", \"female\", or "
                "\"unknown\".")
            //   double getHeightM();
            .def(
                "getHeightM",
                &dart::biomechanics::SubjectOnDisk::getHeightM,
                "This returns the height in meters, or 0.0 if unknown.")
            //   double getMassKg();
            .def(
                "getMassKg",
                &dart::biomechanics::SubjectOnDisk::getMassKg,
                "This returns the mass in kilograms, or 0.0 if unknown.")
            //   /// This gets the tags associated with the subject, if there
            //   are any. std::vector<std::string> getSubjectTags();
            .def(
                "getSubjectTags",
                &dart::biomechanics::SubjectOnDisk::getSubjectTags,
                "This returns the list of tags attached to this subject, which "
                "are arbitrary strings from the AddBiomechanics platform.")

            //   /// This gets the tags associated with the trial, if there are
            //   any. std::vector<std::string> getTrialTags(int trial);
            .def(
                "getTrialTags",
                &dart::biomechanics::SubjectOnDisk::getTrialTags,
                ::py::arg("trial"),
                "This returns the list of tags attached to a given trial "
                "index, which are arbitrary strings from the AddBiomechanics "
                "platform.")
            //   int getAgeYears();
            .def(
                "getAgeYears",
                &dart::biomechanics::SubjectOnDisk::getAgeYears,
                "This returns the age of the subject, or 0 if unknown.")
            //   /// This returns the number of raw force plates that were used
            //   to generate the
            //   /// data, for this trial
            //   int getNumForcePlates(int trial);
            .def(
                "getNumForcePlates",
                &dart::biomechanics::SubjectOnDisk::getNumForcePlates,
                ::py::arg("trial"),
                "The number of force plates in the source data.")
            //   /// This returns the corners (in 3D space) of the selected
            //   force plate, for
            //   /// this trial. Empty arrays on out of bounds.
            //   std::vector<Eigen::Vector3s> getForcePlateCorners(int trial,
            //   int forcePlate);
            .def(
                "getForcePlateCorners",
                &dart::biomechanics::SubjectOnDisk::getForcePlateCorners,
                ::py::arg("trial"),
                ::py::arg("forcePlate"),
                "Get an array of force plate corners (as 3D vectors) for the "
                "given force plate in the given trial. Empty array on "
                "out-of-bounds access.")
            //   /// This gets the href link associated with the subject, if
            //   there is one. std::string getHref();
            .def(
                "getHref",
                &dart::biomechanics::SubjectOnDisk::getHref,
                "The AddBiomechanics link for this subject's data.")
            //   /// This gets the notes associated with the subject, if there
            //   are any. std::string getNotes();
            .def(
                "getNotes",
                &dart::biomechanics::SubjectOnDisk::getNotes,
                "The notes (if any) added by the person who uploaded this data "
                "to AddBiomechanics.");

  subjectOnDisk.doc() = R"doc(
        This is for doing ML and large-scale data analysis. The idea here is to
        create a lazy-loadable view of a subject, where everything remains on disk
        until asked for. That way we can instantiate thousands of these in memory,
        and not worry about OOM'ing a machine.
      )doc";
}
} // namespace python
} // namespace dart