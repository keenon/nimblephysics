/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include <Eigen/Dense>
#include <dart/biomechanics/MarkerFitter.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void MarkerFitter(py::module& m)
{
  auto markerFitter = ::py::class_<
      dart::biomechanics::MarkerFitter,
      std::shared_ptr<dart::biomechanics::MarkerFitter>>(m, "MarkerFitter");

  ::py::class_<dart::biomechanics::MarkerFitterState>(m, "MarkerFitterState")
      .def_readwrite(
          "bodyNames", &dart::biomechanics::MarkerFitterState::bodyNames)
      .def_readwrite(
          "bodyScales", &dart::biomechanics::MarkerFitterState::bodyScales)
      .def_readwrite(
          "markerOrder", &dart::biomechanics::MarkerFitterState::markerOrder)
      .def_readwrite(
          "markerOffsets",
          &dart::biomechanics::MarkerFitterState::markerOffsets)
      .def_readwrite(
          "markerErrorsAtTimesteps",
          &dart::biomechanics::MarkerFitterState::markerErrorsAtTimesteps)
      .def_readwrite(
          "jointErrorsAtTimesteps",
          &dart::biomechanics::MarkerFitterState::jointErrorsAtTimesteps)
      .def_readwrite(
          "jointOrder", &dart::biomechanics::MarkerFitterState::jointOrder)
      .def_readwrite(
          "posesAtTimesteps",
          &dart::biomechanics::MarkerFitterState::posesAtTimesteps)
      // Gradient of current state
      .def_readwrite(
          "bodyScalesGrad",
          &dart::biomechanics::MarkerFitterState::bodyScalesGrad)
      .def_readwrite(
          "markerOffsetsGrad",
          &dart::biomechanics::MarkerFitterState::markerOffsetsGrad)
      .def_readwrite(
          "markerErrorsAtTimestepsGrad",
          &dart::biomechanics::MarkerFitterState::markerErrorsAtTimestepsGrad)
      .def_readwrite(
          "jointErrorsAtTimestepsGrad",
          &dart::biomechanics::MarkerFitterState::jointErrorsAtTimestepsGrad)
      .def_readwrite(
          "posesAtTimestepsGrad",
          &dart::biomechanics::MarkerFitterState::posesAtTimestepsGrad);

  ::py::class_<
      dart::biomechanics::BilevelFitResult,
      std::shared_ptr<dart::biomechanics::BilevelFitResult>>(
      m, "BilevelFitResult")
      .def_readwrite("success", &dart::biomechanics::BilevelFitResult::success)
      .def_readwrite(
          "groupScales", &dart::biomechanics::BilevelFitResult::groupScales)
      .def_readwrite(
          "markerOffsets", &dart::biomechanics::BilevelFitResult::markerOffsets)
      .def_readwrite("poses", &dart::biomechanics::BilevelFitResult::poses)
      .def_readwrite(
          "posesMatrix", &dart::biomechanics::BilevelFitResult::posesMatrix)
      .def_readwrite(
          "rawMarkerOffsets",
          &dart::biomechanics::BilevelFitResult::rawMarkerOffsets);

  ::py::class_<dart::biomechanics::MarkerInitialization>(
      m, "MarkerInitialization")
      .def_readwrite("poses", &dart::biomechanics::MarkerInitialization::poses)
      .def_readwrite(
          "jointCenters",
          &dart::biomechanics::MarkerInitialization::jointCenters)
      .def_readwrite(
          "joints", &dart::biomechanics::MarkerInitialization::joints)
      .def_readwrite(
          "groupScales", &dart::biomechanics::MarkerInitialization::groupScales)
      .def_readwrite(
          "updatedMarkerMap",
          &dart::biomechanics::MarkerInitialization::updatedMarkerMap)
      .def_readwrite(
          "markerOffsets",
          &dart::biomechanics::MarkerInitialization::markerOffsets);

  ::py::class_<
      dart::biomechanics::IMUFineTuneProblem,
      std::shared_ptr<dart::biomechanics::IMUFineTuneProblem>>(
      m, "IMUFineTuneProblem")
      .def(
          "getProblemSize",
          &dart::biomechanics::IMUFineTuneProblem::getProblemSize)
      .def(
          "setWeightGyros",
          &dart::biomechanics::IMUFineTuneProblem::setWeightGyros,
          ::py::arg("weight"))
      .def(
          "setWeightAccs",
          &dart::biomechanics::IMUFineTuneProblem::setWeightAccs,
          ::py::arg("weight"))
      .def(
          "setWeightPoses",
          &dart::biomechanics::IMUFineTuneProblem::setRegularizePoses,
          ::py::arg("weight"))
      .def("flatten", &dart::biomechanics::IMUFineTuneProblem::flatten)
      .def(
          "unflatten",
          &dart::biomechanics::IMUFineTuneProblem::unflatten,
          ::py::arg("x"))
      .def("getLoss", &dart::biomechanics::IMUFineTuneProblem::getLoss)
      .def("getPoses", &dart::biomechanics::IMUFineTuneProblem::getPoses)
      .def("getVels", &dart::biomechanics::IMUFineTuneProblem::getVels)
      .def("getAccs", &dart::biomechanics::IMUFineTuneProblem::getAccs)
      .def("getGrad", &dart::biomechanics::IMUFineTuneProblem::getGrad);

  ::py::class_<dart::biomechanics::InitialMarkerFitParams>(
      m, "InitialMarkerFitParams")
      .def(::py::init<>())
      .def(
          "__repr__",
          [](dart::biomechanics::InitialMarkerFitParams* self) -> std::string {
            // TODO: add remaining params
            return "InitialMarkerFitParams(numBlocks="
                   + std::to_string(self->numBlocks) + ")";
          })
      .def_readwrite(
          "markerWeights",
          &dart::biomechanics::InitialMarkerFitParams::markerWeights)
      .def_readwrite(
          "markerOffsets",
          &dart::biomechanics::InitialMarkerFitParams::markerOffsets)
      .def_readwrite(
          "joints", &dart::biomechanics::InitialMarkerFitParams::joints)
      .def_readwrite(
          "jointCenters",
          &dart::biomechanics::InitialMarkerFitParams::jointCenters)
      .def_readwrite(
          "jointWeights",
          &dart::biomechanics::InitialMarkerFitParams::jointWeights)
      .def_readwrite(
          "numBlocks", &dart::biomechanics::InitialMarkerFitParams::numBlocks)
      .def_readwrite(
          "initPoses", &dart::biomechanics::InitialMarkerFitParams::initPoses)
      .def_readwrite(
          "groupScales",
          &dart::biomechanics::InitialMarkerFitParams::groupScales)
      .def_readwrite(
          "dontRescaleBodies",
          &dart::biomechanics::InitialMarkerFitParams::dontRescaleBodies)
      .def_readwrite(
          "maxTrialsToUseForMultiTrialScaling",
          &dart::biomechanics::InitialMarkerFitParams::
              maxTrialsToUseForMultiTrialScaling)
      .def_readwrite(
          "maxTimestepsToUseForMultiTrialScaling",
          &dart::biomechanics::InitialMarkerFitParams::
              maxTimestepsToUseForMultiTrialScaling)
      .def(
          "setMarkerWeights",
          &dart::biomechanics::InitialMarkerFitParams::setMarkerWeights,
          ::py::arg("markerWeights"))
      .def(
          "setMarkerOffsets",
          &dart::biomechanics::InitialMarkerFitParams::setMarkerOffsets,
          ::py::arg("markerOffsets"))
      .def(
          "setJointCenters",
          &dart::biomechanics::InitialMarkerFitParams::setJointCenters,
          ::py::arg("joints"),
          ::py::arg("jointCenters"),
          ::py::arg("jointAdjacentMarkers"))
      .def(
          "setNumBlocks",
          &dart::biomechanics::InitialMarkerFitParams::setNumBlocks,
          ::py::arg("numBlocks"))
      .def(
          "setNumIKTries",
          &dart::biomechanics::InitialMarkerFitParams::setNumIKTries,
          ::py::arg("tries"))
      .def(
          "setInitPoses",
          &dart::biomechanics::InitialMarkerFitParams::setInitPoses,
          ::py::arg("initPoses"))
      .def(
          "setGroupScales",
          &dart::biomechanics::InitialMarkerFitParams::setGroupScales,
          ::py::arg("groupScales"))
      .def(
          "setDontRescaleBodies",
          &dart::biomechanics::InitialMarkerFitParams::setDontRescaleBodies,
          ::py::arg("dontRescaleBodies"))
      .def(
          "setMaxTrialsToUseForMultiTrialScaling",
          &dart::biomechanics::InitialMarkerFitParams::
              setMaxTrialsToUseForMultiTrialScaling,
          ::py::arg("numTrials"))
      .def(
          "setMaxTimestepsToUseForMultiTrialScaling",
          &dart::biomechanics::InitialMarkerFitParams::
              setMaxTimestepsToUseForMultiTrialScaling,
          ::py::arg("numTimesteps"))
      .def(
          "setJointCentersAndWeights",
          &dart::biomechanics::InitialMarkerFitParams::
              setJointCentersAndWeights,
          ::py::arg("joints"),
          ::py::arg("jointCenters"),
          ::py::arg("jointAdjacentMarkers"),
          ::py::arg("jointWeights"));

  markerFitter
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              std::map<
                  std::string,
                  std::pair<dynamics::BodyNode*, Eigen::Vector3s>>,
              bool>(),
          ::py::arg("skeleton"),
          ::py::arg("markers"),
          ::py::arg("ignoreVirtualJointCenterMarkers") = false)
      .def(
          "setInitialIKSatisfactoryLoss",
          &dart::biomechanics::MarkerFitter::setInitialIKSatisfactoryLoss,
          ::py::arg("loss"))
      .def(
          "setInitialIKMaxRestarts",
          &dart::biomechanics::MarkerFitter::setInitialIKMaxRestarts,
          ::py::arg("starts"))
      .def(
          "setParallelIKWarps",
          &dart::biomechanics::MarkerFitter::setParallelIKWarps,
          ::py::arg("parallelWarps"),
          R"pydoc(If True, this processes "single threaded" IK tasks 32 timesteps at a time
            (a "warp"), in parallel, using the first timestep of the warp as the
            initialization for the whole warp. Defaults to False.
          )pydoc")
      .def(
          "setMaxMarkerOffset",
          &dart::biomechanics::MarkerFitter::setMaxMarkerOffset,
          ::py::arg("offset"))
      .def(
          "setIgnoreJointLimits",
          &dart::biomechanics::MarkerFitter::setIgnoreJointLimits,
          ::py::arg("ignore"))
      .def(
          "setDebugLoss",
          &dart::biomechanics::MarkerFitter::setDebugLoss,
          ::py::arg("debug"))
      .def(
          "setIterationLimit",
          &dart::biomechanics::MarkerFitter::setIterationLimit,
          ::py::arg("iters"))
      .def(
          "setAnthropometricPrior",
          &dart::biomechanics::MarkerFitter::setAnthropometricPrior,
          ::py::arg("prior"),
          ::py::arg("weight") = 0.001)
      .def(
          "setExplicitHeightPrior",
          &dart::biomechanics::MarkerFitter::setExplicitHeightPrior,
          ::py::arg("prior"),
          ::py::arg("weight") = 1e3)
      .def(
          "setStaticTrial",
          &dart::biomechanics::MarkerFitter::setStaticTrial,
          ::py::arg("markerObservationsMapAtStaticPose"),
          ::py::arg("staticPose"))
      .def(
          "setStaticTrialWeight",
          &dart::biomechanics::MarkerFitter::setStaticTrialWeight,
          ::py::arg("weight"))
      .def(
          "setJointForceFieldThresholdDistance",
          &dart::biomechanics::MarkerFitter::
              setJointForceFieldThresholdDistance,
          ::py::arg("minDistance"),
          R"pydoc(
  This sets the minimum distance joints have to be apart in order to get
  zero "force field" loss. Any joints closer than this (in world space) will
  incur a penalty.
          )pydoc")
      .def(
          "setJointForceFieldSoftness",
          &dart::biomechanics::MarkerFitter::setJointForceFieldSoftness,
          ::py::arg("softness"),
          R"pydoc(
  Larger values will increase the softness of the threshold penalty. Smaller
  values, as they approach zero, will have an almost perfectly vertical
  penality for going below the threshold distance. That would be hard to
  optimize, so don't make it too small.
          )pydoc")
      .def(
          "setPostprocessAnatomicalMarkerOffsets",
          &dart::biomechanics::MarkerFitter::
              setPostprocessAnatomicalMarkerOffsets,
          ::py::arg("postprocess"),
          R"pydoc(
  If we set this to true, then after the main optimization completes we will
  do a final step to "center" the error of the anatomical markers. This
  minimizes marker RMSE, but does NOT respect the weights about how far
  markers should be allowed to move.
          )pydoc")
      .def(
          "setPostprocessTrackingMarkerOffsets",
          &dart::biomechanics::MarkerFitter::
              setPostprocessTrackingMarkerOffsets,
          ::py::arg("postprocess"),
          R"pydoc(
  If we set this to true, then after the main optimization completes we will
  do a final step to "center" the error of the tracking markers. This
  minimizes marker RMSE, but does NOT respect the weights about how far
  markers should be allowed to move.
          )pydoc")
      .def(
          "setCustomLossAndGrad",
          &dart::biomechanics::MarkerFitter::setCustomLossAndGrad,
          ::py::arg("loss"))
      .def(
          "addZeroConstraint",
          &dart::biomechanics::MarkerFitter::addZeroConstraint,
          ::py::arg("name"),
          ::py::arg("loss"))
      .def(
          "removeZeroConstraint",
          &dart::biomechanics::MarkerFitter::removeZeroConstraint,
          ::py::arg("name"))
      .def(
          "writeCSVData",
          &dart::biomechanics::MarkerFitter::writeCSVData,
          ::py::arg("path"),
          ::py::arg("init"),
          ::py::arg("rmsMarkerErrors"),
          ::py::arg("maxMarkerErrors"),
          ::py::arg("timestamps"))
      .def(
          "getInitialization",
          &dart::biomechanics::MarkerFitter::getInitialization,
          ::py::arg("markerObservations"),
          ::py::arg("newClip"),
          ::py::arg("params") = dart::biomechanics::InitialMarkerFitParams())
      .def(
          "findJointCenters",
          &dart::biomechanics::MarkerFitter::findJointCenters,
          ::py::arg("initializations"),
          ::py::arg("newClip"),
          ::py::arg("markerObservations"))
      .def(
          "optimizeBilevel",
          &dart::biomechanics::MarkerFitter::optimizeBilevel,
          ::py::arg("markerObservations"),
          ::py::arg("initialization"),
          ::py::arg("numSamples"),
          ::py::arg("applyInnerProblemGradientConstraints") = true)
      .def(
          "checkForEnoughMarkers",
          &dart::biomechanics::MarkerFitter::checkForEnoughMarkers,
          ::py::arg("markerObservations"))
      .def(
          "generateDataErrorsReport",
          &dart::biomechanics::MarkerFitter::generateDataErrorsReport,
          ::py::arg("markerObservations"),
          ::py::arg("dt"),
          ::py::arg("rippleReduce") = true,
          ::py::arg("rippleReduceUseSparse") = true,
          ::py::arg("rippleReduceUseIterativeSolver") = true,
          ::py::arg("rippleReduceSolverIterations") = 1e5)
      .def(
          "checkForFlippedMarkers",
          &dart::biomechanics::MarkerFitter::checkForFlippedMarkers,
          ::py::arg("markerObservations"),
          ::py::arg("init"),
          ::py::arg("report"))
      .def(
          "runMultiTrialKinematicsPipeline",
          &dart::biomechanics::MarkerFitter::runMultiTrialKinematicsPipeline,
          ::py::arg("markerTrials"),
          ::py::arg("params"),
          ::py::arg("numSamples") = 50)
      .def(
          "runKinematicsPipeline",
          &dart::biomechanics::MarkerFitter::runKinematicsPipeline,
          ::py::arg("markerObservations"),
          ::py::arg("newClip"),
          ::py::arg("params"),
          ::py::arg("numSamples") = 20,
          ::py::arg("skipFinalIK") = false)
      .def(
          "runPrescaledPipeline",
          &dart::biomechanics::MarkerFitter::runPrescaledPipeline,
          ::py::arg("markerObservations"),
          ::py::arg("params"))
      .def(
          "setMinJointVarianceCutoff",
          &dart::biomechanics::MarkerFitter::setMinJointVarianceCutoff,
          ::py::arg("cutoff"))
      .def(
          "setMinSphereFitScore",
          &dart::biomechanics::MarkerFitter::setMinSphereFitScore,
          ::py::arg("score"))
      .def(
          "setMinAxisFitScore",
          &dart::biomechanics::MarkerFitter::setMinAxisFitScore,
          ::py::arg("score"))
      .def(
          "setMaxJointWeight",
          &dart::biomechanics::MarkerFitter::setMaxJointWeight,
          ::py::arg("weight"))
      .def(
          "setMaxAxisWeight",
          &dart::biomechanics::MarkerFitter::setMaxAxisWeight,
          ::py::arg("weight"))
      .def(
          "setRegularizeAnatomicalMarkerOffsets",
          &dart::biomechanics::MarkerFitter::
              setRegularizeAnatomicalMarkerOffsets,
          ::py::arg("weight"))
      .def(
          "setRegularizeTrackingMarkerOffsets",
          &dart::biomechanics::MarkerFitter::setRegularizeTrackingMarkerOffsets,
          ::py::arg("weight"))
      .def(
          "setRegularizeIndividualBodyScales",
          &dart::biomechanics::MarkerFitter::setRegularizeIndividualBodyScales,
          ::py::arg("weight"))
      .def(
          "setRegularizeAllBodyScales",
          &dart::biomechanics::MarkerFitter::setRegularizeAllBodyScales,
          ::py::arg("weight"))
      .def(
          "setRegularizeJointBounds",
          &dart::biomechanics::MarkerFitter::setRegularizeJointBounds,
          ::py::arg("weight"))
      .def(
          "setDebugJointVariability",
          &dart::biomechanics::MarkerFitter::setDebugJointVariability,
          ::py::arg("debug"))
      .def(
          "setAnatomicalMarkerDefaultWeight",
          &dart::biomechanics::MarkerFitter::setAnatomicalMarkerDefaultWeight,
          ::py::arg("weight"))
      .def(
          "setTrackingMarkerDefaultWeight",
          &dart::biomechanics::MarkerFitter::setTrackingMarkerDefaultWeight,
          ::py::arg("weight"))
      .def(
          "setRegularizeJointWithVirtualSpring",
          &dart::biomechanics::MarkerFitter::
              setRegularizeJointWithVirtualSpring,
          ::py::arg("jointName"),
          ::py::arg("weight"))
      .def(
          "setRegularizePelvisJointsWithVirtualSpring",
          &dart::biomechanics::MarkerFitter::
              setRegularizePelvisJointsWithVirtualSpring,
          ::py::arg("weight"))
      .def(
          "setJointSphereFitSGDIterations",
          &dart::biomechanics::MarkerFitter::setJointSphereFitSGDIterations,
          ::py::arg("iters"))
      .def(
          "setJointAxisFitSGDIterations",
          &dart::biomechanics::MarkerFitter::setJointAxisFitSGDIterations,
          ::py::arg("iters"))
      .def(
          "debugTrajectoryAndMarkersToGUI",
          &dart::biomechanics::MarkerFitter::debugTrajectoryAndMarkersToGUI,
          ::py::arg("server"),
          ::py::arg("init"),
          ::py::arg("markerObservations"),
          ::py::arg("forcePlates") = nullptr,
          ::py::arg("goldOsim") = nullptr,
          ::py::arg("goldPoses") = Eigen::MatrixXs::Zero(0, 0))
      .def(
          "saveTrajectoryAndMarkersToGUI",
          &dart::biomechanics::MarkerFitter::saveTrajectoryAndMarkersToGUI,
          ::py::arg("path"),
          ::py::arg("init"),
          ::py::arg("markerObservations"),
          ::py::arg("accObservations"),
          ::py::arg("gyroObservations"),
          ::py::arg("frameRate"),
          ::py::arg("forcePlates") = nullptr,
          ::py::arg("goldOsim") = nullptr,
          ::py::arg("goldPoses") = Eigen::MatrixXs::Zero(0, 0))
      .def_static(
          "pickSubset",
          &dart::biomechanics::MarkerFitter::pickSubset,
          ::py::arg("markerObservations"),
          ::py::arg("subsetSize"))
      .def(
          "setMarkerIsTracking",
          &dart::biomechanics::MarkerFitter::setMarkerIsTracking,
          ::py::arg("marker"),
          ::py::arg("isTracking") = true)
      .def(
          "getMarkerIsTracking",
          &dart::biomechanics::MarkerFitter::getMarkerIsTracking,
          ::py::arg("marker"))
      .def(
          "setTriadsToTracking",
          &dart::biomechanics::MarkerFitter::setTriadsToTracking)
      .def(
          "setTrackingMarkers",
          &dart::biomechanics::MarkerFitter::setTrackingMarkers,
          ::py::arg("trackingMarkerNames"))
      .def(
          "autorotateC3D",
          &dart::biomechanics::MarkerFitter::autorotateC3D,
          ::py::arg("c3d"))
      .def("getNumMarkers", &dart::biomechanics::MarkerFitter::getNumMarkers)
      .def(
          "setImuMap",
          &dart::biomechanics::MarkerFitter::setImuMap,
          ::py::arg("imuMap"))
      .def("getImuMap", &dart::biomechanics::MarkerFitter::getImuMap)
      .def("getImuList", &dart::biomechanics::MarkerFitter::getImuList)
      .def("getImuNames", &dart::biomechanics::MarkerFitter::getImuNames)
      .def(
          "rotateIMUs",
          &dart::biomechanics::MarkerFitter::rotateIMUs,
          ::py::arg("accObservations"),
          ::py::arg("gyroObservations"),
          ::py::arg("newClip"),
          ::py::arg("init"),
          ::py::arg("dt"))
      .def(
          "measureAccelerometerRMS",
          &dart::biomechanics::MarkerFitter::measureAccelerometerRMS,
          ::py::arg("accObservations"),
          ::py::arg("newClip"),
          ::py::arg("init"),
          ::py::arg("dt"))
      .def(
          "measureGyroRMS",
          &dart::biomechanics::MarkerFitter::measureGyroRMS,
          ::py::arg("gyroObservations"),
          ::py::arg("newClip"),
          ::py::arg("init"),
          ::py::arg("dt"))
      .def(
          "getIMUFineTuneProblem",
          &dart::biomechanics::MarkerFitter::getIMUFineTuneProblem,
          ::py::arg("accObservations"),
          ::py::arg("gyroObservations"),
          ::py::arg("markerObservations"),
          ::py::arg("init"),
          ::py::arg("dt"),
          ::py::arg("start"),
          ::py::arg("end"))
      .def(
          "fineTuneWithIMU",
          &dart::biomechanics::MarkerFitter::fineTuneWithIMU,
          ::py::arg("accObservations"),
          ::py::arg("gyroObservations"),
          ::py::arg("markerObservations"),
          ::py::arg("newClip"),
          ::py::arg("init"),
          ::py::arg("dt"),
          ::py::arg("weightAccs") = 1.0,
          ::py::arg("weightGyros") = 1.0,
          ::py::arg("weightMarkers") = 100.0,
          ::py::arg("regularizePoses") = 1.0,
          ::py::arg("useIPOPT") = true,
          ::py::arg("iterations") = 300,
          ::py::arg("lbfgsMemory") = 100);
}

} // namespace python
} // namespace dart
