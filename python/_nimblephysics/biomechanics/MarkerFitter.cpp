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

  ::py::class_<dart::biomechanics::InitialMarkerFitParams>(
      m, "InitialMarkerFitParams")
      .def(::py::init<>())
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
          ::py::arg("jointCenters"))
      .def(
          "setNumBlocks",
          &dart::biomechanics::InitialMarkerFitParams::setNumBlocks,
          ::py::arg("numBlocks"))
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
          ::py::arg("jointWeights"));

  ::py::class_<
      dart::biomechanics::MarkerFitter,
      std::shared_ptr<dart::biomechanics::MarkerFitter>>(m, "MarkerFitter")
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
          "setMaxMarkerOffset",
          &dart::biomechanics::MarkerFitter::setMaxMarkerOffset,
          ::py::arg("offset"))
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
          "setDebugJointVariability",
          &dart::biomechanics::MarkerFitter::setDebugJointVariability,
          ::py::arg("debug"))
      .def(
          "debugTrajectoryAndMarkersToGUI",
          &dart::biomechanics::MarkerFitter::debugTrajectoryAndMarkersToGUI,
          ::py::arg("server"),
          ::py::arg("init"),
          ::py::arg("markerObservations"),
          ::py::arg("c3d") = nullptr,
          ::py::arg("goldOsim") = nullptr,
          ::py::arg("goldPoses") = Eigen::MatrixXs::Zero(0, 0))
      .def(
          "saveTrajectoryAndMarkersToGUI",
          &dart::biomechanics::MarkerFitter::saveTrajectoryAndMarkersToGUI,
          ::py::arg("path"),
          ::py::arg("init"),
          ::py::arg("markerObservations"),
          ::py::arg("c3d") = nullptr,
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
      .def("getNumMarkers", &dart::biomechanics::MarkerFitter::getNumMarkers);
}

} // namespace python
} // namespace dart
