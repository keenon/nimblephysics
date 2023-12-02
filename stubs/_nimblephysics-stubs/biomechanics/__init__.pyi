"""This provides biomechanics utilities in Nimble, including inverse dynamics and (eventually) mocap support and muscle estimation."""
from __future__ import annotations
import nimblephysics_libs._nimblephysics.biomechanics
import typing
import nimblephysics_libs._nimblephysics.dynamics
import nimblephysics_libs._nimblephysics.math
import nimblephysics_libs._nimblephysics.neural
import nimblephysics_libs._nimblephysics.server
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "Anthropometrics",
    "BatchGaitInverseDynamics",
    "BilevelFitResult",
    "C3D",
    "C3DLoader",
    "ContactRegimeSection",
    "DynamicsFitProblemConfig",
    "DynamicsFitter",
    "DynamicsInitialization",
    "ForcePlate",
    "Frame",
    "FrameList",
    "FramePass",
    "FramePassList",
    "IKErrorReport",
    "IMUFineTuneProblem",
    "InitialMarkerFitParams",
    "LabelledMarkers",
    "LilypadSolver",
    "MarkerFitter",
    "MarkerFitterState",
    "MarkerFixer",
    "MarkerInitialization",
    "MarkerLabeller",
    "MarkerLabellerMock",
    "MarkerTrace",
    "MarkersErrorReport",
    "MissingGRFReason",
    "MissingGRFStatus",
    "NeuralMarkerLabeller",
    "OpenSimFile",
    "OpenSimMocoTrajectory",
    "OpenSimMot",
    "OpenSimParser",
    "OpenSimScaleAndMarkerOffsets",
    "OpenSimTRC",
    "ProcessingPassType",
    "ResidualForceHelper",
    "SkeletonConverter",
    "SubjectOnDisk",
    "SubjectOnDiskHeader",
    "SubjectOnDiskPassHeader",
    "SubjectOnDiskTrial",
    "SubjectOnDiskTrialPass",
    "forceDiscrepancy",
    "interpolatedClippedGRF",
    "manualReview",
    "measuredGrfZeroWhenAccelerationNonZero",
    "missingBlip",
    "missingImpact",
    "no",
    "notMissingGRF",
    "notOverForcePlate",
    "shiftGRF",
    "torqueDiscrepancy",
    "unknown",
    "unmeasuredExternalForceDetected",
    "yes"
]


class Anthropometrics():
    def addMetric(self, name: str, bodyPose: numpy.ndarray[numpy.float64, _Shape[m, 1]], bodyA: str, offsetA: numpy.ndarray[numpy.float64, _Shape[3, 1]], bodyB: str, offsetB: numpy.ndarray[numpy.float64, _Shape[3, 1]], axis: numpy.ndarray[numpy.float64, _Shape[3, 1]] = array([0., 0., 0.])) -> None: ...
    def condition(self, observedValues: typing.Dict[str, float]) -> Anthropometrics: ...
    def debugToGUI(self, server: nimblephysics_libs._nimblephysics.server.GUIWebsocketServer, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None: ...
    def getDistribution(self) -> nimblephysics_libs._nimblephysics.math.MultivariateGaussian: ...
    def getGradientOfLogPDFWrtBodyScales(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getGradientOfLogPDFWrtGroupScales(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getLogPDF(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, normalized: bool = True) -> float: ...
    def getMetricNames(self) -> typing.List[str]: ...
    def getPDF(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> float: ...
    @staticmethod
    def loadFromFile(uri: str) -> Anthropometrics: ...
    def measure(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> typing.Dict[str, float]: ...
    def setDistribution(self, dist: nimblephysics_libs._nimblephysics.math.MultivariateGaussian) -> None: ...
    pass
class BatchGaitInverseDynamics():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, poses: numpy.ndarray[numpy.float64, _Shape[m, n]], groundContactBodies: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], groundNormal: numpy.ndarray[numpy.float64, _Shape[3, 1]], tileSize: float, maxSectionLength: int = 100, smoothingWeight: float = 1.0, minTorqueWeight: float = 1.0, prevContactWeight: float = 0.1, blendWeight: float = 1.0, blendSteepness: float = 10.0) -> None: ...
    def debugLilypadToGUI(self, gui: nimblephysics_libs._nimblephysics.server.GUIWebsocketServer) -> None: ...
    def debugTimestepToGUI(self, gui: nimblephysics_libs._nimblephysics.server.GUIWebsocketServer, timesteps: int) -> None: ...
    def getContactBodiesAtTimestep(self, timestep: int) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]: ...
    def getContactWrenchesAtTimestep(self, timestep: int) -> typing.List[numpy.ndarray[numpy.float64, _Shape[6, 1]]]: ...
    def getSectionForTimestep(self, timestep: int) -> ContactRegimeSection: ...
    def numTimesteps(self) -> int: ...
    pass
class BilevelFitResult():
    @property
    def groupScales(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @groupScales.setter
    def groupScales(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def markerOffsets(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @markerOffsets.setter
    def markerOffsets(self, arg0: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def poses(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]
        """
    @poses.setter
    def poses(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]) -> None:
        pass
    @property
    def posesMatrix(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @posesMatrix.setter
    def posesMatrix(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def rawMarkerOffsets(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @rawMarkerOffsets.setter
    def rawMarkerOffsets(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def success(self) -> bool:
        """
        :type: bool
        """
    @success.setter
    def success(self, arg0: bool) -> None:
        pass
    pass
class C3D():
    @property
    def dataRotation(self) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 3]]
        """
    @dataRotation.setter
    def dataRotation(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> None:
        pass
    @property
    def forcePlates(self) -> typing.List[ForcePlate]:
        """
        :type: typing.List[ForcePlate]
        """
    @forcePlates.setter
    def forcePlates(self, arg0: typing.List[ForcePlate]) -> None:
        pass
    @property
    def framesPerSecond(self) -> int:
        """
        :type: int
        """
    @framesPerSecond.setter
    def framesPerSecond(self, arg0: int) -> None:
        pass
    @property
    def markerTimesteps(self) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markerTimesteps.setter
    def markerTimesteps(self, arg0: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def markers(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @markers.setter
    def markers(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def shuffledMarkersMatrix(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @shuffledMarkersMatrix.setter
    def shuffledMarkersMatrix(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def shuffledMarkersMatrixMask(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @shuffledMarkersMatrixMask.setter
    def shuffledMarkersMatrixMask(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def timestamps(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @timestamps.setter
    def timestamps(self, arg0: typing.List[float]) -> None:
        pass
    pass
class C3DLoader():
    @staticmethod
    def debugToGUI(file: C3D, server: nimblephysics_libs._nimblephysics.server.GUIWebsocketServer) -> None: ...
    @staticmethod
    def fixupMarkerFlips(c3d: C3D) -> typing.List[typing.List[typing.Tuple[str, str]]]: ...
    @staticmethod
    def loadC3D(uri: str) -> C3D: ...
    pass
class ContactRegimeSection():
    @property
    def endTime(self) -> int:
        """
        :type: int
        """
    @endTime.setter
    def endTime(self, arg0: int) -> None:
        pass
    @property
    def groundContactBodies(self) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]:
        """
        :type: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]
        """
    @groundContactBodies.setter
    def groundContactBodies(self, arg0: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]) -> None:
        pass
    @property
    def startTime(self) -> int:
        """
        :type: int
        """
    @startTime.setter
    def startTime(self, arg0: int) -> None:
        pass
    @property
    def wrenches(self) -> typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[6, 1]]]]:
        """
        :type: typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[6, 1]]]]
        """
    @wrenches.setter
    def wrenches(self, arg0: typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[6, 1]]]]) -> None:
        pass
    pass
class DynamicsFitProblemConfig():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None: ...
    def setBoundMoveDistance(self, distance: float) -> DynamicsFitProblemConfig: ...
    def setConstrainAngularResiduals(self, value: float) -> DynamicsFitProblemConfig: ...
    def setConstrainLinearResiduals(self, value: float) -> DynamicsFitProblemConfig: ...
    def setConstrainResidualsZero(self, constrain: bool) -> DynamicsFitProblemConfig: ...
    def setDefaults(self, useL1: bool = True) -> DynamicsFitProblemConfig: ...
    def setDisableBounds(self, disable: bool) -> DynamicsFitProblemConfig: ...
    def setIncludeBodyScales(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setIncludeCOMs(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setIncludeInertias(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setIncludeMarkerOffsets(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setIncludeMasses(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setIncludePoses(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setJointWeight(self, value: float) -> DynamicsFitProblemConfig: ...
    def setLinearNewtonUseL1(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setLinearNewtonWeight(self, value: float) -> DynamicsFitProblemConfig: ...
    def setLogLossDetails(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setMarkerUseL1(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setMarkerWeight(self, value: float) -> DynamicsFitProblemConfig: ...
    def setMaxBlockSize(self, value: int) -> DynamicsFitProblemConfig: ...
    def setMaxNumBlocksPerTrial(self, value: int) -> DynamicsFitProblemConfig: ...
    def setMaxNumTrials(self, value: int) -> DynamicsFitProblemConfig: ...
    def setNumThreads(self, value: int) -> DynamicsFitProblemConfig: ...
    def setOnlyOneTrial(self, value: int) -> DynamicsFitProblemConfig: ...
    def setPoseSubsetLen(self, value: int) -> DynamicsFitProblemConfig: ...
    def setPoseSubsetStartIndex(self, value: int) -> DynamicsFitProblemConfig: ...
    def setRegularizeAnatomicalMarkerOffsets(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeBodyScales(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeCOMs(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeImpliedDensity(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeInertias(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeJointAcc(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeMasses(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizePoses(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeSpatialAcc(self, value: float) -> DynamicsFitProblemConfig: ...
    def setRegularizeSpatialAccBodyWeights(self, bodyWeights: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> DynamicsFitProblemConfig: ...
    def setRegularizeSpatialAccUseL1(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setRegularizeTrackingMarkerOffsets(self, value: float) -> DynamicsFitProblemConfig: ...
    def setResidualTorqueMultiple(self, value: float) -> DynamicsFitProblemConfig: ...
    def setResidualUseL1(self, value: bool) -> DynamicsFitProblemConfig: ...
    def setResidualWeight(self, value: float) -> DynamicsFitProblemConfig: ...
    pass
class DynamicsFitter():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, footNodes: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], trackingMarkers: typing.List[str]) -> None: ...
    def addJointBoundSlack(self, init: nimblephysics_libs._nimblephysics.dynamics.Skeleton, slack: float) -> None: ...
    def applyInitToSkeleton(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, init: DynamicsInitialization) -> None: ...
    def boundPush(self, init: DynamicsInitialization, boundPush: float = 0.02) -> None: ...
    def checkPhysicalConsistency(self, init: DynamicsInitialization, maxAcceptableErrors: float = 0.001, maxTimestepsToTest: int = 50) -> bool: ...
    def comAccelerations(self, init: DynamicsInitialization, trial: int) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]: ...
    def comPositions(self, init: DynamicsInitialization, trial: int) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]: ...
    def computeAverageCOPChange(self, init: DynamicsInitialization) -> float: ...
    def computeAverageForceMagnitudeChange(self, init: DynamicsInitialization) -> float: ...
    def computeAverageForceVectorChange(self, init: DynamicsInitialization) -> float: ...
    def computeAverageMarkerMaxError(self, init: DynamicsInitialization) -> float: ...
    def computeAverageMarkerRMSE(self, init: DynamicsInitialization) -> float: ...
    def computeAverageRealForce(self, init: DynamicsInitialization) -> typing.Tuple[float, float]: ...
    def computeAverageResidualForce(self, init: DynamicsInitialization) -> typing.Tuple[float, float]: ...
    def computeAverageTrialCOPChange(self, init: DynamicsInitialization, trial: int) -> float: ...
    def computeAverageTrialForceMagnitudeChange(self, init: DynamicsInitialization, trial: int) -> float: ...
    def computeAverageTrialForceVectorChange(self, init: DynamicsInitialization, trial: int) -> float: ...
    def computeAverageTrialMarkerMaxError(self, init: DynamicsInitialization, trial: int) -> float: ...
    def computeAverageTrialMarkerRMSE(self, init: DynamicsInitialization, trial: int) -> float: ...
    def computeAverageTrialRealForce(self, init: DynamicsInitialization, trial: int) -> typing.Tuple[float, float]: ...
    def computeAverageTrialResidualForce(self, init: DynamicsInitialization, trial: int) -> typing.Tuple[float, float]: ...
    def computeInverseDynamics(self, init: DynamicsInitialization, trial: int) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def computePerfectGRFs(self, init: DynamicsInitialization) -> None: ...
    @staticmethod
    @typing.overload
    def createInitialization(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, markerMap: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], trackingMarkers: typing.List[str], grfNodes: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], forcePlateTrials: typing.List[typing.List[ForcePlate]], poseTrials: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]], framesPerSecond: typing.List[int], markerObservationTrials: typing.List[typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]], overrideForcePlateToGRFNodeAssignment: typing.List[typing.List[int]] = [], initializedProbablyMissingGRF: typing.List[typing.List[MissingGRFStatus]] = []) -> DynamicsInitialization: ...
    @staticmethod
    @typing.overload
    def createInitialization(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, kinematicInits: typing.List[MarkerInitialization], trackingMarkers: typing.List[str], grfNodes: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], forcePlateTrials: typing.List[typing.List[ForcePlate]], framesPerSecond: typing.List[int], markerObservationTrials: typing.List[typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]], overrideForcePlateToGRFNodeAssignment: typing.List[typing.List[int]] = [], initializedProbablyMissingGRF: typing.List[typing.List[MissingGRFStatus]] = []) -> DynamicsInitialization: ...
    def estimateFootGroundContactsWithHeightHeuristic(self, init: DynamicsInitialization, ignoreFootNotOverForcePlate: bool = False) -> None: ...
    def estimateFootGroundContactsWithStillness(self, init: DynamicsInitialization, radius: float = 0.05, minTime: float = 0.5) -> None: ...
    def estimateLinkMassesFromAcceleration(self, init: DynamicsInitialization, regularizationWeight: float = 50.0) -> None: ...
    def impliedCOMForces(self, init: DynamicsInitialization, trial: int, includeGravity: numpy.ndarray[numpy.float64, _Shape[3, 1]] = True) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]: ...
    def measuredGRFForces(self, init: DynamicsInitialization, trial: int) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]: ...
    def multimassZeroLinearResidualsOnCOMTrajectory(self, init: DynamicsInitialization, maxTrialsToSolveMassOver: int = 4, boundPush: float = 0.01) -> None: ...
    def optimizeMarkerOffsets(self, init: DynamicsInitialization, reoptimizeAnatomicalMarkers: bool = False, reoptimizeTrackingMarkers: bool = True) -> None: ...
    def optimizeSpatialResidualsOnCOMTrajectory(self, init: DynamicsInitialization, trial: int, satisfactoryThreshold: float = 1e-05, numIters: int = 600, missingResidualRegularization: float = 1000, weightAngular: float = 2.0, weightLastFewTimesteps: float = 5.0, offsetRegularization: float = 0.001, regularizeResiduals: bool = True) -> bool: ...
    def recalibrateForcePlates(self, init: DynamicsInitialization, trial: int, maxMovement: float = 0.03) -> None: ...
    def runConstrainedSGDOptimization(self, init: DynamicsInitialization, config: DynamicsFitProblemConfig) -> None: ...
    def runIPOPTOptimization(self, init: DynamicsInitialization, config: DynamicsFitProblemConfig) -> None: ...
    def runUnconstrainedSGDOptimization(self, init: DynamicsInitialization, config: DynamicsFitProblemConfig) -> None: ...
    def saveDynamicsToGUI(self, path: str, init: DynamicsInitialization, trialIndex: int, framesPerSecond: int) -> None: ...
    def scaleLinkMassesFromGravity(self, init: DynamicsInitialization) -> None: ...
    def setCOMHistogramBuckets(self, buckets: int) -> None: ...
    def setCOMHistogramClipBuckets(self, clipBuckets: int) -> None: ...
    def setCOMHistogramMaxMovement(self, maxMovement: float) -> None: ...
    def setCheckDerivatives(self, value: bool) -> None: ...
    def setDisableLinesearch(self, value: bool) -> None: ...
    def setFillInEndFramesGrfGaps(self, fillInFrames: int) -> None: ...
    def setIterationLimit(self, value: int) -> None: ...
    def setLBFGSHistoryLength(self, value: int) -> None: ...
    def setPrintFrequency(self, value: int) -> None: ...
    def setSilenceOutput(self, value: bool) -> None: ...
    def setTolerance(self, value: float) -> None: ...
    def smoothAccelerations(self, init: DynamicsInitialization, smoothingWeight: float = 10.0, regularizationWeight: float = 0.001) -> None: ...
    def timeSyncAndInitializePipeline(self, init: DynamicsInitialization, useReactionWheels: bool = False, shiftGRF: bool = False, maxShiftGRF: int = 4, iterationsPerShift: int = 20, maxTrialsToSolveMassOver: int = 4, weightLinear: float = 1.0, weightAngular: float = 0.5, regularizeLinearResiduals: float = 0.1, regularizeAngularResiduals: float = 0.1, regularizeCopDriftCompensation: float = 1.0, maxBuckets: int = 100, detectUnmeasuredTorque: bool = True, avgPositionChangeThreshold: float = 0.08, avgAngularChangeThreshold: float = 0.15, reoptimizeAnatomicalMarkers: bool = False, reoptimizeTrackingMarkers: bool = True, tuneLinkMasses: bool = False) -> bool: ...
    def timeSyncTrialGRF(self, init: DynamicsInitialization, trial: int, useReactionWheels: bool = False, maxShiftGRF: int = 4, iterationsPerShift: int = 20, weightLinear: float = 1.0, weightAngular: float = 1.0, regularizeLinearResiduals: float = 0.5, regularizeAngularResiduals: float = 0.5, regularizeCopDriftCompensation: float = 1.0, maxBuckets: int = 20) -> bool: ...
    def writeCSVData(self, path: str, init: DynamicsInitialization, trialIndex: int, useAdjustedGRFs: bool = False, timestamps: typing.List[float] = []) -> None: ...
    def zeroLinearResidualsAndOptimizeAngular(self, init: DynamicsInitialization, trial: int, targetPoses: numpy.ndarray[numpy.float64, _Shape[m, n]], previousTotalResidual: float, iteration: int, useReactionWheels: bool = False, weightLinear: float = 1.0, weightAngular: float = 0.5, regularizeLinearResiduals: float = 0.1, regularizeAngularResiduals: float = 0.1, regularizeCopDriftCompensation: float = 1.0, maxBuckets: int = 40, maxLeastSquaresIters: int = 200, commitCopDriftCompensation: bool = False, detectUnmeasuredTorque: bool = True, avgPositionChangeThreshold: float = 0.08, avgAngularChangeThreshold: float = 0.15) -> typing.Tuple[bool, float]: ...
    def zeroLinearResidualsOnCOMTrajectory(self, init: DynamicsInitialization, maxTrialsToSolveMassOver: int = 4, detectExternalForce: bool = True, driftCorrectionBlurRadius: int = 250, driftCorrectionBlurInterval: int = 250) -> bool: ...
    pass
class DynamicsInitialization():
    @property
    def axisWeights(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]
        """
    @axisWeights.setter
    def axisWeights(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]) -> None:
        pass
    @property
    def bodyCom(self) -> numpy.ndarray[numpy.float64, _Shape[3, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, n]]
        """
    @bodyCom.setter
    def bodyCom(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, n]]) -> None:
        pass
    @property
    def bodyInertia(self) -> numpy.ndarray[numpy.float64, _Shape[6, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[6, n]]
        """
    @bodyInertia.setter
    def bodyInertia(self, arg0: numpy.ndarray[numpy.float64, _Shape[6, n]]) -> None:
        pass
    @property
    def bodyMasses(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @bodyMasses.setter
    def bodyMasses(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def contactBodies(self) -> typing.List[typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]]:
        """
        :type: typing.List[typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]]
        """
    @contactBodies.setter
    def contactBodies(self, arg0: typing.List[typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]]) -> None:
        pass
    @property
    def defaultForcePlateCorners(self) -> typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @defaultForcePlateCorners.setter
    def defaultForcePlateCorners(self, arg0: typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def forcePlateTrials(self) -> typing.List[typing.List[ForcePlate]]:
        """
        :type: typing.List[typing.List[ForcePlate]]
        """
    @forcePlateTrials.setter
    def forcePlateTrials(self, arg0: typing.List[typing.List[ForcePlate]]) -> None:
        pass
    @property
    def grfBodyContactSphereRadius(self) -> typing.List[typing.List[typing.List[float]]]:
        """
        :type: typing.List[typing.List[typing.List[float]]]
        """
    @grfBodyContactSphereRadius.setter
    def grfBodyContactSphereRadius(self, arg0: typing.List[typing.List[typing.List[float]]]) -> None:
        pass
    @property
    def grfBodyForceActive(self) -> typing.List[typing.List[typing.List[bool]]]:
        """
        :type: typing.List[typing.List[typing.List[bool]]]
        """
    @grfBodyForceActive.setter
    def grfBodyForceActive(self, arg0: typing.List[typing.List[typing.List[bool]]]) -> None:
        pass
    @property
    def grfBodyIndices(self) -> typing.List[int]:
        """
        :type: typing.List[int]
        """
    @grfBodyIndices.setter
    def grfBodyIndices(self, arg0: typing.List[int]) -> None:
        pass
    @property
    def grfBodyNodes(self) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]:
        """
        :type: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]
        """
    @grfBodyNodes.setter
    def grfBodyNodes(self, arg0: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]) -> None:
        pass
    @property
    def grfBodyOffForcePlate(self) -> typing.List[typing.List[typing.List[bool]]]:
        """
        :type: typing.List[typing.List[typing.List[bool]]]
        """
    @grfBodyOffForcePlate.setter
    def grfBodyOffForcePlate(self, arg0: typing.List[typing.List[typing.List[bool]]]) -> None:
        pass
    @property
    def grfBodySphereInContact(self) -> typing.List[typing.List[typing.List[bool]]]:
        """
        :type: typing.List[typing.List[typing.List[bool]]]
        """
    @grfBodySphereInContact.setter
    def grfBodySphereInContact(self, arg0: typing.List[typing.List[typing.List[bool]]]) -> None:
        pass
    @property
    def grfTrials(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    @grfTrials.setter
    def grfTrials(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None:
        pass
    @property
    def groupMasses(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @groupMasses.setter
    def groupMasses(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def groupScales(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @groupScales.setter
    def groupScales(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def includeTrialsInDynamicsFit(self) -> typing.List[bool]:
        """
        :type: typing.List[bool]
        """
    @includeTrialsInDynamicsFit.setter
    def includeTrialsInDynamicsFit(self, arg0: typing.List[bool]) -> None:
        pass
    @property
    def initialGroupCOMs(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @initialGroupCOMs.setter
    def initialGroupCOMs(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def initialGroupInertias(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @initialGroupInertias.setter
    def initialGroupInertias(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def initialGroupMasses(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @initialGroupMasses.setter
    def initialGroupMasses(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def initialGroupScales(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @initialGroupScales.setter
    def initialGroupScales(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def initialMarkerOffsets(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @initialMarkerOffsets.setter
    def initialMarkerOffsets(self, arg0: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def jointAxis(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    @jointAxis.setter
    def jointAxis(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None:
        pass
    @property
    def jointCenters(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    @jointCenters.setter
    def jointCenters(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None:
        pass
    @property
    def jointWeights(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]
        """
    @jointWeights.setter
    def jointWeights(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, 1]]]) -> None:
        pass
    @property
    def joints(self) -> typing.List[typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]]:
        """
        :type: typing.List[typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]]
        """
    @joints.setter
    def joints(self, arg0: typing.List[typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]]) -> None:
        pass
    @property
    def jointsAdjacentMarkers(self) -> typing.List[typing.List[typing.List[str]]]:
        """
        :type: typing.List[typing.List[typing.List[str]]]
        """
    @jointsAdjacentMarkers.setter
    def jointsAdjacentMarkers(self, arg0: typing.List[typing.List[typing.List[str]]]) -> None:
        pass
    @property
    def markerObservationTrials(self) -> typing.List[typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]]:
        """
        :type: typing.List[typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]]
        """
    @markerObservationTrials.setter
    def markerObservationTrials(self, arg0: typing.List[typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]]) -> None:
        pass
    @property
    def markerOffsets(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @markerOffsets.setter
    def markerOffsets(self, arg0: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def missingGRFReason(self) -> typing.List[typing.List[MissingGRFReason]]:
        """
        :type: typing.List[typing.List[MissingGRFReason]]
        """
    @missingGRFReason.setter
    def missingGRFReason(self, arg0: typing.List[typing.List[MissingGRFReason]]) -> None:
        pass
    @property
    def originalPoses(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    @originalPoses.setter
    def originalPoses(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None:
        pass
    @property
    def poseTrials(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    @poseTrials.setter
    def poseTrials(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None:
        pass
    @property
    def probablyMissingGRF(self) -> typing.List[typing.List[MissingGRFStatus]]:
        """
        :type: typing.List[typing.List[MissingGRFStatus]]
        """
    @probablyMissingGRF.setter
    def probablyMissingGRF(self, arg0: typing.List[typing.List[MissingGRFStatus]]) -> None:
        pass
    @property
    def regularizeGroupCOMsTo(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @regularizeGroupCOMsTo.setter
    def regularizeGroupCOMsTo(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def regularizeGroupInertiasTo(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @regularizeGroupInertiasTo.setter
    def regularizeGroupInertiasTo(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def regularizeGroupMassesTo(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @regularizeGroupMassesTo.setter
    def regularizeGroupMassesTo(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def regularizeGroupScalesTo(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @regularizeGroupScalesTo.setter
    def regularizeGroupScalesTo(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def regularizeMarkerOffsetsTo(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @regularizeMarkerOffsetsTo.setter
    def regularizeMarkerOffsetsTo(self, arg0: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def trackingMarkers(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @trackingMarkers.setter
    def trackingMarkers(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def trialTimesteps(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @trialTimesteps.setter
    def trialTimesteps(self, arg0: typing.List[float]) -> None:
        pass
    @property
    def trialsOnTreadmill(self) -> typing.List[bool]:
        """
        :type: typing.List[bool]
        """
    @trialsOnTreadmill.setter
    def trialsOnTreadmill(self, arg0: typing.List[bool]) -> None:
        pass
    @property
    def updatedMarkerMap(self) -> typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @updatedMarkerMap.setter
    def updatedMarkerMap(self, arg0: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    pass
class ForcePlate():
    def __init__(self) -> None: ...
    def autodetectNoiseThresholdAndClip(self, percentOfMaxToDetectThumb: float = 0.25, percentOfMaxToCheckThumbRightEdge: float = 0.35) -> None: ...
    @staticmethod
    def copyForcePlate(plate: ForcePlate) -> ForcePlate: ...
    def detectAndFixCopMomentConvention(self, trial: int = -1, i: int = -1) -> None: ...
    def getResamplingMatrixAndGroundHeights(self) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, n]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: ...
    def setResamplingMatrixAndGroundHeights(self, matrix: numpy.ndarray[numpy.float64, _Shape[m, n]], groundHeights: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def trim(self, newStartTime: float, newEndTime: float) -> None: ...
    def trimToIndexes(self, start: int, end: int) -> None: ...
    @property
    def centersOfPressure(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @centersOfPressure.setter
    def centersOfPressure(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def corners(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @corners.setter
    def corners(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def forces(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @forces.setter
    def forces(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def moments(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @moments.setter
    def moments(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def timestamps(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @timestamps.setter
    def timestamps(self, arg0: typing.List[float]) -> None:
        pass
    @property
    def worldOrigin(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @worldOrigin.setter
    def worldOrigin(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    pass
class Frame():
    """
    This is for doing ML and large-scale data analysis. This is a single frame of data, returned in a list by :code:`SubjectOnDisk.readFrames()`, which contains everything needed to reconstruct all the dynamics of a snapshot in time.
    """
    @property
    def accObservations(self) -> typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the accelerometers observations at this frame. Accelerometers that were not observed (perhaps due to time offsets in uploaded data) will not be present in this list. For the full specification of the accelerometer set, load the model from the :code:`SubjectOnDisk`

        :type: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @accObservations.setter
    def accObservations(self, arg0: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the accelerometers observations at this frame. Accelerometers that were not observed (perhaps due to time offsets in uploaded data) will not be present in this list. For the full specification of the accelerometer set, load the model from the :code:`SubjectOnDisk`
        """
    @property
    def customValues(self) -> typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[m, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of unspecified values. The idea here is to allow the format to be easily extensible with unusual data (for example, exoskeleton torques) without bloating ordinary training files.

        :type: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[m, 1]]]]
        """
    @customValues.setter
    def customValues(self, arg0: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[m, 1]]]]) -> None:
        """
        This is list of :code:`Pair[str, np.ndarray]` of unspecified values. The idea here is to allow the format to be easily extensible with unusual data (for example, exoskeleton torques) without bloating ordinary training files.
        """
    @property
    def emgSignals(self) -> typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[m, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the EMG signals at this frame. EMG signals are generally preserved at a higher sampling frequency than the motion capture, so the `np.ndarray` vector will be a number of samples that were captured during this single motion capture frame. For example, if EMG is at 1000Hz and mocap is at 100Hz, the `np.ndarray` vector will be of length 10.

        :type: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[m, 1]]]]
        """
    @emgSignals.setter
    def emgSignals(self, arg0: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[m, 1]]]]) -> None:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the EMG signals at this frame. EMG signals are generally preserved at a higher sampling frequency than the motion capture, so the `np.ndarray` vector will be a number of samples that were captured during this single motion capture frame. For example, if EMG is at 1000Hz and mocap is at 100Hz, the `np.ndarray` vector will be of length 10.
        """
    @property
    def exoTorques(self) -> typing.List[typing.Tuple[int, float]]:
        """
        This is list of :code:`Pair[int, np.ndarray]` of the DOF indices that are actuated by exoskeletons, and the torques on those DOFs.

        :type: typing.List[typing.Tuple[int, float]]
        """
    @exoTorques.setter
    def exoTorques(self, arg0: typing.List[typing.Tuple[int, float]]) -> None:
        """
        This is list of :code:`Pair[int, np.ndarray]` of the DOF indices that are actuated by exoskeletons, and the torques on those DOFs.
        """
    @property
    def gyroObservations(self) -> typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the gyroscope observations at this frame. Gyroscopes that were not observed (perhaps due to time offsets in uploaded data) will not be present in this list. For the full specification of the gyroscope set, load the model from the :code:`SubjectOnDisk`

        :type: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @gyroObservations.setter
    def gyroObservations(self, arg0: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the gyroscope observations at this frame. Gyroscopes that were not observed (perhaps due to time offsets in uploaded data) will not be present in this list. For the full specification of the gyroscope set, load the model from the :code:`SubjectOnDisk`
        """
    @property
    def markerObservations(self) -> typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the marker observations at this frame. Markers that were not observed will not be present in this list. For the full specification of the markerset, load the model from the :code:`SubjectOnDisk`

        :type: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markerObservations.setter
    def markerObservations(self, arg0: typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the marker observations at this frame. Markers that were not observed will not be present in this list. For the full specification of the markerset, load the model from the :code:`SubjectOnDisk`
        """
    @property
    def missingGRFReason(self) -> MissingGRFReason:
        """
                    This is the reason that this frame is missing GRF, or else is the flag notMissingGRF to indicate that this frame has physics.

                    WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on this frame!!
                  

        :type: MissingGRFReason
        """
    @missingGRFReason.setter
    def missingGRFReason(self, arg0: MissingGRFReason) -> None:
        """
        This is the reason that this frame is missing GRF, or else is the flag notMissingGRF to indicate that this frame has physics.

        WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on this frame!!
        """
    @property
    def processingPasses(self) -> FramePassList:
        """
                    The processing passes that were done on this Frame. For example, if we solved for kinematics, then dynamics, 
                    then low pass filtered, this will have 3 entries.
                        

        :type: FramePassList
        """
    @processingPasses.setter
    def processingPasses(self, arg0: FramePassList) -> None:
        """
        The processing passes that were done on this Frame. For example, if we solved for kinematics, then dynamics, 
        then low pass filtered, this will have 3 entries.
            
        """
    @property
    def rawForcePlateCenterOfPressures(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        This is list of :code:`np.ndarray` of the original center of pressure readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.

        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @rawForcePlateCenterOfPressures.setter
    def rawForcePlateCenterOfPressures(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        """
        This is list of :code:`np.ndarray` of the original center of pressure readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.
        """
    @property
    def rawForcePlateForces(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        This is list of :code:`np.ndarray` of the original force readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.

        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @rawForcePlateForces.setter
    def rawForcePlateForces(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        """
        This is list of :code:`np.ndarray` of the original force readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.
        """
    @property
    def rawForcePlateTorques(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        This is list of :code:`np.ndarray` of the original torque readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.

        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @rawForcePlateTorques.setter
    def rawForcePlateTorques(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        """
        This is list of :code:`np.ndarray` of the original torque readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.
        """
    @property
    def t(self) -> int:
        """
        The frame number in this trial.

        :type: int
        """
    @t.setter
    def t(self, arg0: int) -> None:
        """
        The frame number in this trial.
        """
    @property
    def trial(self) -> int:
        """
        The index of the trial in the containing SubjectOnDisk.

        :type: int
        """
    @trial.setter
    def trial(self, arg0: int) -> None:
        """
        The index of the trial in the containing SubjectOnDisk.
        """
    pass
class FrameList():
    def __bool__(self) -> bool: 
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: Frame) -> bool: 
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None: 
        """
        Delete the list elements at index ``i``

        Delete list elements using a slice object
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None: ...
    def __eq__(self, arg0: FrameList) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> FrameList: 
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> Frame: ...
    @typing.overload
    def __init__(self) -> None: 
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: FrameList) -> None: ...
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: FrameList) -> bool: ...
    def __repr__(self) -> str: 
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: Frame) -> None: 
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: FrameList) -> None: ...
    def append(self, x: Frame) -> None: 
        """
        Add an item to the end of the list
        """
    def clear(self) -> None: 
        """
        Clear the contents
        """
    def count(self, x: Frame) -> int: 
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: FrameList) -> None: 
        """
        Extend the list by appending all the items in the given list

        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None: ...
    def insert(self, i: int, x: Frame) -> None: 
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> Frame: 
        """
        Remove and return the last item

        Remove and return the item at index ``i``
        """
    @typing.overload
    def pop(self, i: int) -> Frame: ...
    def remove(self, x: Frame) -> None: 
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    __hash__: typing.ClassVar[None] = None
    pass
class FramePass():
    """
    This is for doing ML and large-scale data analysis. This is a single processing pass on a single frame of data, returned from a list within a :code:`nimblephysics.biomechanics.Frame` (which can be got with :code:`SubjectOnDisk.readFrames()`), which contains the full reconstruction of your subject at this instant created by this processing pass. Earlier processing passes are likely to have more discrepancies with the original data, bet later processing passes require more types of sensor signals that may not always be available.
    """
    @property
    def acc(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        The joint accelerations on this frame.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def accFiniteDifferenced(self) -> numpy.ndarray[numpy.int32, _Shape[m, 1]]:
        """
        A boolean mask of [0,1]s for each DOF, with a 1 indicating that this DOF got its acceleration through finite differencing, and therefore may be somewhat unreliable

        :type: numpy.ndarray[numpy.int32, _Shape[m, 1]]
        """
    @property
    def angularResidual(self) -> float:
        """
        A scalar giving how much angular torque, in Newton-meters, would need to be applied at the root of the skeleton in order to enable the skeleton's observed accelerations (given positions and velocities) on this frame.

        :type: float
        """
    @property
    def comAcc(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        The acceleration of the COM, in world space

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def comAccInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        This is the acceleration of the center of mass of the subject, expressed in the root body frame (which probably means expressed in pelvis coordinates, though some skeletons may use a different body as the root, for instance the torso).

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def comPos(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        The position of the COM, in world space

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def comVel(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        The velocity of the COM, in world space

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def contact(self) -> numpy.ndarray[numpy.int32, _Shape[m, 1]]:
        """
        A vector of [0,1] booleans for if a body is in contact with the ground.

        :type: numpy.ndarray[numpy.int32, _Shape[m, 1]]
        """
    @property
    def groundContactCenterOfPressure(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`CoP` values for each contact body, where :code:`CoP` is a 3 vector representing the center of pressure for a contact measured on the force plate. :code:`CoP` is 
                    expressed in the world frame.
                

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def groundContactCenterOfPressureInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`CoP` values for each contact body, where :code:`CoP` is a 3 vector representing the center of pressure for a contact measured on the force plate. :code:`CoP` is 
                    expressed in the root frame, which is a frame that is rigidly attached to the root body of the skeleton (probably the pelvis).
                

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def groundContactForce(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`f` values for each contact body, where :code:`f` is a 3 vector representing the ground-reaction force from a contact, measured on the force plate. :code:`f` is 
                    expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
                  

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def groundContactForceInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`f` values for each contact body, where :code:`f` is a 3 vector representing the ground-reaction force from a contact, measured on the force plate. :code:`f` is 
                    expressed in the root frame, which is a frame that is rigidly attached to the root body of the skeleton (probably the pelvis), and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
                  

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def groundContactTorque(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`tau` values for each contact body, where :code:`tau` is a 3 vector representing the ground-reaction torque from a contact, measured on the force plate. :code:`tau` is 
                    expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
                  

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def groundContactTorqueInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`tau` values for each contact body, where :code:`tau` is a 3 vector representing the ground-reaction torque from a contact, measured on the force plate. :code:`tau` is 
                    expressed in the root frame, which is a frame that is rigidly attached to the root body of the skeleton (probably the pelvis), and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
                  

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def groundContactWrenches(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
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

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def groundContactWrenchesInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        These are the wrenches (each vectors of length 6, composed of first 3 = torque, last 3 = force) expressed in the root body frame, and concatenated together. The root body is probably the pelvis, but for some skeletons they may use another body as the root, like the torso.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def jointCenters(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        These are the joint center locations, concatenated together, given in the world frame.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def jointCentersInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        These are the joint center locations, concatenated together, given in the root frame. The root body is probably the pelvis, but for some skeletons they may use another body as the root, like the torso.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def linearResidual(self) -> float:
        """
        A scalar giving how much linear force, in Newtons, would need to be applied at the root of the skeleton in order to enable the skeleton's observed accelerations (given positions and velocities) on this frame.

        :type: float
        """
    @property
    def markerMax(self) -> float:
        """
        A scalar indicating the maximum marker error (discrepancy between the model and the experimentally observed marker locations) on this frame, in meters, with these joint positions.

        :type: float
        """
    @property
    def markerRMS(self) -> float:
        """
        A scalar indicating the RMS marker error (discrepancy between the model and the experimentally observed marker locations) on this frame, in meters, with these joint positions.

        :type: float
        """
    @property
    def pos(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        The joint positions on this frame.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def posObserved(self) -> numpy.ndarray[numpy.int32, _Shape[m, 1]]:
        """
        A boolean mask of [0,1]s for each DOF, with a 1 indicating that this DOF was observed on this frame

        :type: numpy.ndarray[numpy.int32, _Shape[m, 1]]
        """
    @property
    def residualWrenchInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]:
        """
        This is the 'residual' force wrench (or 'modelling error' force, the force necessary to make Newton's laws match up with our model, even though it's imaginary) expressed in the root body frame. This is a wrench (vector of length 6, composed of first 3 = torque, last 3 = force). The root body is probably the pelvis, but for some skeletons they may use another body as the root, like the torso.

        :type: numpy.ndarray[numpy.float64, _Shape[6, 1]]
        """
    @property
    def rootAngularAccInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        This is the angular velocity, in an angle-axis representation where the norm of this 3-vector is given in radians per second squared, of the root body of the skeleton (probably the pelvis) expressed in its own coordinate frame.

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def rootAngularVelInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        This is the angular velocity, in an angle-axis representation where the norm of this 3-vector is given in radians per second, of the root body of the skeleton (probably the pelvis) expressed in its own coordinate frame.

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def rootEulerHistoryInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        This is the recent history of the angles (expressed as euler angles) of the root body of the skeleton (probably the pelvis) expressed in its own coordinate frame.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def rootLinearAccInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        This is the linear acceleration, in meters per second squared, of the root body of the skeleton (probably the pelvis) expressed in its own coordinate frame.

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def rootLinearVelInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        This is the linear velocity, in meters per second, of the root body of the skeleton (probably the pelvis) expressed in its own coordinate frame.

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @property
    def rootPosHistoryInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        This is the recent history of the positions of the root body of the skeleton (probably the pelvis) expressed in its own coordinate frame. These are concatenated 3-vectors. The [0:3] of the vector is the most recent, and they get older from there. Vectors  

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def tau(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        The joint control forces on this frame.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def type(self) -> ProcessingPassType:
        """
        The type of processing pass that this data came from. Options include KINEMATICS (for movement only), DYNAMICS (for movement and physics), and LOW_PASS_FILTER (to apply a simple Butterworth to the observed data from the previous pass).

        :type: ProcessingPassType
        """
    @property
    def vel(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        The joint velocities on this frame.

        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @property
    def velFiniteDifferenced(self) -> numpy.ndarray[numpy.int32, _Shape[m, 1]]:
        """
        A boolean mask of [0,1]s for each DOF, with a 1 indicating that this DOF got its velocity through finite differencing, and therefore may be somewhat unreliable

        :type: numpy.ndarray[numpy.int32, _Shape[m, 1]]
        """
    pass
class FramePassList():
    def __bool__(self) -> bool: 
        """
        Check whether the list is nonempty
        """
    def __contains__(self, x: FramePass) -> bool: 
        """
        Return true the container contains ``x``
        """
    @typing.overload
    def __delitem__(self, arg0: int) -> None: 
        """
        Delete the list elements at index ``i``

        Delete list elements using a slice object
        """
    @typing.overload
    def __delitem__(self, arg0: slice) -> None: ...
    def __eq__(self, arg0: FramePassList) -> bool: ...
    @typing.overload
    def __getitem__(self, s: slice) -> FramePassList: 
        """
        Retrieve list elements using a slice object
        """
    @typing.overload
    def __getitem__(self, arg0: int) -> FramePass: ...
    @typing.overload
    def __init__(self) -> None: 
        """
        Copy constructor
        """
    @typing.overload
    def __init__(self, arg0: FramePassList) -> None: ...
    @typing.overload
    def __init__(self, arg0: typing.Iterable) -> None: ...
    def __iter__(self) -> typing.Iterator: ...
    def __len__(self) -> int: ...
    def __ne__(self, arg0: FramePassList) -> bool: ...
    def __repr__(self) -> str: 
        """
        Return the canonical string representation of this list.
        """
    @typing.overload
    def __setitem__(self, arg0: int, arg1: FramePass) -> None: 
        """
        Assign list elements using a slice object
        """
    @typing.overload
    def __setitem__(self, arg0: slice, arg1: FramePassList) -> None: ...
    def append(self, x: FramePass) -> None: 
        """
        Add an item to the end of the list
        """
    def clear(self) -> None: 
        """
        Clear the contents
        """
    def count(self, x: FramePass) -> int: 
        """
        Return the number of times ``x`` appears in the list
        """
    @typing.overload
    def extend(self, L: FramePassList) -> None: 
        """
        Extend the list by appending all the items in the given list

        Extend the list by appending all the items in the given list
        """
    @typing.overload
    def extend(self, L: typing.Iterable) -> None: ...
    def insert(self, i: int, x: FramePass) -> None: 
        """
        Insert an item at a given position.
        """
    @typing.overload
    def pop(self) -> FramePass: 
        """
        Remove and return the last item

        Remove and return the item at index ``i``
        """
    @typing.overload
    def pop(self, i: int) -> FramePass: ...
    def remove(self, x: FramePass) -> None: 
        """
        Remove the first item from the list whose value is x. It is an error if there is no such item.
        """
    __hash__: typing.ClassVar[None] = None
    pass
class IKErrorReport():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, markers: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], poses: numpy.ndarray[numpy.float64, _Shape[m, n]], observations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    def getSortedMarkerRMSE(self) -> typing.List[typing.Tuple[str, float]]: ...
    def printReport(self, limitTimesteps: int = -1) -> None: ...
    def saveCSVMarkerErrorReport(self, path: str) -> None: ...
    @property
    def averageMaxError(self) -> float:
        """
        :type: float
        """
    @averageMaxError.setter
    def averageMaxError(self, arg0: float) -> None:
        pass
    @property
    def averageRootMeanSquaredError(self) -> float:
        """
        :type: float
        """
    @averageRootMeanSquaredError.setter
    def averageRootMeanSquaredError(self, arg0: float) -> None:
        pass
    @property
    def averageSumSquaredError(self) -> float:
        """
        :type: float
        """
    @averageSumSquaredError.setter
    def averageSumSquaredError(self, arg0: float) -> None:
        pass
    @property
    def maxError(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @maxError.setter
    def maxError(self, arg0: typing.List[float]) -> None:
        pass
    @property
    def rootMeanSquaredError(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @rootMeanSquaredError.setter
    def rootMeanSquaredError(self, arg0: typing.List[float]) -> None:
        pass
    @property
    def sumSquaredError(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @sumSquaredError.setter
    def sumSquaredError(self, arg0: typing.List[float]) -> None:
        pass
    pass
class IMUFineTuneProblem():
    def flatten(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getAccs(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getGrad(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getLoss(self) -> float: ...
    def getPoses(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getProblemSize(self) -> int: ...
    def getVels(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def setWeightAccs(self, weight: float) -> None: ...
    def setWeightGyros(self, weight: float) -> None: ...
    def setWeightPoses(self, weight: float) -> None: ...
    def unflatten(self, x: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    pass
class InitialMarkerFitParams():
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def setDontRescaleBodies(self, dontRescaleBodies: bool) -> InitialMarkerFitParams: ...
    def setGroupScales(self, groupScales: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> InitialMarkerFitParams: ...
    def setInitPoses(self, initPoses: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> InitialMarkerFitParams: ...
    def setJointCenters(self, joints: typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint], jointCenters: numpy.ndarray[numpy.float64, _Shape[m, n]], jointAdjacentMarkers: typing.List[typing.List[str]]) -> InitialMarkerFitParams: ...
    def setJointCentersAndWeights(self, joints: typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint], jointCenters: numpy.ndarray[numpy.float64, _Shape[m, n]], jointAdjacentMarkers: typing.List[typing.List[str]], jointWeights: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> InitialMarkerFitParams: ...
    def setMarkerOffsets(self, markerOffsets: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> InitialMarkerFitParams: ...
    def setMarkerWeights(self, markerWeights: typing.Dict[str, float]) -> InitialMarkerFitParams: ...
    def setMaxTimestepsToUseForMultiTrialScaling(self, numTimesteps: int) -> InitialMarkerFitParams: ...
    def setMaxTrialsToUseForMultiTrialScaling(self, numTrials: int) -> InitialMarkerFitParams: ...
    def setNumBlocks(self, numBlocks: int) -> InitialMarkerFitParams: ...
    def setNumIKTries(self, tries: int) -> InitialMarkerFitParams: ...
    @property
    def dontRescaleBodies(self) -> bool:
        """
        :type: bool
        """
    @dontRescaleBodies.setter
    def dontRescaleBodies(self, arg0: bool) -> None:
        pass
    @property
    def groupScales(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @groupScales.setter
    def groupScales(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def initPoses(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @initPoses.setter
    def initPoses(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def jointCenters(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @jointCenters.setter
    def jointCenters(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def jointWeights(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @jointWeights.setter
    def jointWeights(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def joints(self) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]:
        """
        :type: typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]
        """
    @joints.setter
    def joints(self, arg0: typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]) -> None:
        pass
    @property
    def markerOffsets(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @markerOffsets.setter
    def markerOffsets(self, arg0: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def markerWeights(self) -> typing.Dict[str, float]:
        """
        :type: typing.Dict[str, float]
        """
    @markerWeights.setter
    def markerWeights(self, arg0: typing.Dict[str, float]) -> None:
        pass
    @property
    def maxTimestepsToUseForMultiTrialScaling(self) -> int:
        """
        :type: int
        """
    @maxTimestepsToUseForMultiTrialScaling.setter
    def maxTimestepsToUseForMultiTrialScaling(self, arg0: int) -> None:
        pass
    @property
    def maxTrialsToUseForMultiTrialScaling(self) -> int:
        """
        :type: int
        """
    @maxTrialsToUseForMultiTrialScaling.setter
    def maxTrialsToUseForMultiTrialScaling(self, arg0: int) -> None:
        pass
    @property
    def numBlocks(self) -> int:
        """
        :type: int
        """
    @numBlocks.setter
    def numBlocks(self, arg0: int) -> None:
        pass
    pass
class LabelledMarkers():
    @property
    def jointCenterGuesses(self) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @jointCenterGuesses.setter
    def jointCenterGuesses(self, arg0: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def markerObservations(self) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markerObservations.setter
    def markerObservations(self, arg0: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def markerOffsets(self) -> typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markerOffsets.setter
    def markerOffsets(self, arg0: typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def traces(self) -> typing.List[MarkerTrace]:
        """
        :type: typing.List[MarkerTrace]
        """
    @traces.setter
    def traces(self, arg0: typing.List[MarkerTrace]) -> None:
        pass
    pass
class LilypadSolver():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, groundContactBodies: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], groundNormal: numpy.ndarray[numpy.float64, _Shape[3, 1]], tileSize: float) -> None: ...
    def clear(self) -> None: ...
    def debugToGUI(self, gui: nimblephysics_libs._nimblephysics.server.GUIWebsocketServer) -> None: ...
    def getContactBodies(self) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode]: ...
    def process(self, poses: numpy.ndarray[numpy.float64, _Shape[m, n]], startTime: int = 0) -> None: ...
    def setLateralVelThreshold(self, threshold: float) -> None: ...
    def setVerticalAccelerationThreshold(self, threshold: float) -> None: ...
    def setVerticalVelThreshold(self, threshold: float) -> None: ...
    pass
class MarkerFitter():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, markers: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], ignoreVirtualJointCenterMarkers: bool = False) -> None: ...
    def addZeroConstraint(self, name: str, loss: typing.Callable[[MarkerFitterState], float]) -> None: ...
    def autorotateC3D(self, c3d: C3D) -> None: ...
    def checkForEnoughMarkers(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> bool: ...
    def checkForFlippedMarkers(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], init: MarkerInitialization, report: MarkersErrorReport) -> bool: ...
    def debugTrajectoryAndMarkersToGUI(self, server: nimblephysics_libs._nimblephysics.server.GUIWebsocketServer, init: MarkerInitialization, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], forcePlates: typing.List[ForcePlate] = None, goldOsim: OpenSimFile = None, goldPoses: numpy.ndarray[numpy.float64, _Shape[m, n]] = array([], shape=(0, 0), dtype=float64)) -> None: ...
    def findJointCenters(self, initializations: MarkerInitialization, newClip: typing.List[bool], markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    def fineTuneWithIMU(self, accObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], gyroObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], newClip: typing.List[bool], init: MarkerInitialization, dt: float, weightAccs: float = 1.0, weightGyros: float = 1.0, weightMarkers: float = 100.0, regularizePoses: float = 1.0, useIPOPT: bool = True, iterations: int = 300, lbfgsMemory: int = 100) -> MarkerInitialization: ...
    def generateDataErrorsReport(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], dt: float, rippleReduce: bool = True, rippleReduceUseSparse: bool = True, rippleReduceUseIterativeSolver: bool = True, rippleReduceSolverIterations: int = 100000.0) -> MarkersErrorReport: ...
    def getIMUFineTuneProblem(self, accObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], gyroObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], init: MarkerInitialization, dt: float, start: int, end: int) -> IMUFineTuneProblem: ...
    def getImuList(self) -> typing.List[typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, nimblephysics_libs._nimblephysics.math.Isometry3]]: ...
    def getImuMap(self) -> typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, nimblephysics_libs._nimblephysics.math.Isometry3]]: ...
    def getImuNames(self) -> typing.List[str]: ...
    def getInitialization(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], newClip: typing.List[bool], params: InitialMarkerFitParams = InitialMarkerFitParams(numBlocks=12)) -> MarkerInitialization: ...
    def getMarkerIsTracking(self, marker: str) -> bool: ...
    @staticmethod
    def getMarkerLossGradientWrtJoints(skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, markers: typing.List[typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], lossGradWrtMarkerError: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getNumMarkers(self) -> int: ...
    def measureAccelerometerRMS(self, accObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], newClip: typing.List[bool], init: MarkerInitialization, dt: float) -> float: ...
    def measureGyroRMS(self, gyroObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], newClip: typing.List[bool], init: MarkerInitialization, dt: float) -> float: ...
    def optimizeBilevel(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], initialization: MarkerInitialization, numSamples: int, applyInnerProblemGradientConstraints: bool = True) -> BilevelFitResult: ...
    @staticmethod
    def pickSubset(markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], subsetSize: int) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]: ...
    def removeZeroConstraint(self, name: str) -> None: ...
    def rotateIMUs(self, accObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], gyroObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], newClip: typing.List[bool], init: MarkerInitialization, dt: float) -> None: ...
    def runKinematicsPipeline(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], newClip: typing.List[bool], params: InitialMarkerFitParams, numSamples: int = 20, skipFinalIK: bool = False) -> MarkerInitialization: ...
    def runMultiTrialKinematicsPipeline(self, markerTrials: typing.List[typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]], params: InitialMarkerFitParams, numSamples: int = 50) -> typing.List[MarkerInitialization]: ...
    def runPrescaledPipeline(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], params: InitialMarkerFitParams) -> MarkerInitialization: ...
    def saveTrajectoryAndMarkersToGUI(self, path: str, init: MarkerInitialization, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], accObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], gyroObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], frameRate: int, forcePlates: typing.List[ForcePlate] = None, goldOsim: OpenSimFile = None, goldPoses: numpy.ndarray[numpy.float64, _Shape[m, n]] = array([], shape=(0, 0), dtype=float64)) -> None: ...
    def setAnatomicalMarkerDefaultWeight(self, weight: float) -> None: ...
    def setAnthropometricPrior(self, prior: Anthropometrics, weight: float = 0.001) -> None: ...
    def setCustomLossAndGrad(self, loss: typing.Callable[[MarkerFitterState], float]) -> None: ...
    def setDebugJointVariability(self, debug: bool) -> None: ...
    def setDebugLoss(self, debug: bool) -> None: ...
    def setExplicitHeightPrior(self, prior: float, weight: float = 1000.0) -> None: ...
    def setIgnoreJointLimits(self, ignore: bool) -> None: ...
    def setImuMap(self, imuMap: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, nimblephysics_libs._nimblephysics.math.Isometry3]]) -> None: ...
    def setInitialIKMaxRestarts(self, starts: int) -> None: ...
    def setInitialIKSatisfactoryLoss(self, loss: float) -> None: ...
    def setIterationLimit(self, iters: int) -> None: ...
    def setJointAxisFitSGDIterations(self, iters: int) -> None: ...
    def setJointForceFieldSoftness(self, softness: float) -> None: 
        """
        Larger values will increase the softness of the threshold penalty. Smaller
        values, as they approach zero, will have an almost perfectly vertical
        penality for going below the threshold distance. That would be hard to
        optimize, so don't make it too small.
                
        """
    def setJointForceFieldThresholdDistance(self, minDistance: float) -> None: 
        """
        This sets the minimum distance joints have to be apart in order to get
        zero "force field" loss. Any joints closer than this (in world space) will
        incur a penalty.
                
        """
    def setJointSphereFitSGDIterations(self, iters: int) -> None: ...
    def setMarkerIsTracking(self, marker: str, isTracking: bool = True) -> None: ...
    def setMaxAxisWeight(self, weight: float) -> None: ...
    def setMaxJointWeight(self, weight: float) -> None: ...
    def setMaxMarkerOffset(self, offset: float) -> None: ...
    def setMinAxisFitScore(self, score: float) -> None: ...
    def setMinJointVarianceCutoff(self, cutoff: float) -> None: ...
    def setMinSphereFitScore(self, score: float) -> None: ...
    def setParallelIKWarps(self, parallelWarps: bool) -> None: 
        """
        If True, this processes "single threaded" IK tasks 32 timesteps at a time
                    (a "warp"), in parallel, using the first timestep of the warp as the
                    initialization for the whole warp. Defaults to False.
                  
        """
    def setPostprocessAnatomicalMarkerOffsets(self, postprocess: bool) -> None: 
        """
        If we set this to true, then after the main optimization completes we will
        do a final step to "center" the error of the anatomical markers. This
        minimizes marker RMSE, but does NOT respect the weights about how far
        markers should be allowed to move.
                
        """
    def setPostprocessTrackingMarkerOffsets(self, postprocess: bool) -> None: 
        """
        If we set this to true, then after the main optimization completes we will
        do a final step to "center" the error of the tracking markers. This
        minimizes marker RMSE, but does NOT respect the weights about how far
        markers should be allowed to move.
                
        """
    def setRegularizeAllBodyScales(self, weight: float) -> None: ...
    def setRegularizeAnatomicalMarkerOffsets(self, weight: float) -> None: ...
    def setRegularizeIndividualBodyScales(self, weight: float) -> None: ...
    def setRegularizeJointBounds(self, weight: float) -> None: ...
    def setRegularizeJointWithVirtualSpring(self, jointName: str, weight: float) -> None: ...
    def setRegularizePelvisJointsWithVirtualSpring(self, weight: float) -> None: ...
    def setRegularizeTrackingMarkerOffsets(self, weight: float) -> None: ...
    def setStaticTrial(self, markerObservationsMapAtStaticPose: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]], staticPose: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setStaticTrialWeight(self, weight: float) -> None: ...
    def setTrackingMarkerDefaultWeight(self, weight: float) -> None: ...
    def setTrackingMarkers(self, trackingMarkerNames: typing.List[str]) -> None: ...
    def setTriadsToTracking(self) -> None: ...
    def writeCSVData(self, path: str, init: MarkerInitialization, rmsMarkerErrors: typing.List[float], maxMarkerErrors: typing.List[float], timestamps: typing.List[float]) -> None: ...
    pass
class MarkerFitterState():
    @property
    def bodyNames(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @bodyNames.setter
    def bodyNames(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def bodyScales(self) -> numpy.ndarray[numpy.float64, _Shape[3, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, n]]
        """
    @bodyScales.setter
    def bodyScales(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, n]]) -> None:
        pass
    @property
    def bodyScalesGrad(self) -> numpy.ndarray[numpy.float64, _Shape[3, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, n]]
        """
    @bodyScalesGrad.setter
    def bodyScalesGrad(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, n]]) -> None:
        pass
    @property
    def jointErrorsAtTimesteps(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @jointErrorsAtTimesteps.setter
    def jointErrorsAtTimesteps(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def jointErrorsAtTimestepsGrad(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @jointErrorsAtTimestepsGrad.setter
    def jointErrorsAtTimestepsGrad(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def jointOrder(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @jointOrder.setter
    def jointOrder(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def markerErrorsAtTimesteps(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @markerErrorsAtTimesteps.setter
    def markerErrorsAtTimesteps(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def markerErrorsAtTimestepsGrad(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @markerErrorsAtTimestepsGrad.setter
    def markerErrorsAtTimestepsGrad(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def markerOffsets(self) -> numpy.ndarray[numpy.float64, _Shape[3, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, n]]
        """
    @markerOffsets.setter
    def markerOffsets(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, n]]) -> None:
        pass
    @property
    def markerOffsetsGrad(self) -> numpy.ndarray[numpy.float64, _Shape[3, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, n]]
        """
    @markerOffsetsGrad.setter
    def markerOffsetsGrad(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, n]]) -> None:
        pass
    @property
    def markerOrder(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @markerOrder.setter
    def markerOrder(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def posesAtTimesteps(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @posesAtTimesteps.setter
    def posesAtTimesteps(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def posesAtTimestepsGrad(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @posesAtTimestepsGrad.setter
    def posesAtTimestepsGrad(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    pass
class MarkerFixer():
    @staticmethod
    def generateDataErrorsReport(immutableMarkerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], dt: float, dropProlongedStillness: bool = False, rippleReduce: bool = True, rippleReduceUseSparse: bool = True, rippleReduceUseIterativeSolver: bool = True, rippleReduceSolverIterations: int = 100000.0) -> MarkersErrorReport: ...
    pass
class MarkerInitialization():
    def __init__(self) -> None: ...
    @property
    def axisWeights(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @axisWeights.setter
    def axisWeights(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def error(self) -> bool:
        """
        :type: bool
        """
    @error.setter
    def error(self, arg0: bool) -> None:
        pass
    @property
    def errorMsg(self) -> str:
        """
        :type: str
        """
    @errorMsg.setter
    def errorMsg(self, arg0: str) -> None:
        pass
    @property
    def groupScales(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @groupScales.setter
    def groupScales(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def jointAxis(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @jointAxis.setter
    def jointAxis(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def jointCenters(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @jointCenters.setter
    def jointCenters(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def jointWeights(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @jointWeights.setter
    def jointWeights(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def joints(self) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]:
        """
        :type: typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]
        """
    @joints.setter
    def joints(self, arg0: typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]) -> None:
        pass
    @property
    def jointsAdjacentMarkers(self) -> typing.List[typing.List[str]]:
        """
        :type: typing.List[typing.List[str]]
        """
    @jointsAdjacentMarkers.setter
    def jointsAdjacentMarkers(self, arg0: typing.List[typing.List[str]]) -> None:
        pass
    @property
    def markerOffsets(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @markerOffsets.setter
    def markerOffsets(self, arg0: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def poses(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @poses.setter
    def poses(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def updatedMarkerMap(self) -> typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @updatedMarkerMap.setter
    def updatedMarkerMap(self, arg0: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    pass
class MarkerLabeller():
    def evaluate(self, markerOffsets: typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], labeledPointClouds: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    def guessJointLocations(self, pointClouds: typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]: ...
    def labelPointClouds(self, pointClouds: typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]], mergeMarkersThreshold: float = 0.01) -> LabelledMarkers: ...
    def matchUpJointToSkeletonJoint(self, jointName: str, skeletonJointName: str) -> None: ...
    def setSkeleton(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None: ...
    pass
class MarkerLabellerMock(MarkerLabeller):
    def __init__(self) -> None: ...
    def setMockJointLocations(self, jointsOverTime: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    pass
class MarkerTrace():
    @property
    def bodyClosestPointDistance(self) -> typing.Dict[str, float]:
        """
        :type: typing.Dict[str, float]
        """
    @property
    def bodyMarkerOffsetVariance(self) -> typing.Dict[str, float]:
        """
        :type: typing.Dict[str, float]
        """
    @property
    def bodyMarkerOffsets(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @property
    def bodyRootJointDistVariance(self) -> typing.Dict[str, float]:
        """
        :type: typing.Dict[str, float]
        """
    @property
    def markerLabel(self) -> str:
        """
        :type: str
        """
    @property
    def maxTime(self) -> int:
        """
        :type: int
        """
    @property
    def minTime(self) -> int:
        """
        :type: int
        """
    @property
    def points(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @property
    def times(self) -> typing.List[int]:
        """
        :type: typing.List[int]
        """
    pass
class MarkersErrorReport():
    def getMarkerMapOnTimestep(self, t: int) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]: ...
    def getMarkerNamesOnTimestep(self, t: int) -> typing.List[str]: ...
    def getMarkerPositionOnTimestep(self, t: int, marker: str) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def getNumTimesteps(self) -> int: ...
    @property
    def droppedMarkerWarnings(self) -> typing.List[typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]], str]]]:
        """
        :type: typing.List[typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]], str]]]
        """
    @droppedMarkerWarnings.setter
    def droppedMarkerWarnings(self, arg0: typing.List[typing.List[typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]], str]]]) -> None:
        pass
    @property
    def info(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @info.setter
    def info(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def markerObservationsAttemptedFixed(self) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markerObservationsAttemptedFixed.setter
    def markerObservationsAttemptedFixed(self, arg0: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def markersRenamedFromTo(self) -> typing.List[typing.List[typing.Tuple[str, str]]]:
        """
        :type: typing.List[typing.List[typing.Tuple[str, str]]]
        """
    @markersRenamedFromTo.setter
    def markersRenamedFromTo(self, arg0: typing.List[typing.List[typing.Tuple[str, str]]]) -> None:
        pass
    @property
    def warnings(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @warnings.setter
    def warnings(self, arg0: typing.List[str]) -> None:
        pass
    pass
class MissingGRFReason():
    """
    Members:

      notMissingGRF

      measuredGrfZeroWhenAccelerationNonZero

      unmeasuredExternalForceDetected

      torqueDiscrepancy

      forceDiscrepancy

      notOverForcePlate

      missingImpact

      missingBlip

      shiftGRF

      interpolatedClippedGRF

      manualReview
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    __members__: dict # value = {'notMissingGRF': <MissingGRFReason.notMissingGRF: 0>, 'measuredGrfZeroWhenAccelerationNonZero': <MissingGRFReason.measuredGrfZeroWhenAccelerationNonZero: 1>, 'unmeasuredExternalForceDetected': <MissingGRFReason.unmeasuredExternalForceDetected: 2>, 'torqueDiscrepancy': <MissingGRFReason.torqueDiscrepancy: 3>, 'forceDiscrepancy': <MissingGRFReason.forceDiscrepancy: 4>, 'notOverForcePlate': <MissingGRFReason.notOverForcePlate: 5>, 'missingImpact': <MissingGRFReason.missingImpact: 6>, 'missingBlip': <MissingGRFReason.missingBlip: 7>, 'shiftGRF': <MissingGRFReason.shiftGRF: 8>, 'interpolatedClippedGRF': <MissingGRFReason.interpolatedClippedGRF: 10>, 'manualReview': <MissingGRFReason.manualReview: 9>}
    forceDiscrepancy: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.forceDiscrepancy: 4>
    interpolatedClippedGRF: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.interpolatedClippedGRF: 10>
    manualReview: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.manualReview: 9>
    measuredGrfZeroWhenAccelerationNonZero: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.measuredGrfZeroWhenAccelerationNonZero: 1>
    missingBlip: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.missingBlip: 7>
    missingImpact: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.missingImpact: 6>
    notMissingGRF: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.notMissingGRF: 0>
    notOverForcePlate: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.notOverForcePlate: 5>
    shiftGRF: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.shiftGRF: 8>
    torqueDiscrepancy: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.torqueDiscrepancy: 3>
    unmeasuredExternalForceDetected: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.unmeasuredExternalForceDetected: 2>
    pass
class MissingGRFStatus():
    """
    Members:

      no

      unknown

      yes
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    __members__: dict # value = {'no': <MissingGRFStatus.no: 0>, 'unknown': <MissingGRFStatus.unknown: 1>, 'yes': <MissingGRFStatus.yes: 2>}
    no: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFStatus # value = <MissingGRFStatus.no: 0>
    unknown: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFStatus # value = <MissingGRFStatus.unknown: 1>
    yes: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFStatus # value = <MissingGRFStatus.yes: 2>
    pass
class NeuralMarkerLabeller(MarkerLabeller):
    def __init__(self, jointCenterPredictor: typing.Callable[[typing.List[typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]], typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]]) -> None: ...
    pass
class OpenSimFile():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, markers: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    @property
    def anatomicalMarkers(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @anatomicalMarkers.setter
    def anatomicalMarkers(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def ignoredBodies(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @ignoredBodies.setter
    def ignoredBodies(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def jointsDrivenBy(self) -> typing.List[typing.Tuple[str, str]]:
        """
        :type: typing.List[typing.Tuple[str, str]]
        """
    @jointsDrivenBy.setter
    def jointsDrivenBy(self, arg0: typing.List[typing.Tuple[str, str]]) -> None:
        pass
    @property
    def markersMap(self) -> typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markersMap.setter
    def markersMap(self, arg0: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def skeleton(self) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton:
        """
        :type: nimblephysics_libs._nimblephysics.dynamics.Skeleton
        """
    @skeleton.setter
    def skeleton(self, arg0: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None:
        pass
    @property
    def trackingMarkers(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @trackingMarkers.setter
    def trackingMarkers(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def warnings(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @warnings.setter
    def warnings(self, arg0: typing.List[str]) -> None:
        pass
    pass
class OpenSimMocoTrajectory():
    @property
    def activationNames(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @activationNames.setter
    def activationNames(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def activations(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @activations.setter
    def activations(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def excitationNames(self) -> typing.List[str]:
        """
        :type: typing.List[str]
        """
    @excitationNames.setter
    def excitationNames(self, arg0: typing.List[str]) -> None:
        pass
    @property
    def excitations(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @excitations.setter
    def excitations(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def timestamps(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @timestamps.setter
    def timestamps(self, arg0: typing.List[float]) -> None:
        pass
    pass
class OpenSimMot():
    @property
    def poses(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @poses.setter
    def poses(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def timestamps(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @timestamps.setter
    def timestamps(self, arg0: typing.List[float]) -> None:
        pass
    pass
class OpenSimScaleAndMarkerOffsets():
    @property
    def bodyScales(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @bodyScales.setter
    def bodyScales(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def markerOffsets(self) -> typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]:
        """
        :type: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]
        """
    @markerOffsets.setter
    def markerOffsets(self, arg0: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]) -> None:
        pass
    @property
    def markers(self) -> typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markers.setter
    def markers(self, arg0: typing.Dict[str, typing.Tuple[nimblephysics_libs._nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def success(self) -> bool:
        """
        :type: bool
        """
    @success.setter
    def success(self, arg0: bool) -> None:
        pass
    pass
class OpenSimTRC():
    @property
    def framesPerSecond(self) -> int:
        """
        :type: int
        """
    @framesPerSecond.setter
    def framesPerSecond(self, arg0: int) -> None:
        pass
    @property
    def markerLines(self) -> typing.Dict[str, typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.Dict[str, typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markerLines.setter
    def markerLines(self, arg0: typing.Dict[str, typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def markerTimesteps(self) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]:
        """
        :type: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]
        """
    @markerTimesteps.setter
    def markerTimesteps(self, arg0: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
        pass
    @property
    def timestamps(self) -> typing.List[float]:
        """
        :type: typing.List[float]
        """
    @timestamps.setter
    def timestamps(self, arg0: typing.List[float]) -> None:
        pass
    pass
class ProcessingPassType():
    """
    Members:

      KINEMATICS : This is the pass where we solve for kinematics.

      DYNAMICS : This is the pass where we solve for dynamics.

      LOW_PASS_FILTER : This is the pass where we apply a low-pass filter to the kinematics and dynamics.
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    DYNAMICS: nimblephysics_libs._nimblephysics.biomechanics.ProcessingPassType # value = <ProcessingPassType.DYNAMICS: 1>
    KINEMATICS: nimblephysics_libs._nimblephysics.biomechanics.ProcessingPassType # value = <ProcessingPassType.KINEMATICS: 0>
    LOW_PASS_FILTER: nimblephysics_libs._nimblephysics.biomechanics.ProcessingPassType # value = <ProcessingPassType.LOW_PASS_FILTER: 2>
    __members__: dict # value = {'KINEMATICS': <ProcessingPassType.KINEMATICS: 0>, 'DYNAMICS': <ProcessingPassType.DYNAMICS: 1>, 'LOW_PASS_FILTER': <ProcessingPassType.LOW_PASS_FILTER: 2>}
    pass
class ResidualForceHelper():
    def __init__(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton, forceBodies: typing.List[int]) -> None: ...
    def calculateResidual(self, q: numpy.ndarray[numpy.float64, _Shape[m, 1]], dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], forcesConcat: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]: ...
    def calculateResidualJacobianWrt(self, q: numpy.ndarray[numpy.float64, _Shape[m, 1]], dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], forcesConcat: numpy.ndarray[numpy.float64, _Shape[m, 1]], wrt: nimblephysics_libs._nimblephysics.neural.WithRespectTo) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def calculateResidualNorm(self, q: numpy.ndarray[numpy.float64, _Shape[m, 1]], dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], forcesConcat: numpy.ndarray[numpy.float64, _Shape[m, 1]], torquesMultiple: float, useL1: bool = False) -> float: ...
    def calculateResidualNormGradientWrt(self, q: numpy.ndarray[numpy.float64, _Shape[m, 1]], dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], forcesConcat: numpy.ndarray[numpy.float64, _Shape[m, 1]], wrt: nimblephysics_libs._nimblephysics.neural.WithRespectTo, torquesMultiple: float, useL1: bool = False) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    pass
class SkeletonConverter():
    def __init__(self, source: nimblephysics_libs._nimblephysics.dynamics.Skeleton, target: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None: ...
    def convertMotion(self, targetMotion: numpy.ndarray[numpy.float64, _Shape[m, n]], logProgress: bool = True, convergenceThreshold: float = 1e-07, maxStepCount: int = 100, leastSquaresDamping: float = 0.01, lineSearch: bool = True, logIKOutput: bool = False) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def createVirtualMarkers(self, addFakeMarkers: int = 3, weightFakeMarkers: float = 0.1) -> None: ...
    def debugToGUI(self, gui: nimblephysics_libs._nimblephysics.server.GUIWebsocketServer) -> None: ...
    def fitSourceToTarget(self, convergenceThreshold: float = 1e-07, maxStepCount: int = 100, leastSquaresDamping: float = 0.01, lineSearch: bool = True, logOutput: bool = False) -> float: ...
    def fitTargetToSource(self, convergenceThreshold: float = 1e-07, maxStepCount: int = 100, leastSquaresDamping: float = 0.01, lineSearch: bool = True, logOutput: bool = False) -> float: ...
    def getSourceJointWorldPositions(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getSourceJoints(self) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]: ...
    def getTargetJointWorldPositions(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getTargetJoints(self) -> typing.List[nimblephysics_libs._nimblephysics.dynamics.Joint]: ...
    def linkJoints(self, sourceJoint: nimblephysics_libs._nimblephysics.dynamics.Joint, targetJoint: nimblephysics_libs._nimblephysics.dynamics.Joint) -> None: ...
    def rescaleAndPrepTarget(self, addFakeMarkers: int = 3, weightFakeMarkers: float = 0.1, convergenceThreshold: float = 1e-15, maxStepCount: int = 1000, leastSquaresDamping: float = 0.01, lineSearch: bool = True, logOutput: bool = False) -> None: ...
    pass
class SubjectOnDisk():
    """
    This is for doing ML and large-scale data analysis. The idea here is to
    create a lazy-loadable view of a subject, where everything remains on disk
    until asked for. That way we can instantiate thousands of these in memory,
    and not worry about OOM'ing a machine.
    """
    def __init__(self, path: str) -> None: ...
    def getAgeYears(self) -> int: 
        """
        This returns the age of the subject, or 0 if unknown.
        """
    def getBiologicalSex(self) -> str: 
        """
        This returns a string, one of "male", "female", or "unknown".
        """
    def getCustomValueDim(self, valueName: str) -> int: 
        """
        This returns the dimension of the custom value specified by :code:`valueName`
        """
    def getCustomValues(self) -> typing.List[str]: 
        """
        A list of all the different types of custom values that this SubjectOnDisk contains. These are unspecified, and are intended to allow an easy extension of the format to unusual types of data (like exoskeleton torques or unusual physical sensors) that may be present on some subjects but not others.
        """
    def getDofAccelerationsFiniteDifferenced(self, trial: int, processingPass: int) -> typing.List[bool]: 
        """
        This returns the vector of booleans indicating which DOFs have their accelerations from finite-differencing during this trial (as opposed to observed directly through a accelerometer or IMU)
        """
    def getDofPositionsObserved(self, trial: int, processingPass: int) -> typing.List[bool]: 
        """
        This returns the vector of booleans indicating which DOFs have their positions observed during this trial
        """
    def getDofVelocitiesFiniteDifferenced(self, trial: int, processingPass: int) -> typing.List[bool]: 
        """
        This returns the vector of booleans indicating which DOFs have their velocities from finite-differencing during this trial (as opposed to observed directly through a gyroscope or IMU)
        """
    def getForcePlateCorners(self, trial: int, forcePlate: int) -> typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]]: 
        """
        Get an array of force plate corners (as 3D vectors) for the given force plate in the given trial. Empty array on out-of-bounds access.
        """
    def getForceplateCutoffs(self, trial: int, processingPass: int) -> typing.List[float]: 
        """
        If we reprocessed the force plates with a cutoff, then these are the cutoff values we used.
        """
    def getGroundForceBodies(self) -> typing.List[str]: 
        """
        A list of the :code:`body_name`'s for each body that was assumed to be able to take ground-reaction-force from force plates.
        """
    def getHeaderProto(self) -> SubjectOnDiskHeader: 
        """
        This returns the raw proto header for this subject, which can be used to write out a new B3D file
        """
    def getHeightM(self) -> float: 
        """
        This returns the height in meters, or 0.0 if unknown.
        """
    def getHref(self) -> str: 
        """
        The AddBiomechanics link for this subject's data.
        """
    def getLowpassCutoffFrequency(self, trial: int, processingPass: int) -> float: 
        """
        If we're doing a lowpass filter on this pass, then what was the cutoff frequency of that (Butterworth) filter?
        """
    def getLowpassFilterOrder(self, trial: int, processingPass: int) -> int: 
        """
        If we're doing a lowpass filter on this pass, then what was the order of that (Butterworth) filter?
        """
    def getMassKg(self) -> float: 
        """
        This returns the mass in kilograms, or 0.0 if unknown.
        """
    def getMissingGRF(self, trial: int) -> typing.List[MissingGRFReason]: 
        """
        This returns an array of enum values, one per frame in the specified trial,
        each describing whether physics data can be trusted for the corresponding frame of that trial.

        Each frame is either `MissingGRFReason.notMissingGRF`, in which case the physics data is probably trustworthy, or
        some other value indicating why the processing system heuristics believe that there is likely to be unmeasured 
        external force acting on the body at this time.

        WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on the corresponding frame!!

        This method is provided to give a cheaper way to filter out frames we want to ignore for training, without having to call
        the more expensive :code:`loadFrames()` and examine frames individually.
        """
    def getNotes(self) -> str: 
        """
        The notes (if any) added by the person who uploaded this data to AddBiomechanics.
        """
    def getNumDofs(self) -> int: 
        """
        This returns the number of DOFs for the model on this Subject
        """
    def getNumForcePlates(self, trial: int) -> int: 
        """
        The number of force plates in the source data.
        """
    def getNumJoints(self) -> int: 
        """
        This returns the number of joints for the model on this Subject
        """
    def getNumProcessingPasses(self) -> int: 
        """
        This returns the number of processing passes that were successfully completed on this subject. IMPORTANT: Just because a processing pass was done for the subject does not mean that every trial will have successfully completed that processing pass. For example, some trials may lack force plate data, and thus will not have a dynamics pass that requires force plate data.
        """
    def getNumTrials(self) -> int: 
        """
        This returns the number of trials that are in this file.
        """
    def getOpensimFileText(self, processingPass: int) -> str: 
        """
        This will read the raw OpenSim file XML out of the SubjectOnDisk, and return it as a string.
        """
    def getProcessingPassType(self, processingPass: int) -> ProcessingPassType: 
        """
        This returns the type of processing pass at a given index, up to the number of processing passes that were done
        """
    def getSubjectTags(self) -> typing.List[str]: 
        """
        This returns the list of tags attached to this subject, which are arbitrary strings from the AddBiomechanics platform.
        """
    def getTrialAngularResidualNorms(self, trial: int, processingPass: int) -> typing.List[float]: 
        """
        This returns the vector of scalars indicating the norm of the root residual torques on each timestep of a given trial
        """
    def getTrialLength(self, trial: int) -> int: 
        """
        This returns the length of the trial requested
        """
    def getTrialLinearResidualNorms(self, trial: int, processingPass: int) -> typing.List[float]: 
        """
        This returns the vector of scalars indicating the norm of the root residual forces on each timestep of a given trial
        """
    def getTrialMarkerMaxs(self, trial: int, processingPass: int) -> typing.List[float]: 
        """
        This returns the vector of scalars indicating the max marker error on each timestep of a given trial
        """
    def getTrialMarkerRMSs(self, trial: int, processingPass: int) -> typing.List[float]: 
        """
        This returns the vector of scalars indicating the RMS marker error on each timestep of a given trial
        """
    def getTrialMaxJointVelocity(self, trial: int, processingPass: int) -> typing.List[float]: 
        """
        This returns the vector of scalars indicating the maximum absolute velocity of all DOFs on each timestep of a given trial
        """
    def getTrialName(self, trial: int) -> str: 
        """
        This returns the human readable name of the specified trial, given by the person who uploaded the data to AddBiomechanics. This isn't necessary for training, but may be useful for analyzing the data.
        """
    def getTrialNumProcessingPasses(self, trial: int) -> int: 
        """
        This returns the number of processing passes that successfully completed on this trial
        """
    def getTrialOriginalName(self, trial: int) -> str: 
        """
        This returns the original name of the trial before it was (potentially) split into multiple pieces
        """
    def getTrialSplitIndex(self, trial: int) -> int: 
        """
        This returns the index of the split, if this trial was the result of splitting an original trial into multiple pieces
        """
    def getTrialTags(self, trial: int) -> typing.List[str]: 
        """
        This returns the list of tags attached to a given trial index, which are arbitrary strings from the AddBiomechanics platform.
        """
    def getTrialTimestep(self, trial: int) -> float: 
        """
        This returns the timestep size for the trial requested, in seconds per frame
        """
    def loadAllFrames(self, doNotStandardizeForcePlateData: bool = False) -> None: 
        """
        This loads all the frames of data, and fills in the processing pass data matrices in the proto header classes.
        """
    def readForcePlates(self, arg0: int) -> typing.List[ForcePlate]: 
        """
        This reads all the raw sensor data for this trial, and constructs force plates.
        """
    def readFrames(self, trial: int, startFrame: int, numFramesToRead: int = 1, includeSensorData: bool = True, includeProcessingPasses: bool = True, stride: int = 1, contactThreshold: float = 1.0) -> FrameList: 
        """
        This will read from disk and allocate a number of :code:`Frame` objects. These Frame objects are assumed to be short-lived, to save working memory. For example, you might :code:`readFrames()` to construct a training batch, then immediately allow the frames to go out of scope and be released after the batch backpropagates gradient and loss. On OOB access, prints an error and returns an empty vector.
        """
    def readSkel(self, processingPass: int, geometryFolder: str = '') -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: 
        """
        This will read the skeleton from the binary, and optionally use the passed in :code:`geometryFolder` to load meshes. We do not bundle meshes with :code:`SubjectOnDisk` files, to save space. If you do not pass in :code:`geometryFolder`, expect to get warnings about being unable to load meshes, and expect that your skeleton will not display if you attempt to visualize it.
        """
    @staticmethod
    def writeB3D(path: str, header: SubjectOnDiskHeader) -> None: ...
    pass
class SubjectOnDiskHeader():
    def __init__(self) -> None: ...
    def addProcessingPass(self) -> SubjectOnDiskPassHeader: ...
    def addTrial(self) -> SubjectOnDiskTrial: ...
    def getProcessingPasses(self) -> typing.List[SubjectOnDiskPassHeader]: ...
    def getTrials(self) -> typing.List[SubjectOnDiskTrial]: ...
    def recomputeColumnNames(self) -> None: ...
    def setAgeYears(self, ageYears: int) -> SubjectOnDiskHeader: ...
    def setBiologicalSex(self, biologicalSex: str) -> SubjectOnDiskHeader: ...
    def setCustomValueNames(self, customValueNames: typing.List[str]) -> SubjectOnDiskHeader: ...
    def setGroundForceBodies(self, groundForceBodies: typing.List[str]) -> SubjectOnDiskHeader: ...
    def setHeightM(self, heightM: float) -> SubjectOnDiskHeader: ...
    def setHref(self, sourceHref: str) -> SubjectOnDiskHeader: ...
    def setMassKg(self, massKg: float) -> SubjectOnDiskHeader: ...
    def setNotes(self, notes: str) -> SubjectOnDiskHeader: ...
    def setNumDofs(self, dofs: int) -> SubjectOnDiskHeader: ...
    def setNumJoints(self, joints: int) -> SubjectOnDiskHeader: ...
    def setSubjectTags(self, subjectTags: typing.List[str]) -> SubjectOnDiskHeader: ...
    def setTrials(self, trials: typing.List[SubjectOnDiskTrial]) -> None: ...
    pass
class SubjectOnDiskPassHeader():
    def __init__(self) -> None: ...
    def getOpenSimFileText(self) -> str: ...
    def getProcessingPassType(self) -> ProcessingPassType: ...
    def setOpenSimFileText(self, openSimFileText: str) -> None: ...
    def setProcessingPassType(self, type: ProcessingPassType) -> None: ...
    pass
class SubjectOnDiskTrial():
    def __init__(self) -> None: ...
    def addPass(self) -> SubjectOnDiskTrialPass: 
        """
        This creates a new :code:`SubjectOnDiskTrialPass` for this trial, and returns it. That object can store results from IK and ID, as well as other results from the processing pipeline.
        """
    def getForcePlates(self) -> typing.List[ForcePlate]: ...
    def getMarkerObservations(self) -> typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]: ...
    def getMissingGRFReason(self) -> typing.List[MissingGRFReason]: ...
    def getOriginalTrialName(self) -> str: ...
    def getPasses(self) -> typing.List[SubjectOnDiskTrialPass]: ...
    def getSplitIndex(self) -> int: ...
    def getTimestep(self) -> float: ...
    def setAccObservations(self, accObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    def setCustomValues(self, customValues: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None: ...
    def setEmgObservations(self, emgObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[m, 1]]]]) -> None: ...
    def setExoTorques(self, exoTorques: typing.Dict[int, numpy.ndarray[numpy.float64, _Shape[m, 1]]]) -> None: ...
    def setForcePlates(self, forcePlates: typing.List[ForcePlate]) -> None: ...
    def setGyroObservations(self, gyroObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    def setMarkerNamesGuessed(self, markersGuessed: bool) -> None: ...
    def setMarkerObservations(self, markerObservations: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: ...
    def setMissingGRFReason(self, missingGRFReason: typing.List[MissingGRFReason]) -> None: ...
    def setName(self, name: str) -> None: ...
    def setOriginalTrialName(self, name: str) -> None: ...
    def setSplitIndex(self, split: int) -> None: ...
    def setTimestep(self, timestep: float) -> None: ...
    def setTrialTags(self, trialTags: typing.List[str]) -> None: ...
    pass
class SubjectOnDiskTrialPass():
    def __init__(self) -> None: ...
    def computeValues(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, timestep: float, poses: numpy.ndarray[numpy.float64, _Shape[m, n]], footBodyNames: typing.List[str], forces: numpy.ndarray[numpy.float64, _Shape[m, n]], moments: numpy.ndarray[numpy.float64, _Shape[m, n]], cops: numpy.ndarray[numpy.float64, _Shape[m, n]], rootHistoryLen: int = 5, rootHistoryStride: int = 1) -> None: ...
    def computeValuesFromForcePlates(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, timestep: float, poses: numpy.ndarray[numpy.float64, _Shape[m, n]], footBodyNames: typing.List[str], forcePlates: typing.List[ForcePlate], rootHistoryLen: int = 5, rootHistoryStride: int = 1, explicitVels: numpy.ndarray[numpy.float64, _Shape[m, n]] = array([], shape=(0, 0), dtype=float64), explicitAccs: numpy.ndarray[numpy.float64, _Shape[m, n]] = array([], shape=(0, 0), dtype=float64), forcePlateZeroThresholdNewtons: float = 3.0) -> None: ...
    def copyValuesFrom(self, other: SubjectOnDiskTrialPass) -> None: ...
    def getAccs(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getAngularResidual(self) -> typing.List[float]: ...
    def getComAccs(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getComAccsInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getComPoses(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getComVels(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getGroundBodyCopTorqueForce(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getGroundBodyCopTorqueForceInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getGroundBodyWrenches(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getGroundBodyWrenchesInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getJointCenters(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getJointCentersInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getLinearResidual(self) -> typing.List[float]: ...
    def getMarkerMax(self) -> typing.List[float]: ...
    def getMarkerRMS(self) -> typing.List[float]: ...
    def getPoses(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getResamplingMatrix(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getResidualWrenchInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getRootEulerHistoryInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getRootPosHistoryInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getRootSpatialAccInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getRootSpatialVelInRootFrame(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getTaus(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getVels(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def setAccs(self, accs: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setAngularResidual(self, angularResidual: typing.List[float]) -> None: ...
    def setComAccs(self, accs: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setComAccsInRootFrame(self, accs: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setComPoses(self, poses: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setComVels(self, vels: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setDofAccelerationFiniteDifferenced(self, dofAccelerationFiniteDifference: typing.List[bool]) -> None: ...
    def setDofPositionsObserved(self, dofPositionsObserved: typing.List[bool]) -> None: ...
    def setDofVelocitiesFiniteDifferenced(self, dofVelocitiesFiniteDifferenced: typing.List[bool]) -> None: ...
    def setForcePlateCutoffs(self, cutoffs: typing.List[float]) -> None: ...
    def setGroundBodyCopTorqueForce(self, copTorqueForces: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setGroundBodyCopTorqueForceInRootFrame(self, copTorqueForces: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setGroundBodyWrenches(self, wrenches: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setGroundBodyWrenchesInRootFrame(self, wrenches: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setJointCenters(self, centers: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setJointCentersInRootFrame(self, centers: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setLinearResidual(self, linearResidual: typing.List[float]) -> None: ...
    def setLowpassCutoffFrequency(self, freq: float) -> None: ...
    def setLowpassFilterOrder(self, order: int) -> None: ...
    def setMarkerMax(self, markerMax: typing.List[float]) -> None: ...
    def setMarkerRMS(self, markerRMS: typing.List[float]) -> None: ...
    def setPoses(self, poses: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setResamplingMatrix(self, resamplingMatrix: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setResidualWrenchInRootFrame(self, wrenches: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setRootEulerHistoryInRootFrame(self, rootHistory: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setRootPosHistoryInRootFrame(self, rootHistory: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setRootSpatialAccInRootFrame(self, spatialAcc: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setRootSpatialVelInRootFrame(self, spatialVel: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setTaus(self, taus: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def setType(self, type: ProcessingPassType) -> None: ...
    def setVels(self, vels: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    pass
forceDiscrepancy: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.forceDiscrepancy: 4>
interpolatedClippedGRF: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.interpolatedClippedGRF: 10>
manualReview: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.manualReview: 9>
measuredGrfZeroWhenAccelerationNonZero: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.measuredGrfZeroWhenAccelerationNonZero: 1>
missingBlip: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.missingBlip: 7>
missingImpact: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.missingImpact: 6>
no: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFStatus # value = <MissingGRFStatus.no: 0>
notMissingGRF: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.notMissingGRF: 0>
notOverForcePlate: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.notOverForcePlate: 5>
shiftGRF: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.shiftGRF: 8>
torqueDiscrepancy: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.torqueDiscrepancy: 3>
unknown: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFStatus # value = <MissingGRFStatus.unknown: 1>
unmeasuredExternalForceDetected: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFReason # value = <MissingGRFReason.unmeasuredExternalForceDetected: 2>
yes: nimblephysics_libs._nimblephysics.biomechanics.MissingGRFStatus # value = <MissingGRFStatus.yes: 2>
