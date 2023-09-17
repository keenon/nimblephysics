"""
This provides biomechanics utilities in Nimble, including inverse dynamics and (eventually) mocap support and muscle estimation.
"""
from __future__ import annotations
import _nimblephysics.dynamics
import _nimblephysics.math
import _nimblephysics.neural
import _nimblephysics.server
import numpy
import typing
from . import OpenSimParser
__all__ = ['Anthropometrics', 'BatchGaitInverseDynamics', 'BilevelFitResult', 'C3D', 'C3DLoader', 'ContactRegimeSection', 'DynamicsFitProblemConfig', 'DynamicsFitter', 'DynamicsInitialization', 'ForcePlate', 'Frame', 'IKErrorReport', 'IMUFineTuneProblem', 'InitialMarkerFitParams', 'LabelledMarkers', 'LilypadSolver', 'MarkerFitter', 'MarkerFitterState', 'MarkerFixer', 'MarkerInitialization', 'MarkerLabeller', 'MarkerLabellerMock', 'MarkerTrace', 'MarkersErrorReport', 'MissingGRFReason', 'NeuralMarkerLabeller', 'OpenSimFile', 'OpenSimMocoTrajectory', 'OpenSimMot', 'OpenSimParser', 'OpenSimScaleAndMarkerOffsets', 'OpenSimTRC', 'ResidualForceHelper', 'SkeletonConverter', 'SubjectOnDisk', 'forceDiscrepancy', 'measuredGrfZeroWhenAccelerationNonZero', 'missingBlip', 'missingImpact', 'notMissingGRF', 'notOverForcePlate', 'shiftGRF', 'torqueDiscrepancy', 'unmeasuredExternalForceDetected']
class Anthropometrics:
    @staticmethod
    def loadFromFile(uri: str) -> Anthropometrics:
        ...
    def addMetric(self, name: str, bodyPose: numpy.ndarray[numpy.float64[m, 1]], bodyA: str, offsetA: numpy.ndarray[numpy.float64[3, 1]], bodyB: str, offsetB: numpy.ndarray[numpy.float64[3, 1]], axis: numpy.ndarray[numpy.float64[3, 1]] = ...) -> None:
        ...
    def condition(self, observedValues: dict[str, float]) -> Anthropometrics:
        ...
    def debugToGUI(self, server: _nimblephysics.server.GUIWebsocketServer, skel: _nimblephysics.dynamics.Skeleton) -> None:
        ...
    def getDistribution(self) -> _nimblephysics.math.MultivariateGaussian:
        ...
    def getGradientOfLogPDFWrtBodyScales(self, skel: _nimblephysics.dynamics.Skeleton) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def getGradientOfLogPDFWrtGroupScales(self, skel: _nimblephysics.dynamics.Skeleton) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def getLogPDF(self, skel: _nimblephysics.dynamics.Skeleton, normalized: bool = ...) -> float:
        ...
    def getMetricNames(self) -> list[str]:
        ...
    def getPDF(self, skel: _nimblephysics.dynamics.Skeleton) -> float:
        ...
    def measure(self, skel: _nimblephysics.dynamics.Skeleton) -> dict[str, float]:
        ...
    def setDistribution(self, dist: _nimblephysics.math.MultivariateGaussian) -> None:
        ...
class BatchGaitInverseDynamics:
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton, poses: numpy.ndarray[numpy.float64[m, n]], groundContactBodies: list[_nimblephysics.dynamics.BodyNode], groundNormal: numpy.ndarray[numpy.float64[3, 1]], tileSize: float, maxSectionLength: int = ..., smoothingWeight: float = ..., minTorqueWeight: float = ..., prevContactWeight: float = ..., blendWeight: float = ..., blendSteepness: float = ...) -> None:
        ...
    def debugLilypadToGUI(self, gui: _nimblephysics.server.GUIWebsocketServer) -> None:
        ...
    def debugTimestepToGUI(self, gui: _nimblephysics.server.GUIWebsocketServer, timesteps: int) -> None:
        ...
    def getContactBodiesAtTimestep(self, timestep: int) -> list[_nimblephysics.dynamics.BodyNode]:
        ...
    def getContactWrenchesAtTimestep(self, timestep: int) -> list[numpy.ndarray[numpy.float64[6, 1]]]:
        ...
    def getSectionForTimestep(self, timestep: int) -> ContactRegimeSection:
        ...
    def numTimesteps(self) -> int:
        ...
class BilevelFitResult:
    groupScales: numpy.ndarray[numpy.float64[m, 1]]
    markerOffsets: dict[str, numpy.ndarray[numpy.float64[3, 1]]]
    poses: list[numpy.ndarray[numpy.float64[m, 1]]]
    posesMatrix: numpy.ndarray[numpy.float64[m, n]]
    rawMarkerOffsets: numpy.ndarray[numpy.float64[m, 1]]
    success: bool
class C3D:
    dataRotation: numpy.ndarray[numpy.float64[3, 3]]
    forcePlates: list[ForcePlate]
    framesPerSecond: int
    markerTimesteps: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]
    markers: list[str]
    shuffledMarkersMatrix: numpy.ndarray[numpy.float64[m, n]]
    shuffledMarkersMatrixMask: numpy.ndarray[numpy.float64[m, n]]
    timestamps: list[float]
class C3DLoader:
    @staticmethod
    def debugToGUI(file: C3D, server: _nimblephysics.server.GUIWebsocketServer) -> None:
        ...
    @staticmethod
    def fixupMarkerFlips(c3d: C3D) -> list[list[tuple[str, str]]]:
        ...
    @staticmethod
    def loadC3D(uri: str) -> C3D:
        ...
class ContactRegimeSection:
    endTime: int
    groundContactBodies: list[_nimblephysics.dynamics.BodyNode]
    startTime: int
    wrenches: list[list[numpy.ndarray[numpy.float64[6, 1]]]]
class DynamicsFitProblemConfig:
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton) -> None:
        ...
    def setBoundMoveDistance(self, distance: float) -> DynamicsFitProblemConfig:
        ...
    def setConstrainAngularResiduals(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setConstrainLinearResiduals(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setConstrainResidualsZero(self, constrain: bool) -> DynamicsFitProblemConfig:
        ...
    def setDefaults(self, useL1: bool = ...) -> DynamicsFitProblemConfig:
        ...
    def setDisableBounds(self, disable: bool) -> DynamicsFitProblemConfig:
        ...
    def setIncludeBodyScales(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setIncludeCOMs(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setIncludeInertias(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setIncludeMarkerOffsets(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setIncludeMasses(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setIncludePoses(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setJointWeight(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setLinearNewtonUseL1(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setLinearNewtonWeight(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setLogLossDetails(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setMarkerUseL1(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setMarkerWeight(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setMaxBlockSize(self, value: int) -> DynamicsFitProblemConfig:
        ...
    def setMaxNumBlocksPerTrial(self, value: int) -> DynamicsFitProblemConfig:
        ...
    def setMaxNumTrials(self, value: int) -> DynamicsFitProblemConfig:
        ...
    def setNumThreads(self, value: int) -> DynamicsFitProblemConfig:
        ...
    def setOnlyOneTrial(self, value: int) -> DynamicsFitProblemConfig:
        ...
    def setPoseSubsetLen(self, value: int) -> DynamicsFitProblemConfig:
        ...
    def setPoseSubsetStartIndex(self, value: int) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeAnatomicalMarkerOffsets(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeBodyScales(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeCOMs(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeImpliedDensity(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeInertias(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeJointAcc(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeMasses(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizePoses(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeSpatialAcc(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeSpatialAccBodyWeights(self, bodyWeights: numpy.ndarray[numpy.float64[m, 1]]) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeSpatialAccUseL1(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setRegularizeTrackingMarkerOffsets(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setResidualTorqueMultiple(self, value: float) -> DynamicsFitProblemConfig:
        ...
    def setResidualUseL1(self, value: bool) -> DynamicsFitProblemConfig:
        ...
    def setResidualWeight(self, value: float) -> DynamicsFitProblemConfig:
        ...
class DynamicsFitter:
    @staticmethod
    @typing.overload
    def createInitialization(skel: _nimblephysics.dynamics.Skeleton, markerMap: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]], trackingMarkers: list[str], grfNodes: list[_nimblephysics.dynamics.BodyNode], forcePlateTrials: list[list[ForcePlate]], poseTrials: list[numpy.ndarray[numpy.float64[m, n]]], framesPerSecond: list[int], markerObservationTrials: list[list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]], overrideForcePlateToGRFNodeAssignment: list[list[int]] = ...) -> DynamicsInitialization:
        ...
    @staticmethod
    @typing.overload
    def createInitialization(skel: _nimblephysics.dynamics.Skeleton, kinematicInits: list[MarkerInitialization], trackingMarkers: list[str], grfNodes: list[_nimblephysics.dynamics.BodyNode], forcePlateTrials: list[list[ForcePlate]], framesPerSecond: list[int], markerObservationTrials: list[list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]], overrideForcePlateToGRFNodeAssignment: list[list[int]] = ...) -> DynamicsInitialization:
        ...
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton, footNodes: list[_nimblephysics.dynamics.BodyNode], trackingMarkers: list[str]) -> None:
        ...
    def addJointBoundSlack(self, init: _nimblephysics.dynamics.Skeleton, slack: float) -> None:
        ...
    def applyInitToSkeleton(self, skel: _nimblephysics.dynamics.Skeleton, init: DynamicsInitialization) -> None:
        ...
    def boundPush(self, init: DynamicsInitialization, boundPush: float = ...) -> None:
        ...
    def checkPhysicalConsistency(self, init: DynamicsInitialization, maxAcceptableErrors: float = ..., maxTimestepsToTest: int = ...) -> bool:
        ...
    def comAccelerations(self, init: DynamicsInitialization, trial: int) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        ...
    def comPositions(self, init: DynamicsInitialization, trial: int) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        ...
    def computeAverageCOPChange(self, init: DynamicsInitialization) -> float:
        ...
    def computeAverageForceMagnitudeChange(self, init: DynamicsInitialization) -> float:
        ...
    def computeAverageForceVectorChange(self, init: DynamicsInitialization) -> float:
        ...
    def computeAverageMarkerMaxError(self, init: DynamicsInitialization) -> float:
        ...
    def computeAverageMarkerRMSE(self, init: DynamicsInitialization) -> float:
        ...
    def computeAverageRealForce(self, init: DynamicsInitialization) -> tuple[float, float]:
        ...
    def computeAverageResidualForce(self, init: DynamicsInitialization) -> tuple[float, float]:
        ...
    def computeAverageTrialCOPChange(self, init: DynamicsInitialization, trial: int) -> float:
        ...
    def computeAverageTrialForceMagnitudeChange(self, init: DynamicsInitialization, trial: int) -> float:
        ...
    def computeAverageTrialForceVectorChange(self, init: DynamicsInitialization, trial: int) -> float:
        ...
    def computeAverageTrialMarkerMaxError(self, init: DynamicsInitialization, trial: int) -> float:
        ...
    def computeAverageTrialMarkerRMSE(self, init: DynamicsInitialization, trial: int) -> float:
        ...
    def computeAverageTrialRealForce(self, init: DynamicsInitialization, trial: int) -> tuple[float, float]:
        ...
    def computeAverageTrialResidualForce(self, init: DynamicsInitialization, trial: int) -> tuple[float, float]:
        ...
    def computeInverseDynamics(self, init: DynamicsInitialization, trial: int) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def computePerfectGRFs(self, init: DynamicsInitialization) -> None:
        ...
    def estimateFootGroundContacts(self, init: DynamicsInitialization, ignoreFootNotOverForcePlate: bool = ...) -> None:
        ...
    def estimateLinkMassesFromAcceleration(self, init: DynamicsInitialization, regularizationWeight: float = ...) -> None:
        ...
    def impliedCOMForces(self, init: DynamicsInitialization, trial: int, includeGravity: numpy.ndarray[numpy.float64[3, 1]] = ...) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        ...
    def measuredGRFForces(self, init: DynamicsInitialization, trial: int) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        ...
    def multimassZeroLinearResidualsOnCOMTrajectory(self, init: DynamicsInitialization, maxTrialsToSolveMassOver: int = ..., boundPush: float = ...) -> None:
        ...
    def optimizeMarkerOffsets(self, init: DynamicsInitialization, reoptimizeAnatomicalMarkers: bool = ..., reoptimizeTrackingMarkers: bool = ...) -> None:
        ...
    def optimizeSpatialResidualsOnCOMTrajectory(self, init: DynamicsInitialization, trial: int, satisfactoryThreshold: float = ..., numIters: int = ..., missingResidualRegularization: float = ..., weightAngular: float = ..., weightLastFewTimesteps: float = ..., offsetRegularization: float = ..., regularizeResiduals: bool = ...) -> bool:
        ...
    def recalibrateForcePlates(self, init: DynamicsInitialization, trial: int, maxMovement: float = ...) -> None:
        ...
    def runConstrainedSGDOptimization(self, init: DynamicsInitialization, config: DynamicsFitProblemConfig) -> None:
        ...
    def runIPOPTOptimization(self, init: DynamicsInitialization, config: DynamicsFitProblemConfig) -> None:
        ...
    def runUnconstrainedSGDOptimization(self, init: DynamicsInitialization, config: DynamicsFitProblemConfig) -> None:
        ...
    def saveDynamicsToGUI(self, path: str, init: DynamicsInitialization, trialIndex: int, framesPerSecond: int) -> None:
        ...
    def scaleLinkMassesFromGravity(self, init: DynamicsInitialization) -> None:
        ...
    def setCheckDerivatives(self, value: bool) -> None:
        ...
    def setDisableLinesearch(self, value: bool) -> None:
        ...
    def setIterationLimit(self, value: int) -> None:
        ...
    def setLBFGSHistoryLength(self, value: int) -> None:
        ...
    def setPrintFrequency(self, value: int) -> None:
        ...
    def setSilenceOutput(self, value: bool) -> None:
        ...
    def setTolerance(self, value: float) -> None:
        ...
    def smoothAccelerations(self, init: DynamicsInitialization, smoothingWeight: float = ..., regularizationWeight: float = ...) -> None:
        ...
    def timeSyncAndInitializePipeline(self, init: DynamicsInitialization, useReactionWheels: bool = ..., shiftGRF: bool = ..., maxShiftGRF: int = ..., iterationsPerShift: int = ..., maxTrialsToSolveMassOver: int = ..., weightLinear: float = ..., weightAngular: float = ..., regularizeLinearResiduals: float = ..., regularizeAngularResiduals: float = ..., regularizeCopDriftCompensation: float = ..., maxBuckets: int = ..., detectUnmeasuredTorque: bool = ..., avgPositionChangeThreshold: float = ..., avgAngularChangeThreshold: float = ..., reoptimizeAnatomicalMarkers: bool = ..., reoptimizeTrackingMarkers: bool = ...) -> bool:
        ...
    def timeSyncTrialGRF(self, init: DynamicsInitialization, trial: int, useReactionWheels: bool = ..., maxShiftGRF: int = ..., iterationsPerShift: int = ..., weightLinear: float = ..., weightAngular: float = ..., regularizeLinearResiduals: float = ..., regularizeAngularResiduals: float = ..., regularizeCopDriftCompensation: float = ..., maxBuckets: int = ...) -> bool:
        ...
    def writeCSVData(self, path: str, init: DynamicsInitialization, trialIndex: int, useAdjustedGRFs: bool = ..., timestamps: list[float] = ...) -> None:
        ...
    def writeSubjectOnDisk(self, outputPath: str, openSimFilePath: str, init: DynamicsInitialization, biologicalSex: str, massKg: float, heightM: float, ageYears: int, useAdjustedGRFs: bool = ..., trialNames: list[str] = ..., subjectTags: list[str] = ..., trialTags: list[list[str]] = ..., href: str = ..., notes: str = ..., emgObservationTrials: list[list[dict[str, numpy.ndarray[numpy.float64[m, 1]]]]] = ...) -> None:
        ...
    def zeroLinearResidualsAndOptimizeAngular(self, init: DynamicsInitialization, trial: int, targetPoses: numpy.ndarray[numpy.float64[m, n]], previousTotalResidual: float, iteration: int, useReactionWheels: bool = ..., weightLinear: float = ..., weightAngular: float = ..., regularizeLinearResiduals: float = ..., regularizeAngularResiduals: float = ..., regularizeCopDriftCompensation: float = ..., maxBuckets: int = ..., maxLeastSquaresIters: int = ..., commitCopDriftCompensation: bool = ..., detectUnmeasuredTorque: bool = ..., avgPositionChangeThreshold: float = ..., avgAngularChangeThreshold: float = ...) -> tuple[bool, float]:
        ...
    def zeroLinearResidualsOnCOMTrajectory(self, init: DynamicsInitialization, maxTrialsToSolveMassOver: int = ..., detectExternalForce: bool = ..., driftCorrectionBlurRadius: int = ..., driftCorrectionBlurInterval: int = ...) -> bool:
        ...
class DynamicsInitialization:
    axisWeights: numpy.ndarray[numpy.float64[m, 1]]
    bodyCom: numpy.ndarray[numpy.float64[3, n]]
    bodyInertia: numpy.ndarray[numpy.float64[6, n]]
    bodyMasses: numpy.ndarray[numpy.float64[m, 1]]
    contactBodies: list[list[_nimblephysics.dynamics.BodyNode]]
    defaultForcePlateCorners: list[list[numpy.ndarray[numpy.float64[3, 1]]]]
    flatGround: list[bool]
    forcePlateTrials: list[list[ForcePlate]]
    grfBodyContactSphereRadius: list[list[list[float]]]
    grfBodyForceActive: list[list[list[bool]]]
    grfBodyIndices: list[int]
    grfBodyNodes: list[_nimblephysics.dynamics.BodyNode]
    grfBodyOffForcePlate: list[list[list[bool]]]
    grfBodySphereInContact: list[list[list[bool]]]
    grfTrials: list[numpy.ndarray[numpy.float64[m, n]]]
    groundHeight: list[float]
    groupMasses: numpy.ndarray[numpy.float64[m, 1]]
    groupScales: numpy.ndarray[numpy.float64[m, 1]]
    includeTrialsInDynamicsFit: list[bool]
    initialGroupCOMs: numpy.ndarray[numpy.float64[m, 1]]
    initialGroupInertias: numpy.ndarray[numpy.float64[m, 1]]
    initialGroupMasses: numpy.ndarray[numpy.float64[m, 1]]
    initialGroupScales: numpy.ndarray[numpy.float64[m, 1]]
    initialMarkerOffsets: dict[str, numpy.ndarray[numpy.float64[3, 1]]]
    jointAxis: list[numpy.ndarray[numpy.float64[m, n]]]
    jointCenters: list[numpy.ndarray[numpy.float64[m, n]]]
    jointWeights: numpy.ndarray[numpy.float64[m, 1]]
    joints: list[_nimblephysics.dynamics.Joint]
    jointsAdjacentMarkers: list[list[str]]
    markerObservationTrials: list[list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]]
    markerOffsets: dict[str, numpy.ndarray[numpy.float64[3, 1]]]
    missingGRFReason: list[list[MissingGRFReason]]
    originalPoses: list[numpy.ndarray[numpy.float64[m, n]]]
    poseTrials: list[numpy.ndarray[numpy.float64[m, n]]]
    probablyMissingGRF: list[list[bool]]
    regularizeGroupCOMsTo: numpy.ndarray[numpy.float64[m, 1]]
    regularizeGroupInertiasTo: numpy.ndarray[numpy.float64[m, 1]]
    regularizeGroupMassesTo: numpy.ndarray[numpy.float64[m, 1]]
    regularizeGroupScalesTo: numpy.ndarray[numpy.float64[m, 1]]
    regularizeMarkerOffsetsTo: dict[str, numpy.ndarray[numpy.float64[3, 1]]]
    trackingMarkers: list[str]
    trialTimesteps: list[float]
    updatedMarkerMap: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]]
class ForcePlate:
    centersOfPressure: list[numpy.ndarray[numpy.float64[3, 1]]]
    corners: list[numpy.ndarray[numpy.float64[3, 1]]]
    forces: list[numpy.ndarray[numpy.float64[3, 1]]]
    moments: list[numpy.ndarray[numpy.float64[3, 1]]]
    timestamps: list[float]
    worldOrigin: numpy.ndarray[numpy.float64[3, 1]]
    @staticmethod
    def copyForcePlate(plate: ForcePlate) -> ForcePlate:
        ...
    def __init__(self) -> None:
        ...
    def trim(self, newStartTime: float, newEndTime: float) -> None:
        ...
    def trimToIndexes(self, start: int, end: int) -> None:
        ...
class Frame:
    """
    
            This is for doing ML and large-scale data analysis. This is a single frame of data, returned in a list by :code:`SubjectOnDisk.readFrames()`, which contains everything needed to reconstruct all the dynamics of a snapshot in time.
          
    """
    @property
    def acc(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        The joint accelerations on this frame.
        """
    @acc.setter
    def acc(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def accFiniteDifferenced(self) -> numpy.ndarray[numpy.int32[m, 1]]:
        """
        A boolean mask of [0,1]s for each DOF, with a 1 indicating that this DOF got its acceleration through finite differencing, and therefore may be somewhat unreliable
        """
    @accFiniteDifferenced.setter
    def accFiniteDifferenced(self, arg0: numpy.ndarray[numpy.int32[m, 1]]) -> None:
        ...
    @property
    def accObservations(self) -> list[tuple[str, numpy.ndarray[numpy.float64[3, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the accelerometers observations at this frame. Accelerometers that were not observed (perhaps due to time offsets in uploaded data) will not be present in this list. For the full specification of the accelerometer set, load the model from the :code:`SubjectOnDisk`
        """
    @accObservations.setter
    def accObservations(self, arg0: list[tuple[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
    @property
    def comAcc(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        The acceleration of the COM, in world space
        """
    @comAcc.setter
    def comAcc(self, arg0: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    @property
    def comPos(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        The position of the COM, in world space
        """
    @comPos.setter
    def comPos(self, arg0: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    @property
    def comVel(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        The velocity of the COM, in world space
        """
    @comVel.setter
    def comVel(self, arg0: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    @property
    def contact(self) -> numpy.ndarray[numpy.int32[m, 1]]:
        """
        A vector of [0,1] booleans for if a body is in contact with the ground.
        """
    @contact.setter
    def contact(self, arg0: numpy.ndarray[numpy.int32[m, 1]]) -> None:
        ...
    @property
    def customValues(self) -> list[tuple[str, numpy.ndarray[numpy.float64[m, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of unspecified values. The idea here is to allow the format to be easily extensible with unusual data (for example, exoskeleton torques) without bloating ordinary training files.
        """
    @customValues.setter
    def customValues(self, arg0: list[tuple[str, numpy.ndarray[numpy.float64[m, 1]]]]) -> None:
        ...
    @property
    def emgSignals(self) -> list[tuple[str, numpy.ndarray[numpy.float64[m, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the EMG signals at this frame. EMG signals are generally preserved at a higher sampling frequency than the motion capture, so the `np.ndarray` vector will be a number of samples that were captured during this single motion capture frame. For example, if EMG is at 1000Hz and mocap is at 100Hz, the `np.ndarray` vector will be of length 10.
        """
    @emgSignals.setter
    def emgSignals(self, arg0: list[tuple[str, numpy.ndarray[numpy.float64[m, 1]]]]) -> None:
        ...
    @property
    def groundContactCenterOfPressure(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`CoP` values for each contact body, where :code:`CoP` is a 3 vector representing the center of pressure for a contact measured on the force plate. :code:`CoP` is 
                    expressed in the world frame.
        """
    @groundContactCenterOfPressure.setter
    def groundContactCenterOfPressure(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def groundContactForce(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`f` values for each contact body, where :code:`f` is a 3 vector representing the ground-reaction force from a contact, measured on the force plate. :code:`f` is 
                    expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
        """
    @groundContactForce.setter
    def groundContactForce(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def groundContactTorque(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
                    This is a vector of all the concatenated :code:`tau` values for each contact body, where :code:`tau` is a 3 vector representing the ground-reaction torque from a contact, measured on the force plate. :code:`tau` is 
                    expressed in the world frame, and is assumed to be acting at the corresponding :code:`CoP` from the same index in :code:`groundContactCenterOfPressure`.
        """
    @groundContactTorque.setter
    def groundContactTorque(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def groundContactWrenches(self) -> numpy.ndarray[numpy.float64[m, 1]]:
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
        """
    @groundContactWrenches.setter
    def groundContactWrenches(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def gyroObservations(self) -> list[tuple[str, numpy.ndarray[numpy.float64[3, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the gyroscope observations at this frame. Gyroscopes that were not observed (perhaps due to time offsets in uploaded data) will not be present in this list. For the full specification of the gyroscope set, load the model from the :code:`SubjectOnDisk`
        """
    @gyroObservations.setter
    def gyroObservations(self, arg0: list[tuple[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
    @property
    def markerObservations(self) -> list[tuple[str, numpy.ndarray[numpy.float64[3, 1]]]]:
        """
        This is list of :code:`Pair[str, np.ndarray]` of the marker observations at this frame. Markers that were not observed will not be present in this list. For the full specification of the markerset, load the model from the :code:`SubjectOnDisk`
        """
    @markerObservations.setter
    def markerObservations(self, arg0: list[tuple[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
    @property
    def pos(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        The joint positions on this frame.
        """
    @pos.setter
    def pos(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def posObserved(self) -> numpy.ndarray[numpy.int32[m, 1]]:
        """
        A boolean mask of [0,1]s for each DOF, with a 1 indicating that this DOF was observed on this frame
        """
    @posObserved.setter
    def posObserved(self, arg0: numpy.ndarray[numpy.int32[m, 1]]) -> None:
        ...
    @property
    def probablyMissingGRF(self) -> bool:
        """
                    This is true if this frame probably has unmeasured forces acting on the body. For example, if a subject
                    steps off of the available force plates during this frame, this will probably be true.
        
                    WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on this frame!!
        """
    @probablyMissingGRF.setter
    def probablyMissingGRF(self, arg0: bool) -> None:
        ...
    @property
    def rawForcePlateCenterOfPressures(self) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        """
        This is list of :code:`np.ndarray` of the original center of pressure readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.
        """
    @rawForcePlateCenterOfPressures.setter
    def rawForcePlateCenterOfPressures(self, arg0: list[numpy.ndarray[numpy.float64[3, 1]]]) -> None:
        ...
    @property
    def rawForcePlateForces(self) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        """
        This is list of :code:`np.ndarray` of the original force readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.
        """
    @rawForcePlateForces.setter
    def rawForcePlateForces(self, arg0: list[numpy.ndarray[numpy.float64[3, 1]]]) -> None:
        ...
    @property
    def rawForcePlateTorques(self) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        """
        This is list of :code:`np.ndarray` of the original torque readings on each force plate, without any processing by AddBiomechanics. These are the original inputs that were used to create this SubjectOnDisk.
        """
    @rawForcePlateTorques.setter
    def rawForcePlateTorques(self, arg0: list[numpy.ndarray[numpy.float64[3, 1]]]) -> None:
        ...
    @property
    def residual(self) -> float:
        """
        The norm of the root residual force on this trial.
        """
    @residual.setter
    def residual(self, arg0: float) -> None:
        ...
    @property
    def t(self) -> int:
        """
        The frame number in this trial.
        """
    @t.setter
    def t(self, arg0: int) -> None:
        ...
    @property
    def tau(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        The joint control forces on this frame.
        """
    @tau.setter
    def tau(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def trial(self) -> int:
        """
        The index of the trial in the containing SubjectOnDisk.
        """
    @trial.setter
    def trial(self, arg0: int) -> None:
        ...
    @property
    def vel(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        The joint velocities on this frame.
        """
    @vel.setter
    def vel(self, arg0: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    @property
    def velFiniteDifferenced(self) -> numpy.ndarray[numpy.int32[m, 1]]:
        """
        A boolean mask of [0,1]s for each DOF, with a 1 indicating that this DOF got its velocity through finite differencing, and therefore may be somewhat unreliable
        """
    @velFiniteDifferenced.setter
    def velFiniteDifferenced(self, arg0: numpy.ndarray[numpy.int32[m, 1]]) -> None:
        ...
class IKErrorReport:
    averageMaxError: float
    averageRootMeanSquaredError: float
    averageSumSquaredError: float
    maxError: list[float]
    rootMeanSquaredError: list[float]
    sumSquaredError: list[float]
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton, markers: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]], poses: numpy.ndarray[numpy.float64[m, n]], observations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
    def getSortedMarkerRMSE(self) -> list[tuple[str, float]]:
        ...
    def printReport(self, limitTimesteps: int = ...) -> None:
        ...
    def saveCSVMarkerErrorReport(self, path: str) -> None:
        ...
class IMUFineTuneProblem:
    def flatten(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def getAccs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def getGrad(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def getLoss(self) -> float:
        ...
    def getPoses(self) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def getProblemSize(self) -> int:
        ...
    def getVels(self) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def setWeightAccs(self, weight: float) -> None:
        ...
    def setWeightGyros(self, weight: float) -> None:
        ...
    def setWeightPoses(self, weight: float) -> None:
        ...
    def unflatten(self, x: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
class InitialMarkerFitParams:
    dontRescaleBodies: bool
    groupScales: numpy.ndarray[numpy.float64[m, 1]]
    initPoses: numpy.ndarray[numpy.float64[m, n]]
    jointCenters: numpy.ndarray[numpy.float64[m, n]]
    jointWeights: numpy.ndarray[numpy.float64[m, 1]]
    joints: list[_nimblephysics.dynamics.Joint]
    markerOffsets: dict[str, numpy.ndarray[numpy.float64[3, 1]]]
    markerWeights: dict[str, float]
    maxTimestepsToUseForMultiTrialScaling: int
    maxTrialsToUseForMultiTrialScaling: int
    numBlocks: int
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def setDontRescaleBodies(self, dontRescaleBodies: bool) -> InitialMarkerFitParams:
        ...
    def setGroupScales(self, groupScales: numpy.ndarray[numpy.float64[m, 1]]) -> InitialMarkerFitParams:
        ...
    def setInitPoses(self, initPoses: numpy.ndarray[numpy.float64[m, n]]) -> InitialMarkerFitParams:
        ...
    def setJointCenters(self, joints: list[_nimblephysics.dynamics.Joint], jointCenters: numpy.ndarray[numpy.float64[m, n]], jointAdjacentMarkers: list[list[str]]) -> InitialMarkerFitParams:
        ...
    def setJointCentersAndWeights(self, joints: list[_nimblephysics.dynamics.Joint], jointCenters: numpy.ndarray[numpy.float64[m, n]], jointAdjacentMarkers: list[list[str]], jointWeights: numpy.ndarray[numpy.float64[m, 1]]) -> InitialMarkerFitParams:
        ...
    def setMarkerOffsets(self, markerOffsets: dict[str, numpy.ndarray[numpy.float64[3, 1]]]) -> InitialMarkerFitParams:
        ...
    def setMarkerWeights(self, markerWeights: dict[str, float]) -> InitialMarkerFitParams:
        ...
    def setMaxTimestepsToUseForMultiTrialScaling(self, numTimesteps: int) -> InitialMarkerFitParams:
        ...
    def setMaxTrialsToUseForMultiTrialScaling(self, numTrials: int) -> InitialMarkerFitParams:
        ...
    def setNumBlocks(self, numBlocks: int) -> InitialMarkerFitParams:
        ...
    def setNumIKTries(self, tries: int) -> InitialMarkerFitParams:
        ...
class LabelledMarkers:
    jointCenterGuesses: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]
    markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]
    markerOffsets: dict[str, tuple[str, numpy.ndarray[numpy.float64[3, 1]]]]
    traces: list[MarkerTrace]
class LilypadSolver:
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton, groundContactBodies: list[_nimblephysics.dynamics.BodyNode], groundNormal: numpy.ndarray[numpy.float64[3, 1]], tileSize: float) -> None:
        ...
    def clear(self) -> None:
        ...
    def debugToGUI(self, gui: _nimblephysics.server.GUIWebsocketServer) -> None:
        ...
    def getContactBodies(self) -> list[_nimblephysics.dynamics.BodyNode]:
        ...
    def process(self, poses: numpy.ndarray[numpy.float64[m, n]], startTime: int = ...) -> None:
        ...
    def setLateralVelThreshold(self, threshold: float) -> None:
        ...
    def setVerticalAccelerationThreshold(self, threshold: float) -> None:
        ...
    def setVerticalVelThreshold(self, threshold: float) -> None:
        ...
class MarkerFitter:
    @staticmethod
    def pickSubset(markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], subsetSize: int) -> list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]:
        ...
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton, markers: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]], ignoreVirtualJointCenterMarkers: bool = ...) -> None:
        ...
    def addZeroConstraint(self, name: str, loss: typing.Callable[[_nimblephysics.biomechanics.MarkerFitterState], float]) -> None:
        ...
    def autorotateC3D(self, c3d: C3D) -> None:
        ...
    def checkForEnoughMarkers(self, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> bool:
        ...
    def checkForFlippedMarkers(self, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], init: MarkerInitialization, report: MarkersErrorReport) -> bool:
        ...
    def debugTrajectoryAndMarkersToGUI(self, server: _nimblephysics.server.GUIWebsocketServer, init: MarkerInitialization, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], forcePlates: list[ForcePlate] = ..., goldOsim: OpenSimFile = ..., goldPoses: numpy.ndarray[numpy.float64[m, n]] = ...) -> None:
        ...
    def findJointCenters(self, initializations: MarkerInitialization, newClip: list[bool], markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
    def fineTuneWithIMU(self, accObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], gyroObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], newClip: list[bool], init: MarkerInitialization, dt: float, weightAccs: float = ..., weightGyros: float = ..., weightMarkers: float = ..., regularizePoses: float = ..., useIPOPT: bool = ..., iterations: int = ..., lbfgsMemory: int = ...) -> MarkerInitialization:
        ...
    def generateDataErrorsReport(self, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], dt: float) -> MarkersErrorReport:
        ...
    def getIMUFineTuneProblem(self, accObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], gyroObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], init: MarkerInitialization, dt: float, start: int, end: int) -> IMUFineTuneProblem:
        ...
    def getImuList(self) -> list[tuple[_nimblephysics.dynamics.BodyNode, _nimblephysics.math.Isometry3]]:
        ...
    def getImuMap(self) -> dict[str, tuple[_nimblephysics.dynamics.BodyNode, _nimblephysics.math.Isometry3]]:
        ...
    def getImuNames(self) -> list[str]:
        ...
    def getInitialization(self, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], newClip: list[bool], params: InitialMarkerFitParams = ...) -> MarkerInitialization:
        ...
    def getMarkerIsTracking(self, marker: str) -> bool:
        ...
    def getNumMarkers(self) -> int:
        ...
    def measureAccelerometerRMS(self, accObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], newClip: list[bool], init: MarkerInitialization, dt: float) -> float:
        ...
    def measureGyroRMS(self, gyroObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], newClip: list[bool], init: MarkerInitialization, dt: float) -> float:
        ...
    def optimizeBilevel(self, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], initialization: MarkerInitialization, numSamples: int, applyInnerProblemGradientConstraints: bool = ...) -> BilevelFitResult:
        ...
    def removeZeroConstraint(self, name: str) -> None:
        ...
    def rotateIMUs(self, accObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], gyroObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], newClip: list[bool], init: MarkerInitialization, dt: float) -> None:
        ...
    def runKinematicsPipeline(self, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], newClip: list[bool], params: InitialMarkerFitParams, numSamples: int = ..., skipFinalIK: bool = ...) -> MarkerInitialization:
        ...
    def runMultiTrialKinematicsPipeline(self, markerTrials: list[list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]], params: InitialMarkerFitParams, numSamples: int = ...) -> list[MarkerInitialization]:
        ...
    def runPrescaledPipeline(self, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], params: InitialMarkerFitParams) -> MarkerInitialization:
        ...
    def saveTrajectoryAndMarkersToGUI(self, path: str, init: MarkerInitialization, markerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], accObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], gyroObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], frameRate: int, forcePlates: list[ForcePlate] = ..., goldOsim: OpenSimFile = ..., goldPoses: numpy.ndarray[numpy.float64[m, n]] = ...) -> None:
        ...
    def setAnatomicalMarkerDefaultWeight(self, weight: float) -> None:
        ...
    def setAnthropometricPrior(self, prior: Anthropometrics, weight: float = ...) -> None:
        ...
    def setCustomLossAndGrad(self, loss: typing.Callable[[_nimblephysics.biomechanics.MarkerFitterState], float]) -> None:
        ...
    def setDebugJointVariability(self, debug: bool) -> None:
        ...
    def setDebugLoss(self, debug: bool) -> None:
        ...
    def setExplicitHeightPrior(self, prior: float, weight: float = ...) -> None:
        ...
    def setIgnoreJointLimits(self, ignore: bool) -> None:
        ...
    def setImuMap(self, imuMap: dict[str, tuple[_nimblephysics.dynamics.BodyNode, _nimblephysics.math.Isometry3]]) -> None:
        ...
    def setInitialIKMaxRestarts(self, starts: int) -> None:
        ...
    def setInitialIKSatisfactoryLoss(self, loss: float) -> None:
        ...
    def setIterationLimit(self, iters: int) -> None:
        ...
    def setJointAxisFitSGDIterations(self, iters: int) -> None:
        ...
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
    def setJointSphereFitSGDIterations(self, iters: int) -> None:
        ...
    def setMarkerIsTracking(self, marker: str, isTracking: bool = ...) -> None:
        ...
    def setMaxAxisWeight(self, weight: float) -> None:
        ...
    def setMaxJointWeight(self, weight: float) -> None:
        ...
    def setMaxMarkerOffset(self, offset: float) -> None:
        ...
    def setMinAxisFitScore(self, score: float) -> None:
        ...
    def setMinJointVarianceCutoff(self, cutoff: float) -> None:
        ...
    def setMinSphereFitScore(self, score: float) -> None:
        ...
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
    def setRegularizeAllBodyScales(self, weight: float) -> None:
        ...
    def setRegularizeAnatomicalMarkerOffsets(self, weight: float) -> None:
        ...
    def setRegularizeIndividualBodyScales(self, weight: float) -> None:
        ...
    def setRegularizeJointBounds(self, weight: float) -> None:
        ...
    def setRegularizeJointWithVirtualSpring(self, jointName: str, weight: float) -> None:
        ...
    def setRegularizePelvisJointsWithVirtualSpring(self, weight: float) -> None:
        ...
    def setRegularizeTrackingMarkerOffsets(self, weight: float) -> None:
        ...
    def setStaticTrial(self, markerObservationsMapAtStaticPose: dict[str, numpy.ndarray[numpy.float64[3, 1]]], staticPose: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    def setStaticTrialWeight(self, weight: float) -> None:
        ...
    def setTrackingMarkerDefaultWeight(self, weight: float) -> None:
        ...
    def setTrackingMarkers(self, trackingMarkerNames: list[str]) -> None:
        ...
    def setTriadsToTracking(self) -> None:
        ...
    def writeCSVData(self, path: str, init: MarkerInitialization, rmsMarkerErrors: list[float], maxMarkerErrors: list[float], timestamps: list[float]) -> None:
        ...
class MarkerFitterState:
    bodyNames: list[str]
    bodyScales: numpy.ndarray[numpy.float64[3, n]]
    bodyScalesGrad: numpy.ndarray[numpy.float64[3, n]]
    jointErrorsAtTimesteps: numpy.ndarray[numpy.float64[m, n]]
    jointErrorsAtTimestepsGrad: numpy.ndarray[numpy.float64[m, n]]
    jointOrder: list[str]
    markerErrorsAtTimesteps: numpy.ndarray[numpy.float64[m, n]]
    markerErrorsAtTimestepsGrad: numpy.ndarray[numpy.float64[m, n]]
    markerOffsets: numpy.ndarray[numpy.float64[3, n]]
    markerOffsetsGrad: numpy.ndarray[numpy.float64[3, n]]
    markerOrder: list[str]
    posesAtTimesteps: numpy.ndarray[numpy.float64[m, n]]
    posesAtTimestepsGrad: numpy.ndarray[numpy.float64[m, n]]
class MarkerFixer:
    @staticmethod
    def generateDataErrorsReport(immutableMarkerObservations: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]], dt: float, dropProlongedStillness: bool = ...) -> MarkersErrorReport:
        ...
class MarkerInitialization:
    groupScales: numpy.ndarray[numpy.float64[m, 1]]
    jointCenters: numpy.ndarray[numpy.float64[m, n]]
    joints: list[_nimblephysics.dynamics.Joint]
    markerOffsets: dict[str, numpy.ndarray[numpy.float64[3, 1]]]
    poses: numpy.ndarray[numpy.float64[m, n]]
    updatedMarkerMap: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]]
class MarkerLabeller:
    def evaluate(self, markerOffsets: dict[str, tuple[str, numpy.ndarray[numpy.float64[3, 1]]]], labeledPointClouds: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
    def guessJointLocations(self, pointClouds: list[list[numpy.ndarray[numpy.float64[3, 1]]]]) -> list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]:
        ...
    def labelPointClouds(self, pointClouds: list[list[numpy.ndarray[numpy.float64[3, 1]]]], mergeMarkersThreshold: float = ...) -> LabelledMarkers:
        ...
    def matchUpJointToSkeletonJoint(self, jointName: str, skeletonJointName: str) -> None:
        ...
    def setSkeleton(self, skeleton: _nimblephysics.dynamics.Skeleton) -> None:
        ...
class MarkerLabellerMock(MarkerLabeller):
    def __init__(self) -> None:
        ...
    def setMockJointLocations(self, jointsOverTime: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
class MarkerTrace:
    @property
    def bodyClosestPointDistance(self) -> dict[str, float]:
        ...
    @property
    def bodyMarkerOffsetVariance(self) -> dict[str, float]:
        ...
    @property
    def bodyMarkerOffsets(self) -> dict[str, numpy.ndarray[numpy.float64[3, 1]]]:
        ...
    @property
    def bodyRootJointDistVariance(self) -> dict[str, float]:
        ...
    @property
    def markerLabel(self) -> str:
        ...
    @property
    def maxTime(self) -> int:
        ...
    @property
    def minTime(self) -> int:
        ...
    @property
    def points(self) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        ...
    @property
    def times(self) -> list[int]:
        ...
class MarkersErrorReport:
    droppedMarkerWarnings: list[list[tuple[str, numpy.ndarray[numpy.float64[3, 1]], str]]]
    info: list[str]
    markerObservationsAttemptedFixed: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]
    markersRenamedFromTo: list[list[tuple[str, str]]]
    warnings: list[str]
class MissingGRFReason:
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
    """
    __members__: typing.ClassVar[dict[str, MissingGRFReason]]  # value = {'notMissingGRF': <MissingGRFReason.notMissingGRF: 0>, 'measuredGrfZeroWhenAccelerationNonZero': <MissingGRFReason.measuredGrfZeroWhenAccelerationNonZero: 1>, 'unmeasuredExternalForceDetected': <MissingGRFReason.unmeasuredExternalForceDetected: 2>, 'torqueDiscrepancy': <MissingGRFReason.torqueDiscrepancy: 3>, 'forceDiscrepancy': <MissingGRFReason.forceDiscrepancy: 4>, 'notOverForcePlate': <MissingGRFReason.notOverForcePlate: 5>, 'missingImpact': <MissingGRFReason.missingImpact: 6>, 'missingBlip': <MissingGRFReason.missingBlip: 7>, 'shiftGRF': <MissingGRFReason.shiftGRF: 8>}
    forceDiscrepancy: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.forceDiscrepancy: 4>
    measuredGrfZeroWhenAccelerationNonZero: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.measuredGrfZeroWhenAccelerationNonZero: 1>
    missingBlip: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.missingBlip: 7>
    missingImpact: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.missingImpact: 6>
    notMissingGRF: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.notMissingGRF: 0>
    notOverForcePlate: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.notOverForcePlate: 5>
    shiftGRF: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.shiftGRF: 8>
    torqueDiscrepancy: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.torqueDiscrepancy: 3>
    unmeasuredExternalForceDetected: typing.ClassVar[MissingGRFReason]  # value = <MissingGRFReason.unmeasuredExternalForceDetected: 2>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NeuralMarkerLabeller(MarkerLabeller):
    def __init__(self, jointCenterPredictor: typing.Callable[[List[List[numpy.ndarray[numpy.float64[3, 1]]]]], list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]]) -> None:
        ...
class OpenSimFile:
    anatomicalMarkers: list[str]
    ignoredBodies: list[str]
    jointsDrivenBy: list[tuple[str, str]]
    markersMap: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]]
    skeleton: _nimblephysics.dynamics.Skeleton
    trackingMarkers: list[str]
    warnings: list[str]
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton, markers: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]]) -> None:
        ...
class OpenSimMocoTrajectory:
    activationNames: list[str]
    activations: numpy.ndarray[numpy.float64[m, n]]
    excitationNames: list[str]
    excitations: numpy.ndarray[numpy.float64[m, n]]
    timestamps: list[float]
class OpenSimMot:
    poses: numpy.ndarray[numpy.float64[m, n]]
    timestamps: list[float]
class OpenSimScaleAndMarkerOffsets:
    bodyScales: numpy.ndarray[numpy.float64[m, 1]]
    markerOffsets: dict[str, numpy.ndarray[numpy.float64[3, 1]]]
    markers: dict[str, tuple[_nimblephysics.dynamics.BodyNode, numpy.ndarray[numpy.float64[3, 1]]]]
    success: bool
class OpenSimTRC:
    framesPerSecond: int
    markerLines: dict[str, list[numpy.ndarray[numpy.float64[3, 1]]]]
    markerTimesteps: list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]
    timestamps: list[float]
class ResidualForceHelper:
    def __init__(self, skeleton: _nimblephysics.dynamics.Skeleton, forceBodies: list[int]) -> None:
        ...
    def calculateResidual(self, q: numpy.ndarray[numpy.float64[m, 1]], dq: numpy.ndarray[numpy.float64[m, 1]], ddq: numpy.ndarray[numpy.float64[m, 1]], forcesConcat: numpy.ndarray[numpy.float64[m, 1]]) -> numpy.ndarray[numpy.float64[6, 1]]:
        ...
    def calculateResidualJacobianWrt(self, q: numpy.ndarray[numpy.float64[m, 1]], dq: numpy.ndarray[numpy.float64[m, 1]], ddq: numpy.ndarray[numpy.float64[m, 1]], forcesConcat: numpy.ndarray[numpy.float64[m, 1]], wrt: _nimblephysics.neural.WithRespectTo) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def calculateResidualNorm(self, q: numpy.ndarray[numpy.float64[m, 1]], dq: numpy.ndarray[numpy.float64[m, 1]], ddq: numpy.ndarray[numpy.float64[m, 1]], forcesConcat: numpy.ndarray[numpy.float64[m, 1]], torquesMultiple: float, useL1: bool = ...) -> float:
        ...
    def calculateResidualNormGradientWrt(self, q: numpy.ndarray[numpy.float64[m, 1]], dq: numpy.ndarray[numpy.float64[m, 1]], ddq: numpy.ndarray[numpy.float64[m, 1]], forcesConcat: numpy.ndarray[numpy.float64[m, 1]], wrt: _nimblephysics.neural.WithRespectTo, torquesMultiple: float, useL1: bool = ...) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
class SkeletonConverter:
    def __init__(self, source: _nimblephysics.dynamics.Skeleton, target: _nimblephysics.dynamics.Skeleton) -> None:
        ...
    def convertMotion(self, targetMotion: numpy.ndarray[numpy.float64[m, n]], logProgress: bool = ..., convergenceThreshold: float = ..., maxStepCount: int = ..., leastSquaresDamping: float = ..., lineSearch: bool = ..., logIKOutput: bool = ...) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
    def createVirtualMarkers(self, addFakeMarkers: int = ..., weightFakeMarkers: float = ...) -> None:
        ...
    def debugToGUI(self, gui: _nimblephysics.server.GUIWebsocketServer) -> None:
        ...
    def fitSourceToTarget(self, convergenceThreshold: float = ..., maxStepCount: int = ..., leastSquaresDamping: float = ..., lineSearch: bool = ..., logOutput: bool = ...) -> float:
        ...
    def fitTargetToSource(self, convergenceThreshold: float = ..., maxStepCount: int = ..., leastSquaresDamping: float = ..., lineSearch: bool = ..., logOutput: bool = ...) -> float:
        ...
    def getSourceJointWorldPositions(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def getSourceJoints(self) -> list[_nimblephysics.dynamics.Joint]:
        ...
    def getTargetJointWorldPositions(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def getTargetJoints(self) -> list[_nimblephysics.dynamics.Joint]:
        ...
    def linkJoints(self, sourceJoint: _nimblephysics.dynamics.Joint, targetJoint: _nimblephysics.dynamics.Joint) -> None:
        ...
    def rescaleAndPrepTarget(self, addFakeMarkers: int = ..., weightFakeMarkers: float = ..., convergenceThreshold: float = ..., maxStepCount: int = ..., leastSquaresDamping: float = ..., lineSearch: bool = ..., logOutput: bool = ...) -> None:
        ...
class SubjectOnDisk:
    """
    
            This is for doing ML and large-scale data analysis. The idea here is to
            create a lazy-loadable view of a subject, where everything remains on disk
            until asked for. That way we can instantiate thousands of these in memory,
            and not worry about OOM'ing a machine.
          
    """
    @staticmethod
    def writeSubject(outputPath: str, openSimFilePath: str, trialTimesteps: list[float], trialPoses: list[numpy.ndarray[numpy.float64[m, n]]], trialVels: list[numpy.ndarray[numpy.float64[m, n]]], trialAccs: list[numpy.ndarray[numpy.float64[m, n]]], probablyMissingGRF: list[list[bool]], missingGRFReason: list[list[MissingGRFReason]], dofPositionsObserved: list[list[bool]], dofVelocitiesFiniteDifferenced: list[list[bool]], dofAccelerationsFiniteDifferenced: list[list[bool]], trialTaus: list[numpy.ndarray[numpy.float64[m, n]]], trialComPoses: list[numpy.ndarray[numpy.float64[m, n]]], trialComVels: list[numpy.ndarray[numpy.float64[m, n]]], trialComAccs: list[numpy.ndarray[numpy.float64[m, n]]], trialResidualNorms: list[list[float]], groundForceBodies: list[str], trialGroundBodyWrenches: list[numpy.ndarray[numpy.float64[m, n]]], trialGroundBodyCopTorqueForce: list[numpy.ndarray[numpy.float64[m, n]]], customValueNames: list[str], customValues: list[list[numpy.ndarray[numpy.float64[m, n]]]], markerObservations: list[list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]], accObservations: list[list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]], gyroObservations: list[list[dict[str, numpy.ndarray[numpy.float64[3, 1]]]]], emgObservations: list[list[dict[str, numpy.ndarray[numpy.float64[m, 1]]]]], forcePlates: list[list[ForcePlate]], biologicalSex: str, heightM: float, massKg: float, ageYears: int, trialNames: list[str] = ..., subjectTags: list[str] = ..., trialTags: list[list[str]] = ..., sourceHref: str = ..., notes: str = ...) -> None:
        """
        This writes a subject out to disk in a compressed and random-seekable binary format.
        """
    def __init__(self, path: str) -> None:
        ...
    def getAgeYears(self) -> int:
        """
        This returns the age of the subject, or 0 if unknown.
        """
    def getBiologicalSex(self) -> str:
        """
        This returns a string, one of "male", "female", or "unknown".
        """
    def getContactBodies(self) -> list[str]:
        """
        A list of the :code:`body_name`'s for each body that was assumed to be able to take ground-reaction-force from force plates.
        """
    def getCustomValueDim(self, valueName: str) -> int:
        """
        This returns the dimension of the custom value specified by :code:`valueName`
        """
    def getCustomValues(self) -> list[str]:
        """
        A list of all the different types of custom values that this SubjectOnDisk contains. These are unspecified, and are intended to allow an easy extension of the format to unusual types of data (like exoskeleton torques or unusual physical sensors) that may be present on some subjects but not others.
        """
    def getDofAccelerationsFiniteDifferenced(self, trial: int) -> list[bool]:
        """
        This returns the vector of booleans indicating which DOFs have their accelerations from finite-differencing during this trial (as opposed to observed directly through a accelerometer or IMU)
        """
    def getDofPositionsObserved(self, trial: int) -> list[bool]:
        """
        This returns the vector of booleans indicating which DOFs have their positions observed during this trial
        """
    def getDofVelocitiesFiniteDifferenced(self, trial: int) -> list[bool]:
        """
        This returns the vector of booleans indicating which DOFs have their velocities from finite-differencing during this trial (as opposed to observed directly through a gyroscope or IMU)
        """
    def getForcePlateCorners(self, trial: int, forcePlate: int) -> list[numpy.ndarray[numpy.float64[3, 1]]]:
        """
        Get an array of force plate corners (as 3D vectors) for the given force plate in the given trial. Empty array on out-of-bounds access.
        """
    def getHeightM(self) -> float:
        """
        This returns the height in meters, or 0.0 if unknown.
        """
    def getHref(self) -> str:
        """
        The AddBiomechanics link for this subject's data.
        """
    def getMassKg(self) -> float:
        """
        This returns the mass in kilograms, or 0.0 if unknown.
        """
    def getMissingGRFReason(self, trial: int) -> list[MissingGRFReason]:
        """
                    This returns an array of enum values, one per frame in the specified trial,
                    each corresponding to the reason why a frame was marked as `probablyMissingGRF`.
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
    def getNumTrials(self) -> int:
        """
        This returns the number of trials that are in this file.
        """
    def getProbablyMissingGRF(self, trial: int) -> list[bool]:
        """
                    This returns an array of boolean values, one per frame in the specified trial.
                    Each frame is :code:`True` if this frame probably has unmeasured forces acting on the body. For example, if a subject
                    steps off of the available force plates during this frame, this will probably be true.
        
                    WARNING: If this is true, you can't trust the :code:`tau` or :code:`acc` values on the corresponding frame!!
        
                    This method is provided to give a cheaper way to filter out frames we want to ignore for training, without having to call
                    the more expensive :code:`loadFrames()`
        """
    def getSubjectTags(self) -> list[str]:
        """
        This returns the list of tags attached to this subject, which are arbitrary strings from the AddBiomechanics platform.
        """
    def getTrialLength(self, trial: int) -> int:
        """
        This returns the length of the trial requested
        """
    def getTrialMaxJointVelocity(self, trial: int) -> list[float]:
        """
        This returns the vector of scalars indicating the maximum absolute velocity of all DOFs on each timestep of a given trial
        """
    def getTrialName(self, trial: int) -> str:
        """
        This returns the human readable name of the specified trial, given by the person who uploaded the data to AddBiomechanics. This isn't necessary for training, but may be useful for analyzing the data.
        """
    def getTrialResidualNorms(self, trial: int) -> list[float]:
        """
        This returns the vector of scalars indicating the norm of the root residual forces + torques on each timestep of a given trial
        """
    def getTrialTags(self, trial: int) -> list[str]:
        """
        This returns the list of tags attached to a given trial index, which are arbitrary strings from the AddBiomechanics platform.
        """
    def getTrialTimestep(self, trial: int) -> float:
        """
        This returns the timestep size for the trial requested, in seconds per frame
        """
    def readFrames(self, trial: int, startFrame: int, numFramesToRead: int = ..., stride: int = ..., contactThreshold: float = ...) -> list[Frame]:
        """
        This will read from disk and allocate a number of :code:`Frame` objects. These Frame objects are assumed to be short-lived, to save working memory. For example, you might :code:`readFrames()` to construct a training batch, then immediately allow the frames to go out of scope and be released after the batch backpropagates gradient and loss. On OOB access, prints an error and returns an empty vector.
        """
    def readRawOsimFileText(self) -> str:
        """
        This will read the raw OpenSim file XML out of the SubjectOnDisk, and return it as a string.
        """
    def readSkel(self, geometryFolder: str = ...) -> _nimblephysics.dynamics.Skeleton:
        """
        This will read the skeleton from the binary, and optionally use the passed in :code:`geometryFolder` to load meshes. We do not bundle meshes with :code:`SubjectOnDisk` files, to save space. If you do not pass in :code:`geometryFolder`, expect to get warnings about being unable to load meshes, and expect that your skeleton will not display if you attempt to visualize it.
        """
forceDiscrepancy: MissingGRFReason  # value = <MissingGRFReason.forceDiscrepancy: 4>
measuredGrfZeroWhenAccelerationNonZero: MissingGRFReason  # value = <MissingGRFReason.measuredGrfZeroWhenAccelerationNonZero: 1>
missingBlip: MissingGRFReason  # value = <MissingGRFReason.missingBlip: 7>
missingImpact: MissingGRFReason  # value = <MissingGRFReason.missingImpact: 6>
notMissingGRF: MissingGRFReason  # value = <MissingGRFReason.notMissingGRF: 0>
notOverForcePlate: MissingGRFReason  # value = <MissingGRFReason.notOverForcePlate: 5>
shiftGRF: MissingGRFReason  # value = <MissingGRFReason.shiftGRF: 8>
torqueDiscrepancy: MissingGRFReason  # value = <MissingGRFReason.torqueDiscrepancy: 3>
unmeasuredExternalForceDetected: MissingGRFReason  # value = <MissingGRFReason.unmeasuredExternalForceDetected: 2>
