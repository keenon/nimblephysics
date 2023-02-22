from __future__ import annotations
import nimblephysics_libs._nimblephysics.simulation
import typing
import nimblephysics_libs._nimblephysics.collision
import nimblephysics_libs._nimblephysics.constraint
import nimblephysics_libs._nimblephysics.dynamics
import nimblephysics_libs._nimblephysics.neural
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "World"
]


class World():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, arg0: str) -> None: ...
    @typing.overload
    def __init__(self, name: str) -> None: ...
    def addDofToActionSpace(self, dofIndex: int) -> None: ...
    def addSimpleFrame(self, frame: nimblephysics_libs._nimblephysics.dynamics.SimpleFrame) -> str: ...
    def addSkeleton(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> str: ...
    def bake(self) -> None: ...
    @typing.overload
    def checkCollision(self) -> bool: ...
    @typing.overload
    def checkCollision(self, option: nimblephysics_libs._nimblephysics.collision.CollisionOption) -> bool: ...
    @typing.overload
    def checkCollision(self, option: nimblephysics_libs._nimblephysics.collision.CollisionOption, result: nimblephysics_libs._nimblephysics.collision.CollisionResult) -> bool: ...
    def clone(self) -> World: ...
    def colorsToJson(self) -> str: ...
    def getAction(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getActionJacobian(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getActionSize(self) -> int: ...
    def getActionSpace(self) -> typing.List[int]: ...
    def getBodyNodeByIndex(self, arg0: int) -> nimblephysics_libs._nimblephysics.dynamics.BodyNode: ...
    def getCachedLCPSolution(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getConstraintSolver(self) -> nimblephysics_libs._nimblephysics.constraint.ConstraintSolver: ...
    def getContactClippingDepth(self) -> float: ...
    def getControlForceLowerLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getControlForceUpperLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getControlForces(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getCoriolisAndGravityAndExternalForces(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getCoriolisAndGravityForces(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getFallbackConstraintForceMixingConstant(self) -> float: ...
    def getGravity(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def getIndex(self, index: int) -> int: ...
    def getInvMassMatrix(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getLastCollisionResult(self) -> nimblephysics_libs._nimblephysics.collision.CollisionResult: ...
    def getMassLowerLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getMassMatrix(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getMassUpperLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getMasses(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getName(self) -> str: ...
    def getNumBodyNodes(self) -> int: ...
    def getNumDofs(self) -> int: ...
    def getNumSimpleFrames(self) -> int: ...
    def getNumSkeletons(self) -> int: ...
    def getParallelVelocityAndPositionUpdates(self) -> bool: ...
    def getPenetrationCorrectionEnabled(self) -> bool: ...
    def getPositionLowerLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPositionUpperLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPositions(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getSimFrames(self) -> int: ...
    @typing.overload
    def getSimpleFrame(self, index: int) -> nimblephysics_libs._nimblephysics.dynamics.SimpleFrame: ...
    @typing.overload
    def getSimpleFrame(self, name: str) -> nimblephysics_libs._nimblephysics.dynamics.SimpleFrame: ...
    @typing.overload
    def getSkeleton(self, index: int) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def getSkeleton(self, name: str) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: ...
    def getState(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getStateJacobian(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getStateSize(self) -> int: ...
    def getTime(self) -> float: ...
    def getTimeStep(self) -> float: ...
    def getUseFDOverride(self) -> bool: ...
    def getVelocities(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getVelocityLowerLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getVelocityUpperLimits(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getWrtMass(self) -> nimblephysics_libs._nimblephysics.neural.WithRespectToMass: ...
    def hasSkeleton(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> bool: ...
    def integratePositions(self, initialVelocity: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def integrateVelocitiesFromImpulses(self, resetCommand: bool = True) -> None: ...
    @staticmethod
    def loadFrom(arg0: str) -> World: ...
    def loadSkeleton(self, path: str, basePosition: numpy.ndarray[numpy.float64, _Shape[3, 1]] = array([0., 0., 0.]), baseEulerAnglesXYZ: numpy.ndarray[numpy.float64, _Shape[3, 1]] = array([0., 0., 0.])) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: ...
    def positionsToJson(self) -> str: ...
    def removeAllSimpleFrames(self) -> typing.Set[nimblephysics_libs._nimblephysics.dynamics.SimpleFrame]: ...
    def removeAllSkeletons(self) -> typing.Set[nimblephysics_libs._nimblephysics.dynamics.Skeleton]: ...
    def removeDofFromActionSpace(self, dofIndex: int) -> None: ...
    def removeSimpleFrame(self, frame: nimblephysics_libs._nimblephysics.dynamics.SimpleFrame) -> None: ...
    def removeSkeleton(self, skeleton: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None: ...
    def replaceConstraintEngineFn(self, arg0: typing.Callable[[bool], None]) -> None: ...
    def reset(self) -> None: ...
    def runConstraintEngine(self, arg0: bool) -> None: ...
    def runLcpConstraintEngine(self, arg0: bool) -> None: ...
    def setAction(self, action: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setActionSpace(self, actionSpaceMapping: typing.List[int]) -> None: ...
    def setCachedLCPSolution(self, cachedLCPSolution: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setControlForces(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setControlForcesLowerLimits(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setControlForcesUpperLimits(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setFallbackConstraintForceMixingConstant(self, constant: float = 0.001) -> None: ...
    def setGravity(self, gravity: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    def setMasses(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setName(self, newName: str) -> str: ...
    def setParallelVelocityAndPositionUpdates(self, enabled: bool) -> None: ...
    def setPenetrationCorrectionEnabled(self, enabled: bool) -> None: ...
    def setPositionLowerLimits(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setPositionUpperLimits(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setPositions(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    @typing.overload
    def setSlowDebugResultsAgainstFD(self, arg0: bool) -> None: ...
    @typing.overload
    def setSlowDebugResultsAgainstFD(self, setSlowDebugResultsAgainstFD: bool) -> None: ...
    def setState(self, state: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setTime(self, time: float) -> None: ...
    def setTimeStep(self, timeStep: float) -> None: ...
    def setUseFDOverride(self, useFDOverride: bool) -> None: ...
    def setVelocities(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setVelocityLowerLimits(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setVelocityUpperLimits(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    @typing.overload
    def step(self) -> None: ...
    @typing.overload
    def step(self, resetCommand: bool) -> None: ...
    def toJson(self) -> str: ...
    def tuneMass(self, arg0: nimblephysics_libs._nimblephysics.dynamics.BodyNode, arg1: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType, arg2: numpy.ndarray[numpy.float64, _Shape[m, 1]], arg3: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    pass