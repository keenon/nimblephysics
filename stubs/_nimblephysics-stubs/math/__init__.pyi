"""Bindings for Eigen geometric types."""
from __future__ import annotations
import nimblephysics_libs._nimblephysics.math
import typing
import nimblephysics_libs._nimblephysics.dynamics
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "AdR",
    "AdT",
    "AngleAxis",
    "BoundingBox",
    "GraphFlowDiscretizer",
    "Isometry3",
    "MultivariateGaussian",
    "ParticlePath",
    "PolynomialFitter",
    "Quaternion",
    "Random",
    "dAdInvT",
    "dAdT",
    "distancePointToConvexHull2D",
    "distancePointToConvexHullProjectedTo2D",
    "eulerXYXToMatrix",
    "eulerXYZToMatrix",
    "eulerXZXToMatrix",
    "eulerXZYToMatrix",
    "eulerYXYToMatrix",
    "eulerYXZToMatrix",
    "eulerYZXToMatrix",
    "eulerYZYToMatrix",
    "eulerZXYToMatrix",
    "eulerZXZToMatrix",
    "eulerZYXToMatrix",
    "eulerZYZToMatrix",
    "expAngular",
    "expMap",
    "expMapJac",
    "expMapRot",
    "expToQuat",
    "leftMultiplyInFreeJointSpace",
    "logMap",
    "matrixToEulerXYX",
    "matrixToEulerXYZ",
    "matrixToEulerXZY",
    "matrixToEulerYXZ",
    "matrixToEulerYZX",
    "matrixToEulerZXY",
    "matrixToEulerZYX",
    "quatToExp",
    "rightMultiplyInFreeJointSpace",
    "roundEulerAnglesToNearest",
    "transformBy",
    "verifyRotation",
    "verifyTransform"
]


class AngleAxis():
    """
    Bindings for Eigen::AngleAxis<>.
    """
    @staticmethod
    def Identity() -> AngleAxis: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, angle: float, axis: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    @typing.overload
    def __init__(self, quaternion: Quaternion) -> None: ...
    @typing.overload
    def __init__(self, rotation: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> None: ...
    @typing.overload
    def __init__(self, other: AngleAxis) -> None: ...
    def __str__(self) -> str: ...
    def angle(self) -> float: ...
    def axis(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def inverse(self) -> AngleAxis: ...
    def multiply(self, arg0: AngleAxis) -> Quaternion: ...
    def quaternion(self) -> Quaternion: ...
    def rotation(self) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]: ...
    def set_angle(self, angle: float) -> None: ...
    def set_axis(self, axis: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    def set_quaternion(self, arg0: Quaternion) -> None: ...
    def set_rotation(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> None: ...
    def to_rotation_matrix(self) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]: ...
    pass
class BoundingBox():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, min: numpy.ndarray[numpy.float64, _Shape[3, 1]], max: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    def computeCenter(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def computeFullExtents(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def computeHalfExtents(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def getMax(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def getMin(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    pass
class GraphFlowDiscretizer():
    def __init__(self, numNodes: int, arcs: typing.List[typing.Tuple[int, int]], nodeAttachedToSink: typing.List[bool]) -> None: ...
    def cleanUpArcRates(self, energyLevels: numpy.ndarray[numpy.float64, _Shape[m, n]], arcRates: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This will find the least-squares closest rates of transfer across the arcs to end up with the energy levels at each node we got over time. The idea here is that arc rates may not perfectly reflect the observed changes in energy levels.
        """
    def discretize(self, maxSimultaneousParticles: int, energyLevels: numpy.ndarray[numpy.float64, _Shape[m, n]], arcRates: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> typing.List[ParticlePath]: 
        """
        This will attempt to create a set of ParticlePath objects that map the recorded graph node levels and flows as closely as possible. The particles can be created and destroyed within the arcs.
        """
    pass
class Isometry3():
    @staticmethod
    def Identity() -> Isometry3: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, matrix: numpy.ndarray[numpy.float64, _Shape[4, 4]]) -> None: ...
    @typing.overload
    def __init__(self, rotation: numpy.ndarray[numpy.float64, _Shape[3, 3]], translation: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    @typing.overload
    def __init__(self, quaternion: Quaternion, translation: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    @typing.overload
    def __init__(self, other: Isometry3) -> None: ...
    def __str__(self) -> str: ...
    def inverse(self) -> Isometry3: ...
    def matrix(self) -> numpy.ndarray[numpy.float64, _Shape[4, 4]]: ...
    @typing.overload
    def multiply(self, other: Isometry3) -> Isometry3: ...
    @typing.overload
    def multiply(self, position: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def pretranslate(self, other: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    def quaternion(self) -> Quaternion: ...
    def rotation(self) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]: ...
    def set_matrix(self, arg0: numpy.ndarray[numpy.float64, _Shape[4, 4]]) -> None: ...
    def set_quaternion(self, arg0: Quaternion) -> None: ...
    def set_rotation(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> None: ...
    def set_translation(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    def translate(self, other: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None: ...
    def translation(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    pass
class MultivariateGaussian():
    def __init__(self, variables: typing.List[str], mu: numpy.ndarray[numpy.float64, _Shape[m, 1]], cov: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None: ...
    def computeLogPDF(self, values: numpy.ndarray[numpy.float64, _Shape[m, 1]], normalized: bool = True) -> float: ...
    def computeLogPDFGrad(self, x: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def computePDF(self, values: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> float: ...
    def condition(self, observedValues: typing.Dict[str, float]) -> MultivariateGaussian: ...
    def convertFromMap(self, values: typing.Dict[str, float]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def convertToMap(self, values: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Dict[str, float]: ...
    def debugToStdout(self) -> None: ...
    def getCov(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getCovSubset(self, rowIndices: typing.List[int], colIndices: typing.List[int]) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getLogNormalizationConstant(self) -> float: ...
    def getMean(self, variable: str) -> float: ...
    def getMu(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getMuSubset(self, indices: typing.List[int]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getObservedIndices(self, observedValues: typing.Dict[str, float]) -> typing.List[int]: ...
    def getUnobservedIndices(self, observedValues: typing.Dict[str, float]) -> typing.List[int]: ...
    def getVariableNameAtIndex(self, i: int) -> str: ...
    def getVariableNames(self) -> typing.List[str]: ...
    @staticmethod
    def loadFromCSV(file: str, columns: typing.List[str], units: float = 1.0) -> MultivariateGaussian: ...
    pass
class ParticlePath():
    @property
    def energyValue(self) -> float:
        """
        :type: float
        """
    @energyValue.setter
    def energyValue(self, arg0: float) -> None:
        pass
    @property
    def nodeHistory(self) -> typing.List[int]:
        """
        :type: typing.List[int]
        """
    @nodeHistory.setter
    def nodeHistory(self, arg0: typing.List[int]) -> None:
        pass
    @property
    def startTime(self) -> int:
        """
        :type: int
        """
    @startTime.setter
    def startTime(self, arg0: int) -> None:
        pass
    pass
class PolynomialFitter():
    def __init__(self, timesteps: numpy.ndarray[numpy.float64, _Shape[m, 1]], order: int) -> None: ...
    def calcCoeffs(self, values: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def projectPosVelAccAtTime(self, timestep: float, pastValues: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    pass
class Quaternion():
    """
    Provides a unit quaternion binding of Eigen::Quaternion<>.
    """
    @staticmethod
    def Identity() -> Quaternion: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, wxyz: numpy.ndarray[numpy.float64, _Shape[4, 1]]) -> None: ...
    @typing.overload
    def __init__(self, w: float, x: float, y: float, z: float) -> None: ...
    @typing.overload
    def __init__(self, rotation: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> None: ...
    @typing.overload
    def __init__(self, other: Quaternion) -> None: ...
    def __str__(self) -> str: ...
    def conjugate(self) -> Quaternion: ...
    def inverse(self) -> Quaternion: ...
    @typing.overload
    def multiply(self, arg0: Quaternion) -> Quaternion: ...
    @typing.overload
    def multiply(self, position: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def rotation(self) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]: ...
    def set_rotation(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> None: ...
    @typing.overload
    def set_wxyz(self, wxyz: numpy.ndarray[numpy.float64, _Shape[4, 1]]) -> None: ...
    @typing.overload
    def set_wxyz(self, w: float, x: float, y: float, z: float) -> None: ...
    def to_rotation_matrix(self) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]: ...
    def w(self) -> float: ...
    def wxyz(self) -> numpy.ndarray[numpy.float64, _Shape[4, 1]]: ...
    def x(self) -> float: ...
    def xyz(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]: ...
    def y(self) -> float: ...
    def z(self) -> float: ...
    pass
class Random():
    def __init__(self) -> None: ...
    @staticmethod
    def getSeed() -> int: ...
    @staticmethod
    def setSeed(seed: int) -> None: ...
    @staticmethod
    def uniform(min: float, max: float) -> float: ...
    pass
def AdR(R: numpy.ndarray[numpy.float64, _Shape[3, 3]], S: numpy.ndarray[numpy.float64, _Shape[6, 1]]) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]:
    pass
def AdT(R: numpy.ndarray[numpy.float64, _Shape[3, 3]], p: numpy.ndarray[numpy.float64, _Shape[3, 1]], S: numpy.ndarray[numpy.float64, _Shape[6, 1]]) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]:
    pass
def dAdInvT(R: numpy.ndarray[numpy.float64, _Shape[3, 3]], p: numpy.ndarray[numpy.float64, _Shape[3, 1]], S: numpy.ndarray[numpy.float64, _Shape[6, 1]]) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]:
    pass
def dAdT(R: numpy.ndarray[numpy.float64, _Shape[3, 3]], p: numpy.ndarray[numpy.float64, _Shape[3, 1]], S: numpy.ndarray[numpy.float64, _Shape[6, 1]]) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]:
    pass
def distancePointToConvexHull2D(P: numpy.ndarray[numpy.float64, _Shape[2, 1]], points: typing.List[numpy.ndarray[numpy.float64, _Shape[2, 1]]]) -> float:
    pass
def distancePointToConvexHullProjectedTo2D(P: numpy.ndarray[numpy.float64, _Shape[3, 1]], points: typing.List[numpy.ndarray[numpy.float64, _Shape[3, 1]]], normal: numpy.ndarray[numpy.float64, _Shape[3, 1]] = array([0., 1., 0.])) -> float:
    pass
def eulerXYXToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerXYZToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerXZXToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerXZYToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerYXYToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerYXZToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerYZXToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerYZYToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerZXYToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerZXZToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerZYXToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def eulerZYZToMatrix(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def expAngular(s: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> Isometry3:
    pass
def expMap(S: numpy.ndarray[numpy.float64, _Shape[6, 1]]) -> Isometry3:
    pass
def expMapJac(expmap: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def expMapRot(expmap: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 3]]:
    pass
def expToQuat(v: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> Quaternion:
    pass
def leftMultiplyInFreeJointSpace(R: numpy.ndarray[numpy.float64, _Shape[3, 3]], p: numpy.ndarray[numpy.float64, _Shape[3, 1]], S: numpy.ndarray[numpy.float64, _Shape[6, 1]]) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]:
    pass
def logMap(S: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def matrixToEulerXYX(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def matrixToEulerXYZ(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def matrixToEulerXZY(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def matrixToEulerYXZ(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def matrixToEulerYZX(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def matrixToEulerZXY(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def matrixToEulerZYX(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def quatToExp(q: Quaternion) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def rightMultiplyInFreeJointSpace(R: numpy.ndarray[numpy.float64, _Shape[3, 3]], p: numpy.ndarray[numpy.float64, _Shape[3, 1]], S: numpy.ndarray[numpy.float64, _Shape[6, 1]]) -> numpy.ndarray[numpy.float64, _Shape[6, 1]]:
    pass
def roundEulerAnglesToNearest(angle: numpy.ndarray[numpy.float64, _Shape[3, 1]], previousAngle: numpy.ndarray[numpy.float64, _Shape[3, 1]], axisOrder: nimblephysics_libs._nimblephysics.dynamics.AxisOrder = AxisOrder.XYZ) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def transformBy(T: Isometry3, p: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
    pass
def verifyRotation(R: numpy.ndarray[numpy.float64, _Shape[3, 3]]) -> bool:
    pass
def verifyTransform(T: Isometry3) -> bool:
    pass
