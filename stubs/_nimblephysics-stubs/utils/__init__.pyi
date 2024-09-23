from __future__ import annotations
import nimblephysics_libs._nimblephysics.utils
import typing
import nimblephysics_libs._nimblephysics.common
import nimblephysics_libs._nimblephysics.dynamics
import nimblephysics_libs._nimblephysics.simulation
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "AccelerationMinimizer",
    "AccelerationSmoother",
    "AccelerationTrackAndMinimize",
    "AccelerationTrackingResult",
    "DartLoader",
    "MJCFExporter",
    "SdfParser",
    "SkelParser",
    "StringUtils",
    "UniversalLoader"
]


class AccelerationMinimizer():
    def __init__(self, numTimesteps: int, smoothingWeight: float = 1.0, regularizationWeight: float = 0.01, startPositionZeroWeight: float = 0.0, endPositionZeroWeight: float = 0.0, startVelocityZeroWeight: float = 0.0, endVelocityZeroWeight: float = 0.0, numIterations: int = 10000) -> None: ...
    def minimize(self, series: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def setConvergenceTolerance(self, tolerance: float) -> None: ...
    def setDebugIterationBackoff(self, iterations: bool) -> None: ...
    def setNumIterationsBackoff(self, series: int) -> None: ...
    pass
class AccelerationSmoother():
    def __init__(self, timesteps: int, smoothingWeight: float, regularizationWeight: float, useSparse: bool = True, useIterative: bool = True) -> None: ...
    def debugTimeSeries(self, series: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setIterations(self, iterations: int) -> None: ...
    def smooth(self, series: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    pass
class AccelerationTrackAndMinimize():
    def __init__(self, numTimesteps: int, trackAccelerationAtTimesteps: typing.List[bool], zeroUnobservedAccWeight: float = 1.0, trackObservedAccWeight: float = 1.0, regularizationWeight: float = 0.01, dt: float = 1.0, numIterations: int = 10000) -> None: ...
    def minimize(self, series: numpy.ndarray[numpy.float64, _Shape[m, 1]], trackAcc: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> AccelerationTrackingResult: ...
    def setConvergenceTolerance(self, tolerance: float) -> None: ...
    def setDebugIterationBackoff(self, iterations: bool) -> None: ...
    def setNumIterationsBackoff(self, series: int) -> None: ...
    pass
class AccelerationTrackingResult():
    @property
    def accelerationOffset(self) -> float:
        """
        :type: float
        """
    @accelerationOffset.setter
    def accelerationOffset(self, arg0: float) -> None:
        pass
    @property
    def series(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @series.setter
    def series(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    pass
class DartLoader():
    def __init__(self) -> None: ...
    def addPackageDirectory(self, packageName: str, packageDirectory: str) -> None: ...
    @typing.overload
    def parseSkeleton(self, uri: nimblephysics_libs._nimblephysics.common.Uri) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseSkeleton(self, uri: nimblephysics_libs._nimblephysics.common.Uri, resourceRetriever: nimblephysics_libs._nimblephysics.common.ResourceRetriever) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseSkeletonString(self, urdfString: str, baseUri: nimblephysics_libs._nimblephysics.common.Uri) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseSkeletonString(self, urdfString: str, baseUri: nimblephysics_libs._nimblephysics.common.Uri, resourceRetriever: nimblephysics_libs._nimblephysics.common.ResourceRetriever) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseWorld(self, uri: nimblephysics_libs._nimblephysics.common.Uri) -> nimblephysics_libs._nimblephysics.simulation.World: ...
    @typing.overload
    def parseWorld(self, uri: nimblephysics_libs._nimblephysics.common.Uri, resourceRetriever: nimblephysics_libs._nimblephysics.common.ResourceRetriever) -> nimblephysics_libs._nimblephysics.simulation.World: ...
    @typing.overload
    def parseWorldString(self, urdfString: str, baseUri: nimblephysics_libs._nimblephysics.common.Uri) -> nimblephysics_libs._nimblephysics.simulation.World: ...
    @typing.overload
    def parseWorldString(self, urdfString: str, baseUri: nimblephysics_libs._nimblephysics.common.Uri, resourceRetriever: nimblephysics_libs._nimblephysics.common.ResourceRetriever) -> nimblephysics_libs._nimblephysics.simulation.World: ...
    pass
class MJCFExporter():
    @staticmethod
    def writeSkeleton(path: str, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None: ...
    pass
