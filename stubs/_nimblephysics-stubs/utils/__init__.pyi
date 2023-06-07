from __future__ import annotations
import nimblephysics_libs._nimblephysics.utils
import typing
import nimblephysics_libs._nimblephysics.common
import nimblephysics_libs._nimblephysics.dynamics
import nimblephysics_libs._nimblephysics.simulation
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "AccelerationSmoother",
    "DartLoader",
    "MJCFExporter",
    "SdfParser",
    "SkelParser",
    "StringUtils",
    "UniversalLoader"
]


class AccelerationSmoother():
    def __init__(self, timesteps: int, smoothingWeight: float, regularizationWeight: float, useSparse: bool = True, useIterative: bool = True) -> None: ...
    def debugTimeSeries(self, series: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setIterations(self, iterations: int) -> None: ...
    def smooth(self, series: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
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
