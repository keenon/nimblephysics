from __future__ import annotations
import _nimblephysics.utils
import typing
import _nimblephysics.common
import _nimblephysics.dynamics
import _nimblephysics.simulation
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "AccelerationSmoother",
    "DartLoader",
    "MJCFExporter",
    "SdfParser",
    "SkelParser",
    "UniversalLoader"
]


class AccelerationSmoother():
    def __init__(self, timesteps: int, smoothingWeight: float, regularizationWeight: float, useSparse: bool = True, useIterative: bool = True) -> None: ...
    def debugTimeSeries(self, series: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def smooth(self, series: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    pass
class DartLoader():
    def __init__(self) -> None: ...
    def addPackageDirectory(self, packageName: str, packageDirectory: str) -> None: ...
    @typing.overload
    def parseSkeleton(self, uri: _nimblephysics.common.Uri) -> _nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseSkeleton(self, uri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseSkeletonString(self, urdfString: str, baseUri: _nimblephysics.common.Uri) -> _nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseSkeletonString(self, urdfString: str, baseUri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.dynamics.Skeleton: ...
    @typing.overload
    def parseWorld(self, uri: _nimblephysics.common.Uri) -> _nimblephysics.simulation.World: ...
    @typing.overload
    def parseWorld(self, uri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.simulation.World: ...
    @typing.overload
    def parseWorldString(self, urdfString: str, baseUri: _nimblephysics.common.Uri) -> _nimblephysics.simulation.World: ...
    @typing.overload
    def parseWorldString(self, urdfString: str, baseUri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.simulation.World: ...
    pass
class MJCFExporter():
    @staticmethod
    def writeSkeleton(path: str, skel: _nimblephysics.dynamics.Skeleton) -> None: ...
    pass
