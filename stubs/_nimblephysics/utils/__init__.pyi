from __future__ import annotations
import _nimblephysics.common
import _nimblephysics.dynamics
import _nimblephysics.simulation
import numpy
import typing
from . import SdfParser
from . import SkelParser
from . import StringUtils
from . import UniversalLoader
__all__ = ['AccelerationSmoother', 'DartLoader', 'MJCFExporter', 'SdfParser', 'SkelParser', 'StringUtils', 'UniversalLoader']
class AccelerationSmoother:
    def __init__(self, timesteps: int, smoothingWeight: float, regularizationWeight: float, useSparse: bool = ..., useIterative: bool = ...) -> None:
        ...
    def debugTimeSeries(self, series: numpy.ndarray[numpy.float64[m, 1]]) -> None:
        ...
    def setIterations(self, iterations: int) -> None:
        ...
    def smooth(self, series: numpy.ndarray[numpy.float64[m, n]]) -> numpy.ndarray[numpy.float64[m, n]]:
        ...
class DartLoader:
    def __init__(self) -> None:
        ...
    def addPackageDirectory(self, packageName: str, packageDirectory: str) -> None:
        ...
    @typing.overload
    def parseSkeleton(self, uri: _nimblephysics.common.Uri) -> _nimblephysics.dynamics.Skeleton:
        ...
    @typing.overload
    def parseSkeleton(self, uri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.dynamics.Skeleton:
        ...
    @typing.overload
    def parseSkeletonString(self, urdfString: str, baseUri: _nimblephysics.common.Uri) -> _nimblephysics.dynamics.Skeleton:
        ...
    @typing.overload
    def parseSkeletonString(self, urdfString: str, baseUri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.dynamics.Skeleton:
        ...
    @typing.overload
    def parseWorld(self, uri: _nimblephysics.common.Uri) -> _nimblephysics.simulation.World:
        ...
    @typing.overload
    def parseWorld(self, uri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.simulation.World:
        ...
    @typing.overload
    def parseWorldString(self, urdfString: str, baseUri: _nimblephysics.common.Uri) -> _nimblephysics.simulation.World:
        ...
    @typing.overload
    def parseWorldString(self, urdfString: str, baseUri: _nimblephysics.common.Uri, resourceRetriever: _nimblephysics.common.ResourceRetriever) -> _nimblephysics.simulation.World:
        ...
class MJCFExporter:
    @staticmethod
    def writeSkeleton(path: str, skel: _nimblephysics.dynamics.Skeleton) -> None:
        ...
