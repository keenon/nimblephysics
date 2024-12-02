from __future__ import annotations
import nimblephysics_libs._nimblephysics.utils.UniversalLoader
import typing
import nimblephysics_libs._nimblephysics.dynamics
import nimblephysics_libs._nimblephysics.simulation
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "loadMeshShape",
    "loadSkeleton",
    "loadWorld"
]


def loadMeshShape(path: str) -> nimblephysics_libs._nimblephysics.dynamics.MeshShape:
    pass
def loadSkeleton(world: nimblephysics_libs._nimblephysics.simulation.World, path: str, basePosition: numpy.ndarray[numpy.float64, _Shape[3, 1]], baseEulerXYZ: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> nimblephysics_libs._nimblephysics.dynamics.Skeleton:
    pass
def loadWorld(path: str) -> nimblephysics_libs._nimblephysics.simulation.World:
    pass
