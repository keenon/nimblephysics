from __future__ import annotations
import _nimblephysics.utils.UniversalLoader
import typing
import _nimblephysics.dynamics
import _nimblephysics.simulation
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "loadMeshShape",
    "loadSkeleton",
    "loadWorld"
]


def loadMeshShape(path: str) -> _nimblephysics.dynamics.MeshShape:
    pass
def loadSkeleton(world: _nimblephysics.simulation.World, path: str, basePosition: numpy.ndarray[numpy.float64, _Shape[3, 1]], baseEulerXYZ: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> _nimblephysics.dynamics.Skeleton:
    pass
def loadWorld(path: str) -> _nimblephysics.simulation.World:
    pass
