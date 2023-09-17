from __future__ import annotations
import _nimblephysics.dynamics
import _nimblephysics.simulation
import numpy
__all__ = ['loadMeshShape', 'loadSkeleton', 'loadWorld']
def loadMeshShape(path: str) -> _nimblephysics.dynamics.MeshShape:
    ...
def loadSkeleton(world: _nimblephysics.simulation.World, path: str, basePosition: numpy.ndarray[numpy.float64[3, 1]], baseEulerXYZ: numpy.ndarray[numpy.float64[3, 1]]) -> _nimblephysics.dynamics.Skeleton:
    ...
def loadWorld(path: str) -> _nimblephysics.simulation.World:
    ...
