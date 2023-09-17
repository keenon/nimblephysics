from __future__ import annotations
import _nimblephysics.dynamics
import _nimblephysics.simulation
__all__ = ['readSkeleton', 'readWorld', 'writeSkeleton']
def readSkeleton(path: str) -> _nimblephysics.dynamics.Skeleton:
    ...
def readWorld(path: str) -> _nimblephysics.simulation.World:
    ...
def writeSkeleton(path: str, skeleton: _nimblephysics.dynamics.Skeleton) -> None:
    ...
