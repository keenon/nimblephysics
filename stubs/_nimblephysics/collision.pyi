from __future__ import annotations
import _nimblephysics.dynamics
import numpy
import typing
__all__ = ['CollisionDetector', 'CollisionFilter', 'CollisionGroup', 'CollisionObject', 'CollisionOption', 'CollisionResult', 'Contact', 'DARTCollisionDetector', 'DARTCollisionGroup', 'DistanceOption', 'DistanceResult', 'RayHit', 'RaycastOption', 'RaycastResult']
class CollisionDetector:
    def cloneWithoutCollisionObjects(self) -> CollisionDetector:
        ...
    def createCollisionGroup(self) -> CollisionGroup:
        ...
    def getType(self) -> str:
        ...
class CollisionFilter:
    pass
class CollisionGroup:
    def addShapeFrame(self, shapeFrame: _nimblephysics.dynamics.ShapeFrame) -> None:
        ...
    def addShapeFrames(self, shapeFrames: list[_nimblephysics.dynamics.ShapeFrame]) -> None:
        ...
    def addShapeFramesOf(self) -> None:
        ...
    @typing.overload
    def collide(self) -> bool:
        ...
    @typing.overload
    def collide(self, option: CollisionOption) -> bool:
        ...
    @typing.overload
    def collide(self, option: CollisionOption, result: CollisionResult) -> bool:
        ...
    @typing.overload
    def distance(self) -> float:
        ...
    @typing.overload
    def distance(self, option: DistanceOption) -> float:
        ...
    @typing.overload
    def distance(self, option: DistanceOption, result: DistanceResult) -> float:
        ...
    def getAutomaticUpdate(self) -> bool:
        ...
    def getNumShapeFrames(self) -> int:
        ...
    def hasShapeFrame(self, shapeFrame: _nimblephysics.dynamics.ShapeFrame) -> bool:
        ...
    @typing.overload
    def raycast(self, from_point: numpy.ndarray[numpy.float64[3, 1]], to_point: numpy.ndarray[numpy.float64[3, 1]]) -> bool:
        ...
    @typing.overload
    def raycast(self, from_point: numpy.ndarray[numpy.float64[3, 1]], to_point: numpy.ndarray[numpy.float64[3, 1]], option: RaycastOption) -> bool:
        ...
    @typing.overload
    def raycast(self, from_point: numpy.ndarray[numpy.float64[3, 1]], to_point: numpy.ndarray[numpy.float64[3, 1]], option: RaycastOption, result: RaycastResult) -> bool:
        ...
    def removeAllShapeFrames(self) -> None:
        ...
    def removeDeletedShapeFrames(self) -> None:
        ...
    def removeShapeFrame(self, shapeFrame: _nimblephysics.dynamics.ShapeFrame) -> None:
        ...
    def removeShapeFrames(self, shapeFrames: list[_nimblephysics.dynamics.ShapeFrame]) -> None:
        ...
    def removeShapeFramesOf(self) -> None:
        ...
    @typing.overload
    def setAutomaticUpdate(self) -> None:
        ...
    @typing.overload
    def setAutomaticUpdate(self, automatic: bool) -> None:
        ...
    def subscribeTo(self) -> None:
        ...
    def update(self) -> None:
        ...
class CollisionObject:
    def getShape(self) -> _nimblephysics.dynamics.Shape:
        ...
class CollisionOption:
    collisionFilter: CollisionFilter
    enableContact: bool
    maxNumContacts: int
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, enableContact: bool) -> None:
        ...
    @typing.overload
    def __init__(self, enableContact: bool, maxNumContacts: int) -> None:
        ...
    @typing.overload
    def __init__(self, enableContact: bool, maxNumContacts: int, collisionFilter: CollisionFilter) -> None:
        ...
class CollisionResult:
    def __init__(self) -> None:
        ...
    def clear(self) -> None:
        ...
    def getContact(self, arg0: int) -> Contact:
        ...
    def getContacts(self) -> list[Contact]:
        ...
    def getNumContacts(self) -> int:
        ...
    @typing.overload
    def inCollision(self, bn: _nimblephysics.dynamics.BodyNode) -> bool:
        ...
    @typing.overload
    def inCollision(self, frame: _nimblephysics.dynamics.ShapeFrame) -> bool:
        ...
    def isCollision(self) -> bool:
        ...
class Contact:
    collisionObject1: CollisionObject
    collisionObject2: CollisionObject
    force: numpy.ndarray[numpy.float64[3, 1]]
    isFrictionOn: bool
    lcpResult: float
    lcpResultTangent1: float
    lcpResultTangent2: float
    normal: numpy.ndarray[numpy.float64[3, 1]]
    penetrationDepth: float
    point: numpy.ndarray[numpy.float64[3, 1]]
    spatialNormalA: numpy.ndarray[numpy.float64[6, n]]
    spatialNormalB: numpy.ndarray[numpy.float64[6, n]]
    tangent1: numpy.ndarray[numpy.float64[3, 1]]
    tangent2: numpy.ndarray[numpy.float64[3, 1]]
    triID1: int
    triID2: int
    userData: capsule
    @staticmethod
    def getNormalEpsilon() -> float:
        ...
    @staticmethod
    def getNormalEpsilonSquared() -> float:
        ...
    @staticmethod
    def isNonZeroNormal(normal: numpy.ndarray[numpy.float64[3, 1]]) -> bool:
        ...
    @staticmethod
    def isZeroNormal(normal: numpy.ndarray[numpy.float64[3, 1]]) -> bool:
        ...
    def __init__(self) -> None:
        ...
class DARTCollisionDetector(CollisionDetector):
    @staticmethod
    def getStaticType() -> str:
        ...
    def __init__(self) -> None:
        ...
    def cloneWithoutCollisionObjects(self) -> CollisionDetector:
        ...
    def createCollisionGroup(self) -> CollisionGroup:
        ...
    def getType(self) -> str:
        ...
class DARTCollisionGroup(CollisionGroup):
    def __init__(self, collisionDetector: CollisionDetector) -> None:
        ...
class DistanceOption:
    distanceLowerBound: float
    enableNearestPoints: bool
class DistanceResult:
    minDistance: float
    nearestPoint1: numpy.ndarray[numpy.float64[3, 1]]
    nearestPoint2: numpy.ndarray[numpy.float64[3, 1]]
    unclampedMinDistance: float
    def clear(self) -> None:
        ...
    def found(self) -> bool:
        ...
    def isMinDistanceClamped(self) -> bool:
        ...
class RayHit:
    def __init__(self) -> None:
        ...
    @property
    def mFraction(self) -> float:
        """
        The fraction from `from` point to `to` point
        """
    @mFraction.setter
    def mFraction(self, arg0: float) -> None:
        ...
    @property
    def mNormal(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        The normal at the hit point in the world coordinates
        """
    @mNormal.setter
    def mNormal(self, arg0: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    @property
    def mPoint(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        The hit point in the world coordinates
        """
    @mPoint.setter
    def mPoint(self, arg0: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
class RaycastOption:
    mEnableAllHits: bool
    mSortByClosest: bool
class RaycastResult:
    mRayHits: list[RayHit]
    def __init__(self) -> None:
        ...
    def clear(self) -> None:
        ...
    def hasHit(self) -> bool:
        ...
