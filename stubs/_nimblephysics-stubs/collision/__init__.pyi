from __future__ import annotations
import nimblephysics_libs._nimblephysics.collision
import typing
import nimblephysics_libs._nimblephysics.dynamics
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "CollisionDetector",
    "CollisionFilter",
    "CollisionGroup",
    "CollisionObject",
    "CollisionOption",
    "CollisionResult",
    "Contact",
    "DARTCollisionDetector",
    "DARTCollisionGroup",
    "DistanceOption",
    "DistanceResult",
    "RayHit",
    "RaycastOption",
    "RaycastResult"
]


class CollisionDetector():
    def cloneWithoutCollisionObjects(self) -> CollisionDetector: ...
    def createCollisionGroup(self) -> CollisionGroup: ...
    def getType(self) -> str: ...
    pass
class CollisionFilter():
    pass
class CollisionGroup():
    def addShapeFrame(self, shapeFrame: nimblephysics_libs._nimblephysics.dynamics.ShapeFrame) -> None: ...
    def addShapeFrames(self, shapeFrames: typing.List[nimblephysics_libs._nimblephysics.dynamics.ShapeFrame]) -> None: ...
    def addShapeFramesOf(self) -> None: ...
    @typing.overload
    def collide(self) -> bool: ...
    @typing.overload
    def collide(self, option: CollisionOption) -> bool: ...
    @typing.overload
    def collide(self, option: CollisionOption, result: CollisionResult) -> bool: ...
    @typing.overload
    def distance(self) -> float: ...
    @typing.overload
    def distance(self, option: DistanceOption) -> float: ...
    @typing.overload
    def distance(self, option: DistanceOption, result: DistanceResult) -> float: ...
    def getAutomaticUpdate(self) -> bool: ...
    def getNumShapeFrames(self) -> int: ...
    def hasShapeFrame(self, shapeFrame: nimblephysics_libs._nimblephysics.dynamics.ShapeFrame) -> bool: ...
    @typing.overload
    def raycast(self, from_point: numpy.ndarray[numpy.float64, _Shape[3, 1]], to_point: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> bool: ...
    @typing.overload
    def raycast(self, from_point: numpy.ndarray[numpy.float64, _Shape[3, 1]], to_point: numpy.ndarray[numpy.float64, _Shape[3, 1]], option: RaycastOption) -> bool: ...
    @typing.overload
    def raycast(self, from_point: numpy.ndarray[numpy.float64, _Shape[3, 1]], to_point: numpy.ndarray[numpy.float64, _Shape[3, 1]], option: RaycastOption, result: RaycastResult) -> bool: ...
    def removeAllShapeFrames(self) -> None: ...
    def removeDeletedShapeFrames(self) -> None: ...
    def removeShapeFrame(self, shapeFrame: nimblephysics_libs._nimblephysics.dynamics.ShapeFrame) -> None: ...
    def removeShapeFrames(self, shapeFrames: typing.List[nimblephysics_libs._nimblephysics.dynamics.ShapeFrame]) -> None: ...
    def removeShapeFramesOf(self) -> None: ...
    @typing.overload
    def setAutomaticUpdate(self) -> None: ...
    @typing.overload
    def setAutomaticUpdate(self, automatic: bool) -> None: ...
    def subscribeTo(self) -> None: ...
    def update(self) -> None: ...
    pass
class CollisionObject():
    def getShape(self) -> nimblephysics_libs._nimblephysics.dynamics.Shape: ...
    pass
class CollisionOption():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, enableContact: bool) -> None: ...
    @typing.overload
    def __init__(self, enableContact: bool, maxNumContacts: int) -> None: ...
    @typing.overload
    def __init__(self, enableContact: bool, maxNumContacts: int, collisionFilter: CollisionFilter) -> None: ...
    @property
    def collisionFilter(self) -> CollisionFilter:
        """
        :type: CollisionFilter
        """
    @collisionFilter.setter
    def collisionFilter(self, arg0: CollisionFilter) -> None:
        pass
    @property
    def enableContact(self) -> bool:
        """
        :type: bool
        """
    @enableContact.setter
    def enableContact(self, arg0: bool) -> None:
        pass
    @property
    def maxNumContacts(self) -> int:
        """
        :type: int
        """
    @maxNumContacts.setter
    def maxNumContacts(self, arg0: int) -> None:
        pass
    pass
class CollisionResult():
    def __init__(self) -> None: ...
    def clear(self) -> None: ...
    def getContact(self, arg0: int) -> Contact: ...
    def getContacts(self) -> typing.List[Contact]: ...
    def getNumContacts(self) -> int: ...
    @typing.overload
    def inCollision(self, bn: nimblephysics_libs._nimblephysics.dynamics.BodyNode) -> bool: ...
    @typing.overload
    def inCollision(self, frame: nimblephysics_libs._nimblephysics.dynamics.ShapeFrame) -> bool: ...
    def isCollision(self) -> bool: ...
    pass
class Contact():
    def __init__(self) -> None: ...
    @staticmethod
    def getNormalEpsilon() -> float: ...
    @staticmethod
    def getNormalEpsilonSquared() -> float: ...
    @staticmethod
    def isNonZeroNormal(normal: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> bool: ...
    @staticmethod
    def isZeroNormal(normal: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> bool: ...
    @property
    def collisionObject1(self) -> CollisionObject:
        """
        :type: CollisionObject
        """
    @collisionObject1.setter
    def collisionObject1(self, arg0: CollisionObject) -> None:
        pass
    @property
    def collisionObject2(self) -> CollisionObject:
        """
        :type: CollisionObject
        """
    @collisionObject2.setter
    def collisionObject2(self, arg0: CollisionObject) -> None:
        pass
    @property
    def force(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @force.setter
    def force(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    @property
    def isFrictionOn(self) -> bool:
        """
        :type: bool
        """
    @isFrictionOn.setter
    def isFrictionOn(self, arg0: bool) -> None:
        pass
    @property
    def lcpResult(self) -> float:
        """
        :type: float
        """
    @lcpResult.setter
    def lcpResult(self, arg0: float) -> None:
        pass
    @property
    def lcpResultTangent1(self) -> float:
        """
        :type: float
        """
    @lcpResultTangent1.setter
    def lcpResultTangent1(self, arg0: float) -> None:
        pass
    @property
    def lcpResultTangent2(self) -> float:
        """
        :type: float
        """
    @lcpResultTangent2.setter
    def lcpResultTangent2(self, arg0: float) -> None:
        pass
    @property
    def normal(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @normal.setter
    def normal(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    @property
    def penetrationDepth(self) -> float:
        """
        :type: float
        """
    @penetrationDepth.setter
    def penetrationDepth(self, arg0: float) -> None:
        pass
    @property
    def point(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @point.setter
    def point(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    @property
    def spatialNormalA(self) -> numpy.ndarray[numpy.float64, _Shape[6, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[6, n]]
        """
    @spatialNormalA.setter
    def spatialNormalA(self, arg0: numpy.ndarray[numpy.float64, _Shape[6, n]]) -> None:
        pass
    @property
    def spatialNormalB(self) -> numpy.ndarray[numpy.float64, _Shape[6, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[6, n]]
        """
    @spatialNormalB.setter
    def spatialNormalB(self, arg0: numpy.ndarray[numpy.float64, _Shape[6, n]]) -> None:
        pass
    @property
    def tangent1(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @tangent1.setter
    def tangent1(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    @property
    def tangent2(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @tangent2.setter
    def tangent2(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    @property
    def triID1(self) -> int:
        """
        :type: int
        """
    @triID1.setter
    def triID1(self, arg0: int) -> None:
        pass
    @property
    def triID2(self) -> int:
        """
        :type: int
        """
    @triID2.setter
    def triID2(self, arg0: int) -> None:
        pass
    @property
    def userData(self) -> capsule:
        """
        :type: capsule
        """
    @userData.setter
    def userData(self, arg0: capsule) -> None:
        pass
    pass
class DARTCollisionDetector(CollisionDetector):
    def __init__(self) -> None: ...
    def cloneWithoutCollisionObjects(self) -> CollisionDetector: ...
    def createCollisionGroup(self) -> CollisionGroup: ...
    @staticmethod
    def getStaticType() -> str: ...
    def getType(self) -> str: ...
    pass
class DARTCollisionGroup(CollisionGroup):
    def __init__(self, collisionDetector: CollisionDetector) -> None: ...
    pass
class DistanceOption():
    @property
    def distanceLowerBound(self) -> float:
        """
        :type: float
        """
    @distanceLowerBound.setter
    def distanceLowerBound(self, arg0: float) -> None:
        pass
    @property
    def enableNearestPoints(self) -> bool:
        """
        :type: bool
        """
    @enableNearestPoints.setter
    def enableNearestPoints(self, arg0: bool) -> None:
        pass
    pass
class DistanceResult():
    def clear(self) -> None: ...
    def found(self) -> bool: ...
    def isMinDistanceClamped(self) -> bool: ...
    @property
    def minDistance(self) -> float:
        """
        :type: float
        """
    @minDistance.setter
    def minDistance(self, arg0: float) -> None:
        pass
    @property
    def nearestPoint1(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @nearestPoint1.setter
    def nearestPoint1(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    @property
    def nearestPoint2(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @nearestPoint2.setter
    def nearestPoint2(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        pass
    @property
    def unclampedMinDistance(self) -> float:
        """
        :type: float
        """
    @unclampedMinDistance.setter
    def unclampedMinDistance(self, arg0: float) -> None:
        pass
    pass
class RayHit():
    def __init__(self) -> None: ...
    @property
    def mFraction(self) -> float:
        """
        The fraction from `from` point to `to` point

        :type: float
        """
    @mFraction.setter
    def mFraction(self, arg0: float) -> None:
        """
        The fraction from `from` point to `to` point
        """
    @property
    def mNormal(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        The normal at the hit point in the world coordinates

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @mNormal.setter
    def mNormal(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        """
        The normal at the hit point in the world coordinates
        """
    @property
    def mPoint(self) -> numpy.ndarray[numpy.float64, _Shape[3, 1]]:
        """
        The hit point in the world coordinates

        :type: numpy.ndarray[numpy.float64, _Shape[3, 1]]
        """
    @mPoint.setter
    def mPoint(self, arg0: numpy.ndarray[numpy.float64, _Shape[3, 1]]) -> None:
        """
        The hit point in the world coordinates
        """
    pass
class RaycastOption():
    @property
    def mEnableAllHits(self) -> bool:
        """
        :type: bool
        """
    @mEnableAllHits.setter
    def mEnableAllHits(self, arg0: bool) -> None:
        pass
    @property
    def mSortByClosest(self) -> bool:
        """
        :type: bool
        """
    @mSortByClosest.setter
    def mSortByClosest(self, arg0: bool) -> None:
        pass
    pass
class RaycastResult():
    def __init__(self) -> None: ...
    def clear(self) -> None: ...
    def hasHit(self) -> bool: ...
    @property
    def mRayHits(self) -> typing.List[RayHit]:
        """
        :type: typing.List[RayHit]
        """
    @mRayHits.setter
    def mRayHits(self, arg0: typing.List[RayHit]) -> None:
        pass
    pass
