"""This provides gradients to DART, with an eye on embedding DART as a non-linearity in neural networks."""
from __future__ import annotations
import nimblephysics_libs._nimblephysics.neural
import typing
import nimblephysics_libs._nimblephysics.dynamics
import nimblephysics_libs._nimblephysics.performance
import nimblephysics_libs._nimblephysics.simulation
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "BackpropSnapshot",
    "COM",
    "COM_POS",
    "COM_VEL_LINEAR",
    "COM_VEL_SPATIAL",
    "ConvertToSpace",
    "IKMapping",
    "INERTIA_DIAGONAL",
    "INERTIA_FULL",
    "INERTIA_OFF_DIAGONAL",
    "IdentityMapping",
    "KnotJacobian",
    "LossGradient",
    "LossGradientHighLevelAPI",
    "MASS",
    "MappedBackpropSnapshot",
    "Mapping",
    "POS_LINEAR",
    "POS_SPATIAL",
    "VEL_LINEAR",
    "VEL_SPATIAL",
    "WRT_ACCELERATION",
    "WRT_FORCE",
    "WRT_GROUP_INERTIAS",
    "WRT_GROUP_MASSES",
    "WRT_GROUP_SCALES",
    "WRT_LINEARIZED_MASSES",
    "WRT_POSITION",
    "WRT_VELOCITY",
    "WithRespectTo",
    "WithRespectToAcceleration",
    "WithRespectToForce",
    "WithRespectToGroupCOMs",
    "WithRespectToGroupInertias",
    "WithRespectToGroupMasses",
    "WithRespectToGroupScales",
    "WithRespectToLinearizedMasses",
    "WithRespectToMass",
    "WithRespectToPosition",
    "WithRespectToVelocity",
    "WrtMassBodyNodeEntryType",
    "WrtMassBodyNodyEntry",
    "convertJointSpaceToWorldSpace",
    "forwardPass",
    "mappedForwardPass"
]


class BackpropSnapshot():
    def __init__(self, world: nimblephysics_libs._nimblephysics.simulation.World, preStepPosition: numpy.ndarray[numpy.float64, _Shape[m, 1]], preStepVelocity: numpy.ndarray[numpy.float64, _Shape[m, 1]], preStepTorques: numpy.ndarray[numpy.float64, _Shape[m, 1]], preConstraintVelocities: numpy.ndarray[numpy.float64, _Shape[m, 1]], preStepLCPCache: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def backprop(self, world: nimblephysics_libs._nimblephysics.simulation.World, thisTimestepLoss: LossGradient, nextTimestepLoss: LossGradient, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None, exploreAlternateStrategies: bool = False) -> None: ...
    def backpropState(self, world: nimblephysics_libs._nimblephysics.simulation.World, nextTimestepStateLossGrad: numpy.ndarray[numpy.float64, _Shape[m, 1]], perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None, exploreAlternateStrategies: bool = False) -> LossGradientHighLevelAPI: ...
    def benchmarkJacobians(self, world: nimblephysics_libs._nimblephysics.simulation.World, numSamples: int) -> None: ...
    def finiteDifferenceForceVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, useRidders: bool = True) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def finiteDifferencePosPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, subdivisions: int, useRidders: bool = True) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def finiteDifferenceVelPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, subdivisions: int, useRidders: bool = True) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def finiteDifferenceVelVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, useRidders: bool = True) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getActionJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getControlForceVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getInvMassMatrix(self, arg0: nimblephysics_libs._nimblephysics.simulation.World, arg1: bool) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getMassMatrix(self, arg0: nimblephysics_libs._nimblephysics.simulation.World, arg1: bool) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getMassVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPosPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPosVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPostStepPosition(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPostStepTorques(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPostStepVelocity(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPreStepPosition(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPreStepTorques(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPreStepVelocity(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getStateJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getVelPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getVelVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    pass
class ConvertToSpace():
    """
    Members:

      COM_POS

      COM_VEL_LINEAR

      COM_VEL_SPATIAL

      VEL_LINEAR

      VEL_SPATIAL

      POS_LINEAR

      POS_SPATIAL
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    COM_POS: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.COM_POS: 4>
    COM_VEL_LINEAR: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.COM_VEL_LINEAR: 6>
    COM_VEL_SPATIAL: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.COM_VEL_SPATIAL: 5>
    POS_LINEAR: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.POS_LINEAR: 1>
    POS_SPATIAL: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.POS_SPATIAL: 0>
    VEL_LINEAR: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.VEL_LINEAR: 3>
    VEL_SPATIAL: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.VEL_SPATIAL: 2>
    __members__: dict # value = {'COM_POS': <ConvertToSpace.COM_POS: 4>, 'COM_VEL_LINEAR': <ConvertToSpace.COM_VEL_LINEAR: 6>, 'COM_VEL_SPATIAL': <ConvertToSpace.COM_VEL_SPATIAL: 5>, 'VEL_LINEAR': <ConvertToSpace.VEL_LINEAR: 3>, 'VEL_SPATIAL': <ConvertToSpace.VEL_SPATIAL: 2>, 'POS_LINEAR': <ConvertToSpace.POS_LINEAR: 1>, 'POS_SPATIAL': <ConvertToSpace.POS_SPATIAL: 0>}
    pass
class Mapping():
    def getControlForceDim(self) -> int: 
        """
        Gets the dimension of the Force space in this mapping. This will be the length of the getControlForces() vector, and the length of the vector expected by setControlForces().
        """
    def getControlForceLowerLimits(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getControlForceUpperLimits(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getControlForces(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPosDim(self) -> int: 
        """
        Gets the dimension of the Position space in this mapping. This will be the length of the getPositions() vector, and the length of the vector expected by setPositions().
        """
    def getPositionLowerLimits(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPositionUpperLimits(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPositions(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getRealForceToMappedForceJac(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This returns a Jacobian that transforms the rate of change of the force in the 'real' space given by the world to the rate of change of the force in mapped space.
        """
    def getRealPosToMappedPosJac(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This returns a Jacobian that transforms the rate of change of the position in the 'real' space given by the world to the rate of change of the position in mapped space.
        """
    def getRealPosToMappedVelJac(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This returns a Jacobian that transforms the rate of change of the position in the 'real' space given by the world to the rate of change of the velocity in mapped space.
        """
    def getRealVelToMappedPosJac(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This returns a Jacobian that transforms the rate of change of the velocity in the 'real' space given by the world to the rate of change of the position in mapped space.
        """
    def getRealVelToMappedVelJac(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This returns a Jacobian that transforms the rate of change of the velocity in the 'real' space given by the world to the rate of change of the velocity in mapped space.
        """
    def getVelDim(self) -> int: 
        """
        Gets the dimension of the Velocity space in this mapping. This will be the length of the getVelocities() vector, and the length of the vector expected by setVelocities().
        """
    def getVelocities(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getVelocityLowerLimits(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getVelocityUpperLimits(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def setControlForces(self, world: nimblephysics_libs._nimblephysics.simulation.World, forces: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setPositions(self, world: nimblephysics_libs._nimblephysics.simulation.World, positions: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def setVelocities(self, world: nimblephysics_libs._nimblephysics.simulation.World, velocities: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    pass
class IdentityMapping(Mapping):
    def __init__(self, arg0: nimblephysics_libs._nimblephysics.simulation.World) -> None: ...
    pass
class KnotJacobian():
    def __init__(self) -> None: ...
    @property
    def knotPosEndPos(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @knotPosEndPos.setter
    def knotPosEndPos(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def knotPosEndVel(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @knotPosEndVel.setter
    def knotPosEndVel(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def knotVelEndPos(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @knotVelEndPos.setter
    def knotVelEndPos(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def knotVelEndVel(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, n]]
        """
    @knotVelEndVel.setter
    def knotVelEndVel(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
        pass
    @property
    def torquesEndPos(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    @torquesEndPos.setter
    def torquesEndPos(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None:
        pass
    @property
    def torquesEndVel(self) -> typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]:
        """
        :type: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]
        """
    @torquesEndVel.setter
    def torquesEndVel(self, arg0: typing.List[numpy.ndarray[numpy.float64, _Shape[m, n]]]) -> None:
        pass
    pass
class LossGradient():
    def __init__(self) -> None: ...
    @property
    def lossWrtPosition(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @lossWrtPosition.setter
    def lossWrtPosition(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def lossWrtTorque(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @lossWrtTorque.setter
    def lossWrtTorque(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def lossWrtVelocity(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @lossWrtVelocity.setter
    def lossWrtVelocity(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    pass
class LossGradientHighLevelAPI():
    def __init__(self) -> None: ...
    @property
    def lossWrtAction(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @lossWrtAction.setter
    def lossWrtAction(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def lossWrtMass(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @lossWrtMass.setter
    def lossWrtMass(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    @property
    def lossWrtState(self) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]:
        """
        :type: numpy.ndarray[numpy.float64, _Shape[m, 1]]
        """
    @lossWrtState.setter
    def lossWrtState(self, arg0: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None:
        pass
    pass
class MappedBackpropSnapshot():
    def backprop(self, world: nimblephysics_libs._nimblephysics.simulation.World, thisTimestepLoss: LossGradient, nextTimestepLosses: typing.Dict[str, LossGradient], perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None, exploreAlternateStrategies: bool = False) -> None: ...
    def getControlForceMappedVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, mapAfter: str = 'identity', perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getControlForceVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getMappings(self) -> typing.List[str]: ...
    def getMassMappedVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, mapAfter: str = 'identity', perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getMassVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPosMappedPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, mapAfter: str = 'identity', perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPosMappedVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, mapAfter: str = 'identity', perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPosPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPosVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getPostStepPosition(self, mapping: str = 'identity') -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPostStepVelocity(self, mapping: str = 'identity') -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPreStepPosition(self, mapping: str = 'identity') -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPreStepTorques(self, mapping: str = 'identity') -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getPreStepVelocity(self, mapping: str = 'identity') -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def getVelMappedPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, mapAfter: str = 'identity', perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getVelMappedVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, mapAfter: str = 'identity', perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getVelPosJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getVelVelJacobian(self, world: nimblephysics_libs._nimblephysics.simulation.World, perfLog: nimblephysics_libs._nimblephysics.performance.PerformanceLog = None) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    pass
class IKMapping(Mapping):
    def __init__(self, arg0: nimblephysics_libs._nimblephysics.simulation.World) -> None: ...
    def addAngularBodyNode(self, arg0: nimblephysics_libs._nimblephysics.dynamics.BodyNode) -> None: 
        """
        This adds the angular (3D) coordinates of a body node to the mapping, increasing the dimension of the mapped space by 3
        """
    def addLinearBodyNode(self, arg0: nimblephysics_libs._nimblephysics.dynamics.BodyNode) -> None: 
        """
        This adds the linear (3D) coordinates of a body node to the mapping, increasing the dimension of the mapped space by 3
        """
    def addSpatialBodyNode(self, arg0: nimblephysics_libs._nimblephysics.dynamics.BodyNode) -> None: 
        """
        This adds the spatial (6D) coordinates of a body node to the mapping, increasing the dimension of the mapped space by 6
        """
    pass
class WithRespectTo():
    @typing.overload
    def dim(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> int: ...
    @typing.overload
    def dim(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> int: ...
    @typing.overload
    def get(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    @typing.overload
    def get(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def lowerBound(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    def name(self) -> str: ...
    @typing.overload
    def set(self, world: nimblephysics_libs._nimblephysics.simulation.World, value: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    @typing.overload
    def set(self, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, value: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def upperBound(self, world: nimblephysics_libs._nimblephysics.simulation.World) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: ...
    pass
class WithRespectToAcceleration(WithRespectTo):
    pass
class WithRespectToForce(WithRespectTo):
    pass
class WithRespectToGroupCOMs(WithRespectTo):
    pass
class WithRespectToGroupInertias(WithRespectTo):
    pass
class WithRespectToGroupMasses(WithRespectTo):
    pass
class WithRespectToGroupScales(WithRespectTo):
    pass
class WithRespectToLinearizedMasses(WithRespectTo):
    pass
class WithRespectToMass():
    def registerNode(self, node: nimblephysics_libs._nimblephysics.dynamics.BodyNode, type: WrtMassBodyNodeEntryType, upperBound: numpy.ndarray[numpy.float64, _Shape[m, 1]], lowerBound: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> WrtMassBodyNodyEntry: ...
    pass
class WithRespectToPosition(WithRespectTo):
    pass
class WithRespectToVelocity(WithRespectTo):
    pass
class WrtMassBodyNodeEntryType():
    """
    Members:

      MASS

      COM

      INERTIA_DIAGONAL

      INERTIA_OFF_DIAGONAL

      INERTIA_FULL
    """
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    @property
    def name(self) -> str:
        """
        :type: str
        """
    @property
    def value(self) -> int:
        """
        :type: int
        """
    COM: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.COM: 1>
    INERTIA_DIAGONAL: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.INERTIA_DIAGONAL: 3>
    INERTIA_FULL: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.INERTIA_FULL: 5>
    INERTIA_OFF_DIAGONAL: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.INERTIA_OFF_DIAGONAL: 4>
    MASS: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.MASS: 0>
    __members__: dict # value = {'MASS': <WrtMassBodyNodeEntryType.MASS: 0>, 'COM': <WrtMassBodyNodeEntryType.COM: 1>, 'INERTIA_DIAGONAL': <WrtMassBodyNodeEntryType.INERTIA_DIAGONAL: 3>, 'INERTIA_OFF_DIAGONAL': <WrtMassBodyNodeEntryType.INERTIA_OFF_DIAGONAL: 4>, 'INERTIA_FULL': <WrtMassBodyNodeEntryType.INERTIA_FULL: 5>}
    pass
class WrtMassBodyNodyEntry():
    def __init__(self, arg0: str, arg1: WrtMassBodyNodeEntryType) -> None: ...
    @property
    def linkName(self) -> str:
        """
        :type: str
        """
    @linkName.setter
    def linkName(self, arg0: str) -> None:
        pass
    @property
    def type(self) -> WrtMassBodyNodeEntryType:
        """
        :type: WrtMassBodyNodeEntryType
        """
    @type.setter
    def type(self, arg0: WrtMassBodyNodeEntryType) -> None:
        pass
    pass
def convertJointSpaceToWorldSpace(world: nimblephysics_libs._nimblephysics.simulation.World, jointSpace: numpy.ndarray[numpy.float64, _Shape[m, n]], nodes: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], space: ConvertToSpace, backprop: bool = False, useIK: bool = True) -> numpy.ndarray[numpy.float64, _Shape[m, n]]:
    """
    Convert a set of joint positions to a vector of body positions in world space (expressed in log space).
    """
def forwardPass(world: nimblephysics_libs._nimblephysics.simulation.World, idempotent: bool = False) -> BackpropSnapshot:
    pass
def mappedForwardPass(world: nimblephysics_libs._nimblephysics.simulation.World, mappings: typing.Dict[str, Mapping], idempotent: bool = False) -> MappedBackpropSnapshot:
    pass
COM: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.COM: 1>
COM_POS: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.COM_POS: 4>
COM_VEL_LINEAR: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.COM_VEL_LINEAR: 6>
COM_VEL_SPATIAL: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.COM_VEL_SPATIAL: 5>
INERTIA_DIAGONAL: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.INERTIA_DIAGONAL: 3>
INERTIA_FULL: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.INERTIA_FULL: 5>
INERTIA_OFF_DIAGONAL: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.INERTIA_OFF_DIAGONAL: 4>
MASS: nimblephysics_libs._nimblephysics.neural.WrtMassBodyNodeEntryType # value = <WrtMassBodyNodeEntryType.MASS: 0>
POS_LINEAR: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.POS_LINEAR: 1>
POS_SPATIAL: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.POS_SPATIAL: 0>
VEL_LINEAR: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.VEL_LINEAR: 3>
VEL_SPATIAL: nimblephysics_libs._nimblephysics.neural.ConvertToSpace # value = <ConvertToSpace.VEL_SPATIAL: 2>
WRT_ACCELERATION: nimblephysics_libs._nimblephysics.neural.WithRespectToAcceleration
WRT_FORCE: nimblephysics_libs._nimblephysics.neural.WithRespectToForce
WRT_GROUP_INERTIAS: nimblephysics_libs._nimblephysics.neural.WithRespectToGroupInertias
WRT_GROUP_MASSES: nimblephysics_libs._nimblephysics.neural.WithRespectToGroupMasses
WRT_GROUP_SCALES: nimblephysics_libs._nimblephysics.neural.WithRespectToGroupScales
WRT_LINEARIZED_MASSES: nimblephysics_libs._nimblephysics.neural.WithRespectToLinearizedMasses
WRT_POSITION: nimblephysics_libs._nimblephysics.neural.WithRespectToPosition
WRT_VELOCITY: nimblephysics_libs._nimblephysics.neural.WithRespectToVelocity
