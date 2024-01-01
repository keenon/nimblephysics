"""This provides exoskeleton control and design utilities in Nimble."""
from __future__ import annotations
import nimblephysics_libs._nimblephysics.exo
import typing
import nimblephysics_libs._nimblephysics.dynamics
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "ExoSolverPinnedContact"
]


class ExoSolverPinnedContact():
    def __init__(self, realSkel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, virtualSkel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> None: 
        """
        Both the real and virtual skeletons must be identical in their number of DOFs and their structure (names of bodies, etc). The key difference is the masses, inertias, gravity, and spring forces of the virtual skeleton.
        """
    def addMotorDof(self, dofIndex: int) -> None: ...
    def analyticalForwardDynamics(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], tau: numpy.ndarray[numpy.float64, _Shape[m, 1]], exoTorques: numpy.ndarray[numpy.float64, _Shape[m, 1]], contactForces: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        This is only used for testing, to allow us to compare the analytical solution to the numerical solution.
        """
    def estimateHumanTorques(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], contactForces: numpy.ndarray[numpy.float64, _Shape[m, 1]], lastExoTorques: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        This is part of the main exoskeleton solver. It takes in the current joint velocities and accelerations, and the last exoskeleton torques, and returns the estimated human pilot joint torques.
        """
    def estimateTotalTorques(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], contactForces: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        This is part of the main exoskeleton solver. It takes in the current joint velocities and accelerations, and returns the estimated human + exo system joint torques.
        """
    def finiteDifferenceContactJacobian(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This is only used for testing: Get the Jacobian relating world space velocity of the contact points to joint velocities, by finite differencing.
        """
    def getClosestRealAccelerationConsistentWithPinsAndContactForces(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], contactForces: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        Often our estimates for `dq` and `ddq` violate the pin constraints. That leads to exo torques that do not tend to zero as the virtual human exactly matches the real human+exo system. To solve this problem, we can solve a set of least-squares equations to find the best set of ddq values to satisfy the constraint.
        """
    def getContactJacobian(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        Get the Jacobian relating world space velocity of the contact points to joint velocities.
        """
    def getExoToJointTorquesJacobian(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: ...
    def getExoTorquesLinearMap(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, n]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This is the same as solveFromBiologicalTorques, but returns the Ax + b values A and b such that Ax + b = exo_tau, accounting for the pin constraints.
        """
    def getHumanAndExoTorques(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, 1]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        Given the desired end-kinematics, after the human and exoskeleton have finished "negotiating" how they will collaborate, this computes the resulting human and exoskeleton torques.
        """
    def getPinnedForwardDynamicsForExoAndHuman(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], humanTau: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, 1]], numpy.ndarray[numpy.float64, _Shape[m, 1]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This does a simple forward dynamics step, given the current human joint torques, factoring in how the exoskeleton will respond to those torques. This returns the `ddq` that we would see on the human, and the contact forces we see at the pin constraints, and the exo torques we would get.
        """
    def getPinnedForwardDynamicsForExoAndHumanLinearMap(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, n]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This does the same thing as getPinnedForwardDynamicsForExoAndHuman, but returns the Ax + b values A and b such that Ax + b = ddq, accounting for the pin constraints.
        """
    def getPinnedRealDynamics(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], tau: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, 1]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This is not part of the main exoskeleton solver, but is useful for the inverse problem of analyzing the human pilot's joint torques under different assistance strategies.
        """
    def getPinnedRealDynamicsLinearMap(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, n]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This does the same thing as getPinndRealDynamics, but returns the Ax + b values A and b such that Ax + b = ddq, accounting for the pin constraints.
        """
    def getPinnedTotalTorques(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddqDesired: numpy.ndarray[numpy.float64, _Shape[m, 1]], centeringTau: numpy.ndarray[numpy.float64, _Shape[m, 1]], centeringForces: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, 1]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This is part of the main exoskeleton solver. It takes in how the digital twin of the exo pilot is accelerating, and attempts to solve for the torques that the exo needs to apply to get as close to that as possible.
        """
    def getPinnedTotalTorquesLinearMap(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, n]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This does the same thing as getPinnedTotalTorques, but returns the Ax + b values A and b such that Ax + b = tau, accounting for the pin constraints.
        """
    def getPinnedVirtualDynamics(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], tau: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, 1]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This is part of the main exoskeleton solver. It takes in the current estimated human pilot joint torques, and computes the accelerations we would see on the virtual skeleton if we applied those same torques, with the contacts pinned at the CoPs.
        """
    def getPinnedVirtualDynamicsLinearMap(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> typing.Tuple[numpy.ndarray[numpy.float64, _Shape[m, n]], numpy.ndarray[numpy.float64, _Shape[m, 1]]]: 
        """
        This does the same thing as getPinndVirtualDynamics, but returns the Ax + b values A and b such that Ax + b = ddq, accounting for the pin constraints.
        """
    def implicitForwardDynamics(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], tau: numpy.ndarray[numpy.float64, _Shape[m, 1]], exoTorques: numpy.ndarray[numpy.float64, _Shape[m, 1]], contactForces: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        This is only used for testing, to allow us to compare the analytical solution to the numerical solution.
        """
    def projectTorquesToExoControlSpace(self, torques: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        This is part of the main exoskeleton solver. It takes in the desired torques for the exoskeleton, and returns the torques on the actuated DOFs that can be used to drive the exoskeleton.
        """
    def projectTorquesToExoControlSpaceLinearMap(self) -> numpy.ndarray[numpy.float64, _Shape[m, n]]: 
        """
        This does the same thing as projectTorquesToExoControlSpace, but returns the matrix to multiply by the torques to get the exo torques.
        """
    def setContactPins(self, pins: typing.List[typing.Tuple[int, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None: 
        """
        Set the contact points that we will use when solving inverse dynamics.
        """
    def setPositions(self, q: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> None: ...
    def solveFromAccelerations(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], ddq: numpy.ndarray[numpy.float64, _Shape[m, 1]], lastExoTorques: numpy.ndarray[numpy.float64, _Shape[m, 1]], contactForces: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        This runs the entire exoskeleton solver pipeline, spitting out the torques to apply to the exoskeleton actuators.
        """
    def solveFromBiologicalTorques(self, dq: numpy.ndarray[numpy.float64, _Shape[m, 1]], tau: numpy.ndarray[numpy.float64, _Shape[m, 1]], centeringTau: numpy.ndarray[numpy.float64, _Shape[m, 1]], centeringForce: numpy.ndarray[numpy.float64, _Shape[m, 1]]) -> numpy.ndarray[numpy.float64, _Shape[m, 1]]: 
        """
        This is a subset of the steps in solveFromAccelerations, which can take the biological joint torques directly, and solve for the exo torques.
        """
    pass
