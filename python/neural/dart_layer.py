import dartpy as dart
import torch
from typing import Tuple, Callable, List
import numpy as np
import math
import ipyopt


class BackpropSnapshotPointer:
    def __init__(self):
        self.backprop_snapshot = None


class DartLayer(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, world, pos, vel, torque, snapshot_pointer):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:

        world: dart.simulation.World
        pos: torch.Tensor
        vel: torch.Tensor
        torque: torch.Tensor
        snapshot_pointer: BackpropSnapshotPointer
        -> [torch.Tensor, torch.Tensor]
        """

        world.setPositions(pos.detach().numpy())
        world.setVelocities(vel.detach().numpy())
        world.setForces(torque.detach().numpy())
        backprop_snapshot: dart.neural.BackpropSnapshot = dart.neural.forwardPass(world)
        ctx.backprop_snapshot = backprop_snapshot
        ctx.world = world

        if snapshot_pointer is not None:
            snapshot_pointer.backprop_snapshot = backprop_snapshot

        finalPosition = np.array(world.getPositions(), copy=True)
        finalVelocity = np.array(world.getVelocities(), copy=True)
        return (torch.from_numpy(finalPosition), torch.from_numpy(finalVelocity))

    @staticmethod
    def backward(ctx, grad_pos, grad_vel):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        backprop_snapshot: dart.neural.BackpropSnapshot = ctx.backprop_snapshot
        world: dart.simulation.World = ctx.world

        # Set up the gradients for backprop
        nextTimestepGrad: dart.neural.LossGradient = dart.neural.LossGradient()
        nextTimestepGrad.lossWrtPosition = grad_pos.detach().numpy()
        nextTimestepGrad.lossWrtVelocity = grad_vel.detach().numpy()

        # Run backprop, writing values into thisTimestepGrad
        thisTimestepGrad: dart.neural.LossGradient = dart.neural.LossGradient()
        backprop_snapshot.backprop(world, thisTimestepGrad, nextTimestepGrad)

        """
        The forward computation graph looks like this:

        -------> p_t ----+-----------------------> p_t+1 ---->
                  /       \
                 /         \
        v_t ----+-----------+----(LCP Solver)----> v_t+1 ---->
                           /
                          /
        f_t -------------+
        """
        """
        print('vel pos:', backprop_snapshot.getVelPosJacobian(world))
        print('pos pos:', backprop_snapshot.getPosPosJacobian(world))
        print('vel vel:', backprop_snapshot.getVelVelJacobian(world))
        print('force vel:', backprop_snapshot.getForceVelJacobian(world))
        print('M:', backprop_snapshot.getMassMatrix(world))
        print('Minv:', backprop_snapshot.getInvMassMatrix(world))
        print('loss wrt next position:', nextTimestepGrad.lossWrtPosition)
        print('loss wrt next velocity:', nextTimestepGrad.lossWrtVelocity)
        print('loss wrt this position:', thisTimestepGrad.lossWrtPosition)
        print('loss wrt this velocity:', thisTimestepGrad.lossWrtVelocity)
        print('loss wrt this torque:', thisTimestepGrad.lossWrtTorque)
        """

        def normalize(vec):
            return vec
            """
            norm = np.linalg.norm(vec)
            if norm == 0:
                return vec
            return vec / norm
            """

        lossWrtPosition = normalize(thisTimestepGrad.lossWrtPosition)
        lossWrtVelocity = normalize(thisTimestepGrad.lossWrtVelocity)
        lossWrtTorque = normalize(thisTimestepGrad.lossWrtTorque)

        return (
            None,
            torch.tensor(lossWrtPosition, dtype=torch.float64),
            torch.tensor(lossWrtVelocity, dtype=torch.float64),
            torch.tensor(lossWrtTorque, dtype=torch.float64),
        )


def dart_layer(world: dart.simulation.World, pos: torch.Tensor, vel: torch.Tensor,
               torque: torch.Tensor, pointer: BackpropSnapshotPointer = None) -> Tuple[torch.Tensor,
                                                                                       torch.Tensor]:
    """
    This does a forward pass on the `world` that gets passed in, storing information needed
    in order to do a backwards pass.
    """
    return DartLayer.apply(world, pos, vel, torque, pointer)  # type: ignore
