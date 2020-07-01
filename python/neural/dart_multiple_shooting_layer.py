import dartpy as dart
import torch
from typing import Tuple, Callable, List
import numpy as np
import math
import ipyopt


class BackpropSnapshotPointer:
    def __init__(self):
        self.backprop_snapshot = None


class DartMultipleShootingLayer(torch.autograd.Function):
    """
    This implements a batch version of DartLayer, for multiple-shooting trajectory optimization.
    """

    @staticmethod
    def forward(ctx, world, torques, shooting_length, knot_poses, knot_vels):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:

        world: dart.simulation.World
        torques: List[torch.Tensor]
        shooting_length: int
        knot_poses: List[torch.Tensor]
        knot_vels: List[torch.Tensor]
        -> [torch.Tensor, torch.Tensor]
        """

        backprop_snapshots: List[dart.neural.BackpropSnapshot] = dart.neural.bulkForwardPass(
            world, torques, shooting_length, knot_poses, knot_vels)
        ctx.backprop_snapshots = backprop_snapshots
        ctx.world = world

        return (torch.tensor(world.getPositions()), torch.tensor(world.getVelocities()))

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

        lossWrtPosition = thisTimestepGrad.lossWrtPosition
        lossWrtVelocity = thisTimestepGrad.lossWrtVelocity
        lossWrtTorque = thisTimestepGrad.lossWrtTorque

        return (
            None,
            torch.tensor(lossWrtPosition, dtype=torch.float64),
            torch.tensor(lossWrtVelocity, dtype=torch.float64),
            torch.tensor(lossWrtTorque, dtype=torch.float64),
            None
        )


def dart_multiple_shooting_layer(world: dart.simulation.World, pos: torch.Tensor, vel: torch.Tensor,
                                 torque: torch.Tensor, pointer: BackpropSnapshotPointer = None) -> Tuple[torch.Tensor,
                                                                                                         torch.Tensor]:
    """
    This does a forward pass on the `world` that gets passed in, storing information needed
    in order to do a backwards pass.
    """
    return DartLayer.apply(world, pos, vel, torque, pointer)  # type: ignore
