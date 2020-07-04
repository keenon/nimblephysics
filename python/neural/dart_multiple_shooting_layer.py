import dartpy as dart
import torch
from typing import Tuple, Callable, List
import numpy as np
import math
import ipyopt
from dart_layer import BackpropSnapshotPointer


class BulkPassResultPointer:
    def __init__(self):
        self.forwardPassResult: dart.neural.BulkForwardPassResult = None
        self.backwardPassResult: dart.neural.BulkBackwardPassResult = None


class DartMultipleShootingLayer(torch.autograd.Function):
    """
    This implements a batch version of DartLayer, for multiple-shooting trajectory optimization.
    """

    @staticmethod
    def forward(ctx, world, torques, shooting_length, knot_poses, knot_vels, pointer):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:

        world: dart.simulation.World
        torques: torch.Tensor
        shooting_length: int
        knot_poses: torch.Tensor
        knot_vels: torch.Tensor
        pointer: BulkForwardPassResultPointer = None 
        -> [torch.Tensor, torch.Tensor]
        """

        assert(torques.shape[0] == world.getNumDofs())

        result: dart.neural.BulkForwardPassResult = dart.neural.bulkForwardPass(
            world, torques.detach().numpy(),
            shooting_length, knot_poses.detach().numpy(),
            knot_vels.detach().numpy())
        ctx.backprop_snapshots = result.snapshots
        ctx.world = world
        ctx.shooting_length = shooting_length
        ctx.pointer = pointer

        if pointer is not None:
            pointer.forwardPassResult = result

        return (
            torch.tensor(result.postStepPoses),
            torch.tensor(result.postStepVels)
        )

    @staticmethod
    def backward(ctx, grad_poses, grad_vels):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input arrays.

        grad_poses: torch.Tensor
        grad_vels: torch.Tensor
        """
        backprop_snapshots: List[dart.neural.BackpropSnapshot] = ctx.backprop_snapshots
        world: dart.simulation.World = ctx.world
        shooting_length: int = ctx.shooting_length

        result: dart.neural.BulkBackwardPassResult = dart.neural.bulkBackwardPass(
            world, backprop_snapshots, shooting_length, grad_poses.detach().numpy(), grad_vels.detach().numpy())

        if ctx.pointer is not None:
            ctx.pointer.backwardPassResult = result

        return (
            None,
            torch.tensor(result.gradWrtPreStepTorques),
            None,
            torch.tensor(result.gradWrtPreStepKnotPoses),
            torch.tensor(result.gradWrtPreStepKnotVels),
            None
        )


def dart_multiple_shooting_layer(world: dart.simulation.World, torques: List[torch.Tensor],
                                 shooting_length: int, knot_poses: List[torch.Tensor],
                                 knot_vels: List[torch.Tensor],
                                 pointers: List[BackpropSnapshotPointer] = None) -> Tuple[List[torch.Tensor],
                                                                                          List[torch.Tensor]]:
    """
    This does a forward pass on the `world` that gets passed in, storing information needed
    in order to do a backwards pass.
    """
    return DartMultipleShootingLayer.apply(  # type: ignore
        world, torques, shooting_length, knot_poses, knot_vels, pointers)
