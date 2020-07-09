import dartpy as dart
import torch
import numpy as np


class DartWorldSpacePositionLayer(torch.autograd.Function):
    """
    This implements a single, differentiable timestep of DART as a PyTorch layer
    """

    @staticmethod
    def forward(ctx, world, pos):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:

        world: dart.simulation.World
        pos: torch.Tensor
        -> [torch.Tensor]
        """
        return torch.tensor(dart.neural.convertJointSpacePositionsToWorldSpace(
            world, pos.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_world_pos):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return torch.tensor(dart.neural.backpropWorldSpaceToJointSpace(
            world, grad_world_pos.detach().numpy()))


class DartWorldSpaceVelocityLayer(torch.autograd.Function):
    """
    This implements a single, differentiable timestep of DART as a PyTorch layer
    """

    @staticmethod
    def forward(ctx, world, vel):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:

        world: dart.simulation.World
        vel: torch.Tensor
        -> [torch.Tensor]
        """
        return torch.tensor(dart.neural.convertJointSpaceVelocitiesToWorldSpace(
            world, vel.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_world_vel):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return torch.tensor(dart.neural.backpropWorldSpaceToJointSpace(
            world, grad_world_vel.detach().numpy()))


def convert_to_world_space_positions(
        world: dart.simulation.World, jointPos: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint positions (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartWorldSpacePositionLayer.apply(world, jointPos)  # type: ignore


def convert_to_world_space_velocities(
        world: dart.simulation.World, jointVel: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint velocities (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartWorldSpaceVelocityLayer.apply(world, jointVel)  # type: ignore
