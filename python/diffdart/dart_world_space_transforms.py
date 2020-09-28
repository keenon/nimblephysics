import diffdart_libs._diffdart as dart
import torch
import numpy as np
from typing import List


class DartTransformToWorldSpaceLayer(torch.autograd.Function):
    """
    This implements a single, differentiable timestep of DART as a PyTorch layer
    """

    @staticmethod
    def forward(ctx, world, space, nodes, pos):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:

        world: dart.simulation.World
        space: dart.neural.ConvertToSpace
        nodes: List[dart.dynamics.BodyNode]
        pos: torch.Tensor
        -> [torch.Tensor]
        """
        ctx.world = world
        if not isinstance(nodes, list):
            nodes = [nodes]
        ctx.nodes = nodes
        ctx.is_vector = len(pos.shape) == 1
        ctx.space = space
        result = dart.neural.convertJointSpaceToWorldSpace(
            world, pos.detach().numpy(), nodes, space, backprop=False)
        if ctx.is_vector:
            assert result.shape[1] == 1
            result = result.squeeze(1)
        return torch.tensor(result)

    @staticmethod
    def backward(ctx, grad_world_pos):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        world: dart.simulation.World = ctx.world
        nodes: List[dart.dynamics.BodyNode] = ctx.nodes
        space: dart.neural.ConvertToSpace = ctx.space
        is_vector: bool = ctx.is_vector
        gradient = dart.neural.convertJointSpaceToWorldSpace(
            world, grad_world_pos, nodes, space, backprop=True, useIK=True)

        # Normalize by column and multiply by 1e-2, to stabilize learning
        gradient = (gradient / abs(gradient.max(0)[0])) * 1e-2

        if is_vector:
            assert(gradient.shape[1] == 1)
            gradient = gradient.squeeze(1)
        return (None, None, None, torch.tensor(gradient))


def convert_to_world_space_positions_linear(
        world: dart.simulation.World, nodes: List[dart.dynamics.BodyNode],
        jointPos: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint positions (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartTransformToWorldSpaceLayer.apply(  # type: ignore
        world, dart.neural.POS_LINEAR, nodes, jointPos)


def convert_to_world_space_positions_spatial(
        world: dart.simulation.World, nodes: List[dart.dynamics.BodyNode],
        jointPos: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint positions (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartTransformToWorldSpaceLayer.apply(  # type: ignore
        world, dart.neural.POS_SPATIAL, nodes, jointPos)


def convert_to_world_space_velocities_linear(
        world: dart.simulation.World, nodes: List[dart.dynamics.BodyNode],
        jointVel: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint velocities (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartTransformToWorldSpaceLayer.apply(  # type: ignore
        world, dart.neural.VEL_LINEAR, nodes, jointVel)


def convert_to_world_space_velocities_spatial(
        world: dart.simulation.World, nodes: List[dart.dynamics.BodyNode],
        jointVel: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint velocities (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartTransformToWorldSpaceLayer.apply(  # type: ignore
        world, dart.neural.VEL_SPATIAL, nodes, jointVel)


def convert_to_world_space_center_of_mass(
        world: dart.simulation.World, nodes: List[dart.dynamics.BodyNode],
        jointPos: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint positions (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartTransformToWorldSpaceLayer.apply(  # type: ignore
        world, dart.neural.COM_POS, nodes, jointPos)


def convert_to_world_space_center_of_mass_vel_linear(
        world: dart.simulation.World, nodes: List[dart.dynamics.BodyNode],
        jointVel: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint positions (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartTransformToWorldSpaceLayer.apply(  # type: ignore
        world, dart.neural.COM_VEL_LINEAR, nodes, jointVel)


def convert_to_world_space_center_of_mass_vel_spatial(
        world: dart.simulation.World, nodes: List[dart.dynamics.BodyNode],
        jointVel: torch.Tensor) -> torch.Tensor:
    """
    This converts a set of joint positions (each column is an individual example) into
    log-space world space coordinates, concatenated together.
    """
    return DartTransformToWorldSpaceLayer.apply(  # type: ignore
        world, dart.neural.COM_VEL_SPATIAL, nodes, jointVel)
