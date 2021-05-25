import nimblephysics as nimble
import torch
from typing import Tuple, Callable, List
import numpy as np
import math


class MapToPosLayer(torch.autograd.Function):
  """
  This allows you to use an arbitrary mapping as a standalone PyTorch function
  """

  @staticmethod
  def forward(ctx, world, mapping, state):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    world: nimble.simulation.World
    mapping: nimble.neural.Mapping
    pos: torch.Tensor
    -> torch.Tensor
    """
    m: nimble.neural.Mapping = mapping
    world.setState(state.detach().numpy())
    mappedPos = mapping.getPositions(world)
    ctx.into = m.getRealPosToMappedPosJac(world)
    ctx.world = world

    return torch.tensor(mappedPos)

  @staticmethod
  def backward(ctx, grad_pos):
    intoJac = torch.tensor(ctx.into, dtype=torch.float64)
    lossWrtPosition = torch.matmul(torch.transpose(intoJac, 0, 1), grad_pos)
    dofs = ctx.world.getNumDofs()
    lossWrtState = torch.zeros((2 * dofs))
    lossWrtState[:dofs] = lossWrtPosition
    return (
        None,
        None,
        lossWrtState,
    )


def map_to_pos(
        world: nimble.simulation.World, map: nimble.neural.Mapping, state: torch.Tensor) -> torch.Tensor:
  """
  This maps the positions into the mapping passed in, storing necessary info in order to do a backwards pass.
  """
  return MapToPosLayer.apply(world, map, state)  # type: ignore


class MapToVelLayer(torch.autograd.Function):
  """
  This allows you to use an arbitrary mapping as a standalone PyTorch function
  """

  @staticmethod
  def forward(ctx, world, mapping, state):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    world: nimble.simulation.World
    mapping: nimble.neural.Mapping
    pos: torch.Tensor
    -> torch.Tensor
    """
    m: nimble.neural.Mapping = mapping
    world.setState(state.detach().numpy())
    mappedVel = mapping.getVelocities(world)
    ctx.world = world
    ctx.into = m.getRealVelToMappedVelJac(world)

    return torch.tensor(mappedVel)

  @staticmethod
  def backward(ctx, grad_vel):
    intoJac = torch.tensor(ctx.into, dtype=torch.float64)
    lossWrtVelocity = torch.matmul(torch.transpose(intoJac, 0, 1), grad_vel)
    dofs = ctx.world.getNumDofs()
    lossWrtState = torch.zeros((2 * dofs))
    lossWrtState[dofs:] = lossWrtVelocity
    return (
        None,
        None,
        lossWrtState,
    )


def map_to_vel(
        world: nimble.simulation.World, map: nimble.neural.Mapping, state: torch.Tensor) -> torch.Tensor:
  """
  This maps the positions into the mapping passed in, storing necessary info in order to do a backwards pass.
  """
  return MapToVelLayer.apply(world, map, state)  # type: ignore
