import nimblephysics_libs._nimblephysics as nimble
import torch
from typing import Tuple, Callable, List, Optional
import numpy as np
import math


class BackpropSnapshotPointer:
  def __init__(self):
    self.backprop_snapshot = None


class TimestepLayer(torch.autograd.Function):
  """
  This implements a single, differentiable timestep of DART as a PyTorch layer
  """

  @staticmethod
  def forward(ctx, world, state, action, mass):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    world: nimble.simulation.World
    state: torch.Tensor
    action: torch.Tensor
    -> torch.Tensor
    """

    world.setState(state.detach().numpy())
    world.setAction(action.detach().numpy())
    ctx.use_mass = mass is not None
    if ctx.use_mass:
      world.setMasses(mass.detach().numpy())
    backprop_snapshot: nimble.neural.BackpropSnapshot = nimble.neural.forwardPass(world)
    ctx.backprop_snapshot = backprop_snapshot
    ctx.world = world

    return torch.tensor(world.getState())

  @staticmethod
  def backward(ctx, grad_state):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    backprop_snapshot: nimble.neural.BackpropSnapshot = ctx.backprop_snapshot
    world: nimble.simulation.World = ctx.world

    grads: nimble.neural.LossGradientHighLevelAPI = backprop_snapshot.backpropState(
        world, grad_state.detach().numpy())
    
    return (
        None,
        torch.tensor(grads.lossWrtState, dtype=torch.float64),
        torch.tensor(grads.lossWrtAction, dtype=torch.float64),
        torch.tensor(grads.lossWrtMass, dtype=torch.float64) if ctx.use_mass else None
    )


def timestep(world: nimble.simulation.World, state: torch.Tensor, 
    action: torch.Tensor, mass: Optional[torch.Tensor] = None) -> torch.Tensor:
  """
  This does a forward pass on the `world` that gets passed in, storing information needed
  in order to do a backwards pass.
  """
  return TimestepLayer.apply(world, state, action, mass)  # type: ignore
