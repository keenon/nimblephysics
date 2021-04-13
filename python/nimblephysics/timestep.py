import nimblephysics_libs._nimblephysics as nimble
import torch
from typing import Tuple, Callable, List
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
  def forward(ctx, world, pos, vel, torque, snapshot_pointer):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    world: nimble.simulation.World
    pos: torch.Tensor
    vel: torch.Tensor
    torque: torch.Tensor
    snapshot_pointer: BackpropSnapshotPointer
    -> [torch.Tensor, torch.Tensor]
    """

    world.setPositions(pos.detach().numpy())
    world.setVelocities(vel.detach().numpy())
    world.setControlForces(torque.detach().numpy())
    backprop_snapshot: nimble.neural.BackpropSnapshot = nimble.neural.forwardPass(world)
    ctx.backprop_snapshot = backprop_snapshot
    ctx.world = world

    if snapshot_pointer is not None:
      snapshot_pointer.backprop_snapshot = backprop_snapshot

    return (torch.tensor(world.getPositions()), torch.tensor(world.getVelocities()))

  @staticmethod
  def backward(ctx, grad_pos, grad_vel):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    backprop_snapshot: nimble.neural.BackpropSnapshot = ctx.backprop_snapshot
    world: nimble.simulation.World = ctx.world

    # Set up the gradients for backprop
    nextTimestepGrad: nimble.neural.LossGradient = nimble.neural.LossGradient()
    nextTimestepGrad.lossWrtPosition = grad_pos.detach().numpy()
    nextTimestepGrad.lossWrtVelocity = grad_vel.detach().numpy()

    # Run backprop, writing values into thisTimestepGrad
    thisTimestepGrad: nimble.neural.LossGradient = nimble.neural.LossGradient()
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


def timestep(world: nimble.simulation.World, pos: torch.Tensor, vel: torch.Tensor,
               torque: torch.Tensor, pointer: BackpropSnapshotPointer = None) -> Tuple[torch.Tensor,
                                                                                       torch.Tensor]:
  """
  This does a forward pass on the `world` that gets passed in, storing information needed
  in order to do a backwards pass.
  """
  return TimestepLayer.apply(world, pos, vel, torque, pointer)  # type: ignore
