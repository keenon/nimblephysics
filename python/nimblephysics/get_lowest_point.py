import nimblephysics_libs._nimblephysics as nimble
import torch
from typing import Tuple, Callable, List, Optional, Dict
import numpy as np
import math


class GetLowestPointLayer(torch.autograd.Function):
  """
  This implements a differentiable query for the "lowest point" (specified relative to an `up` vector) on a skeleton.
  """

  @staticmethod
  def forward(ctx, skel, position, bodyNames, bodyScales):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    skel: nimble.dynamics.Skeleton
    position: torch.Tensor,
    bodyNames: List[str]
    bodyScales: torch.Tensor
    -> torch.Tensor
    """

    originalScales = skel.getBodyScales()
    originalPosition = skel.getPositions()

    # Set positions

    skel.setPositions(position.detach().numpy())

    # Set body scales

    for i in range(len(bodyNames)):
      body = bodyNames[i]
      skel.getBodyNode(body).setScale(bodyScales.detach().numpy()[i, :])

    # Get lowest point

    lowestPoint = skel.getLowestPoint()
    ctx.gradWrtBodies = skel.getGradientOfLowestPointWrtBodyScales()
    ctx.gradWrtPos = skel.getGradientOfLowestPointWrtJoints()
    ctx.skel = skel
    ctx.bodyNames = bodyNames

    # Reset and return

    skel.setBodyScales(originalScales)
    skel.setPositions(originalPosition)
    return torch.tensor([lowestPoint])

  @staticmethod
  def backward(ctx, grad_lowest_point):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    gradWrtBodies: np.ndarray = ctx.gradWrtBodies
    gradWrtPos: np.ndarray = ctx.gradWrtPos
    skel: nimble.dynamics.Skeleton = ctx.skel
    bodyNames: List[str] = ctx.bodyNames

    lossWrtLowestPoint: float = grad_lowest_point.numpy()[0]
    lossWrtPos: torch.Tensor = torch.from_numpy(
        gradWrtPos * lossWrtLowestPoint)
    lossWrtBodyScales: torch.Tensor = torch.zeros((len(bodyNames), 3), dtype=torch.float64)

    for i in range(len(bodyNames)):
      body = bodyNames[i]
      bodyNode: nimble.dynamics.BodyNode = skel.getBodyNode(body)
      index = bodyNode.getIndexInSkeleton()
      lossWrtBodyScales[i, :] = torch.from_numpy(
          gradWrtBodies[index*3:(index+1)*3] * lossWrtLowestPoint)

    return (
        None,
        lossWrtPos,
        None,
        lossWrtBodyScales
    )


def get_lowest_point(skel: nimble.dynamics.Skeleton, position: torch.Tensor,
                     bodyScales: Dict[str, torch.Tensor]) -> torch.Tensor:
  """
  This does a forward pass on the `world` that gets passed in, storing information needed
  in order to do a backwards pass.
  """
  bodyNames: List[str] = []
  bodyScalesArr: List[torch.Tensor] = []
  for name in bodyScales:
    bodyNames.append(name)
    bodyScalesArr.append(torch.unsqueeze(bodyScales[name], 0))
  bodyScalesTensor: torch.Tensor = torch.cat(bodyScalesArr, dim=0)

  return GetLowestPointLayer.apply(skel, position, bodyNames, bodyScalesTensor)  # type: ignore
