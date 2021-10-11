import nimblephysics_libs._nimblephysics as nimble
import torch
from typing import Tuple, Callable, List, Optional, Dict
import numpy as np
import math


class GetMarkerDistLayer(torch.autograd.Function):
  """
  This implements a single, differentiable timestep of DART as a PyTorch layer
  """

  @staticmethod
  def forward(ctx, bodyNode, markerOffset, bodyScale):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    bodyNode: nimble.dynamics.BodyNode
    markerOffset: torch.Tensor,
    bodyScale: torch.Tensor
    -> torch.Tensor
    """

    originalScale = bodyNode.getScale()
    bodyNode.setScale(bodyScale.detach().numpy())

    # Get distance

    node: nimble.dynamics.BodyNode = bodyNode
    dist = node.getDistToClosestVerticesToMarker(markerOffset.detach().numpy())
    ctx.scaleGrad = node.getGradientOfDistToClosestVerticesToMarkerWrtBodyScale(
        markerOffset.detach().numpy())
    ctx.offsetGrad = node.getGradientOfDistToClosestVerticesToMarkerWrtMarker(
        markerOffset.detach().numpy())
    ctx.bodyNode = bodyNode

    # Reset and return

    bodyNode.setScale(originalScale)

    return torch.tensor([dist])

  @staticmethod
  def backward(ctx, grad_dist):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    scaleGrad: np.ndarray = ctx.scaleGrad
    offsetGrad: np.ndarray = ctx.offsetGrad

    lossWrtDist: float = grad_dist.numpy()[0]

    lossWrtBodyScale: torch.Tensor = torch.from_numpy(scaleGrad * lossWrtDist)
    lossWrtMarkerOffset: torch.Tensor = torch.from_numpy(offsetGrad * lossWrtDist)

    return (
        None,
        lossWrtBodyScale,
        lossWrtMarkerOffset
    )


def get_marker_dist_to_nearest_vertex(
        bodyNode: nimble.dynamics.BodyNode, markerOffset: torch.Tensor, bodyScale: torch.Tensor) -> torch.Tensor:
  """
  This gets the distance between the marker and 
  """
  return GetMarkerDistLayer.apply(bodyNode, markerOffset, bodyScale)  # type: ignore
