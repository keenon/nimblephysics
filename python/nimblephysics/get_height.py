import nimblephysics_libs._nimblephysics as nimble
import torch
from typing import Tuple, Callable, List, Optional, Dict
import numpy as np
import math


class GetHeightLayer(torch.autograd.Function):
  """
  This implements a single, differentiable timestep of DART as a PyTorch layer
  """

  @staticmethod
  def forward(ctx, skel, position, bodyNames, bodyScales):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    skel: nimble.dynamics.Skeleton
    position: np.ndarray,
    bodyNames: List[str]
    bodyScales: torch.Tensor
    -> torch.Tensor
    """

    originalScales = skel.getBodyScales()

    # Set body scales

    for i in range(len(bodyNames)):
      body = bodyNames[i]
      skel.getBodyNode(body).setScale(bodyScales.detach().numpy()[i, :])

    # Get height

    height = skel.getHeight(position)
    ctx.heightGrad = skel.getGradientOfHeightWrtBodyScales(position)
    ctx.skel = skel
    ctx.bodyNames = bodyNames

    # Reset and return

    skel.setBodyScales(originalScales)
    return torch.tensor([height])

  @staticmethod
  def backward(ctx, grad_height):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    heightGrad: np.ndarray = ctx.heightGrad
    skel: nimble.dynamics.Skeleton = ctx.skel
    bodyNames: List[str] = ctx.bodyNames

    lossWrtHeight: float = grad_height.numpy()[0]
    lossWrtBodyScales: torch.Tensor = torch.zeros((len(bodyNames), 3), dtype=torch.float64)

    for i in range(len(bodyNames)):
      body = bodyNames[i]
      bodyNode: nimble.dynamics.BodyNode = skel.getBodyNode(body)
      index = bodyNode.getIndexInSkeleton()
      lossWrtBodyScales[i, :] = torch.from_numpy(heightGrad[index*3:(index+1)*3] * lossWrtHeight)

    return (
        None,
        None,
        None,
        lossWrtBodyScales
    )


def get_height(skel: nimble.dynamics.Skeleton, position: np.ndarray,
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

  return GetHeightLayer.apply(skel, position, bodyNames, bodyScalesTensor)  # type: ignore
