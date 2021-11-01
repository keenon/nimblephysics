import nimblephysics_libs._nimblephysics as nimble
import torch
from typing import Tuple, Callable, List, Optional, Dict
import numpy as np
import math


class GetAnthropometricLogPDF(torch.autograd.Function):
  """
  This implements a single, differentiable call to get the logPDF of an Anthropometric distribution as a PyTorch layer
  """

  @staticmethod
  def forward(ctx, skel, anthro, bodyNames, bodyScales):
    """
    We can't put type annotations on this declaration, because the supertype
    doesn't have any type annotations and otherwise mypy will complain, so here
    are the types:

    skel: nimble.dynamics.Skeleton
    anthro: nimble.biomechanics.Anthropometrics
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

    pdf = anthro.getLogPDF(skel, False)
    ctx.pdfGrad = anthro.getGradientOfLogPDFWrtBodyScales(skel)
    ctx.skel = skel
    ctx.bodyNames = bodyNames

    # Reset and return

    skel.setBodyScales(originalScales)
    return torch.tensor([pdf])

  @staticmethod
  def backward(ctx, grad_pdf):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    pdfGrad: np.ndarray = ctx.pdfGrad
    skel: nimble.dynamics.Skeleton = ctx.skel
    bodyNames: List[str] = ctx.bodyNames

    lossWrtPDF: float = grad_pdf.numpy()[0]
    lossWrtBodyScales: torch.Tensor = torch.zeros((len(bodyNames), 3), dtype=torch.float64)

    for i in range(len(bodyNames)):
      body = bodyNames[i]
      bodyNode: nimble.dynamics.BodyNode = skel.getBodyNode(body)
      index = bodyNode.getIndexInSkeleton()
      lossWrtBodyScales[i, :] = torch.from_numpy(pdfGrad[index*3:(index+1)*3] * lossWrtPDF)

    return (
        None,
        None,
        None,
        lossWrtBodyScales
    )


def get_anthropometric_log_pdf(
        skel: nimble.dynamics.Skeleton, anthro: nimble.biomechanics.Anthropometrics,
        bodyScales: Dict[str, torch.Tensor]) -> torch.Tensor:
  bodyNames: List[str] = []
  bodyScalesArr: List[torch.Tensor] = []
  for name in bodyScales:
    bodyNames.append(name)
    bodyScalesArr.append(torch.unsqueeze(bodyScales[name], 0))
  bodyScalesTensor: torch.Tensor = torch.cat(bodyScalesArr, dim=0)

  return GetAnthropometricLogPDF.apply(skel, anthro, bodyNames, bodyScalesTensor)
