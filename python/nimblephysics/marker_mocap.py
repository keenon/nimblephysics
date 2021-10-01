import torch
import nimblephysics_libs._nimblephysics as nimble
from typing import List, Dict, Callable, Tuple
import numpy as np
from .loader import absPath


class MarkerMocapOptimizationState:
  """
  This wraps a MarkerFitterState, but using PyTorch Tensors, so we can autograd any arbitrary user-supplied loss function
  """
  rawState: nimble.biomechanics.MarkerFitterState
  bodyScales: Dict[str, torch.Tensor]
  markerOffsets: Dict[str, torch.Tensor]
  markerErrorsAtTimesteps: List[Dict[str, torch.Tensor]]
  posesAtTimesteps: List[torch.Tensor]

  def __init__(self, rawState: nimble.biomechanics.MarkerFitterState) -> None:
    self.rawState: nimble.biomechanics.MarkerFitterState = rawState
    self.bodyScales: Dict[str, torch.Tensor] = {}
    self.markerOffsets: Dict[str, torch.Tensor] = {}
    self.markerErrorsAtTimesteps: List[Dict[str, torch.Tensor]] = []
    self.posesAtTimesteps: List[torch.Tensor] = []

    for bodyName in rawState.bodyScales:
      self.bodyScales[bodyName] = torch.tensor(rawState.bodyScales[bodyName], requires_grad=True)

    for markerName in rawState.markerOffsets:
      self.markerOffsets[markerName] = torch.tensor(
          rawState.markerOffsets[markerName], requires_grad=True)

    for t in range(len(rawState.markerErrorsAtTimesteps)):
      markerErrors: Dict[str, torch.Tensor] = {}
      for markerName in rawState.markerErrorsAtTimesteps[t]:
        markerErrors[markerName] = torch.tensor(
            rawState.markerErrorsAtTimesteps[t][markerName],
            requires_grad=True)
      self.markerErrorsAtTimesteps.append(markerErrors)

    for t in range(len(rawState.posesAtTimesteps)):
      self.posesAtTimesteps.append(torch.tensor(rawState.posesAtTimesteps[t], requires_grad=True))

  def fillGradients(self, finalLoss: torch.Tensor) -> None:
    finalLoss.backward()

    bodyScalesGrad: Dict[str, np.ndarray] = {}
    for bodyName in self.bodyScales:
      if self.bodyScales[bodyName].grad is not None:
        bodyScalesGrad[bodyName] = self.bodyScales[bodyName].grad.numpy()
      else:
        bodyScalesGrad[bodyName] = np.zeros(3)
    self.rawState.bodyScalesGrad = bodyScalesGrad

    markerOffsetsGrad: Dict[str, np.ndarray] = {}
    for markerName in self.markerOffsets:
      if self.markerOffsets[markerName].grad is not None:
        markerOffsetsGrad[markerName] = self.markerOffsets[markerName].grad.numpy()
      else:
        markerOffsetsGrad[markerName] = np.zeros(3)
    self.rawState.markerOffsetsGrad = markerOffsetsGrad

    markerErrorsGrad: List[Dict[str, np.ndarray]] = []
    for t in range(len(self.markerErrorsAtTimesteps)):
      grad: Dict[str, torch.Tensor] = {}
      for markerName in self.markerErrorsAtTimesteps[t]:
        if self.markerErrorsAtTimesteps[t][markerName].grad is not None:
          grad[markerName] = self.markerErrorsAtTimesteps[t][markerName].grad.numpy()
        else:
          grad[markerName] = np.zeros(3)
      markerErrorsGrad.append(grad)
    self.rawState.markerErrorsAtTimestepsGrad = markerErrorsGrad

    posesAtTimestepsGrad: List[np.ndarray] = []
    for t in range(len(self.posesAtTimesteps)):
      if self.posesAtTimesteps[t].grad is not None:
        posesAtTimestepsGrad.append(self.posesAtTimesteps[t].grad.numpy())
      else:
        posesAtTimestepsGrad.append(np.zeros_like(self.posesAtTimesteps[t].detach().numpy()))
    self.rawState.posesAtTimestepsGrad = posesAtTimestepsGrad


class MarkerMocap:
  """
  This class encapsulates a lot of the tools in Nimble for loading and processing marker-based mocap data into a useful format.
  """
  fitter: nimble.biomechanics.MarkerFitter
  skel: nimble.dynamics.Skeleton
  markersMap: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]]

  def __init__(self, skel: nimble.dynamics.Skeleton,
               markersMap: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]]) -> None:
    self.skel = skel
    self.markersMap = markersMap
    self.fitter = nimble.biomechanics.MarkerFitter(
        skel, markersMap)
    self.fitter.setInitialIKSatisfactoryLoss(0.05)
    self.fitter.setInitialIKMaxRestarts(50)
    self.fitter.setIterationLimit(100)

  def setCustomLoss(self, lossFn: Callable[[MarkerMocapOptimizationState], torch.Tensor]) -> None:
    def wrappedLoss(rawState: nimble.biomechanics.MarkerFitterState):
      try:
        wrappedState = MarkerMocapOptimizationState(rawState)
        loss: torch.Tensor = lossFn(wrappedState)
        wrappedState.fillGradients(loss)
        return loss.item()
      except Exception as e:
        print(e)
    self.fitter.setCustomLossAndGrad(wrappedLoss)

  def evaluatePerformance(self,
                          markerTrcPath: str,
                          handScaledGoldBodyOsim: str,
                          handScaledGoldIKMot: str):
    """
    This compares the performance of the MarkerMocap system to a manual fit process, and prints a number of stats
    """
    markerTrcPathAbs: str = absPath(markerTrcPath)
    print("Loading "+markerTrcPathAbs)
    markerTrajectories: nimble.biomechanics.OpenSimTRC = nimble.biomechanics.OpenSimParser.loadTRC(
        markerTrcPathAbs)

    handScaledGoldBodyOsimAbs: str = absPath(handScaledGoldBodyOsim)
    print("Loading "+handScaledGoldBodyOsimAbs)
    scaledOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
        handScaledGoldBodyOsimAbs)
    print("Converting markers")
    convertedMarkers: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]
                           ] = self.skel.convertMarkerMap(scaledOsim.markersMap)
    print("Creating OpenSimFile")
    standardOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimFile(
        self.skel, convertedMarkers)
    print("Computing scale and marker offsets")
    config: nimble.biomechanics.OpenSimScaleAndMarkerOffsets = nimble.biomechanics.OpenSimParser.getScaleAndMarkerOffsets(
        standardOsim, scaledOsim)

    """
    handScaledGoldIKMotAbs: str = absPath(handScaledGoldIKMot)
    print("Loading "+handScaledGoldIKMotAbs)
    mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(
        scaledOsim.skeleton,
        handScaledGoldIKMotAbs)
    """

    print("Picking a random subset of the marker data")
    markerObservations: List[Dict[str, np.ndarray]] = nimble.biomechanics.MarkerFitter.pickSubset(
        markerTrajectories.markerTimesteps, 2)

    print("Optimize the fit")
    result: nimble.biomechanics.MarkerFitResult = self.fitter.optimize(markerObservations)
    self.skel.setGroupScales(result.groupScales)
    bodyScales: np.ndarray = self.skel.getBodyScales()

    print("Result scales: " + str(bodyScales))

    groupScaleError: np.ndarray = bodyScales - config.bodyScales
    groupScaleCols: np.ndarray = np.zeros(groupScaleError.shape[0], 4)
    groupScaleCols[:, 0] = config.bodyScales
    groupScaleCols[:, 1] = bodyScales
    groupScaleCols[:, 2] = groupScaleError
    groupScaleCols[:, 3] = groupScaleError / config.bodyScales
    print("gold scales - result scales - error - error %")
    print(groupScaleCols)

    pass
