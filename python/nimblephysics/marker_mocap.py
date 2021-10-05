import torch
import nimblephysics_libs._nimblephysics as nimble
from typing import List, Dict, Callable, Tuple
import numpy as np
from .loader import absPath
from .gui_server import NimbleGUI
import random


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
  skelOriginalPose: np.ndarray
  markersMap: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]]

  def __init__(self, skel: nimble.dynamics.Skeleton,
               markersMap: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]]) -> None:
    self.skel = skel
    self.skelOriginalPose = skel.getPositions()
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

  def debugMotionToGUI(self,
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

    handScaledGoldIKMotAbs: str = absPath(handScaledGoldIKMot)
    print("Loading "+handScaledGoldIKMotAbs)
    mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(
        scaledOsim.skeleton,
        handScaledGoldIKMotAbs)

    print('Num bodies: '+str(scaledOsim.skeleton.getNumBodyNodes()))
    for i in range(scaledOsim.skeleton.getNumBodyNodes()):
      print('BodyNode '+str(i)+': '+scaledOsim.skeleton.getBodyNode(i).getName())

    print('Num joints: '+str(scaledOsim.skeleton.getNumJoints()))
    for i in range(scaledOsim.skeleton.getNumJoints()):
      print('Joint '+str(i)+': '+scaledOsim.skeleton.getJoint(i).getName()+' - '+scaledOsim.skeleton.getJoint(i).getType())

    scaledOsim.skeleton.setPositions(mot.poses[:, 0])
    print('Root joint offset:')
    print(scaledOsim.skeleton.getJoint(0).getRelativeTransform().matrix())

    markerObservations: List[Dict[str, np.ndarray]] = []
    markerObservations.append(markerTrajectories.markerTimesteps[0])
    print("Marker observations:")
    print(markerObservations[0])

    print("Original markers:")
    print(convertedMarkers)

    world: nimble.simulation.World = nimble.simulation.World()
    world.addSkeleton(scaledOsim.skeleton)
    gui: NimbleGUI = NimbleGUI(world)
    for markerName in markerTrajectories.markerLines:
      gui.nativeAPI().createLine(markerName, markerTrajectories.markerLines[markerName], [1, 0, 0])
    gui.loopPosMatrix(mot.poses)
    gui.serve(8080)
    gui.blockWhileServing()

  def evaluatePerformance(self,
                          markerTrcPath: str,
                          handScaledGoldBodyOsim: str,
                          handScaledGoldIKMot: str,
                          numStepsToFit: int = 3,
                          debugToGUI: bool = True) -> nimble.biomechanics.MarkerFitResult:
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

    handScaledGoldIKMotAbs: str = absPath(handScaledGoldIKMot)
    print("Loading "+handScaledGoldIKMotAbs)
    mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(
        scaledOsim.skeleton,
        handScaledGoldIKMotAbs)

    activeMarkers = []
    trimmedConvertedMarkers = {}
    for markerName in scaledOsim.markersMap:
      if '_' not in markerName and markerName.isupper():
        activeMarkers.append(markerName)
        trimmedConvertedMarkers[markerName] = convertedMarkers[markerName]

    originalIK: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
        scaledOsim.skeleton, scaledOsim.markersMap, mot.poses, markerTrajectories.markerTimesteps, activeMarkers)
    print('Original IK:')
    originalIK.printReport(limitTimesteps=10)

    randomIndices = [i for i in range(len(markerTrajectories.markerTimesteps))]
    random.shuffle(randomIndices)
    chosenIndices = randomIndices[:numStepsToFit]

    goldPoses = mot.poses[:, chosenIndices]
    print("Picking a random subset of the marker data")
    markerObservations: List[Dict[str, np.ndarray]
                             ] = [markerTrajectories.markerTimesteps[i] for i in chosenIndices]

    self.fitter = nimble.biomechanics.MarkerFitter(
        self.skel, trimmedConvertedMarkers)
    self.fitter.setInitialIKSatisfactoryLoss(0.05)
    self.fitter.setInitialIKMaxRestarts(50)
    self.fitter.setIterationLimit(100)

    print("Optimize the fit")
    result: nimble.biomechanics.MarkerFitResult = self.fitter.optimize(markerObservations)
    self.skel.setGroupScales(result.groupScales)
    bodyScales: np.ndarray = self.skel.getBodyScales()

    print("Result scales: " + str(bodyScales))

    groupScaleError: np.ndarray = bodyScales - config.bodyScales
    groupScaleCols: np.ndarray = np.zeros((groupScaleError.shape[0], 4))
    groupScaleCols[:, 0] = config.bodyScales
    groupScaleCols[:, 1] = bodyScales
    groupScaleCols[:, 2] = groupScaleError
    groupScaleCols[:, 3] = groupScaleError / config.bodyScales
    print("gold scales - result scales - error - error %")
    print(groupScaleCols)

    resultIK: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
        self.skel, trimmedConvertedMarkers, result.posesMatrix, markerObservations, activeMarkers)
    print('Fine tuned IK:')
    resultIK.printReport(limitTimesteps=10)

    if debugToGUI:
      world: nimble.simulation.World = nimble.simulation.World()
      gui: NimbleGUI = NimbleGUI(world)

      def renderTimestep(timestep):
        gui.nativeAPI().setAutoflush(False)
        # Render our guessed position
        self.skel.setPositions(result.poses[timestep])
        gui.nativeAPI().renderSkeleton(self.skel, 'result')

        # Calculate where we think markers are
        observedMarkers: Dict[str, np.ndarray] = {}
        for markerName in trimmedConvertedMarkers:
          markerBody = trimmedConvertedMarkers[markerName][0]
          originalMarkerOffset = trimmedConvertedMarkers[markerName][1]
          markerError = result.markerErrors[markerName]
          observedMarkers[markerName] = nimble.math.transformBy(
              markerBody.getWorldTransform(), originalMarkerOffset + markerError)

        # Render the gold position
        scaledOsim.skeleton.setPositions(goldPoses[:, timestep])
        gui.nativeAPI().renderSkeleton(scaledOsim.skeleton, 'gold', [0, 1, 0])

        # Render compared marker positions
        goldMarkers: Dict[str, np.ndarray] = scaledOsim.skeleton.getMarkerMapWorldPositions(
            scaledOsim.markersMap)
        realMarkers: Dict[str, np.ndarray] = markerTrajectories.markerTimesteps[timestep]
        for markerName in activeMarkers:
          points = [observedMarkers[markerName], goldMarkers[markerName]]
          gui.nativeAPI().createLine(markerName, points, [1, 0, 0])
          gui.nativeAPI().createBox(
              markerName + "_found", [0.005, 0.005, 0.005],
              observedMarkers[markerName],
              [0, 0, 0],
              [1, 0, 0])
          gui.nativeAPI().createBox(
              markerName + "_gold", [0.01, 0.01, 0.01],
              goldMarkers[markerName],
              [0, 0, 0],
              [0, 1, 0])
          gui.nativeAPI().createBox(
              markerName + "_real", [0.01, 0.01, 0.01],
              observedMarkers[markerName],
              [0, 0, 0],
              [1, 1, 0])
        gui.nativeAPI().setAutoflush(True)
        gui.nativeAPI().flush()

      cursor = 0
      renderTimestep(cursor)

      def keyListener(key: str):
        nonlocal cursor
        if key == " ":
          cursor += 1
          if cursor >= len(result.poses):
            cursor = 0
          renderTimestep(cursor)

      gui.nativeAPI().registerKeydownListener(keyListener)

      gui.serve(8080)
      gui.blockWhileServing()

    return result
