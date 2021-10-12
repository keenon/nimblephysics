import torch
import nimblephysics_libs._nimblephysics as nimble
from typing import List, Dict, Callable, Tuple
import numpy as np
from .loader import absPath
from .gui_server import NimbleGUI
import random
import traceback


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

    for i in range(len(rawState.bodyNames)):
      self.bodyScales[rawState.bodyNames[i]] = torch.tensor(
          np.copy(rawState.bodyScales[:, i]), requires_grad=True)

    for i in range(len(rawState.markerOrder)):
      self.markerOffsets[rawState.markerOrder[i]] = torch.tensor(
          np.copy(rawState.markerOffsets[:, i]), requires_grad=True)

    for t in range(len(rawState.markerErrorsAtTimesteps)):
      markerErrors: Dict[str, torch.Tensor] = {}
      for i in range(len(rawState.markerOrder)):
        markerErrors[rawState.markerOrder[i]] = torch.tensor(
            np.copy(rawState.markerErrorsAtTimesteps[t*3:(t+1)*3, i]),
            requires_grad=True)
      self.markerErrorsAtTimesteps.append(markerErrors)

    for t in range(rawState.posesAtTimesteps.shape[1]):
      self.posesAtTimesteps.append(torch.tensor(
          np.copy(rawState.posesAtTimesteps[:, t]), requires_grad=True))

  def fillGradients(self, finalLoss: torch.Tensor) -> None:
    finalLoss.backward()

    bodyScalesGrad: np.ndarray = np.zeros_like(self.rawState.bodyScales)
    for i in range(len(self.rawState.bodyNames)):
      bodyName = self.rawState.bodyNames[i]
      if self.bodyScales[bodyName].grad is not None:
        bodyScalesGrad[:, i] = self.bodyScales[bodyName].grad.numpy()
    self.rawState.bodyScalesGrad = bodyScalesGrad

    markerOffsetsGrad: np.ndarray = np.zeros_like(self.rawState.markerOffsets)
    for i in range(len(self.rawState.markerOrder)):
      markerName = self.rawState.markerOrder[i]
      if self.markerOffsets[markerName].grad is not None:
        markerOffsetsGrad[:, i] = self.markerOffsets[markerName].grad.numpy()
    self.rawState.markerOffsetsGrad = markerOffsetsGrad

    markerErrorsGrad: np.ndarray = np.zeros_like(self.rawState.markerErrorsAtTimesteps)
    for t in range(len(self.markerErrorsAtTimesteps)):
      for i in range(len(self.rawState.markerOrder)):
        markerName = self.rawState.markerOrder[i]
        if self.markerErrorsAtTimesteps[t][markerName].grad is not None:
          markerErrorsGrad[t*3:(t+1)*3,
                           i] = self.markerErrorsAtTimesteps[t][markerName].grad.numpy()
    self.rawState.markerErrorsAtTimestepsGrad = markerErrorsGrad

    posesAtTimestepsGrad: np.ndarray = np.zeros_like(self.rawState.posesAtTimesteps)
    for t in range(len(self.posesAtTimesteps)):
      if self.posesAtTimesteps[t].grad is not None:
        posesAtTimestepsGrad[:, t] = self.posesAtTimesteps[t].grad.numpy()
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
    self.zeroConstraints: Dict[str,
                               Callable
                               [[nimble.biomechanics.MarkerFitterState],
                                float]] = {}

  def setCustomLoss(self, lossFn: Callable[[MarkerMocapOptimizationState], torch.Tensor]) -> None:
    def wrappedLoss(rawState: nimble.biomechanics.MarkerFitterState) -> float:
      try:
        wrappedState = MarkerMocapOptimizationState(rawState)
        loss: torch.Tensor = lossFn(wrappedState)
        wrappedState.fillGradients(loss)
        return loss.item()
      except Exception as e:
        print(traceback.format_exc())
        return 0
    self.wrappedLoss = wrappedLoss
    self.fitter.setCustomLossAndGrad(self.wrappedLoss)

  def addZeroConstraint(
          self, name: str, lossFn: Callable[[MarkerMocapOptimizationState],
                                            torch.Tensor]) -> None:
    def wrappedLoss(rawState: nimble.biomechanics.MarkerFitterState):
      try:
        wrappedState = MarkerMocapOptimizationState(rawState)
        loss: torch.Tensor = lossFn(wrappedState)
        wrappedState.fillGradients(loss)
        return loss.item()
      except Exception as e:
        print(e)
    self.zeroConstraints[name] = wrappedLoss
    self.fitter.addZeroConstraint(name, wrappedLoss)

  def removeZeroConstraint(self, name: str) -> None:
    self.zeroConstraints.pop(name, None)
    self.fitter.removeZeroConstraint(name)

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
    print("Creating OpenSimFile")
    standardOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimFile(
        self.skel, self.markersMap)
    print("Computing scale and marker offsets")
    config: nimble.biomechanics.OpenSimScaleAndMarkerOffsets = nimble.biomechanics.OpenSimParser.getScaleAndMarkerOffsets(
        standardOsim, scaledOsim)

    handScaledGoldIKMotAbs: str = absPath(handScaledGoldIKMot)
    print("Loading "+handScaledGoldIKMotAbs)
    mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(
        scaledOsim.skeleton,
        handScaledGoldIKMotAbs)

    originalIK: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
        scaledOsim.skeleton, scaledOsim.markersMap, mot.poses, markerTrajectories.markerTimesteps)
    print('Original IK:')
    originalIK.printReport(limitTimesteps=10)

    print("Picking a random subset of the marker data")
    randomIndices = [i for i in range(len(markerTrajectories.markerTimesteps))]
    random.seed(10)
    random.shuffle(randomIndices)
    chosenIndices = randomIndices[: numStepsToFit]

    print('Chosen Indices:')
    print(chosenIndices)

    goldPoses = mot.poses[:, chosenIndices]
    markerObservations: List[Dict[str, np.ndarray]
                             ] = [markerTrajectories.markerTimesteps[i] for i in chosenIndices]

    print("Optimize the fit")
    result: nimble.biomechanics.MarkerFitResult = self.fitter.optimize(markerObservations)
    self.skel.setGroupScales(result.groupScales)
    bodyScales: np.ndarray = self.skel.getBodyScales()

    finalHeightM = self.skel.getHeight(self.skelOriginalPose)
    print('Final height (meters): '+str(finalHeightM))

    fitMarkers: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]] = {}
    for markerName in self.markersMap:
      fitMarkers[markerName] = (
          self.markersMap[markerName][0],
          self.markersMap[markerName][1] + result.markerErrors[markerName])

    print("Result scales: " + str(bodyScales))

    groupScaleError: np.ndarray = bodyScales - config.bodyScales
    groupScaleCols: np.ndarray = np.zeros((groupScaleError.shape[0], 4))
    groupScaleCols[:, 0] = config.bodyScales
    groupScaleCols[:, 1] = bodyScales
    groupScaleCols[:, 2] = groupScaleError
    groupScaleCols[:, 3] = groupScaleError / config.bodyScales
    print("gold scales - result scales - error - error %")
    print(groupScaleCols)

    posesSubset = np.zeros((mot.poses.shape[0], len(chosenIndices)))
    for i in range(len(chosenIndices)):
      posesSubset[:, i] = mot.poses[:, chosenIndices[i]]
    originalIKSubset: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
        scaledOsim.skeleton, scaledOsim.markersMap, posesSubset, markerObservations)
    print('Original IK:')
    originalIKSubset.printReport(limitTimesteps=10)

    resultIK: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
        self.skel, fitMarkers, result.posesMatrix, markerObservations)
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
        observedMarkers: Dict[str, np.ndarray] = self.skel.getMarkerMapWorldPositions(
            fitMarkers)

        # Render the gold position
        scaledOsim.skeleton.setPositions(goldPoses[:, timestep])
        gui.nativeAPI().renderSkeleton(scaledOsim.skeleton, 'gold', [0, 1, 0])

        # Render compared marker positions
        goldMarkers: Dict[str, np.ndarray] = scaledOsim.skeleton.getMarkerMapWorldPositions(
            scaledOsim.markersMap)
        realMarkers: Dict[str, np.ndarray] = markerObservations[timestep]
        for markerName in self.markersMap:
          # Not all markers are observed at all timesteps
          if markerName in realMarkers:
            # Make a triangle between the 3 points
            points = [
                observedMarkers[markerName],
                goldMarkers[markerName],
                realMarkers[markerName],
                observedMarkers[markerName]]
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
                realMarkers[markerName],
                [0, 0, 0],
                [1, 1, 0])
          else:
            gui.nativeAPI().deleteObject(markerName)
            gui.nativeAPI().deleteObject(markerName+"_found")
            gui.nativeAPI().deleteObject(markerName+"_gold")
            gui.nativeAPI().deleteObject(markerName+"_real")
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

      gui.nativeAPI().renderBasis()

      gui.serve(8080)
      gui.blockWhileServing()

    return result
