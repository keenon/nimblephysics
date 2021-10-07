import nimblephysics as nimble
from nimblephysics import absPath
import numpy as np
from typing import List
import torch

height_m = 1.8

osim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
    absPath("./Rajagopal_scaled.osim"))  # nimble.models.RajagopalHumanBodyModel()
skel: nimble.dynamics.Skeleton = osim.skeleton
skel.autogroupSymmetricSuffixes()  # Make scaling symmetric
skel.setScaleGroupUniformScaling(skel.getBodyNode("hand_r"))  # Use uniform scaling for the hands
print('Initial height: '+str(skel.getHeight(skel.getPositions())))
ratio = height_m / skel.getHeight(skel.getPositions())
skel.setBodyScales(skel.getBodyScales() * ratio)
print('Scaled height: '+str(skel.getHeight(skel.getPositions())))
mocap: nimble.MarkerMocap = nimble.MarkerMocap(osim.skeleton, osim.markersMap)

originalPos = osim.skeleton.getPositions()


def customLoss(mocapState: nimble.MarkerMocapOptimizationState):
  sum = torch.zeros(1)

  numMarkers = 0
  for t in range(len(mocapState.markerErrorsAtTimesteps)):
    for markerName in mocapState.markerErrorsAtTimesteps[t]:
      numMarkers += 1
      sum += mocapState.markerErrorsAtTimesteps[t][markerName].dot(
          mocapState.markerErrorsAtTimesteps[t][markerName])

  # Have a strong preferenc that the torso+head doesn't scale too much to get the required height
  for i in range(3):
    sum += (150.0 / numMarkers) * torch.square(mocapState.bodyScales["torso"][i] - ratio)

  # Have a light preference for everything being scaled
  for bodyName in mocapState.bodyScales:
    for i in range(3):
      sum += (1.0 / numMarkers) * torch.square(mocapState.bodyScales[bodyName][i] - ratio)

  # Have a light preference for marker offsets being 0
  for markerName in mocapState.markerOffsets:
    sum += (1.0 / numMarkers) * mocapState.markerOffsets[markerName].dot(
        mocapState.markerOffsets[markerName])

  return sum


def heightConstraint(mocapState: nimble.MarkerMocapOptimizationState):
  height_error = 1.8 - nimble.get_height(osim.skeleton, originalPos, mocapState.bodyScales)
  return torch.square(height_error)


def groundConstraint(mocapState: nimble.MarkerMocapOptimizationState):
  sum = torch.zeros(1)
  for t in range(len(mocapState.posesAtTimesteps)):
    ground_error = nimble.get_lowest_point(
        osim.skeleton, mocapState.posesAtTimesteps[t],
        mocapState.bodyScales)
    # just make sure we never penetrate the ground
    ground_error = ground_error.clamp(max=0.0)
    sum += torch.square(ground_error)
  return sum


mocap.setCustomLoss(customLoss)
mocap.addZeroConstraint("height", heightConstraint)
mocap.addZeroConstraint("ground", groundConstraint)

mocap.evaluatePerformance(
    "./S01DN603.trc",
    "./Rajagopal_scaled.osim",
    "./S01DN603_ik.mot", 5)
