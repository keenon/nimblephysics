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
  for t in range(len(mocapState.markerErrorsAtTimesteps)):
    for markerName in mocapState.markerErrorsAtTimesteps[t]:
      sum += mocapState.markerErrorsAtTimesteps[t][markerName].dot(
          mocapState.markerErrorsAtTimesteps[t][markerName])

  height_error = 1.8 - nimble.get_height(osim.skeleton, originalPos, mocapState.bodyScales)
  sum += torch.square(height_error)
  for bodyName in mocapState.bodyScales:
    for i in range(3):
      sum += 0.5 * torch.square(mocapState.bodyScales[bodyName][i] - ratio)
  for markerName in mocapState.markerOffsets:
    sum += 0.5 * mocapState.markerOffsets[markerName].dot(mocapState.markerOffsets[markerName])

  return sum


mocap.setCustomLoss(customLoss)

mocap.evaluatePerformance(
    "./S01DN603.trc",
    "./Rajagopal_scaled.osim",
    "./S01DN603_ik.mot", 5)
