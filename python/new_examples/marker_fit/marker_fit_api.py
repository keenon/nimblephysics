import nimblephysics as nimble
import numpy as np
from typing import List
import torch

osim: nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
skel: nimble.dynamics.Skeleton = osim.skeleton
skel.autogroupSymmetricSuffixes()  # Make scaling symmetric
mocap: nimble.MarkerMocap = nimble.MarkerMocap(osim.skeleton, osim.markersMap)


def customLoss(mocapState: nimble.MarkerMocapOptimizationState):
  sum = torch.zeros(1)
  for t in range(len(mocapState.markerErrorsAtTimesteps)):
    for markerName in mocapState.markerErrorsAtTimesteps[t]:
      sum += mocapState.markerErrorsAtTimesteps[t][markerName].dot(
          mocapState.markerErrorsAtTimesteps[t][markerName])
  return sum


mocap.setCustomLoss(customLoss)

mocap.evaluatePerformance(
    "./S01DN603.trc",
    "./Rajagopal_scaled.osim",
    "S01DN603_ik.mot")
