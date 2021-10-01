import nimblephysics as nimble
import numpy as np
from typing import List

# Create the world

world: nimble.simulation.World = nimble.simulation.World()
world.setGravity([0, -9.81, 0])
world.setTimeStep(0.01)

# Create the Rajagopal human body model (from the files shipped with Nimble, licensed under a separate MIT license)

osim: nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
skel: nimble.dynamics.Skeleton = osim.skeleton
world.addSkeleton(skel)

original_scales = skel.getGroupScales()

gold_scales = original_scales + ((np.random.rand(original_scales.shape[0]) - 0.5) * 0.2)
skel.setGroupScales(gold_scales)
gold_poses: List[np.ndarray] = []
gold_markers: List[np.ndarray] = []
for i in range(3):
  pose = skel.getRandomPose()
  skel.setPositions(pose)
  markers = skel.getMarkerMapWorldPositions(osim.markersMap)

  gold_poses.append(pose)
  gold_markers.append(markers)

skel.setPositions(np.zeros(skel.getNumDofs()))
skel.setGroupScales(original_scales)

def lossAndGrad(state: nimble.biomechanics.MarkerFitterState):
  loss = 0.0
  try:
    markerErrorsAtTimestepsGrad = []
    for t in range(len(state.markerErrorsAtTimesteps)):
      grad = {}
      for marker in state.markerErrorsAtTimesteps[t]:
        error = state.markerErrorsAtTimesteps[t][marker]
        loss += np.dot(error,error)
        grad[marker] = 2 * error
      markerErrorsAtTimestepsGrad.append(grad)
    state.markerErrorsAtTimestepsGrad = markerErrorsAtTimestepsGrad
  except Exception as e: print(e)
  return loss

fitter: nimble.biomechanics.MarkerFitter = nimble.biomechanics.MarkerFitter(
    skel, osim.markersMap)
fitter.setCustomLossAndGrad(lossAndGrad)
result: nimble.biomechanics.MarkerFitResult = fitter.optimize(gold_markers)

print('Gold group scales:')
print(gold_scales)
print('Found group scales:')
print(result.groupScales)
print('Diff:')
print(gold_scales - result.groupScales)
