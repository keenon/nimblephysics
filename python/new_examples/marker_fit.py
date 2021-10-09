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
for i in range(10):
  pose = skel.getRandomPose()
  skel.setPositions(pose)
  markers = skel.getMarkerMapWorldPositions(osim.markersMap)

  gold_poses.append(pose)
  gold_markers.append(markers)

skel.setPositions(np.zeros(skel.getNumDofs()))
skel.setGroupScales(original_scales)

fitter: nimble.biomechanics.MarkerFitter = nimble.biomechanics.MarkerFitter(
    skel, osim.markersMap)
result: nimble.biomechanics.MarkerFitResult = fitter.optimize(gold_markers)

print('Gold group scales:')
print(gold_scales)
print('Found group scales:')
print(result.groupScales)
print('Diff:')
print(gold_scales - result.groupScales)
