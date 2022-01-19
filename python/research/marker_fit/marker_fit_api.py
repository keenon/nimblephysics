import nimblephysics as nimble
from nimblephysics import absPath
import numpy as np
from typing import List, Dict, Tuple
import torch

height_m = 1.8

osim: nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
skel: nimble.dynamics.Skeleton = osim.skeleton
skel.autogroupSymmetricSuffixes()  # Make scaling symmetric
skel.setScaleGroupUniformScaling(skel.getBodyNode(
    "hand_r"))  # Use uniform scaling for the hands
print('Initial height: '+str(skel.getHeight(skel.getPositions())))
ratio = height_m / skel.getHeight(skel.getPositions())
skel.setBodyScales(skel.getBodyScales() * ratio)
print('Scaled height: '+str(skel.getHeight(skel.getPositions())))

anthro: nimble.biomechanics.Anthropometrics = nimble.models.RajagopalANSURModel().condition({
    'Heightin': height_m * 39.37 * 0.001,
    'Weightlbs': 150 * 0.001,
    'Age': 29 * 0.001})

print('skel DOFs: '+str(skel.getNumDofs()))
print('initial log PDF: '+str(anthro.getLogPDF(skel)))

#scaledModel: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(absPath("./Rajagopal_scaled.osim"))
scaledModel: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
    absPath("./sprinter_scaled.osim"))
scaledHeight = scaledModel.skeleton.getHeight(
    scaledModel.skeleton.getPositions())
print('scaled model height: '+str(scaledHeight))

# markerSet: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(absPath("./Rajagopal2015_passiveCal_hipAbdMoved.osim"))
markerSet: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
    absPath("./sprinter.osim"))
convertedMarkers: Dict[str, Tuple[nimble.dynamics.BodyNode, np.ndarray]
                       ] = skel.convertMarkerMap(markerSet.markersMap)

mocap: nimble.MarkerMocap = nimble.MarkerMocap(skel, convertedMarkers)
mocap.fitter.setTriadsToTracking()
mocap.setAnthropometricPrior(absPath("./ANSUR_metrics.xml"), absPath("./ANSUR_II_MALE_Public.csv"), {
    'Weightlbs': 150,
    'Heightin': height_m * 39.37
})

originalMarkerDistances: Dict[str, float] = {}
for markerName in convertedMarkers:
    originalMarkerDistances[markerName] = convertedMarkers[markerName][0].getDistToClosestVerticesToMarker(
        convertedMarkers[markerName][1])

originalPos = skel.getPositions()


def customLoss(mocapState: nimble.MarkerMocapOptimizationState):
    sum = torch.zeros(1)

    for t in range(mocapState.numTimesteps):
        for markerName in mocapState.markerErrorsAtTimesteps[t]:
            sum += mocapState.markerErrorsAtTimesteps[t][markerName].dot(
                mocapState.markerErrorsAtTimesteps[t][markerName])
        for jointName in mocapState.jointErrorsAtTimesteps[t]:
            sum += mocapState.jointErrorsAtTimesteps[t][jointName].dot(
                mocapState.jointErrorsAtTimesteps[t][jointName])

    """
  # Have a strong preference that the torso+head+pelvis doesn't scale too much to get the required height
  for i in range(3):
    # sum += (1.0 * numMarkerObservations) * torch.square(mocapState.bodyScales["torso"][i] - ratio)
    sum += (0.1 * numMarkerObservations) * torch.square(
        mocapState.bodyScales["pelvis"][i] - ratio)

  # Have a light preference for everything being scaled
  for bodyName in mocapState.bodyScales:
    for i in range(3):
      sum += (0.01 * numMarkerObservations) * torch.square(
          mocapState.bodyScales[bodyName][i] - ratio)

  # Have a light preference for anatomical marker offsets being 0
  for markerName in mocapState.markerOffsets:
    if not mocap.fitter.getMarkerIsTracking(markerName):
      sum += (0.01 * numMarkerObservations) * mocapState.markerOffsets[markerName].dot(
          mocapState.markerOffsets[markerName])
  """

    return sum


def markerDistanceConstraint(mocapState: nimble.MarkerMocapOptimizationState):
    sum = torch.zeros(1)

    # Try to keep markers close to their original distances
    for markerName in mocapState.markerOffsets:
        error_from_original = originalMarkerDistances[markerName] - nimble.get_marker_dist_to_nearest_vertex(
            convertedMarkers[markerName][0],
            mocapState.markerOffsets[markerName],
            mocapState.bodyScales[convertedMarkers[markerName][0].getName()])
        sum += torch.square(error_from_original)

    return sum


def markerZeroConstraint(mocapState: nimble.MarkerMocapOptimizationState):
    sum = torch.zeros(1)
    for markerName in mocapState.markerOffsets:
        if not mocap.fitter.getMarkerIsTracking(markerName):
            sum += mocapState.markerOffsets[markerName].dot(
                mocapState.markerOffsets[markerName])
    return sum


def heightConstraint(mocapState: nimble.MarkerMocapOptimizationState):
    height_error = scaledHeight - \
        nimble.get_height(osim.skeleton, originalPos, mocapState.bodyScales)
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


# mocap.setCustomLoss(customLoss)
mocap.addZeroConstraint("height", heightConstraint)
# mocap.addZeroConstraint("ground", groundConstraint)
# mocap.addZeroConstraint("marker", markerDistanceConstraint)
# mocap.addZeroConstraint("marker_zero", markerZeroConstraint)

mocap.evaluatePerformance(
    "./S01DN603.trc",
    "./Rajagopal_scaled.osim",
    "./S01DN603_ik.mot", 10)

"""
mocap.evaluatePerformance(
    "./run0500cms.trc",
    "./sprinter_scaled.osim",
    "./run0500cms.mot", 10)
"""
