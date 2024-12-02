import nimblephysics as nimble
from nimblephysics import NimbleGUI
from nimblephysics import absPath
from typing import Dict, Tuple, List
import numpy as np

# Load the marker trajectories
c3dFile: nimble.biomechanics.C3D = nimble.biomechanics.C3DLoader.loadC3D(
    absPath('./JA1Gait35.c3d'))

# Set the basic measurements
massKg = 68.0
heightM = 1.8

# Load the unscaled Osim file, which we can then scale and format
customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
    absPath('./sprinter.osim'))

fitter = nimble.biomechanics.MarkerFitter(
    customOsim.skeleton, customOsim.markersMap)
fitter.setTriadsToTracking()
fitter.setInitialIKSatisfactoryLoss(0.05)
fitter.setInitialIKMaxRestarts(50)
fitter.setIterationLimit(300)
# fitter.setInitialIKMaxRestarts(1)
# fitter.setIterationLimit(2)

# Create an anthropometric prior
anthropometrics: nimble.biomechanics.Anthropometrics = nimble.biomechanics.Anthropometrics.loadFromFile(
    absPath('./ANSUR_metrics.xml'))
cols = anthropometrics.getMetricNames()
cols.append('Heightin')
cols.append('Weightlbs')
gauss: nimble.math.MultivariateGaussian = nimble.math.MultivariateGaussian.loadFromCSV(
    absPath('./ANSUR_II_MALE_Public.csv'),
    cols,
    0.001)  # mm -> m
observedValues = {
    'Heightin': heightM * 39.37 * 0.001,
    'Weightlbs': massKg * 0.453 * 0.001,
}
gauss = gauss.condition(observedValues)
anthropometrics.setDistribution(gauss)
fitter.setAnthropometricPrior(anthropometrics, 0.1)

# Run the solver
newClip: List[bool] = [False for _ in range(len(c3dFile.markerTimesteps))]
result: nimble.biomechanics.MarkerInitialization = fitter.runKinematicsPipeline(
    c3dFile.markerTimesteps, newClip, nimble.biomechanics.InitialMarkerFitParams())
customOsim.skeleton.setGroupScales(result.groupScales)
fitMarkers: Dict[str, Tuple[nimble.dynamics.BodyNode,
                            np.ndarray]] = result.updatedMarkerMap
resultIK: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
    customOsim.skeleton, fitMarkers, result.poses, c3dFile.markerTimesteps)
print('automatically scaled average RMSE cm: ' +
      str(resultIK.averageRootMeanSquaredError), flush=True)
print('automatically scaled average max cm: ' +
      str(resultIK.averageMaxError), flush=True)

# Save a results file
fitter.saveTrajectoryAndMarkersToGUI(
    "./results.json", result, c3dFile.markerTimesteps, c3dFile)

# Debug our results to the GUI
world: nimble.simulation.World = nimble.simulation.World()
gui: NimbleGUI = NimbleGUI(world)
gui.serve(8080)
fitter.debugTrajectoryAndMarkersToGUI(
    gui.nativeAPI(), result, c3dFile.markerTimesteps, c3dFile)
gui.blockWhileServing()
