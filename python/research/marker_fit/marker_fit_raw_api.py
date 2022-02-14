import nimblephysics as nimble
from nimblephysics import NimbleGUI
from nimblephysics import absPath
from typing import Dict, Tuple
import numpy as np

# Load the marker trajectories
markerTrajectories: nimble.biomechanics.OpenSimTRC = nimble.biomechanics.OpenSimParser.loadTRC(
    absPath('./S01DN603.trc'))

# Get the manual baseline
goldOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
    absPath('./Rajagopal_scaled.osim'))
goldMot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(
    goldOsim.skeleton, absPath('./S01DN603_ik.mot'))
originalIK: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
    goldOsim.skeleton, goldOsim.markersMap, goldMot.poses, markerTrajectories.markerTimesteps)
print('manually scaled average RMSE cm: ' +
      str(originalIK.averageRootMeanSquaredError), flush=True)
print('manually scaled average max cm: ' +
      str(originalIK.averageMaxError), flush=True)

# Set the basic measurements
massKg = 68.0
heightM = 1.6

# Load the unscaled Osim file, which we can then scale and format
customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
    absPath('./Rajagopal2015_passiveCal_hipAbdMoved.osim'))

fitter = nimble.biomechanics.MarkerFitter(
    customOsim.skeleton, customOsim.markersMap)
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
result: nimble.biomechanics.MarkerInitialization = fitter.runKinematicsPipeline(
    markerTrajectories.markerTimesteps, nimble.biomechanics.InitialMarkerFitParams())
customOsim.skeleton.setGroupScales(result.groupScales)
fitMarkers: Dict[str, Tuple[nimble.dynamics.BodyNode,
                            np.ndarray]] = result.updatedMarkerMap
resultIK: nimble.biomechanics.IKErrorReport = nimble.biomechanics.IKErrorReport(
    customOsim.skeleton, fitMarkers, result.poses, markerTrajectories.markerTimesteps)
print('automatically scaled average RMSE cm: ' +
      str(resultIK.averageRootMeanSquaredError), flush=True)
print('automatically scaled average max cm: ' +
      str(resultIK.averageMaxError), flush=True)

# Save a results file
fitter.saveTrajectoryAndMarkersToGUI(
    "./results.json", result, markerTrajectories.markerTimesteps)

# Debug our results to the GUI
world: nimble.simulation.World = nimble.simulation.World()
gui: NimbleGUI = NimbleGUI(world)
gui.serve(8080)
fitter.debugTrajectoryAndMarkersToGUI(
    gui.nativeAPI(), result, markerTrajectories.markerTimesteps)
gui.blockWhileServing()
