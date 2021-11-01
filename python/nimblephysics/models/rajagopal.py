import os
import pathlib
import nimblephysics as nimble


file_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    'rajagopal_data', 'Rajagopal2015.osim')

anthro_xml_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    'rajagopal_data', 'ANSUR_Rajagopal_metrics.xml')

ansur_male_csv_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    'rajagopal_data', 'ANSUR_II_MALE_Public.csv')


def RajagopalHumanBodyModel():
  return nimble.biomechanics.OpenSimParser.parseOsim(file_path)


def RajagopalANSURModel():
  anthro: nimble.biomechanics.Anthropometrics = nimble.biomechanics.Anthropometrics.loadFromFile(
      anthro_xml_path)
  metricNames: List[str] = anthro.getMetricNames()
  metricNames.append("Age")
  metricNames.append("Weightlbs")
  metricNames.append("Heightin")
  dist: nimble.math.MultivariateGaussian = nimble.math.MultivariateGaussian.loadFromCSV(
      ansur_male_csv_path, metricNames, 0.001)
  anthro.setDistribution(dist)
  return anthro
