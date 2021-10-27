import os
import pathlib
import nimblephysics as nimble


file_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    'rajagopal_data', 'Rajagopal2015.osim')


def RajagopalHumanBodyModel():
  return nimble.biomechanics.OpenSimParser.parseOsim(file_path)
