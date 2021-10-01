import nimblephysics_libs._nimblephysics as nimble
import sys
import os


def absPath(path: str):
  root_file_path = os.path.join(os.getcwd(), sys.argv[0])
  absolute_path = os.path.join(os.path.dirname(root_file_path), path)
  return absolute_path


def loadWorld(path: str):
  root_file_path = os.path.join(os.getcwd(), sys.argv[0])
  absolute_path = os.path.join(os.path.dirname(root_file_path), path)
  return nimble.utils.UniversalLoader.loadWorld(absolute_path)
