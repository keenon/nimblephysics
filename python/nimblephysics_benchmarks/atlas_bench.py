import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
import os
from typing import Dict
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI, GUITrajectoryTrainer


def main():
  world = dart.simulation.World()
  world.setGravity([0, -9.81, 0])
  # world.setSlowDebugResultsAgainstFD(True)

  # Set up skeleton
  atlas: dart.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "../../data/sdf/atlas/atlas_v3_no_head.urdf"))
  atlas.setPosition(0, -0.5 * 3.14159)
  ground: dart.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "../../data/sdf/atlas/ground.urdf"))
  floorBody: dart.dynamics.BodyNode = ground.getBodyNode(0)
  floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)

  snapshot: dart.neural.BackpropSnapshot = dart.neural.forwardPass(world)
  snapshot.benchmarkJacobians(world, 100)

  """
  gui = DartGUI()
  gui.stateMachine().renderWorld(world)
  gui.serve(8080)
  gui.stateMachine().blockWhileServing()
  """


if __name__ == "__main__":
  main()
