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
  world: dart.simulation.World = dart.simulation.World.loadFrom(os.path.join(
      os.path.dirname(__file__), "../../data/skel/half_cheetah.skel"))

  cheetah = world.getSkeleton(1)

  forceLimits = np.ones([cheetah.getNumDofs()]) * 500
  forceLimits[0:1] = 0
  cheetah.setForceUpperLimits(forceLimits)
  cheetah.setForceLowerLimits(forceLimits * -1)

  # Do a benchmark
  cheetah.setPosition(2, 0.03)
  cheetah.setPosition(1, -0.1)

  world.step()

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
