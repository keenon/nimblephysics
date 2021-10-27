import numpy as np
import torch.nn.functional as F
import nimblephysics as dart
import os


def main():
  world: dart.simulation.World = dart.simulation.World.loadFrom(os.path.join(
      os.path.dirname(__file__), "../../data/skel/half_cheetah.skel"))

  cheetah = world.getSkeleton(1)

  forceLimits = np.ones([cheetah.getNumDofs()]) * 500
  forceLimits[0:1] = 0
  cheetah.setControlForceUpperLimits(forceLimits)
  cheetah.setControlForceLowerLimits(forceLimits * -1)

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
