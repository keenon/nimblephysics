import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import nimblephysics as nimble
import os
from typing import Dict
from nimblephysics import NativeLossFn, NativeTrajectoryRollout, NimbleGUI


def main():
  world = nimble.simulation.World()
  world.setGravity([0, -9.81, 0])

  # Set up skeleton
  atlas: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "../../data/sdf/atlas/atlas_v3_no_head.urdf"))
  atlas.setPosition(0, -0.5 * 3.14159)

  # Generate a random imagined acceleration we imagine we saw from data
  acc = np.random.rand(atlas.getNumDofs())
  next_vel = atlas.getVelocities() + atlas.getTimeStep() * acc

  # Pick the body that's in contact, with a contact model we don't understand
  l_foot = atlas.getBodyNode("l_foot")

  # Solve for inverse dynamics, with a 6-dof wrench on the body that's in contact, by no residual forces at the root of the skeleton.
  result = atlas.getContactInverseDynamics(next_vel, l_foot)
  print("joint torques (first 6 should be 0): "+str(result.jointTorques))
  print("contact wrench (applied at the foot): "+str(result.contactWrench))
  print("numerical error: "+str(result.sumError()))


if __name__ == "__main__":
  main()
