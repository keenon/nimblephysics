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
  # world.setSlowDebugResultsAgainstFD(True)

  # Set up skeleton
  atlas: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "../../data/sdf/atlas/atlas_v3_no_head.urdf"))
  atlas.setPosition(0, -0.5 * 3.14159)
  ground: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "../../data/sdf/atlas/ground.urdf"))
  floorBody: nimble.dynamics.BodyNode = ground.getBodyNode(0)
  floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)

  gui = NimbleGUI(world)

  # snapshot: nimble.neural.BackpropSnapshot = nimble.neural.forwardPass(world)
  # snapshot.benchmarkJacobians(world, 100)
  # return

  forceLimits = np.ones([atlas.getNumDofs()]) * 500
  forceLimits[0:6] = 0
  atlas.setControlForceUpperLimits(forceLimits)
  atlas.setControlForceLowerLimits(forceLimits * -1)

  goal_x = 0.0
  goal_y = 0.8
  goal_z = -1.0

  def loss(rollout: NativeTrajectoryRollout):
    pos = rollout.getPoses('ik')
    last_pos_x = pos[0, -1]
    last_pos_y = pos[1, -1]
    last_pos_z = pos[2, -1]
    # gui.stateMachine().setObjectPosition("last_pos", [last_pos_x, last_pos_y, last_pos_z])
    return torch.square(last_pos_x - goal_x) + torch.square(last_pos_y - goal_y) + torch.square(last_pos_z - goal_z)
  nimbleLoss: nimble.trajectory.LossFn = NativeLossFn(loss)

  trajectory = nimble.trajectory.MultiShot(world, nimbleLoss, 400, 20, False)

  ikMap: nimble.neural.IKMapping = nimble.neural.IKMapping(world)
  handNode: nimble.dynamics.BodyNode = atlas.getBodyNode("l_hand")
  ikMap.addLinearBodyNode(handNode)
  trajectory.addMapping('ik', ikMap)
  trajectory.setParallelOperationsEnabled(True)

  optimizer = nimble.trajectory.IPOptOptimizer()
  optimizer.setLBFGSHistoryLength(3)
  optimizer.setTolerance(1e-5)
  optimizer.setCheckDerivatives(False)
  optimizer.setIterationLimit(500)
  optimizer.setRecordPerformanceLog(False)

  gui = NimbleGUI(world)
  gui.serve(8080)

  def onIteration(problem: nimble.trajectory.MultiShot, iter: int, loss: float, infeas: float):
    rollout: nimble.trajectory.TrajectoryRollout = problem.getRolloutCache(world)
    poses: np.ndarray = rollout.getPoses()
    gui.loopPosMatrix(poses)
    return True

  optimizer.registerIntermediateCallback(onIteration)

  # Use a not-too-bright green for the goal sphere
  gui.nativeAPI().createSphere("goal_pos", 0.02, np.array(
      [goal_x, goal_y, goal_z]), np.array([118/255, 224/255, 65/255]), True, False)

  result = optimizer.optimize(trajectory)

  gui.blockWhileServing()


if __name__ == "__main__":
  main()
