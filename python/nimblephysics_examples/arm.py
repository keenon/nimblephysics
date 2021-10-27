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
  # Set up the 2D cartpole
  arm: dart.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "../../data/urdf/KR5/KR5 sixx R650.urdf"))
  ground: dart.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "../../data/sdf/atlas/ground.urdf"))
  floorBody: dart.dynamics.BodyNode = ground.getBodyNode(0)
  floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)
  ticker = dart.realtime.Ticker(world.getTimeStep())

  goal_x = 0.0
  goal_y = 0.8
  goal_z = -1.0

  """
  # Set up a GUI
  gui = DartGUI()
  gui.serve(8080)
  gui.stateMachine().renderWorld(world, "world")
  gui.stateMachine().createSphere(
      "last_pos", 0.1, np.array([0.0, 0.0, 0.0]),
      np.array([1.0, 0.0, 0.0]), True, False)
  """

  forceLimits = np.ones(world.getNumDofs()) * 150
  world.setControlForcesUpperLimits(forceLimits)
  world.setControlForcesLowerLimits(-1 * forceLimits)

  velLimits = np.ones(world.getNumDofs()) * 2
  world.setVelocityUpperLimits(velLimits)
  world.setVelocityLowerLimits(-1 * velLimits)

  # posLimits = np.ones(world.getNumDofs()) * 2
  # world.setPositionUpperLimits(posLimits)
  # world.setPositionLowerLimits(-1 * posLimits)

  def loss(rollout: DartTorchTrajectoryRollout):
    pos = rollout.getPoses('ik')
    last_pos_x = pos[0, -1]
    last_pos_y = pos[1, -1]
    last_pos_z = pos[2, -1]
    last_vel = rollout.getVels()[:, -1]
    # gui.stateMachine().setObjectPosition("last_pos", [last_pos_x, last_pos_y, last_pos_z])
    return torch.square(last_pos_x - goal_x) + torch.square(last_pos_y - goal_y) + torch.square(
        last_pos_z - goal_z)
  dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

  # trajectory = dart.trajectory.MultiShot(world, dartLoss, 500, 50, False)
  trajectory = dart.trajectory.SingleShot(world, dartLoss, 500, False)

  """
  def finalStillness(rollout: DartTorchTrajectoryRollout):
    last_vel = rollout.getVels()[:, -1]
    # gui.stateMachine().setObjectPosition("last_pos", [last_pos_x, last_pos_y, last_pos_z])
    return torch.sum(torch.square(last_vel))
  finalStillnessLoss: dart.trajectory.LossFn = DartTorchLossFn(finalStillness)
  finalStillnessLoss.setLowerBound(0.0)
  finalStillnessLoss.setUpperBound(0.0)
  trajectory.addConstraint(finalStillnessLoss)
  """

  ikMap: dart.neural.IKMapping = dart.neural.IKMapping(world)
  handNode: dart.dynamics.BodyNode = arm.getBodyNode("palm")
  ikMap.addLinearBodyNode(handNode)
  trajectory.addMapping('ik', ikMap)
  # trajectory.setParallelOperationsEnabled(True)

  optimizer = dart.trajectory.IPOptOptimizer()
  optimizer.setLBFGSHistoryLength(5)
  optimizer.setTolerance(1e-5)
  optimizer.setCheckDerivatives(False)
  optimizer.setIterationLimit(500)
  # optimizer.setRecordPerformanceLog(True)

  trainer = GUITrajectoryTrainer(world, trajectory, optimizer)
  trainer.stateMachine().createSphere("goal_pos", 0.1, np.array(
      [goal_x, goal_y, goal_z]), np.array([0.0, 1.0, 0.0]), True, False)
  trainer.train(loopAfterSolve=True)

  """
  def callback(problem: dart.trajectory.MultiShot, iter: int, loss: float, infeas: float):
    print('From Python, iter='+str(iter)+", loss="+str(loss))
    rollout: dart.trajectory.TrajectoryRollout = problem.getRolloutCache(world)
    currentPoses = rollout.getPoses()
    gui.stateMachine().renderTrajectoryLines(world, currentPoses)
    return True
  optimizer.registerIntermediateCallback(callback)

  result = optimizer.optimize(trajectory)

  def onTick(now):
    world.step()
    gui.stateMachine().renderWorld(world, "world")

  def onConnect():
    ticker.start()
  ticker.registerTickListener(onTick)
  gui.stateMachine().registerConnectionListener(onConnect)

  while gui.stateMachine().isServing():
    pass
  """


if __name__ == "__main__":
  main()
