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

  drone = dart.dynamics.Skeleton()
  droneRail, droneBody = drone.createPrismaticJointAndBodyNodePair()
  droneRail.setAxis([0, 1, 0])
  droneShape = droneBody.createShapeNode(dart.dynamics.SphereShape(0.5))
  droneVisual = droneShape.createVisualAspect()
  droneVisual.setColor([0.8, 0.5, 0.5])
  droneShape.createCollisionAspect()
  droneBody.setFrictionCoeff(0.0)

  droneRail.setForceUpperLimit(0, 10)
  droneRail.setForceLowerLimit(0, -10)

  world.addSkeleton(drone)

  floor = dart.dynamics.Skeleton()
  floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
  floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([5.0, 1.0, 5.0]))
  floorVisual = floorShape.createVisualAspect()
  floorVisual.setColor([0.5, 0.5, 0.5])
  floorVisual.setCastShadows(False)
  floorShape.createCollisionAspect()

  world.addSkeleton(floor)

  drone.setPosition(0, 1 - 1e-4)
  drone.setVelocity(0, - 1e-4)
  droneBody.setMass(0.1)

  """
  # Do a benchmark

  drone.setPosition(0, 0.0)
  world.step()

  # snapshot: dart.neural.BackpropSnapshot = dart.neural.forwardPass(world)
  # snapshot.benchmarkJacobians(world, 100)
  # return

  gui = DartGUI()
  gui.stateMachine().renderWorld(world)
  gui.serve(8080)

  ticker = dart.realtime.Ticker(world.getTimeStep())

  def onTick(now):
    world.step()
    gui.stateMachine().renderWorld(world)

  def onConnect():
    ticker.start()

  ticker.registerTickListener(onTick)
  gui.stateMachine().registerConnectionListener(onConnect)
  gui.stateMachine().blockWhileServing()
  """

  goal = 5.0
  world.setTimeStep(0.001)

  def loss(rollout: DartTorchTrajectoryRollout):
    pos = rollout.getPoses()
    last_pos = pos[0, -1]
    vel = rollout.getVels()
    last_vel = pos[0, -1]
    force_loss = rollout.getForces().square().sum() / (10 * rollout.getForces().size()[1])
    return torch.square(goal - last_pos) + torch.square(last_vel) + force_loss
  dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

  trajectory = dart.trajectory.SingleShot(world, dartLoss, 500, False)
  trajectory.setExploreAlternateStrategies(True)

  optimizer = dart.trajectory.SGDOptimizer()
  optimizer.setIterationLimit(500)
  optimizer.setLearningRate(10.0)

  # optimizer.optimize(trajectory)

  trainer = GUITrajectoryTrainer(world, trajectory, optimizer)
  result = trainer.train(loopAfterSolve=True)


if __name__ == "__main__":
  main()
