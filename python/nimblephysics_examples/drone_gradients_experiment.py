import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
import os
from typing import Dict, List
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI, GUITrajectoryTrainer
import matplotlib.pyplot as plt


def optimizeDrone(
        exploreAlternateStrategies: bool, goal: float = 5.0, timesteps: int = 500, iters: int = 500):
  world = dart.simulation.World()
  world.setGravity([0, -9.81, 0])
  # world.setSlowDebugResultsAgainstFD(True)

  drone = dart.dynamics.Skeleton()
  droneRail, droneBody = drone.createPrismaticJointAndBodyNodePair()
  droneRail.setAxis([0, 1, 0])
  droneShape = droneBody.createShapeNode(dart.dynamics.SphereShape(0.3))
  # droneVisual = droneShape.createVisualAspect()
  # droneVisual.setColor([0.8, 0.5, 0.5])
  droneShape.createCollisionAspect()
  droneBody.setFrictionCoeff(0.0)

  droneRail.setControlForceUpperLimit(0, 10)
  droneRail.setControlForceLowerLimit(0, -10)

  world.addSkeleton(drone)

  floor = dart.dynamics.Skeleton()
  floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
  floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([5.0, 1.0, 5.0]))
  floorVisual = floorShape.createVisualAspect()
  floorVisual.setColor([0.5, 0.5, 0.5])
  floorVisual.setCastShadows(False)
  floorShape.createCollisionAspect()

  world.addSkeleton(floor)

  drone.setPosition(0, 0.5 + 0.3 - 1e-4)
  drone.setVelocity(0, - 1e-4)
  droneBody.setMass(0.1)
  world.setTimeStep(0.001)

  def loss(rollout: DartTorchTrajectoryRollout):
    pos = rollout.getPoses()
    last_pos = pos[0, -1]
    # vel = rollout.getVels()
    # last_vel = pos[0, -1]
    # force_loss = rollout.getControlForces().square().sum() / (10 * rollout.getControlForces().size()[1])
    return torch.square(goal - last_pos)  # + 0.1 * torch.square(last_vel)
  dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

  trajectory = dart.trajectory.SingleShot(world, dartLoss, timesteps, False)
  trajectory.setExploreAlternateStrategies(exploreAlternateStrategies)

  optimizer = dart.trajectory.SGDOptimizer()
  optimizer.setIterationLimit(iters)
  optimizer.setLearningRate(10.0)

  losses: List[float] = []

  def afterOptimizationStep(
          problem: dart.trajectory.Problem, n: int, loss: float, infeas: float):
    losses.append(loss)
    return True

  optimizer.registerIntermediateCallback(afterOptimizationStep)
  optimizer.optimize(trajectory)

  rollout: dart.trajectory.TrajectoryRollout = trajectory.getRolloutCache(world)
  poses = rollout.getPoses()[0, :].flatten()

  return (np.array(losses), poses)


def main():
  timesteps = 500
  goal = 5.0
  iters = 50

  naiveLosses, naivePoses = optimizeDrone(
      exploreAlternateStrategies=False, goal=goal, timesteps=timesteps, iters=iters)
  exploreLosses, explorePoses = optimizeDrone(
      exploreAlternateStrategies=True, goal=goal, timesteps=timesteps, iters=iters)

  steps = np.arange(0, iters)
  fig = plt.figure()
  ax1 = fig.add_subplot()
  ax1.plot(steps, naiveLosses)
  ax1.plot(steps, exploreLosses)
  ax1.legend(['Naive gradients', 'Complimentarity aware gradients'])
  ax1.set_title('Drone learns to lift off (SGD, single shooting)')
  ax1.set_xlabel('Iteration of SGD')
  ax1.set_ylabel('Loss')
  ax1.axhline(y=0, color='k')
  ax1.axvline(x=0, color='k')
  plt.show()

  """
  # tmp
  naivePoses = np.linspace(0.5+0.3, 0.5+0.3, timesteps)
  explorePoses = np.linspace(0.5+0.3, 5.0, timesteps)
  # </tmp>
  """

  gui = DartGUI()

  droneBodyMesh = dart.utils.UniversalLoader.loadMeshShape("./data/obj/drone-body.obj")
  droneBodyMesh.setScale([3, 3, 3])
  droneRotorMesh = dart.utils.UniversalLoader.loadMeshShape("./data/obj/drone-rotor.obj")
  droneRotorMesh.setScale([3, 3, 3])

  rotorOffset = 0.325
  scale = np.array([3, 3, 3])

  naiveColor = np.array([176.0/255, 7.0/255, 35.0/255])
  naiveDronePos = np.array([1, naivePoses[0], 0])
  naiveRotorRate = 0.05

  exploreColor = np.array([48.0/255, 140.0/255, 6.0/255])
  exploreDronePos = np.array([-1, explorePoses[0], 0])
  exploreRotorRate = 0.5

  gui.stateMachine().createBox(
      "floor", [5.0, 1.0, 5.0],
      [0, 0, 0],
      [0, 0, 0],
      [0.7, 0.7, 0.7],
      castShadows=False, receiveShadows=True)

  gui.stateMachine().createMeshFromShape("naiveDroneBody", droneBodyMesh,
                                         naiveDronePos, [0, 0, 0], scale, naiveColor)
  gui.stateMachine().createMeshFromShape("naiveDroneRotor1", droneRotorMesh, np.array(
      naiveDronePos + [rotorOffset, 0, rotorOffset]), [0, 0, 0], scale, naiveColor)
  gui.stateMachine().createMeshFromShape("naiveDroneRotor2", droneRotorMesh, np.array(
      naiveDronePos + [rotorOffset, 0, -rotorOffset]), [0, 0, 0], scale, naiveColor)
  gui.stateMachine().createMeshFromShape("naiveDroneRotor3", droneRotorMesh, np.array(
      naiveDronePos + [-rotorOffset, 0, -rotorOffset]), [0, 0, 0], scale, naiveColor)
  gui.stateMachine().createMeshFromShape("naiveDroneRotor4", droneRotorMesh, np.array(
      naiveDronePos + [-rotorOffset, 0, rotorOffset]), [0, 0, 0], scale, naiveColor)

  gui.stateMachine().createMeshFromShape("exploreDroneBody", droneBodyMesh,
                                         exploreDronePos, [0, 0, 0], scale, exploreColor)
  gui.stateMachine().createMeshFromShape("exploreDroneRotor1", droneRotorMesh, np.array(
      exploreDronePos + [rotorOffset, 0, rotorOffset]), [0, 0, 0], scale, exploreColor)
  gui.stateMachine().createMeshFromShape("exploreDroneRotor2", droneRotorMesh, np.array(
      exploreDronePos + [rotorOffset, 0, -rotorOffset]), [0, 0, 0], scale, exploreColor)
  gui.stateMachine().createMeshFromShape("exploreDroneRotor3", droneRotorMesh, np.array(
      exploreDronePos + [-rotorOffset, 0, -rotorOffset]), [0, 0, 0], scale, exploreColor)
  gui.stateMachine().createMeshFromShape("exploreDroneRotor4", droneRotorMesh, np.array(
      exploreDronePos + [-rotorOffset, 0, rotorOffset]), [0, 0, 0], scale, exploreColor)

  gui.stateMachine().createSphere('naiveGoal', 0.1, [
      naiveDronePos[0], goal, naiveDronePos[2]], [1, 0, 0])
  gui.stateMachine().createSphere('exploreGoal', 0.1, [
      exploreDronePos[0], goal, exploreDronePos[2]], [0, 1, 0])

  gui.stateMachine().createLine('naiveLine', [np.array(
      [naiveDronePos[0], pos, naiveDronePos[2]]) for pos in naivePoses], naiveColor)
  gui.stateMachine().createLine('exploreLine', [np.array(
      [exploreDronePos[0], pos, exploreDronePos[2]]) for pos in explorePoses], exploreColor)

  ticker = dart.realtime.Ticker(0.01)

  i = 0

  gui.stateMachine().setAutoflush(False)

  def onTick(now):
    nonlocal i
    nonlocal naiveDronePos
    nonlocal exploreDronePos

    naiveDronePos[1] = naivePoses[i]
    exploreDronePos[1] = explorePoses[i]

    # Move the naive drone
    gui.stateMachine().setObjectPosition("naiveDroneBody", naiveDronePos)
    gui.stateMachine().setObjectPosition("naiveDroneRotor1", naiveDronePos + [
        rotorOffset, 0, rotorOffset])
    gui.stateMachine().setObjectPosition("naiveDroneRotor2", naiveDronePos + [
        rotorOffset, 0, -rotorOffset])
    gui.stateMachine().setObjectPosition("naiveDroneRotor3", naiveDronePos + [
        -rotorOffset, 0, -rotorOffset])
    gui.stateMachine().setObjectPosition("naiveDroneRotor4", naiveDronePos + [
        -rotorOffset, 0, rotorOffset])
    gui.stateMachine().setObjectRotation("naiveDroneRotor1", [0, i*naiveRotorRate, 0])
    gui.stateMachine().setObjectRotation("naiveDroneRotor2", [0, -i*naiveRotorRate, 0])
    gui.stateMachine().setObjectRotation("naiveDroneRotor3", [0, i*naiveRotorRate, 0])
    gui.stateMachine().setObjectRotation("naiveDroneRotor4", [0, -i*naiveRotorRate, 0])

    # Move the explore drone
    gui.stateMachine().setObjectPosition("exploreDroneBody", exploreDronePos)
    gui.stateMachine().setObjectPosition("exploreDroneRotor1", exploreDronePos + [
        rotorOffset, 0, rotorOffset])
    gui.stateMachine().setObjectPosition("exploreDroneRotor2", exploreDronePos + [
        rotorOffset, 0, -rotorOffset])
    gui.stateMachine().setObjectPosition("exploreDroneRotor3", exploreDronePos + [
        -rotorOffset, 0, -rotorOffset])
    gui.stateMachine().setObjectPosition("exploreDroneRotor4", exploreDronePos + [
        -rotorOffset, 0, rotorOffset])
    gui.stateMachine().setObjectRotation("exploreDroneRotor1", [0, i*exploreRotorRate, 0])
    gui.stateMachine().setObjectRotation("exploreDroneRotor2", [0, -i*exploreRotorRate, 0])
    gui.stateMachine().setObjectRotation("exploreDroneRotor3", [0, i*exploreRotorRate, 0])
    gui.stateMachine().setObjectRotation("exploreDroneRotor4", [0, -i*exploreRotorRate, 0])

    gui.stateMachine().flush()

    i += 1
    if i >= timesteps:
      i = 0

  def onConnect():
    ticker.start()

  ticker.registerTickListener(onTick)
  gui.stateMachine().registerConnectionListener(onConnect)
  gui.serve(8080)
  gui.stateMachine().blockWhileServing()


if __name__ == "__main__":
  main()
