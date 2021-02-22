import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
from typing import Dict
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI


def main():
  world = dart.simulation.World()
  world.setGravity([0, -9.81, 0])

  # Set up the 2D cartpole

  jumpworm = dart.dynamics.Skeleton()

  rootJoint, root = jumpworm.createTranslationalJoint2DAndBodyNodePair()
  rootJoint.setXYPlane()
  rootShape = root.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
  rootVisual = rootShape.createVisualAspect()
  rootShape.createCollisionAspect()
  rootVisual.setColor([0.7, 0.7, 0.7])
  rootJoint.setForceUpperLimit(0, 0)
  rootJoint.setForceLowerLimit(0, 0)
  rootJoint.setForceUpperLimit(1, 0)
  rootJoint.setForceLowerLimit(1, 0)
  rootJoint.setVelocityUpperLimit(0, 1000.0)
  rootJoint.setVelocityLowerLimit(0, -1000.0)
  rootJoint.setVelocityUpperLimit(1, 1000.0)
  rootJoint.setVelocityLowerLimit(1, -1000.0)

  def createTailSegment(parent, color):
    poleJoint, pole = jumpworm.createRevoluteJointAndBodyNodePair(parent)
    poleJoint.setAxis([0, 0, 1])
    poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.05, 0.25, .05]))
    poleVisual = poleShape.createVisualAspect()
    poleVisual.setColor(color)
    poleJoint.setForceUpperLimit(0, 1000.0)
    poleJoint.setForceLowerLimit(0, -1000.0)
    poleJoint.setVelocityUpperLimit(0, 10000.0)
    poleJoint.setVelocityLowerLimit(0, -10000.0)

    poleOffset = dart.math.Isometry3()
    poleOffset.set_translation([0, -0.125, 0])
    poleJoint.setTransformFromChildBodyNode(poleOffset)

    poleJoint.setPosition(0, 90 * 3.1415 / 180)
    poleJoint.setPositionUpperLimit(0, 180 * 3.1415 / 180)
    poleJoint.setPositionLowerLimit(0, 0 * 3.1415 / 180)

    poleShape.createCollisionAspect()

    if parent != root:
      childOffset = dart.math.Isometry3()
      childOffset.set_translation([0, 0.125, 0])
      poleJoint.setTransformFromParentBodyNode(childOffset)
    return pole

  tail1 = createTailSegment(root, [182.0/255, 223.0/255, 144.0/255])
  tail2 = createTailSegment(tail1, [223.0/255, 228.0/255, 163.0/255])
  tail3 = createTailSegment(tail2, [221.0/255, 193.0/255, 121.0/255])
  # tail4 = createTailSegment(tail3, [226.0/255, 137.0/255, 79.0/255])

  jumpworm.setPositions(np.array([0, 0, 90, 90, 45]) * 3.1415 / 180)

  world.addSkeleton(jumpworm)

  # Floor

  floor = dart.dynamics.Skeleton()
  floor.setName('floor')  # important for rendering shadows

  floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
  floorOffset = dart.math.Isometry3()
  floorOffset.set_translation([0, -0.7, 0])
  floorJoint.setTransformFromParentBodyNode(floorOffset)
  floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([2.5, 0.25, .5]))
  floorVisual = floorShape.createVisualAspect()
  floorVisual.setColor([0.5, 0.5, 0.5])
  floorShape.createCollisionAspect()
  # floorBody.setFrictionCoeff(0)

  world.addSkeleton(floor)

  # Set up the view

  goal_x = 0.0
  goal_y = 0.3

  def loss(rollout: DartTorchTrajectoryRollout):
    pos = rollout.getPoses()
    head_x = pos[-1, 0]
    head_y = pos[-1, 1]
    diff_x = head_x - goal_x
    diff_y = head_y - goal_y
    return (diff_x * diff_x) + (diff_y * diff_y)
  dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

  gui = DartGUI()
  gui.serve(8080)
  gui.stateMachine().renderWorld(world, "world")
  gui.stateMachine().createSphere("goal_pos", 0.05, np.array(
      [goal_x, goal_y, 0.0]), np.array([118/255, 224/255, 65/255]), True, False)

  def onDrag(pos: np.array):
    goal_x = pos[0]
    goal_y = pos[1]
    gui.stateMachine().setObjectPosition("goal_pos", np.array([goal_x, goal_y, 0.0]))

  gui.stateMachine().registerDragListener("goal_pos", onDrag)

  def onReplan(time: int, rollout: dart.trajectory.TrajectoryRollout, duration: int):
    gui.stateMachine().renderTrajectoryLines(world, rollout.getPoses())
  mpc = dart.realtime.MPCLocal(world.clone(), dartLoss, 3000)
  mpc.registerReplaningListener(onReplan)
  mpc.setSilent(True)

  ticker = dart.realtime.Ticker(world.getTimeStep())
  originalColor = rootVisual.getColor()

  def onTick(now):
    world.setExternalForces(mpc.getForce(now))
    if "a" in gui.stateMachine().getKeysDown():
      perturbedForces = world.getExternalForces()
      perturbedForces[0] = -15.0
      world.setExternalForces(perturbedForces)
      rootVisual.setColor([1, 0, 0])
    elif "e" in gui.stateMachine().getKeysDown():
      perturbedForces = world.getExternalForces()
      perturbedForces[0] = 15.0
      world.setExternalForces(perturbedForces)
      rootVisual.setColor([0, 1, 0])
    else:
      rootVisual.setColor(originalColor)

    world.step()

    mpc.recordGroundTruthState(
        now, world.getPositions(),
        world.getVelocities(),
        world.getMasses())

    gui.stateMachine().renderWorld(world, "world")

  def onConnect():
    ticker.start()
    mpc.start()

  ticker.registerTickListener(onTick)
  gui.stateMachine().registerConnectionListener(onConnect)

  gui.stateMachine().blockWhileServing()

  """
  json = result.toJson(world)
  text_file = open("worm.txt", "w")
  n = text_file.write(json)
  text_file.close()

  dart.dart_serve_optimization_solution(result, world)
  """


if __name__ == "__main__":
  main()
