import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import nimblephysics as nimble
import os
from typing import Dict


def main():
  world = nimble.simulation.World()
  world.setGravity([0, -9.81, 0])
  # Set up the 2D cartpole
  arm: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "./KR5.urdf"))
  ground: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
      os.path.dirname(__file__), "./ground.urdf"))
  floorBody: nimble.dynamics.BodyNode = ground.getBodyNode(0)
  floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)
  ticker = nimble.realtime.Ticker(world.getTimeStep())

  goal_x = 0.0
  goal_y = 0.8
  goal_z = -1.0
  goal: torch.Tensor = torch.tensor([goal_x, goal_y, goal_z])

  gui = nimble.NimbleGUI(world)
  gui.serve(8080)
  gui.nativeAPI().renderWorld(world, "world")
  gui.nativeAPI().createSphere("goal_pos", 0.1, np.array(
      [goal_x, goal_y, goal_z]), np.array([0.0, 1.0, 0.0]), True, False)

  def onDrag(pos):
    nonlocal goal
    goal = torch.tensor(pos)
    gui.nativeAPI().setObjectPosition("goal_pos", pos)
  gui.nativeAPI().registerDragListener("goal_pos", onDrag)

  ikMap: nimble.neural.IKMapping = nimble.neural.IKMapping(world)
  handNode: nimble.dynamics.BodyNode = arm.getBodyNode("palm")
  ikMap.addLinearBodyNode(handNode)

  state: torch.Tensor = torch.randn((world.getStateSize()), requires_grad=True)

  learning_rate = 0.03

  while True:
    hand_pos: torch.Tensor = nimble.map_to_pos(world, ikMap, state)
    loss = (hand_pos - goal).square().sum()
    loss.backward()
    with torch.no_grad():
      state -= learning_rate * state.grad
      state.grad = None
    gui.nativeAPI().renderWorld(world, "world")
    time.sleep(0.01)

  gui.blockWhileServing()


if __name__ == "__main__":
  main()
