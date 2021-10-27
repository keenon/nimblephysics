import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, dart_layer


def main():
  world = dart.simulation.World()
  world.setGravity([0, -9.81, 0])

  # Set up the 2D cartpole

  cartpole = dart.dynamics.Skeleton()
  cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
  cartRail.setAxis([1, 0, 0])
  cartShape = cart.createShapeNode(dart.dynamics.BoxShape([.5, .1, .1]))
  cartVisual = cartShape.createVisualAspect()
  cartVisual.setColor([0.5, 0.5, 0.5])
  cartRail.setPositionUpperLimit(0, 10)
  cartRail.setPositionLowerLimit(0, -10)
  cartRail.setControlForceUpperLimit(0, 10)
  cartRail.setControlForceLowerLimit(0, -10)

  poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
  poleJoint.setAxis([0, 0, 1])
  poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.1, 1.0, .1]))
  poleVisual = poleShape.createVisualAspect()
  poleVisual.setColor([0.7, 0.7, 0.7])
  poleJoint.setControlForceUpperLimit(0, 0)
  poleJoint.setControlForceLowerLimit(0, 0)

  poleOffset = dart.math.Isometry3()
  poleOffset.set_translation([0, -0.5, 0])
  poleJoint.setTransformFromChildBodyNode(poleOffset)

  world.addSkeleton(cartpole)

  # Make simulations repeatable
  random.seed(1234)

  # Make simulations and backprop run faster by using a bigger timestep
  world.setTimeStep(world.getTimeStep()*10)

  first_pos: torch.Tensor = torch.tensor(world.getPositions(), requires_grad=True)
  first_pos.data[1] = 0.1
  first_vel: torch.Tensor = torch.tensor(world.getVelocities(), requires_grad=True)
  torque: torch.Tensor = torch.tensor(world.getExternalForces(), requires_grad=True)

  for _ in range(50):
    pos = first_pos
    vel = first_vel
    for i in range(100):
      next_pos, next_vel = dart_layer(world, pos, vel, torque)
      pos = next_pos
      vel = next_vel

    loss = pos.norm() + vel.norm()
    print('loss: '+str(loss))
    loss.backward()
    first_pos.data.sub_(first_pos.grad * 0.0001)
    first_vel.data.sub_(first_vel.grad * 0.0001)


if __name__ == "__main__":
  main()
