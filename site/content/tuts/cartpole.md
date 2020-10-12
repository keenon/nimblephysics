---
title: "Cartpole Demo"
date: 2020-10-01T11:00:57-07:00
draft: false
menu:
  main:
    parent: "tutorials"
    name: "Cartpole Demo"
---

# Cartpole

## Demo

You're going to be creating this:

{{< viewer3d "Cartpole" "/data/cartpole.txt" >}}

## Code

Here's the code to run the cartpole example

{{< code python >}}

```
import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout


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
    cartRail.setForceUpperLimit(0, 10)
    cartRail.setForceLowerLimit(0, -10)

    poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
    poleJoint.setAxis([0, 0, 1])
    poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.1, 1.0, .1]))
    poleVisual = poleShape.createVisualAspect()
    poleVisual.setColor([0.7, 0.7, 0.7])
    poleJoint.setForceUpperLimit(0, 0)
    poleJoint.setForceLowerLimit(0, 0)

    poleOffset = dart.math.Isometry3()
    poleOffset.set_translation([0, -0.5, 0])
    poleJoint.setTransformFromChildBodyNode(poleOffset)

    world.addSkeleton(cartpole)

    # Make simulations repeatable
    random.seed(1234)

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(world.getTimeStep()*10)

    def loss(rollout: DartTorchTrajectoryRollout):
        posLoss = rollout.getPoses('identity')[:, -1].square().sum()
        velLoss = rollout.getVels('identity')[:, -1].square().sum()
        return posLoss + velLoss
    dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

    world.setPositions([1, 1])

    trajectory = dart.trajectory.MultiShot(world, dartLoss, 500, 50, False)

    optimizer = dart.trajectory.IPOptOptimizer()
    optimizer.setLBFGSHistoryLength(5)
    optimizer.setTolerance(1e-6)
    optimizer.setCheckDerivatives(False)
    optimizer.setIterationLimit(500)
    optimizer.optimize(trajectory)

    json = result.toJson(world)
    text_file = open("cartpole.txt", "w")
    n = text_file.write(json)
    text_file.close()

    dart.dart_serve_web_gui(json)


if __name__ == "__main__":
    main()
```

{{< /code >}}
