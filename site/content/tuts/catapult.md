---
title: "Catapult Demo"
date: 2020-10-01T11:00:57-07:00
draft: false
menu:
  main:
    parent: "tutorials"
    name: "Catapult Demo"
---

# Catapult

## Demo

You're going to be creating this:

{{< viewer3d "Catapult" "/data/catapult.txt" >}}

## Code

Here's the code to run the catapult example, tested against DiffDART version 0.3.4:

{{< warning >}}
**Warning**: DiffDART is still in Alpha, and we reserve the right to change the APIs in breaking ways (like renaming API calls, for example). To ensure that this demo works, make sure you're using version 0.3.4 by running `pip3 install diffdart==0.3.4`
{{< /warning >}}

{{< code python >}}

```
import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI, GUITrajectoryTrainer


def main():
  world = dart.simulation.World()
  world.setGravity([0, -9.81, 0])

  # Set up the projectile

  projectile = dart.dynamics.Skeleton()

  projectileJoint, projectileNode = projectile.createTranslationalJoint2DAndBodyNodePair()
  projectileJoint.setXYPlane()
  projectileShape = projectileNode.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
  projectileVisual = projectileShape.createVisualAspect()
  projectileShape.createCollisionAspect()
  projectileVisual.setColor([0.7, 0.7, 0.7])
  projectileVisual.setCastShadows(False)
  projectileJoint.setForceUpperLimit(0, 0)
  projectileJoint.setForceLowerLimit(0, 0)
  projectileJoint.setForceUpperLimit(1, 0)
  projectileJoint.setForceLowerLimit(1, 0)
  projectileJoint.setVelocityUpperLimit(0, 1000.0)
  projectileJoint.setVelocityLowerLimit(0, -1000.0)
  projectileJoint.setVelocityUpperLimit(1, 1000.0)
  projectileJoint.setVelocityLowerLimit(1, -1000.0)

  projectile.setPositions(np.array([0, 0.1]))

  world.addSkeleton(projectile)

  # Set up catapult

  catapult = dart.dynamics.Skeleton()

  rootJoint, root = catapult.createWeldJointAndBodyNodePair()
  rootOffset = dart.math.Isometry3()
  rootOffset.set_translation([0.5, -0.45, 0])
  rootJoint.setTransformFromParentBodyNode(rootOffset)

  def createTailSegment(parent, color):
    poleJoint, pole = catapult.createRevoluteJointAndBodyNodePair(parent)
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

  catapult.setPositions(np.array([45, 0, 45]) * 3.1415 / 180)

  world.addSkeleton(catapult)

  # Floor

  floor = dart.dynamics.Skeleton()
  floor.setName('floor')  # important for rendering shadows

  floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
  floorOffset = dart.math.Isometry3()
  floorOffset.set_translation([1.2, -0.7, 0])
  floorJoint.setTransformFromParentBodyNode(floorOffset)
  floorShape: dart.dynamics.ShapeNode = floorBody.createShapeNode(dart.dynamics.BoxShape(
      [3.5, 0.25, .5]))
  floorVisual: dart.dynamics.VisualAspect = floorShape.createVisualAspect()
  floorVisual.setColor([0.5, 0.5, 0.5])
  floorVisual.setCastShadows(False)
  floorShape.createCollisionAspect()

  world.addSkeleton(floor)

  # Target

  target_x = 2.2
  target_y = 2.2

  target = dart.dynamics.Skeleton()
  target.setName('target')  # important for rendering shadows

  targetJoint, targetBody = floor.createWeldJointAndBodyNodePair()
  targetOffset = dart.math.Isometry3()
  targetOffset.set_translation([target_x, target_y, 0])
  targetJoint.setTransformFromParentBodyNode(targetOffset)
  targetShape = targetBody.createShapeNode(dart.dynamics.BoxShape([0.1, 0.1, 0.1]))
  targetVisual = targetShape.createVisualAspect()
  targetVisual.setColor([0.8, 0.5, 0.5])

  world.addSkeleton(target)

  # Set up the view

  def loss(rollout: DartTorchTrajectoryRollout):
    last_pos = rollout.getPoses('identity')[:, -1]
    last_x = last_pos[0]
    last_y = last_pos[1]
    final_loss = (target_x - last_x)**2 + (target_y - last_y)**2
    return final_loss
  dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

  trajectory = dart.trajectory.MultiShot(world, dartLoss, 500, 20, False)
  trajectory.setParallelOperationsEnabled(True)

  optimizer = dart.trajectory.IPOptOptimizer()
  optimizer.setLBFGSHistoryLength(5)
  optimizer.setTolerance(1e-4)
  optimizer.setCheckDerivatives(False)
  optimizer.setIterationLimit(500)
  # result: dart.trajectory.Solution = optimizer.optimize(trajectory)

  trainer = GUITrajectoryTrainer(world, trajectory, optimizer)
  trainer.train(loopAfterSolve=True)


if __name__ == "__main__":
  main()
```

{{< /code >}}
