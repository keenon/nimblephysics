import numpy as np
import nimblephysics as dart


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
  projectileJoint.setControlForceUpperLimit(0, 0)
  projectileJoint.setControlForceLowerLimit(0, 0)
  projectileJoint.setControlForceUpperLimit(1, 0)
  projectileJoint.setControlForceLowerLimit(1, 0)
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
    poleJoint.setControlForceUpperLimit(0, 1000.0)
    poleJoint.setControlForceLowerLimit(0, -1000.0)
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
  floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([3.5, 0.25, .5]))
  floorVisual = floorShape.createVisualAspect()
  floorVisual.setColor([0.5, 0.5, 0.5])
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

  # Do a benchmark
  # projectile.setPosition(0, -0.1)
  projectile.setPosition(1, 0.0)
  catapult.setPosition(2, 0.65)

  world.step()

  snapshot: dart.neural.BackpropSnapshot = dart.neural.forwardPass(world)
  snapshot.benchmarkJacobians(world, 100)

  """
  gui = DartGUI()
  gui.stateMachine().renderWorld(world)
  gui.serve(8080)
  gui.stateMachine().blockWhileServing()
  """


if __name__ == "__main__":
  main()
