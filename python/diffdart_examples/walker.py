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

    walker = dart.dynamics.Skeleton()

    rootJoint, root = walker.createFreeJointAndBodyNodePair()
    rootShape = root.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
    rootVisual = rootShape.createVisualAspect()
    rootShape.createCollisionAspect()
    rootVisual.setColor([0.7, 0.7, 0.7])
    rootJoint.setForceUpperLimit(0, 0)
    rootJoint.setForceLowerLimit(0, 0)
    rootJoint.setForceUpperLimit(1, 0)
    rootJoint.setForceLowerLimit(1, 0)
    rootJoint.setForceUpperLimit(2, 0)
    rootJoint.setForceLowerLimit(2, 0)
    rootJoint.setForceUpperLimit(3, 0)
    rootJoint.setForceLowerLimit(3, 0)
    rootJoint.setForceUpperLimit(4, 0)
    rootJoint.setForceLowerLimit(4, 0)
    rootJoint.setForceUpperLimit(5, 0)
    rootJoint.setForceLowerLimit(5, 0)
    rootJoint.setVelocityUpperLimit(0, 1000.0)
    rootJoint.setVelocityLowerLimit(0, -1000.0)
    rootJoint.setVelocityUpperLimit(1, 1000.0)
    rootJoint.setVelocityLowerLimit(1, -1000.0)
    rootJoint.setVelocityUpperLimit(2, 1000.0)
    rootJoint.setVelocityLowerLimit(2, -1000.0)
    rootJoint.setVelocityUpperLimit(3, 1000.0)
    rootJoint.setVelocityLowerLimit(3, -1000.0)
    rootJoint.setVelocityUpperLimit(4, 1000.0)
    rootJoint.setVelocityLowerLimit(4, -1000.0)
    rootJoint.setVelocityUpperLimit(5, 1000.0)
    rootJoint.setVelocityLowerLimit(5, -1000.0)

    def createTailSegment(parent, color, zOffset=0.0):
        poleJoint, pole = walker.createRevoluteJointAndBodyNodePair(parent)
        poleJoint.setAxis([0, 0, 1])
        poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.05, 0.25, .05]))
        poleVisual = poleShape.createVisualAspect()
        poleVisual.setColor(color)
        poleJoint.setForceUpperLimit(0, 100.0)
        poleJoint.setForceLowerLimit(0, -100.0)
        poleJoint.setVelocityUpperLimit(0, 10000.0)
        poleJoint.setVelocityLowerLimit(0, -10000.0)

        poleOffset = dart.math.Isometry3()
        poleOffset.set_translation([0, -0.125, 0])
        poleJoint.setTransformFromChildBodyNode(poleOffset)

        poleJoint.setPosition(0, 90 * 3.1415 / 180)

        poleShape.createCollisionAspect()

        if parent != root:
            childOffset = dart.math.Isometry3()
            childOffset.set_translation([0, 0.125, 0])
            poleJoint.setTransformFromParentBodyNode(childOffset)
            # Knee Joint
            poleJoint.setPositionUpperLimit(0, 3 * 3.1415 / 180)
            poleJoint.setPositionLowerLimit(0, -90 * 3.1415 / 180)
        else:
            childOffset = dart.math.Isometry3()
            childOffset.set_translation([0, 0, zOffset])
            poleJoint.setTransformFromParentBodyNode(childOffset)
            # Hip Joint
            poleJoint.setPositionUpperLimit(0, 250 * 3.1415 / 180)
            poleJoint.setPositionLowerLimit(0, 120 * 3.1415 / 180)
        return pole

    leg1thigh = createTailSegment(root, [182.0/255, 223.0/255, 144.0/255], -0.05)
    leg1shin = createTailSegment(leg1thigh, [223.0/255, 228.0/255, 163.0/255])

    leg2thigh = createTailSegment(root, [221.0/255, 193.0/255, 121.0/255], 0.05)
    leg2shin = createTailSegment(leg2thigh, [226.0/255, 137.0/255, 79.0/255])

    walker.setPositions(np.array([0, 0, 0, 0, 0, 0, 190, -50, 170, -5]) * 3.1415 / 180)

    world.addSkeleton(walker)

    # Floor

    floor = dart.dynamics.Skeleton()
    floor.setName('floor')  # important for rendering shadows

    floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
    floorOffset = dart.math.Isometry3()
    floorOffset.set_translation([0, -0.62, 0])
    floorJoint.setTransformFromParentBodyNode(floorOffset)
    floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([2.5, 0.25, .5]))
    floorVisual = floorShape.createVisualAspect()
    floorVisual.setColor([0.5, 0.5, 0.5])
    floorShape.createCollisionAspect()
    # floorBody.setFrictionCoeff(0)

    world.addSkeleton(floor)

    # Set up the view

    def loss(rollout: DartTorchTrajectoryRollout):
        pos = rollout.getPoses('identity')
        # vel = rollout.getVels('ik')

        # Keep the hips at 0 rotation
        # step_loss = - torch.sum(pos[0, :] * pos[0, :] * torch.sign(pos[0, :]))

        last_segment_pos = pos[:, -1]
        final_loss = -last_segment_pos[0]**2
        # print(pos.shape)
        # final_loss = - 100 * last_segment_pos[1] * \
        # last_segment_pos[1] * torch.sign(last_segment_pos[1])
        # return step_loss + final_loss
        return final_loss
    dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

    trajectory = dart.trajectory.MultiShot(world, dartLoss, 800, 20, False)

    def loopConstraintLoss(rollout: DartTorchTrajectoryRollout):
        pos = rollout.getPoses('identity')
        first_pos = pos[:, 0]
        last_pos = pos[:, -1]
        ignore_joint = 4  # this is the x-axis
        first_pos_trimmed = torch.cat([first_pos[0:ignore_joint], first_pos[ignore_joint+1:]])
        last_pos_trimmed = torch.cat([last_pos[0:ignore_joint], last_pos[ignore_joint+1:]])
        diff = first_pos_trimmed - last_pos_trimmed
        return torch.sum(torch.square(diff))
    loopConstraint: dart.trajectory.LossFn = DartTorchLossFn(loopConstraintLoss)
    loopConstraint.setLowerBound(0)
    loopConstraint.setUpperBound(0)

    # trajectory.addConstraint(loopConstraint)

    # ikMap: dart.neural.IKMapping = dart.neural.IKMapping(world)
    # ikMap.addLinearBodyNode(root)
    # trajectory.addMapping('ik', ikMap)

    optimizer = dart.trajectory.IPOptOptimizer()
    # optimizer.setLBFGSHistoryLength(5)
    optimizer.setTolerance(1e-5)
    optimizer.setCheckDerivatives(True)
    optimizer.setIterationLimit(200)
    result = optimizer.optimize(trajectory)

    json = result.toJson(world)
    text_file = open("worm.txt", "w")
    n = text_file.write(json)
    text_file.close()

    dart.dart_serve_web_gui(json)


if __name__ == "__main__":
    main()
