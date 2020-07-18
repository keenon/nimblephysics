from context import dart_torch
import numpy as np
import dartpy as dart
import torch
import torch.nn.functional as F
import random
import math
import time


class MyWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world):
        super(MyWorldNode, self).__init__(world)

    def customPreStep(self):
        pass


def main():
    world = dart.simulation.World()
    world.setGravity([0, -9.81, 0])

    box = dart.dynamics.BoxShape(np.array([.1, .1, .1]))

    # Set up the 2D cartpole

    jumpworm = dart.dynamics.Skeleton()

    rootJoint, root = jumpworm.createTranslationalJoint2DAndBodyNodePair()
    rootJoint.setXYPlane()
    rootShape = root.createShapeNode(dart.dynamics.BoxShape(np.array([.1, .1, .1])))
    rootVisual = rootShape.createVisualAspect()
    rootShape.createCollisionAspect()
    rootVisual.setColor([0, 0, 0])
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
        poleJoint.setForceUpperLimit(0, 100.0)
        poleJoint.setForceLowerLimit(0, -100.0)
        poleJoint.setVelocityUpperLimit(0, 10000.0)
        poleJoint.setVelocityLowerLimit(0, -10000.0)

        poleOffset = dart.math.Isometry3()
        poleOffset.set_translation([0, -0.125, 0])
        poleJoint.setTransformFromChildBodyNode(poleOffset)

        poleJoint.setPosition(0, 90 * 3.1415 / 180)

        poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.05, 0.25, .05]))
        poleVisual = poleShape.createVisualAspect()
        poleVisual.setColor(color)
        poleShape.createCollisionAspect()

        if parent != root:
            childOffset = dart.math.Isometry3()
            childOffset.set_translation([0, 0.125, 0])
            poleJoint.setTransformFromParentBodyNode(childOffset)
        return pole

    """
/* Color Theme Swatches in Hex */
.April-Picnic-1-hex { color: #323743; }
.April-Picnic-2-hex { color: #B6E091; }
.April-Picnic-3-hex { color: #DFE4A3; }
.April-Picnic-4-hex { color: #DEC179; }
.April-Picnic-5-hex { color: #E3894F; }

/* Color Theme Swatches in RGBA */
.April-Picnic-1-rgba { color: rgba(49, 54, 66, 1); }
.April-Picnic-2-rgba { color: rgba(182, 223, 144, 1); }
.April-Picnic-3-rgba { color: rgba(223, 228, 163, 1); }
.April-Picnic-4-rgba { color: rgba(221, 193, 121, 1); }
.April-Picnic-5-rgba { color: rgba(226, 137, 79, 1); }

/* Color Theme Swatches in HSLA */
.April-Picnic-1-hsla { color: hsla(222, 14, 22, 1); }
.April-Picnic-2-hsla { color: hsla(91, 56, 72, 1); }
.April-Picnic-3-hsla { color: hsla(64, 54, 76, 1); }
.April-Picnic-4-hsla { color: hsla(42, 60, 67, 1); }
.April-Picnic-5-hsla { color: hsla(23, 72, 60, 1); }
    """
    tail1 = createTailSegment(root, [182.0/255, 223.0/255, 144.0/255])
    tail2 = createTailSegment(tail1, [223.0/255, 228.0/255, 163.0/255])
    tail3 = createTailSegment(tail2, [221.0/255, 193.0/255, 121.0/255])
    # tail4 = createTailSegment(tail3, [226.0/255, 137.0/255, 79.0/255])

    jumpworm.setPositions(np.array([0, 0, 90, 90, 45]) * 3.1415 / 180)

    world.addSkeleton(jumpworm)

    # Floor

    floor = dart.dynamics.Skeleton()

    floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
    floorOffset = dart.math.Isometry3()
    floorOffset.set_translation([0, -0.6, 0])
    floorJoint.setTransformFromParentBodyNode(floorOffset)

    floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([2.5, 0.25, .5]))
    floorVisual = floorShape.createVisualAspect()
    floorShape.createCollisionAspect()

    floorBody.setFrictionCoeff(0)

    world.addSkeleton(floor)

    # Set up the view

    node = MyWorldNode(world)
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([0, 0, 5.0], [0, 0, 0], [0, 0.5, 0])
    viewer.realize()

    positions = torch.zeros(world.getNumDofs(), requires_grad=True)

    optimizer = torch.optim.SGD([positions], lr=0.05)

    dt = 1.0 / 50
    clock = 0
    while True:
        optimizer.zero_grad()
        last_segment_pos = dart_torch.convert_to_world_space_positions_linear(
            world, tail3, positions)

        diff = last_segment_pos[0] - math.sin(clock) * 2
        l = diff * diff + (last_segment_pos[1] * last_segment_pos[1])

        l.backward()

        optimizer.step()

        world.setPositions(positions.detach().numpy())
        viewer.frame()
        time.sleep(dt)
        clock += dt


if __name__ == "__main__":
    main()
