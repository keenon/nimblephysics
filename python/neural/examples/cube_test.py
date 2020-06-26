import numpy as np
import dartpy as dart
import torch
from dart_torch import dart_layer, DartTimestepLearnTorque


class MyWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world):
        super(MyWorldNode, self).__init__(world)

    def customPreStep(self):
        pass


def addFreeCube(world: dart.simulation.World, startPos: np.ndarray):
    skel = dart.dynamics.Skeleton()
    joint, body = skel.createTranslationalJointAndBodyNodePair()
    shapeNode = body.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
    shapeNode.createCollisionAspect()
    visual = shapeNode.createVisualAspect()
    visual.setColor([255, 0, 255])
    skel.setPositions(startPos)
    world.addSkeleton(skel)


def addMarker(world: dart.simulation.World, pos: np.ndarray):
    skel = dart.dynamics.Skeleton()
    joint, body = skel.createWeldJointAndBodyNodePair()
    offset = dart.math.Isometry3()
    offset.set_translation(pos)
    joint.setTransformFromParentBodyNode(offset)
    shapeNode = body.createShapeNode(dart.dynamics.BoxShape([.05, .05, .05]))
    visual = shapeNode.createVisualAspect()
    visual.setColor([255, 0, 0])
    world.addSkeleton(skel)


def main():
    world = dart.simulation.World()
    world.setGravity([0, -9.81, 0])

    addFreeCube(world, [0, 0, 0])

    # Set up the view

    node = MyWorldNode(world)
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([1.8, 1.8, 1.8], [0, 0, 0], [0, 0.5, 0])
    viewer.realize()

    # Set up initial conditions

    start_pos = torch.tensor([0, 0, 0], dtype=torch.float64)
    start_vel = torch.tensor([0, 0, 0], dtype=torch.float64, requires_grad=True)
    zero_torques = torch.tensor([0, 0, 0], dtype=torch.float64)

    goal_pos = torch.tensor([0, 0.8, 0.3], dtype=torch.float64)
    addMarker(world, goal_pos.numpy())

    while True:

        # Train the world to do something specific

        pos = start_pos
        vel = start_vel

        print('Running forward:')
        for i in range(1000):
            # gradcheck takes a tuple of tensors as input, check if your gradient
            # evaluated with these tensors are close enough to numerical
            # approximations and returns True if they all verify this condition.
            """
            torch.autograd.gradcheck(
                dart_layer, (world, pos, vel, zero_torques),
                eps=1e-6, atol=1e-4)
            """

            pos, vel = dart_layer(world, pos, vel, zero_torques)
            viewer.frame()

        print('Computing backprop:')
        print(start_vel.grad)
        if start_vel.grad is not None:
            start_vel.grad.data.zero_()
        pos.backward(pos - goal_pos)
        print('Got gradient:')
        print(start_vel.grad)
        print('New start vel:')
        start_vel.data.sub_(start_vel.grad.data * 0.25)


if __name__ == "__main__":
    main()
