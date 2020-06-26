import numpy as np
import dartpy as dart
import torch
from dart_torch import dart_layer, DartTimestepLearnTorque


class MyWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world):
        super(MyWorldNode, self).__init__(world)

    def customPreStep(self):
        pass


def addFreeSphere(world: dart.simulation.World, startPos: np.ndarray, color: np.ndarray):
    skel = dart.dynamics.Skeleton()
    joint, body = skel.createTranslationalJoint2DAndBodyNodePair()
    body.setRestitutionCoeff(0.8)
    shapeNode = body.createShapeNode(dart.dynamics.SphereShape(.1))
    shapeNode.createCollisionAspect()
    visual = shapeNode.createVisualAspect()
    visual.setColor(color)
    skel.setPositions(startPos)
    world.addSkeleton(skel)


def addMarker(world: dart.simulation.World, pos: np.ndarray):
    skel = dart.dynamics.Skeleton()
    joint, body = skel.createWeldJointAndBodyNodePair()
    offset = dart.math.Isometry3()
    offset.set_translation(np.append(pos, [0]))
    joint.setTransformFromParentBodyNode(offset)
    shapeNode = body.createShapeNode(dart.dynamics.BoxShape([.05, .05, .05]))
    visual = shapeNode.createVisualAspect()
    visual.setColor([255, 0, 0])
    world.addSkeleton(skel)


def main():
    world = dart.simulation.World()
    world.setGravity([0, 0, 0])

    # Add our main character
    addFreeSphere(world, [0, 0], [255, 0, 255])

    """
    # Add blockers
    addFreeSphere(world, [1, 0], [0, 0, 255])
    addFreeSphere(world, [1.2, 0.15], [0, 0, 255])
    addFreeSphere(world, [1.2, -0.15], [0, 0, 255])
    addFreeSphere(world, [1.4, 0], [0, 0, 255])
    addFreeSphere(world, [1.4, -0.3], [0, 0, 255])
    """

    # Add our goal
    addFreeSphere(world, [1.4, 0.05], [0, 255, 0])

    # Set up the view

    node = MyWorldNode(world)
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([1, 0, 7.0], [1, 0, 0], [0, 0.5, 0])
    viewer.realize()

    # Set up initial conditions

    start_pos = torch.tensor(world.getPositions(), dtype=torch.float64)
    start_vel = torch.tensor(world.getVelocities(), dtype=torch.float64, requires_grad=True)
    zero_torques = torch.tensor(world.getForces(), dtype=torch.float64, requires_grad=False)

    # Initialize the ball to go hit the main group

    start_vel.data[0] = 2.0

    goal_pos = torch.tensor([3.0, 0.8], dtype=torch.float64)
    addMarker(world, goal_pos.numpy())

    BALL_DOFS = 2
    NUM_BLOCKERS = 0

    while True:

        # Train the world to do something specific

        pos = start_pos
        vel = start_vel

        print('Running forward:')
        positions = []
        for i in range(1000):
            pos, vel = dart_layer(world, pos, vel, zero_torques)
            viewer.frame()

        if start_vel.grad is not None:
            start_vel.grad.data.zero_()
        # print('Computing backprop:')
        # print(start_vel.grad)

        loss = (goal_pos - pos[-BALL_DOFS:]).norm()
        """
        loss_grad = torch.cat(
            (torch.zeros((NUM_BLOCKERS + 1) * BALL_DOFS, dtype=torch.float64),
             (goal_pos - pos.data[-BALL_DOFS:])),
            0)
        print('position loss grad')
        print(loss_grad)
        # pos.backward(loss_grad)
        """
        print('loss: '+str(loss))
        loss.backward()
        start_vel.grad.data[BALL_DOFS:] = 0
        # print('Got gradient:')
        # print(start_vel.grad)
        start_vel.data.sub_(start_vel.grad.data * 0.1)
        # print('New start vel:')
        # print(start_vel.data)


if __name__ == "__main__":
    main()
