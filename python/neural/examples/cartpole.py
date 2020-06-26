import numpy as np
import dartpy as dart
import torch
import torch.nn.functional as F
from dart_torch import dart_layer, DartTimestepLearnTorque
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

    # Set up the 2D cartpole

    cartpole = dart.dynamics.Skeleton()
    cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
    cartRail.setAxis([1, 0, 0])
    cartShape = cart.createShapeNode(dart.dynamics.BoxShape([.5, .1, .1]))
    cartVisual = cartShape.createVisualAspect()
    cartVisual.setColor([0, 0, 0])

    poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
    poleJoint.setAxis([0, 0, 1])
    poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.1, 1.0, .1]))
    poleVisual = poleShape.createVisualAspect()
    poleVisual.setColor([0, 0, 0])

    poleOffset = dart.math.Isometry3()
    poleOffset.set_translation([0, -0.5, 0])
    poleJoint.setTransformFromChildBodyNode(poleOffset)

    world.addSkeleton(cartpole)

    # Set up the force marker, cause I can't figure out how to draw a line

    marker = dart.dynamics.Skeleton()
    markerWeld, markerBody = cartpole.createWeldJointAndBodyNodePair()
    markerShape = markerBody.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
    markerVisual = markerShape.createVisualAspect()
    markerVisual.setColor([255, 0, 0])

    markerOffset = dart.math.Isometry3()
    markerOffset.set_translation([0, -0.5, 0])
    markerWeld.setTransformFromChildBodyNode(markerOffset)

    world.addSkeleton(marker)

    # Set up the view

    node = MyWorldNode(world)
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([0, 0, 5.0], [0, 0, 0], [0, 0.5, 0])
    viewer.realize()

    # Make simulations repeatable
    random.seed(1234)

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(world.getTimeStep()*10)

    steps = 1000
    torque_steps = 1000

    # Set up initial conditions
    start_pos = torch.tensor(
        [random.uniform(-0.3, 0.3),
            random.uniform(-5, 5) * (3.141 / 180)],
        dtype=torch.float, requires_grad=False)
    start_vel = torch.tensor([0, 0], dtype=torch.float, requires_grad=False)

    # Initialize the learnable torques
    torques = [torch.tensor([0], dtype=torch.float, requires_grad=True)
               for _ in range(torque_steps)]
    optimizer = torch.optim.Adam(torques, lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    stepped = False

    # Run cartpole simulations
    iteration = 0
    while True:
        pos = start_pos
        vel = start_vel

        loss = torch.tensor([0], dtype=torch.float, requires_grad=True)
        decay_factor = 1.0

        decay = 1.0
        for i in range(steps):
            if i < torque_steps:
                t = torch.cat((torques[i], torch.zeros(1, dtype=torch.float)), 0)
                # t = torques[i]
            else:
                t = torch.zeros(2, dtype=torch.float)
            pos, vel = dart_layer(world, pos, vel, t)

            markerOffset.set_translation([-pos[0] - t[0], 0.1, 0])
            markerWeld.setTransformFromChildBodyNode(markerOffset)
            if iteration % 100 == 0:
                viewer.frame()
                time.sleep(0.003)

            # + pos[0]*pos[0] + vel[0]*vel[0]
            decay = decay * decay_factor
            if abs(pos[0]) > 1.0 or abs(pos[1]) > 90 * (3.141 / 180):
                loss = pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0]
                break
            else:
                # loss = loss + decay * (pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0])
                pass

        if i == steps-1:
            loss = pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0]

            # Zero the accumulated grad
        optimizer.zero_grad()
        """
        for i in range(steps):
            if torques[i].grad is not None:
                torques[i].grad.data.zero_()
        """

        # Run the backprop
        print('Iteration '+str(iteration)+' survived: '+str(i)+' loss: '+str(loss.item()
                                                                             )+': ['+str(torques[0][0])+']')
        loss.backward()

        if iteration < 399:
            scheduler.step()
        optimizer.step()

        """
        for i in range(steps):
            if torques[i].grad is not None:
                torques[i].data.sub_(torques[i].grad.data * (0.001 / math.sqrt(iteration + 1)))
        """

        iteration += 1

        """
        print(start_vel.grad)
        start_vel.grad = torch.zeros(start_vel.size())
        print('Got gradient:')
        print(start_vel.grad)
        print('New start vel:')
        start_vel.data.sub_(start_vel.grad.data * 0.25)
        """


if __name__ == "__main__":
    main()
