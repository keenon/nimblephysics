import numpy as np
import dartpy as dart
import torch
import torch.nn.functional as F
from dart_torch import dart_layer, DartTimestepLearnTorque
import random
import time


class MyWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world):
        super(MyWorldNode, self).__init__(world)

    def customPreStep(self):
        pass


class CartpoleController(torch.nn.Module):
    def __init__(self):
        super(CartpoleController, self).__init__()
        self.hidden = 20
        # an affine operation: y = Wx + b
        self.fc1 = torch.nn.Linear(4, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.atan(self.fc3(x) / self.hidden)
        return x * 10


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

    # Initialize the NN model
    controller = CartpoleController()
    controller = controller

    # controller = controller.float()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(world.getTimeStep()*10)

    steps = 1000

    iteration = 0

    # Run cartpole simulations
    while True:
        # Set up initial conditions
        pos = torch.tensor(
            [random.uniform(-0.2, 0.2),
             random.uniform(-10, 10) * (3.141 / 180)],
            dtype=torch.float64)
        vel = torch.tensor([0, 0], dtype=torch.float64)

        loss = torch.tensor([0], dtype=torch.float, requires_grad=True)

        for i in range(steps):
            phase = torch.cat((pos, vel), 0).detach().float()
            cart_control = controller(phase)
            t = torch.cat((cart_control, torch.zeros(1)), 0).double()
            # torques = torch.cat((slider_control, torch.zeros(1, dtype=torch.float64)), 0)
            pos, vel = dart_layer(world, pos, vel, t)

            markerOffset.set_translation([- pos[0] - cart_control, 0.1, 0])
            markerWeld.setTransformFromChildBodyNode(markerOffset)

            if iteration % 100 == 0:
                viewer.frame()
                time.sleep(0.003)

            # Very small penalty for using too much "throttle"
            # loss = loss + (cart_control * cart_control * 1e-6)

            if abs(pos[0]) > 0.7 or abs(pos[1]) > 15 * (3.141 / 180):
                loss = pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0]
                break

        if i == steps-1:
            loss = pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0]

        controller.zero_grad()

        # offsets = torch.stack(offsets, dim=0)
        # loss = offsets.sum()

        loss = pos.norm() + vel.norm()
        loss.backward()

        # Run the backprop
        print('Iteration '+str(iteration)+' survived: '+str(i)+' loss: '+str(loss.item()))

        if iteration < 1000:
            scheduler.step()
        optimizer.step()

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
