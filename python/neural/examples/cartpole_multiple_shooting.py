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
    shooting_length = 50

    # Set up initial conditions
    start_pos = torch.tensor(
        [random.uniform(-0.3, 0.3),
            random.uniform(-15, 15) * (3.141 / 180)],
        dtype=torch.float, requires_grad=False)
    start_vel = torch.tensor([0, 0], dtype=torch.float, requires_grad=False)

    world.setPositions(start_pos)
    world.setPositions(start_vel)

    # Create the trajectory
    def step_loss(pos, vel, t, world):
        return 0

    def final_loss(pos, vel, world):
        return torch.norm(pos) + torch.norm(vel)

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, step_loss, final_loss, steps=steps,
        shooting_length=shooting_length, disable_actuators=[1])

    """
    # Initialize the learnable torques
    torques = [torch.tensor([0], dtype=torch.float, requires_grad=True)
               for _ in range(steps)]
    # Initialize knot points
    knot_point_vel = [torch.tensor([0, 0], dtype=torch.float, requires_grad=True)
                      for _ in range(num_shots)]
    knot_point_pos = [torch.tensor([0, 0], dtype=torch.float, requires_grad=True)
                      for _ in range(num_shots)]

    optimizer = torch.optim.Adam(torques + knot_point_vel + knot_point_pos, lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    stepped = False
    """

    optimizer = torch.optim.Adam(trajectory.tensors(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    knot_weight = 100

    # Run cartpole simulations
    iteration = 0
    while True:
        """
        pos = start_pos
        vel = start_vel

        loss = torch.tensor([0], dtype=torch.float, requires_grad=True)
        decay_factor = 1.0

        decay = 1.0
        for i in range(steps):
            t = torch.cat((torques[i], torch.zeros(1, dtype=torch.float)), 0)

            # We're at a knot point
            if i % shooting_length == 0 and i > 0:
                knot_index = math.floor(i / shooting_length) - 1
                knot_pos = knot_point_pos[knot_index]
                knot_vel = knot_point_vel[knot_index]

                # Record the loss from the error at this knot
                knot_loss = 100 * ((knot_pos - pos).norm() + (knot_vel - vel).norm())
                loss = loss + knot_loss

                pos = knot_pos
                vel = knot_vel

            pos, vel = dart_layer(world, pos, vel, t)

            markerOffset.set_translation([-pos[0] - t[0], 0.1, 0])
            markerWeld.setTransformFromChildBodyNode(markerOffset)
            if iteration % 100 == 0:
                viewer.frame()
                time.sleep(0.003)

            # + pos[0]*pos[0] + vel[0]*vel[0]
            decay = decay * decay_factor
            if abs(pos[0]) > 1.0 or abs(pos[1]) > 90 * (3.141 / 180):
                loss = loss + pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0]
                break
            else:
                # loss = loss + decay * (pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0])
                pass

        if i == steps-1:
            loss = loss + pos[1]*pos[1] + vel[1]*vel[1] + pos[0]*pos[0]
        """

        def animate_step(pos, vel, t):
            markerOffset.set_translation([-pos[0] - t[0], 0.1, 0])
            markerWeld.setTransformFromChildBodyNode(markerOffset)
            viewer.frame()
            time.sleep(0.003)

        loss, knot_loss = trajectory.unroll(
            after_step=(animate_step if iteration % 100 == 0 and iteration > 0 else None))

        # Show a trajectory without the knot points
        if iteration % 100 == 0 and iteration > 0:
            trajectory.unroll(use_knots=False, after_step=animate_step)
            knot_weight *= 10

        # Zero the accumulated grad
        optimizer.zero_grad()

        # Run the backprop
        print('Iteration '+str(iteration)+' loss: '+str(loss.item())+', knot loss: '+str(knot_loss.item()))

        l = loss + (knot_loss * iteration)

        l.backward()

        scheduler.step()
        optimizer.step()

        iteration += 1


if __name__ == "__main__":
    main()
