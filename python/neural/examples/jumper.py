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
    world.setGravity([0, 0, -9.81])

    loader = dart.utils.DartLoader()
    loader.addPackageDirectory("main", "/home/keenon/Desktop/dev/dart/python/neural/examples/data/")
    skel: dart.dynamics.Skeleton = loader.parseSkeleton("package://main/human_one_leg_inv.urdf")

    world.addSkeleton(skel)

    # Set up the view

    node = MyWorldNode(world)
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([0, 5.0, 1.0], [0, 0, 1.0], [0, 0, 0.5])
    viewer.realize()

    # Make simulations repeatable
    random.seed(1234)

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(1.0 / 240)

    while True:
        viewer.frame()
        time.sleep(0.003)

    steps = 1000
    shooting_length = 10

    # Set up initial conditions
    start_pos = torch.tensor(
        [random.uniform(-0.3, 0.3),
            random.uniform(-15, 15) * (3.141 / 180)],
        dtype=torch.float, requires_grad=False)
    start_vel = torch.tensor([0, 0], dtype=torch.float, requires_grad=False)

    world.setPositions(start_pos)
    world.setVelocities(start_vel)

    # Create the trajectory
    def step_loss(pos, vel, t, world):
        return 0  # t.norm()

    def final_loss(pos, vel, world):
        return torch.norm(pos) + torch.norm(vel)

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, step_loss, final_loss, steps=steps, shooting_length=shooting_length,
        disable_actuators=[1],
        tune_starting_point=False, enforce_final_state=np.zeros(4))

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

    def animate_step(pos, vel, t):
        markerOffset.set_translation([-pos[0] - t[0], 0.1, 0])
        markerWeld.setTransformFromChildBodyNode(markerOffset)
        viewer.frame()
        time.sleep(0.003)

    # Run cartpole simulations
    """
    iteration = 0
    for i in range(200):
        # while True:

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

        optimizer.step()
        scheduler.step()

        iteration += 1
    """

    trajectory.ipopt()

    print('Optimization complete! Playing trajectories over and over...')
    while True:
        trajectory.unroll(use_knots=True, after_step=animate_step)
        trajectory.unroll(use_knots=False, after_step=animate_step)


if __name__ == "__main__":
    main()
