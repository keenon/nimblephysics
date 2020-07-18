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

    """
    world: dart.simulation.World = loader.parseWorld("package://main/human_one_leg_inv.urdf")
    world.setGravity([0, 0, -9.81])
    skel = world.getSkeleton(0)
    """

    ll = np.array([-0.05, -1.1, -0.05, -1.5])
    skel.setPositionLowerLimits(ll)
    ul = np.array([0.6, 1.1, 2.8, 2.2])
    skel.setPositionUpperLimits(ul)
    vb = np.array([20.0, 15, 15, 5])
    skel.setVelocityUpperLimits(vb)
    skel.setVelocityLowerLimits(-vb)
    max_forces = np.array([30, 250, 250, 200])
    skel.setForceUpperLimits(max_forces)
    skel.setForceLowerLimits(-max_forces)

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
    dt = 1.0 / 240
    world.setTimeStep(dt)

    steps = 240
    shooting_length = 10

    # Set up initial conditions
    start_pos = np.array([0.0, -48.158, 94.737, 77.895]) / 180 * 3.1415
    start_vel = np.zeros(skel.getNumDofs())

    world.setPositions(start_pos)
    world.setVelocities(start_vel)
    positions = torch.zeros(world.getNumDofs(), requires_grad=True)

    head = skel.getBodyNode(7)

    optimizer = torch.optim.SGD([positions], lr=0.05)

    dt = 1.0 / 50
    clock = 0
    while True:
        optimizer.zero_grad()
        """
        last_segment_pos = dart_torch.convert_to_world_space_positions_linear(
            world, head, positions)
        diff = last_segment_pos[0] - math.sin(clock) * 2
        l = diff * diff + (last_segment_pos[1] * last_segment_pos[1])
        """

        com_pos = dart_torch.convert_to_world_space_center_of_mass(
            world, head, positions)
        l = - com_pos[2] * com_pos[2]

        l.backward()

        optimizer.step()

        world.setPositions(positions.detach().numpy())
        viewer.frame()
        time.sleep(dt)
        clock += dt


if __name__ == "__main__":
    main()
