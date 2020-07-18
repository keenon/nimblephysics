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

    nodes = skel.getBodyNodes()
    head = skel.getBodyNode(7)

    # Create the trajectory
    def eval_loss(tau, pos, vel, world):
        com_pos = dart_torch.convert_to_world_space_center_of_mass(
            world, head, pos[:, steps-1])
        com_vel = dart_torch.convert_to_world_space_center_of_mass_vel_linear(
            world, head, vel[:, steps-1])

        # Step Loss, as a soft constraint
        """
        com_dzs = com_vels[2, :]
        # Want z_t+1 - z_t >= -g * DT * shooting_length
        com_next_dzs = torch.roll(com_dzs, 1, 0)
        diff = com_next_dzs - com_dzs
        diff -= 9.81 * dt * shooting_length
        step_loss = torch.sum(torch.exp(diff))
        """

        # Final Loss
        com_z = com_pos[2]
        com_dz = com_vel[2]
        sign = com_dz.item() / abs(com_dz.item())
        final_loss = -100 * com_z  # + sign * com_dz * com_dz / (2 * 9.81))

        return final_loss

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, eval_loss, steps=steps, shooting_length=shooting_length,
        tune_starting_point=False)

    trajectory.create_gui()
    """
    while True:
        trajectory.display_trajectory()
        time.sleep(0.003)
    """

    trajectory.compute_hessian = False
    for i in range(10):
        trajectory.ipopt(20)
        trajectory.playback_trajectory()
        trajectory.display_trajectory()

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
        trajectory.playback_trajectory()


if __name__ == "__main__":
    main()
