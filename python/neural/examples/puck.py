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

    puck = dart.dynamics.Skeleton()

    rootJoint, root = puck.createTranslationalJointAndBodyNodePair()
    rootJoint.setForceUpperLimit(0, 100)
    rootJoint.setForceLowerLimit(0, -100)
    rootJoint.setForceUpperLimit(1, 100)
    rootJoint.setForceLowerLimit(1, -100)
    rootJoint.setForceUpperLimit(2, 100)
    rootJoint.setForceLowerLimit(2, -100)
    rootJoint.setVelocityUpperLimit(0, 1000.0)
    rootJoint.setVelocityLowerLimit(0, -1000.0)
    rootJoint.setVelocityUpperLimit(1, 1000.0)
    rootJoint.setVelocityLowerLimit(1, -1000.0)

    rootShape = root.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
    rootVisual = rootShape.createVisualAspect()
    rootShape.createCollisionAspect()
    rootVisual.setColor([0, 0, 0])

    world.addSkeleton(puck)

    # Floor

    floor = dart.dynamics.Skeleton()

    floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
    floorOffset = dart.math.Isometry3()
    floorOffset.set_translation([0, -0.125 - 0.05, 0])
    floorJoint.setTransformFromParentBodyNode(floorOffset)
    floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([2.5, 0.25, 2.5]))
    floorVisual = floorShape.createVisualAspect()
    floorShape.createCollisionAspect()
    floorBody.setFrictionCoeff(0)

    world.addSkeleton(floor)

    # Set up the view

    """
    node = MyWorldNode(world)
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([0, 0, 5.0], [0, 0, 0], [0, 0.5, 0])
    viewer.realize()
    """

    # Make simulations repeatable
    random.seed(1234)

    steps = 1000
    shooting_length = 10

    # Create the trajectory
    def eval_loss(t, pos, vel, world):
        # DOF x timestep
        step_loss = 0  # torch.sum(t[0, :]*t[0, :]) + torch.sum(t[1, :]*t[1, :])
        # world_vel = dart_torch.convert_to_world_space_velocities(world, vel)
        # world_pos = dart_torch.convert_to_world_space_positions(world, pos)

        last_segment_pos = pos[:, steps-1]
        """dart_torch.convert_to_world_space_positions_linear(
            world, root, pos[:, steps-1])"""
        last_segment_vel = vel[:, steps-1]
        """dart_torch.convert_to_world_space_velocities_linear(
            world, root, vel[:, steps-1])"""

        x_offset = 1 - last_segment_pos[0]
        z_offset = 1 - last_segment_pos[2]
        final_loss = (x_offset * x_offset) + (z_offset * z_offset) + (
            last_segment_vel[0] * last_segment_vel[0]) + (last_segment_vel[2] * last_segment_vel[2])
        return step_loss + final_loss

    """
    while True:
        viewer.frame()
        time.sleep(0.003)
    """

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, eval_loss, steps=steps, shooting_length=shooting_length,
        disable_actuators=[1],
        tune_starting_point=False)

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

    """
    optimizer = torch.optim.Adam(trajectory.tensors(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    knot_weight = 100

    def animate_step(pos, vel, t):
        viewer.frame()
        time.sleep(0.003)

    # Run cartpole simulations
    iteration = 0
    for i in range(201):
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

    """
    trajectory.create_gui()
    trajectory.ipopt(700)
    """

    trajectory.create_gui()
    for i in range(40):
        trajectory.ipopt(20)
        trajectory.display_trajectory()

    """
    for i in range(40):
        trajectory.display_trajectory()
    """

    print('Optimization complete! Playing best found trajectory '+str(trajectory.best_loss)+' over and over...')
    # trajectory.restore_best_loss()
    while True:
        trajectory.playback_trajectory()
        """
        trajectory.unroll(use_knots=True, after_step=animate_step)
        trajectory.unroll(use_knots=False, after_step=animate_step)
        """


if __name__ == "__main__":
    main()
