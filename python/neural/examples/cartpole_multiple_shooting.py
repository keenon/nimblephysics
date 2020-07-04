from context import dart_torch
import numpy as np
import dartpy as dart
import torch
import torch.nn.functional as F
import random
import math
import time
import cProfile


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
    cartRail.setForceUpperLimit(0, 1.0)
    cartRail.setForceLowerLimit(0, -1.0)
    cartRail.setVelocityUpperLimit(0, 1000.0)
    cartRail.setVelocityLowerLimit(0, -1000.0)

    poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
    poleJoint.setAxis([0, 0, 1])
    poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.1, 1.0, .1]))
    poleVisual = poleShape.createVisualAspect()
    poleVisual.setColor([0, 0, 0])
    poleJoint.setForceUpperLimit(0, 0.0)
    poleJoint.setForceLowerLimit(0, 0.0)
    poleJoint.setVelocityUpperLimit(0, 10000.0)
    poleJoint.setVelocityLowerLimit(0, -10000.0)

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

    steps = 500
    shooting_length = 5

    # Set up initial conditions
    start_pos = torch.tensor(
        [random.uniform(-0.3, 0.3),
            random.uniform(-15, 15) * (3.141 / 180)],
        dtype=torch.float, requires_grad=False)
    start_vel = torch.tensor([0, 0], dtype=torch.float, requires_grad=False)

    world.setPositions(start_pos)
    world.setVelocities(start_vel)

    # Create the trajectory
    def eval_loss(t, pos, vel, world):
        # DOF x timestep
        step_loss = torch.sum(t[0, :]*t[0, :]) + torch.sum(t[1, :]*t[1, :])
        final_loss = torch.norm(pos[:, steps-1]) + torch.norm(vel[:, steps-1])
        return step_loss + final_loss

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, eval_loss, steps=steps, shooting_length=shooting_length,
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
    for i in range(201):
        loss, knot_loss = trajectory.parallel_unroll(compute_knot_loss=True)

        # Show a trajectory without the knot points
        if iteration % 100 == 0 and iteration > 0:
            trajectory.unroll(use_knots=True, after_step=animate_step)
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

    trajectory.ipopt(100)

    print('Optimization complete! Playing trajectories over and over...')
    while True:
        trajectory.unroll(use_knots=True, after_step=animate_step)
        trajectory.unroll(use_knots=False, after_step=animate_step)


if __name__ == "__main__":
    # cProfile.run('main()')
    main()
