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

    jumpworm = dart.dynamics.Skeleton()

    rootJoint, root = jumpworm.createTranslationalJoint2DAndBodyNodePair()
    rootJoint.setXYPlane()
    rootShape = root.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
    rootVisual = rootShape.createVisualAspect()
    rootShape.createCollisionAspect()
    rootVisual.setColor([0, 0, 0])
    rootJoint.setForceUpperLimit(0, 0)
    rootJoint.setForceLowerLimit(0, 0)
    rootJoint.setForceUpperLimit(1, 0)
    rootJoint.setForceLowerLimit(1, 0)
    rootJoint.setVelocityUpperLimit(0, 1000.0)
    rootJoint.setVelocityLowerLimit(0, -1000.0)
    rootJoint.setVelocityUpperLimit(1, 1000.0)
    rootJoint.setVelocityLowerLimit(1, -1000.0)

    def createTailSegment(parent, color):
        poleJoint, pole = jumpworm.createRevoluteJointAndBodyNodePair(parent)
        poleJoint.setAxis([0, 0, 1])
        poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.05, 0.25, .05]))
        poleVisual = poleShape.createVisualAspect()
        poleVisual.setColor(color)
        poleJoint.setForceUpperLimit(0, 100.0)
        poleJoint.setForceLowerLimit(0, -100.0)
        poleJoint.setVelocityUpperLimit(0, 10000.0)
        poleJoint.setVelocityLowerLimit(0, -10000.0)

        poleOffset = dart.math.Isometry3()
        poleOffset.set_translation([0, -0.125, 0])
        poleJoint.setTransformFromChildBodyNode(poleOffset)

        poleJoint.setPosition(0, 90 * 3.1415 / 180)
        poleJoint.setPositionUpperLimit(0, 180 * 3.1415 / 180)
        poleJoint.setPositionLowerLimit(0, 0 * 3.1415 / 180)

        poleShape.createCollisionAspect()

        if parent != root:
            childOffset = dart.math.Isometry3()
            childOffset.set_translation([0, 0.125, 0])
            poleJoint.setTransformFromParentBodyNode(childOffset)
        return pole

    """
/* Color Theme Swatches in Hex */
.April-Picnic-1-hex { color: #323743; }
.April-Picnic-2-hex { color: #B6E091; }
.April-Picnic-3-hex { color: #DFE4A3; }
.April-Picnic-4-hex { color: #DEC179; }
.April-Picnic-5-hex { color: #E3894F; }

/* Color Theme Swatches in RGBA */
.April-Picnic-1-rgba { color: rgba(49, 54, 66, 1); }
.April-Picnic-2-rgba { color: rgba(182, 223, 144, 1); }
.April-Picnic-3-rgba { color: rgba(223, 228, 163, 1); }
.April-Picnic-4-rgba { color: rgba(221, 193, 121, 1); }
.April-Picnic-5-rgba { color: rgba(226, 137, 79, 1); }

/* Color Theme Swatches in HSLA */
.April-Picnic-1-hsla { color: hsla(222, 14, 22, 1); }
.April-Picnic-2-hsla { color: hsla(91, 56, 72, 1); }
.April-Picnic-3-hsla { color: hsla(64, 54, 76, 1); }
.April-Picnic-4-hsla { color: hsla(42, 60, 67, 1); }
.April-Picnic-5-hsla { color: hsla(23, 72, 60, 1); }
    """
    tail1 = createTailSegment(root, [182.0/255, 223.0/255, 144.0/255])
    tail2 = createTailSegment(tail1, [223.0/255, 228.0/255, 163.0/255])
    tail3 = createTailSegment(tail2, [221.0/255, 193.0/255, 121.0/255])
    # tail4 = createTailSegment(tail3, [226.0/255, 137.0/255, 79.0/255])

    jumpworm.setPositions(np.array([0, 0, 90, 90, 45]) * 3.1415 / 180)

    world.addSkeleton(jumpworm)

    # Floor

    floor = dart.dynamics.Skeleton()

    floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
    floorOffset = dart.math.Isometry3()
    floorOffset.set_translation([0, -0.7, 0])
    floorJoint.setTransformFromParentBodyNode(floorOffset)
    floorShape = floorBody.createShapeNode(dart.dynamics.BoxShape([2.5, 0.25, .5]))
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

    steps = 400
    shooting_length = 20

    # Create the trajectory
    def eval_loss(t, pos, vel, world):
        # DOF x timestep
        # step_loss = 0  # torch.sum(t[0, :]*t[0, :]) + torch.sum(t[1, :]*t[1, :])
        # world_vel = dart_torch.convert_to_world_space_velocities(world, vel)
        # world_pos = dart_torch.convert_to_world_space_positions(world, pos)
        """
        root_poses = dart_torch.convert_to_world_space_positions_linear(
            world, root, pos)
        """
        step_loss = - torch.sum(pos[1, :] * pos[1, :] * torch.sign(pos[1, :]))
        """
        return loss
        """
        last_segment_pos = pos[-1, :]
        final_loss = - 100 * last_segment_pos[1] * \
            last_segment_pos[1] * torch.sign(last_segment_pos[1])
        return step_loss + final_loss

    """
    while True:
        viewer.frame()
        time.sleep(0.003)
    """

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, eval_loss, steps=steps, shooting_length=shooting_length,
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

    optimizer = torch.optim.SGD(trajectory.tensors(), lr=1e3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    knot_weight = 100

    """
    print('errors:')
    dims = [252, 253, 254, 257]
    for dim in dims:
        print(str(dim)+': '+trajectory.get_flat_dim_name(dim))
    return
    """

    trajectory.create_gui()

    # Run cartpole simulations
    """
    iteration = 0
    for i in range(401):
        loss, knot_loss = trajectory.unroll(use_knots=False)

        # Show a trajectory without the knot points
        if iteration % 100 == 0 and iteration > 0:
            trajectory.playback_trajectory()

        # Zero the accumulated grad
        optimizer.zero_grad()

        # Run the backprop
        print('Iteration '+str(iteration)+' loss: '+str(loss.item()))

        loss.backward()

        trajectory.postprocess_grad()

        optimizer.step()
        scheduler.step()

        trajectory.enforce_limits()

        if i % 10 == 0:
            trajectory.playback_trajectory()
        # time.sleep(1)

        iteration += 1
    """

    trajectory.compute_hessian = False

    trajectory.ipopt(300)

    """
    # trajectory.display_trajectory()
    for i in range(10):
        trajectory.ipopt(10)
        trajectory.playback_trajectory()
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
