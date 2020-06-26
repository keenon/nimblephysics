import unittest
from context import dart_torch
import dartpy as dart
import numpy as np
import torch

##########################################################################
# Utils
##########################################################################


def addFreeCube(world: dart.simulation.World, startPos: np.ndarray):
    skel = dart.dynamics.Skeleton()
    joint, body = skel.createTranslationalJointAndBodyNodePair()
    shapeNode = body.createShapeNode(dart.dynamics.BoxShape([.1, .1, .1]))
    shapeNode.createCollisionAspect()
    visual = shapeNode.createVisualAspect()
    visual.setColor([255, 0, 255])
    skel.setPositions(startPos)
    world.addSkeleton(skel)


def createCubeTrajectory(
        steps: int, shooting_length: int, tune_starting_point: bool, enforce_loop: bool = False):
    world = dart.simulation.World()
    world.setGravity([0, -9.81, 0])
    addFreeCube(world, [0, 0, 0])

    def step_loss(pos: torch.Tensor, vel: torch.Tensor, t: torch.Tensor,
                  world: dart.simulation.World):
        return t.norm()

    def final_loss(pos: torch.Tensor, vel: torch.Tensor, world: dart.simulation.World):
        return (pos - np.array([3, 3, 3])).norm()

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, step_loss, final_loss, steps=steps, shooting_length=shooting_length,
        tune_starting_point=tune_starting_point, enforce_loop=enforce_loop)

    return trajectory, world

##########################################################################
# Main tests
##########################################################################


class TestMultipleShootingTrajectory(unittest.TestCase):
    def test_one_step(self):
        trajectory, world = createCubeTrajectory(
            steps=1, shooting_length=1, tune_starting_point=False)

        # This should only be the first torques
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 1 * world.getNumDofs())

        trajectory.torques[0].data = torch.from_numpy(np.array([3, 2, 1]))

        # This should just be the first torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array([3, 2, 1])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques[0].data.numpy() == np.array([3, 2, 1])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([None], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([None], knot_g_offsets)

    def test_one_step_with_starting_point(self):
        trajectory, world = createCubeTrajectory(
            steps=1, shooting_length=1, tune_starting_point=True)

        # This should be the first torques + first pos + first vel
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 3 * world.getNumDofs())

        trajectory.torques[0].data = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses[0].data = torch.from_numpy(np.array([4, 5, 6]))
        trajectory.knot_vels[0].data = torch.from_numpy(np.array([7, 8, 9]))

        # This should just be the first torque + first pos + first vel
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array([4, 5, 6, 7, 8, 9, 3, 2, 1])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques[0].data.numpy() == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses[0].data.numpy() == np.array([4, 5, 6])).all())
        self.assertTrue((trajectory.knot_vels[0].data.numpy() == np.array([7, 8, 9])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([0], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([0], knot_g_offsets)

    def test_two_step_with_no_knots(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=2, tune_starting_point=False)

        # This should be the first and second torques
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 2 * world.getNumDofs())

        trajectory.torques[0].data = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.torques[1].data = torch.from_numpy(np.array([4, 5, 6]))

        # This should just be the first torque + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array([3, 2, 1, 4, 5, 6])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques[0].data.numpy() == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.torques[1].data.numpy() == np.array([4, 5, 6])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([None], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([None], knot_g_offsets)

    def test_two_step_with_starting_point(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=2, tune_starting_point=True)

        # This should be the first and second torques + first pos + first vel
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 4 * world.getNumDofs())

        trajectory.torques[0].data = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses[0].data = torch.from_numpy(np.array([4, 5, 6]))
        trajectory.knot_vels[0].data = torch.from_numpy(np.array([7, 8, 9]))
        trajectory.torques[1].data = torch.from_numpy(np.array([10, 11, 12]))

        # This should be the first pos + first vel + first torque + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array([4, 5, 6, 7, 8, 9, 3, 2, 1, 10, 11, 12])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques[0].data.numpy() == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses[0].data.numpy() == np.array([4, 5, 6])).all())
        self.assertTrue((trajectory.knot_vels[0].data.numpy() == np.array([7, 8, 9])).all())
        self.assertTrue((trajectory.torques[1].data.numpy() == np.array([10, 11, 12])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([0], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([0], knot_g_offsets)

    def test_two_step_with_two_knots(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=1, tune_starting_point=True)

        # This should be the two full knots (torque + pos + vel)
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 6 * world.getNumDofs())

        trajectory.knot_poses[0].data = torch.from_numpy(np.array([4, 5, 6]))
        trajectory.knot_vels[0].data = torch.from_numpy(np.array([7, 8, 9]))
        trajectory.torques[0].data = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses[1].data = torch.from_numpy(np.array([14, 15, 16]))
        trajectory.knot_vels[1].data = torch.from_numpy(np.array([17, 18, 19]))
        trajectory.torques[1].data = torch.from_numpy(np.array([10, 11, 12]))

        # This should be the first pos + first vel + first torque + second pos + second vel + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array(
            [4, 5, 6, 7, 8, 9, 3, 2, 1, 14, 15, 16, 17, 18, 19, 10, 11, 12])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.knot_poses[0].data.numpy() == np.array([4, 5, 6])).all())
        self.assertTrue((trajectory.knot_vels[0].data.numpy() == np.array([7, 8, 9])).all())
        self.assertTrue((trajectory.torques[0].data.numpy() == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses[1].data.numpy() == np.array([14, 15, 16])).all())
        self.assertTrue((trajectory.knot_vels[1].data.numpy() == np.array([17, 18, 19])).all())
        self.assertTrue((trajectory.torques[1].data.numpy() == np.array([10, 11, 12])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([0, 3 * world.getNumDofs()], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([0, 2 * world.getNumDofs()], knot_g_offsets)

    def test_two_step_with_one_knot_no_start(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=1, tune_starting_point=False)

        # This should be the first torque + one full knot (torque + pos + vel)
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 4 * world.getNumDofs())

        trajectory.torques[0].data = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses[1].data = torch.from_numpy(np.array([14, 15, 16]))
        trajectory.knot_vels[1].data = torch.from_numpy(np.array([17, 18, 19]))
        trajectory.torques[1].data = torch.from_numpy(np.array([10, 11, 12]))

        # This should be the first torque + second pos + second vel + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array(
            [3, 2, 1, 14, 15, 16, 17, 18, 19, 10, 11, 12])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques[0].data.numpy() == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses[1].data.numpy() == np.array([14, 15, 16])).all())
        self.assertTrue((trajectory.knot_vels[1].data.numpy() == np.array([17, 18, 19])).all())
        self.assertTrue((trajectory.torques[1].data.numpy() == np.array([10, 11, 12])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([None, 1 * world.getNumDofs()], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([None, 0], knot_g_offsets)

    def test_two_step_with_two_knots_and_loop(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=1, tune_starting_point=True, enforce_loop=True)

        # TODO: test

    def test_full_opt(self):
        trajectory, world = createCubeTrajectory(
            steps=10, shooting_length=5, tune_starting_point=True)
        trajectory.ipopt()


if __name__ == '__main__':
    unittest.main()
