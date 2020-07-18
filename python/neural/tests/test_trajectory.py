import unittest
from context import dart_torch
import dartpy as dart
import numpy as np
import torch
import os
import random

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
        steps: int, shooting_length: int, tune_starting_point: bool, enforce_loop: bool = False,
        enforce_final_state: np.ndarray = None):
    world = dart.simulation.World()
    # world.setGravity([0, -9.81, 0])
    world.setGravity([0, 0, 0])
    addFreeCube(world, [0, 0, 0])

    def eval_loss(t, pos, vel, world):
        # DOF x timestep
        step_loss = 0  # torch.sum(t[0, :]*t[0, :]) + torch.sum(t[1, :]*t[1, :])
        # final_loss = torch.norm(vel[:, steps-1])  # torch.norm(pos[:, steps-1]) +
        """
        final_loss = torch.sum(pos[:, steps-1] * pos[:, steps-1]
                               ) + torch.sum(vel[:, steps-1] * vel[:, steps-1])
        """
        final_loss = pos[0, steps-1] * pos[0, steps-1] + vel[0, steps-1] * vel[0, steps-1]
        # final_loss = torch.sum(vel[:, steps-1] * vel[:, steps-1])
        # final_loss = vel[2, steps-1]*vel[2, steps-1]
        """
        final_loss = pos[0, steps-1]*pos[0, steps-1]
        final_loss = pos[0, steps-1]*pos[0, steps-1] + pos[1,
                                                           steps-1]*pos[1, steps-1] + pos[2, steps-1]*pos[2, steps-1]
        final_loss = vel[0, steps-1]*vel[0, steps-1] + vel[1,
                                                           steps-1]*vel[1, steps-1] + vel[2, steps-1]*vel[2, steps-1]
        """
        return step_loss + final_loss

    world.setPositions(np.array([3, 3, 3]))

    trajectory = dart_torch.MultipleShootingTrajectory(
        world, eval_loss, steps=steps, shooting_length=shooting_length,
        tune_starting_point=tune_starting_point, enforce_loop=enforce_loop,
        enforce_final_state=enforce_final_state)

    trajectory.knot_poses.data[:, 1] = torch.from_numpy(np.array([1.5, 1.5, 1.5]))

    return trajectory, world


def createCartpoleTrajectory():
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

    # Make simulations repeatable
    random.seed(1234)

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(world.getTimeStep()*10)

    steps = 10
    shooting_length = 5

    # Set up initial conditions
    start_pos = torch.tensor(
        [-0.25, 10 * (3.141 / 180)],
        dtype=torch.float64, requires_grad=False)
    start_vel = torch.tensor([0, 0], dtype=torch.float64, requires_grad=False)

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
        # disable_actuators=[1],
        tune_starting_point=False, enforce_final_state=np.zeros(4))

    return trajectory, world

##########################################################################
# Main tests
##########################################################################


class TestMultipleShootingTrajectory(unittest.TestCase):
    """
    def test_one_step(self):
        trajectory, world = createCubeTrajectory(
            steps=1, shooting_length=1, tune_starting_point=False)

        # This should only be the first torques
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 1 * world.getNumDofs())

        trajectory.torques.data[:, 0] = torch.from_numpy(np.array([3, 2, 1]))

        # This should just be the first torque
        flattened = trajectory.flatten(np.zeros(dim), 'state')
        self.assertTrue((flattened == np.array([3, 2, 1])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques.data.numpy()[:, 0] == np.array([3, 2, 1])).all())

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

        trajectory.torques.data[:, 0] = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses.data[:, 0] = torch.from_numpy(np.array([4, 5, 6]))
        trajectory.knot_vels.data[:, 0] = torch.from_numpy(np.array([7, 8, 9]))

        # This should just be the first torque + first pos + first vel
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array([4, 5, 6, 7, 8, 9, 3, 2, 1])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques.data.numpy()[:, 0] == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses.data.numpy()[:, 0] == np.array([4, 5, 6])).all())
        self.assertTrue((trajectory.knot_vels.data.numpy()[:, 0] == np.array([7, 8, 9])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([0], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([None], knot_g_offsets)

    def test_two_step_with_no_knots(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=2, tune_starting_point=False)

        # This should be the first and second torques
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 2 * world.getNumDofs())

        trajectory.torques.data[:, 0] = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.torques.data[:, 1] = torch.from_numpy(np.array([4, 5, 6]))

        # This should just be the first torque + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array([3, 2, 1, 4, 5, 6])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques.data.numpy()[:, 0] == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.torques.data.numpy()[:, 1] == np.array([4, 5, 6])).all())

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

        trajectory.torques.data[:, 0] = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses.data[:, 0] = torch.from_numpy(np.array([4, 5, 6]))
        trajectory.knot_vels.data[:, 0] = torch.from_numpy(np.array([7, 8, 9]))
        trajectory.torques.data[:, 1] = torch.from_numpy(np.array([10, 11, 12]))

        # This should be the first pos + first vel + first torque + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array([4, 5, 6, 7, 8, 9, 3, 2, 1, 10, 11, 12])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques.data.numpy()[:, 0] == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses.data.numpy()[:, 0] == np.array([4, 5, 6])).all())
        self.assertTrue((trajectory.knot_vels.data.numpy()[:, 0] == np.array([7, 8, 9])).all())
        self.assertTrue((trajectory.torques.data.numpy()[:, 1] == np.array([10, 11, 12])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([0], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([None], knot_g_offsets)

    def test_two_step_with_two_knots(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=1, tune_starting_point=True)

        # This should be the two full knots (torque + pos + vel)
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 6 * world.getNumDofs())

        trajectory.knot_poses.data[:, 0] = torch.from_numpy(np.array([4, 5, 6]))
        trajectory.knot_vels.data[:, 0] = torch.from_numpy(np.array([7, 8, 9]))
        trajectory.torques.data[:, 0] = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses.data[:, 1] = torch.from_numpy(np.array([14, 15, 16]))
        trajectory.knot_vels.data[:, 1] = torch.from_numpy(np.array([17, 18, 19]))
        trajectory.torques.data[:, 1] = torch.from_numpy(np.array([10, 11, 12]))

        # This should be the first pos + first vel + first torque + second pos + second vel + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array(
            [4, 5, 6, 7, 8, 9, 3, 2, 1, 14, 15, 16, 17, 18, 19, 10, 11, 12])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.knot_poses.data.numpy()[:, 0] == np.array([4, 5, 6])).all())
        self.assertTrue((trajectory.knot_vels.data.numpy()[:, 0] == np.array([7, 8, 9])).all())
        self.assertTrue((trajectory.torques.data.numpy()[:, 0] == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses.data.numpy()[:, 1] == np.array([14, 15, 16])).all())
        self.assertTrue((trajectory.knot_vels.data.numpy()[:, 1] == np.array([17, 18, 19])).all())
        self.assertTrue((trajectory.torques.data.numpy()[:, 1] == np.array([10, 11, 12])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([0, 3 * world.getNumDofs()], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([None, 0], knot_g_offsets)

    def test_two_step_with_one_knot_no_start(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=1, tune_starting_point=False)

        # This should be the first torque + one full knot (torque + pos + vel)
        dim = trajectory.get_flat_problem_dim()
        self.assertEqual(dim, 4 * world.getNumDofs())

        trajectory.torques.data[:, 0] = torch.from_numpy(np.array([3, 2, 1]))
        trajectory.knot_poses.data[:, 1] = torch.from_numpy(np.array([14, 15, 16]))
        trajectory.knot_vels.data[:, 1] = torch.from_numpy(np.array([17, 18, 19]))
        trajectory.torques.data[:, 1] = torch.from_numpy(np.array([10, 11, 12]))

        # This should be the first torque + second pos + second vel + second torque
        flattened = trajectory.flatten(np.zeros(dim))
        self.assertTrue((flattened == np.array(
            [3, 2, 1, 14, 15, 16, 17, 18, 19, 10, 11, 12])).all())

        # Test reflating recovery
        for tensor in trajectory.tensors():
            tensor.data.zero_()
        trajectory.unflatten(flattened)
        self.assertTrue((trajectory.torques.data.numpy()[:, 0] == np.array([3, 2, 1])).all())
        self.assertTrue((trajectory.knot_poses.data.numpy()[:, 1] == np.array([14, 15, 16])).all())
        self.assertTrue((trajectory.knot_vels.data.numpy()[:, 1] == np.array([17, 18, 19])).all())
        self.assertTrue((trajectory.torques.data.numpy()[:, 1] == np.array([10, 11, 12])).all())

        # Test offsets for computing Jac g(x)
        knot_x_offsets = trajectory.get_knot_x_offsets()
        self.assertEqual([None, 1 * world.getNumDofs()], knot_x_offsets)
        knot_g_offsets = trajectory.get_knot_g_offsets()
        self.assertEqual([None, 0], knot_g_offsets)

    def test_two_step_with_two_knots_and_loop(self):
        trajectory, world = createCubeTrajectory(
            steps=2, shooting_length=1, tune_starting_point=True, enforce_loop=True)

        # TODO: test

    def test_unflatten(self):
        trajectory, world = createCubeTrajectory(
            steps=4, shooting_length=2, tune_starting_point=True, enforce_loop=False)
        x0 = trajectory.flatten(np.zeros(trajectory.get_flat_problem_dim()))
        x0[7] = 1.0  # this should be the second axis of the first torque
        trajectory.unflatten(x0)
        self.assertEqual(world.getNumDofs(), trajectory.torques.shape[0])
        torques = trajectory.torques.detach().numpy()[:, 0]
        self.assertTrue((np.array([0, 1, 0]) == torques).all())
    """

    """
    def test_box_jac_parallel(self):
        trajectory, world = createCubeTrajectory(
            steps=6, shooting_length=3, tune_starting_point=False, enforce_loop=False,
            enforce_final_state=np.zeros(6))
        x0 = trajectory.flatten(np.zeros(trajectory.get_flat_problem_dim()))

        # Turn on multithreading
        trajectory.multithread = True

        # Check the Jacobian
        dense_jac_g_par = trajectory._test_get_dense_jac_g(x0)
        grad_f_par = trajectory.eval_grad_f(x0, np.zeros(len(x0)))

        # Force a recomputation single threaded
        trajectory.last_x[0] = -1000
        trajectory.multithread = False

        dense_jac_g_serial = trajectory._test_get_dense_jac_g(x0)
        grad_f_serial = trajectory.eval_grad_f(x0, np.zeros(len(x0)))

        grad_equals = np.abs(grad_f_par, grad_f_serial) < 1e-7
        if not grad_equals.all():
            print('grad f(x) par: '+str(grad_f_par))
            print('grad f(x) serial: '+str(grad_f_serial))

        jac_equals = np.abs(dense_jac_g_par - dense_jac_g_serial) < 1e-7

        if not jac_equals.all():
            print('jac g(x) par: '+str(dense_jac_g_par))
            print('jac g(x) serial: '+str(dense_jac_g_serial))
            os.remove('./parallel.csv')
            os.remove('./serial.csv')

            np.savetxt("parallel.csv", dense_jac_g_par, delimiter=",")
            np.savetxt("serial.csv", dense_jac_g_serial, delimiter=",")
        self.assertTrue(jac_equals.all())
    """

    """
    def test_box_grad(self):
        trajectory, world = createCubeTrajectory(
            steps=6, shooting_length=3, tune_starting_point=False, enforce_loop=False,
            enforce_final_state=np.zeros(6))
        x0 = trajectory.flatten(np.zeros(trajectory.get_flat_problem_dim()))

        n = trajectory.get_flat_problem_dim()
        m = trajectory.get_constraint_dim()

        grad_f = trajectory.eval_grad_f(x0, np.zeros(n))
        brute_grad_f = trajectory._eval_brute_force_grad_f(x0, np.zeros(n))
        grad_l = trajectory._eval_grad_lagrange(x0, np.zeros(m), 1.0)

        grad_equals = np.abs(grad_f - brute_grad_f) < 1e-7
        if not grad_equals.all():
            print('grad f(x):\n'+str(grad_f))
            print('grad l(x):\n'+str(grad_l))
            print('brute grad f(x):\n'+str(brute_grad_f))
        self.assertTrue(grad_equals.all())
    """

    def test_box_h(self):
        trajectory, world = createCubeTrajectory(
            steps=6, shooting_length=3, tune_starting_point=False, enforce_loop=False,
            enforce_final_state=np.zeros(6))
        x0 = trajectory.flatten(np.zeros(trajectory.get_flat_problem_dim()))

        # Turn on multithreading
        trajectory.multithread = False

        n = trajectory.get_flat_problem_dim()
        m = trajectory.get_constraint_dim()

        # Test the Hessian for just the objective function

        # raw_h = trajectory._eval_h_lagrange(x0, np.zeros(m), 1.0)
        dense_h = trajectory._test_brute_force_h(x0, np.zeros(m), 1.0)
        main_h = trajectory._test_get_dense_h(x0, np.zeros(m), 1.0)

        threshold = 1e-7

        hess_equals = np.abs(main_h - dense_h) < threshold

        if not hess_equals.all():

            """
            # Check the individual columns one by one to see where a mismatch happened.

            grad_f = trajectory.eval_grad_f(x0, np.zeros(n))
            grad_l = trajectory._eval_grad_lagrange(x0, np.zeros(m), 1.0)
            grad_equals = np.abs(grad_f - grad_l) < threshold
            if not grad_equals.all():
                print('grad f(x) vs l(x):')
                trajectory.debug_flat_compare([grad_f, grad_l], 0)
            self.assertTrue(grad_equals.all())
            eps = 1e-8
            for i in range(n):
                x_prime = x0.copy()
                x_prime[i] += eps
                grad_f_prime = trajectory.eval_grad_f(x_prime, np.zeros(n))
                grad_l_prime = trajectory._eval_grad_lagrange(x_prime, np.zeros(m), 1.0)
                grad_prime_equals = np.abs(grad_f_prime - grad_l_prime) < threshold
                if not grad_prime_equals.all():
                    print('grad '+str(i)+' f(x) vs l(x):')
                    trajectory.debug_flat_compare([grad_f_prime, grad_l_prime], threshold)

            # Print the full matrix

            for i in range(n):
                raw_col = raw_h[:, i]
                dense_col = dense_h[:, i]
                col_equals = np.abs(raw_col - dense_col) < threshold
                if not col_equals.all():
                    print('col '+str(i)+' fd H(x) vs analytic H(x):')
                    trajectory.debug_flat_compare([raw_col, dense_col], threshold)

            # print('raw f(x) hess: \n'+str(raw_h))
            # print('dense f(x) hess: \n'+str(dense_h))
            # os.remove('./raw_hess.csv')
            """
            os.remove('./main_hess.csv')
            os.remove('./dense_hess.csv')

            np.savetxt("main_hess.csv", main_h, delimiter=",")
            np.savetxt("dense_hess.csv", dense_h, delimiter=",")
        self.assertTrue(hess_equals.all())

        # Test the Hessian for each of the individual constraints

        for i in range(m):
            print('Testing lagrange hessian '+str(i)+'/'+str(m))
            lagrange = np.zeros(m)
            lagrange[i] = 1.0
            # raw_h = trajectory._eval_h_lagrange(x0, lagrange, obj_factor)
            dense_h = trajectory._test_brute_force_h(x0, lagrange, 0.0)
            main_h = trajectory._test_get_dense_h(x0, np.zeros(m), 0.0)

            hess_equals = np.abs(main_h - dense_h) < threshold

            if not hess_equals.all():
                os.remove('./main_hess.csv')
                os.remove('./dense_hess.csv')

                np.savetxt("main_hess.csv", main_h, delimiter=",")
                np.savetxt("dense_hess.csv", dense_h, delimiter=",")
            self.assertTrue(hess_equals.all())

    """
    def test_box_jac(self):
        trajectory, world = createCubeTrajectory(
            steps=10, shooting_length=2, tune_starting_point=False, enforce_loop=False,
            enforce_final_state=np.zeros(6))
        x0 = trajectory.flatten(np.zeros(trajectory.get_flat_problem_dim()))
        l = trajectory.eval_f(x0)
        g = trajectory.eval_g(x0, np.zeros(trajectory.get_constraint_dim()))
        grad_f = trajectory.eval_grad_f(x0, np.zeros(len(x0)))
        sparse_g = trajectory.eval_jac_g(x0, np.zeros(
            len(trajectory.get_jac_g_sparsity_indices()[0])))

        # Check the gradient

        if False:
            brute_force_grad_f = trajectory._eval_brute_force_grad_f(x0, np.zeros_like(x0))
            grad_diff = np.abs(grad_f - brute_force_grad_f)
            grad_equals = np.abs(grad_f - brute_force_grad_f) < 1e-10

            if not grad_equals.all():
                print('grad f(x): '+str(grad_f))
                print('brute force grad f(x): '+str(brute_force_grad_f))
                print('grad diff: '+str(grad_diff))
                print('grad equals: '+str(grad_equals))
                trajectory.debug()
            self.assertTrue(grad_equals.all())

        # Check the Jacobian

        dense_jac_g = trajectory._test_get_dense_jac_g(x0)
        brute_force_jac_g = trajectory._test_brute_force_jac_g(x0)
        jac_equals = np.abs(dense_jac_g - brute_force_jac_g) < 1e-7

        if not jac_equals.all():
            print('jac g(x): '+str(dense_jac_g))
            print('brute force jac g(x): '+str(brute_force_jac_g))
            # a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
            os.remove('./dense.csv')
            os.remove('./brute.csv')

            np.savetxt("dense.csv", dense_jac_g, delimiter=",")
            np.savetxt("brute.csv", brute_force_jac_g, delimiter=",")
        self.assertTrue(jac_equals.all())

    def test_cartpole_jac(self):
        trajectory, world = createCartpoleTrajectory()
        x0 = trajectory.flatten(np.zeros(trajectory.get_flat_problem_dim()))
        l = trajectory.eval_f(x0)
        g = trajectory.eval_g(x0, np.zeros(trajectory.get_constraint_dim()))
        grad_f = trajectory.eval_grad_f(x0, np.zeros(len(x0)))
        sparse_g = trajectory.eval_jac_g(x0, np.zeros(
            len(trajectory.get_jac_g_sparsity_indices()[0])))

        # Check the gradient

        if False:
            brute_force_grad_f = trajectory._eval_brute_force_grad_f(x0, np.zeros_like(x0))
            grad_diff = np.abs(grad_f - brute_force_grad_f)
            grad_equals = np.abs(grad_f - brute_force_grad_f) < 1e-10

            if not grad_equals.all():
                print('grad f(x): '+str(grad_f))
                print('brute force grad f(x): '+str(brute_force_grad_f))
                print('grad diff: '+str(grad_diff))
                print('grad equals: '+str(grad_equals))
                trajectory.debug()
            self.assertTrue(grad_equals.all())

        # Check the Jacobian

        dense_jac_g = trajectory._test_get_dense_jac_g(x0)
        brute_force_jac_g = trajectory._test_brute_force_jac_g(x0)
        jac_equals = np.abs(dense_jac_g - brute_force_jac_g) < 1e-7

        if not jac_equals.all():
            print('jac g(x): '+str(dense_jac_g))
            print('brute force jac g(x): '+str(brute_force_jac_g))
            # a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
            os.remove('./dense.csv')
            os.remove('./brute.csv')

            np.savetxt("dense.csv", dense_jac_g, delimiter=",")
            np.savetxt("brute.csv", brute_force_jac_g, delimiter=",")
            np.savetxt(
                "diff.csv", (dense_jac_g - brute_force_jac_g) / (brute_force_jac_g + 1e-15),
                delimiter=",")
        self.assertTrue(jac_equals.all())
    """

    """
    def t_full_opt(self):
        trajectory, world = createCubeTrajectory(
            steps=10, shooting_length=2, tune_starting_point=False, enforce_loop=False,
            enforce_final_state=np.zeros(6))
        x0 = trajectory.flatten(np.zeros(trajectory.get_flat_problem_dim()))
        l = trajectory.eval_f(x0)
        g = trajectory.eval_g(x0, np.zeros(trajectory.get_constraint_dim()))
        grad_f = trajectory.eval_grad_f(x0, np.zeros(len(x0)))
        sparse_g = trajectory.eval_jac_g(x0, np.zeros(
            len(trajectory.get_jac_g_sparsity_indices()[0])))

        # Check the gradient

        if False:
            brute_force_grad_f = trajectory._eval_brute_force_grad_f(x0, np.zeros_like(x0))
            grad_diff = np.abs(grad_f - brute_force_grad_f)
            grad_equals = np.abs(grad_f - brute_force_grad_f) < 1e-10

            if not grad_equals.all():
                print('grad f(x): '+str(grad_f))
                print('brute force grad f(x): '+str(brute_force_grad_f))
                print('grad diff: '+str(grad_diff))
                print('grad equals: '+str(grad_equals))
                trajectory.debug()
            self.assertTrue(grad_equals.all())

        # Check the Jacobian

        dense_jac_g = trajectory._test_get_dense_jac_g(x0)
        brute_force_jac_g = trajectory._test_brute_force_jac_g(x0)
        jac_equals = np.abs(dense_jac_g - brute_force_jac_g) < 1e-7

        if not jac_equals.all():
            print('jac g(x): '+str(dense_jac_g))
            print('brute force jac g(x): '+str(brute_force_jac_g))
            # a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
            os.remove('./dense.csv')
            os.remove('./brute.csv')

            np.savetxt("dense.csv", dense_jac_g, delimiter=",")
            np.savetxt("brute.csv", brute_force_jac_g, delimiter=",")
        self.assertTrue(jac_equals.all())

        # Reset to 0s
        # trajectory.unflatten(x0)

        # trajectory.ipopt()
        # trajectory.debug()

    def test_cartpole_parallel(self):
        trajectory, world = createCartpoleTrajectory()
        par_loss, par_knot_loss = trajectory.parallel_unroll()
        loss, knot_loss = trajectory.unroll()
        print('par loss: '+str(par_loss))
        print('loss: '+str(loss))
        par_loss.backward()
        par_grads = [None if t.grad == None else t.grad.numpy() for t in trajectory.tensors()]
        print(par_grads)

    def test_cube_parallel(self):
        trajectory, world = createCubeTrajectory(
            steps=10, shooting_length=2, tune_starting_point=False, enforce_loop=False,
            enforce_final_state=np.zeros(6))
        par_loss, par_knot_loss = trajectory.parallel_unroll()
        loss, knot_loss = trajectory.unroll()
        print('par loss: '+str(par_loss))
        print('loss: '+str(loss))
        par_loss.backward()
        par_grads = [None if t.grad == None else t.grad.numpy() for t in trajectory.tensors()]
        loss.backward()
        grads = [None if t.grad == None else t.grad.numpy() for t in trajectory.tensors()]
        print(grads)
    """


if __name__ == '__main__':
    unittest.main()
