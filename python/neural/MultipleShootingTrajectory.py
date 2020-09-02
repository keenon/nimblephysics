import dartpy as dart
from dart_layer import dart_layer, BackpropSnapshotPointer
from dart_multiple_shooting_layer import dart_multiple_shooting_layer, BulkPassResultPointer
import torch
from typing import Tuple, Callable, List
import numpy as np
import math
import ipyopt
from concurrent.futures import ThreadPoolExecutor
import time


class TrajectorWorldNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world):
        super(TrajectorWorldNode, self).__init__(world)

    def customPreStep(self):
        pass


class MultipleShootingTrajectory:
    """
    This manages a trajectory optimization problem in a re-usable way
    """

    def __init__(
            self,
            world: dart.simulation.World,
            eval_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, dart.simulation.World],
                                torch.Tensor],
            steps: int = 1000,
            shooting_length: int = 50,
            tune_starting_point: bool = False,
            enforce_loop: bool = False,  # TODO: this should be a list of bools
            enforce_final_state: np.ndarray = None,
            disable_actuators: List[int] = []):
        """
        Constructor
        """
        self.world = world
        self.eval_loss = eval_loss
        self.steps = steps
        self.shooting_length = min(shooting_length, steps)
        self.disable_actuators = disable_actuators
        self.tune_starting_point = tune_starting_point
        self.enforce_loop = enforce_loop
        self.enforce_final_state = enforce_final_state

        assert(not (enforce_loop and enforce_final_state is not None))

        world_dofs = world.getNumDofs()

        self.num_shots = math.ceil(steps / shooting_length)

        # Initialize the learnable torques
        self.torques = torch.zeros((world_dofs, steps), dtype=torch.float64, requires_grad=True)
        self.knot_poses = torch.zeros(
            (world_dofs, self.num_shots),
            dtype=torch.float64, requires_grad=True)
        self.knot_poses.data[:, :] = torch.unsqueeze(torch.from_numpy(world.getPositions()), dim=1)
        self.knot_vels = torch.zeros(
            (world_dofs, self.num_shots),
            dtype=torch.float64, requires_grad=True)
        self.knot_vels.data[:, :] = torch.unsqueeze(torch.from_numpy(world.getVelocities()), dim=1)
        """
        # Initialize knot points
        self.knot_poses = [
            # The first knot point is the starting point
            torch.tensor(
                world.getPositions(),
                dtype=torch.float64, requires_grad=tune_starting_point)] + [
            # The other knot points
            torch.zeros(world_dofs, dtype=torch.float64, requires_grad=True)
            for _ in range(self.num_shots - 1)]
        self.knot_vels = [
            # The first knot point is the starting point
            torch.tensor(
                world.getVelocities(),
                dtype=torch.float64, requires_grad=tune_starting_point)] + [
            # The other knot points
            torch.zeros(world_dofs, dtype=torch.float64, requires_grad=True)
            for _ in range(self.num_shots - 1)]
        """

        self.mask = torch.tensor([1] * world.getNumDofs(), requires_grad=False)
        for j in disable_actuators:
            self.mask[j] = 0

        self.last_x = np.ones(self.get_flat_problem_dim()) * 1e+19
        self.snapshots: List[dart.neural.BackpropSnapshot] = [None] * self.steps
        self.last_loss = torch.tensor([0], requires_grad=False)
        self.already_run_backwards = False
        self.last_preknot_poses = np.zeros((world_dofs, self.num_shots), dtype=np.float64)
        self.last_preknot_vels = np.zeros((world_dofs, self.num_shots), dtype=np.float64)
        self.last_terminal_pos = np.zeros(world_dofs, dtype=np.float64)
        self.last_terminal_vel = np.zeros(world_dofs, dtype=np.float64)

        self.soft_knot_barrier_strength = 1

        self.bulk_result_pointer: BulkPassResultPointer = None

        # An option flag to turn off multithreading forward and backward passes
        self.multithread = False

        # Create a viewer
        self.viewer: dart.gui.osg.Viewer = None
        self.keyframe_worlds: List[dart.simulation.World] = []

        # Create a place to store the lagrange mulipliers for IPOPT, so we can restart cleanly
        self.lower_bound_multipliers = np.zeros(self.get_flat_problem_dim())
        self.upper_bound_multipliers = np.zeros(self.get_flat_problem_dim())
        self.constraint_multipliers = np.zeros(self.get_constraint_dim())

        self.best_loss = 10e15
        self.best_torques = torch.zeros_like(self.torques)
        self.best_knot_poses = torch.zeros_like(self.knot_poses)
        self.best_knot_vels = torch.zeros_like(self.knot_vels)

        # An option to compute the Hessian explicitly
        self.compute_hessian = False

    def tensors(self):
        return [self.torques, self.knot_vels, self.knot_poses]

    def postprocess_grad(self):
        """
        This is useful for basic SGD. It removes gradients for anything we're not allowed to learn.
        """
        if not self.tune_starting_point:
            self.knot_poses.grad[:, 0] = 0
            self.knot_vels.grad[:, 0] = 0

    def enforce_limits(self):
        """
        This is useful for basic SGD. It clamps the trajectory to be at only legal points.
        """
        n = self.get_flat_problem_dim()
        state = self.flatten(np.zeros(n), 'state')
        ub = self.flatten(np.zeros(n), 'upper_bound')
        lb = self.flatten(np.zeros(n), 'lower_bound')
        state = np.maximum(state, lb)
        state = np.minimum(state, ub)
        self.unflatten(state)

    def unroll(
            self,
            use_knots: bool = True,
            after_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                                 None] = None,
            compute_knot_loss: bool = False):
        """
        Returns total loss
        """
        pos = self.knot_poses[:, 0]
        vel = self.knot_vels[:, 0]

        post_step_poses = []
        post_step_vels = []

        knot_loss = torch.tensor([0], dtype=torch.float, requires_grad=True)

        for i in range(self.steps):
            t = self.torques[:, i] * self.mask

            # We're at a knot point
            if i % self.shooting_length == 0 and i > 0 and use_knots:
                knot_index = math.floor(i / self.shooting_length)
                knot_pos = self.knot_poses[:, knot_index]
                knot_vel = self.knot_vels[:, knot_index]

                # Record where we were, so we can calculate error terms later
                self.last_preknot_poses[:, knot_index] = pos.detach().numpy()
                self.last_preknot_vels[:, knot_index] = vel.detach().numpy()

                if compute_knot_loss:
                    # Record the loss from the error at this knot
                    this_knot_loss = torch.exp(
                        (knot_pos - pos).norm() * self.soft_knot_barrier_strength) + torch.exp(
                        10 * (knot_vel - vel).norm() * self.soft_knot_barrier_strength)
                    knot_loss = knot_loss + this_knot_loss

                # Now reset to the actual knot position
                pos = knot_pos
                vel = knot_vel

                # Check for NaNs
                if np.isnan(pos.detach().numpy()).any() or np.isnan(vel.detach().numpy()).any():
                    print('Caught a NaN being introduced at knot['+str(knot_index)+']')

            pointer = BackpropSnapshotPointer()
            next_pos, next_vel = dart_layer(self.world, pos, vel, t, pointer)
            self.snapshots[i] = pointer.backprop_snapshot
            post_step_poses.append(next_pos)
            post_step_vels.append(next_vel)

            # Check for NaNs being newly introduced
            if not(
                    np.isnan(pos.detach().numpy()).any() or np.isnan(vel.detach().numpy()).any()) and (
                    np.isnan(next_pos.detach().numpy()).any() or np.isnan(next_vel.detach().numpy()).
                    any()):
                print('Caught a NaN being introduced at timestep['+str(i)+']:')
                print('   old pos: '+str(pos.detach()))
                print('   old vel: '+str(vel.detach()))
                print('   torque: '+str(t.detach()))
                print('   new pos: '+str(next_pos.detach()))
                print('   new vel: '+str(next_vel.detach()))

            # Move
            pos = next_pos
            vel = next_vel

            if after_step is not None:
                after_step(pos, vel, t)

        loss = self.eval_loss(self.torques, torch.stack(
            post_step_poses, dim=1), torch.stack(post_step_vels, dim=1), self.world)

        self.last_terminal_pos = pos.detach().numpy()
        self.last_terminal_vel = vel.detach().numpy()

        if self.enforce_loop:
            start_pos = self.knot_poses[:, 0]
            start_vel = self.knot_vels[:, 0]
            # Record the loss from the error at the final knot
            loop_loss = torch.exp((pos - start_pos).norm() * self.soft_knot_barrier_strength) + torch.exp(
                10 * (vel - start_vel).norm() * self.soft_knot_barrier_strength)
            knot_loss = knot_loss + loop_loss
        if self.enforce_final_state is not None:
            goal_pos = self.enforce_final_state[:self.world.getNumDofs()]
            goal_vel = self.enforce_final_state[self.world.getNumDofs():]
            loop_loss = torch.exp(
                (pos - torch.from_numpy(goal_pos)).norm() * self.soft_knot_barrier_strength) + torch.exp(
                10 * (vel - torch.from_numpy(goal_vel)).norm() * self.soft_knot_barrier_strength)
            knot_loss = knot_loss + loop_loss

        self.last_loss = loss
        self.already_run_backwards = False

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_torques.copy_(self.torques)  # type: ignore
            self.best_knot_poses.copy_(self.knot_poses)  # type: ignore
            self.best_knot_vels.copy_(self.knot_vels)  # type: ignore

        return loss, knot_loss

    def parallel_unroll(self, compute_knot_loss=False):
        """
        This runs an unroll, parallelizing across knots during the forward and backward pass in native C++.
        Because of this, it doesn't allow you to skip knots or to have a callback after each step.
        """
        self.bulk_result_pointer = BulkPassResultPointer()

        post_step_poses, post_step_vels = dart_multiple_shooting_layer(
            self.world, self.torques, self.shooting_length, self.knot_poses, self.knot_vels, self.bulk_result_pointer)

        self.snapshots = self.bulk_result_pointer.forwardPassResult.snapshots

        loss = self.eval_loss(self.torques, post_step_poses, post_step_vels, self.world)

        # Handle knots, if required

        knot_loss = torch.tensor([0], dtype=torch.float, requires_grad=True)

        post_step_poses_detached = post_step_poses.detach().numpy()
        post_step_vels_detached = post_step_vels.detach().numpy()

        self.last_terminal_pos = post_step_poses_detached[:, self.steps-1]
        self.last_terminal_vel = post_step_vels_detached[:, self.steps-1]

        for knot_index in range(self.num_shots):
            i = knot_index * self.shooting_length
            if i == 0:
                continue

            # Record where we were, so we can calculate error terms later
            self.last_preknot_poses[:, knot_index] = post_step_poses_detached[:, i-1]
            self.last_preknot_vels[:, knot_index] = post_step_vels_detached[:, i-1]

            # Record the loss from the error at this knot
            if compute_knot_loss:
                pos = post_step_poses[:, i-1]
                vel = post_step_vels[:, i-1]
                knot_pos = self.knot_poses[:, knot_index]
                knot_vel = self.knot_vels[:, knot_index]
                this_knot_loss = torch.exp(
                    (knot_pos - pos).norm() * self.soft_knot_barrier_strength) + torch.exp(
                    10 * (knot_vel - vel).norm() * self.soft_knot_barrier_strength)
                knot_loss = knot_loss + this_knot_loss

        if compute_knot_loss:
            if self.enforce_loop:
                start_pos = self.knot_poses[0]
                start_vel = self.knot_vels[0]
                # Record the loss from the error at the final knot
                loop_loss = torch.exp((pos - start_pos).norm() * self.soft_knot_barrier_strength) + torch.exp(
                    10 * (vel - start_vel).norm() * self.soft_knot_barrier_strength)
                knot_loss = knot_loss + loop_loss
            if self.enforce_final_state is not None:
                goal_pos = self.enforce_final_state[:self.world.getNumDofs()]
                goal_vel = self.enforce_final_state[self.world.getNumDofs():]
                loop_loss = torch.exp(
                    (pos - torch.from_numpy(goal_pos)).norm() * self.soft_knot_barrier_strength) + torch.exp(
                    10 * (vel - torch.from_numpy(goal_vel)).norm() * self.soft_knot_barrier_strength)
                knot_loss = knot_loss + loop_loss

        self.last_loss = loss
        self.already_run_backwards = False

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.best_torques.copy_(self.torques)
            self.best_knot_poses.copy_(self.knot_poses)
            self.best_knot_vels.copy_(self.knot_vels)

        return loss, knot_loss

    ##############################################################################
    # Displaying GUI
    ##############################################################################

    def restore_best_loss(self):
        self.torques.data.copy_(self.best_torques)
        self.knot_poses.data.copy_(self.best_knot_poses)
        self.knot_vels.data.copy_(self.best_knot_vels)

    def create_gui(self):
        self.viewer = dart.gui.osg.Viewer()
        self.viewer.setUpViewInWindow(0, 0, 640, 480)
        self.viewer.setCameraHomePosition([0, 0, 5.0], [0, 0, 0], [0, 0.5, 0])
        self.viewer.realize()

    def playback_trajectory(self):
        # Clear out any other stuff we may have in the world
        for keyframe_world in self.keyframe_worlds:
            self.viewer.removeWorldNode(keyframe_world)
        self.viewer.removeWorldNode(self.world)

        # Add our world in
        self.playback_node = TrajectorWorldNode(self.world)
        self.viewer.addWorldNode(self.playback_node)

        def animate_step(pos, vel, t):
            self.viewer.frame()
            time.sleep(0.003)

        # self.unroll(use_knots=True, after_step=animate_step)
        self.unroll(use_knots=False, after_step=animate_step)

    def display_trajectory(self):
        # Clear out any other stuff we may have in the world
        for keyframe_world in self.keyframe_worlds:
            self.viewer.removeWorldNode(keyframe_world)
        self.viewer.removeWorldNode(self.world)

        self.keyframe_worlds = []
        for i in range(self.num_shots):
            keyframe_world: dart.simulation.World = self.world.clone()

            # Set the alpha of everything to something lower
            time_percentage = float(i) / self.num_shots
            alpha = 0.7 - (time_percentage / 2.0)

            for j in range(keyframe_world.getNumSkeletons()):
                skel: dart.dynamics.Skeleton = keyframe_world.getSkeleton(j)
                for k in range(skel.getNumBodyNodes()):
                    body: dart.dynamics.BodyNode = skel.getBodyNode(k)
                    for shape in body.getShapeNodes():
                        if shape.hasVisualAspect():
                            visualAspect: dart.dynamics.VisualAspect = shape.getVisualAspect()
                            visualAspect.setAlpha(visualAspect.getAlpha() * alpha)

            keyframe_world.setPositions(self.knot_poses[:, i].detach().numpy())
            keyframe_world.setVelocities(self.knot_vels[:, i].detach().numpy())

            keyframe_node = TrajectorWorldNode(keyframe_world)
            self.viewer.addWorldNode(keyframe_node)

            self.keyframe_worlds.append(keyframe_world)

        self.viewer.frame()

    ##############################################################################
    # IPOPT logic
    ##############################################################################

    def get_flat_problem_dim(self):
        """
        This gets the total number of decision variables of the trajectory problem
        """
        world_dofs = self.world.getNumDofs()
        knot_phase_dim = world_dofs * 2
        torques_dim = self.shooting_length * world_dofs
        shot_dim = knot_phase_dim + torques_dim

        dims = self.num_shots * shot_dim

        # If we're not tuning the starting point, then remove those variables
        if not self.tune_starting_point:
            dims -= 2 * world_dofs

        # Remove any variables at the end of the last shot that get cut off because of problem length
        last_shot_steps = (self.steps % self.shooting_length)
        if last_shot_steps > 0:
            dead_steps = self.shooting_length - last_shot_steps
            dims -= dead_steps * world_dofs

        return dims

    def get_constraint_dim(self):
        """
        This gets the total number of constraint dimensions
        """
        world_dofs = self.world.getNumDofs()
        knot_phase_dim = world_dofs * 2

        num_knot_points = self.num_shots - 1  # Ignore the starting point

        if self.enforce_loop or (self.enforce_final_state is not None):
            num_knot_points += 1

        return num_knot_points * knot_phase_dim

    def flatten(self, out: np.ndarray, which: str = 'state'):
        """
        This turns our current tensors into a single X vector that we can pass to a solver like IPOPT
        """
        assert len(out) == self.get_flat_problem_dim()
        world_dofs = self.world.getNumDofs()

        flat_cursor = 0
        torque_cursor = 0
        for i in range(self.num_shots):
            if i > 0 or self.tune_starting_point:
                if which == 'grad':
                    out[flat_cursor:flat_cursor+world_dofs] = np.zeros(
                        world_dofs) if self.knot_poses.grad is None else self.knot_poses.grad[:, i]
                    flat_cursor += world_dofs
                    out[flat_cursor:flat_cursor+world_dofs] = np.zeros(
                        world_dofs) if self.knot_vels.grad is None else self.knot_vels.grad[:, i]
                    flat_cursor += world_dofs
                elif which == 'upper_bound':
                    out[flat_cursor:flat_cursor+world_dofs] = self.world.getPositionUpperLimits()
                    flat_cursor += world_dofs
                    out[flat_cursor:flat_cursor+world_dofs] = self.world.getVelocityUpperLimits()
                    flat_cursor += world_dofs
                elif which == 'lower_bound':
                    out[flat_cursor:flat_cursor+world_dofs] = self.world.getPositionLowerLimits()
                    flat_cursor += world_dofs
                    out[flat_cursor:flat_cursor+world_dofs] = self.world.getVelocityLowerLimits()
                    flat_cursor += world_dofs
                else:
                    assert(which == 'state')
                    out[flat_cursor:flat_cursor+world_dofs] = self.knot_poses[:, i].detach()
                    flat_cursor += world_dofs
                    out[flat_cursor:flat_cursor+world_dofs] = self.knot_vels[:, i].detach()
                    flat_cursor += world_dofs

            shot_length = min(self.shooting_length, self.steps - torque_cursor)
            for j in range(shot_length):
                if which == 'grad':
                    out[flat_cursor:flat_cursor+world_dofs] = np.zeros(
                        world_dofs) if self.torques.grad is None else self.torques.grad[:, torque_cursor]
                elif which == 'upper_bound':
                    out[flat_cursor:flat_cursor+world_dofs] = self.world.getForceUpperLimits()
                elif which == 'lower_bound':
                    out[flat_cursor:flat_cursor+world_dofs] = self.world.getForceLowerLimits()
                else:
                    assert(which == 'state')
                    out[flat_cursor:flat_cursor+world_dofs] = self.torques[:, torque_cursor].detach()
                flat_cursor += world_dofs
                torque_cursor += 1
        return out

    def unflatten(self, x: np.ndarray):
        """
        This writes the values in x back into our original tensors
        """
        world_dofs = self.world.getNumDofs()
        flat_cursor = 0
        torque_cursor = 0
        for i in range(self.num_shots):
            if i > 0 or self.tune_starting_point:
                self.knot_poses.data[:, i] = torch.from_numpy(x[flat_cursor:flat_cursor+world_dofs])
                flat_cursor += world_dofs
                self.knot_vels.data[:, i] = torch.from_numpy(x[flat_cursor:flat_cursor+world_dofs])
                flat_cursor += world_dofs

            shot_length = min(self.shooting_length, self.steps - torque_cursor)
            for j in range(shot_length):
                self.torques.data[:, torque_cursor] = torch.from_numpy(
                    x[flat_cursor: flat_cursor + world_dofs])
                flat_cursor += world_dofs
                torque_cursor += 1

    def get_flat_dim_name(self, dim):
        """
        This gets the name of a flat dimension of the problem
        """
        world_dofs = self.world.getNumDofs()
        flat_cursor = 0
        torque_cursor = 0
        for i in range(self.num_shots):
            if i > 0 or self.tune_starting_point:
                # poses
                if flat_cursor <= dim and flat_cursor + world_dofs >= dim:
                    dof = dim - flat_cursor
                    return "knot "+str(i)+" pos of dof "+str(dof)
                flat_cursor += world_dofs
                # vels
                if flat_cursor <= dim and flat_cursor + world_dofs >= dim:
                    dof = dim - flat_cursor
                    return "knot "+str(i)+" vel of dof "+str(dof)
                flat_cursor += world_dofs

            shot_length = min(self.shooting_length, self.steps - torque_cursor)
            for j in range(shot_length):
                if flat_cursor <= dim and flat_cursor + world_dofs >= dim:
                    dof = dim - flat_cursor
                    return "knot "+str(i)+" offset "+str(j)+" torque of dof "+str(dof)
                flat_cursor += world_dofs
                torque_cursor += 1

    def debug_flat_compare(self, xs: List[np.ndarray], threshold=1e-10):
        """
        This prints the values of several different flat vectors, labeling each component
        """
        world_dofs = self.world.getNumDofs()
        flat_cursor = 0
        torque_cursor = 0
        for i in range(self.num_shots):
            if i > 0 or self.tune_starting_point:
                for q in range(world_dofs):
                    values = [x[flat_cursor + q] for x in xs]
                    if max(values) - min(values) > threshold:
                        print(
                            str(flat_cursor + q)+': knot_pos['+str(i * self.shooting_length)+']['+str(q)+']: '+str(values))
                flat_cursor += world_dofs
                for q in range(world_dofs):
                    values = [x[flat_cursor + q] for x in xs]
                    if max(values) - min(values) > threshold:
                        print(
                            str(flat_cursor + q)+': knot_vel['+str(i * self.shooting_length)+']['+str(q)+']: '+str(values))
                flat_cursor += world_dofs

            shot_length = min(self.shooting_length, self.steps - torque_cursor)
            for j in range(shot_length):
                for q in range(world_dofs):
                    values = [x[flat_cursor + q] for x in xs]
                    if max(values) - min(values) > threshold:
                        print(
                            str(flat_cursor + q)+': torques['+str(torque_cursor)+']['+str(q)+']: '+str(values))
                flat_cursor += world_dofs
                torque_cursor += 1

    def get_knot_x_offsets(self):
        """
        This returns a set of offset pointers into the x vector for each knot
        """
        knot_offsets = []

        world_dofs = self.world.getNumDofs()
        flat_cursor = 0
        torque_cursor = 0
        for i in range(self.num_shots):
            if i > 0 or self.tune_starting_point:
                knot_offsets.append(flat_cursor)
                # Account for pos + vel
                flat_cursor += world_dofs * 2
            else:
                knot_offsets.append(None)

            shot_length = min(self.shooting_length, self.steps - torque_cursor)
            torque_cursor += shot_length
            flat_cursor += shot_length * world_dofs

        return knot_offsets

    def get_knot_x_lengths(self):
        n = self.get_flat_problem_dim()
        knot_offsets = self.get_knot_x_offsets()
        lengths = []
        for i in range(len(knot_offsets)):
            if knot_offsets[i] is None or i == 0:
                lengths.append(0)
            else:
                next_offset = 0
                if i+1 < len(knot_offsets):
                    next_offset = knot_offsets[i+1]
                else:
                    next_offset = n
                lengths.append(next_offset - knot_offsets[i])
        return lengths

    def get_knot_g_offsets(self):
        """
        This returns a set of offset pointers into the g(x) vector for each knot
        """
        knot_offsets = []

        world_dofs = self.world.getNumDofs()
        flat_cursor = 0
        if self.enforce_loop or (self.enforce_final_state is not None):
            flat_cursor += world_dofs * 2
        for i in range(self.num_shots):
            if i > 0:
                knot_offsets.append(flat_cursor)
                # Account for pos + vel
                flat_cursor += world_dofs * 2
            else:
                knot_offsets.append(None)

        return knot_offsets

    def ensure_fresh_rollout(self, x: np.ndarray):
        """
        This is a no-op if we've already rolled out for x. Otherwise, this runs an unroll()
        for the trajectory x encodes.
        """
        if (x != self.last_x).any():  # always unroll
            self.last_x = x.copy()
            self.unflatten(x)
            if self.multithread:
                # print('parallel unrolling x='+str(x))
                self.parallel_unroll()
            else:
                # print('unrolling x='+str(x))
                self.unroll()

    def eval_f(self, x: np.ndarray):
        """
        This returns the loss from a given trajectory encoded by x.
        """
        if np.isnan(x).any():
            print('f(x) was fed NaNs!')
            print('x = '+str(x))
        assert not np.isnan(x).any()

        self.ensure_fresh_rollout(x)
        loss = self.last_loss.item()
        # print('f(x): '+str(loss))
        assert not math.isnan(loss)
        return loss

    def eval_grad_f(self, x: np.ndarray, out: np.ndarray):
        """
        This returns the gradient of loss at the trajectory encoded by x.
        """

        if np.isnan(x).any():
            print('grad f(x) was fed NaNs!')
            print('x = '+str(x))
        assert not np.isnan(x).any()
        assert(len(out) == self.get_flat_problem_dim())
        self.ensure_fresh_rollout(x)
        # Zero out the gradient
        for tensor in self.tensors():
            if tensor.grad is not None and tensor.grad.data is not None:
                tensor.grad.data.zero_()
        # Run backprop through our last loss
        if not self.already_run_backwards:
            self.last_loss.backward()
            self.already_run_backwards = True
        # Flatten the gradient into a vector
        out = self.flatten(out, 'grad')
        assert not np.isnan(out).any()
        # print('grad f(x):\nx='+str(x)+'\ngrad f(x)'+str(out))
        return out

    def eval_g(self, x: np.ndarray, out: np.ndarray):
        """
        This returns the gap at knot points for a given trajectory encoded by x.
        """
        if np.isnan(x).any():
            print('g(x) was fed NaNs!')
            print('x = '+str(x))
        assert not np.isnan(x).any()

        self.ensure_fresh_rollout(x)
        assert(len(out) == self.get_constraint_dim())

        world_dofs = self.world.getNumDofs()
        cursor = 0

        if self.enforce_loop:
            out[cursor:cursor+world_dofs] = self.last_terminal_pos - self.knot_poses[:,
                                                                                     0].detach()
            cursor += world_dofs
            out[cursor:cursor+world_dofs] = self.last_terminal_vel - self.knot_vels[:,
                                                                                    0].detach()
            cursor += world_dofs
        elif self.enforce_final_state is not None:
            goal_pos = self.enforce_final_state[:self.world.getNumDofs()]
            goal_vel = self.enforce_final_state[self.world.getNumDofs():]
            out[cursor:cursor+world_dofs] = self.last_terminal_pos - goal_pos
            cursor += world_dofs
            out[cursor:cursor+world_dofs] = self.last_terminal_vel - goal_vel
            cursor += world_dofs

        for i in range(self.num_shots):
            if i == 0:
                continue
            out[cursor:cursor+world_dofs] = self.last_preknot_poses[:,
                                                                    i] - self.knot_poses[:, i].detach().numpy()
            cursor += world_dofs
            out[cursor:cursor+world_dofs] = self.last_preknot_vels[:,
                                                                   i] - self.knot_vels[:, i].detach().numpy()
            cursor += world_dofs

        if np.isnan(out).any():
            print('g(x) found NaNs!')
            print('x = '+str(x))
            print('g(x) = '+str(out))
            for i in range(self.num_shots):
                if i == 0:
                    continue
                print('knot['+str(i)+']:')
                print('   pre-knot pose: '+str(self.last_preknot_poses[i]))
                print('   knot pose: '+str(self.knot_poses[i].detach()))
                print('   pre-knot vel: '+str(self.last_preknot_vels[i]))
                print('   knot vel: '+str(self.knot_vels[i].detach()))
        assert(cursor == len(out))
        assert not np.isnan(out).any()
        # print('g(x):\nx='+str(x)+'\ng(x)'+str(out))
        return out

    def eval_jac_g(self, x: np.ndarray, out: np.ndarray):
        """
        This computes a Jacobian of g(x)
        """
        if np.isnan(x).any():
            print('jac g(x) was fed NaNs!')
            print('x = '+str(x))
        assert not np.isnan(x).any()

        self.ensure_fresh_rollout(x)
        # Run backprop through our last loss
        if self.multithread and self.bulk_result_pointer.backwardPassResult is None and not self.already_run_backwards:
            self.last_loss.backward()
            self.already_run_backwards = True

        world_dofs = self.world.getNumDofs()
        cursor = 0
        dt = self.world.getTimeStep()

        def insertNegativeIdentity(cursor: int):
            """
            This puts a negative identity matrix into the output values
            """
            for i in range(world_dofs*2):
                out[cursor] = -1.0
                cursor += 1
            return cursor

        def insertFullJacobian(last_knot_index: int, cursor: int, includePhasePhase: bool = True):
            """
            This computes a full Jacobian relating the whole shot to the error at this knot-point (last_knot_index + 1)
            """
            start_index = last_knot_index * self.shooting_length
            end_index_exclusive = min((last_knot_index + 1) * self.shooting_length, self.steps)
            if self.multithread and self.bulk_result_pointer is not None and len(
                    self.bulk_result_pointer.backwardPassResult.knotJacobians) > 0:
                # We can use the pre-computed Jacobians! Hooray :)
                J: dart.neural.KnotJacobian = self.bulk_result_pointer.backwardPassResult.knotJacobians[
                    last_knot_index]
                for i in reversed(range(start_index, end_index_exclusive)):
                    # shoot_len = end_index_exclusive - start_index - 1
                    offset = i - start_index
                    posend_force: np.ndarray = J.torquesEndPos[offset]
                    velend_force: np.ndarray = J.torquesEndVel[offset]
                    # Write our force_pos_end and force_vel_end into the output, row by row
                    for row in range(world_dofs):
                        for col in range(world_dofs):
                            out[cursor] = posend_force[row][col]
                            cursor += 1
                    for row in range(world_dofs):
                        for col in range(world_dofs):
                            out[cursor] = velend_force[row][col]
                            cursor += 1
                if includePhasePhase:
                    # Put these so the rows correspond to a whole phase vector
                    posend_phase = np.concatenate([J.knotPosEndPos, J.knotVelEndPos], axis=1)
                    velend_phase = np.concatenate([J.knotPosEndVel, J.knotVelEndVel], axis=1)
                    for row in range(world_dofs):
                        for col in range(world_dofs * 2):
                            out[cursor] = posend_phase[row][col]
                            cursor += 1
                    for row in range(world_dofs):
                        for col in range(world_dofs * 2):
                            out[cursor] = velend_phase[row][col]
                            cursor += 1
            else:
                # We'll have to compute the Jacobians ourselves
                """
                For our purposes here (forward Jacobians), the forward computation 
                graph looks like these two separate processes:

                p_t -------------+
                                  \ 
                                   \                               
                v_t ----------------+----(LCP Solver)----> v_t+1 ---->
                                   /
                                  /
                f_t -------------+

                v_t -------------+
                                  \ 
                                   \                               
                p_t ----------------+--------------------> p_t+1 ---->
                """
                # p_end <-- p_t+1
                posend_posnext = np.identity(world_dofs)
                # p_end <-- v_t+1
                # np.zeros((world_dofs, world_dofs))
                # -dt * dt * np.identity(world_dofs)
                posend_velnext = np.zeros((world_dofs, world_dofs))

                # v_end <-- p_t+1
                velend_posnext = np.zeros((world_dofs, world_dofs))
                # v_end <-- v_t+1
                velend_velnext = np.identity(world_dofs)

                for i in reversed(range(start_index, end_index_exclusive)):
                    snapshot: dart.neural.BackpropSnapshot = self.snapshots[i]

                    # v_t+1 <-- p_t
                    velnext_pos = snapshot.getPosVelJacobian(self.world)
                    # v_t+1 <-- v_t
                    velnext_vel = snapshot.getVelVelJacobian(self.world)
                    # v_t+1 <-- f_t
                    velnext_force = snapshot.getForceVelJacobian(self.world)

                    # p_t+1 <-- p_t
                    # This isn't always Identity, because of bounces
                    posnext_pos = snapshot.getPosPosJacobian(self.world)
                    # p_t+1 <-- v_t
                    posnext_vel = snapshot.getVelPosJacobian(self.world)

                    # p_end <-- f_t = (p_end <-- v_t+1) * (v_t+1 <-- f_t)
                    posend_force = np.matmul(posend_velnext, velnext_force)
                    # v_end <-- f_t = (v_end <-- v_t+1) * (v_t+1 <-- f_t)
                    velend_force = np.matmul(velend_velnext, velnext_force)

                    # Write our force_pos_end and force_vel_end into the output, row by row
                    for row in range(world_dofs):
                        for col in range(world_dofs):
                            out[cursor] = posend_force[row][col]
                            cursor += 1
                    for row in range(world_dofs):
                        for col in range(world_dofs):
                            out[cursor] = velend_force[row][col]
                            cursor += 1

                    if i == end_index_exclusive - 1 and False:
                        print('v_t+1 <-- p_t:\n'+str(velnext_pos))
                        print('v_t+1 <-- v_t:\n'+str(velnext_vel))
                        print('v_t+1 <-- f_t:\n'+str(velnext_force))
                        print('p_t+1 <-- p_t:\n'+str(posnext_pos))
                        print('p_t+1 <-- v_t:\n'+str(posnext_vel))
                        print('p_end <-- f_t:\n'+str(posend_force))
                        print('v_end <-- f_t:\n'+str(velend_force))

                    # Update p_end <-- p_t+1 = ((p_end <-- p_t+1) * (p_t+1 <-- p_t)) + ((p_end <-- v_t+1) * (v_t+1 <-- p_t))
                    posend_posnext = np.matmul(
                        posend_posnext, posnext_pos) + np.matmul(posend_velnext, velnext_pos)
                    # Update v_end <-- p_t+1 ...
                    velend_posnext = np.matmul(
                        velend_posnext, posnext_pos) + np.matmul(velend_velnext, velnext_pos)
                    # Update v_t+1 --> p_end ...
                    posend_velnext = np.matmul(
                        posend_posnext, posnext_vel) + np.matmul(posend_velnext, velnext_vel)
                    # Update v_t+1 --> v_end ...
                    velend_velnext = np.matmul(
                        velend_posnext, posnext_vel) + np.matmul(velend_velnext, velnext_vel)

                if includePhasePhase:
                    # Put these so the rows correspond to a whole phase vector
                    posend_phase = np.concatenate([posend_posnext, posend_velnext], axis=1)
                    velend_phase = np.concatenate([velend_posnext, velend_velnext], axis=1)
                    for row in range(world_dofs):
                        for col in range(world_dofs * 2):
                            out[cursor] = posend_phase[row][col]
                            cursor += 1
                    for row in range(world_dofs):
                        for col in range(world_dofs * 2):
                            out[cursor] = velend_phase[row][col]
                            cursor += 1
            return cursor

        if self.enforce_loop or (self.enforce_final_state is not None):
            # Insert the Jacobian for the last knot point -> the last timestep
            cursor = insertFullJacobian(self.num_shots - 1, cursor)
            # This means we're tuning the initial position, so our Jac for this knot
            # needs to include a negative identity
            if self.tune_starting_point and self.enforce_loop:
                cursor = insertNegativeIdentity(cursor)

        for i in range(self.num_shots):
            if i == 0:
                continue
            # Insert the Jacobian for the last knot point -> the last timestep
            # Ignore the phase-phase Jacobian at the beginning if we're not tuning the starting point
            cursor = insertFullJacobian(
                i - 1, cursor, includePhasePhase=(i > 1 or self.tune_starting_point))
            # Insert a -I relating this knot's position to this knot's constraint
            cursor = insertNegativeIdentity(cursor)

        jac_g_sparsity_indices = self.get_jac_g_sparsity_indices()
        assert(len(out) == len(jac_g_sparsity_indices[0]))
        assert(cursor == len(jac_g_sparsity_indices[0]))
        assert not np.isnan(out).any()

        # print('sparse jac g(x):\nx='+str(x)+'\nsparse jac g(x)'+str(out))

        return out

    def get_jac_g_sparsity_indices(self):
        """
        rows correspond to each constraint
        cols correspond to each variable in x
        """
        row: List[int] = []
        col: List[int] = []

        world_dofs = self.world.getNumDofs()
        n = self.get_flat_problem_dim()

        def insertNegativeIdentity(rowTopLeft: int, colTopLeft: int):
            """
            This puts a space for a negative identity matrix into the sparsity indices
            """
            for i in range(world_dofs*2):
                row.append(rowTopLeft + i)
                col.append(colTopLeft + i)

        def insertPhasePhaseJacobian(rowTopLeft: int, colTopLeft: int):
            """
            This puts a space for a full Jacobian into the sparsity indices

            This relates the previous knot point phase with the error at the next knot point
            """
            for r in range(world_dofs*2):
                for c in range(world_dofs*2):
                    row.append(rowTopLeft + r)
                    col.append(colTopLeft + c)

        def insertTorquePhaseJacobian(rowTopLeft: int, colTopLeft: int):
            """
            This puts a space for a torque-phase Jacobian into the sparsity indices

            This relates a given torque spot with the error at the next knot point
            """
            for r in range(world_dofs*2):
                for c in range(world_dofs):
                    row.append(rowTopLeft + r)
                    col.append(colTopLeft + c)

        def insertFullJacobian(rowTopLeft: int, colTopLeft: int, includePhasePhase: bool = True):
            """
            This puts space for the whole Jacobian for a given constraint into the sparsity indices.
            This is the phase-phase Jacobian, and all the torque-phase Jacobians
            """
            # The torques are all directly to the right of the knot-point phase variables for the last shooting section
            torque_jac_locations: List[int] = []

            cursor = colTopLeft
            if includePhasePhase:
                cursor += world_dofs*2
            for i in range(self.shooting_length):
                torque_jac_locations.append(cursor)
                cursor += world_dofs
                if cursor >= n:
                    break

            # Add the torque Jacobians in reverse, because it will make it more memory efficient to compute
            torque_jac_locations.reverse()
            for cursor in torque_jac_locations:
                insertTorquePhaseJacobian(rowTopLeft, cursor)
            if includePhasePhase:
                insertPhasePhaseJacobian(rowTopLeft, colTopLeft)

        knot_x_offsets = self.get_knot_x_offsets()
        knot_g_offsets = self.get_knot_g_offsets()

        if self.enforce_loop or (self.enforce_final_state is not None):
            # Insert the Jacobian for the last knot point -> the last timestep
            insertFullJacobian(0, knot_x_offsets[len(knot_x_offsets)-1])
            # This means we're not tuning the initial position, so our Jac for this knot
            # can just be the Jac for the last timestep
            if knot_x_offsets[0] is None:
                assert(not self.tune_starting_point)
            # This means this is basically an ordinary constraint, and also has a negative
            # identity relationship to the original knot
            elif self.enforce_loop:
                insertNegativeIdentity(0, knot_x_offsets[0])

        for i in range(self.num_shots):
            if i == 0:
                continue
            # Insert the Jacobian for the last knot point -> the last timestep
            if knot_x_offsets[i-1] is None:
                # This indicates that we're mapping against a phase that doesn't exist, which
                # can only be the start phase when we're not allowing tuning of the start position
                insertFullJacobian(knot_g_offsets[i], 0, includePhasePhase=False)
            else:
                insertFullJacobian(knot_g_offsets[i], knot_x_offsets[i-1])
            # Insert a -I relating this knot's position to this knot's constraint
            insertNegativeIdentity(knot_g_offsets[i], knot_x_offsets[i])

        return (np.array(row), np.array(col))

    def get_h_sparsity_indices(self):
        """
        Returns the array of indices for the sparse Hessian. This is just block-diagonal with each
        shot getting its own independent Hessian.
        """
        rows: List[int] = []
        cols: List[int] = []

        world_dofs = self.world.getNumDofs()
        n = self.get_flat_problem_dim()

        knot_x_offsets = self.get_knot_x_offsets()
        knot_x_length = self.get_knot_x_lengths()

        for i in range(self.num_shots):
            shot_length = knot_x_length[i]
            if shot_length == 0:
                continue
            # Organize as column-major to make finite-differencing easier
            for col in range(knot_x_offsets[i], knot_x_offsets[i] + shot_length):
                for row in range(knot_x_offsets[i], knot_x_offsets[i] + shot_length):
                    rows.append(row)
                    cols.append(col)

        return (np.array(rows), np.array(cols))

    def eval_h(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float, out: np.ndarray):
        """
        This evaluates the Hessian, in the block diagonal sparsity structure
        """
        rows: List[int] = []
        cols: List[int] = []

        world_dofs = self.world.getNumDofs()
        n = self.get_flat_problem_dim()

        knot_x_offsets = self.get_knot_x_offsets()
        knot_x_length = self.get_knot_x_lengths()
        hess_offsets = []
        hess_cursor = 0
        for i in range(self.num_shots):
            hess_offsets.append(hess_cursor)
            hess_cursor += knot_x_length[i] * knot_x_length[i]
        jac_rows, jac_cols = self.get_jac_g_sparsity_indices()

        block_size = max(knot_x_length)

        eps = 1e-12

        grad_f = self.eval_grad_f(x, np.zeros(n))
        jac_g = self.eval_jac_g(x, np.zeros(len(jac_rows)))

        # We do all the shots in parallel, so our outer iteration is just over the point into the knots
        # that we're going to perturb. Each knot is independent for the Hessian, so we can finite
        # difference all the knots in the same place at the same time, and get meaningful results.
        for q in range(block_size):
            diff = np.zeros_like(x)
            for i in range(self.num_shots):
                if knot_x_offsets[i] is None:
                    continue
                offset = knot_x_offsets[i] + q
                if offset < n:
                    diff[offset] = eps

            x_prime = x + diff
            x_prime_minus = x - diff

            # Compute the gradients in two passes, with cacheing

            grad_f_prime = self.eval_grad_f(x_prime, np.zeros(n))
            jac_g_prime = self.eval_jac_g(x_prime, np.zeros(len(jac_rows)))
            grad_f_prime_minus = self.eval_grad_f(x_prime_minus, np.zeros(n))
            jac_g_prime_minus = self.eval_jac_g(x_prime_minus, np.zeros(len(jac_rows)))

            # Compute Hess(f(x)) with finite-differencing

            grad_diff = obj_factor * (grad_f_prime - grad_f_prime_minus) / (2 * eps)

            # Compute Hess(g(x)) with finite-differencing

            jac_g_diff = np.zeros(n)
            for j in range(len(jac_rows)):
                row = jac_rows[j]
                col = jac_cols[j]
                # row -> variable in g (sum over these)
                # col -> variable in x
                grad_g_elem = (jac_g_prime[j] - jac_g_prime_minus[j]) / (2 * eps)
                jac_g_diff[col] += grad_g_elem * lagrange[row]

            # Reassemble our vector into the sparse output

            for i in range(self.num_shots):
                if knot_x_offsets[i] is None:
                    continue
                col_len = knot_x_length[i]
                hess_col_offset = hess_offsets[i] + (q * col_len)
                grad_f_col = grad_diff[knot_x_offsets[i]:knot_x_offsets[i]+col_len]
                jac_g_col = jac_g_diff[knot_x_offsets[i]:knot_x_offsets[i]+col_len]
                out[hess_col_offset:hess_col_offset+col_len] = grad_f_col + jac_g_col

        return out

    def _test_get_dense_jac_g(self, x: np.ndarray):
        """
        This computes the dense version of the Jac of g(x).

        You shouldn't need this, except for testing.
        """
        n = self.get_flat_problem_dim()
        m = self.get_constraint_dim()
        dense = np.zeros((m, n))
        rows, cols = self.get_jac_g_sparsity_indices()
        values = self.eval_jac_g(x, np.zeros(len(rows)))
        for i in range(len(values)):
            dense[rows[i]][cols[i]] = values[i]
        return dense

    def _test_get_dense_h(
            self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float):
        """
        This computes the dense version of the Jac of g(x).

        You shouldn't need this, except for testing.
        """
        n = self.get_flat_problem_dim()
        dense = np.zeros((n, n))
        rows, cols = self.get_h_sparsity_indices()
        values = self.eval_h(x, lagrange, obj_factor, np.zeros(len(rows)))
        for i in range(len(values)):
            dense[rows[i]][cols[i]] = values[i]
        return dense

    def _test_brute_force_jac_g(self, x: np.ndarray):
        n = self.get_flat_problem_dim()
        m = self.get_constraint_dim()
        eps = 1e-8
        g = self.eval_g(x, np.zeros(m))
        cols = []
        for i in range(n):
            x_prime = x.copy()
            x_prime[i] += eps
            g_prime = self.eval_g(x_prime, np.zeros(m))
            g_diff = (g_prime - g) / eps
            cols.append(g_diff)
        self.unflatten(x)
        return np.stack(cols, axis=0).T

    def _brute_force_jac_sparsity(self):
        n = self.get_flat_problem_dim()
        m = self.get_constraint_dim()

        rows = []
        cols = []
        for row in range(m):
            for col in range(n):
                rows.append(row)
                cols.append(col)

        return (np.array(rows), np.array(cols))

    def _eval_brute_force_jac_g(self, x: np.ndarray, out: np.ndarray):
        rows, cols = self._brute_force_jac_sparsity()
        brute = self._test_brute_force_jac_g(x)
        for i in range(len(out)):
            out[i] = brute[rows[i]][cols[i]]
        assert not np.isnan(out).any()
        return out

    def _eval_dense_jac_g(self, x: np.ndarray, out: np.ndarray):
        rows, cols = self._brute_force_jac_sparsity()
        brute = self._test_get_dense_jac_g(x)
        # print('dense jac g(x): '+str(brute))
        for i in range(len(out)):
            out[i] = brute[rows[i]][cols[i]]
        assert(len(out) == len(rows))
        assert not np.isnan(out).any()

        # brute_out = self._eval_brute_force_jac_g(x, np.zeros_like(out))
        # diff = out - brute_out
        # print('diff: '+str(diff))

        return out

    def _eval_brute_force_grad_f(self, x: np.ndarray, out: np.ndarray):
        n = self.get_flat_problem_dim()
        eps = 1e-7
        f = self.eval_f(x)
        for i in range(n):
            x_prime = x.copy()
            x_prime[i] += eps
            f_prime = self.eval_f(x_prime)
            x_prime_minus = x.copy()
            x_prime_minus[i] -= eps
            f_prime_minus = self.eval_f(x_prime_minus)
            f_diff = (f_prime - f_prime_minus) / (2 * eps)
            out[i] = f_diff
        self.unflatten(x)
        return out

    def _brute_force_h_sparsity(self):
        n = self.get_flat_problem_dim()

        rows = []
        cols = []
        for row in range(n):
            for col in range(n):
                rows.append(row)
                cols.append(col)

        return (np.array(rows), np.array(cols))

    def _eval_brute_force_h(
            self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float, out: np.ndarray):
        rows, cols = self._brute_force_h_sparsity()
        brute = self._test_brute_force_h(x, lagrange, obj_factor)
        assert len(out) == len(rows)
        for i in range(len(out)):
            out[i] = brute[rows[i]][cols[i]]
        assert not np.isnan(out).any()
        return out

    def _test_brute_force_h(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float):
        n = self.get_flat_problem_dim()
        eps = 1e-6
        jac_rows, jac_cols = self.get_jac_g_sparsity_indices()
        grad_f = self.eval_grad_f(x, np.zeros(n))
        jac_g = self.eval_jac_g(x, np.zeros(len(jac_rows)))
        out = np.ndarray((n, n))
        print('attempting to brute force H:')
        for i in range(n):
            if i % 250 == 0:
                print('  test '+str(i)+'/'+str(n)+'')
            x_prime = x.copy()
            x_prime[i] += eps
            grad_f_prime = self.eval_grad_f(x_prime, np.zeros(n))
            jac_g_prime = self.eval_jac_g(x_prime, np.zeros(len(jac_rows)))
            x_prime_minus = x.copy()
            x_prime_minus[i] -= eps
            grad_f_prime_minus = self.eval_grad_f(x_prime_minus, np.zeros(n))
            jac_g_prime_minus = self.eval_jac_g(x_prime_minus, np.zeros(len(jac_rows)))
            grad_f_diff = obj_factor * (grad_f_prime - grad_f_prime_minus) / (2 * eps)
            # grad_f_diff = obj_factor * (grad_f_prime - grad_f) / eps
            jac_g_diff = np.zeros(n)
            for j in range(len(jac_rows)):
                row = jac_rows[j]
                col = jac_cols[j]
                # row -> variable in g (sum over these)
                # col -> variable in x
                grad_g_elem = (jac_g_prime[j] - jac_g_prime_minus[j]) / (2 * eps)
                jac_g_diff[col] += grad_g_elem * lagrange[row]
            out[:, i] = grad_f_diff + jac_g_diff

        """
        # Add small values to the diagonal to prevent singular-ness
        for i in range(n):
            out[i, i] += 1e-6
        """

        self.unflatten(x)

        return out

    def _eval_lagrange(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float):
        m = self.get_constraint_dim()
        f = self.eval_f(x)
        g = self.eval_g(x, np.zeros(m))
        return (obj_factor * f) + sum([lagrange[i] * g[i] for i in range(m)])

    def _eval_grad_lagrange(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float):
        n = self.get_flat_problem_dim()
        eps = 1e-6
        out = np.ndarray(n)
        dual = self._eval_lagrange(x, lagrange, obj_factor)
        for i in range(n):
            x_prime = x.copy()
            x_prime[i] += eps
            dual_prime = self._eval_lagrange(x_prime, lagrange, obj_factor)
            # out[i] = (dual_prime - dual) / eps
            x_prime_minus = x.copy()
            x_prime_minus[i] -= eps
            dual_prime_minus = self._eval_lagrange(x_prime_minus, lagrange, obj_factor)
            out[i] = (dual_prime - dual_prime_minus) / (eps * 2)
        self.unflatten(x)
        return out

    def _eval_h_lagrange(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float):
        n = self.get_flat_problem_dim()
        eps = 1e-6
        grad_l = self._eval_grad_lagrange(x, lagrange, obj_factor)
        out = np.ndarray((n, n))
        for i in range(n):
            x_prime = x.copy()
            x_prime[i] += eps
            grad_l_prime = self._eval_grad_lagrange(x_prime, lagrange, obj_factor)
            x_prime_minus = x.copy()
            x_prime_minus[i] -= eps
            grad_l_prime_minus = self._eval_grad_lagrange(x_prime_minus, lagrange, obj_factor)
            out[:, i] = (grad_l_prime - grad_l_prime_minus) / (2 * eps)
        self.unflatten(x)
        return out

    def ipopt(self, max_iter=300):
        """
        This uses IPOPT to try to get the bounds really tight on the knot points
        """
        n = self.get_flat_problem_dim()
        eps = 0  # 1e-8

        lower_bounds = self.flatten(np.zeros(n), 'lower_bound')
        upper_bounds = self.flatten(np.zeros(n), 'upper_bound')

        # skel->getJoints()
        # joint.getForceUpperLimits()
        # joint.getPositionLowerLimits()

        num_constraints = self.get_constraint_dim()
        constraint_upper_bounds = np.array([eps] * num_constraints)
        constraint_lower_bounds = np.array([-eps] * num_constraints)

        jac_g_sparsity_indices = self.get_jac_g_sparsity_indices()
        h_sparsity_indices = self.get_h_sparsity_indices()

        def eval_f(x: np.ndarray):
            try:
                return self.eval_f(x)
            except Exception as e:
                print('ERROR when IPOPT was evaluating our f(x)!')
                print(e)

        def eval_grad_f(x: np.ndarray, out: np.ndarray):
            try:
                return self.eval_grad_f(x, out)
            except Exception as e:
                print('ERROR when IPOPT was evaluating our gradient of f(x)!')
                print(e)

        def eval_g(x: np.ndarray, out: np.ndarray):
            try:
                return self.eval_g(x, out)
            except Exception as e:
                print('ERROR when IPOPT was evaluating our g(x)!')
                print(e)

        def eval_jac_g(x: np.ndarray, out: np.ndarray):
            try:
                return self.eval_jac_g(x, out)
            except Exception as e:
                print('ERROR when IPOPT was evaluating our Jacobian of g(x)!')
                print(e)

        def eval_h(x: np.ndarray, lagrange: np.ndarray, obj_factor: float, out: np.ndarray):
            try:
                return self.eval_h(x, lagrange, obj_factor, out)
            except Exception as e:
                print('ERROR when IPOPT was evaluating our Hessian of f(x)!')
                print(e)

        if self.compute_hessian:
            nlp = ipyopt.Problem(
                n, lower_bounds, upper_bounds, num_constraints, constraint_lower_bounds,
                constraint_upper_bounds, jac_g_sparsity_indices, h_sparsity_indices,
                eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)
            # nlp.set(print_level=7)
        else:
            nlp = ipyopt.Problem(
                n, lower_bounds, upper_bounds, num_constraints, constraint_lower_bounds,
                constraint_upper_bounds, jac_g_sparsity_indices, 0,
                eval_f, eval_grad_f, eval_g, eval_jac_g)
        nlp.set(max_iter=max_iter)
        nlp.set(
            print_info_string='yes', check_derivatives_for_naninf='yes',
            derivative_test='first-order', derivative_test_perturbation=1e-6)

        x0 = self.flatten(np.zeros(n), 'state')
        eval_jac_g(x0, np.zeros(len(jac_g_sparsity_indices[0])))

        print("Going to call IPOPT solve")
        if self.compute_hessian:
            _x, loss, status = nlp.solve(x0, mult_g=self.constraint_multipliers,
                                         mult_x_L=self.lower_bound_multipliers,
                                         mult_x_U=self.upper_bound_multipliers)
        else:
            _x, loss, status = nlp.solve(x0)
        print("IPOPT finished with final loss "+str(loss)+", "+str(status))
        self.unflatten(_x)

    def debug(self, fd=None, error=None):
        """
        This prints the current trajectory parameters
        """
        n = self.get_flat_problem_dim()
        m = self.get_constraint_dim()
        print('****************')
        print('Problem Size:')
        print('****************')
        print('n - num variables: '+str(n))
        print('m - num constraints: '+str(m))
        print('****************')
        print('Debug Trajectory Data:')
        print('****************')
        for i in range(self.steps):
            if i % self.shooting_length == 0:
                knot_index = math.floor(i / self.shooting_length)
                if (knot_index > 0):
                    print(
                        '      (end-pos): ' +
                        str(self.last_preknot_poses[knot_index]))
                    print(
                        '      (end-vel): ' +
                        str(self.last_preknot_vels[knot_index]))
                print('   knot '+str(knot_index)+' :')
                print('      pos: '+str(self.knot_poses[knot_index].detach().numpy()))
                print('      vel: '+str(self.knot_vels[knot_index].detach().numpy()))
            print('         f['+str(i)+']: '+str(self.torques[:, i].detach().numpy()))
        print('   (end-pos): '+str(self.last_terminal_pos))
        print('   (end-vel): '+str(self.last_terminal_vel))
        flat = self.flatten(np.zeros(n), 'state')
        print('flat: '+str(flat))

        # Do a rollout
        self.unroll()
        # Zero out the gradient
        for tensor in self.tensors():
            if tensor.grad is not None and tensor.grad.data is not None:
                tensor.grad.data.zero_()
        # Run backprop through our last loss
        self.last_loss.backward()

        world_dofs = self.world.getNumDofs()

        print('****************')
        print('Loss:')
        print('****************')
        print(self.last_loss.item())

        print('****************')
        print('Debug Trajectory Gradients:')
        print('****************')
        for i in range(self.steps):
            if i % self.shooting_length == 0:
                knot_index = math.floor(i / self.shooting_length)
                print('   knot '+str(knot_index)+':')
                print('      pos: '+str(np.zeros(world_dofs)
                                        if self.knot_poses.grad is None else self.knot_poses.grad.numpy()[:, knot_index]))
                print(
                    '      vel: ' +
                    str(
                        np.zeros(world_dofs)
                        if self.knot_vels[knot_index].grad is None else self.knot_vels
                        [knot_index].grad.detach().numpy()))
            print('         f['+str(i)+']: '+str(np.zeros(world_dofs)
                                                 if self.torques.grad is None else self.torques.grad.detach().numpy()[:, i]))
        flat_grad = self.flatten(np.zeros(n), 'grad')
        print('flat grad: '+str(flat_grad))

        if fd is not None:
            self.unflatten(fd)
            print('****************')
            print('Finite Difference Gradient:')
            print('****************')
            for i in range(self.steps):
                if i % self.shooting_length == 0:
                    knot_index = math.floor(i / self.shooting_length)
                    print('   knot '+str(knot_index)+' :')
                    if i > 0 or self.tune_starting_point:
                        print('      pos: '+str(self.knot_poses[:, knot_index].detach().numpy()))
                        print('      vel: '+str(self.knot_vels[:, knot_index].detach().numpy()))
                print('         f['+str(i)+']: '+str(self.torques[:, i].detach().numpy()))
            print('flat grad: '+str(fd))

        if error is not None:
            self.unflatten(error)
            print('****************')
            print('Error Data:')
            print('****************')
            for i in range(self.steps):
                if i % self.shooting_length == 0:
                    knot_index = math.floor(i / self.shooting_length)
                    print('   knot '+str(knot_index)+' :')
                    if i > 0 or self.tune_starting_point:
                        print('      pos: '+str(self.knot_poses[:, knot_index].detach().numpy()))
                        print('      vel: '+str(self.knot_vels[:, knot_index].detach().numpy()))
                print('         f['+str(i)+']: '+str(self.torques[:, i].detach().numpy()))
            print('flat grad: '+str(error))

        self.unflatten(flat)
