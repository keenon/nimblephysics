import dartpy as dart
from dart_layer import dart_layer, BackpropSnapshotPointer
import torch
from typing import Tuple, Callable, List
import numpy as np
import math
import ipyopt


class MultipleShootingTrajectory:
    """
    This manages a trajectory optimization problem in a re-usable way
    """

    def __init__(
            self,
            world: dart.simulation.World,
            step_loss: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, dart.simulation.World],
                                float],
            final_loss: Callable[[torch.Tensor, torch.Tensor, dart.simulation.World],
                                 float],
            steps: int = 1000,
            shooting_length: int = 50,
            tune_starting_point: bool = False,
            enforce_loop: bool = False,
            disable_actuators: List[int] = []):
        """
        Constructor
        """
        self.world = world
        self.step_loss = step_loss
        self.final_loss = final_loss
        self.steps = steps
        self.shooting_length = min(shooting_length, steps)
        self.disable_actuators = disable_actuators
        self.tune_starting_point = tune_starting_point
        self.enforce_loop = enforce_loop

        zero_vector = [0] * world.getNumDofs()

        self.num_shots = math.floor(steps / shooting_length)

        # Initialize the learnable torques
        self.torques = [torch.tensor(zero_vector, dtype=torch.float64, requires_grad=True)
                        for _ in range(steps)]
        # Initialize knot points
        self.knot_poses = [
            # The first knot point is the starting point
            torch.tensor(
                world.getPositions(),
                dtype=torch.float64, requires_grad=tune_starting_point)] + [
            # The other knot points
            torch.tensor(zero_vector, dtype=torch.float64, requires_grad=True)
            for _ in range(self.num_shots - 1)]
        self.knot_vels = [
            # The first knot point is the starting point
            torch.tensor(
                world.getVelocities(),
                dtype=torch.float64, requires_grad=tune_starting_point)] + [
            # The other knot points
            torch.tensor(zero_vector, dtype=torch.float64, requires_grad=True)
            for _ in range(self.num_shots - 1)]

        self.mask = torch.tensor([1] * world.getNumDofs(), requires_grad=False)
        for j in disable_actuators:
            self.mask[j] = 0

        self.last_x = np.zeros(self.get_flat_problem_dim())
        self.snapshots: List[dart.neural.BackpropSnapshot] = [None] * self.steps
        self.last_loss = torch.tensor([0], requires_grad=False)
        self.last_preknot_poses = [torch.tensor(
            zero_vector, dtype=torch.float64, requires_grad=False)] * self.num_shots
        self.last_preknot_vels = [torch.tensor(
            zero_vector, dtype=torch.float64, requires_grad=False)] * self.num_shots
        self.last_terminal_pos = torch.tensor(
            zero_vector, dtype=torch.float64, requires_grad=False)
        self.last_terminal_vel = torch.tensor(
            zero_vector, dtype=torch.float64, requires_grad=False)

    def tensors(self):
        t = self.torques + self.knot_vels[1:] + self.knot_poses[1:]
        if self.tune_starting_point:
            t = [self.knot_poses[0], self.knot_poses[0]] + t
        return t

    def unroll(
            self,
            use_knots: bool = True,
            after_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor],
                                 None] = None):
        """
        Returns total loss
        """
        pos = self.knot_poses[0]
        vel = self.knot_vels[0]

        loss = torch.tensor([0], dtype=torch.float, requires_grad=True)
        knot_loss = torch.tensor([0], dtype=torch.float, requires_grad=True)

        for i in range(self.steps):
            t = self.torques[i] * self.mask

            # We're at a knot point
            if i % self.shooting_length == 0 and i > 0 and use_knots:
                knot_index = math.floor(i / self.shooting_length)
                knot_pos = self.knot_poses[knot_index]
                knot_vel = self.knot_vels[knot_index]

                # Record the loss from the error at this knot
                this_knot_loss = (knot_pos - pos).norm() + 10 * (knot_vel - vel).norm()
                knot_loss = knot_loss + this_knot_loss

                # Now reset to the actual knot position
                pos = knot_pos
                vel = knot_vel

            loss = loss + self.step_loss(pos, vel, t, self.world)

            pointer = BackpropSnapshotPointer()
            pos, vel, snapshot = dart_layer(self.world, pos, vel, t, pointer)
            self.snapshots[i] = snapshot.backprop_snapshot

            if after_step is not None:
                after_step(pos, vel, t)

        loss = loss + self.final_loss(pos, vel, self.world)

        if self.enforce_loop:
            self.last_terminal_pos = pos
            self.last_terminal_vel = vel
            start_pos = self.knot_poses[0]
            start_vel = self.knot_vels[0]
            # Record the loss from the error at the final knot
            loop_loss = (pos - start_pos).norm() + 10 * (vel - start_vel).norm()
            knot_loss = knot_loss + loop_loss

        self.last_loss = loss

        return loss, knot_loss

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
        dims -= (self.steps % self.shooting_length) * world_dofs

        return dims

    def get_constraint_dim(self):
        """
        This gets the total number of constraint dimensions
        """
        world_dofs = self.world.getNumDofs()
        knot_phase_dim = world_dofs * 2

        num_knot_points = self.num_shots - 1  # Ignore the starting point

        if self.enforce_loop:
            num_knot_points += 1

        return num_knot_points * knot_phase_dim

    def flatten(self, out: np.ndarray, grad: bool = False):
        """
        This turns our current tensors into a single X vector that we can pass to a solver like IPOPT
        """
        assert len(out) == self.get_flat_problem_dim()
        world_dofs = self.world.getNumDofs()

        flat_cursor = 0
        torque_cursor = 0
        for i in range(self.num_shots):
            if i > 0 or self.tune_starting_point:
                out[flat_cursor:flat_cursor+world_dofs] = self.knot_poses[i].detach() if not grad else self.knot_poses[i].grad
                flat_cursor += world_dofs
                out[flat_cursor:flat_cursor+world_dofs] = self.knot_vels[i].detach() if not grad else self.knot_vels[i].grad
                flat_cursor += world_dofs

            shot_length = min(self.shooting_length, len(self.torques) - torque_cursor)
            for j in range(shot_length):
                out[flat_cursor:flat_cursor+world_dofs] = self.torques[torque_cursor].detach() if not grad else self.torques[i].grad
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
                self.knot_poses[i].data = torch.tensor(x[flat_cursor:flat_cursor+world_dofs])
                flat_cursor += world_dofs
                self.knot_vels[i].data = torch.tensor(x[flat_cursor:flat_cursor+world_dofs])
                flat_cursor += world_dofs

            shot_length = min(self.shooting_length, len(self.torques) - torque_cursor)
            for j in range(shot_length):
                self.torques[torque_cursor].data = torch.tensor(
                    x[flat_cursor: flat_cursor + world_dofs])
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

            shot_length = min(self.shooting_length, len(self.torques) - torque_cursor)
            torque_cursor += shot_length
            flat_cursor += shot_length * world_dofs

        return knot_offsets

    def get_knot_g_offsets(self):
        """
        This returns a set of offset pointers into the g(x) vector for each knot
        """
        knot_offsets = []

        world_dofs = self.world.getNumDofs()
        flat_cursor = 0
        if self.enforce_loop:
            flat_cursor += world_dofs * 2
        for i in range(self.num_shots):
            if i > 0 or self.tune_starting_point:
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
        if (x != self.last_x).any():
            self.last_x = x
            self.unflatten(x)
            self.unroll()

    def eval_f(self, x: np.ndarray):
        """
        This returns the loss from a given trajectory encoded by x.
        """
        self.ensure_fresh_rollout(x)
        return self.last_loss.item()

    def eval_grad_f(self, x: np.ndarray, out: np.ndarray):
        """
        This returns the gradient of loss at the trajectory encoded by x.
        """
        assert(len(out) == self.get_flat_problem_dim())
        self.ensure_fresh_rollout(x)
        # Zero out the gradient
        for tensor in self.tensors():
            tensor.grad.data.zero_()
        # Run backprop through our last loss
        self.last_loss.backward()
        # Flatten the gradient into a vector
        return self.flatten(out, grad=True)

    def eval_g(self, x: np.ndarray, out: np.ndarray):
        """
        This returns the gap at knot points for a given trajectory encoded by x.
        """
        self.ensure_fresh_rollout(x)
        assert(len(out) == self.get_constraint_dim())

        world_dofs = self.world.getNumDofs()
        cursor = 0

        if self.enforce_loop:
            out[cursor:cursor+world_dofs] = self.last_terminal_pos - self.knot_poses[0]
            out[cursor:cursor+world_dofs] = self.last_terminal_vel - self.knot_vels[0]

        for i in range(self.num_shots):
            if i == 0:
                continue
            out[cursor:cursor+world_dofs] = self.last_preknot_poses[i] - self.knot_poses[i]
            out[cursor:cursor+world_dofs] = self.last_preknot_vels[i] - self.knot_vels[i]

        return out

    def eval_jac_g(self, x: np.ndarray, out: np.ndarray):
        """
        This computes a Jacobian of g(x)
        """
        self.ensure_fresh_rollout(x)

        world_dofs = self.world.getNumDofs()
        cursor = 0
        dt = self.world.getTimeStep()

        def insertNegativeIdentity():
            """
            This puts a negative identity matrix into the output values
            """
            for i in range(world_dofs*2):
                out[cursor] = -1.0
                cursor += 1

        def insertFullJacobian(last_knot_index: int):
            """
            This computes a full Jacobian relating the whole shot to the error at this knot-point (last_knot_index + 1)
            """
            start_index = last_knot_index * self.shooting_length
            end_index_exclusive = (last_knot_index + 1) * self.shooting_length
            """
            For our purposes here (forward Jacobians), the forward computation 
            graph looks like this:

            p_t -------------+--------------------------------> p_t+1 ---->
                              \                                   /
                               \                                 /
            v_t ----------------+----(LCP Solver)----> v_t+1 ---+---->
                               /
                              /
            f_t -------------+
            """

            # p_t+1 --> p_end
            pos_pos_end = np.identity(world_dofs)
            # p_t+1 --> v_end
            pos_vel_end = np.identity(world_dofs)
            # v_t+1 --> p_end
            vel_pos_end = np.identity(world_dofs)
            # v_t+1 --> v_end
            vel_vel_end = np.identity(world_dofs)

            for i in reversed(range(start_index, end_index_exclusive)):
                snapshot: dart.neural.BackpropSnapshot = self.snapshots[i]

                # p_t --> v_t+1
                pos_vel = snapshot.getPosVelJacobian()
                # v_t --> v_t+1
                vel_vel = snapshot.getVelVelJacobian()
                # f_t --> v_t+1
                force_vel = snapshot.getForceVelJacobian()

                # p_t --> p_t+1 = (p_t --> p_t+1) + dt*(p_t --> v_t+1)
                pos_pos = snapshot.getPosPosJacobian() + dt * pos_vel
                # v_t --> p_t+1 = dt*(v_t --> v_t+1)
                vel_pos = dt * vel_vel
                # f_t --> p_t+1
                force_pos = dt * force_vel

                # f_t --> p_end = ((p_t+1 --> p_end) * (f_t --> p_t+1)) + ((v_t+1 --> p_end) * (f_t --> v_t+1))
                force_pos_end = (pos_pos_end * force_pos) + (vel_pos_end * force_vel)
                # f_t --> v_end ...
                force_vel_end = (pos_vel_end * force_pos) + (vel_vel_end * force_vel)

                # Write our force_pos_end and force_vel_end into the output, row by row
                for row in range(world_dofs):
                    for col in range(world_dofs):
                        out[cursor] = force_pos_end[row][col]
                        cursor += 1
                for row in range(world_dofs):
                    for col in range(world_dofs):
                        out[cursor] = force_vel_end[row][col]
                        cursor += 1

                # Update p_t+1 --> p_end = ((p_t+1 --> p_end) * (p_t --> p_t+1)) + (v_t+1 --> p_end) * (p_t --> v_t+1)
                pos_pos_end = (pos_pos_end * pos_pos) + (vel_pos_end * pos_vel)
                # Update p_t+1 --> v_end ...
                pos_vel_end = (pos_vel_end * pos_pos) + (vel_vel_end * pos_vel)
                # Update v_t+1 --> p_end ...
                vel_pos_end = (pos_pos_end * vel_pos) + (vel_pos_end * vel_vel)
                # Update v_t+1 --> v_end ...
                vel_vel_end = (pos_vel_end * vel_pos) + (vel_vel_end * vel_vel)

            # Put these so the rows correspond to a whole phase vector
            phase_pos_end = np.concatenate([pos_pos_end, vel_pos_end], axis=1)
            phase_vel_end = np.concatenate([pos_vel_end, vel_vel_end], axis=1)
            for row in range(world_dofs):
                for col in range(world_dofs * 2):
                    out[cursor] = phase_pos_end[row][col]
                    cursor += 1
            for row in range(world_dofs):
                for col in range(world_dofs * 2):
                    out[cursor] = phase_vel_end[row][col]
                    cursor += 1

        if self.enforce_loop:
            # Insert the Jacobian for the last knot point -> the last timestep
            insertFullJacobian(self.num_shots - 1)
            # This means we're tuning the initial position, so our Jac for this knot
            # needs to include a negative identity
            if self.tune_starting_point:
                insertNegativeIdentity()

        for i in range(self.num_shots):
            if i == 0:
                continue
            # Insert the Jacobian for the last knot point -> the last timestep
            insertFullJacobian(i-1)
            # Insert a -I relating this knot's position to this knot's constraint
            insertNegativeIdentity()

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

        def insertFullJacobian(rowTopLeft: int, colTopLeft: int):
            """
            This puts space for the whole Jacobian for a given constraint into the sparsity indices.
            This is the phase-phase Jacobian, and all the torque-phase Jacobians
            """
            # The torques are all directly to the right of the knot-point phase variables for the last shooting section
            torque_jac_locations: List[int] = []

            cursor = colTopLeft + world_dofs*2
            for i in range(self.shooting_length):
                torque_jac_locations.append(cursor)
                cursor += world_dofs
                if cursor >= n:
                    break

            # Add the torque Jacobians in reverse, because it will make it more memory efficient to compute
            torque_jac_locations.reverse()
            for cursor in torque_jac_locations:
                insertTorquePhaseJacobian(rowTopLeft, cursor)
            insertPhasePhaseJacobian(rowTopLeft, colTopLeft)

        knot_x_offsets = self.get_knot_x_offsets()
        knot_g_offsets = self.get_knot_g_offsets()

        if self.enforce_loop:
            # Insert the Jacobian for the last knot point -> the last timestep
            insertFullJacobian(0, knot_x_offsets[len(knot_x_offsets)-1])
            # This means we're not tuning the initial position, so our Jac for this knot
            # can just be the Jac for the last timestep
            if knot_x_offsets[0] is None:
                assert(not self.tune_starting_point)
            # This means this is basically an ordinary constraint, and also has a negative
            # identity relationship to the original knot
            else:
                insertNegativeIdentity(0, knot_x_offsets[0])

        for i in range(self.num_shots):
            if i == 0:
                continue
            # Insert the Jacobian for the last knot point -> the last timestep
            insertFullJacobian(knot_g_offsets[i], knot_x_offsets[i-1])
            # Insert a -I relating this knot's position to this knot's constraint
            insertNegativeIdentity(knot_g_offsets[i], knot_x_offsets[i])

        return (np.array(row), np.array(col))

    def ipopt(self):
        """
        This uses IPOPT to try to get the bounds really tight on the knot points
        """
        n = self.get_flat_problem_dim()
        eps = 1e-2

        lower_bounds = np.array([-2e19] * n)
        upper_bounds = np.array([2e19] * n)

        num_constraints = self.get_constraint_dim()
        constraint_upper_bounds = np.array([eps] * num_constraints)
        constraint_lower_bounds = np.array([-eps] * num_constraints)

        jac_g_sparsity_indices = self.get_jac_g_sparsity_indices()

        nlp = ipyopt.Problem(
            n, lower_bounds, upper_bounds, num_constraints, constraint_lower_bounds,
            constraint_upper_bounds, jac_g_sparsity_indices, 0,
            self.eval_f, self.eval_grad_f, self.eval_g, self.eval_jac_g)

        print("Going to call IPOPT solve")
        x0 = self.flatten(np.zeros(n))
        _x, loss, status = nlp.solve(x0)
        print("IPOPT finished with final loss "+str(loss)+", "+str(status))
        self.unflatten(_x)
