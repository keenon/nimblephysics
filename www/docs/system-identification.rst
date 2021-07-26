System Identification
================================

It's possible to use Nimble to fit the parameters of your simulation to observed real-world data.
We're actively working on adding more features to Nimble, but so far we support backprop into mass and inertia properties.

This document will walk you through constructing a super simple example to illustrate using backprop to learn inertia properties.
We'll again use a simple box, just like in :ref:`Backprop`.
We'll try to recover what the mass of our box should've been in order to reach a target position after 100 timesteps of a [10, 10] force in the x and y directions.

Note: Much like in :ref:`Backprop`, solving for mass like this is trivial and could be solved with simble analytical models. You `can` do much more complex optimization in Nimble, but this is a tutorial, so we're keeping it simple to start out.

If you're the type of person who prefers to just look at complete Python code, here's :download:`what we'll be building <./_static/robots/tune_mass.py>`.

Tuning mass
#################################################

Now that we've learned how to tune the initial velocity in :ref:`Backprop`, let's take a look at 
how to do the same for mass. The code is largely the same, except for a few
changes. First, we need to register with Nimble that we're going to be learning the mass of our box body::

  bound = np.zeros((1,))  # This is not used in the PyTorch API
  world.getWrtMass().registerNode(
      boxBody, 
      nimble.neural.WrtMassBodyNodeEntryType.MASS, 
      bound, 
      bound)

We also need to create a learnable `torch.Tensor` for mass::

  mass: torch.Tensor = torch.tensor([1.0], requires_grad=True)  # True mass is 2.0
  
Finally, pass the learnable mass tensor into the timestep function::

  state = nimble.timestep(world, state, action, mass)

Since mass is an optional argument, we only need to pass it in when we want to
optimize the mass value.

Here's the complete code::

  import torch
  import numpy as np
  import nimblephysics as nimble

  # Set up the world
  world = nimble.simulation.World()
  world.setGravity([0, -9.81, 0])
  world.setTimeStep(0.01)

  # Set up initial conditions for optimization
  initial_position: torch.Tensor = torch.tensor([3.0, 0.0])
  initial_velocity: torch.Tensor = torch.tensor([-3.0011, 4.8577])
  mass: torch.Tensor = torch.tensor([1.0], requires_grad=True)  # True mass is 2.0
  goal: torch.Tensor = torch.Tensor([[2.4739, 2.4768]])
  # We apply nonzero force so that mass can be determined from the trajectory.
  action: torch.Tensor = torch.tensor([10.0, 10.0])

  # Set up the box
  box = nimble.dynamics.Skeleton()
  boxJoint, boxBody = box.createTranslationalJoint2DAndBodyNodePair()
  world.addSkeleton(box)
  bound = np.zeros((1,))  # This is not used, so we just pass in zeros
  world.getWrtMass().registerNode(
      boxBody, 
      nimble.neural.WrtMassBodyNodeEntryType.MASS, 
      bound, 
      bound)


  while True:
      state: torch.Tensor = torch.cat((initial_position, initial_velocity), 0)
      states = [state]

      num_timesteps = 100
      for i in range(num_timesteps):
          state = nimble.timestep(world, state, action, mass)
          states.append(state)

      # Our loss is just the distance to the origin at the final step
      final_position = state[:world.getNumDofs()]  # Position is the first half of the state vector
      loss = (goal - final_position).norm()
      print('loss: '+str(loss))

      loss.backward()

      # Manually update weights using gradient descent. Wrap in torch.no_grad()
      # because weights have requires_grad=True, but we don't need to track this
      # in autograd.
      with torch.no_grad():
          learning_rate = 0.01
          mass -= learning_rate * mass.grad
          mass.grad = None

Automatically Initializing Inertia
#################################################

If you have custom colliders and you'd like to automatically compute inertia values for them, it's a straightforward process.

Recall how in :ref:`Backprop` you created :code:`boxBody: nimble.dynamics.BodyNode` and :code:`boxShape: nimble.dynamics.ShapeNode`.

To automatically compute and set inertia from the shape of colliders, all you need to do is::

  massOfBox = 1.0
  centerOfMass = [0.0, 0.0, 0.0]
  momentOfInertia = boxShape.getShape().computeInertia(massOfBox)
  boxBody.setInertia(nimble.dynamics.Inertia(massOfBox, centerOfMass, momentOfInertia))