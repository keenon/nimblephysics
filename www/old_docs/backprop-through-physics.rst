.. _Backprop:

Backprop through Physics Timesteps
==========================================

This tutorial assumes you're already familiar with PyTorch. If not, we recommend `this tutorial <https://pytorch.org/tutorials/beginner/pytorch_with_examples.html>`_ as a good place to start.

At a high level, all you need to know about using Nimble is that :code:`nimble.timestep(...)` is a valid PyTorch function, and everything else is going to work how you'd expect.

This document will walk you through constructing a super simple example to illustrate passing gradients through physics.
We'll have a box starting at a fixed location, flying through space towards a target for 100 timesteps.
We'll use PyTorch to help us find an initial velocity for the box so that it ends up as close to the target as possible.

Note: Yes, solving for initial conditions like this is trivial and could be solved with simble analytical models. You `can` do much more complex optimization in Nimble, but this is a tutorial, so we're keeping it simple to start out.

If you're the type of person who prefers to just look at complete Python code, here's :download:`what we'll be building <./_static/robots/initial_conditions.py>`.

Setting up the optimization problem
#################################################

As always, we start by importing Nimble and PyTorch::

  import torch
  import nimblephysics as nimble

Next up, we'll create our :code:`world` and our :code:`box` (to better understand this code make sure you've read :ref:`Worlds`). We'll build our skeletons manually::

  # Set up the world
  world = nimble.simulation.World()
  world.setGravity([0, -9.81, 0])
  world.setTimeStep(0.01)

  # Set up the box
  box = nimble.dynamics.Skeleton()
  boxJoint, boxBody = box.createTranslationalJoint2DAndBodyNodePair()
  boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([.1, .1, .1]))
  boxVisual = boxShape.createVisualAspect()
  boxVisual.setColor([0.5, 0.5, 0.5])
  world.addSkeleton(box)

Now that we've got a :code:`world` and a :code:`box`, we're going to set up our :code:`torch.Tensor`'s that we're going to optimize over::

  # Set up initial conditions for optimization
  initial_position: torch.Tensor = torch.tensor([3.0, 0.0])
  initial_velocity: torch.Tensor = torch.zeros((world.getNumDofs()), requires_grad=True)

Note that we don't set :code:`requires_grad=True` for the :code:`initial_position` Tensor, because we won't be optimizing the :code:`initial_position`.

Once we've set up our world and our tensors, we'll set up a GUI so that we can see what we're doing::

  # Set up the GUI
  gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
  gui.serve(8080)
  gui.nativeAPI().createSphere("goal", radius=0.1, pos=[0, 0, 0], color=[0, 255, 0])

That last line, where we access the :code:`gui.nativeAPI()`, is to create a little green sphere in the GUI so that we can easily see where we want the box to end up.

Doing the actual optimization
#################################################

Now that we've got all the preliminaries out of the way, we can start doing the exciting part!
We'll do the following steps over and over again until we're happy with the result:

1. Run a simulation for 100 timesteps
2. Calculate loss (the difference between the box's final position and our goal)
3. Backpropagate the loss through time (using PyTorch's :code:`loss.backward()`)
4. Update our initial conditions by taking a step in the direction of the gradient

To run a forward simulation, we'll just call :code`nimble.timestep()` over and over again. Remember from :ref:`Worlds` that "state" is just "position" and "velocity" concatenated together::

  state: torch.Tensor = torch.cat((initial_position, initial_velocity), 0)

  num_timesteps = 100
  for i in range(num_timesteps):
    state = nimble.timestep(world, state, torch.zeros((world.getNumDofs())))

As a quality of life improvement, in order to actually see what we're doing while learning is happening, we'll need to save the intermediate states into a List, and pass them to our GUI. Let's do that::

  state: torch.Tensor = torch.cat((initial_position, initial_velocity), 0)
  states = [state]

  num_timesteps = 100
  for i in range(num_timesteps):
    state = nimble.timestep(world, state, torch.zeros((world.getNumDofs())))
    states.append(state)

  # This call will overwrite any previous set of states we were looping from
  # a previous iteration of gradient descent.
  gui.loopStates(states)

With that improvement in place, we can visit `http://localhost:8080 <http://localhost:8080>`_ during training and see the current trajectory that the optimizer is exploring, and watch as it updates over time.

After we've run our simulation, we need to compute the loss from our simulation::

  # Our loss is just the distance to the origin at the final step
  final_position = state[:world.getNumDofs()]  # Position is the first half of the state vector
  loss = final_position.norm()
  print('loss: '+str(loss))

Since our goal position is just the origin, it's sufficient to call :code:`final_position.norm()` to get the distance to the origin for our trajectory.

Now that we've got a PyTorch Tensor holding a single value representing loss, we can use :code:`loss.backward()` to have PyTorch run backprop for us::

  loss.backward()

The only reason that the above call works is because :code:`nimble.timestep()` is a fully differentiable operator that PyTorch can understand. Very cool stuff.

Last but not least, we can't forget to actually update the initial velocity of our box! We'll use an incredibly primitive learning algorithm to update our weights. We'll just multiply the gradient by 0.01, and add it to the old value::

  # Manually update weights using gradient descent. Wrap in torch.no_grad()
  # because weights have requires_grad=True, but we don't need to track this
  # in autograd.
  with torch.no_grad():
    learning_rate = 0.01
    initial_velocity -= learning_rate * initial_velocity.grad
    initial_velocity.grad = None

That's it! Now if we perform those operations over and over again, we'll be able to watch our initial velocity change until the box's arc ends up exactly at the origin.

Complete Code
#################################################

Here's :download:`the complete code <./_static/robots/initial_conditions.py>` we just wrote, or if you'd prefer to copy-paste::

  import torch
  import nimblephysics as nimble

  # Set up the world
  world = nimble.simulation.World()
  world.setGravity([0, -9.81, 0])
  world.setTimeStep(0.01)

  # Set up the box
  box = nimble.dynamics.Skeleton()
  boxJoint, boxBody = box.createTranslationalJoint2DAndBodyNodePair()
  boxShape = boxBody.createShapeNode(nimble.dynamics.BoxShape([.1, .1, .1]))
  boxVisual = boxShape.createVisualAspect()
  boxVisual.setColor([0.5, 0.5, 0.5])
  world.addSkeleton(box)

  # Set up initial conditions for optimization
  initial_position: torch.Tensor = torch.tensor([3.0, 0.0])
  initial_velocity: torch.Tensor = torch.zeros((world.getNumDofs()), requires_grad=True)

  # Set up the GUI
  gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
  gui.serve(8080)
  gui.nativeAPI().createSphere("goal", radius=0.1, pos=[0, 0, 0], color=[0, 255, 0])

  while True:
    state: torch.Tensor = torch.cat((initial_position, initial_velocity), 0)
    states = [state]

    num_timesteps = 100
    for i in range(num_timesteps):
      state = nimble.timestep(world, state, torch.zeros((world.getNumDofs())))
      states.append(state)

    # This call will overwrite any previous set of states we were looping from
    # a previous iteration of gradient descent.
    gui.loopStates(states)

    # Our loss is just the distance to the origin at the final step
    final_position = state[:world.getNumDofs()]  # Position is the first half of the state vector
    loss = final_position.norm()
    print('loss: '+str(loss))

    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
      learning_rate = 0.01
      initial_velocity -= learning_rate * initial_velocity.grad
      initial_velocity.grad = None
