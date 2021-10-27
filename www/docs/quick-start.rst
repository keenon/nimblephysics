Get Started in 2 Minutes
================================

Let's get the `Half Cheetah <https://gym.openai.com/envs/HalfCheetah-v2/>`_ from OpenAI Gym running in Nimble.

We'll start a new Python script by importing Nimble and PyTorch::

  import nimblephysics as nimble
  import torch

Download the :download:`half_cheetah.skel <./_static/robots/half_cheetah.skel>` file, and place it in the same directory as your Python script.

We'll load :code:`half_cheetah.skel`, and that'll define a world containing a half-cheetah robot and a floor::

  # Load the world

  world = nimble.loadWorld("./half_cheetah.skel")

Next we'll set up some PyTorch tensors to hold the state of our physics engine, and the current action command we're sending to the engine.
These are PyTorch tensors, and not numpy arrays, because we can backprop into them! (We'll do that in :ref:`Backprop`)

.. code-block:: python

  # Set up initial state

  initialState = torch.zeros((world.getStateSize()))
  action = torch.zeros((world.getActionSize()))

Now we're done with setup, and we're ready to run a physics simulation. A simulation is just a sequence of calls to the :code:`nimble.timestep()` function::

  # Run a simulation for 300 timesteps

  state = initialState
  states = []
  for _ in range(300):
    state = nimble.timestep(world, state, action)
    states.append(state)

The above code block runs 300 timesteps of simulation, and saves each state to the List :code:`states`.
Once we've saved a list of states from a simulation, we can use Nimble's built-in web-based GUI to see what we just built::

  # Display our trajectory in a GUI

  gui = nimble.NimbleGUI(world)
  gui.serve(8080) # host the GUI on localhost:8080
  gui.loopStates(states) # tells the GUI to animate our list of states
  gui.blockWhileServing() # block here so we don't exit the program

If you run the above code, and open up your web browser and visit `http://localhost:8080 <http://localhost:8080>`_ you should
see a looping rendering of a `Half Cheetah <https://gym.openai.com/envs/HalfCheetah-v2/>`_ robot collapsing onto the floor. You can use your left mouse button to rotate the
view, and the right mouse button to move the camera around.

Here's the whole file we just created, which loads and simulates the :download:`half_cheetah.skel <./_static/robots/half_cheetah.skel>` world, then displays the results::

  import nimblephysics as nimble
  import torch

  # Load the world

  world = nimble.loadWorld("./half_cheetah.skel")

  # Set up initial state

  initialState = torch.zeros((world.getStateSize()))
  action = torch.zeros((world.getActionSize()))

  # Run a simulation for 300 timesteps

  state = initialState
  states = []
  for _ in range(300):
    state = nimble.timestep(world, state, action)
    states.append(state)

  # Display our trajectory in a GUI

  gui = nimble.NimbleGUI(world)
  gui.serve(8080) # host the GUI on localhost:8080
  gui.loopStates(states) # tells the GUI to animate our list of states
  gui.blockWhileServing() # block here so we don't exit the program

To make robots do something more interesting than just falling on the floor, read the following tutorials for
how to implement different popular control strategies in Nimble. Or you could stop here and just invent your own!