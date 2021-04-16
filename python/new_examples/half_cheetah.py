import nimblephysics as nimble
import torch

# Load the world

world = nimble.loadWorld("half_cheetah.skel")

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
gui.loopStates(states)
gui.serve(8080)
gui.blockWhileServing()
