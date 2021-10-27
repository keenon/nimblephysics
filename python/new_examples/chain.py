import nimblephysics as nimble
import torch

# Load the world

world = nimble.loadWorld("./chain.skel")
world.setTimeStep(0.0002)

# Fix weird shadows on the ground

world.getSkeleton("ground skeleton").getBodyNode(
    "ground").getShapeNode(0).getVisualAspect().setCastShadows(False)

# Set up initial state

initialState = torch.rand((world.getStateSize()))
action = torch.zeros((world.getActionSize()))

# Run a simulation for 300 timesteps

state = initialState
states = []
for _ in range(2000):
  state = nimble.timestep(world, state, action)
  states.append(state)

# Display our trajectory in a GUI

gui = nimble.NimbleGUI(world)
gui.serve(8080)  # host the GUI on localhost:8080
gui.loopStates(states)  # tells the GUI to animate our list of states
gui.blockWhileServing()  # block here so we don't exit the program
