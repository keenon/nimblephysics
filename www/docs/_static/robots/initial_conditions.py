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
