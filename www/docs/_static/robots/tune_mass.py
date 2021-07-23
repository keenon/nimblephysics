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
