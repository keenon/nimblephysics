import torch
import nimblephysics as nimble


def main():
  # Set up the world
  world = nimble.simulation.World()
  world.setGravity([0, -9.81, 0])
  world.setTimeStep(0.01)

  # Set up the box
  skel: nimble.dynamics.Skeleton = nimble.models.RajagopalHumanBodyModel().skeleton
  world.addSkeleton(skel)

  # Set up initial conditions for optimization
  scales: Dict[str, torch.Tensor] = {}
  for i in range(skel.getNumBodyNodes()):
    body: nimble.dynamics.BodyNode = skel.getBodyNode(i)
    scales[body.getName()] = torch.tensor(body.getScale(), requires_grad=True)

  while True:
    # Position is the first half of the state vector
    height: torch.Tensor = nimble.get_height(skel, skel.getPositions(), scales)
    print('height: '+str(height))

    # Our loss is just the distance to the origin at the final step
    loss = torch.square(height - 1.8)
    print('loss: '+str(loss))

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    loss.backward()
    with torch.no_grad():
      learning_rate = 0.01
      for key in scales:
        scales[key] -= learning_rate * scales[key].grad
        scales[key].grad = None


if __name__ == "__main__":
  main()
