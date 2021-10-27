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

  pos: torch.Tensor = torch.zeros((skel.getNumDofs()), requires_grad=True)

  gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
  gui.serve(8080)
  gui.nativeAPI().renderBasis()

  while True:
    world.setPositions(pos.detach().numpy())
    gui.nativeAPI().renderWorld(world)

    # Position is the first half of the state vector
    lowestPoint: torch.Tensor = nimble.get_lowest_point(skel, pos, scales)
    print('lowest point: '+str(lowestPoint))

    # Our loss is just the distance to the origin at the final step
    loss = torch.square(lowestPoint - 1.0)
    print('loss: '+str(loss))

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    loss.backward()
    with torch.no_grad():
      learning_rate = 0.01
      pos -= learning_rate * pos.grad
      pos.grad = None
      """
      for key in scales:
        scales[key] -= learning_rate * scales[key].grad
        scales[key].grad = None
      """


if __name__ == "__main__":
  main()
