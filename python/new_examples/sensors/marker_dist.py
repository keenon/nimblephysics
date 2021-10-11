import torch
import nimblephysics as nimble


def main():
  # Set up the world
  world = nimble.simulation.World()
  world.setGravity([0, -9.81, 0])
  world.setTimeStep(0.01)

  # Set up the box
  osim: nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
  skel: nimble.dynamics.Skeleton = osim.skeleton
  world.addSkeleton(skel)

  # Set up initial conditions for optimization
  bodyNode: nimble.dynamics.BodyNode = osim.markersMap['C7'][0]
  markerOffset: torch.Tensor = torch.zeros((3), requires_grad=True)
  bodyScale: torch.Tensor = torch.zeros((3), requires_grad=True)
  with torch.no_grad():
    markerOffset[:] = torch.from_numpy(osim.markersMap['C7'][1])
    bodyScale[:] = torch.from_numpy(bodyNode.getScale())

  while True:
    # Position is the first half of the state vector
    closestPoint: torch.Tensor = nimble.get_marker_dist_to_nearest_vertex(
        bodyNode, markerOffset, bodyScale)
    print('closest point dist: '+str(closestPoint))

    # Our loss is just the distance to the origin at the final step
    loss = torch.square(closestPoint - 1.0)
    print('loss: '+str(loss))

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    loss.backward()
    with torch.no_grad():
      learning_rate = 0.01
      markerOffset -= learning_rate * markerOffset.grad
      markerOffset.grad = None
      bodyScale -= learning_rate * bodyScale.grad
      bodyScale.grad = None


if __name__ == "__main__":
  main()
