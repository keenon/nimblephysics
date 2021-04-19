import nimblephysics as nimble
import os


world: nimble.simulation.World = nimble.simulation.World()
arm: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
    os.path.dirname(__file__), "KR5.urdf"))

# Your code here
# arm.setPositions([0, 0, -90*(3.1415/180), 0, 0, 0])

gui = nimble.NimbleGUI(world)
gui.serve(8080)
gui.blockWhileServing()
