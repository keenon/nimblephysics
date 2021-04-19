import nimblephysics as nimble
import os


world: nimble.simulation.World = nimble.simulation.World()
arm: nimble.dynamics.Skeleton = world.loadSkeleton(os.path.join(
    os.path.dirname(__file__), "./KR5.urdf"))

# Your code here

gui = nimble.NimbleGUI(world)
gui.serve(8080)
gui.blockWhileServing()
