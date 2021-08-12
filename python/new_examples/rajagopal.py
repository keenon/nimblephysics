import nimblephysics as nimble

# Create the world

world: nimble.simulation.World = nimble.simulation.World()
world.setGravity([0, -9.81, 0])
world.setTimeStep(0.01)

# Create the Rajagopal human body model (from the files shipped with Nimble, licensed under a separate MIT license)

skel: nimble.dynamics.Skeleton = nimble.models.RajagopalHumanBodyModel()
world.addSkeleton(skel)

# Run a gui

gui = nimble.NimbleGUI(world)
gui.serve(8080)
gui.blockWhileServing()
