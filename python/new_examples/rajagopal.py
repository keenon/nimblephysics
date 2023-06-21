import nimblephysics as nimble

# Create the world

world: nimble.simulation.World = nimble.simulation.World()
world.setGravity([0, -9.81, 0])
world.setTimeStep(0.01)

# Create the Rajagopal human body model (from the files shipped with Nimble, licensed under a separate MIT license)

skel: nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
world.addSkeleton(skel.skeleton)

# Run a gui

gui = nimble.NimbleGUI(world)
gui.serve(8080)
gui.nativeAPI().renderWorld(world)

# Animate the knees back and forth
ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(
    world.getTimeStep() * 10)


def onTick(now):
    progress = (now % 2000) / 2000.0


ticker.registerTickListener(onTick)
ticker.start()

# Don't immediately exit

gui.blockWhileServing()
