import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI


def main():
    world = dart.simulation.World()
    world.setGravity([0, -9.81, 0])

    # Set up the 2D cartpole

    cartpole = dart.dynamics.Skeleton()
    cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
    cartRail.setAxis([1, 0, 0])
    cartShape = cart.createShapeNode(dart.dynamics.BoxShape([.5, .1, .1]))
    cartVisual = cartShape.createVisualAspect()
    cartVisual.setColor([0.5, 0.5, 0.5])
    cartRail.setPositionUpperLimit(0, 10)
    cartRail.setPositionLowerLimit(0, -10)
    cartRail.setForceUpperLimit(0, 10)
    cartRail.setForceLowerLimit(0, -10)

    poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
    poleJoint.setAxis([0, 0, 1])
    poleShape = pole.createShapeNode(dart.dynamics.BoxShape([.1, 1.0, .1]))
    poleVisual = poleShape.createVisualAspect()
    poleVisual.setColor([0.7, 0.7, 0.7])
    poleJoint.setForceUpperLimit(0, 0)
    poleJoint.setForceLowerLimit(0, 0)

    poleOffset = dart.math.Isometry3()
    poleOffset.set_translation([0, -0.5, 0])
    poleJoint.setTransformFromChildBodyNode(poleOffset)

    world.addSkeleton(cartpole)

    # Make simulations repeatable
    random.seed(1234)

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(world.getTimeStep()*10)

    def loss(rollout: DartTorchTrajectoryRollout):
        posLoss = rollout.getPoses('identity')[:, -1].square().sum()
        velLoss = rollout.getVels('identity')[:, -1].square().sum()
        return posLoss + velLoss
    dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

    world.setPositions([1, 1])

    def onReplan(time: int, rollout: dart.trajectory.TrajectoryRollout, duration: int):
        gui.stateMachine().renderTrajectoryLines(world, rollout.getPoses())
    mpc = dart.realtime.MPCLocal(world.clone(), dartLoss, 3000)
    mpc.registerReplaningListener(onReplan)
    mpc.setSilent(True)

    gui = DartGUI()
    gui.serve(8080)
    gui.stateMachine().renderWorld(world, "world")

    ticker = dart.realtime.Ticker(world.getTimeStep())
    originalColor = cartVisual.getColor()

    def onTick(now):
        world.setExternalForces(mpc.getForce(now))
        if "a" in gui.stateMachine().getKeysDown():
            perturbedForces = world.getExternalForces()
            perturbedForces[0] = -15.0
            world.setExternalForces(perturbedForces)
            cartVisual.setColor([1, 0, 0])
        elif "e" in gui.stateMachine().getKeysDown():
            perturbedForces = world.getExternalForces()
            perturbedForces[0] = 15.0
            world.setExternalForces(perturbedForces)
            cartVisual.setColor([0, 1, 0])
        else:
            cartVisual.setColor(originalColor)

        world.step()

        mpc.recordGroundTruthState(
            now, world.getPositions(),
            world.getVelocities(),
            world.getMasses())

        gui.stateMachine().renderWorld(world, "world")

    def onConnect():
        ticker.start()
        mpc.start()

    ticker.registerTickListener(onTick)
    gui.stateMachine().registerConnectionListener(onConnect)


if __name__ == "__main__":
    main()
