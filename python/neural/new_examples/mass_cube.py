import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout


def main():
    world = dart.simulation.World()
    world.setGravity([0, -9.81, 0])

    # Set up the 2D cube

    cube = dart.dynamics.Skeleton()
    cubeAxis, cubeBody = cube.createPrismaticJointAndBodyNodePair()
    cubeAxis.setAxis([1, 0, 0])
    cubeShape = cubeBody.createShapeNode(dart.dynamics.BoxShape([.5, .1, .1]))
    cubeVisual = cubeShape.createVisualAspect()
    cubeVisual.setColor([0.5, 0.5, 0.5])
    cubeAxis.setPositionUpperLimit(0, 10)
    cubeAxis.setPositionLowerLimit(0, -10)
    cubeAxis.setForceUpperLimit(0, 10)
    cubeAxis.setForceLowerLimit(0, -10)
    world.addSkeleton(cube)

    # Set the cube body mass as tunable
    world.tuneMass(cubeBody, dart.neural.WrtMassBodyNodeEntryType.MASS, [5.0], [0.1])

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(1e-2)

    # Get original data
    TRUE_MASS = 2.5
    MASS_PRIOR = 2.0
    STEPS = 25
    USE_MASS_PRIOR = True

    cubeBody.setMass(TRUE_MASS)
    world.setPositions([0])
    world.setVelocities([0])
    trueForces = [2.0]
    for i in range(STEPS):
        world.setForces(trueForces)
        world.step()
    finalPosition = world.getPositions()[0]

    # Reset world, and scramble mass
    world.setPositions([0])
    world.setVelocities([0])
    cubeBody.setMass(0.2)

    def loss(rollout: DartTorchTrajectoryRollout):
        diff = rollout.getPoses()[0, -1] - finalPosition
        if USE_MASS_PRIOR:
            massPrior = 1e-2 * (MASS_PRIOR - rollout.getMasses()[0])
            return (diff * diff) + (massPrior * massPrior)
        else:
            return diff * diff

    dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

    trajectory = dart.trajectory.MultiShot(world, dartLoss, STEPS, 5, False)
    for i in range(STEPS):
        trajectory.pinForce(i, trueForces)

    optimizer = dart.trajectory.IPOptOptimizer()
    optimizer.setLBFGSHistoryLength(5)
    optimizer.setTolerance(1e-6)
    optimizer.setCheckDerivatives(False)
    optimizer.setIterationLimit(500)
    result = optimizer.optimize(trajectory)

    print('Recovered mass: ' + str(cubeBody.getMass()))

    # json = result.toJson(world)
    # text_file = open("cube.txt", "w")
    # n = text_file.write(json)
    # text_file.close()

    # dart.dart_serve_web_gui(json)


if __name__ == "__main__":
    main()
