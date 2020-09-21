import dartpy as dart
from context import dart_torch
from dart_torch import DartTorchLossFn, DartTorchTrajectoryRollout
import torch


def main():
    world = dart.simulation.World()
    cartpole = dart.dynamics.Skeleton()
    cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
    world.addSkeleton(cartpole)

    # Set up the view

    def loss(rollout: DartTorchTrajectoryRollout):
        # print(rollout.getPoses('identity'))
        # return rollout.getPoses('ik')[2, -1]
        return rollout.getPoses('identity')[0, -1]

    dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)
    trajectory: dart.trajectory.MultiShot = dart.trajectory.MultiShot(
        world, dartLoss, 50, 10, False)

    ikMap: dart.neural.IKMapping = dart.neural.IKMapping(world)
    ikMap.addLinearBodyNode(cart)
    trajectory.addMapping('ik', ikMap)

    trajectory.getLoss(world)
    cache = trajectory.getRolloutCache(world)
    copy = cache.copy()
    loss = dartLoss.getLossAndGradient(cache, copy)

    optimizer = dart.trajectory.IPOptOptimizer()
    # optimizer.setCheckDerivatives(True)
    optimizer.setLBFGSHistoryLength(10)
    optimizer.setIterationLimit(10)
    optimizer.optimize(trajectory)

    dart.gui.glut.displayTrajectoryInGUI(world, trajectory)

    print("We're done!")


if __name__ == "__main__":
    main()
