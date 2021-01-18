import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import diffdart as dart
from typing import Dict
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI


def main():
    world = dart.simulation.World()
    world.setGravity([0, -9.81, 0])

    # Set up the 2D cartpole

    atlas = world.loadSkeleton("dart://sample/sdf/atlas/atlas_v3_no_head.sdf")
    ground = world.loadSkeleton("dart://sample/sdf/atlas/ground.urdf")

    # Set up a GUI

    gui = DartGUI()
    gui.serve(8080)
    gui.stateMachine().renderWorld(world, "world")

    ticker = dart.realtime.Ticker(world.getTimeStep())

    def onTick(now):
        world.step()
        gui.stateMachine().renderWorld(world, "world")

    def onConnect():
        ticker.start()
    ticker.registerTickListener(onTick)
    gui.stateMachine().registerConnectionListener(onConnect)

    # Set up the view

    """
    def loss(rollout: DartTorchTrajectoryRollout):
        pos = rollout.getPoses('ik')
        vel = rollout.getVels('ik')
        step_loss = - torch.sum(pos[1, :] * pos[1, :] * torch.sign(pos[1, :]))
        last_pos_y = pos[1, -1]
        last_vel_y = vel[1, -1]
        final_loss = - 100 * torch.square(last_pos_y) * torch.sign(last_pos_y)
        return step_loss + final_loss
    dartLoss: dart.trajectory.LossFn = DartTorchLossFn(loss)

    trajectory = dart.trajectory.MultiShot(world, dartLoss, 400, 20, False)

    ikMap: dart.neural.IKMapping = dart.neural.IKMapping(world)
    ikMap.addLinearBodyNode(root)
    trajectory.addMapping('ik', ikMap)

    trajectory.setParallelOperationsEnabled(True)
    optimizer = dart.trajectory.IPOptOptimizer()
    optimizer.setLBFGSHistoryLength(5)
    optimizer.setTolerance(1e-5)
    optimizer.setCheckDerivatives(False)
    optimizer.setIterationLimit(150)
    optimizer.setRecordPerformanceLog(True)
    result: dart.trajectory.Solution = optimizer.optimize(trajectory)
    # perflogs: Dict[str, dart.performance.FinalizedPerformanceLog] = result.getPerfLog().finalize()
    # print(perflog)

    json = result.toJson(world)
    text_file = open("worm.txt", "w")
    n = text_file.write(json)
    text_file.close()

    dart.dart_serve_optimization_solution(result, world)
    """


if __name__ == "__main__":
    main()
