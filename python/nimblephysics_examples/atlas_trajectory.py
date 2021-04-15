import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import nimblephysics as nimble
import os
from typing import Dict
from nimblephysics import NativeLossFn, NativeTrajectoryRollout, NativeTrajectoryTrainer


def get_bodies(world):
  all_body_nodes = []
  for skeleton_idx in range(world.getNumSkeletons()):
    all_body_nodes.extend(world.getSkeleton(skeleton_idx).getBodyNodes())
  return all_body_nodes


def set():
  world.setPositions(pose)
  gui.stateMachine().renderWorld(world, "world")
  print(pose)


def adv(i):
  for _ in range(i):
    world.step()
    gui.stateMachine().renderWorld(world, "world")


warrior2_pose = np.array([
    -1.570795,
    0.,
    0.,
    0.,
    -0.321,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    0.,
    1.57079633,
    0.,
    -1.57079633,
    1.57079633,
    0.,
    0.,
    -1.57079633,
    0.,
    -1.,
    0.,
    1.,
    0.,
])

warrior1_pose = np.array([
    -1.570795,
    0.0,
    0.0,
    0.0,
    -0.321,
    0.0,
    0.0,
    0.0,
    0.1,
    0.0,
    1.0,
    0.0,
    0.47079632679489647,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.4,
    0.0,
    0.0,
    1.57079633,
    0.0,
    -1.57079633,
    1.57079633,
    0.0,
    0.0,
    -1.57079633,
    0.0,
    -1.0,
    0.0,
    1.0,
    0.0,
])


world = nimble.simulation.World()
world.setGravity([0, -9.81, 0])
# world.setSlowDebugResultsAgainstFD(True)

# Set up skeleton
atlas: nimble.dynamics.Skeleton = world.loadSkeleton('../../data/sdf/atlas/atlas_v3_no_head.urdf')
atlas.setPosition(0, -0.5 * 3.14159)

ground: nimble.dynamics.Skeleton = world.loadSkeleton('../../data/sdf/atlas/ground.urdf')
floorBody: nimble.dynamics.BodyNode = ground.getBodyNode(0)
floorBody.getShapeNode(0).getVisualAspect().setColor([248./255., 248./255., 248./255.])
floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)


def computeStabilizingForces():
  snapshot: nimble.neural.BackpropSnapshot = nimble.neural.forwardPass(world, idempotent=True)

  forceVel = snapshot.getControlForceVelJacobian(world)
  velPreStep = snapshot.getPreStepVelocity()
  velPostStep = snapshot.getPostStepVelocity()
  velDelta = velPostStep - velPreStep

  # Trim small singular values manually, to avoid having explosions when we invert
  U, S, V = np.linalg.svd(forceVel)
  Sinv = S
  for i in range(len(S)):
    if abs(S[i]) < 1e-5:
      Sinv[i] = 0
    else:
      Sinv[i] = 1/S[i]
  # Invert, with tiny singular values now set to 0, instead of massive values
  forceVelInv = np.matmul(np.transpose(V), np.diag(Sinv), np.transpose(U))
  # Compute the control forces we'd need in order to stabilize
  stabilizingControls = np.matmul(forceVelInv, -velDelta)
  # Check how much velocity we'd actually accumulate here
  inverted = np.matmul(forceVel, stabilizingControls)
  atlas.setControlForces(stabilizingControls)

# forceLimits = np.ones([atlas.getNumDofs()]) * 500
# forceLimits[0:6] = 0
# atlas.setControlForceUpperLimits(forceLimits)
# atlas.setControlForceLowerLimits(forceLimits * -1)

# goal = torch.tensor([0.0, 2.0, -0.2])
# def loss(rollout: DartTorchTrajectoryRollout):
#   pos = rollout.getPoses('ik') # dofs x num_steps
#   last_pos = pos[:, -1]
#   return torch.sum(torch.square(last_pos - goal))

# nimbleLoss: nimble.trajectory.LossFn = DartTorchLossFn(loss)

# trajectory = nimble.trajectory.MultiShot(world, nimbleLoss, 400, 20, False)

# ikMap: nimble.neural.IKMapping = nimble.neural.IKMapping(world)
# handNode: nimble.dynamics.BodyNode = atlas.getBodyNode("l_hand")
# ikMap.addLinearBodyNode(handNode)
# trajectory.addMapping('ik', ikMap)
# trajectory.setParallelOperationsEnabled(True)


world.setPositions(warrior2_pose)
goal = torch.tensor(world.getPositions())


def loss(rollout: NativeTrajectoryRollout):
  pos = rollout.getPoses('identity')  # dofs x num_steps
  vel = rollout.getVels('identity')  # dofs x num_steps
  last_pos = pos[:, -1]
  last_vel = vel[:, -1]

  return torch.sum(torch.square(last_pos - goal)) + torch.sum(torch.square(last_vel))


nimbleLoss: nimble.trajectory.LossFn = NativeLossFn(loss)

trajectory = nimble.trajectory.SingleShot(world, nimbleLoss, 1, False)
optimizer = nimble.trajectory.SGDOptimizer()
optimizer.optimize(trajectory)


world.setPositions(warrior1_pose)
world.setPositions(warrior2_pose)

gui = NimbleGUI()
gui.serve(8080)
gui.stateMachine().renderWorld(world, "world")

ticker = nimble.realtime.Ticker(world.getTimeStep() / 10)

computeStabilizingForces()


def onTick(now):
  world.step()
  gui.stateMachine().renderWorld(world, "world")


def onConnect():
  ticker.start()


ticker.registerTickListener(onTick)
gui.stateMachine().registerConnectionListener(onConnect)

# adv(15)

gui.stateMachine().blockWhileServing()


# optimizer = nimble.trajectory.IPOptOptimizer()
# optimizer.setLBFGSHistoryLength(3)
# optimizer.setTolerance(1e-5)
# optimizer.setCheckDerivatives(False)
# optimizer.setIterationLimit(500)
# optimizer.setRecordPerformanceLog(False)

# trainer = GUITrajectoryTrainer(world, trajectory, optimizer)
# # Use a not-too-bright green for the goal sphere
# trainer.stateMachine().createSphere("goal_pos", 0.02, np.array(goal),
#  np.array([118/255, 224/255, 65/255]), True, False)
# result = trainer.train(loopAfterSolve=True)
