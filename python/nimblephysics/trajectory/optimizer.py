import nimblephysics as nimble
from .loss_fn import NativeLossFn, NativeTrajectoryRollout
import numpy as np


class NativeTrajectoryTrainer:
  def __init__(self, world: nimble.simulation.World, problem: nimble.trajectory.Problem,
               optimizer: nimble.trajectory.IPOptOptimizer, gui: nimble.NimbleGUI):
    self.world = world.clone()
    self.problem = problem
    self.optimizer = optimizer

    self.poses = np.zeros([world.getNumDofs(), self.problem.getNumSteps()])
    for i in range(self.problem.getNumSteps()):
      self.poses[:, i] = self.world.getPositions()

    # Set up a GUI
    self.gui = gui
    self.gui.stateMachine().renderWorld(self.world, "world")
    self.optimizer.registerIntermediateCallback(self.afterOptimizationStep)

    # Set up the realtime animation
    self.i = 0
    self.ticker.registerTickListener(self.onTick)
    self.gui.stateMachine().registerConnectionListener(self.onConnect)

    self.renderDuringTraining = True
    self.training = False

  def train(self, loopAfterSolve=False) -> nimble.trajectory.Solution:
    self.training = True
    result = self.optimizer.optimize(self.problem)
    self.training = False
    if loopAfterSolve:
      rollout: nimble.trajectory.TrajectoryRollout = self.problem.getRolloutCache(self.world)
      self.poses[:, :] = rollout.getPoses()
      self.gui.stateMachine().renderTrajectoryLines(self.world, self.poses)
      # Loop
      self.gui.stateMachine().blockWhileServing()
    return result

  def afterOptimizationStep(
          self, problem: nimble.trajectory.MultiShot, iter: int, loss: float, infeas: float):
    """
    This gets called after each step of optimization
    """
    if not self.renderDuringTraining:
      return True
    rollout: nimble.trajectory.TrajectoryRollout = problem.getRolloutCache(self.world)
    self.poses[:, :] = rollout.getPoses()
    self.gui.stateMachine().renderTrajectoryLines(self.world, self.poses)
    return True

  def stateMachine(self) -> nimble.server.GUIWebsocketServer:
    return self.gui.stateMachine()
