import diffdart as dart
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI
import numpy as np


class GUITrajectoryTrainer:
  def __init__(self, world: dart.simulation.World, problem: dart.trajectory.Problem,
               optimizer: dart.trajectory.IPOptOptimizer):
    self.world = world.clone()
    self.problem = problem
    self.optimizer = optimizer

    self.poses = np.zeros([world.getNumDofs(), self.problem.getNumSteps()])
    for i in range(self.problem.getNumSteps()):
      self.poses[:, i] = self.world.getPositions()

    # Set up a GUI
    self.gui = DartGUI()
    self.gui.serve(8080)
    self.gui.stateMachine().renderWorld(self.world, "world")
    self.optimizer.registerIntermediateCallback(self.afterOptimizationStep)

    # Set up the realtime animation
    self.ticker = dart.realtime.Ticker(self.world.getTimeStep() * 10)
    self.i = 0
    self.ticker.registerTickListener(self.onTick)
    self.gui.stateMachine().registerConnectionListener(self.onConnect)

    self.renderDuringTraining = True
    self.training = False

  def train(self, loopAfterSolve=False) -> dart.trajectory.Solution:
    self.training = True
    result = self.optimizer.optimize(self.problem)
    self.training = False
    if loopAfterSolve:
      rollout: dart.trajectory.TrajectoryRollout = self.problem.getRolloutCache(self.world)
      self.poses[:, :] = rollout.getPoses()
      self.gui.stateMachine().renderTrajectoryLines(self.world, self.poses)
      # Loop
      self.gui.stateMachine().blockWhileServing()
    return result

  def onTick(self, now):
    """
    This gets called periodically by our Ticker
    """
    if self.training and not self.renderDuringTraining:
      return
    self.world.setPositions(self.poses[:, self.i])
    # world.setVelocities(vels[:, i])
    self.gui.stateMachine().renderWorld(self.world, "world")
    self.i += 1
    if self.i >= self.poses.shape[1]:
      self.i = 0

  def onConnect(self):
    """
    This gets called whenever someone connects to the GUI. This should
    be idempotent code, since multiple people can connect / reconnect.
    """
    self.ticker.start()

  def afterOptimizationStep(
          self, problem: dart.trajectory.MultiShot, iter: int, loss: float, infeas: float):
    """
    This gets called after each step of optimization
    """
    if not self.renderDuringTraining:
      return True
    rollout: dart.trajectory.TrajectoryRollout = problem.getRolloutCache(self.world)
    self.poses[:, :] = rollout.getPoses()
    self.gui.stateMachine().renderTrajectoryLines(self.world, self.poses)
    return True

  def stateMachine(self) -> dart.server.GUIWebsocketServer:
    return self.gui.stateMachine()
