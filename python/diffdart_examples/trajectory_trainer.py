import diffdart as dart
from diffdart import DartTorchLossFn, DartTorchTrajectoryRollout, DartGUI
import numpy as np
import time


class GUITrajectoryTrainer:
  def __init__(self, world: dart.simulation.World, problem: dart.trajectory.Problem,
               optimizer: dart.trajectory.IPOptOptimizer):
    self.world = world.clone()
    self.problem = problem
    self.optimizer = optimizer

    self.poses = np.zeros([world.getNumDofs(), self.problem.getNumSteps()])
    for i in range(self.problem.getNumSteps()):
      self.poses[:, i] = self.world.getPositions()

    self.poses_history = []
    self.loss_history = []
    self.time_history = []

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

    self.startTime = 0.0

  def train(self, loopAfterSolve=False) -> dart.trajectory.Solution:
    self.training = True
    self.startTime = time.time()
    result = self.optimizer.optimize(self.problem)
    self.training = False
    """
    with open('M_ctplt.npz', 'wb') as f:
      np.savez(f, T_m=self.time_history, C_m=self.loss_history)
    """
    if loopAfterSolve:
      rollout: dart.trajectory.TrajectoryRollout = self.problem.getRolloutCache(self.world)
      self.poses[:, :] = rollout.getPoses()
      self.gui.stateMachine().renderTrajectoryLines(self.world, self.poses)

      def onSlide(val: float):
        step = int(val)
        print('Sliding to '+str(step))
        try:
          self.poses[:, :] = self.poses_history[step]
          self.gui.stateMachine().renderTrajectoryLines(self.world, self.poses)
        except Exception as inst:
          print(type(inst))    # the exception instance
          print(inst.args)     # arguments stored in .args
          print(inst)

      self.gui.stateMachine().createSlider(
          'slider', [20, 20],
          [20, 150],
          0, len(self.loss_history) - 1,
          len(self.loss_history) - 1, onlyInts=True, horizontal=False, onChange=onSlide)

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
    rollout: dart.trajectory.TrajectoryRollout = problem.getRolloutCache(self.world)
    self.poses[:, :] = rollout.getPoses()
    self.time_history.append(time.time() - self.startTime)
    self.poses_history.append(np.copy(self.poses))
    self.loss_history.append(loss)
    if not self.renderDuringTraining:
      return True
    self.gui.stateMachine().renderTrajectoryLines(self.world, self.poses)
    return True

  def stateMachine(self) -> dart.server.GUIWebsocketServer:
    return self.gui.stateMachine()
