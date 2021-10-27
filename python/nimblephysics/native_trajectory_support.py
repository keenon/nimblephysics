import torch
import nimblephysics_libs._nimblephysics as nimble
from typing import Dict, Callable
import numpy as np


class NativeTrajectoryRollout:
  def __init__(self, rollout: nimble.trajectory.TrajectoryRollout):
    self.rollout = rollout

    self.posTensors: Dict[str, torch.Tensor] = {}
    self.velTensors: Dict[str, torch.Tensor] = {}
    self.forceTensors: Dict[str, torch.Tensor] = {}
    self.massTensor: torch.Tensor = torch.tensor(
        rollout.getMasses(), requires_grad=True)

    for mapping in rollout.getMappings():
      self.posTensors[mapping] = torch.tensor(
          rollout.getPoses(mapping), requires_grad=True)
      self.velTensors[mapping] = torch.tensor(rollout.getVels(mapping), requires_grad=True)
      self.forceTensors[mapping] = torch.tensor(
          rollout.getControlForces(mapping), requires_grad=True)

  def getPoses(self, mapping: str = "identity") -> torch.Tensor:
    return self.posTensors[mapping]

  def getVels(self, mapping: str = "identity") -> torch.Tensor:
    return self.velTensors[mapping]

  def getControlForces(self, mapping: str = "identity") -> torch.Tensor:
    return self.forceTensors[mapping]

  def getMasses(self) -> torch.Tensor:
    return self.massTensor

  def fill_gradients(self, gradWrtRollout: nimble.trajectory.TrajectoryRollout):
    for mapping in gradWrtRollout.getMappings():
      posGrad = self.getPoses(mapping).grad
      if posGrad is not None:
        np.copyto(gradWrtRollout.getPoses(mapping), posGrad.numpy())
      velGrad = self.getVels(mapping).grad
      if velGrad is not None:
        np.copyto(gradWrtRollout.getVels(mapping), velGrad.numpy())
      forceGrad = self.getControlForces(mapping).grad
      if forceGrad is not None:
        np.copyto(gradWrtRollout.getControlForces(mapping), forceGrad.numpy())
    massGrad = self.getMasses().grad
    if massGrad is not None:
      np.copyto(gradWrtRollout.getMasses(), massGrad.numpy())


def NativeLossFn(fn: Callable[[NativeTrajectoryRollout], torch.Tensor]):
  def loss(trajectory):
    nimbleTorchTrajectory = NativeTrajectoryRollout(trajectory)
    return fn(nimbleTorchTrajectory).item()

  def gradAndLoss(trajectory, gradWrtTrajectory):
    nimbleTorchTrajectory = NativeTrajectoryRollout(trajectory)
    loss = fn(nimbleTorchTrajectory)
    loss.backward()
    nimbleTorchTrajectory.fill_gradients(gradWrtTrajectory)
    return loss.item()

  return nimble.trajectory.LossFn(loss, gradAndLoss)
