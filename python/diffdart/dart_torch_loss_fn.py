import torch
import diffdart_libs._diffdart as dart
from typing import Dict, Callable
import numpy as np


class DartTorchTrajectoryRollout:
    def __init__(self, rollout: dart.trajectory.TrajectoryRollout):
        self.rollout = rollout

        self.posTensors: Dict[str, torch.Tensor] = {}
        self.velTensors: Dict[str, torch.Tensor] = {}
        self.forceTensors: Dict[str, torch.Tensor] = {}

        for mapping in rollout.getMappings():
            self.posTensors[mapping] = torch.tensor(
                rollout.getPoses(mapping), requires_grad=True)
            self.velTensors[mapping] = torch.tensor(rollout.getVels(mapping), requires_grad=True)
            self.forceTensors[mapping] = torch.tensor(
                rollout.getForces(mapping), requires_grad=True)

    def getPoses(self, mapping: str) -> torch.Tensor:
        return self.posTensors[mapping]

    def getVels(self, mapping: str) -> torch.Tensor:
        return self.velTensors[mapping]

    def getForces(self, mapping: str) -> torch.Tensor:
        return self.forceTensors[mapping]

    def fill_gradients(self, gradWrtRollout: dart.trajectory.TrajectoryRollout):
        for mapping in gradWrtRollout.getMappings():
            posGrad = self.getPoses(mapping).grad
            if posGrad is not None:
                np.copyto(gradWrtRollout.getPoses(mapping), posGrad.numpy())
            velGrad = self.getVels(mapping).grad
            if velGrad is not None:
                np.copyto(gradWrtRollout.getVels(mapping), velGrad.numpy())
            forceGrad = self.getForces(mapping).grad
            if forceGrad is not None:
                np.copyto(gradWrtRollout.getForces(mapping), forceGrad.numpy())


def DartTorchLossFn(fn: Callable[[DartTorchTrajectoryRollout], torch.Tensor]):
    def loss(trajectory):
        dartTorchTrajectory = DartTorchTrajectoryRollout(trajectory)
        return fn(dartTorchTrajectory).item()

    def gradAndLoss(trajectory, gradWrtTrajectory):
        dartTorchTrajectory = DartTorchTrajectoryRollout(trajectory)
        loss = fn(dartTorchTrajectory)
        loss.backward()
        dartTorchTrajectory.fill_gradients(gradWrtTrajectory)
        return loss.item()

    return dart.trajectory.LossFn(loss, gradAndLoss)
