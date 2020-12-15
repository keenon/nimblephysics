import diffdart as dart
import torch
from typing import Tuple, Callable, List
import numpy as np
import math


class DartMapPosition(torch.autograd.Function):
    """
    This allows you to use an arbitrary mapping as a standalone PyTorch function
    """

    @staticmethod
    def forward(ctx, world, mapping, pos):
        """
        We can't put type annotations on this declaration, because the supertype
        doesn't have any type annotations and otherwise mypy will complain, so here
        are the types:

        world: dart.simulation.World
        mapping: dart.neural.Mapping
        pos: torch.Tensor
        -> torch.Tensor
        """
        m: dart.neural.Mapping = mapping
        world.setPositions(pos.detach().numpy())
        mappedPos = mapping.getPositions(world)
        ctx.into = m.getRealPosToMappedPosJac(world)

        return torch.tensor(mappedPos)

    @staticmethod
    def backward(ctx, grad_pos):
        intoJac = torch.tensor(ctx.into, dtype=torch.float64)
        lossWrtPosition = torch.matmul(torch.transpose(intoJac, 0, 1), grad_pos)
        return (
            None,
            None,
            lossWrtPosition,
        )


def dart_map_pos(
        world: dart.simulation.World, map: dart.neural.Mapping, pos: torch.Tensor) -> torch.Tensor:
    """
    This maps the positions into the mapping passed in, storing necessary info in order to do a backwards pass.
    """
    return DartMapPosition.apply(world, map, pos)  # type: ignore
