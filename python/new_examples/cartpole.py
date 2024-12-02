import numpy as np
import torch
import torch.nn.functional as F
import random
import math
import time
import nimblephysics as nimble
from typing import List


def main():
    world = nimble.simulation.World()
    world.setGravity([0, -9.81, 0])

    # Set up the 2D cartpole

    cartpole = nimble.dynamics.Skeleton()
    cartRail, cart = cartpole.createPrismaticJointAndBodyNodePair()
    cartRail.setAxis([1, 0, 0])
    cartShape = cart.createShapeNode(nimble.dynamics.BoxShape([.5, .1, .1]))
    cartVisual = cartShape.createVisualAspect()
    cartVisual.setColor([0.5, 0.5, 0.5])
    cartRail.setPositionUpperLimit(0, 10)
    cartRail.setPositionLowerLimit(0, -10)
    cartRail.setControlForceUpperLimit(0, 10)
    cartRail.setControlForceLowerLimit(0, -10)

    poleJoint, pole = cartpole.createRevoluteJointAndBodyNodePair(cart)
    poleJoint.setAxis([0, 0, 1])
    poleShape = pole.createShapeNode(nimble.dynamics.BoxShape([.1, 1.0, .1]))
    poleVisual = poleShape.createVisualAspect()
    poleVisual.setColor([0.7, 0.7, 0.7])
    poleJoint.setControlForceUpperLimit(0, 0)
    poleJoint.setControlForceLowerLimit(0, 0)

    poleOffset = nimble.math.Isometry3()
    poleOffset.set_translation([0, -0.5, 0])
    poleJoint.setTransformFromChildBodyNode(poleOffset)

    world.addSkeleton(cartpole)

    # Make simulations repeatable
    random.seed(1234)

    # Make simulations and backprop run faster by using a bigger timestep
    world.setTimeStep(world.getTimeStep()*10)

    num_timesteps = 100

    action_size = world.getActionSize()
    print('action size: '+str(action_size))
    learning_rate = 0.01

    first_state: torch.Tensor = torch.randn(
        (world.getStateSize()), requires_grad=True)
    actions: List[torch.Tensor] = [torch.zeros(
        (action_size), requires_grad=True) for _ in range(num_timesteps)]

    gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
    gui.serve(8080)

    while True:
        state: torch.Tensor = first_state
        states = [state]
        for i in range(num_timesteps):
            state = nimble.timestep(world, state, actions[i])
            states.append(state)
        gui.loopStates(states)

        loss = state.norm()
        print('loss: '+str(loss))

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        loss.backward()
        with torch.no_grad():
            first_state -= learning_rate * first_state.grad
            first_state.grad = None

            for action in actions:
                action -= learning_rate * action.grad
                action.grad = None
        # time.sleep(0.1)

    gui.blockWhileServing()


if __name__ == "__main__":
    main()
