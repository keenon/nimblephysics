import nimblephysics as nimble
import numpy as np
import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--state", type = str, default = None, required = True)
parser.add_argument("--port", type = int, default = 8090)
parser.add_argument("--frame_time", type = float, default = None)
args = parser.parse_args()

def createCartpole():
    world = nimble.simulation.World()
    world.setGravity([0, -9.81, 0])
    cartpole = nimble.dynamics.Skeleton("cartpole")
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
    world.setTimeStep(0.01)
    return world
    
states = np.genfromtxt(f'raw_data/States/{args.state}.csv',delimiter=',').T
world = createCartpole()
gui = nimble.NimbleGUI(world)
gui.serve(args.port)

for state in states:
    fullstate = np.concatenate((state,np.zeros_like(state)))
    gui.displayState(torch.from_numpy(fullstate))
    if args.frame_time == None:
        input("Press Any Button for next state")
    else:
        time.sleep(args.frame_time)