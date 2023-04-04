import time

import nimblephysics as nimble
import numpy as np
import torch


def create_world():
    # Create and configure world.
    world: nimble.simulation.World = nimble.simulation.World()
    world.setGravity([0, -9.81, 0])
    world.setTimeStep(0.01)
    return world


def create_projectile():
    projectile = nimble.dynamics.Skeleton()

    projectileJoint, projectileNode = projectile.createTranslationalJoint2DAndBodyNodePair()
    projectileJoint.setXYPlane()
    projectileShape = projectileNode.createShapeNode(nimble.dynamics.BoxShape([.1, .1, .1]))
    projectileVisual = projectileShape.createVisualAspect()
    projectileShape.createCollisionAspect()
    projectileVisual.setColor([0.7, 0.7, 0.7])
    projectileVisual.setCastShadows(False)
    projectileJoint.setControlForceUpperLimit(0, 0)
    projectileJoint.setControlForceLowerLimit(0, 0)
    projectileJoint.setControlForceUpperLimit(1, 0)
    projectileJoint.setControlForceLowerLimit(1, 0)
    projectileJoint.setVelocityUpperLimit(0, 1000.0)
    projectileJoint.setVelocityLowerLimit(0, -1000.0)
    projectileJoint.setVelocityUpperLimit(1, 1000.0)
    projectileJoint.setVelocityLowerLimit(1, -1000.0)

    projectile.setPositions(np.array([0, 0.1]))
    return projectile



def create_catapult():
    catapult = nimble.dynamics.Skeleton()

    rootJoint, root = catapult.createWeldJointAndBodyNodePair()
    rootOffset = nimble.math.Isometry3()
    rootOffset.set_translation([0.5, -0.45, 0])
    rootJoint.setTransformFromParentBodyNode(rootOffset)

    def createTailSegment(parent, color):
        poleJoint, pole = catapult.createRevoluteJointAndBodyNodePair(parent)
        poleJoint.setAxis([0, 0, 1])
        poleShape = pole.createShapeNode(nimble.dynamics.BoxShape([.05, 0.25, .05]))
        poleVisual = poleShape.createVisualAspect()
        poleVisual.setColor(color)
        poleJoint.setControlForceUpperLimit(0, 1000.0)
        poleJoint.setControlForceLowerLimit(0, -1000.0)
        poleJoint.setVelocityUpperLimit(0, 10000.0)
        poleJoint.setVelocityLowerLimit(0, -10000.0)

        poleOffset = nimble.math.Isometry3()
        poleOffset.set_translation([0, -0.125, 0])
        poleJoint.setTransformFromChildBodyNode(poleOffset)

        poleJoint.setPosition(0, 90 * 3.1415 / 180)
        poleJoint.setPositionUpperLimit(0, 180 * 3.1415 / 180)
        poleJoint.setPositionLowerLimit(0, 0 * 3.1415 / 180)

        poleShape.createCollisionAspect()

        if parent != root:
            childOffset = nimble.math.Isometry3()
            childOffset.set_translation([0, 0.125, 0])
            poleJoint.setTransformFromParentBodyNode(childOffset)
        return pole

    tail1 = createTailSegment(root, [182.0/255, 223.0/255, 144.0/255])
    tail2 = createTailSegment(tail1, [223.0/255, 228.0/255, 163.0/255])
    tail3 = createTailSegment(tail2, [221.0/255, 193.0/255, 121.0/255])

    catapult.setPositions(np.array([45, 0, 45]) * 3.1415 / 180)

    return catapult


def create_floor():
    floor = nimble.dynamics.Skeleton()
    floor.setName('floor')  # important for rendering shadows

    floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
    floorOffset = nimble.math.Isometry3()
    floorOffset.set_translation([1.2, -0.7, 0])
    floorJoint.setTransformFromParentBodyNode(floorOffset)
    floorShape: nimble.dynamics.ShapeNode = floorBody.createShapeNode(nimble.dynamics.BoxShape(
        [3.5, 0.25, .5]))
    floorVisual: nimble.dynamics.VisualAspect = floorShape.createVisualAspect()
    floorVisual.setColor([0.5, 0.5, 0.5])
    floorVisual.setCastShadows(False)
    floorShape.createCollisionAspect()

    return floor


def main():
    # Load the world
    world: nimble.simulation.World = create_world()

    # Create cube
    projectile = create_projectile()
    world.addSkeleton(projectile)

    # Create catapult
    catapult = create_catapult()
    world.addSkeleton(catapult)

    floor = create_floor()
    world.addSkeleton(floor)

    # Target

    target_x = 2.2
    target_y = 2.2

    target = nimble.dynamics.Skeleton()
    target.setName('target')  # important for rendering shadows

    targetJoint, targetBody = floor.createWeldJointAndBodyNodePair()
    targetOffset = nimble.math.Isometry3()
    targetOffset.set_translation([target_x, target_y, 0])
    targetJoint.setTransformFromParentBodyNode(targetOffset)
    targetShape = targetBody.createShapeNode(nimble.dynamics.BoxShape([0.1, 0.1, 0.1]))
    targetVisual = targetShape.createVisualAspect()
    targetVisual.setColor([0.8, 0.5, 0.5])

    world.addSkeleton(target)

    gui: nimble.NimbleGUI = nimble.NimbleGUI(world)
    gui.serve(8080)

    # Set up the view

    # Define the loss function which is norm squared on final pose
    def loss(rollout: nimble.trajectory.TrajectoryRollout):
        poses = rollout.getPoses('identity')  # [n_bodies, steps]
        last_pos = poses[:, -1]
        last_x = last_pos[0]
        last_y = last_pos[1]
        final_loss = (target_x - last_x)**2 + (target_y - last_y)**2
        return final_loss
    dartLoss: nimble.trajectory.LossFn = nimble.trajectory.LossFn(loss)

    timesteps = 50
    shot_length = 20
    iteration_limit = 500
    trajectory = nimble.trajectory.MultiShot(world, dartLoss, timesteps, shot_length, False)
    trajectory.setParallelOperationsEnabled(True)

    # Initialize the optimizer
    optimizer = nimble.trajectory.IPOptOptimizer()
    optimizer.setLBFGSHistoryLength(5)
    optimizer.setTolerance(1e-4)
    optimizer.setCheckDerivatives(False)
    optimizer.setIterationLimit(iteration_limit)
    start = time.time()
    result: nimble.trajectory.Solution = optimizer.optimize(trajectory)
    print(f'Finished. Took: {time.time() - start}')

    # Get the rollout from the last optimization step.
    n_registered_opt_steps = result.getNumSteps()
    rollout: nimble.trajectory.TrajectoryRollout = result.getStep(n_registered_opt_steps - 1).rollout
    poses = rollout.getPoses()  # Get poses
    vels = rollout.getVels()  # Get velocities
    states = torch.cat((torch.tensor(poses), torch.tensor(vels)), 0).transpose(1, 0)

    dofs = world.getNumDofs()
    poses = np.zeros((dofs, len(states)))
    for i in range(len(states)):
        # Take the top-half of each state vector, since this is the position component
        poses[:, i] = states[i].detach().numpy()[:dofs]

    gui.loopStates(states) # tells the GUI to animate our list of states
    gui.blockWhileServing() # block here so we don't exit the program
    



if __name__ == '__main__':
    main()
