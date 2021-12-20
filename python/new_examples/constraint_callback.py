"""An example of overriding the ConstraintSolver's default callback function."""
import nimblephysics as nimble
import torch


def dummy_callback(world: nimble.simulation.World):
    pass

def lcp_callback(world: nimble.simulation.World):
    world.getConstraintSolver().lcpSolveCallback(world)

def main():
    world = nimble.loadWorld("../../data/skel/test/colliding_cube.skel")
    state = torch.tensor(world.getState())
    action = torch.zeros((world.getNumDofs()))
    solver = world.getConstraintSolver()
    callbacks = [None, dummy_callback, lcp_callback]
    for callback in callbacks:
        if callback is not None:
            solver.replaceSolveCallback(callback)
        new_state = nimble.timestep(world, state, action)
        print(new_state)


if __name__ == '__main__':
    main()
