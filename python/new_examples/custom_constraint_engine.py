"""An example of overriding the ConstraintSolver's default callback function."""
import nimblephysics as nimble
import torch


def runDummyConstraintEngine(resetCommand):
    pass


def main():
    world = nimble.loadWorld("../../data/skel/test/colliding_cube.skel")
    state = torch.tensor(world.getState())
    action = torch.zeros((world.getNumDofs()))
    solver = world.getConstraintSolver()

    def full_frictionless_lcp_engine(resetCommand):
        world.runLcpConstraintEngine(resetCommand)
        world.runFrictionlessLcpConstraintEngine(resetCommand)

    def frictionless_full_lcp_engine(resetCommand):
        world.runFrictionlessLcpConstraintEngine(resetCommand)
        world.runLcpConstraintEngine(resetCommand)

    engines = [
        None,  # Use default (don't replace)
        runDummyConstraintEngine,  # Replace with dummy engine
        world.runLcpConstraintEngine,  # Replace with LCP engine (same as default)
        full_frictionless_lcp_engine,  # LCP + FrictionlessLCP
        frictionless_full_lcp_engine,  # FrictionlessLCP + LCP
    ]
    for engine in engines:
        if engine is not None:
            world.replaceConstraintEngineFn(engine)
            print(engine.__name__)
        else:
            print("None")
        new_state = nimble.timestep(world, state, action)
        print(new_state)


if __name__ == "__main__":
    main()
