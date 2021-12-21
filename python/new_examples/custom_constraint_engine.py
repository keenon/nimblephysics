"""An example of overriding the ConstraintSolver's default callback function."""
import nimblephysics as nimble
import torch


def runDummyConstraintEngine(resetCommand):
    pass


def main():
    world = nimble.loadWorld("../../data/skel/test/colliding_cube.skel")
    skel = world.getSkeleton("box skeleton")
    cube = skel.getBodyNode("box")
    state = torch.tensor(world.getState())
    action = torch.zeros((world.getNumDofs()))
    solver = world.getConstraintSolver()
        
    def friction_frictionless_lcp_engine(resetCommand, use_tauxz):
        solver.runEnforceContactAndJointAndCustomConstraintsFn()
        local_impulse = cube.getConstraintImpulse()
        world_impulse = nimble.math.dAdInvT(cube.getWorldTransform(), local_impulse)
        cube.clearConstraintImpulse()
        taux, tauy, tauz, fx, fy, fz = world_impulse
        y = skel.getPositions()[4]
        tauzx, tauxz = 0, 0
        if use_tauxz:
            tauzx = -y * fz
            tauxz = y * fx
        world_friction_impulse = [tauzx, tauy, tauxz, fx, 0, fz]
        local_friction_impulse = nimble.math.dAdInvT(cube.getWorldTransform().inverse(), world_friction_impulse)
        cube.addConstraintImpulse(local_friction_impulse)
        world.integrateVelocitiesFromImpulses(resetCommand)
        world.runFrictionlessLcpConstraintEngine(resetCommand)

    def friction_frictionless_lcp_engine_with_tauxz(resetCommand):
        friction_frictionless_lcp_engine(resetCommand, use_tauxz=True)

    def friction_frictionless_lcp_engine_without_tauxz(resetCommand):
        friction_frictionless_lcp_engine(resetCommand, use_tauxz=False)

    engines = [
        None,  # Use default (don't replace)
        runDummyConstraintEngine,  # Replace with dummy engine
        world.runLcpConstraintEngine,  # Replace with LCP engine (same as default)
        world.runFrictionlessLcpConstraintEngine,  # Replace with FrictionlessLCP
        friction_frictionless_lcp_engine_with_tauxz,
        friction_frictionless_lcp_engine_without_tauxz
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
