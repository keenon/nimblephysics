"""An example of overriding the ConstraintSolver's default callback function."""
import nimblephysics as nimble
import torch


def runDummyConstraintEngine(reset_command):
    pass


# def frictionless_lcp_callback():
#     # Backup and remove friction.
#     friction_coefs = []
#     bodies = []
#     for i in range(world.getNumBodyNodes()):
#         body = world.getBodyNodeIndex(i)
#         bodies.append(body)
#         friction_coefs.append(body.getFrictionCoeff())
#         body.setFrictionCoeff(0.0)

#     # Frictionless LCP
#     lcp_callback()

#     # Restore friction.
#     for friction_coef, body in zip(friction_coefs, bodies):
#         body.setFrictionCoeff(friction_coef)


def main():
    world = nimble.loadWorld("../../data/skel/test/colliding_cube.skel")
    state = torch.tensor(world.getState())
    action = torch.zeros((world.getNumDofs()))
    solver = world.getConstraintSolver()

    engines = [
        None,  # Use default (don't replace)
        runDummyConstraintEngine,  # Replace with dummy engine
        world.runLcpConstraintEngine,  # Replace with LCP engine (same as default)
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
