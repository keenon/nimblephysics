"""An example of overriding the ConstraintSolver's default callback function."""
import nimblephysics as nimble
import torch


def dummy_callback():
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
    callbacks = [
        None,  # Use default LCP, don't replace
        dummy_callback,  # Replace with dummy function
        solver.enforceContactAndJointAndCustomConstraintsWithLcp,  # Replace with the same function as default
        # frictionless_lcp_callback,
    ]
    for callback in callbacks:
        if callback is not None:
            solver.replaceEnforceContactAndJointAndCustomConstraintsFn(callback)
            print(callback.__name__)
        else:
            print("None")
        new_state = nimble.timestep(world, state, action)
        print(new_state)


if __name__ == "__main__":
    main()
