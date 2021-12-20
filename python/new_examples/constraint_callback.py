"""An example of overriding the ConstraintSolver's default callback function."""
import nimblephysics as nimble
import torch


def dummy_callback():
    pass

def lcp_callback(reset_command):
    world.lcpConstraintEngine(reset_command)

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

def dummy_callback(world: nimble.simulation.World, reset: bool):
    pass

def lcp_callback(world: nimble.simulation.World, reset: bool):
    world.lcpConstraintEngine(reset)

def frictionless_lcp_callback(world: nimble.simulation.World, reset: bool):
    # Backup and remove friction.
    friction_coefs = []
    bodies = []
    for i in range(world.getNumBodyNodes()):
        body = world.getBodyNodeIndex(i)
        bodies.append(body)
        friction_coefs.append(body.getFrictionCoeff())
        body.setFrictionCoeff(0.0)

    # Frictionless LCP
    world.lcpConstraintEngine(reset)


def full_frictionless_lcp_callback(world: nimble.simulation.World, reset: bool):
    lcp_callback(world, reset)
    frictionless_lcp_callback(world, reset)


def main():
    world = nimble.loadWorld("../../data/skel/test/colliding_cube.skel")
    state = torch.tensor(world.getState())
    action = torch.zeros((world.getNumDofs()))
    solver = world.getConstraintSolver()

    # Try the default arg
    world.integrateVelocitiesFromImpulses()
    callbacks = [None, dummy_callback, world.lcpConstraintEngine, lcp_callback, frictionless_lcp_callback]
    callbacks = [
        None, 
        dummy_callback, 
        world.lcpConstraintEngine, 
        lcp_callback,
        frictionless_lcp_callback, 
        full_frictionless_lcp_callback,
    ]
    for callback in callbacks:
        if callback is not None:
            world.replaceConstraintEngine(callback)
            print(callback.__name__)
        else:
            print("None")
        new_state = nimble.timestep(world, state, action)
        print(new_state)


if __name__ == "__main__":
    main()
