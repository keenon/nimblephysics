import ipyopt
from numpy import ones, float_, array, zeros

steps = 3
eps = 1e-2

# Initial setup

xs = [4.0, 3.0, 2.0]
vs = [-1.0, -1.0, -1.0]
x = [xs[0], vs[0], xs[1], vs[1], xs[2], vs[2]]

# Have a very simple setup, we've got position and velocity for each timestep

n = steps * 2
lower_bounds = array([-2e19] * n)
upper_bounds = array([2e19] * n)

# Have one constraint per timestep

num_constraints = steps
constraint_upper_bounds = array([eps] * num_constraints)
constraint_lower_bounds = array([-eps] * num_constraints)

# Functions


def f(x):
    assert len(x) == n
    print('f(x): '+str(x))
    # normalize velocities
    return (x[n-2] * x[n-2])/2 + (x[1]*x[1] + x[3]*x[3] + x[5]*x[5])/20


def grad_f(x, out):
    assert len(x) == n
    print('grad_f(x): '+str(x))
    out[0] = 0
    out[1] = x[1] / 10
    out[2] = 0
    out[3] = x[3] / 10
    out[4] = x[4]
    out[5] = x[5] / 10
    return out


def g(x, out):
    assert len(x) == n
    print('g(x): '+str(x))
    for i in range(num_constraints):
        last_pos = 4 if i == 0 else x[(i-1)*2]
        this_pos = x[i*2]
        this_vel = x[(i*2)+1]
        out[i] = last_pos + this_vel - this_pos
    return out


def jac_g(x, out):
    assert len(x) == n
    print('jac_g(x): '+str(x))

    # xs = [x[0], x[2], x[4]]
    # vs = [x[1], x[3], x[5]]

    out[()] = [
        # Timestep 0
        -1,  # xs[0]
        1,  # vs[0]
        0,
        0,
        0,
        0,
        # Timestep 1
        1,   # xs[0]
        0,
        -1,  # xs[1]
        1,  # vs[1]
        0,
        0,
        # Timestep 2
        0,
        0,
        1,  # xs[1]
        0,
        -1,  # xs[2]
        1,  # vs[2]
    ]
    return out


jac_g_sparsity_indices = (array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]),  # row
                          array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]))  # col

ipyopt.set_loglevel(ipyopt.LOGGING_DEBUG)
nlp = ipyopt.Problem(
    n, lower_bounds, upper_bounds, num_constraints, constraint_lower_bounds,
    constraint_upper_bounds, jac_g_sparsity_indices, 0,
    f, grad_f, g, jac_g)

x0 = array(x)
# x0[1] = 0
# x0[3] = 0
# x0[5] = 0

print("Going to call solve")
print("x0 = {}".format(x0))
_x, obj, status = nlp.solve(x0)

print(_x)
