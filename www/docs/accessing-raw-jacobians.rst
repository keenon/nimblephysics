Accessing Derivatives and Jacobians
==========================================

In order to do gradient-based optimization, you need access to derivatives of the quantities you are optimizing.

For some "old school" control algorithms like `Differential Dynamic Programming (DDP) <https://inst.eecs.berkeley.edu/~cs294-40/fa08/scribes/lecture7.pdf>`_, you need access to the Jacobians of dynamics.

If we say the state at time :math:`t` is :math:`s_t`, and the action is :math:`a_t`, then our timestep can be thought of as:

:math:`s_{t+1} = f(s_t, a_t)`

We'd like to be able to access the Jacobians:

:math:`\frac{\partial s_{t+1}}{\partial s_t}` - which we'll call the "state Jacobian" (:code:`stateJac` below)

:math:`\frac{\partial s_{t+1}}{\partial a_t}` - which we'll call the "action Jacobian" (:code:`actionJac` below)

Getting these quantities out of Nimble is easy! Simply call::

  stateJac = world.getStateJacobian()
  actionJac = world.getActionJacobian()

That'll return numpy arrays (`not` PyTorch tensors, computing these Jacobians is not itself differentiable) of the requested Jacobians.

These Jacobians will change as you change either the state or the action, so remember to recompute!

**Performance note**: internally, a call to either :code:`world.getStateJacobian()` or :code:`world.getActionJacobian()` must run a timestep and cache the result.
This means that if you call both without changing either your state or your action, the second Jacobian requested is (almost) free.
Be aware though that the first time you call either :code:`world.getStateJacobian()` or :code:`world.getActionJacobian()` in a given state and action, that'll cost about the same as a call to :code:`nimble.timestep(...)`.