Manual Equations of Motion
==========================================

Often, it is useful to be able to peer inside the black box of a simulation and see how the sausage is made.
This section will provide a brief overview of the equations of motion that are used in the simulation,
and how to access the quantities you need in order to recreate the simulation results manually with Numpy.

The equations of motion are derived from the Lagrangian formulation of classical mechanics. For details
on how these equations are implemented at a low level, see the `Nimble Wiki <https://github.com/keenon/nimblephysics/wiki/3.1.-How-Featherstone-is-Organized>`_.
This in turn is based on the `GEAR Notes <https://www.cs.cmu.edu/~junggon/tools/liegroupdynamics.pdf>`_.

At its heart, we are generally using the following dynamics equation:

:math:`M(q) \ddot{q} + C(q, \dot{q}) = \tau`

Or rearranging for forward dynamics:

:math:`\ddot{q} = M(q)^{-1} (\tau - C(q, \dot{q}))`

Where:

- :math:`q` is the generalized coordinates of the system, which can be got with :code:`skeleton.getPositions()` or :code:`world.getPositions()`
- :math:`\dot{q}` is the generalized velocities of the system, which can be got with :code:`skeleton.getVelocities()` or :code:`world.getVelocities()`
- :math:`\ddot{q}` is the generalized accelerations of the system, which can be got with :code:`skeleton.getAccelerations()` or :code:`world.getAccelerations()`
- :math:`M(q)` is the mass matrix of the system, which can be got with :code:`skeleton.getMassMatrix()` or :code:`world.getMassMatrix()`
- :math:`C(q, \dot{q})` is the Coriolis and centrifugal forces of the system, which can be got with :code:`skeleton.getCoriolisAndGravityForces()` or :code:`world.getCoriolisAndGravityForces()`

To put this into code::

  import numpy as np
  import nimblephysics as nimble

  rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
  skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton

  # Set the generalized coordinates, velocities, and accelerations
  q = skeleton.getPositions()
  dq = skeleton.getVelocities()
  ddq = skeleton.getAccelerations()

  # Get the mass matrix
  M = skeleton.getMassMatrix()

  # Get the Coriolis and centrifugal forces
  C = skeleton.getCoriolisAndGravityForces() - skeleton.getExternalForces() + skeleton.getDampingForce() + skeleton.getSpringForce()

  # Get the generalized forces
  tau = skeleton.getForces()

  # Calculate the accelerations
  ddq = np.linalg.solve(M, tau - C)
