Processing Mocap Data
=======================

*This documentation is WIP*:

Nimble is focused on human motion and biomechanics analysis. We include many utilities that will be familiar to biomechanics 
practitioners: loading `C3D files <https://www.c3d.org/>`_, loading multiple `OpenSim <https://simtk.org/projects/opensim>`_ 
formats, and the algorithms that power `AddBiomechanics <https://addbiomechanics.org/>`_.

To load C3D files, check out:

.. autoclass:: nimblephysics.biomechanics.C3DLoader
   :members:
   :undoc-members:
   :private-members:

To run some heuristics to clean up the C3D data, you can use:

.. autoclass:: nimblephysics.biomechanics.MarkerFixer
   :members:
   :undoc-members:
   :private-members:

This will return you an object with a field `markerObservationsAttemptedFixed`

.. autoclass:: nimblephysics.biomechanics.MarkersErrorReport
   :members:
   :undoc-members:
   :private-members:

For OpenSim files, check out:

.. autoclass:: nimblephysics.biomechanics.OpenSimParser
   :members:
   :undoc-members:
   :private-members:

This class is the basis of the kinematic fit (just bone scaling, marker offsets, and IK, *no dynamics*) for AddBiomechanics.

.. autoclass:: nimblephysics.biomechanics.MarkerFitter
   :members:
   :undoc-members:
   :private-members:

