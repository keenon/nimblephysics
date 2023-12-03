.. Nimble Physics documentation master file, created by
   sphinx-quickstart on Tue Apr 13 16:51:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

What is Nimble?
==========================================

Nimble is a *toolkit for doing AI on human biomechanics* (physically simulated realistic human bodies), written in C++ for speed, but with nice Python bindings. It focuses on studying real physical human bodies.

Nimble started life as a general purpose differentiable physics engine, as a fork of the (not differentiable) `DART physics engine <https://dartsim.github.io/>`_.
We wrote a paper about it, which you can find `here <https://arxiv.org/pdf/2103.16021.pdf>`_.
Over the years, to support projects like `AddBiomechanics <https://addbiomechanics.org/>`_, we've added a lot of biomechanics-focused functionality to Nimble, including:

- New types of biological joints (:code:`CustomJoint` for knees, :code:`ConstantCurvatureJoint` for spines, :code:`EllipsoidJoint` for shoulders, etc), which are all differentiable
- Support for loading, modifying, and saving OpenSim skeleton models
- Support for handling raw motion capture data
- Treating bone scales and optical marker offsets as first class differentiable quantities, even through physics, which is important when system-identifying a human body
- Various optimization algorithms useful for biomechanics
- Optimized computations to get Jacobians and gradients through most quantities of interest in the human body
- ... a bunch of other features too numerous to list here

Does Nimble do general purpose robotics simulation?
######################################################

This documentation is focused on the tools you need when doing AI for human biomechanics.
While Nimble is also a general purpose differentiable physics engine capable of simulating complex scenes 
with robots and collisions, we don't cover that much here, because the past few years of research with differentiable physics engines have shown that 
the way Nimble gets gradients through rigid collisions is `probably not the best way for complex trajectory optimization <https://arxiv.org/abs/2109.05143>`_ (TLDR: RL is onto something, and it has advantages even if the system is technically differentiable).
If you are doing general purpose robotics research, we recommend looking at some of the following engines:

- `MuJoCo <http://www.mujoco.org/>`_
- `Brax <https://github.com/google/brax>`_
- `Drake <https://drake.mit.edu/>`_
- `DART <https://dartsim.github.io/>`_
- `TDS <https://github.com/erwincoumans/tiny-differentiable-simulator>`_
- `PyBullet <https://pybullet.org/wordpress/>`_

If you are here because you are interested in studying physical human bodies with AI, then read on!