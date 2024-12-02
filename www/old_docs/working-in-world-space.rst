Specify Loss in World Space
==========================================

Once you start using Nimble to work on non-trivial problems, pretty soon you'll run into an issue:

Nimble works in `generalized coordinates <https://en.wikipedia.org/wiki/Generalized_coordinates>`_, which is great for efficiency and simulation accuracy, but is terrible if you want to know where things are in world coordinates.
For example, you might want to know the :code:`(X,Y,Z)` coordinates of the gripper-hand on your robot arm, so you can use it to specify loss. It's not trivial to go from a set of joint angles to the :code:`(X,Y,Z)` position of your gripper-hand, so you're stuck.

We've built a tool to help you with this! It's called :code:`nimble.neural.IKMapping`.

If you're impatient, you can download an example :download:`using IKMapping on the KR5 robot <./_static/robots/IK_example.zip>` and figure out how to use :code:`IKMapping` from context. If you prefer reading a full explanation, read on :)

As a recap from :ref:`Worlds`, remember that we specify ordinary position and velocity in joint space. For example:

.. image:: _static/figures/generalized_coords.png
   :width: 600

The goal of our :code:`nimble.neural.IKMapping` object is to define a map from joint space to world space.

To create a simple :code:`IKMapping`, if you've got the KR5 robot loaded in the variable :code:`arm`, you could write::

  ikMap = nimble.neural.IKMapping(world)
  ikMap.addLinearBodyNode(arm.getBodyNode("palm"))

That results in the mapping below where we go from the joint space configuration of the KR5 to a world-space that has just the KR5's palm position.

.. image:: _static/figures/ik_mapping.svg
   :width: 800

The :code:`IKMapping` can map multiple nodes at once, so you could instead say::

  ikMap = nimble.neural.IKMapping(world)
  ikMap.addLinearBodyNode(arm.getBodyNode("palm"))
  ikMap.addLinearBodyNode(arm.getBodyNode("elbow"))

That results in the mapping below where we go from the joint space configuration of the KR5 to a world-space that has just the KR5's palm position concatenated with the KR5's elbow position. (Note the indices in the figure start from the bottom!)

.. image:: _static/figures/ik_mapping_2.svg
   :width: 800

Now let's use our mapping with PyTorch.
Suppose we have an :code:`IKMapping` object called :code:`ikMap`. We can call::

  world_pos = nimble.map_to_pos(world, ikMap, state)

That'll give us the world positions corresponding to a joint-space vector :code:`state`.
If instead we want the world velocities, we can call::

  world_vel = nimble.map_to_vel(world, ikMap, state)

To recap, to map joint-space into world-space, follow three steps:

1. Create and configure an :code:`mapping = nimble.neural.IKMapping(world)` object, to specify how you'd like to map objects to world space.
2. Configure the :code:`mapping` by adding one or more nodes to it. To add the (X,Y,Z) position of a node named :code:`head`, you could write :code:`mapping.addLinearBodyNode(head)`.
3. Call :code:`nimble.map_to_pos(world, mapping, state)` to map (in a PyTorch friendly way) a state vector :code:`state` to a vector of world space positions.
   Alternatively, you can call :code:`nimble.map_to_vel(world, mapping, state)` to get the velocities in world space for your mappings.

You can download a working example of :download:`using IKMapping on the KR5 robot <./_static/robots/IK_example.zip>` and play around with it.