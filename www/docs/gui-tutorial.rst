Using the Nimble GUI
=======================

Nimble has a powerful and simple web GUI built in.
At its heart, it is a simple hash table that maps keys to 3D objects.
Whenever you change the hash table, any connected web browsers will update in real time.

The boilerplate to use the GUI is the following::

  import nimblephysics as nimble

  # Create a GUI
  gui = nimble.NimbleGUI()

  # Serve the GUI on port 8080
  gui.serve(8080)

  ############################
  # Do something useful here!
  ############################

  # Do not immediately exit
  gui.blockWhileServing()

To access the hash table directly, you can use the :code:`gui.nativeAPI()` command.
Methods like :code:`gui.nativeAPI().createBox(...)` will create a box in the hash table at the `key` argument.
If the key already exists, it will be overwritten.
Higher level methods like :code:`gui.nativeAPI().renderSkeleton(...)` will create a bunch of objects in the hash table, for each mesh in the skeleton.
If the keys already exist (if you have rendered the skeleton before), they will be overwritten.

You can delete objects by key, with :code:`gui.nativeAPI().deleteObject(key)`, or delete groups of objects with :code:`gui.nativeAPI().deleteObjectsByPrefix(prefix)`, or delete everything with :code:`gui.nativeAPI().clear()`.
You can also update individual attributes of objects by key (without entirely recreating them), with for example :code:`gui.nativeAPI().setObjectPosition(key, position)`.

.. autoclass:: nimblephysics.server.GUIStateMachine
   :members:
   :undoc-members:
   :private-members: