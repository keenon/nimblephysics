Working with AddBiomechanics Data
====================================

You can process motion capture in `AddBiomechanics <https://addbiomechanics.org/>`_, and get binary files out.
If you process a subject using dynamics, you'll see the following button appear:

.. image:: _static/figures/download_addbiomechanics.png
   :width: 500

If you click to download this file, you'll get a single :code:`*.bin` file. This file is laid out on disk so
that you can efficiently load frames at random from the file, without loading the entire file into memory. This
is helpful for training ML systems on large amounts of training data, which might otherwise overwhelm the amount
of RAM available on your machine.

To load a file at :code:`your/path/your_subject_name.bin`, simply instantiate a :code:`your_subject = nimble.biomechanics.SubjectOnDisk("your/path/your_subject_name.bin")`.
Note that instantiating :code:`SubjectOnDisk` *does not* load all the trials into memory, it merely keeps a lightweight index of the file in memory, which can then load 
arbitrary frames of trials quickly and efficiently. It's safe to load an enormous number of :code:`SubjectOnDisk` files simultaneously, even with very limited RAM.

Once you have a :code:`SubjectOnDisk`, the main point of a :code:`SubjectOnDisk` is to load arrays of :code:`Frame` objects by calling :code:`frames = your_subject.loadFrames(...)`.
Each :code:`Frame` contains all the information to set the state of the skeleton corresponding to this subject, which you can get copies of by calling :code:`skel = your_subject.readSkel(...)`.
With a skeleton set in the correct state, with contact and dynamics information known, you're ready to derive any additional information you need to train your ML system!

.. autoclass:: nimblephysics.biomechanics.SubjectOnDisk
   :members:
   :undoc-members:
   :private-members:


.. autoclass:: nimblephysics.biomechanics.Frame
   :members:
   :undoc-members:
   :private-members: