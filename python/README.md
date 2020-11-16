This is the home for the Python code.

The `_diffdart` folder holds the `pybind11` binding code for our C++. This needs to be updated to change the native methods we expose to Python.

The `diffdart` folder holds the Python code that gets bundled into our package that is installed by `pip install diffdart`. New Python code that's supposed to ship out globally should go in there.

The `diffdart_examples` folder holds the Python examples for DiffDART usage.

The `old_dart_examples` folder holds original examples from DART for how to use DART's Python bindings.
