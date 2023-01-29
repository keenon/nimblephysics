#!/bin/bash
set -e

PYTHONPATH=`pwd`/build/python/_nimblephysics/ python3 -c "import _nimblephysics"
PYTHONPATH=`pwd`/build/python/_nimblephysics/ pybind11-stubgen -o stubs _nimblephysics