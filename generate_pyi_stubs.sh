#!/bin/bash
# Remove set -e and handle errors manually
# set -e
export LC_CTYPE=C
export LANG=C

# Check for the directory argument
if [ -z "$1" ]; then
  # Updated usage message to show the optional python executable
  echo "Usage: $0 <directory> [<python_executable>]"
  exit 1
fi

# Use the second argument as the Python command, or default to "python3"
PYTHON_EXEC=${2:-python3}
echo "Using Python executable: $PYTHON_EXEC"

PP=$1:`pwd`/build/python/_nimblephysics/
echo $PP

# Check if the file exists
if [ ! -f "$1/_nimblephysics.so" ]; then
  echo "$1/_nimblephysics.so does not exist. Exiting."
  exit 1
fi

# Use the PYTHON_EXEC variable instead of the hardcoded "python3"
PYTHONPATH=$PP $PYTHON_EXEC -c "import _nimblephysics" || { echo "Python import failed. Exiting."; exit 1; }
# It's also more robust to run pybind11-stubgen as a module
PYTHONPATH=$PP $PYTHON_EXEC -m pybind11_stubgen --no-setup-py -o stubs _nimblephysics || { echo "pybind11-stubgen failed. Exiting."; exit 1; }

touch stubs/_nimblephysics-stubs/py.typed
mv stubs/_nimblephysics-stubs/__init__.pyi stubs/_nimblephysics-stubs/_nimblephysics.pyi

if [[ $OSTYPE == 'darwin'* ]]; then
  find stubs/_nimblephysics-stubs -type f | xargs sed -i '.bak' 's/_nimblephysics/nimblephysics_libs\._nimblephysics/g'
else
  find stubs/_nimblephysics-stubs -type f | xargs sed -i 's/_nimblephysics/nimblephysics_libs\._nimblephysics/g'
fi

find stubs/_nimblephysics-stubs -name "*.bak" -type f -delete