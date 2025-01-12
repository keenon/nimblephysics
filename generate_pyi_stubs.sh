#!/bin/bash
# Remove set -e and handle errors manually
# set -e
export LC_CTYPE=C 
export LANG=C

if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

PP=$1:`pwd`/build/python/_nimblephysics/
echo $PP

# Check if the file exists
if [ ! -f "$1/_nimblephysics.so" ]; then
  echo "$1/_nimblephysics.so does not exist. Exiting."
  exit 1
fi

# Continue if the file exists
# file $1/_nimblephysics.so
# lipo -info $1/_nimblephysics.so
# otool -L $1/_nimblephysics.so
PYTHONPATH=$PP python3 -c "import _nimblephysics" || { echo "Python import failed. Exiting."; exit 1; }
PYTHONPATH=$PP python3 -m pybind11_stubgen --no-setup-py --bare-numpy-ndarray -o stubs _nimblephysics || { echo "pybind11-stubgen failed. Exiting."; exit 1; }
touch stubs/_nimblephysics-stubs/py.typed
mv stubs/_nimblephysics-stubs/__init__.pyi stubs/_nimblephysics-stubs/_nimblephysics.pyi

if [[ $OSTYPE == 'darwin'* ]]; then
  find stubs/_nimblephysics-stubs -type f | xargs sed -i '.bak' 's/_nimblephysics/nimblephysics_libs\._nimblephysics/g'
else
  find stubs/_nimblephysics-stubs -type f | xargs sed -i 's/_nimblephysics/nimblephysics_libs\._nimblephysics/g'
fi

find stubs/_nimblephysics-stubs -name "*.bak" -type f -delete
