#!/bin/bash
set -e
export LC_CTYPE=C 
export LANG=C

PP=$1:`pwd`/build/python/_nimblephysics/
echo $PP
PYTHONPATH=$PP python3 -c "import _nimblephysics"
PYTHONPATH=$PP pybind11-stubgen --no-setup-py -o stubs _nimblephysics
touch stubs/_nimblephysics-stubs/py.typed
mv stubs/_nimblephysics-stubs/__init__.pyi stubs/_nimblephysics-stubs/_nimblephysics.pyi
# rm -rf stubs/nimblephysics-stubs
# mv stubs/_nimblephysics-stubs stubs/nimblephysics-stubs
if [[ $OSTYPE == 'darwin'* ]]; then
  find stubs/_nimblephysics-stubs -type f | xargs sed -i '.bak' 's/_nimblephysics/nimblephysics_libs\._nimblephysics/g'
else
  find stubs/_nimblephysics-stubs -type f | xargs sed -i 's/_nimblephysics/nimblephysics_libs\._nimblephysics/g'
fi
find stubs/_nimblephysics-stubs -name "*.bak" -type f -delete