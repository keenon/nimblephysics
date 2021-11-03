#!/bin/bash

set -e

path="./public"

echo "Rebuilding docs"
pushd docs
make html
popd
rm -rf public/docs
mkdir public/docs
cp -r docs/_build/html/* public/docs/
echo "Rebuilt docs"

if [ ! -d nimblephysics.github.io ]; then
  echo "Cloning Github Pages repo"
  git clone git@github.com:nimblephysics/nimblephysics.github.io.git
  echo "Cloned Github Pages repo"
fi

echo "Copying built website contents into repo"
cp -r public/* nimblephysics.github.io
echo "Done copying built website contents into repo"

CURRENT_HASH=$(git rev-parse HEAD)

echo "Committing changes to Github Pages repo"
cd nimblephysics.github.io
git add .
git commit -m "Published website changes from main repo ${CURRENT_HASH}"
git push
cd ..
echo "Committed changes to Github Pages repo"

echo "All done!"