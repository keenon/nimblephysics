#!/bin/bash

# pip3 install meshio

CONVERT="assimp export "

rm ./Geometry/*.stl
rm ./Geometry/*.obj
rm ./Geometry/*.mtl
for file in ./Geometry/*.vtk
do
  meshio convert $file ${file}.stl
done