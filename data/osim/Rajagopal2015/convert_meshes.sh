#!/bin/bash

CONVERT="assimp export "

rm ./Geometry/*.stl
rm ./Geometry/*.obj
rm ./Geometry/*.mtl
for file in ./Geometry/*.ply
do
  assimp export $file ${file}.stl -fstlb
done