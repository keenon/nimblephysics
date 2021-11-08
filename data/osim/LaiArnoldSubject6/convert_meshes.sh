#!/bin/bash

CONVERT=~/Desktop/libs/ConvertFile/build/ConvertFile

$CONVERT
for file in ./Geometry/*.vtp
do
  $CONVERT "$file" "${file}"
done