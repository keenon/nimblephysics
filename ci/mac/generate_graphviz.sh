#!/bin/bash
set -e

pushd ../..
mkdir build_viz
pushd build_viz
cmake .. --graphviz=viz.out
