#!/bin/bash
mkdir -p perf_results
pushd perf_results
rm tmp.perf
rm tmp.folded
rm perf.data
sudo -E perf record -F 99 -a -g /home/keenon/Desktop/dev/nimblephysics/build/unittests/unit/test_MarkerMultiBeamSearch