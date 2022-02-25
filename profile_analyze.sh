#!/bin/bash

FLAMEGRAPH_HOME=/home/keenon/Desktop/libs/FlameGraph

pushd perf_results
sudo perf script > tmp.perf
sudo chown $(whoami) tmp.perf
${FLAMEGRAPH_HOME}/stackcollapse-perf.pl tmp.perf > tmp.folded
timestamp=$(date +%s)
${FLAMEGRAPH_HOME}/flamegraph.pl tmp.folded > perf_${timestamp}.svg