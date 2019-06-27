#!/usr/bin/env bash
set -ex

brew update > /dev/null
brew bundle
brew install dartsim/dart/filament

pip3 install -U pytest
