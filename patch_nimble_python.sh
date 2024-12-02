#!/bin/bash

PIP_HOME=$(pip3 show nimblephysics | grep "Location" | sed 's/^\(Location: \)*//')
TARGET=$PIP_HOME/nimblephysics
rm -rf $TARGET/*

SOURCE=python/nimblephysics

cp -r $SOURCE/* $TARGET/
python3 -c "import nimblephysics"