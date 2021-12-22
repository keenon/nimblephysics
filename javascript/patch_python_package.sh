#!/bin/bash
set -e

rm -rf dist
npm run build-for-python
rm ../python/nimblephysics/web_gui/bundle.js
cp dist/live.js ../python/nimblephysics/web_gui/bundle.js