#!/bin/bash
set -e

rm -rf dist
npm run build
rm ../python/nimblephysics/web_gui/bundle.js
cp dist/bundle.js ../python/nimblephysics/web_gui