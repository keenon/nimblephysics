#!/bin/bash
set -e

rm -rf dist
npm run build
rm ../python/diffdart/web_gui/bundle.js
cp dist/bundle.js ../python/diffdart/web_gui