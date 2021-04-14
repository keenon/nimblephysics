#!/bin/bash

set -e

path="./public"
s3Dir="s3://nimblephysics.org"

echo "Rebuilding docs"
pushd docs
make html
popd
rm -rf public/docs
mkdir public/docs
cp -r docs/_build/html/* public/docs/
echo "Rebuilt docs"

for entry in "$path"/*; do
    name=`echo $entry | sed 's/.*\///'`  # getting the name of the file or directory
    if [[ -d  $entry ]]; then  # if it is a directory
        aws s3 cp  --recursive "$path/$name" "$s3Dir/$name/"
    else  # if it is a file
        aws s3 cp "$path/$name" "$s3Dir/"
    fi
done
