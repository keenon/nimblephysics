#!/bin/bash

# pushd ../javascript/
# npm run build
# popd

# cp ../javascript/dist/bundle.js ./themes/prettydoc/static/assets/js/

hugo

path="./public"
s3Dir="s3://www.diffdart.org"

for entry in "$path"/*; do
    name=`echo $entry | sed 's/.*\///'`  # getting the name of the file or directory
    if [[ -d  $entry ]]; then  # if it is a directory
        aws s3 cp  --recursive "$path/$name" "$s3Dir/$name/"
    else  # if it is a file
        aws s3 cp "$path/$name" "$s3Dir/"
    fi
done
