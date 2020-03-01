#!/usr/bin/env bash
set -ex

if [ "${OSTYPE//[0-9.]/}" == "linux-gnu" ]; then
  if [ $(lsb_release -si) = "Ubuntu" ]; then
    '.ci/install_ubuntu.sh'
  elif [ $(lsb_release -si) = "Arch" ]; then
    '.ci/install_archlinux.sh'
  fi
elif [ "${OSTYPE//[0-9.]/}" == "darwin"  ]; then
  '.ci/install_osx.sh'
fi
