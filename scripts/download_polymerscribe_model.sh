#!/bin/bash

mkdir -p ./output/r1a

if [ ! -f ./output/r1a/best.pth ]; then
    echo "./output/r1a/best.pth not found. Downloading.."
    wget -q --show-progress -O output/r1a/best.pth \
      "https://www.dropbox.com/scl/fi/s5gnw3zknwh3c1x4cz195/best.pth?rlkey=axhsn51ikzimfz7jdhai5gv3l&st=igdvq9cm&dl=1"
    echo "Model checkpoint best.pth Downloaded."
fi
