#!/bin/bash

curl -X 'POST' \
  'http://0.0.0.0:3310/predict-polymer-image/?return_bigsmiles=true' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'png_file=@assets/bigsmiles_manuscript_1.png;type=image/png'
