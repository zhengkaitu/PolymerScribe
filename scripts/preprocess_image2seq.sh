#!/bin/bash
source /etc/profile

module load anaconda/2023b
source activate polymerscribe

python preprocess_for_polybert.py
