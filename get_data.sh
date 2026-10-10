#!/bin/sh
# get_data.sh - download the FlyWire connectome files Bob needs (~135 MB) into ./data, then build Bob's cache.
# The data is NOT part of this repo; it comes from its original sources under their own terms (see THIRD_PARTY_NOTICES.md).
set -e
mkdir -p data
B=https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main
curl -fL -o data/Completeness_783.csv $B/Completeness_783.csv
curl -fL -o data/Connectivity_783.parquet $B/Connectivity_783.parquet
curl -fL -o data/neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
${PYTHON:-python3} build_connectome.py
