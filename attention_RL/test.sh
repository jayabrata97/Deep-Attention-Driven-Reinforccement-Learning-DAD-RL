#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate smarts
export PYTHONPATH="${PYTHONPATH}:../SMARTS"

## list of commands
python test.py "test_config.yaml"