#!/bin/bash

train(){
  python -u train.py "$KEY"
}

# actiate conda env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smarts
export PYTHONPATH="${PYTHONPATH}:../SMARTS"

# resume training in case of a crash
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1

KEY=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 5 | head -n 1) # generate key

crash_count=0
until [ $PYTHON_RETURN == 0 ]; do
  train KEY
  PYTHON_RETURN=$?
  if [ $PYTHON_RETURN != 0 ]
  then
    echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
    echo "${KEY}: Training crashed, resuming from last timestep ($(date))" >> crash_report.txt
    crash_count=$((crash_count + 1))
  fi
  sleep 2
  if [ $crash_count -gt 3 ]
  then
  echo "Program crashed too many times!"
    break
  fi
done

echo "Bash script done."