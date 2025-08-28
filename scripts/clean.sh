#!/usr/bin/env bash

# Usage function
usage() {
  echo "Usage: $0 [-b] [-i]"
  echo "  -i    delete all child files/directories under results/ipbes"
  exit 1
}

# Parse options
b_flag=0
i_flag=0
while getopts "bi" opt; do
  case $opt in
    b) b_flag=1 ;;
    i) i_flag=1 ;;
    *) usage ;;
  esac
done

# If neither -b nor -i was provided, show usage and exit
if [[ $b_flag -eq 0 && $i_flag -eq 0 ]]; then
  usage
fi

# Base directory for both BiomoQA and IPBES
BASE="./results"

# If -b was passed, delete all children under biomoqa/*
if [[ $b_flag -eq 1 ]]; then
  rm -r "$BASE/ray_results/"*    \
         "$BASE/final_model/"*    \
         "$BASE/models/"*
fi

# If -i was passed, delete all children under ipbes/*
if [[ $i_flag -eq 1 ]]; then
  rm -r "$BASE/ray_results/"*      \
         "$BASE/final_model/"*      \
         "$BASE/models/"* \
          "$BASE/test preds/"* 
fi

# Always clean /tmp/ray/*
rm -r /tmp/ray/*
