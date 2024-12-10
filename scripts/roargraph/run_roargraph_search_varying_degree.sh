#!/bin/bash

# Path to the script
script_path="./run_roargraph_search_expr.sh"

# Parameter values to pass to script.sh
for M_PJBP in {5..35..5}; do
  echo "Running $script_path with M_PJBP=${M_PJBP}..."
  bash "$script_path" "$M_PJBP"
  echo "Completed $script_path with M_PJBP=${M_PJBP}."
done

echo "All experiments completed!"