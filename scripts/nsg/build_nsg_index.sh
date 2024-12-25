#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 config.yaml"
    exit 1
fi

config_file="$1"

if [ ! -f "$config_file" ]; then
    echo "Error: Configuration file '$config_file' does not exist."
    exit 1
fi

if ! command -v yq &> /dev/null; then
    echo "Error: yq is not installed."
    exit 1
fi

# Load configuration values from the YAML file
build_dir="$(dirname "$(realpath "$0")")/../../build/"
base_path=$(yq e '.base_path' "$config_file")
dist=$(yq e '.distance_metric' "$config_file")
data_types_length=$(yq e '.data_types | length' "$config_file")

# KNN graph params
knn_K=400
knn_L=400
knn_iter=12
knn_S=15
knn_R=100

# NSG graph params
nsg_L=60
nsg_R=70
nsg_C=500

# Iterate through the multivector sizes and data types
multivector_sizes=($(yq e '.multivector_sizes[]' "$config_file"))
for mvs in "${multivector_sizes[@]}"; do
  for (( dt_idx=0; dt_idx<data_types_length; dt_idx++ )); do
    datatype_name=$(yq e ".data_types[$dt_idx].name" "$config_file")
    data_file=$(yq e ".data_types[$dt_idx].data_file" "$config_file")

    path="${base_path}${mvs}/"
    mkdir -p ${path}/index

    ${build_dir}/tests/build_nsg \
      --base_data_path ${path}/${data_file} \
      --index_save_path ${path}/index/${datatype_name}_nsg.index \
      --knn_K ${knn_K} --knn_L ${knn_L} --knn_iter ${knn_iter} --knn_S ${knn_S} --knn_R ${knn_R} \
      --nsg_L ${nsg_L} --nsg_R ${nsg_R} --nsg_C ${nsg_C} \
      --dist ${dist}
  done
done
