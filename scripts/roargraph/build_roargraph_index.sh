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

# RoarGraph-specific params
M_SQ=100
M_PJBP=35
L_PJPQ=100

# Iterate through the multivector sizes and data types
multivector_sizes=($(yq e '.multivector_sizes[]' "$config_file"))
for mvs in "${multivector_sizes[@]}"; do
  for (( dt_idx=0; dt_idx<data_types_length; dt_idx++ )); do
    datatype_name=$(yq e ".data_types[$dt_idx].name" "$config_file")
    data_file=$(yq e ".data_types[$dt_idx].data_file" "$config_file")

    build_query_file=$(yq e ".data_types[$dt_idx].nonquery_file" "$config_file")
    build_query_groundtruth_file=$(yq e ".data_types[$dt_idx].nonquery_vector_gt_file" "$config_file")

    path="${base_path}${mvs}/"
    index_save_path="${path}/index/${datatype_name}_roargraph.index"

    mkdir -p ${path}/index

    ${build_dir}/tests/test_build_roargraph \
      --data_type float \
      --dist ${dist} \
      --base_data_path ${path}/${data_file} \
      --sampled_query_data_path ${path}/${build_query_file} \
      --projection_index_save_path ${index_save_path} \
      --learn_base_nn_path ${path}${build_query_groundtruth_file} \
      --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

    echo "Completed building RoarGraph index for ${datatype_name}, multivector size ${mvs}"
  done
done
