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
k=$(yq e '.k' "$config_file")
data_types_length=$(yq e '.data_types | length' "$config_file")
total_beam_widths=($(yq e '.total_beam_widths[]' "$config_file"))

# HNSW-specific params
M=35
index_prefix="_hnsw"
timestamp=$(date +%Y%m%d%H%M)

# Iterate through the multivector sizes and data types
multivector_sizes=($(yq e '.multivector_sizes[]' "$config_file"))
for mvs in "${multivector_sizes[@]}"; do
  for (( dt_idx=0; dt_idx<data_types_length; dt_idx++ )); do
    datatype_name=$(yq e ".data_types[$dt_idx].name" "$config_file")
    data_file=$(yq e ".data_types[$dt_idx].data_file" "$config_file")
    query_file=$(yq e ".data_types[$dt_idx].query_file" "$config_file")
    set_gt_file=$(yq e ".data_types[$dt_idx].set_gt_file" "$config_file")
    result_prefix=$(yq e ".data_types[$dt_idx].result_prefix" "$config_file")

    iteration_path="${base_path}${mvs}/"
    result_dir="${iteration_path}/result_hnsw_${timestamp}/"
    mkdir -p "${result_dir}"

    index_path="${iteration_path}index/${datatype_name}${index_prefix}.index"
    data_path="${iteration_path}${data_file}"
    query_path="${iteration_path}${query_file}"
    gt_path="${iteration_path}${set_gt_file}"
    result_agg_path="${result_dir}/aggregated_results.txt"

    for bw in "${total_beam_widths[@]}"; do
      ${build_dir}/tests/search_rerank_hnsw \
        --base_data_path ${data_path} \
        --query_path ${query_path} \
        --index_path ${index_path} \
        --set_gt_path ${gt_path} \
        --evaluation_save_path ${result_agg_path} \
        --k ${k} \
        --dist ${dist} \
        --query_multivector_size ${mvs} \
        --evaluation_save_prefix ${result_dir}/${result_prefix}_hnsw_${bw} \
        --total_beam_width ${bw}

      echo "Completed search for ${result_prefix} with beam width ${bw}"
    done
  done
done
