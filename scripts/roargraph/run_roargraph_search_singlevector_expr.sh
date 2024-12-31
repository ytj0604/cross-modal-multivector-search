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
beam_width_budget=($(yq e '.total_beam_widths[]' "$config_file"))
multivector_sizes=($(yq e '.multivector_sizes[]' "$config_file"))


enable_adaptive_expansion=false  # Default setting, can toggle manually
min_beam_width=5

timestamp=$(date +%Y%m%d%H%M)

echo "Running search experiments with adaptive expansion set to ${enable_adaptive_expansion}"...

for mvs in "${multivector_sizes[@]}"; do
  for (( dt_idx=0; dt_idx<data_types_length; dt_idx++ )); do
    datatype_name=$(yq e ".data_types[$dt_idx].name" "$config_file")
    data_file=$(yq e ".data_types[$dt_idx].data_file" "$config_file")
    query_file=$(yq e ".data_types[$dt_idx].query_file" "$config_file")
    gt_file=$(yq e ".data_types[$dt_idx].vector_gt_file" "$config_file")
    result_prefix=$(yq e ".data_types[$dt_idx].result_prefix" "$config_file")

    path="${base_path}${mvs}/"
    result_dir=${path}/result_roargraph_${enable_adaptive_expansion}_${timestamp}/
    mkdir -p ${result_dir}

    index_path="${path}index/${datatype_name}_roargraph.index"

    for budget in "${beam_width_budget[@]}"; do
      ${build_dir}/tests/test_search_singlevector_rerank \
        --data_type float \
        --dist ${dist} \
        --base_data_path ${path}/${data_file} \
        --projection_index_save_path ${index_path} \
        --query_path ${path}/${query_file} \
        --k ${k} -T 1 \
        --max_pq ${budget} \
        --min_pq ${min_beam_width} \
        --max_pq_size_budget ${budget} \
        --evaluation_save_prefix ${result_dir}/${result_prefix}_roargraph_${budget} \
        --evaluation_save_path ${result_dir}/aggregated_results.txt \
        --enable_adaptive_expansion ${enable_adaptive_expansion} \
        --vector_gt_path ${path}/${gt_file}

      echo "Completed search for ${result_prefix} with beam width ${budget}, multivector size ${mvs}"
    done
  done
done

echo "Search experiments completed."
