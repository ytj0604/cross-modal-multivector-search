#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
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

# Helper function to read YAML values
get_yaml_value() {
    yq e "$1" "$config_file"
}

# Extract general configuration values
base_path=$(get_yaml_value '.base_path')
k=$(get_yaml_value '.k')
distance_metric=$(get_yaml_value '.distance_metric')
data_types_length=$(yq e '.data_types | length' "$config_file")
multivector_sizes=($(yq e '.multivector_sizes[]' "$config_file"))
aggregated_result_filename=$(get_yaml_value '.aggregated_result_filename')

for mvs in "${multivector_sizes[@]}"; do
    iteration_path="${base_path}${mvs}/"
    result_path="${iteration_path}/pair_recall_results/"
    mkdir -p "$result_path"

    echo "Running multivector size: $mvs"

    for (( dt_idx=0; dt_idx<data_types_length; dt_idx++ )); do
        datatype_name=$(get_yaml_value ".data_types[$dt_idx].name")
        data_file=$(get_yaml_value ".data_types[$dt_idx].data_file")
        query_file=$(get_yaml_value ".data_types[$dt_idx].query_file")
        set_gt_file=$(get_yaml_value ".data_types[$dt_idx].set_gt_file")
        result_prefix=$(get_yaml_value ".data_types[$dt_idx].result_prefix")

        echo "Processing $datatype_name"
        /mnt/CrossModalMultivectorSearch/build/tests/test_dive_emb_pair_recall \
            --query_path "${iteration_path}/${query_file}" \
            --base_data_path "${iteration_path}/${data_file}" \
            --query_multivector_size "$mvs" \
            --evaluation_save_path "${result_path}/${aggregated_result_filename}" \
            --evaluation_save_prefix "${result_path}/${result_prefix}" \
            --set_gt_path "${iteration_path}/${set_gt_file}" \
            --k "$k" \
            --dist "$distance_metric"
        echo "$datatype_name done"
    done
done

echo "All experiments completed successfully."
