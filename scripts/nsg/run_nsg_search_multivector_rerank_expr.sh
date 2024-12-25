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

index_name="nsg"
index_suffix="nsg"
index_prefix="_nsg"
build_dir="$(dirname "$(realpath "$0")")/../../build/"
timestamp=$(date +%Y%m%d%H%M)

get_yaml_value() {
    yq e "$1" "$config_file"
}

experiment_name=$(get_yaml_value '.experiment_name')
base_path=$(get_yaml_value '.base_path')
k=$(get_yaml_value '.k')
distance_metric=$(get_yaml_value '.distance_metric')
aggregated_result_filename=$(get_yaml_value '.aggregated_result_filename')
multivector_sizes=($(yq e '.multivector_sizes[]' "$config_file"))
total_beam_widths=($(yq e '.total_beam_widths[]' "$config_file"))
data_types_length=$(yq e '.data_types | length' "$config_file")

# echo "Starting Experiment: $experiment_name"
# echo "Index: $index_name"
# echo "Base Path: $base_path"
# echo "Ground Truth Path: $ground_truth_path"
# echo "k: $k"
# echo "Distance Metric: $distance_metric"
# echo "Build Directory: $build_dir"
# echo "Aggregated Result Filename: $aggregated_result_filename"
# echo "Multivector Sizes: ${multivector_sizes[*]}"
# echo "Beam Widths: ${total_beam_widths[*]}"
# echo "Timestamp: $timestamp"
# echo "----------------------------------------------"

for mvs in "${multivector_sizes[@]}"; do
    echo "Processing Multivector Size: $mvs"
    for (( dt_idx=0; dt_idx<data_types_length; dt_idx++ )); do
        datatype_name=$(get_yaml_value ".data_types[$dt_idx].name")
        data_file=$(get_yaml_value ".data_types[$dt_idx].data_file")
        query_file=$(get_yaml_value ".data_types[$dt_idx].query_file")
        set_gt_file=$(get_yaml_value ".data_types[$dt_idx].set_gt_file")
        result_prefix=$(get_yaml_value ".data_types[$dt_idx].result_prefix")

        echo "  Data Type: $datatype_name"

        iteration_path="${base_path}${mvs}/"
        result_dir="${iteration_path}/result_${index_suffix}_${timestamp}/"
        mkdir -p "${result_dir}"

        index_path="${iteration_path}index/${datatype_name}${index_prefix}.index"
        data_path="${iteration_path}${data_file}"
        query_path="${iteration_path}${query_file}"
        gt_path="${iteration_path}${set_gt_file}"
        result_agg_path="${result_dir}/${aggregated_result_filename}"

        # echo " Index path: $index_path"

        for bw in "${total_beam_widths[@]}"; do
            evaluation_save_prefix="${result_dir}/${result_prefix}_${index_suffix}_${bw}"
            "${build_dir}/tests/search_rerank_${index_suffix}" \
                --base_data_path "${data_path}" \
                --query_path "${query_path}" \
                --index_path "${index_path}" \
                --set_gt_path "${gt_path}" \
                --evaluation_save_path "${result_agg_path}" \
                --k "${k}" \
                --dist "${distance_metric}" \
                --query_multivector_size "${mvs}" \
                --evaluation_save_prefix "${evaluation_save_prefix}" \
                --total_beam_width "${bw}"

            echo "    Completed: mvs=${mvs}, datatype=${datatype_name}, beam_width=${bw}"
        done
    done
done

echo "Experiment '${experiment_name}' completed successfully."
