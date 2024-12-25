#!/bin/bash
# for i in {1..5}; do
#   prefix=/mnt/dive/${i}
#   image=coco_test_${i}_img_embs.fbin
#   text=coco_test_${i}_txt_embs.fbin
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${text} --query_file ${prefix}/${text}  --gt_file ${prefix}/t2t.gt.bin --K $((i * 25000))
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${image} --query_file ${prefix}/${image} --gt_file ${prefix}/i2i.gt.bin --K $((i * 5000))
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${image} --query_file ${prefix}/${text} --gt_file ${prefix}/t2i.gt.bin --K $((i * 5000))
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${text} --query_file ${prefix}/${image} --gt_file ${prefix}/i2t.gt.bin --K $((i * 25000))
# done

# /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file /mnt/rand_vec/4/vec_2.fbin --query_file /mnt/rand_vec/4/vec_1.fbin  --gt_file /mnt/rand_vec/4/1to2.gt.bin --K 40000
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

base_path=$(get_yaml_value '.base_path')
k=$(get_yaml_value '.k')
distance_metric=$(get_yaml_value '.distance_metric')
data_types_length=$(yq e '.data_types | length' "$config_file")
multivector_sizes=($(yq e '.multivector_sizes[]' "$config_file"))

for mvs in "${multivector_sizes[@]}"; do
    iteration_path="${base_path}${mvs}/"
    result_path="${iteration_path}/pair_recall_results/"
    mkdir -p "$result_path"

    echo "Running multivector size: $mvs"

    for (( dt_idx=0; dt_idx<data_types_length; dt_idx++ )); do
        datatype_name=$(get_yaml_value ".data_types[$dt_idx].name")
        data_file=$(get_yaml_value ".data_types[$dt_idx].data_file")
        query_file=$(get_yaml_value ".data_types[$dt_idx].nonquery_file")
        vector_gt_file=$(get_yaml_value ".data_types[$dt_idx].nonquery_vector_gt_file")

        echo "Processing $datatype_name"
        /mnt/CrossModalMultivectorSearch/build/tests/gen_vector_groundtruth \
            --query_path "${iteration_path}/${query_file}" \
            --base_data_path "${iteration_path}/${data_file}" \
            --vector_gt_path "${iteration_path}/${vector_gt_file}" \
            --k "$k" \
            --dist "$distance_metric"
        echo "$datatype_name done"
    done
done

echo "All experiments completed successfully."
