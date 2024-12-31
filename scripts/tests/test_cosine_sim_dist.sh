#!/bin/bash
build_dir="$(dirname "$(realpath "$0")")/../../build/"
i=2
# for qt in {'img','txt'}; do
#   for dt in {'img','txt'}; do
    # query_file=/mnt/datasets/dive_coco_train/${i}/${qt}_query.fbin
    # data_file=/mnt/datasets/dive_coco_train/${i}/${dt}.fbin
    # output_file=/mnt/tjyoon/cosine_dist/result_dive_coco_train_${i}/${qt}2${dt}_result.txt

    query_file=/mnt/CrossModalMultivectorSearch/data/laion-10M/query.10k.fbin
    data_file=/mnt/CrossModalMultivectorSearch/data/laion-10M/base.10M.fbin
    output_file=/mnt/tjyoon/cosine_dist/result_laion/result.txt

    gdb --args ${build_dir}/tests/test_rand_vector \
      --query ${query_file} \
      --data ${data_file} \
      --output ${output_file}
  # done
# done