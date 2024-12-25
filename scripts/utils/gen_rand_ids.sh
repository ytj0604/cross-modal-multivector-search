#!/bin/bash

num_index=566435
num_sample=10000
rand_id_path=/mnt/datasets/dive_coco_train/txt_query_ids.txt
build_dir="$(dirname "$(realpath "$0")")/../../build/"

$build_dir/tests/gen_rand_ids $num_index $num_sample $rand_id_path