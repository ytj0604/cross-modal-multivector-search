#!/bin/bash

num_index=566435
num_sample=113287
rand_id_path=/mnt/datasets/dive_coco_train/2_test/txt_smpl_ids.txt
build_dir="$(dirname "$(realpath "$0")")/../../build/"

$build_dir/tests/gen_rand_ids $num_index $num_sample $rand_id_path