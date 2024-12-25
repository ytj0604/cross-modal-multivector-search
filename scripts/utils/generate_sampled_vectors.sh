#!/bin/bash

build_dir="$(dirname "$(realpath "$0")")/../../build/"
base_path=/mnt/datasets/dive_coco_train/
for i in {1..5}; do
    ${build_dir}/tests/extract_vectorsets \
    ${base_path}img_query_ids.txt \
    $i \
    ${base_path}${i}/img.fbin \
    ${base_path}${i}/img_query.fbin \
    ${base_path}${i}/img_nonquery.fbin

    ${build_dir}/tests/extract_vectorsets \
    ${base_path}txt_query_ids.txt \
    $i \
    ${base_path}${i}/txt.fbin \
    ${base_path}${i}/txt_query.fbin \
    ${base_path}${i}/txt_nonquery.fbin

done