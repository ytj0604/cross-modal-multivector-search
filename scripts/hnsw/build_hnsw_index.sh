#!/bin/bash

build_dir="$(dirname "$(realpath "$0")")/../../build/"

M=35
ef_construction=500

dist=ip

prefix=/mnt/datasets/dive_coco_train/

for i in {1..5}; do
  for datatype in "img" "txt"; do
    path=${prefix}${i}/
    mkdir -p ${path}/index
    ${build_dir}/tests/build_hnsw \
      --base_data_path ${path}/${datatype}.fbin \
      --index_save_path ${path}/index/${datatype}_nsg.index \
      --M ${M} --ef_construction ${ef_construction} \
      --dist ${dist}
  done
done