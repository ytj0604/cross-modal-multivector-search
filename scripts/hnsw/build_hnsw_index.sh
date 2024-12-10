#!/bin/bash

build_dir="$(dirname "$(realpath "$0")")/../../build/"

M=35
ef_construction=500

dist=ip

prefix=/mnt/dive/

for i in {1..5}; do
  for datatype in "img" "txt"; do
    path=${prefix}${i}/
    
    ${build_dir}/tests/build_hnsw \
      --base_data_path ${path}/coco_test_${i}_${datatype}_embs.fbin \
      --index_save_path ${path}/${datatype}_hnsw_${M}.index \
      --M ${M} --ef_construction ${ef_construction} \
      --dist ${dist}
  done
done