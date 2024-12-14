#!/bin/bash

build_dir="$(dirname "$(realpath "$0")")/../../build/"

dist=ip

prefix=/mnt/dive/

# KNN graph params
knn_K=400
knn_L=400
knn_iter=12
knn_S=15
knn_R=100

# nsg graph params
nsg_L=60
nsg_R=70
nsg_C=500

for i in {1..5}; do
  for datatype in "img" "txt"; do
    path=${prefix}${i}/
    
    ${build_dir}/tests/build_nsg \
      --base_data_path ${path}/coco_test_${i}_${datatype}_embs.fbin \
      --index_save_path ${path}/${datatype}_nsg.index \
      --knn_K ${knn_K} --knn_L ${knn_L} --knn_iter ${knn_iter} --knn_S ${knn_S} --knn_R ${knn_R} \
      --nsg_L ${nsg_L} --nsg_R ${nsg_R} --nsg_C ${nsg_C} \
      --dist ${dist}
  done
done