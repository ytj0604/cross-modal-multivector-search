#!/bin/bash
base_path=/mnt/datasets/dive_coco_train/
for i in {1..5}; do
  path=${base_path}/${i}/
  res_path=$path/pair_recall_results/
  mkdir -p $res_path
  gdb --args /mnt/CrossModalMultivectorSearch/build/tests/test_dive_emb_pair_recall \
    --query_path ${path}/img.fbin \
    --base_data_path ${path}/txt.fbin \
    --query_multivector_size $i \
    --evaluation_save_path $res_path/aggregated_results.txt \
    --evaluation_save_prefix $res_path/i2t \
    --k 10 \
    --dist ip \
    --num_samples 1000
    echo "i2t done"

  /mnt/CrossModalMultivectorSearch/build/tests/test_dive_emb_pair_recall \
    --query_path ${path}/txt.fbin \
    --base_data_path ${path}/img.fbin \
    --query_multivector_size $i \
    --evaluation_save_path $res_path/aggregated_results.txt \
    --evaluation_save_prefix $res_path/t2i \
    --k 10 \
    --dist ip \
    --num_samples 1000
    echo "t2i done"
done
