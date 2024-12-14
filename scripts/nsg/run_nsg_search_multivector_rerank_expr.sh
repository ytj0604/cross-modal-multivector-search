#!/bin/bash

build_dir="$(dirname "$(realpath "$0")")/../../build/"

dist=ip
base_path=/mnt/dive/
k=10

for i in {1..5}; do
  for datatype in "img" "txt"; do
    total_beam_widths=(20 40 80 120 160 200 400 600 800 1000)
    for bw in "${total_beam_widths[@]}"; do
      path=${base_path}${i}/
      result_dir=${path}/result_nsg/
      if [ ! -d ${result_dir} ]; then
        mkdir ${result_dir}
      fi
      if [ $datatype == "img" ]; then
        query_type="txt"
        res_filename_prefix="t2i"
      else
        query_type="img"
        res_filename_prefix="i2t"
      fi
      index_path=${path}/${datatype}_nsg.index
      data_path=${path}/coco_test_${i}_${datatype}_embs.fbin
      query_path=${path}/coco_test_${i}_${query_type}_embs.fbin
      gt_path=/mnt/dive/ground_truth/coco_test_${i}_${query_type}_gts.ibin
      result_agg_path=${result_dir}/aggregated_results.txt

      ${build_dir}/tests/search_rerank_nsg \
        --base_data_path ${data_path} \
        --query_path ${query_path} \
        --index_path ${index_path} \
        --set_gt_path ${gt_path} \
        --evaluation_save_path ${result_agg_path} \
        --k ${k} \
        --dist ${dist} \
        --query_multivector_size $i \
        --evaluation_save_prefix ${result_dir}/${res_filename_prefix}_nsg_${M}_${bw} \
        --total_beam_width ${bw}
      echo "Completed search for ${res_filename_prefix} with beam width ${bw}"
    done
  done
done