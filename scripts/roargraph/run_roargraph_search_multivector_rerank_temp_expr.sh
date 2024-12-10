#!/bin/bash

# Set paths and parameters
# prefix=/mnt/dive/4/
# roargraph_dir=/mnt/RoarGraph/build/
# text_data=${prefix}/coco_test_4_txt_embs.fbin
# image_data=${prefix}/coco_test_4_img_embs.fbin
# result_dir=${prefix}/results/

# if [ ! -d ${result_dir} ]; then
#   mkdir -p ${result_dir}
# fi

# num_threads=1
# topk=20

# M_PJBP=${1:-35}
# query_multivector_size=4
# beam_width_budget=(20 40 80 120 160 200)
# min_beam_width=5
# max_beam_width=2000

# evaluation_save_path=${prefix}/results/aggregated_results.txt

# # Function to run a single search experiment
# run_search() {
#   local query_data=$1
#   local base_data=$2
#   local index_name=$3
#   local gt_path=$4
#   local evaluation_save_prefix=$5
#   local budget=$6

#   ${roargraph_dir}/tests/test_search_multivector_roargraph --data_type float \
#     --dist ip --base_data_path ${base_data} \
#     --projection_index_save_path ${index_name} \
#     --gt_path ${gt_path} \
#     --query_path ${query_data} \
#     --k ${topk} -T ${num_threads} \
#     --max_pq ${max_beam_width} \
#     --min_pq ${min_beam_width} \
#     --max_pq_size_budget ${budget} \
#     --evaluation_save_prefix ${evaluation_save_prefix} \
#     --evaluation_save_path ${evaluation_save_path} \
#     --query_multivector_size ${query_multivector_size}
# }

# # Run experiments for all 4 combinations
# echo "Running search experiments with num_threads=${num_threads} and topk=${topk}..."

# for budget in "${beam_width_budget[@]}"; do
#   echo "Testing with max_pq_size_budget=${budget}..."

#   # Image to Image (i2i)
#   run_search ${image_data} ${image_data} \
#     ${prefix}/i2i_Roar_${M_PJBP}.index \
#     ${prefix}/i2i.gt.bin \
#     ${result_dir}/i2i_Roar_${M_PJBP} \
#     ${budget}

#   # Text to Text (t2t)
#   run_search ${text_data} ${text_data} \
#     ${prefix}/t2t_Roar_${M_PJBP}.index \
#     ${prefix}/t2t.gt.bin \
#     ${result_dir}/t2t_Roar_${M_PJBP} \
#     ${budget}

#   # Text to Image (t2i)
#   run_search ${text_data} ${image_data} \
#     ${prefix}/t2i_Roar_${M_PJBP}.index \
#     ${prefix}/t2i.gt.bin \
#     ${result_dir}/t2i_Roar_${M_PJBP} \
#     ${budget}

#   # Image to Text (i2t)
#   run_search ${image_data} ${text_data} \
#     ${prefix}/i2t_Roar_${M_PJBP}.index \
#     ${prefix}/i2t.gt.bin \
#     ${result_dir}/i2t_Roar_${M_PJBP} \
#     ${budget}
# done
# echo "Search experiments completed!"

for i in {3..5}; do

  # Set paths and parameters
  prefix=/mnt/dive/${i}/
  roargraph_dir=/mnt/RoarGraph/build/
  text_data=${prefix}/coco_test_${i}_txt_embs.fbin
  image_data=${prefix}/coco_test_${i}_img_embs.fbin
  result_dir=${prefix}/result_hj_alg/
  gt_dir=/mnt/dive/ground_truth/

  if [ ! -d ${result_dir} ]; then
    mkdir -p ${result_dir}
  fi

  MKL_NUM_THREADS=1
  num_threads=1
  topk=10

  M_PJBP=${1:-35}
  query_multivector_size=${i}
  beam_width_budget=(20 40 80 120 160 200)
  # beam_width_budget=(1500 2000 3000 5000)
  # beam_width_budget=(1000)
  min_beam_width=5
  # max_beam_width=1000
  enable_adaptive_expansion=true

  evaluation_save_path=${result_dir}/aggregated_results.txt

  # Function to run a single search experiment
  run_search() {
    local query_data=$1
    local base_data=$2
    local index_name=$3
    local gt_path=$4
    local evaluation_save_prefix=$5
    local budget=$6
    local set_gt_path=$7

    ${roargraph_dir}/tests/test_search_multivector_rerank_temp --data_type float \
      --dist cosine --base_data_path ${base_data} \
      --projection_index_save_path ${index_name} \
      --gt_path ${gt_path} \
      --query_path ${query_data} \
      --k ${topk} -T ${num_threads} \
      --max_pq ${budget} \
      --min_pq ${min_beam_width} \
      --max_pq_size_budget ${budget} \
      --evaluation_save_prefix ${evaluation_save_prefix} \
      --evaluation_save_path ${evaluation_save_path} \
      --query_multivector_size ${query_multivector_size} \
      --enable_adaptive_expansion ${enable_adaptive_expansion} \
      --set_gt_path ${set_gt_path}
  }

  # Run experiments for all 4 combinations
  echo "Running search experiments with num_threads=${num_threads} and topk=${topk}..."

  for budget in "${beam_width_budget[@]}"; do
    run_search ${text_data} ${image_data} \
      ${prefix}/t2i_Roar_${M_PJBP}.index \
      ${prefix}/t2i.gt.bin \
      ${result_dir}/t2i_Roar_${M_PJBP} \
      ${budget} \
      ${gt_dir}/coco_test_${i}_txt_gts.ibin
  done

  for budget in "${beam_width_budget[@]}"; do
      run_search ${image_data} ${text_data} \
      ${prefix}/i2t_Roar_${M_PJBP}.index \
      ${prefix}/i2t.gt.bin \
      ${result_dir}/i2t_Roar_${M_PJBP} \
      ${budget} \
      ${gt_dir}/coco_test_${i}_img_gts.ibin
  done

  echo "Search experiments completed!"
done