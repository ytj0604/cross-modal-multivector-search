#!/bin/bash

# Set paths and parameters
prefix=/mnt/dive/
roargraph_dir=/mnt/RoarGraph/build/
text_data=${prefix}/coco_test_txt_embs.fbin
image_data=${prefix}/coco_test_img_embs.fbin

num_threads=1
topk=20

M_PJBP=${1:-20}

# beam_width=(48 96 128)
beam_width=(20 40 60 80 100 120 140 160 180 200 2000)
L_pq_values=$(IFS=" "; echo "${beam_width[*]}")

evaluation_save_path=${prefix}/results/aggregated_results.txt

# Function to run a single search experiment
run_search() {
  local query_data=$1
  local base_data=$2
  local index_name=$3
  local gt_path=$4
  local evaluation_save_prefix=$5

  ${roargraph_dir}/tests/test_search_roargraph --data_type float \
    --dist ip --base_data_path ${base_data} \
    --projection_index_save_path ${index_name} \
    --gt_path ${gt_path} \
    --query_path ${query_data} \
    --L_pq ${L_pq_values} \
    --k ${topk} -T ${num_threads} \
    --evaluation_save_prefix ${evaluation_save_prefix} \
    --evaluation_save_path ${evaluation_save_path}
}

# Run experiments for all 4 combinations
echo "Running search experiments with num_threads=${num_threads} and topk=${topk}..."

# Image to Image (i2i)
run_search ${image_data} ${image_data} \
  ${prefix}/i2i_Roar_${M_PJBP}.index \
  ${prefix}/i2i.gt.bin \
  ${prefix}/results/i2i_Roar_${M_PJBP}

# Text to Text (t2t)
run_search ${text_data} ${text_data} \
  ${prefix}/t2t_Roar_${M_PJBP}.index \
  ${prefix}/t2t.gt.bin \
  ${prefix}/results/t2t_Roar_${M_PJBP}

# Text to Image (t2i)
run_search ${text_data} ${image_data} \
  ${prefix}/t2i_Roar_${M_PJBP}.index \
  ${prefix}/t2i.gt.bin \
  ${prefix}/results/t2i_Roar_${M_PJBP}

# Image to Text (i2t)
run_search ${image_data} ${text_data} \
  ${prefix}/i2t_Roar_${M_PJBP}.index \
  ${prefix}/i2t.gt.bin \
  ${prefix}/results/i2t_Roar_${M_PJBP}

echo "Search experiments completed!"
