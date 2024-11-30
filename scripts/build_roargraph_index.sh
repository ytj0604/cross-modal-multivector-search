#!/bin/bash

# roargraph_dir=/mnt/RoarGraph/build/

# # Set the prefix for the dataset path
# prefix=/mnt/dive/4/

# # Define the value
# M_SQ=100
# M_PJBP=35
# L_PJPQ=100

# # File paths for text and image datasets
# text_data=${prefix}/coco_test_4_txt_embs.fbin
# image_data=${prefix}/coco_test_4_img_embs.fbin

# # Generate RoarGraph datasets for all 4 combinations
# echo "Generating RoarGraph datasets with M_PJBP=${M_PJBP}..."

# # Image to Image (i2i)
# ${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
#   --base_data_path ${image_data} \
#   --sampled_query_data_path ${image_data} \
#   --projection_index_save_path ${prefix}/i2i_Roar_${M_PJBP}.index \
#   --learn_base_nn_path ${prefix}/i2i.gt.bin \
#   --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

# # Text to Text (t2t)
# ${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
#   --base_data_path ${text_data} \
#   --sampled_query_data_path ${text_data} \
#   --projection_index_save_path ${prefix}/t2t_Roar_${M_PJBP}.index \
#   --learn_base_nn_path ${prefix}/t2t.gt.bin \
#   --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

# # Text to Image (t2i)
# ${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
#   --base_data_path ${image_data} \
#   --sampled_query_data_path ${text_data} \
#   --projection_index_save_path ${prefix}/t2i_Roar_${M_PJBP}.index \
#   --learn_base_nn_path ${prefix}/t2i.gt.bin \
#   --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

# # Image to Text (i2t)
# ${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
#   --base_data_path ${text_data} \
#   --sampled_query_data_path ${image_data} \
#   --projection_index_save_path ${prefix}/i2t_Roar_${M_PJBP}.index \
#   --learn_base_nn_path ${prefix}/i2t.gt.bin \
#   --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

# echo "RoarGraph generation completed!"

for i in {1..5}; do 
roargraph_dir=/mnt/RoarGraph/build/

# Set the prefix for the dataset path
prefix=/mnt/dive/${i}/

# Define the value
M_SQ=100
M_PJBP=35
L_PJPQ=100

# File paths for text and image datasets
text_data=${prefix}/coco_test_${i}_txt_embs.fbin
image_data=${prefix}/coco_test_${i}_img_embs.fbin

# Generate RoarGraph datasets for all 4 combinations
echo "Generating RoarGraph datasets with M_PJBP=${M_PJBP}..."

# Image to Image (i2i)
${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
  --base_data_path ${image_data} \
  --sampled_query_data_path ${image_data} \
  --projection_index_save_path ${prefix}/i2i_Roar_${M_PJBP}.index \
  --learn_base_nn_path ${prefix}/i2i.gt.bin \
  --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

# Text to Text (t2t)
${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
  --base_data_path ${text_data} \
  --sampled_query_data_path ${text_data} \
  --projection_index_save_path ${prefix}/t2t_Roar_${M_PJBP}.index \
  --learn_base_nn_path ${prefix}/t2t.gt.bin \
  --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

# Text to Image (t2i)
${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
  --base_data_path ${image_data} \
  --sampled_query_data_path ${text_data} \
  --projection_index_save_path ${prefix}/t2i_Roar_${M_PJBP}.index \
  --learn_base_nn_path ${prefix}/t2i.gt.bin \
  --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

# Image to Text (i2t)
${roargraph_dir}/tests/test_build_roargraph --data_type float --dist cosine \
  --base_data_path ${text_data} \
  --sampled_query_data_path ${image_data} \
  --projection_index_save_path ${prefix}/i2t_Roar_${M_PJBP}.index \
  --learn_base_nn_path ${prefix}/i2t.gt.bin \
  --M_sq ${M_SQ} --M_pjbp ${M_PJBP} --L_pjpq ${L_PJPQ} -T 64

echo "RoarGraph generation completed!"

done