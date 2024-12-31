build_dir="$(dirname "$(realpath "$0")")/../../build/"
i=2
base=/mnt/datasets/dive_coco_train/2_test
query_path=$base/img_query.fbin
data_path=$base/img.fbin
${build_dir}/tests/test_NN_cosine_sim_dist \
  --query_path $query_path \
  --data_path $data_path \
  --output /mnt/tjyoon/nnwise_cosine_dist/result_dive_coco_train_2/img2img_result.txt

base=/mnt/datasets/dive_coco_train/2_test
query_path=$base/img_query.fbin
data_path=$base/txt_smpl.fbin
${build_dir}/tests/test_NN_cosine_sim_dist \
  --query_path $query_path \
  --data_path $data_path \
  --output /mnt/tjyoon/nnwise_cosine_dist/result_dive_coco_train_2/img2txt_result.txt

