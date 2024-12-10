# for i in {1..5}; do
#   prefix=/mnt/dive/${i}
#   image=coco_test_${i}_img_embs.fbin
#   text=coco_test_${i}_txt_embs.fbin
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${text} --query_file ${prefix}/${text}  --gt_file ${prefix}/t2t.gt.bin --K $((i * 25000))
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${image} --query_file ${prefix}/${image} --gt_file ${prefix}/i2i.gt.bin --K $((i * 5000))
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${image} --query_file ${prefix}/${text} --gt_file ${prefix}/t2i.gt.bin --K $((i * 5000))
#   /mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${text} --query_file ${prefix}/${image} --gt_file ${prefix}/i2t.gt.bin --K $((i * 25000))
# done

/mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file /mnt/rand_vec/4/vec_2.fbin --query_file /mnt/rand_vec/4/vec_1.fbin  --gt_file /mnt/rand_vec/4/1to2.gt.bin --K 40000