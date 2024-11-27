prefix=/mnt/dive/4
image=coco_test_4_img_embs.fbin
text=coco_test_4_txt_embs.fbin
/mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${text} --query_file ${prefix}/${text}  --gt_file ${prefix}/t2t.gt.bin --K 1000
/mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${image} --query_file ${prefix}/${image} --gt_file ${prefix}/i2i.gt.bin --K 1000
/mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${image} --query_file ${prefix}/${text} --gt_file ${prefix}/t2i.gt.bin --K 1000
/mnt/RoarGraph/build/thirdparty/DiskANN/tests/utils/compute_groundtruth --data_type float --dist_fn mips --base_file ${prefix}/${text} --query_file ${prefix}/${image} --gt_file ${prefix}/i2t.gt.bin --K 1000