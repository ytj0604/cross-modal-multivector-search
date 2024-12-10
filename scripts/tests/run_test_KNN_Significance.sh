#!/bin/bash
# k=10
# type=t2i
# card=4
# num_raw_vector_results=100

# if [ "$type" == "i2t" ]; then
#   type_name="img"
# else
#   type_name="txt"
# fi

# /mnt/RoarGraph/build/tests/test_KNN_significance\
#   --set_gt_path /mnt/dive/ground_truth//coco_test_${card}_${type_name}_gts.ibin \
#   --vector_gt_path /mnt/dive/${card}/${type}.gt.bin \
#   --multivector_cardinality ${card} \
#   --output /mnt/tjyoon/knn_significance/${card}_${type}_${k}.txt \
#   --k ${k} \
#   --num_raw_vector_results ${num_raw_vector_results}
k=10
/mnt/RoarGraph/build/tests/test_KNN_significance\
  --set_gt_path /mnt/rand_vec/4/set_1to2.ibin \
  --vector_gt_path /mnt/rand_vec/4/1to2.gt.bin \
  --multivector_cardinality 4 \
  --output /mnt/tjyoon/knn_significance/rand_4_1to2_${k}.txt \
  --k ${k} \
  --num_raw_vector_results 10