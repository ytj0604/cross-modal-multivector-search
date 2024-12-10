#!/bin/bash
beam_width_budget=120
result_path=/mnt/rand_vec/4/result_enable_adaptive_expansion/
/mnt/RoarGraph/build/tests/test_search_multivector_rerank --data_type float \
      --dist cosine --base_data_path /mnt/rand_vec/4/vec_2.fbin \
      --projection_index_save_path /mnt/rand_vec/4/1to2.index \
      --gt_path /mnt/rand_vec/4/1to2.gt.bin \
      --query_path /mnt/rand_vec/4/vec_1.fbin \
      --k 10 -T 1 \
      --max_pq $beam_width_budget \
      --min_pq 5 \
      --max_pq_size_budget $beam_width_budget \
      --evaluation_save_prefix $result_path \
      --evaluation_save_path ${result_path}/aggregated_results.txt \
      --query_multivector_size 4 \
      --enable_adaptive_expansion true \
      --set_gt_path /mnt/rand_vec/4/set_1to2.ibin \