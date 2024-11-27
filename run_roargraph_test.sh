mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
prefix=../data/laion-10M
./tests/test_build_roargraph --data_type float --dist cosine \
--base_data_path ${prefix}/base.10M.fbin  \
--sampled_query_data_path ${prefix}/query.train.10M.fbin \
--projection_index_save_path ${prefix}/laion_10M_roar.index \
--learn_base_nn_path ${prefix}/train.gt.bin \
--M_sq 100 --M_pjbp 35 --L_pjpq 500 -T 64