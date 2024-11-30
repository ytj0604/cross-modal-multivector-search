#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_bipartite.h"

namespace po = boost::program_options;

float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t *res, uint32_t *gt) {
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<uint32_t> intersection;
        std::vector<uint32_t> temp_res(res + i * k, res + i * k + k);
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) intersection.push_back(p);
        }

        total_count += static_cast<uint32_t>(intersection.size());
    }
    return static_cast<float>(total_count) / (float)(k * q_num);
}

double ComputeRderr(float* gt_dist, uint32_t gt_dim, std::vector<std::vector<float>>& res_dists, uint32_t k, efanna2e::Metric metric) {
    double total_err = 0;
    uint32_t q_num = res_dists.size();

    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<float> one_gt(gt_dist + i * gt_dim, gt_dist + i * gt_dim + k);
        std::vector<float> temp_res(res_dists[i].begin(), res_dists[i].end());
        if (metric == efanna2e::INNER_PRODUCT) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = -1.0 * temp_res[j];
            }
        } else if (metric == efanna2e::COSINE) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = 2.0 * ( 1.0 - (-1.0 * temp_res[j]));
            }
        }
        double err = 0.0;
        for (uint32_t j = 0; j < k; j++) {
            err += std::fabs(temp_res[j] - one_gt[j]) / double(one_gt[j]);
        }
        err = err / static_cast<double>(k);
        total_err = total_err + err;
    }
    return total_err / static_cast<double>(q_num);
}

int main(int argc, char **argv) {
    std::string base_data_file;
    std::string query_file;
    std::string sampled_query_data_file;
    std::string gt_file;

    std::string bipartite_index_save_file, projection_index_save_file;
    std::string data_type;
    std::string dist;
    std::vector<uint32_t> L_vec;
    // uint32_t L_pq;
    uint32_t num_threads;
    uint32_t k;
    std::string evaluation_save_path = "";
    std::string evaluation_save_prefix = "";
    uint32_t max_pq;
    uint32_t min_pq;
    uint32_t max_pq_size_budget;
    uint32_t query_multivector_size;
    bool enable_adaptive_expansion;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), "distance function <l2/ip>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        // desc.add_options()("sampled_query_data_path", po::value<std::string>(&sampled_query_data_file)->required(),
        //                    "Sampled query file in bin format");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(), "Query file in bin format");
        desc.add_options()("gt_path", po::value<std::string>(&gt_file)->required(), "Groundtruth file in bin format");
        // desc.add_options()("query_data_path",
        //                    po::value<std::string>(&query_data_file)->required(),
        //                    "Query file in bin format");
        // desc.add_options()("bipartite_index_save_path", po::value<std::string>(&bipartite_index_save_file)->required(),
        //                    "Path prefix for saving bipartite index file components");
        desc.add_options()("projection_index_save_path",
                           po::value<std::string>(&projection_index_save_file)->required(),
                           "Path prefix for saving projetion index file components");
        // desc.add_options()("L_pq", po::value<std::vector<uint32_t>>(&L_vec)->multitoken()->required(),
        //                    "Priority queue length for searching");
        desc.add_options()("k", po::value<uint32_t>(&k)->default_value(1)->required(), "k nearest neighbors");
        desc.add_options()("evaluation_save_path", po::value<std::string>(&evaluation_save_path),
                           "Path prefix for saving evaluation results");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");
        desc.add_options()("evaluation_save_prefix", po::value<std::string>(&evaluation_save_prefix),
                           "Path prefix for saving evaluation results");
        desc.add_options()("max_pq", po::value<uint32_t>(&max_pq)->default_value(2000), "max priority queue length");
        desc.add_options()("min_pq", po::value<uint32_t>(&min_pq)->default_value(5), "min priority queue length");
        desc.add_options()("max_pq_size_budget", po::value<uint32_t>(&max_pq_size_budget)->default_value(10000), "max priority queue size budget");
        desc.add_options()("query_multivector_size", po::value<uint32_t>(&query_multivector_size)->default_value(4), "query multivector size");
        desc.add_options()("enable_adaptive_expansion", po::value<bool>(&enable_adaptive_expansion)->default_value(true), "enable adaptive expansion");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    uint32_t base_num, base_dim, sq_num = 0, sq_dim = 0;
    efanna2e::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    if (!sampled_query_data_file.empty()) {
        efanna2e::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);
    }

    efanna2e::Parameters parameters;

    parameters.Set<uint32_t>("num_threads", num_threads);
    omp_set_num_threads(num_threads);
    uint32_t q_pts, q_dim;
    efanna2e::load_meta<float>(query_file.c_str(), q_pts, q_dim);
    float *query_data = nullptr;
    efanna2e::load_data<float>(query_file.c_str(), q_pts, q_dim, query_data);
    float *aligned_query_data = efanna2e::data_align(query_data, q_pts, q_dim);
    assert(q_pts % query_multivector_size == 0);
    q_pts = q_pts / query_multivector_size;
    std::cout << "q_pts: " << q_pts << std::endl;
    uint32_t gt_pts, gt_dim;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    efanna2e::load_gt_meta<uint32_t>(gt_file.c_str(), gt_pts, gt_dim);
    // efanna2e::load_gt_data<uint32_t>(gt_file.c_str(), gt_pts, gt_dim, gt_ids);
    efanna2e::load_gt_data_with_dist<uint32_t, float>(gt_file.c_str(), gt_pts, gt_dim, gt_ids, gt_dists);
    efanna2e::Metric dist_metric = efanna2e::INNER_PRODUCT;
    if (dist == "l2") {
        dist_metric = efanna2e::L2;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        dist_metric = efanna2e::INNER_PRODUCT;
        std::cout << "Using inner product as distance metric" << std::endl;
    } else if (dist == "cosine") {
        dist_metric = efanna2e::COSINE;
            std::cout << "Using cosine as distance metric" << std::endl;
    } else {
        std::cout << "Unknown distance type: " << dist << std::endl;
        return -1;
    }

    if (!std::filesystem::exists(projection_index_save_file.c_str())) {
        std::cout << "projection index file does not exist." << std::endl;
        return -1;
    }

    efanna2e::IndexBipartite index(q_dim, base_num + sq_num, dist_metric, nullptr);

    index.LoadSearchNeededData(base_data_file.c_str(), sampled_query_data_file.c_str());

    std::cout << "Load graph index: " << projection_index_save_file << std::endl;
    index.LoadProjectionGraph(projection_index_save_file.c_str());

    if (index.need_normalize) {
        std::cout << "Normalizing query data" << std::endl;
        for (uint32_t i = 0; i < q_pts; i++) {
            efanna2e::normalize<float>(aligned_query_data + i * q_dim, q_dim);
        }
    }
    index.InitVisitedListPool(query_multivector_size);

    // Search
    std::cout << "k: " << k << std::endl;
    uint32_t *res = new uint32_t[q_pts * k];
    memset(res, 0, sizeof(uint32_t) * q_pts * k);
    // std::vector<std::vector<float>> res_dists(q_pts, std::vector<float>(k, 0.0));
    uint32_t *projection_cmps_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    memset(projection_cmps_vec, 0, sizeof(uint32_t) * q_pts);
    uint32_t *hops_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    float *projection_latency_vec = (float *)aligned_alloc(4, sizeof(float) * q_pts);
    memset(projection_latency_vec, 0, sizeof(float) * q_pts);
    std::ofstream evaluation_out;
    std::vector<std::vector<unsigned int>> indices(query_multivector_size);
    for (auto &ind : indices) {
        ind.reserve(max_pq);
    }
    std::vector<std::vector<float>> res_dists(query_multivector_size);
    for (auto &res_dist : res_dists) {
        res_dist.reserve(max_pq);
    }
    // if (!evaluation_save_path.empty()) {
    //     evaluation_out.open(evaluation_save_path, std::ios::out);
    // }
    // std::cout << "Using thread: " << num_threads << std::endl;
    // std::cout << "L_pq" << "\t\tQPS" << "\t\t\tavg_visited" << "\tmean_latency" << "\trecall@" << k << "\tavg_hops" << std::endl;

    if (!evaluation_save_path.empty()) {
        // Open evaluation file in append mode
        evaluation_out.open(evaluation_save_path, std::ios::app);
        if (!evaluation_out.is_open()) {
            std::cerr << "Error opening evaluation file: " << evaluation_save_path << std::endl;
            return -1;
        }

        // Write the name of the projection index file to the evaluation file
        // evaluation_out << "Index File: " << projection_index_save_file << "\n";
    }

    parameters.Set<uint32_t>("min_pq", min_pq);
    parameters.Set<uint32_t>("max_pq", max_pq);
    parameters.Set<uint32_t>("max_pq_size_budget", max_pq_size_budget);


    // Construct the evaluation file path dynamically
    std::string evaluation_file_path =
        evaluation_save_prefix + "_" + std::to_string(max_pq_size_budget) + ".tsv";

    std::ofstream tsv_out(evaluation_file_path, std::ios::out);
    if (!tsv_out.is_open()) {
        std::cerr << "Error opening output file: " << evaluation_file_path << std::endl;
        return -1;
    }

    // Start measuring total time
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<const float *> queries;
    queries.resize(query_multivector_size);
    int64_t total_comparison = 0;

#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < q_pts; ++i) {
        // Start timing for the individual query
        auto query_start = std::chrono::high_resolution_clock::now();

        // Set the query
        for(int32_t j = 0; j < query_multivector_size; j++) {
            queries[j] = aligned_query_data + i * q_dim * query_multivector_size + j * q_dim;
        }

        for(auto &ind : indices) {
            ind.clear();
        }
        for (auto &res_dist : res_dists) {
            res_dist.clear();
        }
        // Perform the search for the current query
        auto ret_val = index.SearchMultivectorOnRoarGraph(queries, k, i, parameters, indices, res_dists, enable_adaptive_expansion);
        // End timing for the individual query
        auto query_end = std::chrono::high_resolution_clock::now();
        auto query_diff = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start).count();
        double query_time_seconds = static_cast<double>(query_diff) / 1'000'000;  // Convert to seconds with precision
        for(auto &pair: ret_val) {
            total_comparison += pair.first;
        }
        // Save the results
        // projection_cmps_vec[i] = ret_val.first;
        // hops_vec[i] = ret_val.second;

        // Write the results to the TSV file
    #pragma omp critical
        {
            tsv_out << i << "\t" << query_time_seconds << "\n";  // Write the query execution time in microseconds
            for(int32_t j = 0; j < query_multivector_size; j++) {
                tsv_out << j;
                for (int k = 0; k < indices[j].size(); k++) {
                    tsv_out << "\t" << indices[j][k] << "\t" << -res_dists[j][k];
                }
                tsv_out << "\n";
            }
            // for (int j = 0; j < k; j++) {  // Adjusted for min-heap
            //     tsv_out << "\t" << res[i * k + j] << "\t" << -res_dists[i][j];
            // }
        }
        // }

        // End measuring total time
        auto end = std::chrono::high_resolution_clock::now();
        auto total_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double total_time_seconds = static_cast<double>(total_diff) / 1000.0;  // Convert to seconds


        // Calculate QPS
        double qps = static_cast<double>(q_pts) / total_time_seconds;

        // Print summary
        // float recall = ComputeRecall(q_pts, k, gt_dim, res, gt_ids);
        // float avg_projection_cmps = std::accumulate(projection_cmps_vec, projection_cmps_vec + q_pts, 0.0) / q_pts;
        // float avg_hops = std::accumulate(hops_vec, hops_vec + q_pts, 0.0) / q_pts;

        // std::cout << "L_pq: " << L_pq 
        //         << ", QPS: " << qps  // Print QPS
        //         << ", Recall@" << k << ": " << recall
        //         << ", Avg Hops: " << avg_hops << std::endl;
        // std::cout << "done" << std::endl;
            // Write the results to the evaluation file
        // if (evaluation_out.is_open()) {
        //     evaluation_out << "L_pq: " << L_pq 
        //                 << ", QPS: " << qps
        //                 << ", Recall@" << k << ": " << recall
        //                 << ", Avg Hops: " << avg_hops << "\n";
        // }
    }
    tsv_out.close();  

    evaluation_out << max_pq_size_budget << "\t" << total_comparison << std::endl;
    if (evaluation_out.is_open()) {
        evaluation_out.close();
    }

    delete[] res;
    free(projection_cmps_vec);
    free(hops_vec);
    free(projection_latency_vec);
    delete[] aligned_query_data;
    delete[] gt_ids;

    return 0;
}