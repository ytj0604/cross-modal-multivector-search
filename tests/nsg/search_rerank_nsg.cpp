#include <efanna2e-nsg/index_nsg.h>
#include <efanna2e-nsg/util.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "multivector_reranker.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string index_path;
  std::string query_file;
  std::string base_data_file;
  std::string evaluation_save_path = "";
  std::string evaluation_save_prefix = "";
  std::string set_gt_path = "";
  uint32_t query_multivector_size;
  uint32_t k;
  std::string dist;
  uint32_t total_beam_width;
  // Currently not used; will be used in future after implementing adaptive
  // expansion.
  uint32_t max_pq;
  uint32_t min_pq;
  uint32_t max_pq_size_budget;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("index_path",
                       po::value<std::string>(&index_path)->required(),
                       "Path to the index file");
    desc.add_options()("query_path",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in bin format");
    desc.add_options()("base_data_path",
                       po::value<std::string>(&base_data_file)->required(),
                       "Input data file in bin format");
    desc.add_options()(
        "query_multivector_size",
        po::value<uint32_t>(&query_multivector_size)->default_value(4),
        "Query multivector size");
    desc.add_options()("evaluation_save_path",
                       po::value<std::string>(&evaluation_save_path),
                       "Path to save evaluation results");
    desc.add_options()("evaluation_save_prefix",
                       po::value<std::string>(&evaluation_save_prefix),
                       "Prefix for saving evaluation results");
    desc.add_options()("set_gt_path", po::value<std::string>(&set_gt_path),
                       "Set ground truth path");
    desc.add_options()("max_pq",
                       po::value<uint32_t>(&max_pq)->default_value(2000),
                       "Max priority queue length");
    desc.add_options()("min_pq", po::value<uint32_t>(&min_pq)->default_value(5),
                       "Min priority queue length");
    desc.add_options()(
        "max_pq_size_budget",
        po::value<uint32_t>(&max_pq_size_budget)->default_value(10000),
        "Max priority queue size budget");
    desc.add_options()("k", po::value<uint32_t>(&k)->default_value(10),
                       "k nearest neighbors");
    desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                       "Distance function <l2/ip>");
    desc.add_options()("total_beam_width",
                       po::value<uint32_t>(&total_beam_width)->required(),
                       "Beam width for search");
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

  auto query_matrix = Loader::LoadEmbeddingVector(query_file);
  auto data_matrix = Loader::LoadEmbeddingVector(base_data_file);
  auto queries_ = Loader::LoadEmbeddingVectorAsFloatVector(query_file);
  auto data_ = Loader::LoadEmbeddingVectorAsFloatVector(base_data_file);

  // This might need to be changed.
  uint32_t num_query_sets = query_matrix.rows() / query_multivector_size;
  assert(num_query_sets * query_multivector_size == query_matrix.rows());

  uint32_t dim = queries_->at(0).size();
  uint32_t num_query_vectors = queries_->size();
  float *query_load = new float[num_query_vectors * dim];
  for (size_t i = 0; i < num_query_vectors; i++) {
    for (size_t j = 0; j < dim; j++) {
      query_load[i * dim + j] = queries_->at(i)[j];
    }
  }

  // Dimension should be the same for l2 or ip distance.
  assert(dim = data_->at(0).size());
  uint32_t num_data_vectors = data_->size();
  float *data_load = new float[num_data_vectors * dim];
  for (size_t i = 0; i < num_data_vectors; i++) {
    for (size_t j = 0; j < dim; j++) {
      data_load[i * dim + j] = data_->at(i)[j];
    }
  }

  MultiVectorReranker reranker;
  reranker.SetDataVector(data_matrix);
  reranker.SetQueryVector(query_matrix);
  reranker.SetMultiVectorCardinality(query_multivector_size);
  reranker.SetK(k);
  reranker.SetDistanceMetric(
      "smooth_chamfer", (dist == "cosine" || dist == "ip") ? "cosine" : "l2");
  reranker.SetVectorID2VectorSetIDMapping(
      [&query_multivector_size](VectorID vid) -> VectorSetID {
        return vid / query_multivector_size;
      });

  auto set_level_ground_truth = Loader::LoadGroundTruth(set_gt_path);
  RecallCalculator recall_calculator;
  recall_calculator.SetGroundTruth(set_level_ground_truth);
  recall_calculator.SetK(k);

  std::string evaluation_file_path = evaluation_save_prefix + "_" + ".tsv";
  std::ofstream tsv_out(evaluation_file_path, std::ios::out);
  if (!tsv_out.is_open()) {
    std::cerr << "Error opening output file: " << evaluation_file_path
              << std::endl;
    return -1;
  }

  efanna2e_nsg::Metric metric;
  if (dist == "l2") {
    metric = efanna2e_nsg::FAST_L2;
  } else if (dist == "ip") {
    metric = efanna2e_nsg::INNER_PRODUCT;
  } else {
    std::cerr << "Invalid distance metric: " << dist << std::endl;
    return -1;
  }
  efanna2e_nsg::IndexNSG index(dim, num_data_vectors, metric, nullptr);
  index.Load(index_path.c_str());
  index.OptimizeGraph(data_load);
  efanna2e_nsg::Parameters params;
  auto per_query_vector_beam_width = total_beam_width / query_multivector_size;
  params.Set<unsigned>("L_search", per_query_vector_beam_width);
  params.Set<unsigned>("P_search", per_query_vector_beam_width);

  double total_search_time = 0;
  double total_rerank_time = 0;
  double total_recall = 0;

  std::vector<VectorSetID> reranked_indices;
  std::vector<std::vector<VectorID>> indices(
      query_multivector_size,
      std::vector<VectorID>(per_query_vector_beam_width, 0));

  for (uint32_t i = 0; i < num_query_sets; ++i) {
    auto query_vec_index_start = i * query_multivector_size;
    auto search_start = std::chrono::high_resolution_clock::now();
    for (int32_t j = 0; j < query_multivector_size; j++) {
      std::fill(indices[j].begin(), indices[j].end(), 0);
      index.SearchWithOptGraph(
          query_load + (i * query_multivector_size + j) * dim,
          per_query_vector_beam_width, params, indices[j].data());
    }
    auto search_end = std::chrono::high_resolution_clock::now();
    reranker.Rerank(i, indices, reranked_indices);
    auto rerank_end = std::chrono::high_resolution_clock::now();
    auto search_time_seconds =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(search_end -
                                                                  search_start)
                .count()) /
        1e6;
    auto rerank_time_seconds =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(rerank_end -
                                                                  search_end)
                .count()) /
        1e6;
    total_search_time += search_time_seconds;
    total_rerank_time += rerank_time_seconds;
    double recall = recall_calculator.ComputeRecall(i, reranked_indices);
    total_recall += recall;
    tsv_out << i << "\t" << search_time_seconds << "\t" << rerank_time_seconds
            << "\n";
    tsv_out << recall << "\n";
    for (int32_t j = 0; j < query_multivector_size; j++) {
      tsv_out << j;
      for (int k = 0; k < indices[j].size(); k++) {
        tsv_out << "\t" << indices[j][k];
      }
      tsv_out << "\n";
    }
    for (auto in = 0; in < reranked_indices.size(); in++) {
      tsv_out << reranked_indices[in] << "\t";
    }
    tsv_out << "\n";
  }
  tsv_out.close();
  delete[] query_load;
  delete[] data_load;
  double QPS = num_query_sets / (total_search_time + total_rerank_time);
  double recall = total_recall / num_query_sets;
  std::ofstream evaluation_out(evaluation_save_path, std::ios::app);
  if (!evaluation_out.is_open()) {
    std::cerr << "Error: Unable to open or create the file at "
              << evaluation_save_path << std::endl;
    return -1;
  }

  evaluation_out << total_beam_width << "\t";
  evaluation_out << recall << "\t" << QPS << "\t"
                 << total_rerank_time / (total_rerank_time + total_search_time)
                 << "\n";
  if (evaluation_out.is_open()) {
    evaluation_out.close();
  }

  return 0;
}