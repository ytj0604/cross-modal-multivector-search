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
#include <thread>
#include <vector>

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_bipartite.h"
#include "multivector_reranker.h"
#include "utility_methods.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string vector_gt_path;
  std::string query_path;
  std::string data_path;
  std::string dist;
  uint32_t k;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("vector_gt_path",
                       po::value<std::string>(&vector_gt_path)->required(),
                       "Path to the vector ground truth file");
    desc.add_options()("query_path",
                       po::value<std::string>(&query_path)->required(),
                       "Query file in bin format");
    desc.add_options()("base_data_path",
                       po::value<std::string>(&data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("k", po::value<uint32_t>(&k)->required(),
                       "k nearest neighbors");
    desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                       "Distance function <l2/ip>");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }
  auto data_matrix = Loader::LoadEmbeddingVector(data_path);
  auto query_matrix = Loader::LoadEmbeddingVector(query_path);

  MultiVectorReranker reranker;
  reranker.SetDataVector(data_matrix);
  reranker.SetQueryVector(query_matrix);
  reranker.SetK(k);
  reranker.SetGPUBatchSize(10000);
  reranker.SetUseGPU(true);
  const Cardinality query_batch_size = 100;

  reranker.SetDistanceMetric(
      "smooth_chamfer",
      (dist == "cosine" || dist == "ip") ? "cosine_gpu" : "l2");
  std::ofstream out(vector_gt_path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Cannot open file: " + vector_gt_path);
  }

  uint32_t num_queries = query_matrix.rows();
  uint32_t num_gt_per_query = k;
  out.write(reinterpret_cast<const char*>(&num_queries), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&num_gt_per_query), sizeof(uint32_t));
  std::vector<std::vector<VectorID>> ground_truth(num_queries);
  std::atomic<uint32_t> progress(0);

  auto start_real_time = std::chrono::high_resolution_clock::now();

  std::thread progress_thread([&progress, &num_queries, &start_real_time]() {
    while (true) {
      auto current = progress.load();
      if (current > 0) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                                current_time - start_real_time)
                                .count();

        double rate = static_cast<double>(current) / elapsed_time;
        uint32_t remaining_queries = num_queries - current;
        uint32_t remaining_seconds =
            static_cast<uint32_t>(remaining_queries / rate);
        uint32_t minutes = remaining_seconds / 60;
        uint32_t seconds = remaining_seconds % 60;
        std::cout << "Progress: " << current << "/" << num_queries
                  << " \t Remaining: " << minutes << ":" << std::setfill('0')
                  << std::setw(2) << seconds << std::endl;
      } else {
        std::cout << "Progress: " << current << "/" << num_queries << std::endl;
      }

      if (current >= num_queries) {
        break;
      }

      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });

  omp_set_num_threads(80);

#pragma omp parallel for
  for (VectorID i = 0; i < num_queries; i += query_batch_size) {
    Cardinality size = std::min(query_batch_size, num_queries - i);
    reranker.RankAllVectorsBySequentialScan(i, size, ground_truth);
    progress.fetch_add(size);
  }
  for (const auto& gt : ground_truth) {
    if (gt.size() != k) {
      std::cerr << "Ground truth size mismatch. Expected: " << k
                << " but got: " << gt.size() << std::endl;
      throw std::runtime_error("Ground truth size mismatch.");
    }
    out.write(reinterpret_cast<const char*>(gt.data()), k * sizeof(VectorID));
  }
  progress_thread.join();

  out.close();
}