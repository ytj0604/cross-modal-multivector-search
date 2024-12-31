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
  std::string query_path;
  std::string data_path;
  std::string output;
  uint32_t k = 100;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("query_path",
                       po::value<std::string>(&query_path)->required(),
                       "set ground truth path");
    desc.add_options()("data_path",
                       po::value<std::string>(&data_path)->required(),
                       "vector ground truth path");
    desc.add_options()("output", po::value<std::string>(&output)->required(),
                       "output file");
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

  auto query_matrix = Loader::LoadEmbeddingVector(query_path);
  auto data_matrix = Loader::LoadEmbeddingVector(data_path);
  uint32_t num_query_sets = query_matrix.rows();

  MultiVectorReranker reranker;
  reranker.SetDataVector(data_matrix);
  reranker.SetQueryVector(query_matrix);
  reranker.SetQueryMultiVectorCardinality(1);
  reranker.SetK(k);
  reranker.SetUseGPU(true);
  reranker.SetGPUBatchSize(10000);
  reranker.SetDistanceMetric("smooth_chamfer", "cosine_gpu");
  std::vector<std::vector<VectorID>> ground_truth(num_query_sets);

  std::atomic<uint32_t> progress(0);
  std::atomic<bool> running(true);

  std::thread progress_thread([&running, &progress]() {
    while (running.load()) {
      auto current = progress.load();
      std::cout << "Progress: " << current << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });

  omp_set_num_threads(80);
#pragma omp parallel for
  for (uint32_t i = 0; i < num_query_sets; ++i) {
    reranker.RankAllVectorsBySequentialScan(i, 1, ground_truth);
    progress.fetch_add(1);
  }

  // From here, do not use GPU since the cudamemcpy overhead would be
  // significant.
  reranker.SetUseGPU(false);
  reranker.SetDistanceMetric("smooth_chamfer", "cosine");
  progress.store(0);

  std::vector<float> agg_distances;
#pragma omp parallel for
  for (uint32_t i = 0; i < num_query_sets; ++i) {
    std::vector<float> avg_distances;
    reranker.GetNNWiseDistance(ground_truth[i], avg_distances);
    progress.fetch_add(1);
#pragma omp critical
    {
      agg_distances.insert(agg_distances.end(), avg_distances.begin(),
                           avg_distances.end());
    }
  }
  running.store(false);
  auto csv_out = std::ofstream(output);
  if (!csv_out.is_open()) {
    throw std::runtime_error("Cannot open file: " + output);
  }
  for (auto i = 0; i < agg_distances.size(); i++) {
    csv_out << agg_distances[i] << ",";
  }
  progress_thread.join();
}