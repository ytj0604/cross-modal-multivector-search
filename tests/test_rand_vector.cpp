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

#include "efanna2e/distance.h"
#include "efanna2e/neighbor.h"
#include "efanna2e/parameters.h"
#include "efanna2e/util.h"
#include "index_bipartite.h"
#include "multivector_reranker.h"
#include "utility_methods.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string set_gt_path;
  std::string vector_gt_path;
  uint32_t multi_vector_cardinality;
  std::string output;
  std::string query;
  std::string data;
  uint32_t k;
  uint32_t num_raw_vector_results;
  po::options_description desc{"Arguments"};
  try {
    // desc.add_options()("set_gt_path", po::value<std::string>(&set_gt_path)->required(), "set ground truth path");
    // desc.add_options()("vector_gt_path", po::value<std::string>(&vector_gt_path)->required(), "vector ground truth path");
    // desc.add_options()("multivector_cardinality", po::value<uint32_t>(&multi_vector_cardinality)->required(),
    //                    "multivector cardinality");
    desc.add_options()("query", po::value<std::string>(&query)->required(), "input file");
    desc.add_options()("data", po::value<std::string>(&data)->required(), "input file");
    desc.add_options()("output", po::value<std::string>(&output)->required(), "output file");

    // desc.add_options()("k", po::value<uint32_t>(&k)->required(), "k");
    // desc.add_options()("num_raw_vector_results", po::value<uint32_t>(&num_raw_vector_results)->required(), "num raw vector results");
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
  // Generate a random vectors:
  // 10000 vectorset, 4 vectors per vectorset, dim = 1024
  // UtilityMethods::GenerateRandomVectorsAndStore(output1, 10000 * 4, 1024);
  // UtilityMethods::GenerateRandomVectorsAndStore(output2, 10000 * 4, 1024);

  // Generate Set ground truth; vec1 is query and vec2 is base data.
  auto query_ = Loader::LoadEmbeddingVector(query);
  auto base_ = Loader::LoadEmbeddingVector(data);
  // MultiVectorReranker reranker;
  // reranker.SetQueryMultiVectorCardinality(4);
  // reranker.SetQueryVector(query);
  // reranker.SetDataVector(base);
  // reranker.SetDistanceMetric("smooth_chamfer", "cosine");
  // reranker.RerankAllAndGenerateSetGroundTruth(set_gt_path);
  UtilityMethods::TestCosineSimilarityDist(query_, base_, output);
}