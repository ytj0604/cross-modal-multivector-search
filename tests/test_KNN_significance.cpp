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
#include "test_class.h"

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string seg_gt_path;
  std::string vector_gt_path;
  uint32_t multi_vector_cardinality;
  std::string output;
  uint32_t k;
  uint32_t num_raw_vector_results;
  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("set_gt_path", po::value<std::string>(&seg_gt_path)->required(), "set ground truth path");
    desc.add_options()("vector_gt_path", po::value<std::string>(&vector_gt_path)->required(), "vector ground truth path");
    desc.add_options()("multivector_cardinality", po::value<uint32_t>(&multi_vector_cardinality)->required(),
                       "multivector cardinality");
    desc.add_options()("output", po::value<std::string>(&output)->required(), "output file");
    desc.add_options()("k", po::value<uint32_t>(&k)->required(), "k");
    desc.add_options()("num_raw_vector_results", po::value<uint32_t>(&num_raw_vector_results)->required(), "num raw vector results");
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
  auto sgt = Loader::LoadGroundTruth(seg_gt_path);
  auto vgt = Loader::LoadVectorGroundTruth(vector_gt_path);
  TestClass::TestKNNSignificance(sgt, vgt, multi_vector_cardinality, output, k, num_raw_vector_results);
}