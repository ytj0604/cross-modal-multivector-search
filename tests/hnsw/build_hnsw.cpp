#include <gtest/gtest.h>
#include <hnswlib/hnswlib.h>
#include <multivector_reranker.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string base_data_file;                             // Read from argument
  int M;                                                  // Read from argument
  int ef_construction;                                    // Read from argument
  int num_threads = std::thread::hardware_concurrency();  // Read from argument
  int dim;                                                // Read from data file
  int max_elements;                                       // Read from data file
  std::string index_save_path;
  std::string dist;

  po::options_description desc("Allowed options");
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("base_data_path",
                       po::value<std::string>(&base_data_file)->required(),
                       "Input data file in bin format");
    desc.add_options()(
        "M", po::value<int>(&M)->required(),
        "Number of neighbors for base points to build the graph");
    desc.add_options()(
        "ef_construction", po::value<int>(&ef_construction)->required(),
        "Number of neighbors for base points to build the graph");
    desc.add_options()("num_threads", po::value<int>(&num_threads));
    desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                       "distance function <l2/ip>");
    desc.add_options()("index_save_path",
                       po::value<std::string>(&index_save_path)->required(),
                       "Path prefix for saving index file components");
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

  auto data = Loader::LoadEmbeddingVectorAsFloatVector(base_data_file);
  dim = data->at(0).size();
  max_elements = data->size();

  std::shared_ptr<hnswlib::SpaceInterface<float>> sp;
  if (dist == "l2") {
    sp = std::make_shared<hnswlib::L2Space>(dim);
  } else if (dist == "ip") {
    sp = std::make_shared<hnswlib::InnerProductSpace>(dim);
  } else {
    std::cout << "Unknown distance type: " << dist << std::endl;
    return -1;
  }

  hnswlib::HierarchicalNSW<float> *hnsw = new hnswlib::HierarchicalNSW<float>(
      sp.get(), max_elements, M, ef_construction);

  omp_set_num_threads(num_threads);
#pragma omp parallel for
  for (int i = 0; i < max_elements; ++i) {
    hnsw->addPoint(data->at(i).data(), i);
  }

  hnsw->saveIndex(index_save_path);
  delete hnsw;
  std::cout << "Index saved to " << index_save_path << std::endl;
  return 0;
}