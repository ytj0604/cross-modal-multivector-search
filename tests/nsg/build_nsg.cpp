#include <efanna2e-graph/index_graph.h>
#include <efanna2e-graph/index_random.h>
#include <efanna2e-nsg/index_nsg.h>
#include <efanna2e/util.h>
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
  std::string base_data_file;
  int knn_K;
  int knn_L;
  int knn_iter;
  int knn_S;
  int knn_R;
  int nsg_L;
  int nsg_R;
  int nsg_C;
  std::string index_save_path;
  std::string dist;

  po::options_description desc("Allowed options");
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("base_data_path",
                       po::value<std::string>(&base_data_file)->required(),
                       "Input data file in bin format");
    desc.add_options()("knn_K", po::value<int>(&knn_K)->required());
    desc.add_options()("knn_L", po::value<int>(&knn_L)->required());
    desc.add_options()("knn_iter", po::value<int>(&knn_iter)->required());
    desc.add_options()("knn_S", po::value<int>(&knn_S)->required());
    desc.add_options()("knn_R", po::value<int>(&knn_R)->required());
    desc.add_options()("nsg_L", po::value<int>(&nsg_L)->required());
    desc.add_options()("nsg_R", po::value<int>(&nsg_R)->required());
    desc.add_options()("nsg_C", po::value<int>(&nsg_C)->required());
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
  uint32_t dim = data->at(0).size();
  uint32_t max_elements = data->size();
  float *data_load = new float[max_elements * dim];
  for (size_t i = 0; i < max_elements; i++) {
    for (size_t j = 0; j < dim; j++) {
      data_load[i * dim + j] = data->at(i)[j];
    }
  }
  efanna2e_graph::Metric dist_metric_knn;
  efanna2e_nsg::Metric dist_metric_nsg;
  if (dist == "l2") {
    dist_metric_knn = efanna2e_graph::L2;
    dist_metric_nsg = efanna2e_nsg::L2;
  } else if (dist == "ip") {
    dist_metric_knn = efanna2e_graph::INNER_PRODUCT;
    dist_metric_nsg = efanna2e_nsg::INNER_PRODUCT;
  } else {
    std::cout << "Unknown distance type: " << dist << std::endl;
    return -1;
  }
  efanna2e_graph::IndexRandom init_index(dim, max_elements);
  efanna2e_graph::IndexGraph index(dim, max_elements, dist_metric_knn,
                                   (efanna2e_graph::Index *)(&init_index));
  efanna2e_graph::Parameters params_knn;
  params_knn.Set<unsigned>("K", knn_K);
  params_knn.Set<unsigned>("L", knn_L);
  params_knn.Set<unsigned>("iter", knn_iter);
  params_knn.Set<unsigned>("S", knn_S);
  params_knn.Set<unsigned>("R", knn_R);

  index.Build(max_elements, data_load, params_knn);
  auto idx = index.GetFinalGraph();

  efanna2e_nsg::IndexNSG nsg_index(dim, max_elements, dist_metric_nsg, nullptr);
  efanna2e_nsg::Parameters params_nsg;
  params_nsg.Set<unsigned>("L", nsg_L);
  params_nsg.Set<unsigned>("R", nsg_R);
  params_nsg.Set<unsigned>("C", nsg_C);
  params_nsg.Set<std::string>("nn_graph_path", index_save_path);

  nsg_index.SetInitialGraph(idx);
  nsg_index.BuildLoadedGraph(max_elements, data_load, params_nsg);
  nsg_index.Save(index_save_path.c_str());
}