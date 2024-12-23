#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "multivector_reranker.h"
namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string query_file;
  std::string base_data_file;
  std::string evaluation_save_path = "";
  std::string evaluation_save_prefix = "";
  std::string set_gt_path = "";
  uint32_t query_multivector_size;
  uint32_t k;
  std::string dist;
  // uint32_t num_samples;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
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
    desc.add_options()("k", po::value<uint32_t>(&k)->default_value(10),
                       "k nearest neighbors");
    desc.add_options()("dist", po::value<std::string>(&dist)->required(),
                       "Distance function <l2/ip>");
    desc.add_options()("set_gt_path",
                       po::value<std::string>(&set_gt_path)->required(),
                       "Set ground truth path");
    // desc.add_options()("num_samples",
    //                    po::value<uint32_t>(&num_samples)->required(),
    //                    "Dimension of the vectors");
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
  bool query_is_image;
  if (query_file.find("img") != std::string::npos) {
    query_is_image = true;
  } else {
    query_is_image = false;
  }

  auto query_matrix = Loader::LoadEmbeddingVector(query_file);
  auto data_matrix = Loader::LoadEmbeddingVector(base_data_file);
  uint32_t num_query_sets = query_matrix.rows() / query_multivector_size;

  MultiVectorReranker reranker;
  reranker.SetDataVector(data_matrix);
  reranker.SetQueryVector(query_matrix);
  reranker.SetQueryMultiVectorCardinality(query_multivector_size);
  reranker.SetK(k);
  reranker.SetUseGPU(true);
  // reranker.SetUseGPU(false);
  reranker.SetGPUBatchSize(10000);
  reranker.SetDistanceMetric(
      "smooth_chamfer",
      (dist == "cosine" || dist == "ip") ? "cosine_gpu" : "l2");
  reranker.SetVectorID2VectorSetIDMapping(
      [&query_multivector_size](VectorID vid) -> VectorSetID {
        return vid / query_multivector_size;
      });
  RecallCalculator recall_calculator;
  recall_calculator.SetPairedGroundTruth(
      [&query_multivector_size, &query_is_image](VectorSetID vsid) {
        if (query_is_image) {
          return std::make_pair(vsid * 5, 5);
        } else {
          return std::make_pair(vsid / 5, 1);
        }
      });
  recall_calculator.SetK(k);

  std::string evaluation_file_path = evaluation_save_prefix + "_" + ".tsv";
  std::ofstream tsv_out(evaluation_file_path, std::ios::out);
  if (!tsv_out.is_open()) {
    std::cerr << "Error opening output file: " << evaluation_file_path
              << std::endl;
    return -1;
  }
  double total_search_time = 0;
  double total_recall = 0;

  std::ofstream gt_out(set_gt_path, std::ios::binary);
  gt_out.write(reinterpret_cast<const char *>(&num_query_sets),
               sizeof(uint32_t));
  gt_out.write(reinterpret_cast<const char *>(&k), sizeof(uint32_t));
  // std::random_device rd;   // Non-deterministic generator
  // std::mt19937 gen(rd());  // Mersenne Twister engine seeded with rd()
  // std::uniform_int_distribution<> distr(0, num_query_sets - 1);

  // num_samples = num_query_sets;

  auto start_real_time = std::chrono::high_resolution_clock::now();

  std::atomic<uint32_t> progress(0);

  // This is only for printing progress.
  std::thread progress_thread([&progress, &num_query_sets, &start_real_time]() {
    while (true) {
      auto current = progress.load();
      if (current > 0) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                                current_time - start_real_time)
                                .count();

        double rate = static_cast<double>(current) / elapsed_time;
        uint32_t remaining_queries = num_query_sets - current;
        uint32_t remaining_seconds =
            static_cast<uint32_t>(remaining_queries / rate);
        uint32_t minutes = remaining_seconds / 60;
        uint32_t seconds = remaining_seconds % 60;
        std::cout << "Progress: " << current << "/" << num_query_sets
                  << " \t Remaining: " << minutes << ":" << std::setfill('0')
                  << std::setw(2) << seconds << std::endl;
      } else {
        std::cout << "Progress: " << current << "/" << num_query_sets
                  << std::endl;
      }

      if (current >= num_query_sets) {
        break;
      }

      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  });

  // To store the groundtruth info, we first store them in pre-allocated
  // vectors.
  std::vector<std::vector<VectorSetID>> ground_truth(num_query_sets);
  for (uint32_t i = 0; i < num_query_sets; i++) {
    ground_truth[i].reserve(k);
  }

  // Set omp thread
  omp_set_num_threads(80);
#pragma omp parallel for
  for (uint32_t i = 0; i < num_query_sets; ++i) {
    // printf("i: %d\n", i);
    // std::vector<VectorSetID> reranked_indices;
    // #pragma omp critical
    // rand_index = static_cast<uint32_t>(distr(gen));
    auto start = std::chrono::high_resolution_clock::now();
    reranker.RerankAllBySequentialScan(i, ground_truth[i]);
    auto end = std::chrono::high_resolution_clock::now();
    auto search_time_seconds =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count()) /
        1e6;
    double recall = recall_calculator.ComputePairedRecall(i, ground_truth[i]);
#pragma omp critical
    {
      total_search_time += search_time_seconds;
      total_recall += recall;
      tsv_out << i << "\t" << search_time_seconds << "\n";
      tsv_out << recall << "\n";
      for (auto in = 0; in < ground_truth[i].size(); in++) {
        tsv_out << ground_truth[i][in] << "\t";
      }
      tsv_out << "\n";
    }
    // Progress print
    auto prog = progress.fetch_add(1);
  }

  auto end_real_time = std::chrono::high_resolution_clock::now();
  tsv_out.close();

  progress_thread.join();

  double recall = total_recall / num_query_sets;
  double QPS = num_query_sets / total_search_time;
  std::ofstream evaluation_out(evaluation_save_path, std::ios::app);
  if (!evaluation_out.is_open()) {
    std::cerr << "Error: Unable to open or create the file at "
              << evaluation_save_path << std::endl;
    return -1;
  }

  std::cout << "Now writing to file: " << evaluation_save_path << std::endl;
  for (auto i = 0; i < num_query_sets; i++) {
    gt_out.write(reinterpret_cast<const char *>(ground_truth[i].data()),
                 k * sizeof(VectorSetID));
  }

  auto total_real_search_time_seconds =
      static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                              end_real_time - start_real_time)
                              .count()) /
      1e6;

  evaluation_out << recall << "\t" << QPS << "\t"
                 << total_real_search_time_seconds << "\n";
  if (evaluation_out.is_open()) {
    evaluation_out.close();
  }

  return 0;
}