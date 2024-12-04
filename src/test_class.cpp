#include <test_class.h>

#include <mutex>
#include <thread>

void TestClass::TestKNNSignificance(GroundTruthType sgt,
                                    VectorGroundTruthType vgt_,
                                    uint32_t multivector_cardinality,
                                    std::string output, uint32_t k,
                                    uint32_t num_raw_vector_results) {
  auto VectorIDToVectorSetID = [&](VectorID vid) -> VectorSetID {
    return vid / multivector_cardinality;
  };
  auto weighted_sum = [&](const std::vector<uint32_t> &vec) -> double {
    double sum = 0.0;
    double total_weight = 0.0;
    for (size_t i = 0; i < vec.size(); i++) {
      double weight = static_cast<double>(vec.size() - i) / vec.size();
      sum += vec[i] * weight;
      total_weight += weight;
    }
    return sum / total_weight;  // Normalize by total weight
  };

  auto &vgt = *vgt_;
  std::vector<uint32_t> agg_count(vgt[0].size(), 0);
  std::vector<std::vector<uint32_t>> count_per_vector(
      vgt.size(), std::vector<uint32_t>(vgt[0].size(), 0));
  std::mutex mutex;  // Mutex for thread-safe access to shared data

  // Function to process a chunk of `vgt`
  auto process_chunk = [&](size_t start, size_t end) {
    for (size_t vectorid = start; vectorid < end; vectorid++) {
      if (vectorid % 1000 == 0) {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << "[Progress] Processed " << vectorid << " / " << vgt.size()
                  << " vectors.\n";
      }

      auto current_vector_set_id = VectorIDToVectorSetID(vectorid);
      auto &current_sgt = (*sgt)[current_vector_set_id];
      auto top_10_end =
          current_sgt.begin() + std::min<size_t>(k, current_sgt.size());

      for (size_t j = 0; j < vgt[vectorid].size(); j++) {
        auto KNN_ID = vgt[vectorid][j];
        auto KNN_set_id = VectorIDToVectorSetID(KNN_ID);
        auto it = std::find(current_sgt.begin(), top_10_end, KNN_set_id);
        if (it != top_10_end) {
          std::lock_guard<std::mutex> lock(mutex);
          agg_count[j]++;
          count_per_vector[vectorid][j]++;
        }
      }
    }
  };

  // Determine the number of threads
  size_t num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4;  // Default to 4 threads if hardware info is unavailable

  size_t chunk_size = vgt.size() / num_threads;
  std::vector<std::thread> threads;

  // Launch threads
  for (size_t t = 0; t < num_threads; t++) {
    size_t start = t * chunk_size;
    size_t end = (t == num_threads - 1) ? vgt.size() : (t + 1) * chunk_size;
    threads.emplace_back(process_chunk, start, end);
  }

  // Join threads
  for (auto &thread : threads) {
    thread.join();
  }

  // Write results to file
  std::ofstream file(output);
  if (!file.is_open()) {
    std::cerr << "[Error] Unable to open output file: " << output << "\n";
    return;
  }

  // Write aggregated counts
  for (size_t i = 0; i < agg_count.size(); i++) {
    file << agg_count[i] << ",";
  }

  // Write counts per vector
  for (size_t i = 0; i < multivector_cardinality * num_raw_vector_results;
       i++) {
    file << "\n";
    for (size_t j = 0; j < count_per_vector[i].size(); j++) {
      file << count_per_vector[i][j] << ",";
    }
  }

  file.close();
  std::cout << "[Info] Results written to " << output << "\n";
}
