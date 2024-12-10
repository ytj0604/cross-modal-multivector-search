#include <utility_methods.h>

#include <mutex>
#include <random>
#include <thread>
#include <condition_variable>

void UtilityMethods::TestKNNSignificance(GroundTruthType sgt,
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
void UtilityMethods::GenerateRandomVectorsAndStore(std::string output,
                                                   uint32_t num_vectors,
                                                   uint32_t vector_dimension) {
  std::ofstream file(output, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[Error] Unable to open output file: " << output << "\n";
    return;
  }
  file.write(reinterpret_cast<const char *>(&num_vectors), sizeof(uint32_t));
  file.write(reinterpret_cast<const char *>(&vector_dimension),
             sizeof(uint32_t));

  for (uint32_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vector = generate_normalized_vector(vector_dimension);
    file.write(reinterpret_cast<const char *>(vector.data()),
               vector_dimension * sizeof(float));
  }

  file.close();
  std::cout << "[Info] Random vectors written to " << output << "\n";
}

std::vector<float> UtilityMethods::generate_normalized_vector(int d) {
  if (d <= 0) {
    throw std::invalid_argument("Dimension must be positive.");
  }

  // Create a vector to hold the random components
  std::vector<float> vec(d);

  // Random number generator with normal distribution
  std::random_device rd;   // Seed for randomness
  std::mt19937 gen(rd());  // Mersenne Twister random generator
  std::normal_distribution<> dis(
      0.0, 1.0);  // Normal distribution with mean 0 and stddev 1

  // Generate random components and compute their norm
  double norm = 0.0;
  for (int i = 0; i < d; ++i) {
    vec[i] = dis(gen);
    norm += vec[i] * vec[i];  // Sum of squares for norm calculation
  }

  // Normalize the vector
  norm = std::sqrt(norm);
  for (int i = 0; i < d; ++i) {
    vec[i] /= norm;
  }

  return vec;
}

// void UtilityMethods::TestCosineSimilarityDist(MatrixType query_vector,
//                                               MatrixType data_vector,
//                                               std::string output_file_path) {
//   const int num_query_to_test = 1;
//   const uint32_t num_division = 200;
//   const double total_angle = 2.0;  // Cosine similarity range: -1 to 1

//   std::ofstream file(output_file_path);
//   if (!file.is_open()) {
//     std::cerr << "[Error] Unable to open output file: " << output_file_path
//               << "\n";
//     return;
//   }
//   std::cout << "[Info] Computing cosine similarity distances using "
//             << std::thread::hardware_concurrency() << " threads...\n";

//   // Limit the number of query vectors
//   size_t num_query_vectors =
//       std::min((size_t)num_query_to_test, (size_t)query_vector.rows());

//   // Global histogram to store the results
//   std::vector<double> global_divisions(num_division, 0.0);
//   std::mutex histogram_mutex;
//   std::atomic<size_t> processed_queries(0);

//   auto process_chunk = [&](size_t start, size_t end) {
//     std::vector<double> local_divisions(num_division, 0.0);

//     for (size_t i = start; i < end; ++i) {
//       // Extract the current query vector
//       auto query = query_vector.row(i);

//       // Compute cosine similarity for the current query with all data vectors
//       auto cosine_chunk_ = query * data_vector.transpose();
//       auto cosine_chunk = cosine_chunk_.eval();

//       // Update the histogram for the current query
//       for (int j = 0; j < cosine_chunk.cols(); ++j) {
//         if (j % 1000 == 0) {
//           std::lock_guard<std::mutex> lock(histogram_mutex);
//           std::cout << "[Progress] Processed " << j << " / " << cosine_chunk.cols()
//                     << ".\n";
//         }
//         double cosine = cosine_chunk(0, j);  // Since `query` is a single row
//         int slot =
//             static_cast<int>((cosine + 1.0) / total_angle * num_division);
//         if (slot >= 0 && slot < num_division) {
//           local_divisions[slot]++;
//         }
//       }

//       // Update progress
//       size_t current_progress = ++processed_queries;
//       if (current_progress % 10 == 0) {  // Print progress every 10 queries
//         std::cout << "[Progress] Processed " << current_progress << " / "
//                   << num_query_vectors << " queries.\n";
//       }
//     }

//     // Merge local histogram into the global histogram
//     {
//       std::lock_guard<std::mutex> lock(histogram_mutex);
//       for (size_t i = 0; i < num_division; ++i) {
//         global_divisions[i] += local_divisions[i];
//       }
//     }
//   };

//   // Divide queries among threads
//   std::vector<std::thread> threads;
//   size_t num_threads = std::thread::hardware_concurrency();
//   size_t chunk_size = (num_query_vectors + num_threads - 1) / num_threads;

//   for (size_t t = 0; t < num_threads; ++t) {
//     size_t start = t * chunk_size;
//     size_t end = std::min(start + chunk_size, num_query_vectors);
//     if (start < end) {
//       threads.emplace_back(process_chunk, start, end);
//     }
//   }

//   // Wait for all threads to finish
//   for (auto &thread : threads) {
//     thread.join();
//   }

//   // Write results to the file
//   for (int i = 0; i < num_division; i++) {
//     file << global_divisions[i] << ",";
//   }
//   file.close();
//   std::cout << "[Info] Cosine similarity distances written to "
//             << output_file_path << "\n";
// }
void UtilityMethods::TestCosineSimilarityDist(MatrixType query_vector,
                                              MatrixType data_vector,
                                              std::string output_file_path) {
  const int num_query_to_test = 400;
  const uint32_t num_division = 200;
  const double total_angle = 2.0;  // Cosine similarity range: -1 to 1

  std::mutex file_mutex;
  std::condition_variable cv;
  std::atomic<size_t> next_chunk_to_write(0);

  std::ofstream file(output_file_path);
  if (!file.is_open()) {
    std::cerr << "[Error] Unable to open output file: " << output_file_path
              << "\n";
    return;
  }
  std::cout << "[Info] Computing cosine similarity distances using "
            << std::thread::hardware_concurrency() << " threads...\n";

  // Limit the number of query vectors
  size_t num_query_vectors =
      std::min((size_t)num_query_to_test, (size_t)query_vector.rows());

  auto process_chunk = [&](size_t chunk_index, size_t start, size_t end) {
    std::ostringstream local_output;
    for (size_t i = start; i < end; ++i) {
      // Extract the current query vector
      auto query = query_vector.row(i);

      // Compute cosine similarity for the current query with all data vectors
      auto cosine_chunk_ = query * data_vector.transpose();
      auto cosine_chunk = cosine_chunk_.eval();

      // Create a histogram for the current query
      std::vector<double> divisions(num_division, 0.0);

      // Populate the histogram
      for (int j = 0; j < cosine_chunk.cols(); ++j) {
        double cosine = cosine_chunk(0, j);  // Since `query` is a single row
        int slot =
            static_cast<int>((cosine + 1.0) / total_angle * num_division);
        if (slot >= 0 && slot < num_division) {
          divisions[slot]++;
        }
      }

      // Store the histogram in the local output stream
      for (size_t d = 0; d < divisions.size(); ++d) {
        local_output << divisions[d];
        if (d < divisions.size() - 1) {
          local_output << ",";
        }
      }
      local_output << "\n";  // End the line for the current query
    }

    // Write the local output to the global file in the correct order
    {
      std::unique_lock<std::mutex> lock(file_mutex);
      cv.wait(lock, [&] { return next_chunk_to_write == chunk_index; });
      file << local_output.str();
      ++next_chunk_to_write;
      cv.notify_all();
    }
  };

  // Divide queries among threads
  std::vector<std::thread> threads;
  size_t num_threads = std::thread::hardware_concurrency();
  size_t chunk_size = (num_query_vectors + num_threads - 1) / num_threads;

  for (size_t t = 0; t < num_threads; ++t) {
    size_t start = t * chunk_size;
    size_t end = std::min(start + chunk_size, num_query_vectors);
    if (start < end) {
      threads.emplace_back(process_chunk, t, start, end);
    }
  }

  // Wait for all threads to finish
  for (auto &thread : threads) {
    thread.join();
  }

  file.close();
  std::cout << "[Info] Cosine similarity distances written to "
            << output_file_path << "\n";
}
