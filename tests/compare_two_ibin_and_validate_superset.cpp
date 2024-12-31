#include <multivector_reranker.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <query file path> <data file path>"
              << std::endl;
    return 1;
  }
  std::string query_file_path = argv[1];
  std::string data_file_path = argv[2];

  auto query_vectors_ =
      Loader::LoadEmbeddingVectorAsFloatVector(query_file_path);
  auto query_vectors = query_vectors_.get();
  auto data_vectors_ = Loader::LoadEmbeddingVectorAsFloatVector(data_file_path);
  auto data_vectors = data_vectors_.get();

  // For each query vector, do brute-force scan to find the same vector from
  // data vectors.
  int count = 0;
  for (const auto& query_vector : *query_vectors) {
    bool found = false;
    for (const auto& data_vector : *data_vectors) {
      if (query_vector == data_vector) {
        found = true;
        break;
      }
    }
    if (!found) {
      std::cerr << "Error: Query vector not found in data vectors."
                << std::endl;
      return 1;
    }
    if (found) {
      count++;
      if (count % 100 == 0) {
        std::cout << "Found " << count << " query vectors in data vectors."
                  << std::endl;
      }
    }
  }
}