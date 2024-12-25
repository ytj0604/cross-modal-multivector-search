#include <multivector_reranker.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <sampled id path> <multivector card> <original ibin file "
                 "path> <sampled ibin path> <remaining ibin path>"
              << std::endl;
    return 1;
  }

  std::string sampled_id_path = argv[1];
  int multivector_card = std::stoi(argv[2]);
  std::string original_ibin_path = argv[3];
  std::string sampled_ibin_path = argv[4];
  std::string remaining_ibin_path = argv[5];

  auto vectors_ = Loader::LoadEmbeddingVectorAsFloatVector(original_ibin_path);
  auto vectors = vectors_.get();

  if (vectors->empty()) {
    std::cerr << "Error: The input ibin file is empty." << std::endl;
    return 1;
  }

  std::ifstream sampled_id_file(sampled_id_path);
  if (!sampled_id_file.is_open()) {
    std::cerr << "Error: Could not open sampled id file: " << sampled_id_path
              << std::endl;
    return 1;
  }

  std::unordered_set<int> sampled_ids;
  int max_id = -1;  // Initialize to a valid minimum
  int id;
  while (sampled_id_file >> id) {
    sampled_ids.insert(id);
    max_id = std::max(max_id, id);
  }
  sampled_id_file.close();

  if (vectors->size() / multivector_card <= max_id) {
    std::cerr
        << "Error: sampled id exceeds the number of vectors in the ibin file"
        << std::endl;
    return 1;
  }

  uint32_t num_samples = sampled_ids.size() * multivector_card;
  uint32_t num_remainings = vectors->size() - num_samples;
  uint32_t dim = vectors->at(0).size();

  // Open binary files
  std::ofstream sampled_ibin(sampled_ibin_path, std::ios::binary);
  std::ofstream remaining_ibin(remaining_ibin_path, std::ios::binary);
  if (!sampled_ibin.is_open() || !remaining_ibin.is_open()) {
    std::cerr << "Error: Could not open output files." << std::endl;
    return 1;
  }

  // Write headers
  sampled_ibin.write(reinterpret_cast<const char*>(&num_samples),
                     sizeof(uint32_t));
  sampled_ibin.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
  remaining_ibin.write(reinterpret_cast<const char*>(&num_remainings),
                       sizeof(uint32_t));
  remaining_ibin.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));

  // Write vectors
  for (int vector_set_id = 0;
       vector_set_id < vectors->size() / multivector_card; vector_set_id++) {
    bool is_sampled = sampled_ids.find(vector_set_id) != sampled_ids.end();
    std::ofstream& target_file = is_sampled ? sampled_ibin : remaining_ibin;

    for (int i = 0; i < multivector_card; i++) {
      const auto& vector = vectors->at(vector_set_id * multivector_card + i);
      target_file.write(reinterpret_cast<const char*>(vector.data()),
                        vector.size() * sizeof(float));
    }
  }

  sampled_ibin.close();
  remaining_ibin.close();

  std::cout << "Processing completed successfully." << std::endl;
  return 0;
}
