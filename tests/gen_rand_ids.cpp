#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <total_points> <sample_size> <result_path>" << std::endl;
    return 1;
  }
  int total_points = std::stoi(argv[1]);
  int sample_size = std::stoi(argv[2]);
  std::string result_path = argv[3];
  auto out = std::ofstream(result_path);
  int seed = 42;

  // Create a vector with all indices
  std::vector<int> indices(total_points);
  for (int i = 0; i < total_points; ++i) {
    indices[i] = i;
  }
  std::vector<int> sampled_indices;

  // Shuffle the indices with a fixed seed
  std::mt19937 gen(
      seed);  // Initialize the random number generator with a fixed seed
  std::shuffle(indices.begin(), indices.end(), gen);

  sampled_indices =
      std::vector<int>(indices.begin(), indices.begin() + sample_size);
  std::sort(sampled_indices.begin(), sampled_indices.end());

  for (int i = 0; i < sample_size; ++i) {
    out << sampled_indices[i] << "\n";
  }
  out.close();
  return 0;
}
