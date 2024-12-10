#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <multivector_reranker.h>

#include <iostream>

// A class that contains functions for pure test purposes; not a part of main
// logic.
class UtilityMethods {
 public:
  static void TestKNNSignificance(GroundTruthType sgt,
                                  VectorGroundTruthType vgt,
                                  uint32_t multivector_cardinality,
                                  std::string output, uint32_t k,
                                  uint32_t num_raw_vector_results);
  static void GenerateRandomVectorsAndStore(std::string output,
                                            uint32_t num_vectors,
                                            uint32_t vector_dimension);
  static void TestCosineSimilarityDist(MatrixType query_vector,
                                       MatrixType data_vector,
                                       std::string output_file_path);

 private:
  static std::vector<float> generate_normalized_vector(int d);
};
#endif  // TEST_CLASS_H