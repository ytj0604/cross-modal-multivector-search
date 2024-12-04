#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <multivector_reranker.h>

#include <iostream>

// A class that contains functions for pure test purposes; not a part of main
// logic.
class TestClass {
 public:
  static void TestKNNSignificance(GroundTruthType sgt,
                                  VectorGroundTruthType vgt,
                                  uint32_t multivector_cardinality,
                                  std::string output, uint32_t k, uint32_t num_raw_vector_results);
};
#endif  // TEST_CLASS_H