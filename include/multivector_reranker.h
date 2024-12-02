#ifndef MULTI_VECTOR_RERANKER_H
#define MULTI_VECTOR_RERANKER_H
#define EIGEN_USE_BLAS

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <vector>

using VectorSetID = unsigned int;
using VectorID = unsigned int;
using MatrixType =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class MultiVectorReranker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void SetVectorID2VectorSetIDMapping(std::function<VectorSetID(VectorID)> f);
  void SetMultiVectorCardinality(uint32_t cardinality);
  void SetDistanceMetric(const std::string& set_to_set_metric,
                         const std::string& vector_dist_metric);
  void SetDataVector(const MatrixType& data_matrix);
  void SetQueryVector(const MatrixType& query_matrix);
  void SetK(uint32_t k) { this->k = k; }
  void Rerank(VectorSetID& query_id,
              const std::vector<std::vector<VectorID>>& indices,
              std::vector<VectorSetID>& reranked_indices);

 private:
  void computeCosineSimilarity(const Eigen::Ref<const MatrixType>&,
                               const Eigen::Ref<const MatrixType>&,
                               Eigen::Ref<MatrixType>);
  float computeSmoothChamferDistance(
      const Eigen::Ref<const MatrixType>& img_embs,
      const Eigen::Ref<const MatrixType>& txt_embs);
  uint32_t multi_vector_cardinality;
  std::function<VectorSetID(VectorID)> vector_id_to_vector_set_id;
  MatrixType data_matrix;
  MatrixType query_matrix;
  std::function<float(const Eigen::Ref<const MatrixType>&,
                      const Eigen::Ref<const MatrixType>&)>
      set_to_set_distance_metric;
  std::function<void(const Eigen::Ref<const MatrixType>&,
                     const Eigen::Ref<const MatrixType>&,
                     Eigen::Ref<MatrixType>)>
      vector_distance_metric;
  // Smooth-Chamfer distance parameters
  float temperature = 16.0f;
  float temperature_txt_scale = 1.0;
  float denominator = 2;
  uint32_t k;
};

// class RecallCalculator {
//  public:
//   void SetGroundTruth(std::vector<float> gt, std::vector<uint32_t> indices);
//   float ComputeRecall(uint32_t query_id, std::vector<VectorSetID> res);

//  private:
//   std::vector<float> gt;
// };

class Loader {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static MatrixType LoadEmbeddingVector(const std::string& file_path);
};
#endif  // MULTI_VECTOR_RERANKER_H