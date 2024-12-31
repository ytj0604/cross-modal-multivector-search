#ifndef MULTI_VECTOR_RERANKER_H
#define MULTI_VECTOR_RERANKER_H
#define EIGEN_USE_BLAS

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

using VectorSetID = unsigned int;
using VectorID = unsigned int;
using Cardinality = unsigned int;
using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FloatVectorPtr = std::shared_ptr<std::vector<std::vector<float>>>;
using SetGroundTruthVectorPtr =
    std::shared_ptr<std::vector<std::vector<VectorSetID>>>;
using VectorGroundTruthVectorPtr =
    std::shared_ptr<std::vector<std::vector<VectorID>>>;
using VID2VSIDMapping = std::function<VectorSetID(VectorID)>;
using VSID2VIDMapping =
    std::function<std::pair<VectorID, Cardinality>(VectorSetID)>;
using GroundTruthMapping =
    std::function<std::pair<VectorSetID, Cardinality>(VectorSetID)>;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));                            \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

namespace std {
template <>
struct hash<std::pair<float*, unsigned int>> {
  size_t operator()(const std::pair<float*, unsigned int>& p) const {
    // Combine the hashes of the individual components
    return hash<float*>()(p.first) ^ (hash<unsigned int>()(p.second) << 1);
  }
};
}  // namespace std

class MultiVectorReranker {
 public:
  MultiVectorReranker();
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  void SetVectorID2VectorSetIDMapping(VID2VSIDMapping f);
  void SetVectorSetID2VectorIDMapping(VSID2VIDMapping f);
  void SetQueryMultiVectorCardinality(uint32_t cardinality);
  void SetDistanceMetric(const std::string& set_to_set_metric,
                         const std::string& vector_dist_metric);
  void SetDataVector(const Matrix& data_matrix);
  void SetQueryVector(const Matrix& query_matrix);
  void SetK(uint32_t k) { this->k = k; }
  template <typename IDType>
  void Rerank(VectorSetID& query_id,
              const std::vector<std::vector<IDType>>& indices,
              std::vector<VectorSetID>& reranked_indices);
  void RerankAllBySequentialScan(VectorSetID& query_id,
                                 std::vector<VectorSetID>& reranked_indices);
  void RankAllVectorsBySequentialScan(
      VectorID& query_vec_id, Cardinality query_batch_size,
      std::vector<std::vector<VectorID>>& top_k_indices);
  void RerankAllAndGenerateSetGroundTruth(const std::string& ground_truth_file);
  void GenerateVectorGroundTruth(const std::string& ground_truth_file);
  void SetGPUBatchSize(Cardinality batch_size);
  void SetUseGPU(bool use_gpu) { this->use_gpu = use_gpu; }
  void GetNNWiseDistance(std::vector<VectorID>& indices, std::vector<float>& avg_distances);
 private:
  void SetQueryOnGPU(const Eigen::Ref<const Matrix>& query);
  float* SetDataBatchOnGPU(const Eigen::Ref<const Matrix>& data);
  void AllocateResultBufferIfNeeded(int query_rows, int data_rows);
  void computeCosineSimilarity(const Eigen::Ref<const Matrix>&,
                               const Eigen::Ref<const Matrix>&,
                               Eigen::Ref<Matrix>);
  void computeCosineSimilarityGPU(const Eigen::Ref<const Matrix>&,
                                  const Eigen::Ref<const Matrix>&,
                                  Eigen::Ref<Matrix>);
  // For following functions, the "img_embs" and "txt_embs" are not correct.
  // They are actually query and data.
  float computeSmoothChamferDistance(const Eigen::Ref<const Matrix>& img_embs,
                                     const Eigen::Ref<const Matrix>& txt_embs);
  std::vector<float> ComputeSmoothChamferDistanceBatch(
      const Eigen::Ref<const Matrix>& img_embs,
      const Eigen::Ref<const Matrix>& txt_embs,
      std::vector<Cardinality>& cardinalities);
  float ComputeSummedMaxSimilarity(const Eigen::Ref<const Matrix>& img_embs,
                                   const Eigen::Ref<const Matrix>& txt_embs);
  uint32_t multi_vector_cardinality = 0;
  VID2VSIDMapping vector_id_to_vector_set_id = nullptr;
  VSID2VIDMapping vector_set_id_to_vector_id = nullptr;
  Matrix data_matrix;
  Matrix query_matrix;
  std::function<float(const Eigen::Ref<const Matrix>&,
                      const Eigen::Ref<const Matrix>&)>
      set_to_set_distance_metric = nullptr;
  std::function<void(const Eigen::Ref<const Matrix>&,
                     const Eigen::Ref<const Matrix>&, Eigen::Ref<Matrix>)>
      vector_distance_metric = nullptr;
  // This is used only when GPU is used.
  std::function<std::vector<float>(const Eigen::Ref<const Matrix>&,
                                   const Eigen::Ref<const Matrix>&,
                                   std::vector<Cardinality>&)>
      set_to_set_distance_metric_batch = nullptr;
  // Smooth-Chamfer distance parameters
  float temperature = 16.0f;
  float temperature_txt_scale = 1.0;
  float denominator = 2;
  uint32_t k = 0;

  // GPU related.
  bool use_gpu = false;

  // cuBLAS handle; thread safe.
  cublasHandle_t handle;
  Cardinality gpu_batch_size = 10000;

  // Device memory for query matrix
  thread_local static float* d_query;
  thread_local static int query_rows;

  // Device memory for batched data and result matrices
  //   float* d_data_batch = nullptr;
  thread_local static float* d_result_batch;

  // Preallocated sizes
  //   int max_batch_size = -1;
  thread_local static int max_result_rows;
  thread_local static int max_result_cols;

  // Instead of unloading and loading data to GPU for each query, we can keep
  // the data in GPU memory.
  // Eviction is TODO.
  // Currently assume that data is not too large.
  std::mutex map_mutex;
  std::unordered_map<std::pair<float*, Cardinality>, float*> d_data_batch_map;
  size_t allocated_GPU_memory = 0;
};

class RecallCalculator {
 public:
  void SetK(uint32_t k) { this->k = k; }
  // Set level gt.
  void SetGroundTruth(SetGroundTruthVectorPtr ground_truth);
  void SetPairedGroundTruth(GroundTruthMapping f) { paired_ground_truth = f; }
  template <typename IDType>
  double ComputeRecall(VectorSetID query_id,
                       const std::vector<IDType>& reranked_indices);
  double ComputePairedRecall(VectorSetID query_id,
                             const std::vector<VectorSetID>& reranked_indices);

 private:
  uint32_t k = 0;
  SetGroundTruthVectorPtr ground_truth = nullptr;
  GroundTruthMapping paired_ground_truth = nullptr;
};

class Loader {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static Matrix LoadEmbeddingVector(const std::string& file_path);
  static FloatVectorPtr LoadEmbeddingVectorAsFloatVector(
      const std::string& file_path);
  // Loads a set-level ground truth file
  static SetGroundTruthVectorPtr LoadGroundTruth(const std::string& file_path);
  // Loads a vector-level ground truth file (generated by the RoarGraph using
  // DiskANN's code)
  static VectorGroundTruthVectorPtr LoadVectorGroundTruth(
      const std::string& file_path);
  // Should be used for re-ranking.
  static std::pair<VID2VSIDMapping, VSID2VIDMapping>
  LoadVectorCardinalityMappingAndGetBothMappings(std::string& file_path);
  static GroundTruthMapping LoadQueryDataPairMappingAsFunction(
      std::string& file_path);
};
#endif  // MULTI_VECTOR_RERANKER_H