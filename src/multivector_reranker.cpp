#include <multivector_reranker.h>

#include <unordered_map>

MultiVectorReranker::MultiVectorReranker() { cublasCreate(&handle); }

void MultiVectorReranker::SetVectorID2VectorSetIDMapping(VID2VSIDMapping f) {
  vector_id_to_vector_set_id = f;
}

void MultiVectorReranker::SetVectorSetID2VectorIDMapping(VSID2VIDMapping f) {
  vector_set_id_to_vector_id = f;
}

void MultiVectorReranker::SetQueryMultiVectorCardinality(uint32_t cardinality) {
  this->multi_vector_cardinality = cardinality;
}

void MultiVectorReranker::SetDataVector(const Matrix& data_matrix) {
  this->data_matrix = data_matrix;
}

void MultiVectorReranker::SetQueryVector(const Matrix& query_matrix) {
  this->query_matrix = query_matrix;
}

void MultiVectorReranker::SetGPUBatchSize(Cardinality batch_size) {
  gpu_batch_size = batch_size;
}

template <typename IDType>
void MultiVectorReranker::Rerank(
    VectorSetID& query_id, const std::vector<std::vector<IDType>>& indices,
    std::vector<VectorSetID>& reranked_indices) {
  std::vector<VectorSetID> deduplicated_indices;
  deduplicated_indices.reserve(indices.size());
  for (const auto& vector : indices) {
    for (VectorID vector_id : vector)
      deduplicated_indices.push_back(vector_id_to_vector_set_id(vector_id));
  }
  std::sort(deduplicated_indices.begin(), deduplicated_indices.end());
  deduplicated_indices.erase(
      std::unique(deduplicated_indices.begin(), deduplicated_indices.end()),
      deduplicated_indices.end());

  // Handle edge case: No valid vector set IDs
  if (deduplicated_indices.empty()) {
    reranked_indices.clear();
    return;
  }

  Eigen::Map<const Matrix> query_set(
      query_matrix.data() +
          query_id * multi_vector_cardinality * query_matrix.cols(),
      multi_vector_cardinality, query_matrix.cols());

  std::vector<std::pair<float, VectorSetID>> relevance_scores;
  relevance_scores.reserve(deduplicated_indices.size());

  for (VectorSetID data_set_id : deduplicated_indices) {
    if (vector_set_id_to_vector_id) {
      auto [vector_id, cardinality] = vector_set_id_to_vector_id(data_set_id);
      Eigen::Map<const Matrix> data_set(
          data_matrix.data() + vector_id * data_matrix.cols(), cardinality,
          data_matrix.cols());
      float relevance = set_to_set_distance_metric(query_set, data_set);
      relevance_scores.emplace_back(relevance, data_set_id);
    } else {
      Eigen::Map<const Matrix> data_set(
          data_matrix.data() +
              data_set_id * multi_vector_cardinality * data_matrix.cols(),
          multi_vector_cardinality, data_matrix.cols());
      float relevance = set_to_set_distance_metric(query_set, data_set);
      relevance_scores.emplace_back(relevance, data_set_id);
    }
  }

  size_t top_k =
      std::min(this->k, static_cast<uint32_t>(relevance_scores.size()));

  std::partial_sort(relevance_scores.begin(), relevance_scores.begin() + top_k,
                    relevance_scores.end(), [](const auto& a, const auto& b) {
                      return a.first > b.first;  // Higher relevance first
                    });

  reranked_indices.clear();
  reranked_indices.reserve(top_k);
  for (size_t i = 0; i < top_k; ++i) {
    reranked_indices.push_back(relevance_scores[i].second);
  }
}

void MultiVectorReranker::RerankAllBySequentialScan(
    VectorSetID& query_id, std::vector<VectorSetID>& reranked_indices) {
  Eigen::Map<const Matrix> query_set(
      query_matrix.data() +
          query_id * multi_vector_cardinality * query_matrix.cols(),
      multi_vector_cardinality, query_matrix.cols());

  std::vector<std::pair<float, VectorSetID>> relevance_scores;
  relevance_scores.reserve(data_matrix.rows() / multi_vector_cardinality);

  if (use_gpu) {
    SetQueryOnGPU(query_set);
    for (VectorSetID data_set_id = 0;
         data_set_id < data_matrix.rows() / multi_vector_cardinality;
         data_set_id += gpu_batch_size) {
      Cardinality batch_size = std::min(
          gpu_batch_size, static_cast<Cardinality>(data_matrix.rows()) /
                                  multi_vector_cardinality -
                              data_set_id);
      Eigen::Map<const Matrix> data_batch(
          data_matrix.data() +
              data_set_id * multi_vector_cardinality * data_matrix.cols(),
          batch_size * multi_vector_cardinality, data_matrix.cols());
      // Following should be changed if data multivector cardinality is not
      // fixed.
      std::vector<Cardinality> cardinalities(batch_size,
                                             multi_vector_cardinality);
      auto relevance_batch = set_to_set_distance_metric_batch(
          query_set, data_batch, cardinalities);
      for (size_t i = 0; i < batch_size; ++i) {
        relevance_scores.emplace_back(relevance_batch[i], data_set_id + i);
      }
    }
  } else {
    for (VectorSetID data_set_id = 0;
         data_set_id < data_matrix.rows() / multi_vector_cardinality;
         ++data_set_id) {
      Eigen::Map<const Matrix> data_set(
          data_matrix.data() +
              data_set_id * multi_vector_cardinality * data_matrix.cols(),
          multi_vector_cardinality, data_matrix.cols());
      float relevance = set_to_set_distance_metric(query_set, data_set);
      relevance_scores.emplace_back(relevance, data_set_id);
    }
  }
  size_t top_k =
      std::min(this->k, static_cast<uint32_t>(relevance_scores.size()));

  std::partial_sort(relevance_scores.begin(), relevance_scores.begin() + top_k,
                    relevance_scores.end(), [](const auto& a, const auto& b) {
                      return a.first > b.first;  // Higher relevance first
                    });

  reranked_indices.clear();
  reranked_indices.reserve(top_k);
  for (size_t i = 0; i < top_k; ++i) {
    reranked_indices.push_back(relevance_scores[i].second);
  }
}

void MultiVectorReranker::RerankAllAndGenerateSetGroundTruth(
    const std::string& ground_truth_file) {
  if (multi_vector_cardinality == 0) {
    throw std::runtime_error("Multi-vector cardinality not set.");
  }
  std::ofstream out(ground_truth_file, std::ios::binary);
  uint32_t num_queries = query_matrix.rows() / multi_vector_cardinality;
  uint32_t num_gt_per_query = data_matrix.rows() / multi_vector_cardinality;
  out.write(reinterpret_cast<const char*>(&num_queries), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&num_gt_per_query), sizeof(uint32_t));
  if (!out.is_open()) {
    throw std::runtime_error("Cannot open file: " + ground_truth_file);
  }
  this->k = num_gt_per_query;
  std::vector<VectorSetID> reranked_indices;
  for (uint32_t i = 0; i < num_queries; ++i) {
    RerankAllBySequentialScan(i, reranked_indices);
    out.write(reinterpret_cast<const char*>(reranked_indices.data()),
              num_gt_per_query * sizeof(VectorSetID));
  }
  out.close();
}

void MultiVectorReranker::SetQueryOnGPU(const Eigen::Ref<const Matrix>& query) {
  if (query.rows() != query_rows) {
    if (d_query) cudaFree(d_query);
    query_rows = query.rows();
    cudaMalloc((void**)&d_query, query_rows * query.cols() * sizeof(float));
  }
  cudaMemcpy(d_query, query.data(), query_rows * query.cols() * sizeof(float),
             cudaMemcpyHostToDevice);
}

float* MultiVectorReranker::SetDataBatchOnGPU(
    const Eigen::Ref<const Matrix>& data) {
  std::pair<float*, Cardinality> key = std::make_pair(
      const_cast<float*>(data.data()), static_cast<Cardinality>(data.rows()));
  if (d_data_batch_map.find(key) == d_data_batch_map.end()) {
    float* d_data_batch = nullptr;
    cudaMalloc((void**)&d_data_batch,
               data.rows() * data.cols() * sizeof(float));
    cudaMemcpy(d_data_batch, data.data(),
               data.rows() * data.cols() * sizeof(float),
               cudaMemcpyHostToDevice);
    d_data_batch_map[key] = d_data_batch;
    allocated_GPU_memory += data.rows() * data.cols() * sizeof(float);
  }
  return d_data_batch_map[key];
}

void MultiVectorReranker::AllocateResultBufferIfNeeded(int query_rows,
                                                       int data_rows) {
  if (query_rows > max_result_rows || data_rows > max_result_cols) {
    // Free existing memory if it exists
    if (d_result_batch) cudaFree(d_result_batch);

    // Update maximum dimensions
    max_result_rows = query_rows;
    max_result_cols = data_rows;

    // Allocate new GPU memory for the result matrix
    cudaMalloc((void**)&d_result_batch,
               max_result_rows * max_result_cols * sizeof(float));
  }
}

void MultiVectorReranker::computeCosineSimilarity(
    const Eigen::Ref<const Matrix>& X, const Eigen::Ref<const Matrix>& Y,
    Eigen::Ref<Matrix> result) {
  result = X * Y.transpose();
}

void MultiVectorReranker::computeCosineSimilarityGPU(
    const Eigen::Ref<const Matrix>& X, const Eigen::Ref<const Matrix>& Y,
    Eigen::Ref<Matrix> result) {
  // The query should be already set!
  // The caller should explicitly invoke SetQueryOnGPU() before calling this
  // function. It is to avoid unnecessary memory transfers.
  // SetQueryOnGPU(X);
  auto data_mem = SetDataBatchOnGPU(Y);
  AllocateResultBufferIfNeeded(X.rows(), Y.rows());
  // Scalars for cuBLAS
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Y.cols(), X.rows(), X.cols(),
              &alpha, data_mem, Y.cols(), d_query, X.cols(), &beta,
              d_result_batch, Y.cols());

  cudaMemcpy(result.data(), d_result_batch, X.rows() * Y.rows() * sizeof(float),
             cudaMemcpyDeviceToHost);
}

float MultiVectorReranker::computeSmoothChamferDistance(
    const Eigen::Ref<const Matrix>& img_embs,
    const Eigen::Ref<const Matrix>& txt_embs) {
  // Compute cosine similarity
  Matrix dist(img_embs.rows(), txt_embs.rows());
  vector_distance_metric(img_embs, txt_embs, dist);

  // Compute temp_dist1 and temp_dist2
  Matrix temp_dist1 = temperature * temperature_txt_scale * dist;
  Matrix temp_dist2 = temperature * dist;

  // Compute row-wise max for temp_dist1
  Eigen::VectorXf row_max = temp_dist1.rowwise().maxCoeff();  // Size: (n_img)

  // Subtract row_max and exponentiate for temp_dist1
  Matrix exp1 =
      (temp_dist1.colwise() - row_max).array().exp();  // Size: (n_img, n_txt)

  // Compute row-wise sum and log for temp_dist1
  Eigen::VectorXf row_sum = exp1.rowwise().sum();  // Size: (n_img)
  Eigen::VectorXf row_log =
      row_sum.array().log() + row_max.array();  // Size: (n_img)

  // Compute term1
  float term1 = row_log.sum() / (multi_vector_cardinality * temperature *
                                 temperature_txt_scale);

  // Compute column-wise max for temp_dist2
  Eigen::VectorXf col_max = temp_dist2.colwise().maxCoeff();  // Size: (n_txt)

  // Subtract col_max and exponentiate for temp_dist2
  Matrix exp2 = (temp_dist2.rowwise() - col_max.transpose())
                    .array()
                    .exp();  // Size: (n_img, n_txt)

  // Compute column-wise sum and log for temp_dist2
  Eigen::VectorXf col_sum = exp2.colwise().sum();  // Size: (n_txt)
  Eigen::VectorXf col_log =
      col_sum.array().log() + col_max.array();  // Size: (n_txt)

  // Compute term2
  float term2 = col_log.sum() / (multi_vector_cardinality * temperature);

  // Smooth Chamfer Distance
  return (term1 + term2) / denominator;
}

std::vector<float> MultiVectorReranker::ComputeSmoothChamferDistanceBatch(
    const Eigen::Ref<const Matrix>& img_embs,
    const Eigen::Ref<const Matrix>& txt_embs,
    std::vector<Cardinality>& cardinalities) {
  auto ret = std::vector<float>(cardinalities.size());
  Matrix dists(img_embs.rows(), txt_embs.rows());
  vector_distance_metric(img_embs, txt_embs, dists);
  // Now split the result and do per data computation
  auto offset = 0;
  for (auto data_id = 0; data_id < cardinalities.size(); ++data_id) {
    // Note that, this involves non-contiguous memory access.
    // For now I think it is fine, but if it becomes a bottleneck, we can
    // consider optimizing it.
    auto dist = dists.block(0, offset, img_embs.rows(), cardinalities[data_id]);
    Matrix temp_dist1 = (temperature * temperature_txt_scale * dist);
    Matrix temp_dist2 = (temperature * dist);

    // Compute row-wise max for temp_dist1
    Eigen::VectorXf row_max = temp_dist1.rowwise().maxCoeff();  // Size: (n_img)

    // Subtract row_max and exponentiate for temp_dist1
    Matrix exp1 =
        (temp_dist1.colwise() - row_max).array().exp();  // Size: (n_img, n_txt)

    // Compute row-wise sum and log for temp_dist1
    Eigen::VectorXf row_sum = exp1.rowwise().sum();  // Size: (n_img)
    Eigen::VectorXf row_log =
        row_sum.array().log() + row_max.array();  // Size: (n_img)

    // Compute term1
    float term1 = row_log.sum() / (multi_vector_cardinality * temperature *
                                   temperature_txt_scale);

    // Compute column-wise max for temp_dist2
    Eigen::VectorXf col_max = temp_dist2.colwise().maxCoeff();  // Size: (n_txt)

    // Subtract col_max and exponentiate for temp_dist2
    Matrix exp2 = (temp_dist2.rowwise() - col_max.transpose())
                      .array()
                      .exp();  // Size: (n_img, n_txt)

    // Compute column-wise sum and log for temp_dist2
    Eigen::VectorXf col_sum = exp2.colwise().sum();  // Size: (n_txt)
    Eigen::VectorXf col_log =
        col_sum.array().log() + col_max.array();  // Size: (n_txt)

    // Compute term2
    float term2 = col_log.sum() / (multi_vector_cardinality * temperature);

    // Smooth Chamfer Distance
    ret[data_id] = (term1 + term2) / denominator;

    offset += cardinalities[data_id];
  }
  return ret;
}

float MultiVectorReranker::ComputeSummedMaxSimilarity(
    const Eigen::Ref<const Matrix>& img_embs,
    const Eigen::Ref<const Matrix>& txt_embs) {
  Matrix dist(img_embs.rows(), txt_embs.rows());
  vector_distance_metric(img_embs, txt_embs, dist);
  return dist.rowwise().maxCoeff().sum();
}

void MultiVectorReranker::SetDistanceMetric(
    const std::string& set_to_set_metric,
    const std::string& vector_dist_metric) {
  // Map for vector distance metrics (pairwise distances)
  static const std::unordered_map<
      std::string,
      std::function<void(const Eigen::Ref<const Matrix>&,
                         const Eigen::Ref<const Matrix>&, Eigen::Ref<Matrix>)>>
      vector_metric_map = {
          {"cosine",
           [this](const Eigen::Ref<const Matrix>& X,
                  const Eigen::Ref<const Matrix>& Y,
                  Eigen::Ref<Matrix> result) {
             this->computeCosineSimilarity(X, Y, result);
           }},
          {"cosine_gpu", [this](const Eigen::Ref<const Matrix>& X,
                                const Eigen::Ref<const Matrix>& Y,
                                Eigen::Ref<Matrix> result) {
             this->computeCosineSimilarityGPU(X, Y, result);
           }}};

  // Map for set-to-set distance metrics
  static const std::unordered_map<
      std::string, std::function<float(const Eigen::Ref<const Matrix>&,
                                       const Eigen::Ref<const Matrix>&)>>
      set_to_set_metric_map = {
          {"smooth_chamfer",
           [this](const Eigen::Ref<const Matrix>& X,
                  const Eigen::Ref<const Matrix>& Y) {
             return this->computeSmoothChamferDistance(X, Y);
           }},
          {"summed_max_similarity", [this](const Eigen::Ref<const Matrix>& X,
                                           const Eigen::Ref<const Matrix>& Y) {
             return this->ComputeSummedMaxSimilarity(X, Y);
           }}};
  static const std::unordered_map<
      std::string,
      std::function<std::vector<float>(const Eigen::Ref<const Matrix>&,
                                       const Eigen::Ref<const Matrix>&,
                                       std::vector<Cardinality>&)>>
      set_to_set_batch_map = {
          {"smooth_chamfer", [this](const Eigen::Ref<const Matrix>& X,
                                    const Eigen::Ref<const Matrix>& Y,
                                    std::vector<Cardinality>& cardinalities) {
             return this->ComputeSmoothChamferDistanceBatch(X, Y,
                                                            cardinalities);
           }}};

  // Resolve vector distance metric
  auto vector_metric_it = vector_metric_map.find(vector_dist_metric);
  auto set_to_set_metric_it = set_to_set_metric_map.find(set_to_set_metric);
  auto set_to_set_metric_batch_it =
      set_to_set_batch_map.find(set_to_set_metric);
  if (vector_metric_it == vector_metric_map.end() ||
      set_to_set_metric_it == set_to_set_metric_map.end()) {
    throw std::invalid_argument("Unsupported metric");
  }
  vector_distance_metric = vector_metric_it->second;
  set_to_set_distance_metric = set_to_set_metric_it->second;
  // set_to_set_metric_batch can be nullptr if not implemented yet.
  if (set_to_set_metric_batch_it != set_to_set_batch_map.end()) {
    set_to_set_distance_metric_batch = set_to_set_metric_batch_it->second;
  }
}

Matrix Loader::LoadEmbeddingVector(const std::string& file_path) {
  unsigned points_num = 0;
  unsigned dim = 0;

  // Open the file in binary mode
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file: " + file_path);
  }

  // Read the number of points and dimensions (8 bytes)
  in.read(reinterpret_cast<char*>(&points_num), sizeof(unsigned));
  in.read(reinterpret_cast<char*>(&dim), sizeof(unsigned));

  // Verify that the file size matches the expected size
  in.seekg(0, std::ios::end);
  std::streampos file_size = in.tellg();
  std::size_t expected_size =
      sizeof(unsigned) * 2 +
      static_cast<std::size_t>(points_num) * dim * sizeof(float);
  if (static_cast<std::size_t>(file_size) != expected_size) {
    std::cerr << "File size does not match expected size.\n";
    std::cerr << "Expected size: " << expected_size
              << " bytes, but file size is " << file_size << " bytes.\n";
    throw std::runtime_error("Data file size mismatch.");
  }

  // Return to the position after the header
  in.seekg(sizeof(unsigned) * 2, std::ios::beg);

  // Allocate the Eigen matrix
  Matrix data_matrix(points_num, dim);

  // Read the data directly into the Eigen matrix
  in.read(reinterpret_cast<char*>(data_matrix.data()),
          static_cast<std::streamsize>(points_num * dim * sizeof(float)));

  if (!in) {
    throw std::runtime_error("Error reading data from file: " + file_path);
  }

  in.close();

  return data_matrix;
}

FloatVectorPtr Loader::LoadEmbeddingVectorAsFloatVector(
    const std::string& file_path) {
  unsigned points_num = 0;
  unsigned dim = 0;

  // Open the file in binary mode
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file: " + file_path);
  }

  // Read the number of points and dimensions (8 bytes)
  in.read(reinterpret_cast<char*>(&points_num), sizeof(unsigned));
  in.read(reinterpret_cast<char*>(&dim), sizeof(unsigned));

  // Verify that the file size matches the expected size
  in.seekg(0, std::ios::end);
  std::streampos file_size = in.tellg();
  std::size_t expected_size =
      sizeof(unsigned) * 2 +
      static_cast<std::size_t>(points_num) * dim * sizeof(float);
  if (static_cast<std::size_t>(file_size) != expected_size) {
    std::cerr << "File size does not match expected size.\n";
    std::cerr << "Expected size: " << expected_size
              << " bytes, but file size is " << file_size << " bytes.\n";
    throw std::runtime_error("Data file size mismatch.");
  }

  // Return to the position after the header
  in.seekg(sizeof(unsigned) * 2, std::ios::beg);

  // Allocate the outer vector
  auto result = std::make_shared<std::vector<std::vector<float>>>();
  result->reserve(points_num);

  // Allocate a temporary buffer to read all the data
  std::vector<float> buffer(points_num * dim);
  in.read(reinterpret_cast<char*>(buffer.data()),
          static_cast<std::streamsize>(points_num * dim * sizeof(float)));

  if (!in) {
    throw std::runtime_error("Error reading data from file: " + file_path);
  }

  // Populate the result with individual vectors for each point
  for (unsigned i = 0; i < points_num; ++i) {
    result->emplace_back(buffer.begin() + i * dim,
                         buffer.begin() + (i + 1) * dim);
  }

  in.close();

  return result;
}

void RecallCalculator::SetGroundTruth(SetGroundTruthVectorPtr ground_truth) {
  this->ground_truth = ground_truth;
}

double RecallCalculator::ComputeRecall(
    VectorSetID query_id, const std::vector<VectorSetID>& reranked_indices) {
  if (ground_truth == nullptr) {
    throw std::runtime_error("Ground truth not set.");
  }
  if (k == 0) {
    throw std::runtime_error("Recall@k not set.");
  }

  // Access the ground truth for the given query
  const auto& gt_for_this_query = (*ground_truth)[query_id];

  if (k > gt_for_this_query.size()) {
    throw std::invalid_argument(
        "k is greater than the number of ground truth items.");
  }

  std::unordered_set<VectorSetID> gt_set(gt_for_this_query.begin(),
                                         gt_for_this_query.begin() + k);

  // Compute the number of relevant items in the top-k reranked indices
  uint32_t num_relevant = 0;
  const size_t actual_k =
      std::min(static_cast<size_t>(k), reranked_indices.size());
  for (size_t i = 0; i < actual_k; ++i) {
    if (gt_set.find(reranked_indices[i]) != gt_set.end()) {
      ++num_relevant;
    }
  }

  // Return the recall as the fraction of top-k ground truth items found
  return static_cast<double>(num_relevant) / k;
}

double RecallCalculator::ComputePairedRecall(
    VectorSetID query_id, const std::vector<VectorSetID>& reranked_indices) {
  if (paired_ground_truth == nullptr) {
    throw std::runtime_error("Paired ground truth not set.");
  }
  if (k == 0) {
    throw std::runtime_error("Recall@k not set.");
  }

  // Access the ground truth for the given query
  const auto& gt_for_this_query = paired_ground_truth(query_id);
  auto duration_start = gt_for_this_query.first;
  auto duration_end = gt_for_this_query.first + gt_for_this_query.second;
  const size_t actual_k =
      std::min(static_cast<size_t>(k), reranked_indices.size());
  // In the context of retrieval task, if any of the relevant data is found,
  // then recall is 1, otherwise 0.
  for (size_t i = 0; i < actual_k; ++i) {
    if (reranked_indices[i] >= duration_start &&
        reranked_indices[i] < duration_end) {
      return 1.0;
    }
  }
  return 0.0;
}

SetGroundTruthVectorPtr Loader::LoadGroundTruth(const std::string& file_path) {
  // Open the file in binary mode
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file: " + file_path);
  }

  unsigned int num_queries = 0;
  unsigned int num_gt_per_query = 0;

  // Read the number of queries and number of ground truth per query (8 bytes)
  in.read(reinterpret_cast<char*>(&num_queries), sizeof(unsigned int));
  in.read(reinterpret_cast<char*>(&num_gt_per_query), sizeof(unsigned int));

  // Verify that the file size matches the expected size
  in.seekg(0, std::ios::end);
  std::streampos file_size = in.tellg();
  std::size_t expected_size =
      sizeof(unsigned int) * 2 + static_cast<std::size_t>(num_queries) *
                                     num_gt_per_query * sizeof(unsigned int);

  if (static_cast<std::size_t>(file_size) != expected_size) {
    std::cerr << "File size does not match expected size.\n";
    std::cerr << "Expected size: " << expected_size
              << " bytes, but file size is " << file_size << " bytes.\n";
    throw std::runtime_error("Ground truth file size mismatch.");
  }

  // Return to the position after the header
  in.seekg(sizeof(unsigned int) * 2, std::ios::beg);

  // Allocate the ground truth data structure
  auto ground_truth =
      std::make_shared<std::vector<std::vector<VectorSetID>>>(num_queries);

  // Read the ground truth data
  for (unsigned int i = 0; i < num_queries; ++i) {
    std::vector<VectorSetID> gt_vector(num_gt_per_query);
    in.read(
        reinterpret_cast<char*>(gt_vector.data()),
        static_cast<std::streamsize>(num_gt_per_query * sizeof(unsigned int)));

    if (!in) {
      throw std::runtime_error("Error reading ground truth data from file: " +
                               file_path);
    }

    // Store the ground truth vector
    (*ground_truth)[i] = std::move(gt_vector);
  }

  in.close();

  return ground_truth;
}

VectorGroundTruthVectorPtr Loader::LoadVectorGroundTruth(
    const std::string& file_path) {
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error: Unable to open file " << file_path << std::endl;
    throw std::runtime_error("File open error");
  }

  // Read metadata: points_num and dim
  uint32_t points_num, dim;
  in.read(reinterpret_cast<char*>(&points_num), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

  auto result = std::make_shared<std::vector<std::vector<VectorID>>>(
      points_num, std::vector<VectorID>(dim));

  for (size_t i = 0; i < points_num; ++i) {
    in.read(reinterpret_cast<char*>((*result)[i].data()),
            dim * sizeof(uint32_t));
  }

  in.seekg(0, std::ios::end);
  std::ios::pos_type file_size = in.tellg();

  size_t expected_size =
      sizeof(uint32_t) * 2                   // Metadata
      + points_num * dim * sizeof(uint32_t)  // Nearest neighbor IDs
      + points_num * dim * sizeof(float);    // Distances (skipped)

  if (static_cast<size_t>(file_size) != expected_size) {
    std::cerr << "Error: File size mismatch. Expected " << expected_size
              << " bytes but got " << file_size << " bytes." << std::endl;
    throw std::runtime_error("File size validation failed");
  }

  in.close();
  return result;
}

std::pair<std::function<VectorSetID(VectorID)>,
          std::function<std::pair<VectorID, Cardinality>(VectorSetID)>>
Loader::LoadVectorCardinalityMappingAndGetBothMappings(std::string& file_path) {
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error: Unable to open file " << file_path << std::endl;
    throw std::runtime_error("File open error");
  }
  uint32_t num_data;
  in.read(reinterpret_cast<char*>(&num_data), sizeof(uint32_t));

  auto vector_id_to_vector_set_id_map =
      std::make_shared<std::unordered_map<VectorID, VectorSetID>>();
  auto vector_set_id_to_vector_id_map = std::make_shared<
      std::unordered_map<VectorSetID, std::pair<VectorID, Cardinality>>>();
  VectorID vector_id_offset = 0;
  for (VectorSetID i = 0; i < num_data; ++i) {
    Cardinality cardinality;
    in.read(reinterpret_cast<char*>(&cardinality), sizeof(cardinality));
    (*vector_set_id_to_vector_id_map)[i] =
        std::make_pair(vector_id_offset, cardinality);
    for (uint32_t j = 0; j < cardinality; ++j) {
      (*vector_id_to_vector_set_id_map)[vector_id_offset] = i;
      ++vector_id_offset;
    }
  }

  size_t expected_size = sizeof(uint32_t) + num_data * sizeof(uint32_t);
  in.seekg(0, std::ios::end);
  if (in.tellg() != expected_size) {
    std::cerr << "Error: File size mismatch. Expected " << expected_size
              << " bytes but got " << in.tellg() << " bytes." << std::endl;
    throw std::runtime_error("File size validation failed");
  }
  in.close();
  return std::make_pair(
      [vector_id_to_vector_set_id_map](VectorID vector_id) {
        return vector_id_to_vector_set_id_map->at(vector_id);
      },
      [vector_set_id_to_vector_id_map](VectorSetID vector_set_id) {
        return vector_set_id_to_vector_id_map->at(vector_set_id);
      });
}

GroundTruthMapping Loader::LoadQueryDataPairMappingAsFunction(
    std::string& file_path) {
  std::ifstream in(file_path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error: Unable to open file " << file_path << std::endl;
    throw std::runtime_error("File open error");
  }
  uint32_t num_data;
  in.read(reinterpret_cast<char*>(&num_data), sizeof(uint32_t));
  auto mapping = std::make_shared<
      std::unordered_map<VectorSetID, std::pair<VectorSetID, Cardinality>>>();
  VectorSetID offset = 0;
  for (VectorSetID i = 0; i < num_data; ++i) {
    Cardinality cardinality;
    in.read(reinterpret_cast<char*>(&cardinality), sizeof(cardinality));
    (*mapping)[i] = std::make_pair(offset, cardinality);
    offset += cardinality;
  }
  size_t expected_size = sizeof(uint32_t) + num_data * sizeof(uint32_t);
  in.seekg(0, std::ios::end);
  if (in.tellg() != expected_size) {
    std::cerr << "Error: File size mismatch. Expected " << expected_size
              << " bytes but got " << in.tellg() << " bytes." << std::endl;
    throw std::runtime_error("File size validation failed");
  }
  in.close();
  return [mapping](VectorSetID vector_set_id) {
    return mapping->at(vector_set_id);
  };
}

template void MultiVectorReranker::Rerank<unsigned int>(
    VectorSetID& query_id,
    const std::vector<std::vector<unsigned int>>& indices,
    std::vector<VectorSetID>& reranked_indices);

template void MultiVectorReranker::Rerank<size_t>(
    VectorSetID& query_id, const std::vector<std::vector<size_t>>& indices,
    std::vector<VectorSetID>& reranked_indices);
