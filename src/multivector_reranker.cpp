#include <multivector_reranker.h>

void MultiVectorReranker::SetVectorID2VectorSetIDMapping(
    std::function<VectorSetID(VectorID)> f) {
  vector_id_to_vector_set_id = f;
}

void MultiVectorReranker::SetMultiVectorCardinality(uint32_t cardinality) {
  this->multi_vector_cardinality = cardinality;
}

void MultiVectorReranker::SetDataVector(const MatrixType& data_matrix) {
  this->data_matrix = data_matrix;
}

void MultiVectorReranker::SetQueryVector(const MatrixType& query_matrix) {
  this->query_matrix = query_matrix;
}

void MultiVectorReranker::Rerank(
    VectorSetID& query_id, const std::vector<std::vector<VectorID>>& indices,
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

  Eigen::Map<const MatrixType> query_set(
      query_matrix.data() +
          query_id * multi_vector_cardinality * query_matrix.cols(),
      multi_vector_cardinality, query_matrix.cols());

  std::vector<std::pair<float, VectorSetID>> relevance_scores;
  relevance_scores.reserve(deduplicated_indices.size());

  for (VectorSetID data_set_id : deduplicated_indices) {
    Eigen::Map<const MatrixType> data_set(
        data_matrix.data() +
            data_set_id * multi_vector_cardinality * data_matrix.cols(),
        multi_vector_cardinality, data_matrix.cols());
    float relevance = set_to_set_distance_metric(query_set, data_set);
    relevance_scores.emplace_back(relevance, data_set_id);
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
  Eigen::Map<const MatrixType> query_set(
      query_matrix.data() +
          query_id * multi_vector_cardinality * query_matrix.cols(),
      multi_vector_cardinality, query_matrix.cols());

  std::vector<std::pair<float, VectorSetID>> relevance_scores;
  relevance_scores.reserve(data_matrix.rows() / multi_vector_cardinality);

  for (VectorSetID data_set_id = 0;
       data_set_id < data_matrix.rows() / multi_vector_cardinality;
       ++data_set_id) {
    Eigen::Map<const MatrixType> data_set(
        data_matrix.data() +
            data_set_id * multi_vector_cardinality * data_matrix.cols(),
        multi_vector_cardinality, data_matrix.cols());
    float relevance = set_to_set_distance_metric(query_set, data_set);
    relevance_scores.emplace_back(relevance, data_set_id);
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

void MultiVectorReranker::computeCosineSimilarity(
    const Eigen::Ref<const MatrixType>& X,
    const Eigen::Ref<const MatrixType>& Y, Eigen::Ref<MatrixType> result) {
  result = X * Y.transpose();
}

float MultiVectorReranker::computeSmoothChamferDistance(
    const Eigen::Ref<const MatrixType>& img_embs,
    const Eigen::Ref<const MatrixType>& txt_embs) {
  // Compute cosine similarity
  MatrixType dist(img_embs.rows(), txt_embs.rows());
  vector_distance_metric(img_embs, txt_embs, dist);

  // Compute temp_dist1 and temp_dist2
  MatrixType temp_dist1 = temperature * temperature_txt_scale * dist;
  MatrixType temp_dist2 = temperature * dist;

  // Compute row-wise max for temp_dist1
  Eigen::VectorXf row_max = temp_dist1.rowwise().maxCoeff();  // Size: (n_img)

  // Subtract row_max and exponentiate for temp_dist1
  MatrixType exp1 =
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
  MatrixType exp2 = (temp_dist2.rowwise() - col_max.transpose())
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

void MultiVectorReranker::SetDistanceMetric(
    const std::string& set_to_set_metric,
    const std::string& vector_dist_metric) {
  // Map for vector distance metrics (pairwise distances)
  static const std::unordered_map<
      std::string, std::function<void(const Eigen::Ref<const MatrixType>&,
                                      const Eigen::Ref<const MatrixType>&,
                                      Eigen::Ref<MatrixType>)>>
      vector_metric_map = {
          {"cosine", [this](const Eigen::Ref<const MatrixType>& X,
                            const Eigen::Ref<const MatrixType>& Y,
                            Eigen::Ref<MatrixType> result) {
             this->computeCosineSimilarity(X, Y, result);
           }}};

  // Map for set-to-set distance metrics
  static const std::unordered_map<
      std::string, std::function<float(const Eigen::Ref<const MatrixType>&,
                                       const Eigen::Ref<const MatrixType>&)>>
      set_to_set_metric_map = {
          {"smooth_chamfer", [this](const Eigen::Ref<const MatrixType>& X,
                                    const Eigen::Ref<const MatrixType>& Y) {
             return this->computeSmoothChamferDistance(X, Y);
           }}};

  // Resolve vector distance metric
  auto vector_metric_it = vector_metric_map.find(vector_dist_metric);
  auto set_to_set_metric_it = set_to_set_metric_map.find(set_to_set_metric);
  if (vector_metric_it == vector_metric_map.end() ||
      set_to_set_metric_it == set_to_set_metric_map.end()) {
    throw std::invalid_argument("Unsupported metric");
  }
  vector_distance_metric = vector_metric_it->second;
  set_to_set_distance_metric = set_to_set_metric_it->second;
}

MatrixType Loader::LoadEmbeddingVector(const std::string& file_path) {
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
  MatrixType data_matrix(points_num, dim);

  // Read the data directly into the Eigen matrix
  in.read(reinterpret_cast<char*>(data_matrix.data()),
          static_cast<std::streamsize>(points_num * dim * sizeof(float)));

  if (!in) {
    throw std::runtime_error("Error reading data from file: " + file_path);
  }

  in.close();

  return data_matrix;
}

void RecallCalculator::SetGroundTruth(GroundTruthType ground_truth) {
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

GroundTruthType Loader::LoadGroundTruth(const std::string& file_path) {
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

VectorGroundTruthType Loader::LoadVectorGroundTruth(
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
    in.read(reinterpret_cast<char*>((*result)[i].data()), dim * sizeof(uint32_t));
  }

  // Validate file size to ensure it matches the expected structure
  std::ios::pos_type cursor_position = in.tellg();
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