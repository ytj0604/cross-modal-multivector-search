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

void MultiVectorReranker::computeCosineSimilarity(const Eigen::Ref<const MatrixType>& X,
                             const Eigen::Ref<const MatrixType>& Y,
                             Eigen::Ref<MatrixType> result) {
  result = X * Y.transpose();
}

float MultiVectorReranker::computeSmoothChamferDistance(
    const Eigen::Ref<const MatrixType>& img_embs,
    const Eigen::Ref<const MatrixType>& txt_embs) {
  MatrixType dist(img_embs.rows(), txt_embs.rows());
  vector_distance_metric(img_embs, txt_embs, dist);

  MatrixType temp_dist1 = temperature * temperature_txt_scale * dist;
  MatrixType temp_dist2 = temperature * dist;

  Eigen::VectorXf col_max = temp_dist1.colwise().maxCoeff();
  Eigen::VectorXf row_max = temp_dist2.rowwise().maxCoeff();

  MatrixType exp1 = (temp_dist1.colwise() - col_max).array().exp();
  MatrixType exp2 = (temp_dist2.rowwise() - row_max.transpose()).array().exp();

  // Compute sums
  Eigen::VectorXf row_sum = exp1.rowwise().sum();
  Eigen::VectorXf col_sum = exp2.colwise().sum();

  // Compute logarithms
  Eigen::VectorXf row_log = row_sum.array().log() + col_max.transpose().array();
  Eigen::VectorXf col_log = col_sum.array().log() + row_max.array();

  // Compute terms
  float term1 = row_log.sum() / (multi_vector_cardinality * temperature *
                                 temperature_txt_scale);
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
