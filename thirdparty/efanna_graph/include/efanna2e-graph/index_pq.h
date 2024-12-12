//
// Created by 付聪 on 2017/6/26.
//

#pragma once
#include <faiss/AutoTune.h>
#include "efanna2e-graph/util.h"
#include "efanna2e-graph/parameters.h"
#include "efanna2e-graph/neighbor.h"
#include "efanna2e-graph/index.h"
#include <cassert>

namespace efanna2e_graph {
class IndexPQ : public Index {
 public:
  explicit IndexPQ(const size_t dimension, const size_t n, Metric m, Index *initializer);


  virtual ~IndexPQ();

  virtual void Save(const char *filename)override;
  virtual void Load(const char *filename)override;


  virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

  virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override;

 protected:

  Index *initializer_;
  faiss::Index* index;
  void compute_gt_for_tune(const float* q,
                           const unsigned nq,
                          const unsigned k,
                           unsigned *gt);
};
}

