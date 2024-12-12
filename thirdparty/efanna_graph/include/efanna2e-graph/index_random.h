//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//

#pragma once

#include "efanna2e-graph/index.h"
#include "efanna2e-graph/util.h"

namespace efanna2e_graph {

class IndexRandom : public Index {
public:
    IndexRandom(const size_t dimension, const size_t n);
    virtual ~IndexRandom();
    std::mt19937 rng;
    void Save(const char *filename)override{}
    void Load(const char *filename)override{}
    virtual void Build(size_t n, const float *data, const Parameters &parameters) override;

    virtual void Search(
      const float *query,
      const float *x,
      size_t k,
      const Parameters &parameters,
      unsigned *indices) override ;

};

}
