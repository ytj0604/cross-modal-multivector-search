//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//
#include <efanna2e-nsg/index.h>
namespace efanna2e_nsg {
Index::Index(const size_t dimension, const size_t n, Metric metric = L2)
  : dimension_ (dimension), nd_(n), has_built(false) {
    switch (metric) {
      case L2:distance_ = new DistanceL2();
        break;
      case INNER_PRODUCT:distance_ = new DistanceInnerProduct();
        break;
      default:distance_ = new DistanceL2();
        break;
    }
}
Index::~Index() {}
}
