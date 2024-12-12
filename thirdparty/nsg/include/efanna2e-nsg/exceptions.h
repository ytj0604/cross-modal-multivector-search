//
// Copyright (c) 2017 ZJULearning. All rights reserved.
//
// This source code is licensed under the MIT license.
//
#pragma once

#include <stdexcept>

namespace efanna2e_nsg {

class NotImplementedException : public std::logic_error {
 public:
  NotImplementedException() : std::logic_error("Function not yet implemented.") {}
};

}
