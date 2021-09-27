#ifndef DART_NEURAL_DIFF_NODE_HPP_
#define DART_NEURAL_DIFF_NODE_HPP_

#include <memory>
#include <optional>

#include <Eigen/Dense>

namespace dart {

namespace neural {

class DiffNode
{
public:
  DiffNode();

  virtual ~DiffNode();
};

} // namespace neural
} // namespace dart

#endif