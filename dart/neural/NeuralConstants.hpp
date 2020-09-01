#ifndef DART_NEURAL_CONSTANTS_HPP_
#define DART_NEURAL_CONSTANTS_HPP_

#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace dart {
namespace dynamics {
class BodyNode;
}

namespace neural {

enum WithRespectTo
{
  POSITION,
  VELOCITY,
  FORCE,
  LINK_MASSES,
  LINK_COMS,
  LINK_MOIS
};

} // namespace neural
} // namespace dart

#endif