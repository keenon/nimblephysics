#ifndef DART_NEURAL_DIFF_CONSTRAINT_HPP_
#define DART_NEURAL_DIFF_CONSTRAINT_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

namespace constraint {
class ConstrainedGroup;
class ConstraintBase;
class ContactConstraint;
} // namespace constraint

namespace collision {
class Contact;
}

namespace neural {
class DifferentiableConstraint
{

public:
  DifferentiableConstraint(
      std::shared_ptr<constraint::ConstraintBase> constraint, int index);

  Eigen::Vector3d getContactWorldPosition();

  Eigen::Vector3d getContactWorldNormal();

protected:
  std::shared_ptr<constraint::ConstraintBase> mConstraint;
  std::shared_ptr<constraint::ContactConstraint> mContactConstraint;
  std::shared_ptr<collision::Contact> mContact;
  int mIndex;
};
} // namespace neural
} // namespace dart

#endif