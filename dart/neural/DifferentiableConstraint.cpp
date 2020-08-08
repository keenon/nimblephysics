#include "dart/neural/DifferentiableConstraint.hpp"

#include "dart/collision/Contact.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/constraint/ContactConstraint.hpp"

namespace dart {
namespace neural {

DifferentiableConstraint::DifferentiableConstraint(
    std::shared_ptr<constraint::ConstraintBase> constraint, int index)
{
  mConstraint = constraint;
  mIndex = index;
  if (mConstraint->isContactConstraint())
  {
    mContactConstraint
        = std::static_pointer_cast<constraint::ContactConstraint>(mConstraint);
    // This needs to be explicitly copied, otherwise the memory is overwritten
    mContact = std::make_shared<collision::Contact>(
        mContactConstraint->getContact());
  }
}

Eigen::Vector3d DifferentiableConstraint::getContactWorldPosition()
{
  if (!mConstraint->isContactConstraint())
  {
    return Eigen::Vector3d::Zero();
  }
  return mContact->point;
}

Eigen::Vector3d DifferentiableConstraint::getContactWorldNormal()
{
  if (!mConstraint->isContactConstraint())
  {
    return Eigen::Vector3d::Zero();
  }
  if (mIndex == 0)
  {
    return mContact->normal;
  }
  else
  {
    return mContactConstraint->getTangentBasisMatrixODE(mContact->normal)
        .col(mIndex - 1);
  }
}

} // namespace neural
} // namespace dart
