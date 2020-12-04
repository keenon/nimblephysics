#ifndef DART_FAST_FEATHERSTONE
#define DART_FAST_FEATHERSTONE

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "dart/math/Geometry.hpp"

namespace dart {
namespace dynamics {

class Skeleton;

struct JointAndBody
{
  // This is a normalized transform, represented in log-space. You can recover
  // the transform by dart::math::expMap(axis * position)
  Eigen::Vector6d axis;
  // This is the transform from the parent
  Eigen::Isometry3d transformFromParent;
  // This is the transform from the children
  Eigen::Isometry3d transformFromChildren;
  // This is the spatial inertia matrix for the body node
  Eigen::Matrix6d inertia;
  // -1 indicates this is the root element, otherwise this is the index into
  // SimpleFeatherstone::mJointsAndBodies where the parent lives
  int parentIndex;
};

struct FeatherstoneScratchSpace
{
  Eigen::Isometry3d transformFromParent;
  Eigen::Vector6d spatialVelocity;
  Eigen::Vector6d spatialAcceleration;

  Eigen::Matrix6d articulatedInertia;
  Eigen::Vector6d articulatedBiasForce;

  // Intermediate values without convenient names. From the symbols on
  // page 12 of http://www.cs.cmu.edu/~junggon/tools/liegroupdynamics.pdf
  double psi;
  double totalForce;
  Eigen::Vector6d partialAcceleration; // = eta
  Eigen::Matrix6d phi;
};

class SimpleFeatherstone
{
public:
  // This creates a new JointAndBody object in our vector, and returns it by
  // reference
  JointAndBody& emplaceBack();

  // The number of joints in this skeleton
  int len();

  // This computes accelerations. All the pointer arguments are assumed to point
  // to arrays of length len()
  void forwardDynamics(
      double* pos,
      double* vel,
      double* force,
      /* OUT */ double* accelerations);

  // This gets the values from a DART skeleton to populate our Featherstone
  // implementation
  void populateFromSkeleton(
      const std::shared_ptr<dynamics::Skeleton>& skeleton);

  // protected:
  std::vector<JointAndBody> mJointsAndBodies;
  std::vector<FeatherstoneScratchSpace> mScratchSpace;
};

} // namespace dynamics
} // namespace dart

#endif