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
  Eigen::Vector6s axis;
  // This is the transform from the parent
  Eigen::Isometry3s transformFromParent;
  // This is the transform from the children
  Eigen::Isometry3s transformFromChildren;
  // This is the spatial inertia matrix for the body node
  Eigen::Matrix6s inertia;
  // -1 indicates this is the root element, otherwise this is the index into
  // SimpleFeatherstone::mJointsAndBodies where the parent lives
  int parentIndex;
};

struct FeatherstoneScratchSpace
{
  Eigen::Isometry3s transformFromParent;
  Eigen::Vector6s spatialVelocity;
  Eigen::Vector6s spatialAcceleration;

  Eigen::Matrix6s articulatedInertia;
  Eigen::Vector6s articulatedBiasForce;

  // Intermediate values without convenient names. From the symbols on
  // page 12 of http://www.cs.cmu.edu/~junggon/tools/liegroupdynamics.pdf
  s_t psi;
  s_t totalForce;
  Eigen::Vector6s partialAcceleration; // = eta
  Eigen::Matrix6s phi;
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
      s_t* pos,
      s_t* vel,
      s_t* force,
      /* OUT */ s_t* accelerations);

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