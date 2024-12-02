#ifndef DART_NEURAL_WRT_DAMPING_HPP_
#define DART_NEURAL_WRT_DAMPING_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/WithRespectTo.hpp"

namespace dart {
namespace simulation {
class World;
}

namespace dynamics {
class Skeleton;
class BodyNode;
class Joint;
} // namespace dynamics

namespace neural {

enum WrtDampingJointEntryType
{
  DAMPING
};

struct WrtDampingJointEntry
{
  std::string jointName;
  WrtDampingJointEntryType type;
  int mDofs;
  Eigen::VectorXi mWorldDofs;

  WrtDampingJointEntry(
      std::string jointName,
      WrtDampingJointEntryType type,
      int dofs,
      Eigen::VectorXi worldDofs);

  int dim();

  void get(dynamics::Skeleton* skel, Eigen::Ref<Eigen::VectorXs> out);

  Eigen::VectorXi getDofMapping();

  void set(dynamics::Skeleton* skel, const Eigen::Ref<Eigen::VectorXs>& val);
};

class WithRespectToDamping : public WithRespectTo
{
public:
  WithRespectToDamping();

  /// This registers that we'd like to keep track of this node's mass in this
  /// way in this differentiation
  /// Need to set the dofs
  WrtDampingJointEntry& registerJoint(
      dynamics::Joint* joint,
      WrtDampingJointEntryType type,
      Eigen::VectorXi dofs_index,
      Eigen::VectorXs upperBound,
      Eigen::VectorXs lowerBound);

  /// This returns the entry object corresponding to this node. Throws an
  /// assertion if this node doesn't exist
  WrtDampingJointEntry& getJoint(dynamics::Joint* joint);

  //////////////////////////////////////////////////////////////
  // Implement all the methods we need
  //////////////////////////////////////////////////////////////

  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;

  Eigen::VectorXi getDofsMapping(simulation::World* world);

  Eigen::VectorXi getDofsMapping(dynamics::Skeleton* skel);

protected:
  std::unordered_map<std::string, std::vector<WrtDampingJointEntry>> mEntries;
  Eigen::VectorXs mUpperBounds;
  Eigen::VectorXs mLowerBounds;
};

} // namespace neural
} // namespace dart

#endif