#include "dart/neural/WithRespectToDamping.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace neural {

//==============================================================================
/// Basic constructor
WithRespectToDamping::WithRespectToDamping()
{
}

//==============================================================================
WrtDampingJointEntry::WrtDampingJointEntry(
    std::string jointName, WrtDampingJointEntryType type, int dofs, Eigen::VectorXi worldDofs)
  : jointName(jointName), type(type), mDofs(dofs), mWorldDofs(worldDofs)
{
}

//==============================================================================
int WrtDampingJointEntry::dim()
{
  return mDofs;
}

//==============================================================================
void WrtDampingJointEntry::set(
    dynamics::Skeleton* skel, const Eigen::Ref<Eigen::VectorXs>& value)
{
  dynamics::Joint* joint = skel->getJoint(jointName);
  for(int i = 0; i < joint->getNumDofs(); i++)
  {
    joint->setDampingCoefficient(i, value(i));
  }
  return;
  

}

//==============================================================================
void WrtDampingJointEntry::get(
    dynamics::Skeleton* skel, Eigen::Ref<Eigen::VectorXs> out)
{
  dynamics::Joint* joint = skel->getJoint(jointName);
  for(int i = 0; i < joint->getNumDofs(); i++)
  {
    out(i) = joint->getDampingCoefficient(i);
  }
  return;
}

Eigen::VectorXi WrtDampingJointEntry::getDofMapping()
{
  return mWorldDofs;
}

//==============================================================================
/// This registers that we'd like to keep track of this node's mass in this
/// way in this differentiation
WrtDampingJointEntry& WithRespectToDamping::registerJoint(
    dynamics::Joint* joint,
    WrtDampingJointEntryType type,
    Eigen::VectorXi dofs_index,
    Eigen::VectorXs upperBound,
    Eigen::VectorXs lowerBound)
{
  std::string skelName = joint->getSkeleton()->getName();
  std::vector<WrtDampingJointEntry>& skelEntries = mEntries[skelName];
  skelEntries.emplace_back(joint->getName(), type, joint->getNumDofs(), dofs_index);

  WrtDampingJointEntry& entry = skelEntries[skelEntries.size() - 1];

  int dim = entry.dim();
  assert(
      dim == lowerBound.size()
      && "Lower bound must be same size as requested type would imply");
  assert(
      dim == upperBound.size()
      && "Upper bound must be same size as requested type would imply");

  assert(mUpperBounds.size() == mLowerBounds.size());

  // Append to the lower bound vector
  Eigen::VectorXs newDampingLowerBound
      = Eigen::VectorXs::Zero(mLowerBounds.size() + dim);
  newDampingLowerBound.segment(0, mLowerBounds.size()) = mLowerBounds;
  newDampingLowerBound.segment(mLowerBounds.size(), dim) = lowerBound;
  mLowerBounds = newDampingLowerBound;

  // Append to the upper bound vector
  Eigen::VectorXs newDampingUpperBound
      = Eigen::VectorXs::Zero(mUpperBounds.size() + dim);
  newDampingUpperBound.segment(0, mUpperBounds.size()) = mUpperBounds;
  newDampingUpperBound.segment(mUpperBounds.size(), dim) = upperBound;
  mUpperBounds = newDampingUpperBound;

  assert(mUpperBounds.size() == mLowerBounds.size());

  return entry;
}

//==============================================================================
/// This returns the entry object corresponding to this node
WrtDampingJointEntry& WithRespectToDamping::getJoint(dynamics::Joint* joint)
{
  std::string skelName = joint->getSkeleton()->getName();
  std::vector<WrtDampingJointEntry>& skelEntries = mEntries[skelName];
  for (WrtDampingJointEntry& entry : skelEntries)
  {
    if (entry.jointName == joint->getName())
    {
      return entry;
    }
  }
  assert(false);
  // The code should never reach this point, but this is here to keep the
  // compiler happy
  throw std::runtime_error{"Execution should never reach this point"};
}

//==============================================================================
/// This returns this WRT from the world as a vector
Eigen::VectorXs WithRespectToDamping::get(simulation::World* world)
{
  int worldDim = dim(world);
  Eigen::VectorXs result = Eigen::VectorXs::Zero(worldDim);
  int cursor = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    dynamics::Skeleton* skel = world->getSkeleton(i).get();
    int skelDim = dim(skel);
    result.segment(cursor, skelDim) = get(skel);
    cursor += skelDim;
  }
  assert(cursor == worldDim);
  return result;
}

//==============================================================================
/// This returns this WRT from this skeleton as a vector
Eigen::VectorXs WithRespectToDamping::get(dynamics::Skeleton* skel)
{
  std::vector<WrtDampingJointEntry>& skelEntries = mEntries[skel->getName()];
  if (skelEntries.size() == 0)
    return Eigen::VectorXs::Zero(0);
  int cursor = 0;
  int skelDim = dim(skel);
  Eigen::VectorXs result = Eigen::VectorXs::Zero(skelDim);
  for (WrtDampingJointEntry& entry : skelEntries)
  {
    entry.get(skel, result.segment(cursor, entry.dim()));
    cursor += entry.dim();
  }
  assert(cursor == skelDim);
  return result;
}

//==============================================================================
/// This sets the world's state based on our WRT
void WithRespectToDamping::set(simulation::World* world, Eigen::VectorXs value)
{
  int cursor = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    dynamics::Skeleton* skel = world->getSkeleton(i).get();
    int skel_dim = dim(skel);
    set(skel, value.segment(cursor, skel_dim));
    cursor += skel_dim;
  }
  assert(cursor == value.size());
}

//==============================================================================
/// This sets the skeleton's state based on our WRT
void WithRespectToDamping::set(dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  std::vector<WrtDampingJointEntry>& skelEntries = mEntries[skel->getName()];
  if (skelEntries.size() == 0)
    return;
  int cursor = 0;
  for (WrtDampingJointEntry& entry : skelEntries)
  {
    entry.set(skel, value.segment(cursor, entry.dim()));
    cursor += entry.dim();
  }
  assert(cursor == value.size());
}

//==============================================================================
/// This gives the dimensions of the WRT
int WithRespectToDamping::dim(simulation::World* world)
{
  int worldDim = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    worldDim += dim(world->getSkeleton(i).get());
  }
  return worldDim;
}

//==============================================================================
/// This gives the dimensions of the WRT
int WithRespectToDamping::dim(dynamics::Skeleton* skel)
{
  std::vector<WrtDampingJointEntry>& skelEntries = mEntries[skel->getName()];
  int skelDim = 0;
  for (WrtDampingJointEntry& entry : skelEntries)
  {
    skelDim += entry.dim();
  }
  return skelDim;
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToDamping::upperBound(simulation::World* /* world */)
{
  return mUpperBounds;
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToDamping::lowerBound(simulation::World* /* world */)
{
  return mLowerBounds;
}

Eigen::VectorXi WithRespectToDamping::getDofsMapping(simulation::World* world)
{
  int worldDim = dim(world);
  Eigen::VectorXi result = Eigen::VectorXi::Zero(worldDim);
  int cursor = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    dynamics::Skeleton* skel = world->getSkeleton(i).get();
    int skelDim = dim(skel);
    result.segment(cursor, skelDim) = getDofsMapping(skel);
    cursor += skelDim;
  }
  assert(cursor == worldDim);
  return result;
}

Eigen::VectorXi WithRespectToDamping::getDofsMapping(dynamics::Skeleton* skel)
{
  std::vector<WrtDampingJointEntry>& skelEntries = mEntries[skel->getName()];
  if (skelEntries.size() == 0)
    return Eigen::VectorXi::Zero(0);
  int cursor = 0;
  int skelDim = dim(skel);
  Eigen::VectorXi result = Eigen::VectorXi::Zero(skelDim);
  for (WrtDampingJointEntry& entry : skelEntries)
  {
    result.segment(cursor, entry.dim()) = entry.getDofMapping();
    cursor += entry.dim();
  }
  assert(cursor == skelDim);
  return result;
}

} // namespace neural
} // namespace dart