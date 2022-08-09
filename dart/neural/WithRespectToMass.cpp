#include "dart/neural/WithRespectToMass.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace neural {

//==============================================================================
/// Basic constructor
WithRespectToMass::WithRespectToMass()
{
}

//==============================================================================
/// A printable name for this WRT object
std::string WithRespectToMass::name()
{
  return "MASS";
}

//==============================================================================
WrtMassBodyNodyEntry::WrtMassBodyNodyEntry(
    std::string linkName, WrtMassBodyNodeEntryType type)
  : linkName(linkName), type(type)
{
}

//==============================================================================
int WrtMassBodyNodyEntry::dim()
{
  if (type == INERTIA_MASS)
    return 1;
  if (type == INERTIA_COM)
    return 3;
  if (type == INERTIA_COM_MU)
    return 1;
  if (type == INERTIA_DIAGONAL)
    return 3;
  if (type == INERTIA_OFF_DIAGONAL)
    return 3;
  if (type == INERTIA_FULL)
    return 10;
  assert(false);
  return 0;
}

//==============================================================================
void WrtMassBodyNodyEntry::set(
    dynamics::Skeleton* skel, const Eigen::Ref<Eigen::VectorXs>& value)
{
  dynamics::BodyNode* node = skel->getBodyNode(linkName);
  if (type == INERTIA_MASS)
  {
    node->setMass(value(0));
    return;
  }

  const dynamics::Inertia& inertia = node->getInertia();
  if (type == INERTIA_COM)
  {
    dynamics::Inertia newInertia(
        inertia.getParameter(dynamics::Inertia::Param::MASS),
        value(0), // COM_X
        value(1), // COM_Y
        value(2), // COM_Z
        inertia.getParameter(dynamics::Inertia::Param::I_XX),
        inertia.getParameter(dynamics::Inertia::Param::I_YY),
        inertia.getParameter(dynamics::Inertia::Param::I_ZZ),
        inertia.getParameter(dynamics::Inertia::Param::I_XY),
        inertia.getParameter(dynamics::Inertia::Param::I_XZ),
        inertia.getParameter(dynamics::Inertia::Param::I_YZ));
    node->setInertia(newInertia);
  }
  if (type == INERTIA_COM_MU)
  {
    Eigen::Vector3s beta = node->getBeta();
    dynamics::Inertia newInertia(
        inertia.getParameter(dynamics::Inertia::Param::MASS),
        beta(0) * value(0), // COM_X
        beta(1) * value(0), // COM_Y
        beta(2) * value(0), // COM_Z
        inertia.getParameter(dynamics::Inertia::Param::I_XX),
        inertia.getParameter(dynamics::Inertia::Param::I_YY),
        inertia.getParameter(dynamics::Inertia::Param::I_ZZ),
        inertia.getParameter(dynamics::Inertia::Param::I_XY),
        inertia.getParameter(dynamics::Inertia::Param::I_XZ),
        inertia.getParameter(dynamics::Inertia::Param::I_YZ));
    node->setInertia(newInertia);
  }
  else if (type == INERTIA_DIAGONAL)
  {
    dynamics::Inertia newInertia(
        inertia.getParameter(dynamics::Inertia::Param::MASS),
        inertia.getParameter(dynamics::Inertia::Param::COM_X),
        inertia.getParameter(dynamics::Inertia::Param::COM_Y),
        inertia.getParameter(dynamics::Inertia::Param::COM_Z),
        value(0), // I_XX
        value(1), // I_YY
        value(2), // I_ZZ
        inertia.getParameter(dynamics::Inertia::Param::I_XY),
        inertia.getParameter(dynamics::Inertia::Param::I_XZ),
        inertia.getParameter(dynamics::Inertia::Param::I_YZ));
    node->setInertia(newInertia);
  }
  else if (type == INERTIA_OFF_DIAGONAL)
  {
    dynamics::Inertia newInertia(
        inertia.getParameter(dynamics::Inertia::Param::MASS),
        inertia.getParameter(dynamics::Inertia::Param::COM_X),
        inertia.getParameter(dynamics::Inertia::Param::COM_Y),
        inertia.getParameter(dynamics::Inertia::Param::COM_Z),
        inertia.getParameter(dynamics::Inertia::Param::I_XX),
        inertia.getParameter(dynamics::Inertia::Param::I_YY),
        inertia.getParameter(dynamics::Inertia::Param::I_ZZ),
        value(0), // I_XY
        value(1), // I_XZ
        value(2)  // I_YZ
    );
    node->setInertia(newInertia);
  }
  else if (type == INERTIA_FULL)
  {
    dynamics::Inertia newInertia(
        value(0), // MASS,
        value(1), // COM_X,
        value(2), // COM_Y,
        value(3), // COM_Z,
        value(4), // I_XX,
        value(5), // I_YY,
        value(6), // I_ZZ,
        value(7), // I_XY
        value(8), // I_XZ
        value(9)  // I_YZ
    );
    node->setInertia(newInertia);
  }
}

//==============================================================================
void WrtMassBodyNodyEntry::get(
    dynamics::Skeleton* skel, Eigen::Ref<Eigen::VectorXs> out)
{
  dynamics::BodyNode* node = skel->getBodyNode(linkName);
  if (type == INERTIA_MASS)
  {
    out(0) = node->getMass();
    return;
  }

  if (type == INERTIA_COM)
  {
    out = node->getInertia().getLocalCOM();
    return;
  }
  if (type == INERTIA_COM_MU)
  {
    if (node->getBeta()(0) != 0)
      out(0) = node->getInertia().getLocalCOM()(0) / node->getBeta()(0);
    else if (node->getBeta()(1) != 0)
      out(0) = node->getInertia().getLocalCOM()(1) / node->getBeta()(1);
    else
      out(0) = node->getInertia().getLocalCOM()(2) / node->getBeta()(2);
    return;
  }

  Eigen::MatrixXs moment = node->getInertia().getMoment();
  if (type == INERTIA_DIAGONAL)
  {
    out(0) = moment(0, 0); // I_XX
    out(1) = moment(1, 1); // I_YY
    out(2) = moment(2, 2); // I_ZZ
  }
  else if (type == INERTIA_OFF_DIAGONAL)
  {
    out(0) = moment(0, 1); // I_XY
    out(1) = moment(0, 2); // I_XZ
    out(2) = moment(1, 2); // I_YZ
  }
  else if (type == INERTIA_FULL)
  {
    out(0) = node->getMass();
    out.segment(1, 3) = node->getInertia().getLocalCOM();
    out(4) = moment(0, 0); // I_XX
    out(5) = moment(1, 1); // I_YY
    out(6) = moment(2, 2); // I_ZZ
    out(7) = moment(0, 1); // I_XY
    out(8) = moment(0, 2); // I_XZ
    out(9) = moment(1, 2); // I_YZ
  }
}

//==============================================================================
/// This registers that we'd like to keep track of this node's mass in this
/// way in this differentiation
WrtMassBodyNodyEntry& WithRespectToMass::registerNode(
    dynamics::BodyNode* node,
    WrtMassBodyNodeEntryType type,
    Eigen::VectorXs upperBound,
    Eigen::VectorXs lowerBound)
{
  std::string skelName = node->getSkeleton()->getName();
  std::vector<WrtMassBodyNodyEntry>& skelEntries = mEntries[skelName];
  skelEntries.emplace_back(node->getName(), type);

  WrtMassBodyNodyEntry& entry = skelEntries[skelEntries.size() - 1];

  int dim = entry.dim();
  assert(
      dim == lowerBound.size()
      && "Lower bound must be same size as requested type would imply");
  assert(
      dim == upperBound.size()
      && "Upper bound must be same size as requested type would imply");

  assert(mUpperBounds.size() == mLowerBounds.size());

  // Append to the lower bound vector
  Eigen::VectorXs newMassLowerBound
      = Eigen::VectorXs::Zero(mLowerBounds.size() + dim);
  newMassLowerBound.segment(0, mLowerBounds.size()) = mLowerBounds;
  newMassLowerBound.segment(mLowerBounds.size(), dim) = lowerBound;
  mLowerBounds = newMassLowerBound;

  // Append to the upper bound vector
  Eigen::VectorXs newMassUpperBound
      = Eigen::VectorXs::Zero(mUpperBounds.size() + dim);
  newMassUpperBound.segment(0, mUpperBounds.size()) = mUpperBounds;
  newMassUpperBound.segment(mUpperBounds.size(), dim) = upperBound;
  mUpperBounds = newMassUpperBound;

  assert(mUpperBounds.size() == mLowerBounds.size());

  return entry;
}

//==============================================================================
/// This returns the entry object corresponding to this node
WrtMassBodyNodyEntry& WithRespectToMass::getNode(dynamics::BodyNode* node)
{
  std::string skelName = node->getSkeleton()->getName();
  std::vector<WrtMassBodyNodyEntry>& skelEntries = mEntries[skelName];
  for (WrtMassBodyNodyEntry& entry : skelEntries)
  {
    if (entry.linkName == node->getName())
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
Eigen::VectorXs WithRespectToMass::get(simulation::World* world)
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
Eigen::VectorXs WithRespectToMass::get(dynamics::Skeleton* skel)
{
  std::vector<WrtMassBodyNodyEntry>& skelEntries = mEntries[skel->getName()];
  if (skelEntries.size() == 0)
    return Eigen::VectorXs::Zero(0);
  int cursor = 0;
  int skelDim = dim(skel);
  Eigen::VectorXs result = Eigen::VectorXs::Zero(skelDim);
  for (WrtMassBodyNodyEntry& entry : skelEntries)
  {
    entry.get(skel, result.segment(cursor, entry.dim()));
    cursor += entry.dim();
  }
  assert(cursor == skelDim);
  return result;
}

//==============================================================================
/// This sets the world's state based on our WRT
void WithRespectToMass::set(simulation::World* world, Eigen::VectorXs value)
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
void WithRespectToMass::set(dynamics::Skeleton* skel, Eigen::VectorXs value)
{
  std::vector<WrtMassBodyNodyEntry>& skelEntries = mEntries[skel->getName()];
  if (skelEntries.size() == 0)
    return;
  int cursor = 0;
  for (WrtMassBodyNodyEntry& entry : skelEntries)
  {
    entry.set(skel, value.segment(cursor, entry.dim()));
    cursor += entry.dim();
  }
  assert(cursor == value.size());
}

//==============================================================================
/// This gives the dimensions of the WRT
int WithRespectToMass::dim(simulation::World* world)
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
int WithRespectToMass::dim(dynamics::Skeleton* skel)
{
  std::vector<WrtMassBodyNodyEntry>& skelEntries = mEntries[skel->getName()];
  int skelDim = 0;
  for (WrtMassBodyNodyEntry& entry : skelEntries)
  {
    skelDim += entry.dim();
  }
  return skelDim;
}

/// This gives a vector of upper bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToMass::upperBound(simulation::World* /* world */)
{
  return mUpperBounds;
}

/// This gives a vector of lower bound values for this WRT, given state in the
/// world
Eigen::VectorXs WithRespectToMass::lowerBound(simulation::World* /* world */)
{
  return mLowerBounds;
}

} // namespace neural
} // namespace dart