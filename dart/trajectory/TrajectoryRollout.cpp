#include "dart/trajectory/TrajectoryRollout.hpp"

#include "dart/trajectory/AbstractShot.hpp"

using namespace dart;

namespace dart {
namespace trajectory {

//==============================================================================
TrajectoryRollout::~TrajectoryRollout()
{
}

//==============================================================================
TrajectoryRolloutRef TrajectoryRollout::slice(int start, int len)
{
  return TrajectoryRolloutRef(this, start, len);
}

//==============================================================================
/// This returns a trajectory rollout ref, corresponding to a slice
/// of this trajectory rollout
const TrajectoryRolloutConstRef TrajectoryRollout::sliceConst(
    int start, int len) const
{
  return TrajectoryRolloutConstRef(this, start, len);
}

//==============================================================================
/// This returns a copy of the trajectory rollout
TrajectoryRollout* TrajectoryRollout::copy() const
{
  return new TrajectoryRolloutReal(this);
}

//==============================================================================
TrajectoryRolloutReal::TrajectoryRolloutReal(
    std::unordered_map<std::string, std::shared_ptr<neural::Mapping>> mappings,
    int steps,
    std::string representationMapping)
{
  mRepresentationMapping = representationMapping;
  for (auto pair : mappings)
  {
    mPoses[pair.first] = Eigen::MatrixXd::Zero(pair.second->getPosDim(), steps);
    mVels[pair.first] = Eigen::MatrixXd::Zero(pair.second->getVelDim(), steps);
    mForces[pair.first]
        = Eigen::MatrixXd::Zero(pair.second->getForceDim(), steps);
    mMappings.push_back(pair.first);
  }
}

//==============================================================================
TrajectoryRolloutReal::TrajectoryRolloutReal(AbstractShot* shot)
  : TrajectoryRolloutReal(
      shot->getMappings(), shot->getNumSteps(), shot->getRepresentationName())
{
}

//==============================================================================
const std::string& TrajectoryRolloutReal::getRepresentationMapping() const
{
  return mRepresentationMapping;
}

//==============================================================================
const std::vector<std::string>& TrajectoryRolloutReal::getMappings() const
{
  return mMappings;
}

//==============================================================================
/// Deep copy constructor
TrajectoryRolloutReal::TrajectoryRolloutReal(const TrajectoryRollout* copy)
{
  mRepresentationMapping = copy->getRepresentationMapping();
  mMappings = copy->getMappings();
  for (std::string key : copy->getMappings())
  {
    mPoses[key] = copy->getPosesConst(key);
    mVels[key] = copy->getVelsConst(key);
    mForces[key] = copy->getForcesConst(key);
  }
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutReal::getPoses(
    const std::string& mapping)
{
  return mPoses.at(mapping);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutReal::getVels(
    const std::string& mapping)
{
  return mVels.at(mapping);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutReal::getForces(
    const std::string& mapping)
{
  return mForces.at(mapping);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutReal::getPosesConst(
    const std::string& mapping) const
{
  return mPoses.at(mapping);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutReal::getVelsConst(
    const std::string& mapping) const
{
  return mVels.at(mapping);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutReal::getForcesConst(
    const std::string& mapping) const
{
  return mForces.at(mapping);
}

//==============================================================================
/// Slice constructor
TrajectoryRolloutRef::TrajectoryRolloutRef(
    TrajectoryRollout* toSlice, int start, int len)
  : mToSlice(toSlice), mStart(start), mLen(len)
{
}

//==============================================================================
const std::string& TrajectoryRolloutRef::getRepresentationMapping() const
{
  return mToSlice->getRepresentationMapping();
}

//==============================================================================
const std::vector<std::string>& TrajectoryRolloutRef::getMappings() const
{
  return mToSlice->getMappings();
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutRef::getPoses(
    const std::string& mapping)
{
  return mToSlice->getPoses(mapping).block(
      0, mStart, mToSlice->getPosesConst(mapping).rows(), mLen);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutRef::getVels(
    const std::string& mapping)
{
  return mToSlice->getVels(mapping).block(
      0, mStart, mToSlice->getVelsConst(mapping).rows(), mLen);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutRef::getForces(
    const std::string& mapping)
{
  return mToSlice->getForces(mapping).block(
      0, mStart, mToSlice->getForcesConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutRef::getPosesConst(
    const std::string& mapping) const
{
  return mToSlice->getPosesConst(mapping).block(
      0, mStart, mToSlice->getPosesConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutRef::getVelsConst(
    const std::string& mapping) const
{
  return mToSlice->getVelsConst(mapping).block(
      0, mStart, mToSlice->getVelsConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutRef::getForcesConst(
    const std::string& mapping) const
{
  return mToSlice->getForcesConst(mapping).block(
      0, mStart, mToSlice->getForcesConst(mapping).rows(), mLen);
}

//==============================================================================
/// Slice constructor
TrajectoryRolloutConstRef::TrajectoryRolloutConstRef(
    const TrajectoryRollout* toSlice, int start, int len)
  : mToSlice(toSlice), mStart(start), mLen(len)
{
}

//==============================================================================
const std::string& TrajectoryRolloutConstRef::getRepresentationMapping() const
{
  return mToSlice->getRepresentationMapping();
}

//==============================================================================
const std::vector<std::string>& TrajectoryRolloutConstRef::getMappings() const
{
  return mToSlice->getMappings();
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutConstRef::getPoses(
    const std::string& /* mapping */)
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutConstRef::getVels(
    const std::string& /* mapping */)
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutConstRef::getForces(
    const std::string& /* mapping */)
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd>
TrajectoryRolloutConstRef::getPosesConst(const std::string& mapping) const
{
  return mToSlice->getPosesConst(mapping).block(
      0, mStart, mToSlice->getPosesConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutConstRef::getVelsConst(
    const std::string& mapping) const
{
  return mToSlice->getVelsConst(mapping).block(
      0, mStart, mToSlice->getVelsConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd>
TrajectoryRolloutConstRef::getForcesConst(const std::string& mapping) const
{
  return mToSlice->getForcesConst(mapping).block(
      0, mStart, mToSlice->getForcesConst(mapping).rows(), mLen);
}

} // namespace trajectory
} // namespace dart