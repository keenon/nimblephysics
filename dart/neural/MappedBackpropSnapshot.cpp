#include "dart/neural/MappedBackpropSnapshot.hpp"

#include "dart/neural/BackpropSnapshot.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
MappedBackpropSnapshot::MappedBackpropSnapshot(
    std::shared_ptr<BackpropSnapshot> backpropSnapshot,
    std::string representation,
    std::unordered_map<std::string, PreStepMapping> preStepMappings,
    std::unordered_map<std::string, PostStepMapping> postStepMappings)
  : mBackpropSnapshot(backpropSnapshot),
    mRepresentation(representation),
    mPreStepMappings(preStepMappings),
    mPostStepMappings(postStepMappings)
{
  for (auto pair : postStepMappings)
    mMappings.push_back(pair.first);
}

//==============================================================================
const std::vector<std::string>& MappedBackpropSnapshot::getMappings()
{
  return mMappings;
}

//==============================================================================
const std::string& MappedBackpropSnapshot::getRepresentation()
{
  return mRepresentation;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getPosPosJacobian(
    std::shared_ptr<simulation::World> world, const std::string& mapping)
{
  return mPostStepMappings[mapping].posInJacWrtPos
             * mBackpropSnapshot->getPosPosJacobian(world)
             * mPreStepMappings[mapping].posOutJac
         + mPostStepMappings[mapping].posInJacWrtVel
               * mBackpropSnapshot->getPosVelJacobian(world)
               * mPreStepMappings[mapping].posOutJac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getPosVelJacobian(
    std::shared_ptr<simulation::World> world, const std::string& mapping)
{
  return mPostStepMappings[mapping].velInJacWrtVel
             * mBackpropSnapshot->getPosVelJacobian(world)
             * mPreStepMappings[mapping].posOutJac
         + mPostStepMappings[mapping].velInJacWrtPos
               * mBackpropSnapshot->getPosPosJacobian(world)
               * mPreStepMappings[mapping].posOutJac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getVelPosJacobian(
    std::shared_ptr<simulation::World> world, const std::string& mapping)
{
  return mPostStepMappings[mapping].posInJacWrtPos
             * mBackpropSnapshot->getVelPosJacobian(world)
             * mPreStepMappings[mapping].velOutJac
         + mPostStepMappings[mapping].posInJacWrtVel
               * mBackpropSnapshot->getVelVelJacobian(world)
               * mPreStepMappings[mapping].velOutJac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getVelVelJacobian(
    std::shared_ptr<simulation::World> world, const std::string& mapping)
{
  return mPostStepMappings[mapping].velInJacWrtVel
             * mBackpropSnapshot->getVelVelJacobian(world)
             * mPreStepMappings[mapping].velOutJac
         + mPostStepMappings[mapping].velInJacWrtPos
               * mBackpropSnapshot->getVelPosJacobian(world)
               * mPreStepMappings[mapping].velOutJac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getForceVelJacobian(
    std::shared_ptr<simulation::World> world, const std::string& mapping)
{
  return mPostStepMappings[mapping].velInJacWrtVel
         * mBackpropSnapshot->getForceVelJacobian(world)
         * mPreStepMappings[mapping].forceOutJac;
}

//==============================================================================
/// This computes the implicit backprop without forming intermediate
/// Jacobians. It takes a LossGradient with the position and velocity vectors
/// filled it, though the loss with respect to torque is ignored and can be
/// null. It returns a LossGradient with all three values filled in, position,
/// velocity, and torque.
void MappedBackpropSnapshot::backprop(
    simulation::WorldPtr world,
    LossGradient& thisTimestepLoss,
    const std::unordered_map<std::string, LossGradient> nextTimestepLosses)
{
  LossGradient nextTimestepRealLoss;
  nextTimestepRealLoss.lossWrtPosition
      = Eigen::VectorXd::Zero(world->getNumDofs());
  nextTimestepRealLoss.lossWrtVelocity
      = Eigen::VectorXd::Zero(world->getNumDofs());
  for (auto pair : nextTimestepLosses)
  {
    nextTimestepRealLoss.lossWrtPosition
        += mPostStepMappings[pair.first].posInJacWrtPos.transpose()
               * pair.second.lossWrtPosition
           + mPostStepMappings[pair.first].velInJacWrtPos.transpose()
                 * pair.second.lossWrtVelocity;
    nextTimestepRealLoss.lossWrtVelocity
        = mPostStepMappings[pair.first].posInJacWrtVel.transpose()
              * pair.second.lossWrtPosition
          + mPostStepMappings[pair.first].velInJacWrtVel.transpose()
                * pair.second.lossWrtVelocity;
  }
  LossGradient thisTimestepRealLoss;
  mBackpropSnapshot->backprop(
      world, thisTimestepRealLoss, nextTimestepRealLoss);

  thisTimestepLoss.lossWrtPosition
      = mPreStepMappings[mRepresentation].posOutJac.transpose()
        * thisTimestepRealLoss.lossWrtPosition;
  thisTimestepLoss.lossWrtVelocity
      = mPreStepMappings[mRepresentation].posOutJac.transpose()
        * thisTimestepRealLoss.lossWrtVelocity;
  thisTimestepLoss.lossWrtTorque
      = mPreStepMappings[mRepresentation].forceOutJac.transpose()
        * thisTimestepRealLoss.lossWrtTorque;
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' position()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, BEFORE the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPreStepPosition(
    const std::string& mapping)
{
  return mPreStepMappings[mapping].pos;
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' velocity()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, BEFORE the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPreStepVelocity(
    const std::string& mapping)
{
  return mPreStepMappings[mapping].vel;
}

//==============================================================================
/// Returns a concatenated vector of all the joint torques that were applied
/// during the forward pass, BEFORE the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPreStepTorques(
    const std::string& mapping)
{
  return mPreStepMappings[mapping].force;
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' position()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, AFTER the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPostStepPosition(
    const std::string& mapping)
{
  return mPostStepMappings[mapping].pos;
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' velocity()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, AFTER the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPostStepVelocity(
    const std::string& mapping)
{
  return mPostStepMappings[mapping].vel;
}

//==============================================================================
/// Returns a concatenated vector of all the joint torques that were applied
/// during the forward pass, AFTER the timestep. This is necessarily identical
/// to getPreStepTorques(), since the timestep doesn't change the applied
/// forces.
const Eigen::VectorXd& MappedBackpropSnapshot::getPostStepTorques(
    const std::string& mapping)
{
  return getPreStepTorques(mapping);
}

} // namespace neural
} // namespace dart