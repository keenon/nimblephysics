#include "dart/neural/MappedBackpropSnapshot.hpp"

#include "dart/neural/BackpropSnapshot.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
MappedBackpropSnapshot::MappedBackpropSnapshot(
    std::shared_ptr<BackpropSnapshot> backpropSnapshot,
    PreStepMapping preStepRepresentation,
    PostStepMapping postStepRepresentation,
    std::unordered_map<std::string, PreStepMapping> preStepLosses,
    std::unordered_map<std::string, PostStepMapping> postStepLosses)
  : mBackpropSnapshot(backpropSnapshot),
    mPreStepRepresentation(preStepRepresentation),
    mPostStepRepresentation(postStepRepresentation),
    mPreStepLosses(preStepLosses),
    mPostStepLosses(postStepLosses)
{
}

Eigen::MatrixXd MappedBackpropSnapshot::getPosPosJacobian(
    std::shared_ptr<simulation::World> world)
{
  return mPostStepRepresentation.posInJacWrtPos
             * mBackpropSnapshot->getPosPosJacobian(world)
             * mPreStepRepresentation.posOutJac
         + mPostStepRepresentation.posInJacWrtVel
               * mBackpropSnapshot->getPosVelJacobian(world)
               * mPreStepRepresentation.posOutJac;
}

Eigen::MatrixXd MappedBackpropSnapshot::getPosVelJacobian(
    std::shared_ptr<simulation::World> world)
{
  return mPostStepRepresentation.velInJacWrtVel
             * mBackpropSnapshot->getPosVelJacobian(world)
             * mPreStepRepresentation.posOutJac
         + mPostStepRepresentation.velInJacWrtPos
               * mBackpropSnapshot->getPosPosJacobian(world)
               * mPreStepRepresentation.posOutJac;
}

Eigen::MatrixXd MappedBackpropSnapshot::getVelPosJacobian(
    std::shared_ptr<simulation::World> world)
{
  return mPostStepRepresentation.posInJacWrtPos
             * mBackpropSnapshot->getVelPosJacobian(world)
             * mPreStepRepresentation.velOutJac
         + mPostStepRepresentation.posInJacWrtVel
               * mBackpropSnapshot->getVelVelJacobian(world)
               * mPreStepRepresentation.velOutJac;
}

Eigen::MatrixXd MappedBackpropSnapshot::getVelVelJacobian(
    std::shared_ptr<simulation::World> world)
{
  return mPostStepRepresentation.velInJacWrtVel
             * mBackpropSnapshot->getVelVelJacobian(world)
             * mPreStepRepresentation.velOutJac
         + mPostStepRepresentation.velInJacWrtPos
               * mBackpropSnapshot->getVelPosJacobian(world)
               * mPreStepRepresentation.velOutJac;
}

Eigen::MatrixXd MappedBackpropSnapshot::getForceVelJacobian(
    std::shared_ptr<simulation::World> world)
{
  return mPostStepRepresentation.velInJacWrtVel
         * mBackpropSnapshot->getForceVelJacobian(world)
         * mPreStepRepresentation.forceOutJac;
}

/// This computes the implicit backprop without forming intermediate
/// Jacobians. It takes a LossGradient with the position and velocity vectors
/// filled it, though the loss with respect to torque is ignored and can be
/// null. It returns a LossGradient with all three values filled in, position,
/// velocity, and torque.
void MappedBackpropSnapshot::backprop(
    simulation::WorldPtr world,
    LossGradient& thisTimestepLoss,
    const LossGradient& nextTimestepLoss)
{
  thisTimestepLoss.lossWrtPosition
      = Eigen::VectorXd(mPreStepRepresentation.pos.size());
  thisTimestepLoss.lossWrtVelocity
      = Eigen::VectorXd(mPreStepRepresentation.vel.size());
  thisTimestepLoss.lossWrtTorque
      = Eigen::VectorXd(mPreStepRepresentation.force.size());

  // TODO: replace with a factored view that passes down to backprop()

  const Eigen::MatrixXd& posPos = getPosPosJacobian(world);
  const Eigen::MatrixXd& posVel = getPosVelJacobian(world);
  const Eigen::MatrixXd& velPos = getVelPosJacobian(world);
  const Eigen::MatrixXd& velVel = getVelVelJacobian(world);
  const Eigen::MatrixXd& forceVel = getForceVelJacobian(world);

  thisTimestepLoss.lossWrtPosition
      = posPos.transpose() * nextTimestepLoss.lossWrtPosition
        + posVel.transpose() * nextTimestepLoss.lossWrtVelocity;
  thisTimestepLoss.lossWrtVelocity
      = velPos.transpose() * nextTimestepLoss.lossWrtPosition
        + velVel.transpose() * nextTimestepLoss.lossWrtVelocity;
  thisTimestepLoss.lossWrtTorque
      = forceVel.transpose() * nextTimestepLoss.lossWrtVelocity;
}

/// Returns a concatenated vector of all the Skeletons' position()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, BEFORE the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPreStepPosition()
{
  return mPreStepRepresentation.pos;
}

/// Returns a concatenated vector of all the Skeletons' velocity()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, BEFORE the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPreStepVelocity()
{
  return mPreStepRepresentation.vel;
}

/// Returns a concatenated vector of all the joint torques that were applied
/// during the forward pass, BEFORE the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPreStepTorques()
{
  return mPreStepRepresentation.force;
}

/// Returns a concatenated vector of all the Skeletons' position()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, AFTER the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPostStepPosition()
{
  return mPostStepRepresentation.pos;
}

/// Returns a concatenated vector of all the Skeletons' velocity()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, AFTER the timestep.
const Eigen::VectorXd& MappedBackpropSnapshot::getPostStepVelocity()
{
  return mPostStepRepresentation.vel;
}

/// Returns a concatenated vector of all the joint torques that were applied
/// during the forward pass, AFTER the timestep. This is necessarily identical
/// to getPreStepTorques(), since the timestep doesn't change the applied
/// forces.
const Eigen::VectorXd& MappedBackpropSnapshot::getPostStepTorques()
{
  return getPreStepTorques();
}

} // namespace neural
} // namespace dart