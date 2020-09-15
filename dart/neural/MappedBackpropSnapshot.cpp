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

} // namespace neural
} // namespace dart