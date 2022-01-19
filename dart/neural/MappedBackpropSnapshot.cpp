#include "dart/neural/MappedBackpropSnapshot.hpp"

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/math/FiniteDifference.hpp"

// Make production builds happy with asserts
#define _unused(x) ((void)(x))

#define LOG_PERFORMANCE_MAPPED_BACKPROP_SNAPSHOT ;

using namespace dart;
using namespace performance;
using namespace math;

namespace dart {
namespace neural {

//==============================================================================
MappedBackpropSnapshot::MappedBackpropSnapshot(
    std::shared_ptr<BackpropSnapshot> backpropSnapshot,
    std::unordered_map<std::string, std::shared_ptr<Mapping>> mappings,
    std::unordered_map<std::string, PreStepMapping> preStepMappings,
    std::unordered_map<std::string, PostStepMapping> postStepMappings)
  : mBackpropSnapshot(backpropSnapshot),
    mMappings(mappings),
    mPreStepMappings(preStepMappings),
    mPostStepMappings(postStepMappings)
{
  for (auto pair : postStepMappings)
    mMappingsSet.push_back(pair.first);
}

//==============================================================================
const std::vector<std::string>& MappedBackpropSnapshot::getMappings()
{
  return mMappingsSet;
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getPosPosJacobian(
    std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getPosPosJacobian(world, perfLog);
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getPosVelJacobian(
    std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getPosVelJacobian(world, perfLog);
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getVelPosJacobian(
    std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getVelPosJacobian(world, perfLog);
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getVelVelJacobian(
    std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getVelVelJacobian(world, perfLog);
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getControlForceVelJacobian(
    std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getControlForceVelJacobian(world, perfLog);
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getMassVelJacobian(
    std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getMassVelJacobian(world, perfLog);
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getDampingVelJacobian(
    std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getDampingVelJacobian(world, perfLog);
}

//==============================================================================
const Eigen::MatrixXs& MappedBackpropSnapshot::getSpringVelJacobian(
  std::shared_ptr<simulation::World> world, PerformanceLog* perfLog)
{
  return mBackpropSnapshot->getSpringVelJacobian(world, perfLog);
}
//==============================================================================
Eigen::MatrixXs MappedBackpropSnapshot::getPosMappedPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXs jac
      = mPostStepMappings[mapAfter].posInJacWrtPos
            * mBackpropSnapshot->getPosPosJacobian(world, perfLog)
        + mPostStepMappings[mapAfter].posInJacWrtVel
              * mBackpropSnapshot->getPosVelJacobian(world, perfLog);
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXs fd = finiteDifferencePosPosJacobian(world, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "pos->mapped pos");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXs MappedBackpropSnapshot::getPosMappedVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXs jac
      = mPostStepMappings[mapAfter].velInJacWrtPos
            * mBackpropSnapshot->getPosPosJacobian(world, perfLog)
        + mPostStepMappings[mapAfter].velInJacWrtVel
              * mBackpropSnapshot->getPosVelJacobian(world, perfLog);
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXs fd = finiteDifferencePosVelJacobian(world, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "pos->mapped vel");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXs MappedBackpropSnapshot::getVelMappedPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXs jac
      = mPostStepMappings[mapAfter].posInJacWrtPos
            * mBackpropSnapshot->getVelPosJacobian(world, perfLog)
        + mPostStepMappings[mapAfter].posInJacWrtVel
              * mBackpropSnapshot->getVelVelJacobian(world, perfLog);
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXs fd = finiteDifferenceVelPosJacobian(world, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "vel->mapped pos");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXs MappedBackpropSnapshot::getVelMappedVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXs jac
      = mPostStepMappings[mapAfter].velInJacWrtPos
            * mBackpropSnapshot->getVelPosJacobian(world, perfLog)
        + mPostStepMappings[mapAfter].velInJacWrtVel
              * mBackpropSnapshot->getVelVelJacobian(world, perfLog);
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXs fd = finiteDifferenceVelVelJacobian(world, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "vel->mapped vel");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXs MappedBackpropSnapshot::getControlForceMappedVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXs jac
      = mPostStepMappings[mapAfter].velInJacWrtVel
        * mBackpropSnapshot->getControlForceVelJacobian(world, perfLog);
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXs fd = finiteDifferenceForceVelJacobian(world, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "force->mapped vel");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXs MappedBackpropSnapshot::getMassMappedVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXs jac = mPostStepMappings[mapAfter].velInJacWrtVel
                        * mBackpropSnapshot->getMassVelJacobian(world, perfLog);
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXs fd = finiteDifferenceForceVelJacobian(world, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "mass->mapped vel");
  }
  return jac;
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
    const std::unordered_map<std::string, LossGradient> nextTimestepLosses,
    PerformanceLog* perfLog,
    bool exploreAlternateStrategies)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MAPPED_BACKPROP_SNAPSHOT
  if (perfLog != nullptr)
  {
    thisLog = perfLog->startRun("MappedBackpropSnapshot.backprop");
  }
#endif

  /*
  thisTimestepLoss.lossWrtPosition
      = Eigen::VectorXs::Zero(mMappings[mRepresentation]->getPosDim());
  thisTimestepLoss.lossWrtVelocity
      = Eigen::VectorXs::Zero(mMappings[mRepresentation]->getVelDim());
  thisTimestepLoss.lossWrtTorque
      = Eigen::VectorXs::Zero(mMappings[mRepresentation]->getControlForceDim());
  thisTimestepLoss.lossWrtMass
      = Eigen::VectorXs::Zero(mMappings[mRepresentation]->getMassDim());

  // Cleaner, slower way to compute backprop
  for (auto pair : nextTimestepLosses)
  {
    const std::string& mapAfter = pair.first;
    LossGradient& nextTimestepLoss = pair.second;

    const Eigen::MatrixXs posPos
        = getPosPosJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXs posVel
        = getPosVelJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXs velPos
        = getVelPosJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXs velVel
        = getVelVelJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXs forceVel
        = getControlForceVelJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXs massVel
        = getMassVelJacobian(world, mRepresentation, mapAfter);

    thisTimestepLoss.lossWrtPosition
        += posPos.transpose() * nextTimestepLoss.lossWrtPosition
           + posVel.transpose() * nextTimestepLoss.lossWrtVelocity;
    thisTimestepLoss.lossWrtVelocity
        += velPos.transpose() * nextTimestepLoss.lossWrtPosition
           + velVel.transpose() * nextTimestepLoss.lossWrtVelocity;
    thisTimestepLoss.lossWrtTorque
        += forceVel.transpose() * nextTimestepLoss.lossWrtVelocity;
    thisTimestepLoss.lossWrtMass
        += massVel.transpose() * nextTimestepLoss.lossWrtVelocity;
  }
  */

  // Faster, but not obviously correct way
  LossGradient nextTimestepRealLoss;
  nextTimestepRealLoss.lossWrtPosition
      = Eigen::VectorXs::Zero(world->getNumDofs());
  nextTimestepRealLoss.lossWrtVelocity
      = Eigen::VectorXs::Zero(world->getNumDofs());
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
  mBackpropSnapshot->backprop(
      world,
      thisTimestepLoss,
      nextTimestepRealLoss,
      thisLog,
      exploreAlternateStrategies);

#ifdef LOG_PERFORMANCE_MAPPED_BACKPROP_SNAPSHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' position()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, BEFORE the timestep.
const Eigen::VectorXs& MappedBackpropSnapshot::getPreStepPosition(
    const std::string& mapping)
{
  return mPreStepMappings[mapping].pos;
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' velocity()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, BEFORE the timestep.
const Eigen::VectorXs& MappedBackpropSnapshot::getPreStepVelocity(
    const std::string& mapping)
{
  return mPreStepMappings[mapping].vel;
}

//==============================================================================
/// Returns a concatenated vector of all the joint torques that were applied
/// during the forward pass, BEFORE the timestep.
const Eigen::VectorXs& MappedBackpropSnapshot::getPreStepTorques(
    const std::string& mapping)
{
  return mPreStepMappings[mapping].force;
}

//==============================================================================
/// Returns the LCP's cached solution from before the step
const Eigen::VectorXs& MappedBackpropSnapshot::getPreStepLCPCache()
{
  return mBackpropSnapshot->mPreStepLCPCache;
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' position()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, AFTER the timestep.
const Eigen::VectorXs& MappedBackpropSnapshot::getPostStepPosition(
    const std::string& mapping)
{
  return mPostStepMappings[mapping].pos;
}

//==============================================================================
/// Returns a concatenated vector of all the Skeletons' velocity()'s in the
/// World, in order in which the Skeletons appear in the World's
/// getSkeleton(i) returns them, AFTER the timestep.
const Eigen::VectorXs& MappedBackpropSnapshot::getPostStepVelocity(
    const std::string& mapping)
{
  return mPostStepMappings[mapping].vel;
}

//==============================================================================
/// This computes and returns the whole vel-vel jacobian by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXs MappedBackpropSnapshot::finiteDifferenceVelVelJacobian(
    simulation::WorldPtr world, const std::string& mapAfter, bool useRidders)
{
  RestorableSnapshot snapshot(world);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getVelDim();
  Eigen::MatrixXs result(outDim, inDim);
  s_t eps = useRidders ? 1e-4 : 1e-7;
  try
  {
    finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        world->setPositions(mBackpropSnapshot->mPreStepPosition);
        world->setControlForces(mBackpropSnapshot->mPreStepTorques);
        world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
        Eigen::VectorXs tweakedVel = 
            Eigen::VectorXs(mBackpropSnapshot->mPreStepVelocity);
        tweakedVel(dof) += eps;
        world->setVelocities(tweakedVel);
        BackpropSnapshotPtr snapshot = neural::forwardPass(world, false);
        perturbed = mMappings[mapAfter]->getVelocities(world);
        return (!mBackpropSnapshot->areResultsStandardized() 
                  || snapshot->areResultsStandardized())
               && snapshot->getNumClamping() 
                  == mBackpropSnapshot->getNumClamping()
               && snapshot->getNumUpperBound() 
                  == mBackpropSnapshot->getNumUpperBound();
      },
      result,
      eps,
      useRidders);
    snapshot.restore();
    world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
    return result;
  }
  catch(const std::exception& e)
  {
    std::cout << "Error in finiteDifferenceVelVelJacobian(): " 
        << e.what() << std::endl;
    throw e;
  }
}

//==============================================================================
/// This computes and returns the whole pos-C(pos,vel) jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXs MappedBackpropSnapshot::finiteDifferencePosVelJacobian(
    simulation::WorldPtr world, const std::string& mapAfter, bool useRidders)
{
  RestorableSnapshot snapshot(world);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);
  bool oldPenetrationCorrectionEnabled
      = world->getPenetrationCorrectionEnabled();
  world->setPenetrationCorrectionEnabled(false);

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getVelDim();
  Eigen::MatrixXs result(outDim, inDim);
  s_t eps = useRidders ? 1e-4 : 1e-7;
  try
  {
    finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        world->setControlForces(mBackpropSnapshot->mPreStepTorques);
        world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
        world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
        Eigen::VectorXs tweakedPos =
            Eigen::VectorXs(mBackpropSnapshot->mPreStepPosition);
        tweakedPos(dof) += eps;
        world->setPositions(tweakedPos);
        BackpropSnapshotPtr snapshot = neural::forwardPass(world, false);
        perturbed = mMappings[mapAfter]->getVelocities(world);
        return (!mBackpropSnapshot->areResultsStandardized() 
                  || snapshot->areResultsStandardized())
               && snapshot->getNumClamping() 
                  == mBackpropSnapshot->getNumClamping()
               && snapshot->getNumUpperBound() 
                  == mBackpropSnapshot->getNumUpperBound();
      },
      result,
      eps,
      useRidders);
    snapshot.restore();
    world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
    world->setPenetrationCorrectionEnabled(oldPenetrationCorrectionEnabled);
    return result;
  }
  catch(const std::exception& e)
  {
    std::cout << "Error in finiteDifferencePosVelJacobian(): " 
        << e.what() << std::endl;
    throw e;
  }
}

//==============================================================================
/// This computes and returns the whole force-vel jacobian by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXs MappedBackpropSnapshot::finiteDifferenceForceVelJacobian(
    simulation::WorldPtr world, const std::string& mapAfter, bool useRidders)
{
  RestorableSnapshot snapshot(world);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getVelDim();
  Eigen::MatrixXs result(outDim, inDim);
  s_t eps = useRidders ? 1e-4 : 1e-7;
  try
  {
    finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        world->setPositions(mBackpropSnapshot->mPreStepPosition);
        world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
        world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
        Eigen::VectorXs tweakedForces
            = Eigen::VectorXs(mBackpropSnapshot->mPreStepTorques);
        tweakedForces(dof) += eps;
        world->setControlForces(tweakedForces);
        BackpropSnapshotPtr snapshot = neural::forwardPass(world, false);
        perturbed = mMappings[mapAfter]->getVelocities(world);
        return (!mBackpropSnapshot->areResultsStandardized() 
                  || snapshot->areResultsStandardized())
               && snapshot->getNumClamping() 
                  == mBackpropSnapshot->getNumClamping()
               && snapshot->getNumUpperBound() 
                  == mBackpropSnapshot->getNumUpperBound();
      },
      result,
      eps,
      useRidders);
    snapshot.restore();
    world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
    return result;
  }
  catch(const std::exception& e)
  {
    std::cout << "Error in finiteDifferenceForceVelJacobian(): " 
        << e.what() << std::endl;
    throw e;
  }
}

//==============================================================================
/// This computes a finite differenced Jacobian for pos_t->mapped_pos_{t+1}
Eigen::MatrixXs MappedBackpropSnapshot::finiteDifferencePosPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapAfter,
    std::size_t subdivisions,
    bool useRidders)
{
  RestorableSnapshot snapshot(world);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getPosDim();
  Eigen::MatrixXs result(outDim, inDim);
  s_t eps = useRidders ? 1e-3 / subdivisions : 
      ((subdivisions > 1) ? (1e-2 / subdivisions) : 1e-6);
  s_t oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
    finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
        world->setControlForces(mBackpropSnapshot->mPreStepTorques);
        world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
        Eigen::VectorXs tweakedPos
            = Eigen::VectorXs(mBackpropSnapshot->mPreStepPosition);
        tweakedPos(dof) += eps;
        world->setPositions(tweakedPos);
        for (std::size_t j = 0; j < subdivisions; j++)
          world->step(false);
        perturbed = mMappings[mapAfter]->getPositions(world);
        return true;
      },
      result,
      eps,
      useRidders);
    world->setTimeStep(oldTimestep);
    world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
    snapshot.restore();
    return result;
}

//==============================================================================
/// This computes and returns the whole vel-pos jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXs MappedBackpropSnapshot::finiteDifferenceVelPosJacobian(
    simulation::WorldPtr world,
    const std::string& mapAfter,
    std::size_t subdivisions,
    bool useRidders)
{
  RestorableSnapshot snapshot(world);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getPosDim();
  Eigen::MatrixXs result(outDim, inDim);
  s_t eps = useRidders ? 1e-3 / subdivisions : 
      ((subdivisions > 1) ? (1e-2 / subdivisions) : 1e-6);
  s_t oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  finiteDifference(
    [&](/* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed) {
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setControlForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXs tweakedVel
          = Eigen::VectorXs(mBackpropSnapshot->mPreStepVelocity);
      tweakedVel(dof) += eps;
      world->setVelocities(tweakedVel);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      perturbed = mMappings[mapAfter]->getPositions(world);
      return true;
    },
    result,
    eps,
    useRidders);
  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();
  return result;
}

//==============================================================================
/// Returns the underlying BackpropSnapshot, without the mappings
std::shared_ptr<BackpropSnapshot>
MappedBackpropSnapshot::getUnderlyingSnapshot()
{
  return mBackpropSnapshot;
}

} // namespace neural
} // namespace dart