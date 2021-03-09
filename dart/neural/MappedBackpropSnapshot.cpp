#include "dart/neural/MappedBackpropSnapshot.hpp"

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"

// Make production builds happy with asserts
#define _unused(x) ((void)(x))

#define LOG_PERFORMANCE_MAPPED_BACKPROP_SNAPSHOT ;

using namespace dart;
using namespace performance;

namespace dart {
namespace neural {

//==============================================================================
MappedBackpropSnapshot::MappedBackpropSnapshot(
    std::shared_ptr<BackpropSnapshot> backpropSnapshot,
    std::string representation,
    std::unordered_map<std::string, std::shared_ptr<Mapping>> mappings,
    std::unordered_map<std::string, PreStepMapping> preStepMappings,
    std::unordered_map<std::string, PostStepMapping> postStepMappings)
  : mBackpropSnapshot(backpropSnapshot),
    mRepresentation(representation),
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
const std::string& MappedBackpropSnapshot::getRepresentation()
{
  return mRepresentation;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getPosPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXd jac
      = mPostStepMappings[mapAfter].posInJacWrtPos
            * mBackpropSnapshot->getPosPosJacobian(world, perfLog)
            * mPreStepMappings[mapBefore].posOutJac
        + mPostStepMappings[mapAfter].posInJacWrtVel
              * mBackpropSnapshot->getPosVelJacobian(world, perfLog)
              * mPreStepMappings[mapBefore].posOutJac;
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXd fd
        = finiteDifferencePosPosJacobian(world, mapBefore, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "Mapped pos-pos");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getPosVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXd jac
      = mPostStepMappings[mapAfter].velInJacWrtVel
            * mBackpropSnapshot->getPosVelJacobian(world, perfLog)
            * mPreStepMappings[mapBefore].posOutJac
        + mPostStepMappings[mapAfter].velInJacWrtPos
              * mBackpropSnapshot->getPosPosJacobian(world, perfLog)
              * mPreStepMappings[mapBefore].posOutJac;
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXd fd
        = finiteDifferencePosVelJacobian(world, mapBefore, mapAfter);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "Mapped pos-vel");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getVelPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXd jac
      = mPostStepMappings[mapAfter].posInJacWrtPos
            * mBackpropSnapshot->getVelPosJacobian(world, perfLog)
            * mPreStepMappings[mapBefore].velOutJac
        + mPostStepMappings[mapAfter].posInJacWrtVel
              * mBackpropSnapshot->getVelVelJacobian(world, perfLog)
              * mPreStepMappings[mapBefore].velOutJac;
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXd fd
        = finiteDifferenceVelPosJacobian(world, mapBefore, mapAfter, 1);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "Mapped vel-pos");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getVelVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXd jac
      = mPostStepMappings[mapAfter].velInJacWrtVel
            * mBackpropSnapshot->getVelVelJacobian(world, perfLog)
            * mPreStepMappings[mapBefore].velOutJac
        + mPostStepMappings[mapAfter].velInJacWrtPos
              * mBackpropSnapshot->getVelPosJacobian(world, perfLog)
              * mPreStepMappings[mapBefore].velOutJac;
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXd fd
        = finiteDifferenceVelVelJacobian(world, mapBefore, mapAfter);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "Mapped vel-vel");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getForceVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    PerformanceLog* perfLog)
{
  Eigen::MatrixXd jac = mPostStepMappings[mapAfter].velInJacWrtVel
                        * mBackpropSnapshot->getForceVelJacobian(world, perfLog)
                        * mPreStepMappings[mapBefore].forceOutJac;
  if (world->getSlowDebugResultsAgainstFD())
  {
    Eigen::MatrixXd fd
        = finiteDifferenceForceVelJacobian(world, mapBefore, mapAfter);
    mBackpropSnapshot->equalsOrCrash(world, jac, fd, "Mapped force-vel");
  }
  return jac;
}

//==============================================================================
Eigen::MatrixXd MappedBackpropSnapshot::getMassVelJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& /* mapBefore */,
    const std::string& mapAfter,
    PerformanceLog* /* perfLog */)
{
  // No pre-step mapping necessary, because mass doesn't support mappings
  int massDim = world->getMassDims();
  if (massDim == 0)
  {
    int velDim = mPostStepMappings[mapAfter].velInJacWrtVel.rows();
    return Eigen::MatrixXd::Zero(velDim, 0);
  }
  return mPostStepMappings[mapAfter].velInJacWrtVel
         * mBackpropSnapshot->getMassVelJacobian(world);
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
      = Eigen::VectorXd::Zero(mMappings[mRepresentation]->getPosDim());
  thisTimestepLoss.lossWrtVelocity
      = Eigen::VectorXd::Zero(mMappings[mRepresentation]->getVelDim());
  thisTimestepLoss.lossWrtTorque
      = Eigen::VectorXd::Zero(mMappings[mRepresentation]->getForceDim());
  thisTimestepLoss.lossWrtMass
      = Eigen::VectorXd::Zero(mMappings[mRepresentation]->getMassDim());

  // Cleaner, slower way to compute backprop
  for (auto pair : nextTimestepLosses)
  {
    const std::string& mapAfter = pair.first;
    LossGradient& nextTimestepLoss = pair.second;

    const Eigen::MatrixXd posPos
        = getPosPosJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXd posVel
        = getPosVelJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXd velPos
        = getVelPosJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXd velVel
        = getVelVelJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXd forceVel
        = getForceVelJacobian(world, mRepresentation, mapAfter);
    const Eigen::MatrixXd massVel
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
      world,
      thisTimestepRealLoss,
      nextTimestepRealLoss,
      thisLog,
      exploreAlternateStrategies);

  thisTimestepLoss.lossWrtPosition
      = mPreStepMappings[mRepresentation].posOutJac.transpose()
        * thisTimestepRealLoss.lossWrtPosition;
  thisTimestepLoss.lossWrtVelocity
      = mPreStepMappings[mRepresentation].velOutJac.transpose()
        * thisTimestepRealLoss.lossWrtVelocity;
  thisTimestepLoss.lossWrtTorque
      = mPreStepMappings[mRepresentation].forceOutJac.transpose()
        * thisTimestepRealLoss.lossWrtTorque;
  thisTimestepLoss.lossWrtMass
      = mPreStepMappings[mRepresentation].massOutJac.transpose()
        * thisTimestepRealLoss.lossWrtMass;

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

//==============================================================================
/// This computes and returns the whole vel-vel jacobian by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceVelVelJacobian(
    simulation::WorldPtr world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersVelVelJacobian(world, mapBefore, mapAfter);
  }
  RestorableSnapshot snapshot(world);

  int inDim = mMappings[mapBefore]->getVelDim();
  int outDim = mMappings[mapAfter]->getVelDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");

  Eigen::MatrixXd J(outDim, inDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd perturbedVelPos = mMappings[mapAfter]->getVelocities(world);
    Eigen::VectorXd perturbedVelNeg = mMappings[mapAfter]->getVelocities(world);

    double epsPos = EPSILON;
    while (true)
    {
      // Get predicted next vel
      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedVel
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      tweakedVel(i) += epsPos;
      world->setVelocities(tweakedVel);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, false);
      if ((!mBackpropSnapshot->areResultsStandardized()
           || ptr->areResultsStandardized())
          && ptr->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && ptr->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelPos = mMappings[mapAfter]->getVelocities(world);
        break;
      }
      epsPos *= 0.5;

      assert(std::abs(epsPos) > 1e-20);
    }

    double epsNeg = EPSILON;
    while (true)
    {
      // Get predicted next vel
      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedVel
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      tweakedVel(i) -= epsNeg;
      world->setVelocities(tweakedVel);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, false);
      if ((!mBackpropSnapshot->areResultsStandardized()
           || ptr->areResultsStandardized())
          && ptr->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && ptr->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelNeg = mMappings[mapAfter]->getVelocities(world);
        break;
      }
      epsNeg *= 0.5;

      assert(std::abs(epsNeg) > 1e-20);
    }

    J.col(i).noalias()
        = (perturbedVelPos - perturbedVelNeg) / (epsPos + epsNeg);
  }

  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();
  return J;
}

//==============================================================================
/// This computes and returns the whole vel-vel jacobian by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceRiddersVelVelJacobian(
    simulation::WorldPtr world,
    const std::string& mapBefore,
    const std::string& mapAfter)
{
  RestorableSnapshot snapshot(world);

  int inDim = mMappings[mapBefore]->getVelDim();
  int outDim = mMappings[mapAfter]->getVelDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");

  Eigen::MatrixXd J(outDim, inDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double originalStepSize = 1e-4;
  const double con = 1.4, con2 = (con * con);
  const double safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXd, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXd perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
    Eigen::VectorXd perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      bool plusGood, minusGood = false;
      // Get predicted next vel
      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedPlus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      perturbedPlus(i) += originalStepSize;
      world->setVelocities(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, false);

      if ((!mBackpropSnapshot->areResultsStandardized()
           || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotPlus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
        plusGood = true;
      }

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedMinus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      perturbedMinus(i) -= originalStepSize;
      world->setVelocities(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, false);

      if ((!mBackpropSnapshot->areResultsStandardized()
           || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotMinus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);
        minusGood = true;
      }
      if (plusGood && minusGood) break;

      originalStepSize *= 0.5;

      assert(std::abs(originalStepSize) > 1e-20);
    }

    tab[0][0] = (perturbedVelPlus - perturbedVelMinus) / (2 * originalStepSize);

    double stepSize = originalStepSize;
    double bestError = std::numeric_limits<double>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedPlus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      perturbedPlus(i) += stepSize;
      world->setVelocities(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, false);
      perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
      if (!((!mBackpropSnapshot->areResultsStandardized()
           || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotPlus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersVelVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedMinus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      perturbedMinus(i) -= stepSize;
      world->setVelocities(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, false);
      perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);
      if (!((!mBackpropSnapshot->areResultsStandardized()
           || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotMinus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersVelVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      tab[0][iTab] = (perturbedVelPlus - perturbedVelMinus) / (2 * stepSize);

      double fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        double currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();
  return J;
}

//==============================================================================
/// This computes and returns the whole pos-C(pos,vel) jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferencePosVelJacobian(
    simulation::WorldPtr world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersPosVelJacobian(world, mapBefore, mapAfter);
  }

  RestorableSnapshot snapshot(world);

  int inDim = mMappings[mapBefore]->getPosDim();
  int outDim = mMappings[mapAfter]->getVelDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");

  Eigen::MatrixXd J(outDim, inDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd perturbedVelPos = world->getVelocities();
    Eigen::VectorXd perturbedVelNeg = world->getVelocities();

    double epsPos = EPSILON;
    while (true)
    {
      // Get predicted next vel
      snapshot.restore();
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedPos
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      tweakedPos(i) += epsPos;
      world->setPositions(tweakedPos);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, false);
      if ((!mBackpropSnapshot->areResultsStandardized()
           || ptr->areResultsStandardized())
          && ptr->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && ptr->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelPos = mMappings[mapAfter]->getVelocities(world);
        break;
      }
      epsPos *= 0.5;

      assert(std::abs(epsPos) > 1e-20);
    }

    double epsNeg = EPSILON;
    while (true)
    {
      // Get predicted next vel
      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedPos
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      tweakedPos(i) -= epsNeg;
      world->setPositions(tweakedPos);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, false);
      if ((!mBackpropSnapshot->areResultsStandardized()
           || ptr->areResultsStandardized())
          && ptr->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && ptr->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelNeg = mMappings[mapAfter]->getVelocities(world);
        break;
      }
      epsNeg *= 0.5;

      assert(std::abs(epsNeg) > 1e-20);
    }

    J.col(i).noalias()
        = (perturbedVelPos - perturbedVelNeg) / (epsPos + epsNeg);
  }

  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the whole pos-C(pos,vel) jacobian by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceRiddersPosVelJacobian(
    simulation::WorldPtr world,
    const std::string& mapBefore,
    const std::string& mapAfter)
{
  RestorableSnapshot snapshot(world);

  int inDim = mMappings[mapBefore]->getVelDim();
  int outDim = mMappings[mapAfter]->getVelDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");

  Eigen::MatrixXd J(outDim, inDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double originalStepSize = 1e-4;
  const double con = 1.4, con2 = (con * con);
  const double safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXd, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXd perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
    Eigen::VectorXd perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      bool plusGood, minusGood = false;
      // Get predicted next vel
      snapshot.restore();
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedPlus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      perturbedPlus(i) += originalStepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, false);

      if ((!mBackpropSnapshot->areResultsStandardized()
           || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotPlus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
        plusGood = true;
      }

      snapshot.restore();
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedMinus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      perturbedMinus(i) -= originalStepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, false);

      if ((!mBackpropSnapshot->areResultsStandardized()
           || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotMinus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);
        minusGood = true;
      }
      if (plusGood && minusGood) break;

      originalStepSize *= 0.5;

      assert(std::abs(originalStepSize) > 1e-20);
    }

    tab[0][0] = (perturbedVelPlus - perturbedVelMinus) / (2 * originalStepSize);

    double stepSize = originalStepSize;
    double bestError = std::numeric_limits<double>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      snapshot.restore();
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedPlus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      perturbedPlus(i) += stepSize;
      world->setPositions(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, false);
      perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
      if (!((!mBackpropSnapshot->areResultsStandardized()
           || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotPlus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersPosVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      snapshot.restore();
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedMinus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      perturbedMinus(i) -= stepSize;
      world->setPositions(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, false);
      perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);
      if (!((!mBackpropSnapshot->areResultsStandardized()
           || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotMinus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersPosVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      tab[0][iTab] = (perturbedVelPlus - perturbedVelMinus) / (2 * stepSize);

      double fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        double currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();
  return J;
}

//==============================================================================
/// This computes and returns the whole force-vel jacobian by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceForceVelJacobian(
    simulation::WorldPtr world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersForceVelJacobian(world, mapBefore, mapAfter);
  }

  RestorableSnapshot snapshot(world);

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getVelDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");
  _unused(mapBefore);

  Eigen::MatrixXd J(outDim, inDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd perturbedVelPos = world->getVelocities();
    Eigen::VectorXd perturbedVelNeg = world->getVelocities();

    double epsPos = EPSILON;
    while (true)
    {
      // Get predicted next vel
      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedForces
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepTorques);
      tweakedForces(i) += epsPos;
      world->setExternalForces(tweakedForces);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, false);
      if ((!mBackpropSnapshot->areResultsStandardized()
           || ptr->areResultsStandardized())
          && ptr->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && ptr->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelPos = mMappings[mapAfter]->getVelocities(world);
        break;
      }
      epsPos *= 0.5;

      assert(std::abs(epsPos) > 1e-20);
    }

    double epsNeg = EPSILON;
    while (true)
    {
      // Get predicted next vel
      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedForces
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepTorques);
      tweakedForces(i) -= epsNeg;
      world->setExternalForces(tweakedForces);

      BackpropSnapshotPtr ptr = neural::forwardPass(world, false);
      if ((!mBackpropSnapshot->areResultsStandardized()
           || ptr->areResultsStandardized())
          && ptr->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && ptr->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelNeg = mMappings[mapAfter]->getVelocities(world);
        break;
      }
      epsNeg *= 0.5;

      assert(std::abs(epsNeg) > 1e-20);
    }

    J.col(i).noalias()
        = (perturbedVelPos - perturbedVelNeg) / (epsPos + epsNeg);
  }

  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the whole force-vel jacobian by finite
/// differences. This is SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceRiddersForceVelJacobian(
    simulation::WorldPtr world,
    const std::string& mapBefore,
    const std::string& mapAfter)
{
  RestorableSnapshot snapshot(world);

  int inDim = mMappings[mapBefore]->getVelDim();
  int outDim = mMappings[mapAfter]->getVelDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");

  Eigen::MatrixXd J(outDim, inDim);

  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
  world->step(false);

  Eigen::VectorXd originalVel = world->getVelocities();

  double originalStepSize = 1e-4;
  const double con = 1.4, con2 = (con * con);
  const double safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXd, tabSize>, tabSize> tab;

    snapshot.restore();

    Eigen::VectorXd perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
    Eigen::VectorXd perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);

    // Find largest original step size which doesn't change numClamping
    while (true)
    {
      bool plusGood, minusGood = false;
      // Get predicted next vel
      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedPlus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepTorques);
      perturbedPlus(i) += originalStepSize;
      world->setExternalForces(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, false);

      if ((!mBackpropSnapshot->areResultsStandardized()
           || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotPlus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
        plusGood = true;
      }

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedMinus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepTorques);
      perturbedMinus(i) -= originalStepSize;
      world->setExternalForces(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, false);

      if ((!mBackpropSnapshot->areResultsStandardized()
           || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotMinus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound())
      {
        perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);
        minusGood = true;
      }
      if (plusGood && minusGood) break;

      originalStepSize *= 0.5;

      assert(std::abs(originalStepSize) > 1e-20);
    }

    tab[0][0] = (perturbedVelPlus - perturbedVelMinus) / (2 * originalStepSize);

    double stepSize = originalStepSize;
    double bestError = std::numeric_limits<double>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedPlus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepTorques);
      perturbedPlus(i) += stepSize;
      world->setExternalForces(perturbedPlus);
      BackpropSnapshotPtr snapshotPlus = neural::forwardPass(world, false);
      perturbedVelPlus = mMappings[mapAfter]->getVelocities(world);
      if (!((!mBackpropSnapshot->areResultsStandardized()
           || snapshotPlus->areResultsStandardized())
          && snapshotPlus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotPlus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersForceVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd perturbedMinus
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepTorques);
      perturbedMinus(i) -= stepSize;
      world->setExternalForces(perturbedMinus);
      BackpropSnapshotPtr snapshotMinus = neural::forwardPass(world, false);
      perturbedVelMinus = mMappings[mapAfter]->getVelocities(world);
      if (!((!mBackpropSnapshot->areResultsStandardized()
           || snapshotMinus->areResultsStandardized())
          && snapshotMinus->getNumClamping() == mBackpropSnapshot->getNumClamping()
          && snapshotMinus->getNumUpperBound() == mBackpropSnapshot->getNumUpperBound()))
      {
        assert(false && "Lowering EPS in finiteDifferenceRiddersForceVelJacobian() "
                      "caused numClamping() or numUpperBound() to change.");
      }

      tab[0][iTab] = (perturbedVelPlus - perturbedVelMinus) / (2 * stepSize);

      double fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        double currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();
  return J;
}

//==============================================================================
/// This computes a finite differenced Jacobian for pos_t->mapped_pos_{t+1}
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferencePosPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    std::size_t subdivisions,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersPosPosJacobian(
      world, mapBefore, mapAfter, subdivisions);
  }

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getPosDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");
  _unused(mapBefore);

  RestorableSnapshot snapshot(world);

  double oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  Eigen::MatrixXd J(outDim, inDim);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXd originalPosition = mMappings[mapAfter]->getPositions(world);

  // IMPORTANT: EPSILON must be larger than the distance traveled in a single
  // subdivided timestep. Ideally much larger.
  double EPSILON = (subdivisions > 1) ? (1e-2 / subdivisions) : 1e-6;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();
    world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    Eigen::VectorXd tweakedPositions
        = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
    tweakedPositions(i) += EPSILON;
    world->setPositions(tweakedPositions);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);

    Eigen::VectorXd pos = mMappings[mapAfter]->getPositions(world);

    snapshot.restore();
    world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    tweakedPositions = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
    tweakedPositions(i) -= EPSILON;
    world->setPositions(tweakedPositions);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);

    Eigen::VectorXd neg = mMappings[mapAfter]->getPositions(world);

    J.col(i).noalias() = (pos - neg) / (2 * EPSILON);
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes a finite differenced Jacobian for pos_t->mapped_pos_{t+1}
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceRiddersPosPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    std::size_t subdivisions)
{
  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getPosDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");
  _unused(mapBefore);

  RestorableSnapshot snapshot(world);

  double oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  Eigen::MatrixXd J(outDim, inDim);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  const double originalStepSize = 1e-3 / subdivisions;
  const double con = 1.4, con2 = (con * con);
  const double safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    double stepSize = originalStepSize;
    double bestError = std::numeric_limits<double>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXd, tabSize>, tabSize> tab;

    snapshot.restore();
    world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    Eigen::VectorXd tweakedPositions
        = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
    tweakedPositions(i) += stepSize;
    world->setPositions(tweakedPositions);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXd pos = mMappings[mapAfter]->getPositions(world);

    snapshot.restore();
    world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    tweakedPositions = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
    tweakedPositions(i) -= stepSize;
    world->setPositions(tweakedPositions);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXd neg = mMappings[mapAfter]->getPositions(world);

    tab[0][0] = (pos - neg) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      snapshot.restore();
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedPositions
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      tweakedPositions(i) += stepSize;
      world->setPositions(tweakedPositions);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      Eigen::VectorXd pos = mMappings[mapAfter]->getPositions(world);

      snapshot.restore();
      world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      tweakedPositions = Eigen::VectorXd(mBackpropSnapshot->mPreStepPosition);
      tweakedPositions(i) -= stepSize;
      world->setPositions(tweakedPositions);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      Eigen::VectorXd neg = mMappings[mapAfter]->getPositions(world);

      tab[0][iTab] = (pos - neg) / (2 * stepSize);

      double fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        double currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the whole vel-pos jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceVelPosJacobian(
    simulation::WorldPtr world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    std::size_t subdivisions,
    bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersVelPosJacobian(
      world, mapBefore, mapAfter, subdivisions);
  }

  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getPosDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");
  _unused(mapBefore);

  RestorableSnapshot snapshot(world);

  double oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  Eigen::MatrixXd J(outDim, inDim);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  Eigen::VectorXd originalPosition = mMappings[mapAfter]->getPositions(world);

  // IMPORTANT: EPSILON must be larger than the distance traveled in a single
  // subdivided timestep. Ideally much larger.
  double EPSILON = (subdivisions > 1) ? (1e-2 / subdivisions) : 1e-6;
  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    snapshot.restore();
    world->setPositions(mBackpropSnapshot->mPreStepPosition);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    Eigen::VectorXd tweakedVelocities
        = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
    tweakedVelocities(i) += EPSILON;
    world->setVelocities(tweakedVelocities);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);

    Eigen::VectorXd pos = mMappings[mapAfter]->getPositions(world);

    snapshot.restore();
    world->setPositions(mBackpropSnapshot->mPreStepPosition);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    tweakedVelocities = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
    tweakedVelocities(i) -= EPSILON;
    world->setVelocities(tweakedVelocities);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);

    Eigen::VectorXd neg = mMappings[mapAfter]->getPositions(world);

    J.col(i).noalias() = (pos - neg) / (2 * EPSILON);
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
}

//==============================================================================
/// This computes and returns the whole vel-pos jacobian by finite
/// differences. This is SUPER SUPER SLOW, and is only here for testing.
Eigen::MatrixXd MappedBackpropSnapshot::finiteDifferenceRiddersVelPosJacobian(
    std::shared_ptr<simulation::World> world,
    const std::string& mapBefore,
    const std::string& mapAfter,
    std::size_t subdivisions)
{
  int inDim = world->getNumDofs();
  int outDim = mMappings[mapAfter]->getPosDim();

  // TODO: this needs to support non-identity mapIns
  assert(mapBefore == "identity" && "Non-identity map ins are currently not supported by finite differencing");
  _unused(mapBefore);

  RestorableSnapshot snapshot(world);

  double oldTimestep = world->getTimeStep();
  world->setTimeStep(oldTimestep / subdivisions);
  bool oldGradientEnabled = world->getConstraintSolver()->getGradientEnabled();
  world->getConstraintSolver()->setGradientEnabled(true);

  Eigen::MatrixXd J(outDim, inDim);

  world->setPositions(mBackpropSnapshot->mPreStepPosition);
  world->setVelocities(mBackpropSnapshot->mPreStepVelocity);
  world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
  world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);

  for (std::size_t j = 0; j < subdivisions; j++)
    world->step(false);

  const double originalStepSize = 1e-3 / subdivisions;
  const double con = 1.4, con2 = (con * con);
  const double safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    double stepSize = originalStepSize;
    double bestError = std::numeric_limits<double>::max();

    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXd, tabSize>, tabSize> tab;

    snapshot.restore();
    world->setPositions(mBackpropSnapshot->mPreStepPosition);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    Eigen::VectorXd tweakedVelocities
        = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
    tweakedVelocities(i) += stepSize;
    world->setVelocities(tweakedVelocities);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXd pos = mMappings[mapAfter]->getPositions(world);

    snapshot.restore();
    world->setPositions(mBackpropSnapshot->mPreStepPosition);
    world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
    world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
    tweakedVelocities = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
    tweakedVelocities(i) -= stepSize;
    world->setVelocities(tweakedVelocities);
    for (std::size_t j = 0; j < subdivisions; j++)
      world->step(false);
    Eigen::VectorXd neg = mMappings[mapAfter]->getPositions(world);

    tab[0][0] = (pos - neg) / (2 * stepSize);

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      Eigen::VectorXd tweakedVelocities
          = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      tweakedVelocities(i) += stepSize;
      world->setVelocities(tweakedVelocities);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      Eigen::VectorXd pos = mMappings[mapAfter]->getPositions(world);

      snapshot.restore();
      world->setPositions(mBackpropSnapshot->mPreStepPosition);
      world->setExternalForces(mBackpropSnapshot->mPreStepTorques);
      world->setCachedLCPSolution(mBackpropSnapshot->mPreStepLCPCache);
      tweakedVelocities = Eigen::VectorXd(mBackpropSnapshot->mPreStepVelocity);
      tweakedVelocities(i) -= stepSize;
      world->setVelocities(tweakedVelocities);
      for (std::size_t j = 0; j < subdivisions; j++)
        world->step(false);
      Eigen::VectorXd neg = mMappings[mapAfter]->getPositions(world);

      tab[0][iTab] = (pos - neg) / (2 * stepSize);

      double fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        double currError = std::max(
            (tab[jTab][iTab] - tab[jTab - 1][iTab]).array().abs().maxCoeff(),
            (tab[jTab][iTab] - tab[jTab - 1][iTab - 1])
                .array()
                .abs()
                .maxCoeff());
        if (currError < bestError)
        {
          bestError = currError;
          J.col(i).noalias() = tab[jTab][iTab];
        }
      }

      // If higher order is worse by a significant factor, quit early.
      if ((tab[iTab][iTab] - tab[iTab - 1][iTab - 1]).array().abs().maxCoeff()
          >= safeThreshold * bestError)
      {
        break;
      }
    }
  }

  world->setTimeStep(oldTimestep);
  world->getConstraintSolver()->setGradientEnabled(oldGradientEnabled);
  snapshot.restore();

  return J;
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