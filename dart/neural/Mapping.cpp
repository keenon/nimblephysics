#include "dart/math/FiniteDifference.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
Mapping::Mapping()
{
}

//==============================================================================
Mapping::~Mapping()
{
}

//==============================================================================
/// Check if a Jacobian is equal
void Mapping::equalsOrCrash(
    Eigen::MatrixXs analytical,
    Eigen::MatrixXs bruteForce,
    std::shared_ptr<simulation::World> world,
    const std::string& name)
{
  Eigen::MatrixXs diff = bruteForce - analytical;
  double eps = 1e-8;
  if (diff.maxCoeff() > eps || diff.minCoeff() < -eps)
  {
    std::cout << "Mapping Jacobian disagrees on " << name << ": " << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "Brute Force: " << std::endl << bruteForce << std::endl;
    std::cout << "Diff (" << diff.minCoeff() << " - " << diff.maxCoeff()
              << "): " << std::endl
              << diff << std::endl;
    std::cout << "Code to replicate:" << std::endl;
    std::cout << "--------------------" << std::endl;
    int mNumDOFs = world->getNumDofs();
    std::cout << "Eigen::VectorXs brokenPos = Eigen::VectorXs::Zero("
              << mNumDOFs << ");" << std::endl;
    std::cout << "brokenPos <<" << std::endl;
    Eigen::VectorXs pos = world->getPositions();
    for (int i = 0; i < mNumDOFs; i++)
    {
      std::cout << "  " << pos(i);
      if (i == mNumDOFs - 1)
      {
        std::cout << ";" << std::endl;
      }
      else
      {
        std::cout << "," << std::endl;
      }
    }
    std::cout << "Eigen::VectorXs brokenVel = Eigen::VectorXs::Zero("
              << mNumDOFs << ");" << std::endl;
    std::cout << "brokenVel <<" << std::endl;
    Eigen::VectorXs vel = world->getVelocities();
    for (int i = 0; i < mNumDOFs; i++)
    {
      std::cout << "  " << vel(i);
      if (i == mNumDOFs - 1)
      {
        std::cout << ";" << std::endl;
      }
      else
      {
        std::cout << "," << std::endl;
      }
    }
    std::cout << "Eigen::VectorXs brokenForce = Eigen::VectorXs::Zero("
              << mNumDOFs << ");" << std::endl;
    std::cout << "brokenForce <<" << std::endl;
    Eigen::VectorXs force = world->getControlForces();
    for (int i = 0; i < mNumDOFs; i++)
    {
      std::cout << "  " << force(i);
      if (i == mNumDOFs - 1)
      {
        std::cout << ";" << std::endl;
      }
      else
      {
        std::cout << "," << std::endl;
      }
    }
    /*
    std::cout << "Eigen::VectorXs brokenLCPCache = Eigen::VectorXs::Zero("
              << mPreStepLCPCache.size() << ");" << std::endl;
    if (mPreStepLCPCache.size() > 0)
    {
      std::cout << "brokenLCPCache <<" << std::endl;
      for (int i = 0; i < mPreStepLCPCache.size(); i++)
      {
        std::cout << "  " << mPreStepLCPCache(i);
        if (i == mPreStepLCPCache.size() - 1)
        {
          std::cout << ";" << std::endl;
        }
        else
        {
          std::cout << "," << std::endl;
        }
      }
    }
    */

    std::cout << "world->setPositions(brokenPos);" << std::endl;
    std::cout << "world->setVelocities(brokenVel);" << std::endl;
    std::cout << "world->setControlForces(brokenForce);" << std::endl;
    std::cout << "world->setCachedLCPSolution(brokenLCPCache);" << std::endl;

    std::cout << "--------------------" << std::endl;
    exit(1);
  }
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealPosToMappedPosJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  RestorableSnapshot snapshot(world);
  Eigen::VectorXs originalPosWorld = world->getPositions();
  Eigen::MatrixXs result(getPosDim(), world->getNumDofs());

  s_t eps = useRidders ? 1e-4 : 1e-5;
  math::finiteDifference(
    [&](/* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed) {
      Eigen::VectorXs tweakedPos = originalPosWorld;
      tweakedPos(dof) += eps;
      world->setPositions(tweakedPos);
      perturbed = getPositions(world);
      return true;
    },
    result,
    eps,
    useRidders);
  snapshot.restore();
  return result;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealVelToMappedPosJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  RestorableSnapshot snapshot(world);
  Eigen::VectorXs originalVelWorld = world->getVelocities();
  Eigen::MatrixXs result(getPosDim(), world->getNumDofs());

  s_t eps = useRidders ? 1e-4 : 1e-5;
  math::finiteDifference(
    [&](/* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed) {
      Eigen::VectorXs tweakedVel = originalVelWorld;
      tweakedVel(dof) += eps;
      world->setVelocities(tweakedVel);
      perturbed = getPositions(world);
      return true;
    },
    result,
    eps,
    useRidders);
  snapshot.restore();
  return result;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealVelToMappedVelJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  RestorableSnapshot snapshot(world);
  Eigen::VectorXs originalVelWorld = world->getVelocities();
  Eigen::MatrixXs result(getVelDim(), world->getNumDofs());

  s_t eps = useRidders ? 1e-4 : 1e-5;
  math::finiteDifference(
    [&](/* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed) {
      Eigen::VectorXs tweakedVel = originalVelWorld;
      tweakedVel(dof) += eps;
      world->setVelocities(tweakedVel);
      perturbed = getVelocities(world);
      return true;
    },
    result,
    eps,
    useRidders);
  snapshot.restore();
  return result;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealPosToMappedVelJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  RestorableSnapshot snapshot(world);
  Eigen::VectorXs originalPosWorld = world->getPositions();
  Eigen::MatrixXs result(getVelDim(), world->getNumDofs());

  s_t eps = useRidders ? 1e-4 : 1e-5;
  math::finiteDifference(
    [&](/* in*/ s_t eps,
        /* in*/ int dof,
        /*out*/ Eigen::VectorXs& perturbed) {
      Eigen::VectorXs tweakedPos = originalPosWorld;
      tweakedPos(dof) += eps;
      world->setPositions(tweakedPos);
      perturbed = getVelocities(world);
      return true;
    },
    result,
    eps,
    useRidders);
  snapshot.restore();
  return result;
}

//==============================================================================
Eigen::VectorXs Mapping::getPositions(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs poses = Eigen::VectorXs::Zero(getPosDim());
  getPositionsInPlace(world, poses);
  return poses;
}

//==============================================================================
Eigen::VectorXs Mapping::getVelocities(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs vels = Eigen::VectorXs::Zero(getVelDim());
  getVelocitiesInPlace(world, vels);
  return vels;
}

//==============================================================================
Eigen::VectorXs Mapping::getControlForces(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs forces = Eigen::VectorXs::Zero(getControlForceDim());
  getControlForcesInPlace(world, forces);
  return forces;
}

//==============================================================================
Eigen::VectorXs Mapping::getMasses(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs masses = Eigen::VectorXs::Zero(getMassDim());
  getMassesInPlace(world, masses);
  return masses;
}

//==============================================================================
Eigen::VectorXs Mapping::getPositionLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getPosDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getPositionUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getPosDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getVelocityLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getVelDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getVelocityUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getVelDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getControlForceLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getControlForceDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getControlForceUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getControlForceDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getMassLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  // Non-zero lower bound by default
  return Eigen::VectorXs::Ones(getMassDim()) * 1e-7;
}

//==============================================================================
Eigen::VectorXs Mapping::getMassUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getMassDim())
         * std::numeric_limits<s_t>::infinity();
}

} // namespace neural
} // namespace dart