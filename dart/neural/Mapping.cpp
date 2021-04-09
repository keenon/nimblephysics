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
    Eigen::VectorXs force = world->getExternalForces();
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
    std::cout << "world->setExternalForces(brokenForce);" << std::endl;
    std::cout << "world->setCachedLCPSolution(brokenLCPCache);" << std::endl;

    std::cout << "--------------------" << std::endl;
    exit(1);
  }
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealPosToMappedPosJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRealPosToMappedPosJac(world);
  }
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getPositions();
  Eigen::VectorXs originalMapped = getPositions(world);
  int dofs = world->getNumDofs();
  int mappedDim = getPosDim();

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mappedDim, dofs);

  const s_t EPS = 1e-5;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += EPS;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldPos = getPositions(world);

    perturbed = originalWorld;
    perturbed(i) -= EPS;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getPositions(world);

    J.col(i) = (perturbedWorldPos - perturbedWorldNeg) / (2 * EPS);
  }

  snapshot.restore();
  return J;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRiddersRealPosToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getPositions();
  Eigen::VectorXs originalMapped = getPositions(world);
  int mappedDim = getPosDim();

  Eigen::MatrixXs J(mappedDim, world->getNumDofs());

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    // Find original FD column
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += originalStepSize;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldPos = getPositions(world);

    perturbed = originalWorld;
    perturbed(i) -= originalStepSize;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getPositions(world);

    tab[0][0]
        = (perturbedWorldPos - perturbedWorldNeg) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbed = originalWorld;
      perturbed(i) += originalStepSize;
      world->setPositions(perturbed);
      Eigen::VectorXs perturbedWorldPos = getPositions(world);

      perturbed = originalWorld;
      perturbed(i) -= originalStepSize;
      world->setPositions(perturbed);
      Eigen::VectorXs perturbedWorldNeg = getPositions(world);

      tab[0][iTab] = (perturbedWorldPos - perturbedWorldNeg) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
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

  snapshot.restore();
  return J;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealVelToMappedPosJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRealVelToMappedPosJac(world);
  }
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getVelocities();
  Eigen::VectorXs originalMapped = getPositions(world);
  int dofs = world->getNumDofs();
  int mappedDim = getPosDim();

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mappedDim, dofs);

  const s_t EPS = 1e-5;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += EPS;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldPos = getPositions(world);

    perturbed = originalWorld;
    perturbed(i) -= EPS;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getPositions(world);

    J.col(i) = (perturbedWorldPos - perturbedWorldNeg) / (2 * EPS);
  }

  snapshot.restore();
  return J;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRiddersRealVelToMappedPosJac(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getVelocities();
  Eigen::VectorXs originalMapped = getPositions(world);
  int mappedDim = getPosDim();

  Eigen::MatrixXs J(mappedDim, world->getNumDofs());

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    // Find original FD column
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += originalStepSize;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldPos = getPositions(world);

    perturbed = originalWorld;
    perturbed(i) -= originalStepSize;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getPositions(world);

    tab[0][0]
        = (perturbedWorldPos - perturbedWorldNeg) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbed = originalWorld;
      perturbed(i) += originalStepSize;
      world->setVelocities(perturbed);
      Eigen::VectorXs perturbedWorldPos = getPositions(world);

      perturbed = originalWorld;
      perturbed(i) -= originalStepSize;
      world->setVelocities(perturbed);
      Eigen::VectorXs perturbedWorldNeg = getPositions(world);

      tab[0][iTab] = (perturbedWorldPos - perturbedWorldNeg) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
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

  snapshot.restore();
  return J;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealVelToMappedVelJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRealVelToMappedVelJac(world);
  }
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getVelocities();
  Eigen::VectorXs originalMapped = getPositions(world);
  int dofs = world->getNumDofs();
  int mappedDim = getPosDim();

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mappedDim, dofs);

  const s_t EPS = 1e-5;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += EPS;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldPos = getVelocities(world);

    perturbed = originalWorld;
    perturbed(i) -= EPS;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getVelocities(world);

    J.col(i) = (perturbedWorldPos - perturbedWorldNeg) / (2 * EPS);
  }

  snapshot.restore();
  return J;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRiddersRealVelToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getVelocities();
  Eigen::VectorXs originalMapped = getPositions(world);
  int mappedDim = getPosDim();

  Eigen::MatrixXs J(mappedDim, world->getNumDofs());

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    // Find original FD column
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += originalStepSize;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldPos = getVelocities(world);

    perturbed = originalWorld;
    perturbed(i) -= originalStepSize;
    world->setVelocities(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getVelocities(world);

    tab[0][0]
        = (perturbedWorldPos - perturbedWorldNeg) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbed = originalWorld;
      perturbed(i) += originalStepSize;
      world->setVelocities(perturbed);
      Eigen::VectorXs perturbedWorldPos = getVelocities(world);

      perturbed = originalWorld;
      perturbed(i) -= originalStepSize;
      world->setVelocities(perturbed);
      Eigen::VectorXs perturbedWorldNeg = getVelocities(world);

      tab[0][iTab] = (perturbedWorldPos - perturbedWorldNeg) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
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

  snapshot.restore();
  return J;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRealPosToMappedVelJac(
    std::shared_ptr<simulation::World> world, bool useRidders)
{
  if (useRidders)
  {
    return finiteDifferenceRiddersRealPosToMappedVelJac(world);
  }
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getPositions();
  Eigen::VectorXs originalMapped = getPositions(world);
  int dofs = world->getNumDofs();
  int mappedDim = getPosDim();

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(mappedDim, dofs);

  const s_t EPS = 1e-5;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += EPS;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldPos = getVelocities(world);

    perturbed = originalWorld;
    perturbed(i) -= EPS;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getVelocities(world);

    J.col(i) = (perturbedWorldPos - perturbedWorldNeg) / (2 * EPS);
  }

  snapshot.restore();
  return J;
}

//==============================================================================
Eigen::MatrixXs Mapping::finiteDifferenceRiddersRealPosToMappedVelJac(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalWorld = world->getPositions();
  Eigen::VectorXs originalMapped = getPositions(world);
  int mappedDim = getPosDim();

  Eigen::MatrixXs J(mappedDim, world->getNumDofs());

  s_t originalStepSize = 1e-4;
  const s_t con = 1.4, con2 = (con * con);
  const s_t safeThreshold = 2.0;
  const int tabSize = 10;

  for (std::size_t i = 0; i < world->getNumDofs(); i++)
  {
    // Neville tableau of finite difference results
    std::array<std::array<Eigen::VectorXs, tabSize>, tabSize> tab;

    snapshot.restore();

    // Find original FD column
    Eigen::VectorXs perturbed = originalWorld;
    perturbed(i) += originalStepSize;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldPos = getVelocities(world);

    perturbed = originalWorld;
    perturbed(i) -= originalStepSize;
    world->setPositions(perturbed);
    Eigen::VectorXs perturbedWorldNeg = getVelocities(world);

    tab[0][0]
        = (perturbedWorldPos - perturbedWorldNeg) / (2 * originalStepSize);

    s_t stepSize = originalStepSize;
    s_t bestError = std::numeric_limits<s_t>::max();

    // Iterate over smaller and smaller step sizes
    for (int iTab = 1; iTab < tabSize; iTab++)
    {
      stepSize /= con;

      Eigen::VectorXs perturbed = originalWorld;
      perturbed(i) += originalStepSize;
      world->setPositions(perturbed);
      Eigen::VectorXs perturbedWorldPos = getVelocities(world);

      perturbed = originalWorld;
      perturbed(i) -= originalStepSize;
      world->setPositions(perturbed);
      Eigen::VectorXs perturbedWorldNeg = getVelocities(world);

      tab[0][iTab] = (perturbedWorldPos - perturbedWorldNeg) / (2 * stepSize);

      s_t fac = con2;
      // Compute extrapolations of increasing orders, requiring no new
      // evaluations
      for (int jTab = 1; jTab <= iTab; jTab++)
      {
        tab[jTab][iTab] = (tab[jTab - 1][iTab] * fac - tab[jTab - 1][iTab - 1])
                          / (fac - 1.0);
        fac = con2 * fac;
        s_t currError = max(
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

  snapshot.restore();
  return J;
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
Eigen::VectorXs Mapping::getForces(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs forces = Eigen::VectorXs::Zero(getForceDim());
  getForcesInPlace(world, forces);
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
Eigen::VectorXs Mapping::getForceLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getForceDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getForceUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getForceDim())
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