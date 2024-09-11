#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <IpAlgTypes.hpp>
#include <gtest/gtest.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/MarkerFixer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/DifferentiableExternalForce.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/AccelerationSmoother.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/MJCFExporter.hpp"
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

#define JACOBIAN_TESTS
// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

void applyExternalForces(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<int, Eigen::Vector6s> worldForces)
{
  skel->clearExternalForces();
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* body = skel->getBodyNode(i);
    if (worldForces.count(i))
    {
      body->setExtWrench(
          math::dAdT(body->getWorldTransform(), worldForces.at(i)));
    }
  }
}

bool testApplyWorldForces(std::shared_ptr<dynamics::Skeleton> skel)
{
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = Eigen::VectorXs::Zero(skel->getNumDofs());
  skel->setControlForces(originalTau);

  // Compute normal forward dynamics
  Eigen::Vector6s upwardsForce = Eigen::Vector6s::Unit(5) * 10;
  Eigen::Vector3s g = skel->getGravity();
  (void)g;
  Eigen::Vector3s totalForce = skel->getGravity() * skel->getMass();

  // skel->setGravity(Eigen::Vector3s::Zero());

  /*
  skel->getBodyNode(0)->setExtForce(
      upwardsForce, Eigen::Vector3s::Zero(), false, true);
  totalForce += upwardsForce;

  Eigen::Vector6s worldWrench = Eigen::Vector6s::Zero();
  worldWrench.tail<3>() = upwardsForce;
  Eigen::Vector6s localWrench
      = math::dAdT(skel->getBodyNode(0)->getWorldTransform(), worldWrench);

  Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(6, 3);
  compare.col(0) = skel->getBodyNode(0)->getExternalForceLocal();
  compare.col(1) = localWrench;
  compare.col(2) = skel->getBodyNode(0)->getExternalForceLocal() - localWrench;
  std::cout << "mFext - computed - diff" << std::endl << compare << std::endl;

  skel->getBodyNode(0)->setExtWrench(localWrench);
  */

  std::map<int, Eigen::Vector6s> worldForces;
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    worldForces[i] = upwardsForce;

    // auto* body = skel->getBodyNode(i);

    // TODO: compare
    // Eigen::Vector6s wrench
    //     = math::dAdT(body->getWorldTransform(), worldForces.at(i));

    // body->setExtForce(
    //     upwardsForce.tail<3>(), Eigen::Vector3s::Zero(), false, true);
    totalForce += upwardsForce.tail<3>();
  }
  Eigen::Vector3s expectedAcc = totalForce.tail<3>() / skel->getMass();

  applyExternalForces(skel, worldForces);
  skel->computeForwardDynamics();
  Eigen::Vector3s realAcc = skel->getCOMLinearAcceleration();

  if (!equals(realAcc, expectedAcc, 1e-8))
  {
    Eigen::Matrix3s compare = Eigen::Matrix3s::Zero();
    compare.col(0) = expectedAcc;
    compare.col(1) = realAcc;
    compare.col(2) = expectedAcc - realAcc;
    std::cout << "F/m - Acc - Diff" << std::endl << compare << std::endl;
    return false;
  }

  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  skel->setControlForces(originalTau);

  return true;
}

bool testForwardDynamicsFormula(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<int, Eigen::Vector6s> worldForces)
{
  (void)skel;
  (void)worldForces;
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = skel->getRandomPose();

  // Compute normal forward dynamics
  (void)worldForces;
  applyExternalForces(skel, worldForces);
  skel->setControlForces(originalTau);
  skel->computeForwardDynamics();
  skel->integrateVelocities(skel->getTimeStep());
  Eigen::VectorXs nextVel = skel->getVelocities();

  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  skel->setControlForces(originalTau);

  // Compute analytical forward dynamics
  Eigen::MatrixXs Minv = skel->getInvMassMatrix();
  Eigen::VectorXs tau = skel->getControlForces();
  Eigen::VectorXs C = skel->getCoriolisAndGravityForces();
  s_t dt = skel->getTimeStep();

  Eigen::VectorXs preSolveV = originalVel + dt * Minv * (tau - C);
  Eigen::VectorXs f_cDeltaV = Eigen::VectorXs::Zero(preSolveV.size());

  Eigen::MatrixXi parents = skel->getDofParentMap();

  for (auto& pair : worldForces)
  {
    DifferentiableExternalForce force(skel, pair.first);
    Eigen::VectorXs fTaus = force.computeTau(pair.second);

#ifndef NDEBUG
    dynamics::BodyNode* body = skel->getBodyNode(pair.first);
    std::vector<int> parentDofs;
    for (int i = 0; i < body->getParentJoint()->getNumDofs(); i++)
    {
      parentDofs.push_back(
          body->getParentJoint()->getDof(i)->getIndexInSkeleton());
    }
    Eigen::VectorXs rawTaus = Eigen::VectorXs::Zero(skel->getNumDofs());
    for (int i = 0; i < skel->getNumDofs(); i++)
    {
      bool isParent = false;
      for (int j : parentDofs)
      {
        if (parents(i, j) == 1 || i == j)
        {
          isParent = true;
          break;
        }
      }
      if (!isParent)
        continue;
      rawTaus(i) = DifferentiableContactConstraint::getWorldScrewAxisForForce(
                       skel->getDof(i))
                       .dot(pair.second);
    }
    if (!equals(rawTaus, fTaus, 1e-8))
    {
      std::cout << "Taus don't equal!" << std::endl;
    }
#endif

    f_cDeltaV += dt * Minv * fTaus;
  }
  Eigen::VectorXs postSolveV = preSolveV + f_cDeltaV;

  if (!equals(nextVel, postSolveV, 1e-8))
  {
    std::cout << "Failed to recover next V!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(nextVel.size(), 4);
    compare.col(0) = nextVel;
    compare.col(1) = postSolveV;
    compare.col(2) = nextVel - postSolveV;
    compare.col(3) = f_cDeltaV;
    std::cout << "Forward - Analytical - Diff - Delta V: " << std::endl
              << compare << std::endl;
    return false;
  }
  return true;
}

bool testInverseDynamicsFormula(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<int, Eigen::Vector6s> worldForces)
{
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = skel->getRandomPose();
  applyExternalForces(skel, worldForces);
  skel->setControlForces(originalTau);
  skel->computeForwardDynamics();
  Eigen::VectorXs acc = skel->getAccelerations();

  // Reset
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  applyExternalForces(skel, worldForces);
  skel->setControlForces(Eigen::VectorXs::Zero(skel->getNumDofs()));

  // ID method
  Eigen::VectorXs taus = skel->getInverseDynamics(acc);

  if (!equals(taus, originalTau, 1e-8))
  {
    std::cout << "Failed to recover ID forces!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(taus.size(), 3);
    compare.col(0) = originalTau;
    compare.col(1) = taus;
    compare.col(2) = originalTau - taus;
    std::cout << "Original - Recovered - Diff: " << std::endl
              << compare << std::endl;
    return false;
  }

  // Decomposed ID formula

  // Eigen::VectorXs nextV = originalVel + dt * Minv * (tau - C +
  // sum(J[i]*f[i]));
  //
  // Eigen::VectorXs (M * (nextV - originalVel) / dt) + C - Fs = tau;

  Eigen::MatrixXs M = skel->getMassMatrix();
  Eigen::VectorXs C = skel->getCoriolisAndGravityForces();
  Eigen::VectorXs Fs = Eigen::VectorXs::Zero(skel->getNumDofs());
  for (auto& pair : worldForces)
  {
    DifferentiableExternalForce force(skel, pair.first);
    Eigen::VectorXs fTaus = force.computeTau(pair.second);
    Fs += fTaus;
  }

  Eigen::VectorXs manualTau = M * acc + C - Fs;

  if (!equals(manualTau, originalTau, 1e-8))
  {
    std::cout << "Failed to recover ID forces with manual ID formula!"
              << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(taus.size(), 3);
    compare.col(0) = originalTau;
    compare.col(1) = manualTau;
    compare.col(2) = originalTau - manualTau;
    std::cout << "Original - Recovered - Diff: " << std::endl
              << compare << std::endl;
    return false;
  }

  return true;
}

bool testResidualAgainstID(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<int, Eigen::Vector6s> worldForces)
{
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = skel->getRandomPose();
  Eigen::Vector6s originalResidual = originalTau.head<6>();
  applyExternalForces(skel, worldForces);
  skel->setControlForces(originalTau);
  skel->computeForwardDynamics();
  Eigen::VectorXs acc = skel->getAccelerations();

  // Reset
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  applyExternalForces(skel, worldForces);
  skel->setControlForces(Eigen::VectorXs::Zero(skel->getNumDofs()));

  std::vector<int> forceBodies;
  Eigen::VectorXs concatForces = Eigen::VectorXs::Zero(worldForces.size() * 6);
  for (auto& pair : worldForces)
  {
    concatForces.segment<6>(forceBodies.size() * 6) = pair.second;
    forceBodies.push_back(pair.first);
  }

  ResidualForceHelper helper(skel, forceBodies);

  Eigen::Vector6s manualResidual
      = helper.calculateResidual(originalPos, originalVel, acc, concatForces);

  if (!equals(manualResidual, originalResidual, 1e-8))
  {
    std::cout << "Failed to recover residual!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(6, 3);
    compare.col(0) = originalResidual;
    compare.col(1) = manualResidual;
    compare.col(2) = originalResidual - manualResidual;
    std::cout << "Original - Recovered - Diff: " << std::endl
              << compare << std::endl;
    return false;
  }

  return true;
}

bool testSpatialNewtonGrad(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<int, Eigen::Vector6s> worldForces,
    neural::WithRespectTo* wrt)
{
  Eigen::VectorXs concatForces = Eigen::VectorXs::Zero(worldForces.size() * 6);
  int i = 0;
  for (auto& pair : worldForces)
  {
    concatForces.segment<6>(i * 6) = pair.second;
    i++;
  }

  SpatialNewtonHelper helper(skel);

  Eigen::VectorXs accWeights = Eigen::VectorXs::Random(skel->getNumBodyNodes());

  Eigen::VectorXs acc = helper.calculateAccelerationNormGradient(
      skel->getPositions(),
      skel->getVelocities(),
      skel->getAccelerations(),
      accWeights,
      wrt,
      true);
  Eigen::VectorXs acc_fd = helper.finiteDifferenceAccelerationNormGradient(
      skel->getPositions(),
      skel->getVelocities(),
      skel->getAccelerations(),
      accWeights,
      wrt,
      true);

  if (!equals(acc, acc_fd, 1e-8))
  {
    std::cout << "Acceleration regularization gradient wrt " << wrt->name()
              << " failed!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(acc.size(), 3);
    compare.col(0) = acc_fd;
    compare.col(1) = acc;
    compare.col(2) = acc_fd - acc;
    std::cout << "FD - Analytical - Diff: " << std::endl
              << compare << std::endl;
    return false;
  }

  Eigen::VectorXs linForce = helper.calculateLinearForceGapNormGradientWrt(
      skel->getPositions(),
      skel->getVelocities(),
      skel->getAccelerations(),
      concatForces,
      wrt,
      true);
  Eigen::VectorXs linForce_fd
      = helper.finiteDifferenceLinearForceGapNormGradientWrt(
          skel->getPositions(),
          skel->getVelocities(),
          skel->getAccelerations(),
          concatForces,
          wrt,
          true);

  if (!equals(linForce, linForce_fd, 1e-8))
  {
    std::cout << "Linear force gap gradient wrt " << wrt->name() << " failed!"
              << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(linForce.size(), 3);
    compare.col(0) = linForce_fd;
    compare.col(1) = linForce;
    compare.col(2) = linForce_fd - linForce;
    std::cout << "FD - Analytical - Diff: " << std::endl
              << compare << std::endl;
    return false;
  }

  return true;
}

bool testBodyScaleJointJacobians(
    std::shared_ptr<dynamics::Skeleton> skel, int joint)
{
  Eigen::Isometry3s originalParent
      = skel->getJoint(joint)->getTransformFromParentBodyNode();
  Eigen::Isometry3s originalChild
      = skel->getJoint(joint)->getTransformFromChildBodyNode();

  Eigen::Isometry3s randomParent = originalParent;
  randomParent.translation() = Eigen::Vector3s::Random();
  Eigen::Isometry3s randomChild = originalChild;
  randomChild.translation() = Eigen::Vector3s::Random();

  skel->getJoint(joint)->setTransformFromParentBodyNode(randomParent);
  skel->getJoint(joint)->setTransformFromChildBodyNode(randomChild);

  for (int axis = -1; axis < 3; axis++)
  {
    Eigen::Vector6s normal
        = skel->getJoint(joint)->getLocalTransformScrewWrtChildScale(axis);
    Eigen::Vector6s fd
        = skel->getJoint(joint)
              ->finiteDifferenceLocalTransformScrewWrtChildScale(axis);

    if (!equals(fd, normal, 1e-8))
    {
      Eigen::Vector3s offset
          = skel->getJoint(joint)->getOriginalTransformFromChildBodyNode();

      std::cout << "Screw of relative transform "
                << skel->getJoint(joint)->getName() << " (type "
                << skel->getJoint(joint)->getType()
                << ") with respect to child scale axis " << axis
                << " didn't equal analytical!" << std::endl;
      std::cout << "Original child offset (mag = " << offset.norm()
                << "): " << std::endl
                << offset << std::endl;
      std::cout << "Analytical (mag = " << normal.norm() << "): " << std::endl
                << normal << std::endl;
      std::cout << "FD (mag = " << fd.norm() << "): " << std::endl
                << fd << std::endl;
      std::cout << "Diff: " << std::endl << fd - normal << std::endl;
      return false;
    }
  }

  for (int axis = -1; axis < 3; axis++)
  {
    Eigen::Vector6s normal
        = skel->getJoint(joint)->getLocalTransformScrewWrtParentScale(axis);
    Eigen::Vector6s fd
        = skel->getJoint(joint)
              ->finiteDifferenceLocalTransformScrewWrtParentScale(axis);
    if (!equals(fd, normal, 1e-8))
    {
      Eigen::Vector3s offset
          = skel->getJoint(joint)->getOriginalTransformFromParentBodyNode();

      std::cout << "Screw of relative transform "
                << skel->getJoint(joint)->getName() << " (type "
                << skel->getJoint(joint)->getType()
                << ") with respect to parent scale axis " << axis
                << " didn't equal analytical!" << std::endl;
      std::cout << "Original child offset (mag = " << offset.norm()
                << "): " << std::endl
                << offset << std::endl;
      std::cout << "Analytical (mag = " << normal.norm() << "): " << std::endl
                << normal << std::endl;
      std::cout << "FD (mag = " << fd.norm() << "): " << std::endl
                << fd << std::endl;
      std::cout << "Diff: " << std::endl << fd - normal << std::endl;
      return false;
    }
  }

  for (int axis = -1; axis < 3; axis++)
  {
    Eigen::MatrixXs normal
        = skel->getJoint(joint)->getRelativeJacobianDerivWrtParentScale(axis);
    Eigen::MatrixXs fd
        = skel->getJoint(joint)
              ->finiteDifferenceRelativeJacobianDerivWrtParentScale(axis);

    if (!equals(fd, normal, 1e-8))
    {
      std::cout << "Gradient of relative jacobian "
                << skel->getJoint(joint)->getName() << " (type "
                << skel->getJoint(joint)->getType()
                << ") with respect to parent scale axis " << axis
                << " didn't equal analytical!" << std::endl;
      std::cout << "Analytical: " << std::endl << normal << std::endl;
      std::cout << "FD: " << std::endl << fd << std::endl;
      std::cout << "Diff: " << std::endl << fd - normal << std::endl;
      return false;
    }
  }

  for (int axis = -1; axis < 3; axis++)
  {
    Eigen::MatrixXs normal
        = skel->getJoint(joint)->getRelativeJacobianDerivWrtChildScale(axis);
    Eigen::MatrixXs fd
        = skel->getJoint(joint)
              ->finiteDifferenceRelativeJacobianDerivWrtChildScale(axis);

    if (!equals(fd, normal, 1e-8))
    {
      std::cout << "Gradient of relative jacobian "
                << skel->getJoint(joint)->getName() << " (type "
                << skel->getJoint(joint)->getType()
                << ") with respect to child scale axis " << axis
                << " didn't equal analytical!" << std::endl;
      std::cout << "Analytical: " << std::endl << normal << std::endl;
      std::cout << "FD: " << std::endl << fd << std::endl;
      std::cout << "Diff: " << std::endl << fd - normal << std::endl;
      return false;
    }
  }

  for (int axis = -1; axis < 3; axis++)
  {
    Eigen::MatrixXs normal
        = skel->getJoint(joint)
              ->getRelativeJacobianTimeDerivDerivWrtParentScale(axis);
    Eigen::MatrixXs fd
        = skel->getJoint(joint)
              ->finiteDifferenceRelativeJacobianTimeDerivDerivWrtParentScale(
                  axis);

    if (!equals(fd, normal, 1e-8))
    {
      std::cout << "Gradient of time derivative of relative jacobian "
                << skel->getJoint(joint)->getName() << " (type "
                << skel->getJoint(joint)->getType()
                << ") with respect to parent scale axis " << axis
                << " didn't equal analytical!" << std::endl;
      std::cout << "Analytical: " << std::endl << normal << std::endl;
      std::cout << "FD: " << std::endl << fd << std::endl;
      std::cout << "Diff: " << std::endl << fd - normal << std::endl;
      return false;
    }
  }

  for (int axis = -1; axis < 3; axis++)
  {
    Eigen::MatrixXs normal
        = skel->getJoint(joint)->getRelativeJacobianTimeDerivDerivWrtChildScale(
            axis);
    Eigen::MatrixXs fd
        = skel->getJoint(joint)
              ->finiteDifferenceRelativeJacobianTimeDerivDerivWrtChildScale(
                  axis);

    if (!equals(fd, normal, 1e-8))
    {
      std::cout << "Gradient of time derivative of relative jacobian "
                << skel->getJoint(joint)->getName() << " (type "
                << skel->getJoint(joint)->getType()
                << ") with respect to child scale axis " << axis
                << " didn't equal analytical!" << std::endl;
      std::cout << "Analytical: " << std::endl << normal << std::endl;
      std::cout << "FD: " << std::endl << fd << std::endl;
      std::cout << "Diff: " << std::endl << fd - normal << std::endl;
      return false;
    }
  }

  skel->getJoint(joint)->setTransformFromParentBodyNode(originalParent);
  skel->getJoint(joint)->setTransformFromChildBodyNode(originalChild);

  return true;
}

bool testMassJacobian(
    std::shared_ptr<dynamics::Skeleton> skel, neural::WithRespectTo* wrt)
{
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = skel->getRandomPose();
  skel->setControlForces(originalTau);
  skel->clearExternalForces();
  skel->computeForwardDynamics();
  Eigen::VectorXs acc = skel->getAccelerations();
  skel->integrateVelocities(skel->getTimeStep());
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  skel->setControlForces(Eigen::VectorXs::Zero(skel->getNumDofs()));
  Eigen::MatrixXs dM = skel->getJacobianOfM(acc, wrt);
  Eigen::MatrixXs dM_fd = skel->finiteDifferenceJacobianOfM(acc, wrt);

  if (!equals(dM, dM_fd, 1e-8))
  {
    std::cout << "dM and dM_fd with respect to " << wrt->name() << " not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl
              << dM.block(0, 0, 6, 6) << std::endl;
    std::cout << "FD:" << std::endl << dM_fd.block(0, 0, 6, 6) << std::endl;
    std::cout << "Diff (" << (dM_fd - dM).minCoeff() << " - "
              << (dM_fd - dM).maxCoeff() << "):" << std::endl
              << (dM_fd - dM).block(0, 0, 6, 6) << std::endl;

    for (int i = 0; i < skel->getNumBodyNodes(); i++)
    {
      if (!skel->getBodyNode(i)->debugJacobianOfMForward(wrt, acc))
      {
        return false;
      }
    }
    for (int i = skel->getNumBodyNodes() - 1; i >= 0; i--)
    {
      if (!skel->getBodyNode(i)->debugJacobianOfMBackward(wrt, acc, dM_fd))
      {
        return false;
      }
    }

    return false;
  }
  return true;
}

bool testCoriolisJacobian(
    std::shared_ptr<dynamics::Skeleton> skel, neural::WithRespectTo* wrt)
{
  skel->clearExternalForces();
  Eigen::MatrixXs dC = skel->getJacobianOfC(wrt);
  Eigen::MatrixXs dC_fd = skel->finiteDifferenceJacobianOfC(wrt);

  if (!equals(dC, dC_fd, 1e-8))
  {
    std::cout << "dC and dC_fd not equal (with respect to " << wrt->name()
              << ")!" << std::endl;
    std::cout << "Analytical:" << std::endl
              << dC.block(0, 0, 6, 6) << std::endl;
    std::cout << "FD:" << std::endl << dC_fd.block(0, 0, 6, 6) << std::endl;
    std::cout << "Diff (" << (dC_fd - dC).minCoeff() << " - "
              << (dC_fd - dC).maxCoeff() << "):" << std::endl
              << (dC_fd - dC).block(0, 0, 6, 6) << std::endl;

    for (int i = 0; i < skel->getNumBodyNodes(); i++)
    {
      if (!skel->getBodyNode(i)->debugJacobianOfCForward(wrt))
      {
        return false;
      }
    }
    for (int i = skel->getNumBodyNodes() - 1; i >= 0; i--)
    {
      if (!skel->getBodyNode(i)->debugJacobianOfCBackward(wrt))
      {
        return false;
      }
    }
    return false;
  }
  return true;
}

bool testResidualJacWrt(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<int, Eigen::Vector6s> worldForces,
    neural::WithRespectTo* wrt)
{
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = skel->getRandomPose();
  Eigen::Vector6s originalResidual = originalTau.head<6>();
  (void)originalResidual;
  applyExternalForces(skel, worldForces);
  skel->setControlForces(originalTau);
  skel->computeForwardDynamics();
  Eigen::VectorXs acc = skel->getAccelerations();
  skel->integrateVelocities(skel->getTimeStep());

  // Reset
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  applyExternalForces(skel, worldForces);
  skel->clearExternalForces();
  skel->setControlForces(Eigen::VectorXs::Zero(skel->getNumDofs()));

  std::vector<int> forceBodies;
  Eigen::VectorXs concatForces = Eigen::VectorXs::Zero(worldForces.size() * 6);
  for (auto& pair : worldForces)
  {
    concatForces.segment<6>(forceBodies.size() * 6) = pair.second;
    forceBodies.push_back(pair.first);
  }
  ResidualForceHelper helper(skel, forceBodies);

  Eigen::MatrixXs analytical = helper.calculateResidualJacobianWrt(
      originalPos, originalVel, acc, concatForces, wrt);
  Eigen::MatrixXs fd = helper.finiteDifferenceResidualJacobianWrt(
      originalPos, originalVel, acc, concatForces, wrt);

  if (!equals(analytical, fd, 2e-8))
  {
    std::cout << "Jacobian of tau wrt " << wrt->name() << " not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl
              << analytical.block(0, 0, 6, 10) << std::endl;
    std::cout << "FD:" << std::endl << fd.block(0, 0, 6, 10) << std::endl;
    std::cout << "Diff (" << (fd - analytical).minCoeff() << " - "
              << (fd - analytical).maxCoeff() << "):" << std::endl
              << (fd - analytical).block(0, 0, 6, 10) << std::endl;
    return false;
  }

  return true;
}

bool testResidualGradWrt(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<int, Eigen::Vector6s> worldForces,
    neural::WithRespectTo* wrt,
    Eigen::VectorXs originalForceVector = Eigen::VectorXs::Zero(0))
{
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = skel->getRandomPose();
  Eigen::Vector6s originalResidual = originalTau.head<6>();
  (void)originalResidual;
  applyExternalForces(skel, worldForces);
  skel->setControlForces(originalTau);
  skel->computeForwardDynamics();
  Eigen::VectorXs acc = skel->getAccelerations();
  skel->integrateVelocities(skel->getTimeStep());

  // Reset
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  applyExternalForces(skel, worldForces);
  skel->clearExternalForces();
  skel->setControlForces(Eigen::VectorXs::Zero(skel->getNumDofs()));

  std::vector<int> forceBodies;
  Eigen::VectorXs concatForces = Eigen::VectorXs::Zero(worldForces.size() * 6);
  for (auto& pair : worldForces)
  {
    concatForces.segment<6>(forceBodies.size() * 6) = pair.second;
    forceBodies.push_back(pair.first);
  }
  if (originalForceVector.size() != 0)
  {
    if (!equals(originalForceVector, concatForces))
    {
      std::cout << "Got an error reconstructing original forces!" << std::endl;
      Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(concatForces.size(), 3);
      compare.col(0) = concatForces;
      compare.col(1) = originalForceVector;
      compare.col(2) = concatForces - originalForceVector;
      std::cout << "reconstructed - original - diff" << std::endl
                << compare << std::endl;
      return false;
    }
  }
  ResidualForceHelper helper(skel, forceBodies);

  s_t residualTorqueMultiple = 10.0;
  Eigen::VectorXs analytical = helper.calculateResidualNormGradientWrt(
      originalPos,
      originalVel,
      acc,
      concatForces,
      wrt,
      residualTorqueMultiple,
      true);
  Eigen::VectorXs fd = helper.finiteDifferenceResidualNormGradientWrt(
      originalPos,
      originalVel,
      acc,
      concatForces,
      wrt,
      residualTorqueMultiple,
      true);

  s_t max = fd.cwiseAbs().maxCoeff();
  analytical /= max;
  fd /= max;

  if (!equals(analytical, fd, 5e-7))
  {
    std::cout << "Gradient of norm(tau) (divided by max coeff=" << max
              << ") wrt " << wrt->name() << " not equal!" << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(analytical.size(), 4);
    compare.col(0) = analytical;
    compare.col(1) = fd;
    compare.col(2) = fd - analytical;
    compare.col(3) = wrt->get(skel.get());
    std::cout << "Analytical - FD - Diff (" << (fd - analytical).minCoeff()
              << " - " << (fd - analytical).maxCoeff()
              << ") - Value:" << std::endl
              << compare << std::endl;
    return false;
  }

  return true;
}

bool testResidualRootJacobianWrt(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<int> contactBodies,
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forces)
{
  ResidualForceHelper helper(skel, contactBodies);

  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalAcc = skel->getAccelerations();
  skel->setPositions(q);
  skel->setVelocities(dq);
  skel->setAccelerations(ddq);

  std::vector<neural::WithRespectTo*> wrts;
  wrts.push_back(neural::WithRespectTo::LINEARIZED_MASSES);

  for (neural::WithRespectTo* wrt : wrts)
  {
    Eigen::MatrixXs angWrt = helper.calculateRootAngularResidualJacobianWrt(
        q, dq, ddq, forces, wrt);
    Eigen::MatrixXs angWrt_fd
        = helper.finiteDifferenceRootAngularResidualJacobianWrt(
            q, dq, ddq, forces, wrt);

    if (!equals(angWrt, angWrt_fd, 2e-8))
    {
      std::cout << "Jacobian of root angular residual wrt " << wrt->name()
                << " not equal!" << std::endl;
      std::cout << "Analytical:" << std::endl << angWrt << std::endl;
      std::cout << "FD:" << std::endl << angWrt_fd << std::endl;
      std::cout << "Diff (" << (angWrt_fd - angWrt).minCoeff() << " - "
                << (angWrt_fd - angWrt).maxCoeff() << "):" << std::endl
                << (angWrt_fd - angWrt) << std::endl;
      return false;
    }

    Eigen::MatrixXs angAccWrt
        = helper.calculateResidualFreeRootAngularAccelerationJacobianWrt(
            q, dq, ddq, forces, wrt);
    Eigen::MatrixXs angAccWrt_fd
        = helper.finiteDifferenceResidualFreeRootAngularAccelerationJacobianWrt(
            q, dq, ddq, forces, wrt);

    if (!equals(angAccWrt, angAccWrt_fd, 2e-8))
    {
      std::cout << "Jacobian of residual free angular acceleration wrt "
                << wrt->name() << " not equal!" << std::endl;
      std::cout << "Analytical:" << std::endl << angAccWrt << std::endl;
      std::cout << "FD:" << std::endl << angAccWrt_fd << std::endl;
      std::cout << "Diff (" << (angAccWrt_fd - angAccWrt).minCoeff() << " - "
                << (angAccWrt_fd - angAccWrt).maxCoeff() << "):" << std::endl
                << (angAccWrt_fd - angAccWrt) << std::endl;
      return false;
    }

    // Check that the linear position offset has a linear effect on angular
    // residual

    Eigen::VectorXs originalWrt = wrt->get(skel.get());

    Eigen::Vector3s originalAngAcc
        = helper.calculateResidualFreeAngularAcceleration(
            skel->getPositions(),
            skel->getVelocities(),
            skel->getAccelerations(),
            forces);

    for (int i = 0; i < 10; i++)
    {
      Eigen::VectorXs offset = Eigen::VectorXs::Random(originalWrt.size());

      // Need to do special work to keep this in bounds
      if (wrt == neural::WithRespectTo::LINEARIZED_MASSES)
      {
        offset(0) = abs(offset(0)) * 0.001;
        offset.segment(1, offset.size() - 1).setZero();

        // offset(0) = 0.0;
        // // Make changes small
        // offset.segment(1, offset.size() - 1) *= 0.001;
        // // Make sure changes sum to 0
        // offset.segment(1, offset.size() - 1)
        //     -= Eigen::VectorXs::Ones(offset.size() - 1)
        //        * (offset.segment(1, offset.size() - 1).sum()
        //           / (offset.size() - 1));
      }

      Eigen::VectorXs offsetWrt = originalWrt;
      offsetWrt += offset;
      wrt->set(skel.get(), offsetWrt);

      Eigen::Vector3s predictedChange = angWrt * offset;
      Eigen::Vector3s actualChange
          = helper.calculateResidualFreeAngularAcceleration(
                skel->getPositions(),
                skel->getVelocities(),
                skel->getAccelerations(),
                forces)
            - originalAngAcc;

      if (!equals(predictedChange, actualChange, 1e-8))
      {
        std::cout << "Relationship between " << wrt->name()
                  << " and residual free angular acc is "
                     "not linear! Change:"
                  << std::endl
                  << offset << std::endl;
        Eigen::Matrix3s compare;
        compare.col(0) = predictedChange;
        compare.col(1) = actualChange;
        compare.col(2) = predictedChange - actualChange;
        std::cout << "Predicted - Actual - Diff" << std::endl
                  << compare << std::endl;
        return false;
      }
    }

    wrt->set(skel.get(), originalWrt);
  }

  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  skel->setAccelerations(originalAcc);

  return true;
}

bool testResidualRootJacobians(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<int> contactBodies,
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forces)
{
  ResidualForceHelper helper(skel, contactBodies);

  Eigen::MatrixXs angWrtPos
      = helper.calculateRootAngularResidualJacobianWrtLinearPosition(
          q, dq, ddq, forces);
  Eigen::MatrixXs angWrtPos_fd
      = helper.finiteDifferenceRootAngularResidualJacobianWrtLinearPosition(
          q, dq, ddq, forces);

  if (!equals(angWrtPos, angWrtPos_fd, 2e-8))
  {
    std::cout
        << "Jacobian of root angular residual wrt linear position not equal!"
        << std::endl;
    std::cout << "Analytical:" << std::endl << angWrtPos << std::endl;
    std::cout << "FD:" << std::endl << angWrtPos_fd << std::endl;
    std::cout << "Diff (" << (angWrtPos_fd - angWrtPos).minCoeff() << " - "
              << (angWrtPos_fd - angWrtPos).maxCoeff() << "):" << std::endl
              << (angWrtPos_fd - angWrtPos) << std::endl;
    return false;
  }

  // Check that the linear position offset has a linear effect on angular
  // residual

  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s offset = Eigen::Vector3s::Random();

    Eigen::VectorXs offsetQ = q;
    offsetQ.segment<3>(3) += offset;

    Eigen::Vector3s predictedChange = angWrtPos * offset;
    Eigen::Vector3s actualChange
        = helper.calculateResidual(offsetQ, dq, ddq, forces).head<3>()
          - helper.calculateResidual(q, dq, ddq, forces).head<3>();
    if (!equals(predictedChange, actualChange, 1e-8))
    {
      std::cout << "Relationship between root position and angular residual is "
                   "not linear! Change:"
                << std::endl
                << offset << std::endl;
      Eigen::Matrix3s compare;
      compare.col(0) = predictedChange;
      compare.col(1) = actualChange;
      compare.col(2) = predictedChange - actualChange;
      std::cout << "Predicted - Actual - Diff" << std::endl
                << compare << std::endl;
      return false;
    }
  }

  Eigen::MatrixXs angWrtVel
      = helper.calculateRootAngularResidualJacobianWrtLinearVelocity(
          q, dq, ddq, forces);
  Eigen::MatrixXs angWrtVel_fd
      = helper.finiteDifferenceRootAngularResidualJacobianWrtLinearVelocity(
          q, dq, ddq, forces);

  if (!equals(angWrtVel, angWrtVel_fd, 2e-8))
  {
    std::cout
        << "Jacobian of root angular residual wrt linear velocity not equal!"
        << std::endl;
    std::cout << "Analytical:" << std::endl << angWrtVel << std::endl;
    std::cout << "FD:" << std::endl << angWrtVel_fd << std::endl;
    std::cout << "Diff (" << (angWrtVel_fd - angWrtVel).minCoeff() << " - "
              << (angWrtVel_fd - angWrtVel).maxCoeff() << "):" << std::endl
              << (angWrtVel_fd - angWrtVel) << std::endl;
    return false;
  }

  // Check that the linear velocity offset has a linear effect on angular
  // residual

  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s offset = Eigen::Vector3s::Random();

    Eigen::VectorXs offsetDq = dq;
    offsetDq.segment<3>(3) += offset;

    Eigen::Vector3s predictedChange = angWrtVel * offset;
    Eigen::Vector3s actualChange
        = helper.calculateResidual(q, offsetDq, ddq, forces).head<3>()
          - helper.calculateResidual(q, dq, ddq, forces).head<3>();
    if (!equals(predictedChange, actualChange, 1e-8))
    {
      std::cout << "Relationship between root velocity and angular residual is "
                   "not linear! Change:"
                << std::endl
                << offset << std::endl;
      Eigen::Matrix3s compare;
      compare.col(0) = predictedChange;
      compare.col(1) = actualChange;
      compare.col(2) = predictedChange - actualChange;
      std::cout << "Predicted - Actual - Diff" << std::endl
                << compare << std::endl;
      return false;
    }
  }

  Eigen::MatrixXs angWrtAcc
      = helper.calculateRootAngularResidualJacobianWrtLinearAcceleration(
          q, dq, ddq, forces);
  Eigen::MatrixXs angWrtAcc_fd
      = helper.finiteDifferenceRootAngularResidualJacobianWrtLinearAcceleration(
          q, dq, ddq, forces);

  if (!equals(angWrtAcc, angWrtAcc_fd, 2e-8))
  {
    std::cout << "Jacobian of root angular residual wrt linear acceleration "
                 "not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << angWrtAcc << std::endl;
    std::cout << "FD:" << std::endl << angWrtAcc_fd << std::endl;
    std::cout << "Diff (" << (angWrtAcc_fd - angWrtAcc).minCoeff() << " - "
              << (angWrtAcc_fd - angWrtAcc).maxCoeff() << "):" << std::endl
              << (angWrtAcc_fd - angWrtAcc) << std::endl;
    return false;
  }

  Eigen::Vector3s f = Eigen::Vector3s::Random();
  Eigen::Matrix<s_t, 3, 2> basis = Eigen::Matrix<s_t, 3, 2>::Random();
  basis.col(0).normalize();
  basis.col(1).normalize();
  for (int footIndex = 0; footIndex < 2; footIndex++)
  {
    Eigen::Matrix<s_t, 3, 2> angWrtCoP
        = helper.calculateRootAngularResidualJacobianWrtCoPChange(
            q, dq, ddq, forces, f, footIndex, basis);
    Eigen::Matrix<s_t, 3, 2> angWrtCoP_fd
        = helper.finiteDifferenceRootAngularResidualJacobianWrtCoPChange(
            q, dq, ddq, forces, f, footIndex, basis);

    if (!equals(angWrtCoP, angWrtCoP_fd, 2e-8))
    {
      std::cout << "Jacobian of root angular residual wrt CoP movement "
                   "not equal!"
                << std::endl;
      std::cout << "Analytical:" << std::endl << angWrtCoP << std::endl;
      std::cout << "FD:" << std::endl << angWrtCoP_fd << std::endl;
      std::cout << "Diff (" << (angWrtCoP_fd - angWrtCoP).minCoeff() << " - "
                << (angWrtCoP_fd - angWrtCoP).maxCoeff() << "):" << std::endl
                << (angWrtCoP_fd - angWrtCoP) << std::endl;
      return false;
    }

    Eigen::Matrix<s_t, 3, 2> accWrtCoP
        = helper
              .calculateResidualFreeRootAngularAccelerationJacobianWrtCoPChange(
                  q, dq, ddq, forces, f, footIndex, basis);
    Eigen::Matrix<s_t, 3, 2> accWrtCoP_fd
        = helper
              .finiteDifferenceResidualFreeRootAngularAccelerationJacobianWrtCoPChange(
                  q, dq, ddq, forces, f, footIndex, basis);

    if (!equals(accWrtCoP, accWrtCoP_fd, 2e-8))
    {
      std::cout
          << "Jacobian of root angular residual-free-acc wrt CoP movement "
             "not equal!"
          << std::endl;
      std::cout << "Analytical:" << std::endl << accWrtCoP << std::endl;
      std::cout << "FD:" << std::endl << accWrtCoP_fd << std::endl;
      std::cout << "Diff (" << (accWrtCoP_fd - accWrtCoP).minCoeff() << " - "
                << (accWrtCoP_fd - accWrtCoP).maxCoeff() << "):" << std::endl
                << (accWrtCoP_fd - accWrtCoP) << std::endl;
      return false;
    }
  }

  // Check that the linear acceleration offset has a linear effect on angular
  // residual

  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s offset = Eigen::Vector3s::Random();

    Eigen::VectorXs offsetDdq = ddq;
    offsetDdq.segment<3>(3) += offset;

    Eigen::Vector3s predictedChange = angWrtAcc * offset;
    Eigen::Vector3s actualChange
        = helper.calculateResidual(q, dq, offsetDdq, forces).head<3>()
          - helper.calculateResidual(q, dq, ddq, forces).head<3>();
    if (!equals(predictedChange, actualChange, 1e-8))
    {
      std::cout
          << "Relationship between root acceleration and angular residual is "
             "not linear! Change:"
          << std::endl
          << offset << std::endl;
      Eigen::Matrix3s compare;
      compare.col(0) = predictedChange;
      compare.col(1) = actualChange;
      compare.col(2) = predictedChange - actualChange;
      std::cout << "Predicted - Actual - Diff" << std::endl
                << compare << std::endl;
      return false;
    }
  }

  // Eigen::MatrixXs fullResidual
  //     = helper.calculateRootResidualJacobianWrtPosition(q, dq, ddq, forces);
  // Eigen::MatrixXs fullResidual_fd
  //     = helper.finiteDifferenceRootResidualJacobianWrtPosition(
  //         q, dq, ddq, forces);

  // if (!equals(fullResidual, fullResidual_fd, 2e-8))
  // {
  //   std::cout << "Jacobian of root residual wrt position not equal!"
  //             << std::endl;
  //   std::cout << "Analytical:" << std::endl << fullResidual << std::endl;
  //   std::cout << "FD:" << std::endl << fullResidual_fd << std::endl;
  //   std::cout << "Diff (" << (fullResidual_fd - fullResidual).minCoeff()
  //             << " - " << (fullResidual_fd - fullResidual).maxCoeff()
  //             << "):" << std::endl
  //             << (fullResidual_fd - fullResidual) << std::endl;
  //   return false;
  // }

  Eigen::Vector6s noResidualRoot
      = helper.calculateResidualFreeRootAcceleration(q, dq, ddq, forces);
  Eigen::VectorXs noResidualAcc = ddq;
  noResidualAcc.head<6>() = noResidualRoot;
  Eigen::Vector6s residual
      = helper.calculateResidual(q, dq, noResidualAcc, forces);
  Eigen::Vector6s zero6 = Eigen::Vector6s::Zero();
  if (!equals(residual, zero6, 1e-8))
  {
    std::cout << "Residual free root acceleration did not remove residuals!"
              << std::endl;
    std::cout << "Remaining residuals: " << residual << std::endl;
    return false;
  }

  Eigen::Vector3s noResidualAngular
      = helper.calculateResidualFreeAngularAcceleration(q, dq, ddq, forces);
  noResidualAcc = ddq;
  noResidualAcc.head<3>() = noResidualAngular;
  Eigen::Vector3s residualAng
      = helper.calculateResidual(q, dq, noResidualAcc, forces).head<3>();
  Eigen::Vector3s zero3 = Eigen::Vector3s::Zero();
  if (!equals(residualAng, zero3, 1e-8))
  {
    std::cout << "Residual free root acceleration did not remove residuals!"
              << std::endl;
    std::cout << "Remaining residuals: " << residualAng << std::endl;
    return false;
  }

  Eigen::MatrixXs angAccWrtPos
      = helper
            .calculateResidualFreeRootAngularAccelerationJacobianWrtLinearPosition(
                q, dq, ddq, forces);
  Eigen::MatrixXs angAccWrtPos_fd
      = helper
            .finiteDifferenceResidualFreeRootAngularAccelerationJacobianWrtLinearPosition(
                q, dq, ddq, forces);

  if (!equals(angAccWrtPos, angAccWrtPos_fd, 2e-8))
  {
    std::cout << "Jacobian of root angular residual wrt position not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << angAccWrtPos << std::endl;
    std::cout << "FD:" << std::endl << angAccWrtPos_fd << std::endl;
    std::cout << "Diff (" << (angAccWrtPos_fd - angAccWrtPos).minCoeff()
              << " - " << (angAccWrtPos_fd - angAccWrtPos).maxCoeff()
              << "):" << std::endl
              << (angAccWrtPos_fd - angAccWrtPos) << std::endl;
    return false;
  }

  // Check that the linear position offset has a linear effect on residual-free
  // angular acc

  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s offset = Eigen::Vector3s::Random();

    Eigen::VectorXs offsetQ = q;
    offsetQ.segment<3>(3) += offset;

    Eigen::Vector3s predictedChange = angAccWrtPos * offset;
    Eigen::Vector3s actualChange
        = helper
              .calculateResidualFreeAngularAcceleration(
                  offsetQ, dq, ddq, forces)
              .head<3>()
          - helper.calculateResidualFreeAngularAcceleration(q, dq, ddq, forces)
                .head<3>();
    if (!equals(predictedChange, actualChange, 1e-8))
    {
      std::cout << "Relationship between root position and angular residual is "
                   "not linear! Change:"
                << std::endl
                << offset << std::endl;
      Eigen::Matrix3s compare;
      compare.col(0) = predictedChange;
      compare.col(1) = actualChange;
      compare.col(2) = predictedChange - actualChange;
      std::cout << "Predicted - Actual - Diff" << std::endl
                << compare << std::endl;
      return false;
    }
  }

  Eigen::MatrixXs angAccWrtVel
      = helper
            .calculateResidualFreeRootAngularAccelerationJacobianWrtLinearVelocity(
                q, dq, ddq, forces);
  Eigen::MatrixXs angAccWrtVel_fd
      = helper
            .finiteDifferenceResidualFreeRootAngularAccelerationJacobianWrtLinearVelocity(
                q, dq, ddq, forces);

  if (!equals(angAccWrtVel, angAccWrtVel_fd, 2e-8))
  {
    std::cout << "Jacobian of root angular residual wrt velocity not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << angAccWrtVel << std::endl;
    std::cout << "FD:" << std::endl << angAccWrtVel_fd << std::endl;
    std::cout << "Diff (" << (angAccWrtVel_fd - angAccWrtVel).minCoeff()
              << " - " << (angAccWrtVel_fd - angAccWrtVel).maxCoeff()
              << "):" << std::endl
              << (angAccWrtVel_fd - angAccWrtVel) << std::endl;
    return false;
  }

  // Check that the linear velocity offset has a linear effect on residual-free
  // angular acc

  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s offset = Eigen::Vector3s::Random();

    Eigen::VectorXs offsetDq = dq;
    offsetDq.segment<3>(3) += offset;

    Eigen::Vector3s predictedChange = angAccWrtVel * offset;
    Eigen::Vector3s actualChange
        = helper
              .calculateResidualFreeAngularAcceleration(
                  q, offsetDq, ddq, forces)
              .head<3>()
          - helper.calculateResidualFreeAngularAcceleration(q, dq, ddq, forces)
                .head<3>();
    if (!equals(predictedChange, actualChange, 1e-8))
    {
      std::cout << "Relationship between root velocity and angular residual is "
                   "not linear! Change:"
                << std::endl
                << offset << std::endl;
      Eigen::Matrix3s compare;
      compare.col(0) = predictedChange;
      compare.col(1) = actualChange;
      compare.col(2) = predictedChange - actualChange;
      std::cout << "Predicted - Actual - Diff" << std::endl
                << compare << std::endl;
      return false;
    }
  }

  Eigen::MatrixXs angAccWrtAcc
      = helper
            .calculateResidualFreeRootAngularAccelerationJacobianWrtLinearAcceleration(
                q, dq, ddq, forces);
  Eigen::MatrixXs angAccWrtAcc_fd
      = helper
            .finiteDifferenceResidualFreeRootAngularAccelerationJacobianWrtLinearAcceleration(
                q, dq, ddq, forces);

  if (!equals(angAccWrtAcc, angAccWrtAcc_fd, 2e-8))
  {
    std::cout << "Jacobian of root angular residual wrt acceleration not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << angAccWrtAcc << std::endl;
    std::cout << "FD:" << std::endl << angAccWrtAcc_fd << std::endl;
    std::cout << "Diff (" << (angAccWrtAcc_fd - angAccWrtAcc).minCoeff()
              << " - " << (angAccWrtAcc_fd - angAccWrtAcc).maxCoeff()
              << "):" << std::endl
              << (angAccWrtAcc_fd - angAccWrtAcc) << std::endl;
    return false;
  }

  // Check that the linear acceleration offset has a linear effect on
  // residual-free angular acc

  for (int i = 0; i < 10; i++)
  {
    Eigen::Vector3s offset = Eigen::Vector3s::Random();

    Eigen::VectorXs offsetDdq = ddq;
    offsetDdq.segment<3>(3) += offset;

    Eigen::Vector3s predictedChange = angAccWrtAcc * offset;
    Eigen::Vector3s actualChange
        = helper
              .calculateResidualFreeAngularAcceleration(
                  q, dq, offsetDdq, forces)
              .head<3>()
          - helper.calculateResidualFreeAngularAcceleration(q, dq, ddq, forces)
                .head<3>();
    if (!equals(predictedChange, actualChange, 1e-8))
    {
      std::cout
          << "Relationship between root acceleration and angular residual is "
             "not linear! Change:"
          << std::endl
          << offset << std::endl;
      Eigen::Matrix3s compare;
      compare.col(0) = predictedChange;
      compare.col(1) = actualChange;
      compare.col(2) = predictedChange - actualChange;
      std::cout << "Predicted - Actual - Diff" << std::endl
                << compare << std::endl;
      return false;
    }
  }

  /*
  // TODO: these tests are numerically finnicky (hard to choose good finite
  // differencing epsilons), and we don't use these
  // Jacobians anyways, so this is disabled for now.

  Eigen::MatrixXs rootAccResidual
      = helper.calculateResidualFreeRootAccelerationJacobianWrtPosition(
          q, dq, ddq, forces);
  Eigen::MatrixXs rootAccResidual_fd
      = helper.finiteDifferenceResidualFreeRootAccelerationJacobianWrtPosition(
          q, dq, ddq, forces);

  if (!equals(rootAccResidual_fd, rootAccResidual, 5e-8))
  {
    std::cout << "Jacobian of root acceleration wrt position not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl << rootAccResidual << std::endl;
    std::cout << "FD:" << std::endl << rootAccResidual_fd << std::endl;
    std::cout << "Diff (" << (rootAccResidual_fd - rootAccResidual).minCoeff()
              << " - " << (rootAccResidual_fd - rootAccResidual).maxCoeff()
              << "):" << std::endl
              << (rootAccResidual_fd - rootAccResidual) << std::endl;
    return false;
  }

  Eigen::MatrixXs rootAccResidualWrtVel
      = helper.calculateResidualFreeRootAccelerationJacobianWrtVelocity(
          q, dq, ddq, forces);
  Eigen::MatrixXs rootAccResidualWrtVel_fd
      = helper.finiteDifferenceResidualFreeRootAccelerationJacobianWrtVelocity(
          q, dq, ddq, forces);

  if (!equals(rootAccResidualWrtVel, rootAccResidualWrtVel_fd, 2e-8))
  {
    std::cout << "Jacobian of root acceleration wrt velocity not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl
              << rootAccResidualWrtVel << std::endl;
    std::cout << "FD:" << std::endl << rootAccResidualWrtVel_fd << std::endl;
    std::cout << "Diff ("
              << (rootAccResidualWrtVel_fd - rootAccResidualWrtVel).minCoeff()
              << " - "
              << (rootAccResidualWrtVel_fd - rootAccResidualWrtVel).maxCoeff()
              << "):" << std::endl
              << (rootAccResidualWrtVel_fd - rootAccResidualWrtVel)
              << std::endl;
    return false;
  }

  Eigen::VectorXs scratchWrtInvMass
      = helper.calculateScratchJacobianWrtInvMass(q, dq, ddq, forces);
  Eigen::VectorXs scratchWrtInvMass_fd
      = helper.finiteDifferenceScratchJacobianWrtInvMass(q, dq, ddq, forces);

  if (!equals(scratchWrtInvMass, scratchWrtInvMass_fd, 6e-8))
  {
    std::cout
        << "Scratch Jacobian of root acceleration wrt inverse mass not equal!"
        << std::endl;
    std::cout << "Analytical:" << std::endl << scratchWrtInvMass << std::endl;
    std::cout << "FD:" << std::endl << scratchWrtInvMass_fd << std::endl;
    std::cout << "Diff ("
              << (scratchWrtInvMass_fd - scratchWrtInvMass).minCoeff() << " - "
              << (scratchWrtInvMass_fd - scratchWrtInvMass).maxCoeff()
              << "):" << std::endl
              << (scratchWrtInvMass_fd - scratchWrtInvMass) << std::endl;
    return false;
  }

  Eigen::VectorXs rootAccResidualWrtInvMass
      = helper.calculateResidualFreeRootAccelerationJacobianWrtInvMass(
          q, dq, ddq, forces);
  Eigen::VectorXs rootAccResidualWrtInvMass_fd
      = helper.finiteDifferenceResidualFreeRootAccelerationJacobianWrtInvMass(
          q, dq, ddq, forces);

  if (!equals(rootAccResidualWrtInvMass, rootAccResidualWrtInvMass_fd, 5e-8))
  {
    std::cout << "Jacobian of root acceleration wrt inverse mass not equal!"
              << std::endl;
    std::cout << "Analytical:" << std::endl
              << rootAccResidualWrtInvMass << std::endl;
    std::cout << "FD:" << std::endl
              << rootAccResidualWrtInvMass_fd << std::endl;
    std::cout
        << "Diff ("
        << (rootAccResidualWrtInvMass_fd - rootAccResidualWrtInvMass).minCoeff()
        << " - "
        << (rootAccResidualWrtInvMass_fd - rootAccResidualWrtInvMass).maxCoeff()
        << "):" << std::endl
        << (rootAccResidualWrtInvMass_fd - rootAccResidualWrtInvMass)
        << std::endl;
    return false;
  }
  */

  return true;
}

bool testResidualTrajectoryTaylorExpansionWithRandomTrajectory(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<int> collisionBodies,
    int numTimesteps)
{
  s_t dt = skel->getTimeStep();
  Eigen::MatrixXs qs = Eigen::MatrixXs::Zero(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs dqs = Eigen::MatrixXs::Zero(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs ddqs
      = Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs forces
      = 0.001
        * Eigen::MatrixXs::Random(collisionBodies.size() * 6, numTimesteps);
  std::vector<bool> probablyMissingGRF;
  for (int i = 0; i < numTimesteps; i++)
  {
    probablyMissingGRF.push_back(i % 4 == 0);
  }

  // Generate q,dq from integrating the given accelerations
  qs.col(0) = skel->getRandomPose();
  dqs.col(0) = skel->getRandomVelocity();
  for (int i = 1; i < numTimesteps; i++)
  {
    dqs.col(i) = dqs.col(i - 1) + ddqs.col(i - 1) * dt;
    qs.col(i) = qs.col(i - 1) + dqs.col(i) * dt;
  }
  // Add some random noise
  qs += Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps) * 0.0001;
  dqs += Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps) * 0.001;

  for (int t = 0; t < numTimesteps; t++)
  {
    std::cout << "Testing individual jacobians at t=" << t << std::endl;
    bool success = testResidualRootJacobians(
        skel,
        collisionBodies,
        qs.col(t),
        dqs.col(t),
        ddqs.col(t),
        forces.col(t));
    if (!success)
      return false;
  }

  ResidualForceHelper helper(skel, collisionBodies);

  Eigen::Vector6s posOffset = Eigen::Vector6s::Zero();
  Eigen::Vector6s velOffset = Eigen::Vector6s::Zero();
  Eigen::MatrixXs qsLin = helper.getRootTrajectoryLinearSystemPoses(
      posOffset, velOffset, qs, dqs, ddqs, forces, probablyMissingGRF);
  Eigen::MatrixXs qsFwd = helper.getResidualFreePoses(
      posOffset, velOffset, qs, forces, probablyMissingGRF);
  Eigen::MatrixXs diff = qsLin - qsFwd;

  // Note that the forward dynamics version only begins changing at the [2]
  // timestep and beyond (3rd timestep)
  std::cout << "Diff between linear and fwd definitions of trajectory:"
            << std::endl
            << diff.block(0, 0, 6, std::min((int)diff.cols(), 10)) << std::endl;

  std::vector<bool> includeAllResidualsList;
  includeAllResidualsList.push_back(true);
  includeAllResidualsList.push_back(false);
  for (bool includeAllResiduals : includeAllResidualsList)
  {
    std::pair<Eigen::MatrixXs, Eigen::VectorXs> taylor
        = helper.getRootTrajectoryLinearSystem(
            qs, dqs, ddqs, forces, probablyMissingGRF, includeAllResiduals);
    std::pair<Eigen::MatrixXs, Eigen::VectorXs> taylor_fd
        = helper.finiteDifferenceRootTrajectoryLinearSystem(
            qs, dqs, ddqs, forces, probablyMissingGRF, includeAllResiduals);

    if (!equals(taylor.second, taylor_fd.second, 1e-8))
    {
      std::cout << "Linear system b vector is not equal (includeAllResiduals="
                << includeAllResiduals << ")!" << std::endl;
      return false;
    }

    if (!equals(taylor.first, taylor_fd.first, 1e-8))
    {
      std::cout << "Linear system A matrix is not equal (includeAllResiduals="
                << includeAllResiduals << ")!" << std::endl;
      Eigen::MatrixXs A = taylor.first;
      Eigen::MatrixXs A_fd = taylor_fd.first;

      for (int t = 0; t < numTimesteps; t++)
      {
        Eigen::Matrix6s posOffsetPos = A.block<6, 6>(t * 6, 0);
        Eigen::Matrix6s posOffsetPos_fd = A_fd.block<6, 6>(t * 6, 0);

        if (!equals(posOffsetPos, posOffsetPos_fd, 1e-8))
        {
          std::cout << "Linear system error at t=" << t << std::endl;
          std::cout << "Analytical dPos[" << t << "]/dOffsetPos:" << std::endl
                    << posOffsetPos << std::endl;
          std::cout << "FD dPos[" << t << "]/dOffsetPos:" << std::endl
                    << posOffsetPos_fd << std::endl;
          std::cout << "Extra (Analytical - FD):" << std::endl
                    << posOffsetPos - posOffsetPos_fd << std::endl;
          for (int i = 0; i <= t; i++)
          {
            const Eigen::Matrix6s dAcc_dOffsetPos
                = helper
                      .calculateResidualFreeRootAccelerationJacobianWrtPosition(
                          qs.col(i), dqs.col(i), ddqs.col(i), forces.col(i));
            std::cout << "dt * dt * dAcc[" << i << "]/dOffsetPos:" << std::endl
                      << dt * dt * dAcc_dOffsetPos << std::endl;
          }
          return false;
        }

        Eigen::Matrix6s posOffsetVel = A.block<6, 6>(t * 6, 6);
        Eigen::Matrix6s posOffsetVel_fd = A_fd.block<6, 6>(t * 6, 6);

        if (!equals(posOffsetVel, posOffsetVel_fd, 1e-8))
        {
          std::cout << "Linear system error at t=" << t << std::endl;
          std::cout << "Analytical dPos[" << t << "]/dOffsetVel:" << std::endl
                    << posOffsetVel << std::endl;
          std::cout << "FD dPos[" << t << "]/dOffsetVel:" << std::endl
                    << posOffsetVel_fd << std::endl;
          std::cout << "Diff:" << std::endl
                    << posOffsetVel - posOffsetVel_fd << std::endl;
          return false;
        }

        int numMissing = 0;
        for (bool m : probablyMissingGRF)
          if (m)
            numMissing++;

        int numResiduals = includeAllResiduals ? numTimesteps : numMissing;
        assert(A.cols() == 12 + (numResiduals * 6));
        assert(A_fd.cols() == 12 + (numResiduals * 6));
        for (int r = 0; r < numResiduals; r++)
        {
          Eigen::Matrix6s posOffsetResidual
              = A.block<6, 6>(t * 6, 12 + (r * 6));
          Eigen::Matrix6s posOffsetResidual_fd
              = A_fd.block<6, 6>(t * 6, 12 + (r * 6));

          if (!equals(posOffsetResidual, posOffsetResidual_fd, 1e-8))
          {
            std::cout << "Linear system error at t=" << t << std::endl;
            std::cout << "Analytical dPos[" << t << "]/dResidual[" << r
                      << "]:" << std::endl
                      << posOffsetResidual << std::endl;
            std::cout << "FD dPos[" << t << "]/dResidual[" << r
                      << "]:" << std::endl
                      << posOffsetResidual_fd << std::endl;
            std::cout << "Diff:" << std::endl
                      << posOffsetResidual - posOffsetResidual_fd << std::endl;
            std::cout << "Analytical allTimesteps/dResidual[" << r
                      << "]:" << std::endl
                      << A.block(0, 12 + (r * 6), A.rows(), 6) << std::endl;
            std::cout << "FD allTimesteps/dResidual[" << r << "]:" << std::endl
                      << A_fd.block(0, 12 + (r * 6), A_fd.rows(), 6)
                      << std::endl;
            return false;
          }
        }
      }
      return false;
    }
  }

  return true;
}

bool testLinearTrajectorLinearMapWithRandomTrajectory(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<int> collisionBodies,
    int numTimesteps,
    bool useReactionWheels)
{
  s_t dt = skel->getTimeStep();
  Eigen::MatrixXs qs = Eigen::MatrixXs::Zero(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs dqs = Eigen::MatrixXs::Zero(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs ddqs
      = Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs forces
      = 0.001
        * Eigen::MatrixXs::Random(collisionBodies.size() * 6, numTimesteps);
  std::vector<bool> probablyMissingGRF;
  int numMissing = 0;
  std::vector<int> missingIndices;
  for (int i = 0; i < numTimesteps; i++)
  {
    bool missing = i % 4 == 0;
    probablyMissingGRF.push_back(missing);
    if (missing)
    {
      numMissing++;
      missingIndices.push_back(i);
    }
  }

  int maxBuckets = 2;
  if (numMissing > maxBuckets)
  {
    numMissing = maxBuckets;
  }

  std::vector<ForcePlate> driftCorrectForcePlates;
  std::vector<std::vector<int>> driftCorrectForcePlatesAssignedToContactBody;
  int driftCorrectionBlurRadius = 3;
  int driftCorrectionBlurInterval = 3;
  for (int i = 0; i < 3; i++)
  {
    driftCorrectForcePlates.emplace_back();
    std::vector<int> footAssignment;
    for (int t = 0; t < numTimesteps; t++)
    {
      driftCorrectForcePlates[i].forces.emplace_back(Eigen::Vector3s::Random());
      driftCorrectForcePlates[i].moments.emplace_back(
          Eigen::Vector3s::Random());
      driftCorrectForcePlates[i].centersOfPressure.emplace_back(
          Eigen::Vector3s::Random());
      footAssignment.push_back(i % 2);
    }
    driftCorrectForcePlatesAssignedToContactBody.push_back(footAssignment);
  }

  // Generate q,dq from integrating the given accelerations
  qs.col(0) = skel->getRandomPose();
  dqs.col(0) = skel->getRandomVelocity();
  for (int i = 1; i < numTimesteps; i++)
  {
    dqs.col(i) = dqs.col(i - 1) + ddqs.col(i - 1) * dt;
    qs.col(i) = qs.col(i - 1) + dqs.col(i) * dt;
  }
  // Add some random noise
  qs += Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps) * 0.0001;
  dqs += Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps) * 0.001;

  for (int t = 0; t < numTimesteps; t++)
  {
    std::cout << "Testing individual jacobians at t=" << t << std::endl;
    bool success = testResidualRootJacobians(
        skel,
        collisionBodies,
        qs.col(t),
        dqs.col(t),
        ddqs.col(t),
        forces.col(t));
    if (!success)
      return false;
  }

  ResidualForceHelper helper(skel, collisionBodies);

  std::tuple<Eigen::MatrixXs, Eigen::VectorXs, std::vector<Eigen::MatrixXs>>
      linear = helper.getLinearTrajectoryLinearSystem(
          dt,
          qs,
          dqs,
          ddqs,
          forces,
          probablyMissingGRF,
          useReactionWheels,
          driftCorrectForcePlates,
          driftCorrectForcePlatesAssignedToContactBody,
          driftCorrectionBlurRadius,
          driftCorrectionBlurInterval,
          maxBuckets);
  Eigen::MatrixXs A = std::get<0>(linear);
  Eigen::VectorXs b = std::get<1>(linear);
  std::vector<Eigen::MatrixXs> recoverCops = std::get<2>(linear);
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> linearParallel
      = helper.getLinearTrajectoryLinearSystemParallel(
          dt, qs, dqs, ddqs, forces, probablyMissingGRF, useReactionWheels, 2);
  std::tuple<Eigen::MatrixXs, Eigen::VectorXs, std::vector<Eigen::MatrixXs>>
      linear_fd = helper.finiteDifferenceLinearTrajectoryLinearSystem(
          dt,
          qs,
          dqs,
          ddqs,
          forces,
          probablyMissingGRF,
          useReactionWheels,
          driftCorrectForcePlates,
          driftCorrectForcePlatesAssignedToContactBody,
          driftCorrectionBlurRadius,
          driftCorrectionBlurInterval,
          maxBuckets);
  Eigen::MatrixXs A_fd = std::get<0>(linear_fd);
  Eigen::VectorXs b_fd = std::get<1>(linear_fd);
  std::vector<Eigen::MatrixXs> recoverCops_fd = std::get<2>(linear_fd);

  /*
  if (!equals(A, linearParallel.first, 1e-8))
  {
    std::cout << "Linear system A matrix not the same as parallel" << std::endl;
    return false;
  }
  if (!equals(b, linearParallel.second, 1e-8))
  {
    std::cout << "Linear system b vector not the same as parallel" << std::endl;
    return false;
  }
  */

  if (!equals(b, b_fd, 1e-8))
  {
    std::cout << "Linear system b vector is not equal!" << std::endl;
    for (int t = 0; t < numTimesteps; t++)
    {
      Eigen::Vector3s pos = b.segment<3>(t * 3);
      Eigen::Vector3s pos_fd = b_fd.segment<3>(t * 3);
      if (!equals(pos, pos_fd, 1e-8))
      {
        std::cout << "Linear system b vector pos not equal at t=" << t
                  << std::endl;
        Eigen::Matrix3s compare;
        compare.col(0) = pos;
        compare.col(1) = pos_fd;
        compare.col(2) = pos - pos_fd;
        std::cout << "Analytical - FD - Diff" << std::endl
                  << compare << std::endl;
        return false;
      }
    }
    for (int t = 0; t < numTimesteps; t++)
    {
      Eigen::Vector3s pos = b.segment<3>((numTimesteps + t) * 3);
      Eigen::Vector3s pos_fd = b_fd.segment<3>((numTimesteps + t) * 3);
      if (!equals(pos, pos_fd, 1e-8))
      {
        std::cout << "Linear system b vector ang not equal at t=" << t
                  << std::endl;
        Eigen::Matrix3s compare;
        compare.col(0) = pos;
        compare.col(1) = pos_fd;
        compare.col(2) = pos - pos_fd;
        std::cout << "Analytical - FD - Diff" << std::endl
                  << compare << std::endl;
        return false;
      }
    }
    return false;
  }

  if (!equals(A, A_fd, 1e-8))
  {
    std::cout << "Linear system A matrix is not equal!" << std::endl;

    /// Check the linear pose map
    for (int t = 0; t < numTimesteps; t++)
    {
      Eigen::Matrix3s posOffsetPos = A.block<3, 3>(t * 3, 0);
      Eigen::Matrix3s posOffsetPos_fd = A_fd.block<3, 3>(t * 3, 0);

      if (!equals(posOffsetPos, posOffsetPos_fd, 1e-8))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dPos[" << t << "]/dOffsetPos:" << std::endl
                  << posOffsetPos << std::endl;
        std::cout << "FD dPos[" << t << "]/dOffsetPos:" << std::endl
                  << posOffsetPos_fd << std::endl;
        std::cout << "Extra (Analytical - FD):" << std::endl
                  << posOffsetPos - posOffsetPos_fd << std::endl;
        return false;
      }

      Eigen::Matrix3s posOffsetVel = A.block<3, 3>(t * 3, 3);
      Eigen::Matrix3s posOffsetVel_fd = A_fd.block<3, 3>(t * 3, 3);

      if (!equals(posOffsetVel, posOffsetVel_fd, 1e-8))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dPos[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetVel << std::endl;
        std::cout << "FD dPos[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetVel_fd << std::endl;
        std::cout << "Diff:" << std::endl
                  << posOffsetVel - posOffsetVel_fd << std::endl;
        return false;
      }

      for (int i = 0; i < numMissing; i++)
      {
        Eigen::Matrix3s posOffsetResidual = A.block<3, 3>(t * 3, 6 + i * 3);
        Eigen::Matrix3s posOffsetResidual_fd
            = A_fd.block<3, 3>(t * 3, 6 + i * 3);

        if (!equals(posOffsetResidual, posOffsetResidual_fd, 1e-8))
        {
          std::cout << "Linear system error at t=" << t << std::endl;
          std::cout << "Analytical dPos[" << t << "]/dLinResidual["
                    << missingIndices[i] << "]:" << std::endl
                    << posOffsetResidual << std::endl;
          std::cout << "FD dPos[" << t << "]/dLinResidual[" << missingIndices[i]
                    << "]:" << std::endl
                    << posOffsetResidual_fd << std::endl;
          std::cout << "Diff:" << std::endl
                    << posOffsetResidual - posOffsetResidual_fd << std::endl;
          return false;
        }
      }
    }
    std::cout << "Linear pose map passes!" << std::endl;

    int rowOffset = numTimesteps * 3;
    int colOffset = 6 + numMissing * 3;
    int numBlurs = (int)floor((s_t)qs.cols() / driftCorrectionBlurInterval);

    for (int i = 0; i < driftCorrectForcePlates.size(); i++)
    {
      for (int b = 0; b < numBlurs; b++)
      {
        int col = (colOffset * 2) + ((i * numBlurs + b) * 2);

        Eigen::MatrixXs offsetOverTime = Eigen::MatrixXs::Zero(3, numTimesteps);
        Eigen::MatrixXs offsetOverTime_fd
            = Eigen::MatrixXs::Zero(3, numTimesteps);
        for (int t = 0; t < numTimesteps; t++)
        {
          offsetOverTime.col(t) = A.block<3, 1>(rowOffset + (t * 3), col);
          offsetOverTime_fd.col(t) = A_fd.block<3, 1>(rowOffset + (t * 3), col);
        }

        std::cout << "Plate " << i << " blur " << b
                  << " ang offsets over time:" << std::endl
                  << offsetOverTime << std::endl;
        std::cout << "FD Plate " << i << " blur " << b
                  << " ang offsets over time:" << std::endl
                  << offsetOverTime_fd << std::endl;
      }
    }

    /// Check the angular pose map
    for (int t = 0; t < numTimesteps; t++)
    {
      Eigen::Matrix3s posOffsetPos = A.block<3, 3>(rowOffset + t * 3, 0);
      Eigen::Matrix3s posOffsetPos_fd = A_fd.block<3, 3>(rowOffset + t * 3, 0);

      if (!equals(posOffsetPos, posOffsetPos_fd, 1e-8))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dAng[" << t << "]/dOffsetPos:" << std::endl
                  << posOffsetPos << std::endl;
        std::cout << "FD dAng[" << t << "]/dOffsetPos:" << std::endl
                  << posOffsetPos_fd << std::endl;
        std::cout << "Extra (Analytical - FD):" << std::endl
                  << posOffsetPos - posOffsetPos_fd << std::endl;
        return false;
      }

      Eigen::Matrix3s posOffsetVel = A.block<3, 3>(rowOffset + t * 3, 3);
      Eigen::Matrix3s posOffsetVel_fd = A_fd.block<3, 3>(rowOffset + t * 3, 3);

      if (!equals(posOffsetVel, posOffsetVel_fd, 1e-8))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dAng[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetVel << std::endl;
        std::cout << "FD dAng[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetVel_fd << std::endl;
        std::cout << "Diff:" << std::endl
                  << posOffsetVel - posOffsetVel_fd << std::endl;
        return false;
      }

      for (int i = 0; i < numMissing; i++)
      {
        Eigen::Matrix3s posOffsetResidual
            = A.block<3, 3>(rowOffset + t * 3, 6 + i * 3);
        Eigen::Matrix3s posOffsetResidual_fd
            = A_fd.block<3, 3>(rowOffset + t * 3, 6 + i * 3);

        if (!equals(posOffsetResidual, posOffsetResidual_fd, 1e-8))
        {
          std::cout << "Linear system error at t=" << t << std::endl;
          std::cout << "Analytical dAng[" << t << "]/dLinResidual["
                    << missingIndices[i] << "]:" << std::endl
                    << posOffsetResidual << std::endl;
          std::cout << "FD dAng[" << t << "]/dLinResidual[" << missingIndices[i]
                    << "]:" << std::endl
                    << posOffsetResidual_fd << std::endl;
          std::cout << "Diff:" << std::endl
                    << posOffsetResidual - posOffsetResidual_fd << std::endl;
          return false;
        }
      }

      Eigen::Matrix3s posOffsetAng
          = A.block<3, 3>(rowOffset + t * 3, colOffset);
      Eigen::Matrix3s posOffsetAng_fd
          = A_fd.block<3, 3>(rowOffset + t * 3, colOffset);

      if (!equals(posOffsetAng, posOffsetAng_fd, 1e-8))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dAng[" << t << "]/dOffsetAng:" << std::endl
                  << posOffsetAng << std::endl;
        std::cout << "FD dAng[" << t << "]/dOffsetAng:" << std::endl
                  << posOffsetAng_fd << std::endl;
        std::cout << "Extra (Analytical - FD):" << std::endl
                  << posOffsetAng - posOffsetAng_fd << std::endl;
        return false;
      }

      Eigen::Matrix3s posOffsetAngVel
          = A.block<3, 3>(rowOffset + t * 3, colOffset + 3);
      Eigen::Matrix3s posOffsetAngVel_fd
          = A_fd.block<3, 3>(rowOffset + t * 3, colOffset + 3);

      if (!equals(posOffsetAngVel, posOffsetAngVel_fd, 1e-8))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dAng[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetAngVel << std::endl;
        std::cout << "FD dAng[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetAngVel_fd << std::endl;
        std::cout << "Diff:" << std::endl
                  << posOffsetAngVel - posOffsetAngVel_fd << std::endl;
        return false;
      }

      for (int i = 0; i < numMissing; i++)
      {
        Eigen::Matrix3s posOffsetResidual
            = A.block<3, 3>(rowOffset + t * 3, colOffset + 6 + i * 3);
        Eigen::Matrix3s posOffsetResidual_fd
            = A_fd.block<3, 3>(rowOffset + t * 3, colOffset + 6 + i * 3);

        if (!equals(posOffsetResidual, posOffsetResidual_fd, 1e-8))
        {
          std::cout << "Linear system error at t=" << t << std::endl;
          std::cout << "Analytical dAng[" << t << "]/dAngResidual["
                    << missingIndices[i] << "]:" << std::endl
                    << posOffsetResidual << std::endl;
          std::cout << "FD dAng[" << t << "]/dAngResidual[" << missingIndices[i]
                    << "]:" << std::endl
                    << posOffsetResidual_fd << std::endl;
          std::cout << "Diff:" << std::endl
                    << posOffsetResidual - posOffsetResidual_fd << std::endl;
          return false;
        }
      }

      for (int i = 0; i < driftCorrectForcePlates.size(); i++)
      {
        for (int b = 0; b < numBlurs; b++)
        {
          int col = (colOffset * 2) + ((i * numBlurs + b) * 2);
          Eigen::Vector3s copChange = A.block<3, 1>(rowOffset + (t * 3), col);
          Eigen::Vector3s copChange_fd
              = A_fd.block<3, 1>(rowOffset + (t * 3), col);
          if (!equals(copChange, copChange_fd, 1e-8))
          {
            std::cout << "Linear system error at t=" << t << std::endl;
            std::cout << "Analytical dAng[" << t << "]/dCop[plate=" << i
                      << ",blur=" << b << "]:" << std::endl
                      << copChange << std::endl;
            std::cout << "FD dAng[" << t << "]/dCop[plate=" << i
                      << ",blur=" << b << "]:" << std::endl
                      << copChange_fd << std::endl;
            std::cout << "Diff:" << std::endl
                      << copChange - copChange_fd << std::endl;
            return false;
          }
        }
      }
    }
    std::cout << "Angular pose map passes!" << std::endl;

    return false;
  }

  for (int i = 0; i < recoverCops.size(); i++)
  {
    if (!equals(recoverCops[i], recoverCops_fd[i], 1e-8))
    {
      std::cout << "Recover CoPs " << i << " not equal!" << std::endl;
      std::cout << "Analytical:" << std::endl << recoverCops[i] << std::endl;
      std::cout << "FD:" << std::endl << recoverCops_fd[i] << std::endl;
      return false;
    }
  }

  return true;
}

bool testMultiMassLinearMapWithRandomTrajectory(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<int> collisionBodies,
    int numTimesteps)
{
  s_t dt = skel->getTimeStep();
  Eigen::MatrixXs qs = Eigen::MatrixXs::Zero(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs dqs = Eigen::MatrixXs::Zero(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs ddqs
      = Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps);
  Eigen::MatrixXs forces
      = 0.001
        * Eigen::MatrixXs::Random(collisionBodies.size() * 6, numTimesteps);
  std::vector<bool> probablyMissingGRF;
  int numMissing = 0;
  std::vector<int> missingIndices;
  for (int i = 0; i < numTimesteps; i++)
  {
    bool missing = i % 4 == 0;
    probablyMissingGRF.push_back(missing);
    if (missing)
    {
      numMissing++;
      missingIndices.push_back(i);
    }
  }

  // Generate q,dq from integrating the given accelerations
  qs.col(0) = skel->getRandomPose();
  dqs.col(0) = skel->getRandomVelocity();
  for (int i = 1; i < numTimesteps; i++)
  {
    dqs.col(i) = dqs.col(i - 1) + ddqs.col(i - 1) * dt;
    qs.col(i) = qs.col(i - 1) + dqs.col(i) * dt;
  }
  // Add some random noise
  qs += Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps) * 0.0001;
  dqs += Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps) * 0.001;

  skel->setLinearizedMasses(skel->getLinearizedMasses());
  skel->setGravity(Eigen::Vector3s(0, 0, -9.81));
  ResidualForceHelper helper(skel, collisionBodies);

  std::pair<Eigen::MatrixXs, Eigen::VectorXs> linear
      = helper.getMultiMassLinearSystem(
          dt, qs, dqs, ddqs, forces, probablyMissingGRF, 2);
  std::pair<Eigen::MatrixXs, Eigen::VectorXs> linear_fd
      = helper.finiteDifferenceMultiMassLinearSystem(
          dt, qs, dqs, ddqs, forces, probablyMissingGRF, 2);

  if (!equals(linear.second, linear_fd.second, 1e-8))
  {
    std::cout << "Multi-mass system b vector is not equal!" << std::endl;
    for (int t = 0; t < numTimesteps; t++)
    {
      Eigen::Vector3s pos = linear.second.segment<3>(t * 3);
      Eigen::Vector3s pos_fd = linear_fd.second.segment<3>(t * 3);
      if (!equals(pos, pos_fd, 1e-8))
      {
        std::cout << "Multi-mass system b vector pos not equal at t=" << t
                  << std::endl;
        Eigen::Matrix3s compare;
        compare.col(0) = pos;
        compare.col(1) = pos_fd;
        compare.col(2) = pos - pos_fd;
        std::cout << "Analytical - FD - Diff" << std::endl
                  << compare << std::endl;
        return false;
      }
    }
    return false;
  }

  const s_t tol = 1e-8;
  if (!equals(linear.first, linear_fd.first, tol))
  {
    std::cout << "Linear system A matrix is not equal!" << std::endl;
    Eigen::MatrixXs A = linear.first;
    Eigen::MatrixXs A_fd = linear_fd.first;

    /// Check the linear pose map
    for (int t = 0; t < numTimesteps; t++)
    {
      Eigen::Matrix3s posOffsetPos = A.block<3, 3>(t * 3, 0);
      Eigen::Matrix3s posOffsetPos_fd = A_fd.block<3, 3>(t * 3, 0);

      if (!equals(posOffsetPos, posOffsetPos_fd, tol))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dPos[" << t << "]/dOffsetPos:" << std::endl
                  << posOffsetPos << std::endl;
        std::cout << "FD dPos[" << t << "]/dOffsetPos:" << std::endl
                  << posOffsetPos_fd << std::endl;
        std::cout << "Extra (Analytical - FD):" << std::endl
                  << posOffsetPos - posOffsetPos_fd << std::endl;
        return false;
      }

      Eigen::Matrix3s posOffsetVel = A.block<3, 3>(t * 3, 3);
      Eigen::Matrix3s posOffsetVel_fd = A_fd.block<3, 3>(t * 3, 3);

      if (!equals(posOffsetVel, posOffsetVel_fd, tol))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dPos[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetVel << std::endl;
        std::cout << "FD dPos[" << t << "]/dOffsetVel:" << std::endl
                  << posOffsetVel_fd << std::endl;
        std::cout << "Diff:" << std::endl
                  << posOffsetVel - posOffsetVel_fd << std::endl;
        return false;
      }

      Eigen::Vector3s posOffsetInvMass = A.block<3, 1>(t * 3, 6);
      Eigen::Vector3s posOffsetInvMass_fd = A_fd.block<3, 1>(t * 3, 6);
      if (!equals(posOffsetInvMass, posOffsetInvMass_fd, tol))
      {
        std::cout << "Linear system error at t=" << t << std::endl;
        std::cout << "Analytical dPos[" << t << "]/dInvMass:" << std::endl
                  << posOffsetInvMass << std::endl;
        std::cout << "FD dPos[" << t << "]/dInvMass:" << std::endl
                  << posOffsetInvMass_fd << std::endl;
        std::cout << "Diff:" << std::endl
                  << posOffsetInvMass - posOffsetInvMass_fd << std::endl;
        return false;
      }

      for (int i = 0; i < skel->getNumScaleGroups(); i++)
      {
        Eigen::Vector3s posOffsetMassPerc = A.block<3, 1>(t * 3, 6 + i);
        Eigen::Vector3s posOffsetMassPerc_fd = A_fd.block<3, 1>(t * 3, 6 + i);
        if (!equals(posOffsetMassPerc, posOffsetMassPerc_fd, tol))
        {
          std::cout << "Linear system error at t=" << t << std::endl;
          std::cout << "Analytical dPos[" << t << "]/dMassPerc[" << i
                    << "]:" << std::endl
                    << posOffsetMassPerc << std::endl;
          std::cout << "FD dPos[" << t << "]/dMassPerc[" << i
                    << "]:" << std::endl
                    << posOffsetMassPerc_fd << std::endl;
          std::cout << "Diff:" << std::endl
                    << posOffsetMassPerc - posOffsetMassPerc_fd << std::endl;
          return false;
        }
      }

      int numVariables = 7 + skel->getNumScaleGroups();

      for (int i = 0; i < numMissing; i++)
      {
        Eigen::Matrix3s posOffsetResidual
            = A.block<3, 3>(t * 3, numVariables + i * 3);
        Eigen::Matrix3s posOffsetResidual_fd
            = A_fd.block<3, 3>(t * 3, numVariables + i * 3);

        if (!equals(posOffsetResidual, posOffsetResidual_fd, tol))
        {
          std::cout << "Linear system error at t=" << t << std::endl;
          std::cout << "Analytical dPos[" << t << "]/dLinResidual["
                    << missingIndices[i] << "]:" << std::endl
                    << posOffsetResidual << std::endl;
          std::cout << "FD dPos[" << t << "]/dLinResidual[" << missingIndices[i]
                    << "]:" << std::endl
                    << posOffsetResidual_fd << std::endl;
          std::cout << "Diff:" << std::endl
                    << posOffsetResidual - posOffsetResidual_fd << std::endl;
          return false;
        }
      }
    }
    std::cout << "Linear pose map passes!" << std::endl;

    return false;
  }

  // Test some random points to ensure that the linear map is in fact linear

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs random = Eigen::VectorXs::Random(linear.first.cols());

    // These should give the same result
    Eigen::VectorXs randomLinearOutput = linear.first * random + linear.second;
    Eigen::VectorXs randomTestOutput
        = helper.getMultiMassLinearSystemTestOutput(
            dt,
            random.segment<3>(0),
            random.segment<3>(3),
            random.segment(6 + numMissing * 3, 1 + skel->getNumScaleGroups()),
            random.segment(6, numMissing * 3),
            qs,
            dqs,
            ddqs,
            forces,
            probablyMissingGRF);
    if (!equals(randomLinearOutput, randomTestOutput, 1e-8))
    {
      std::cout << "Not linear! Diff norm: "
                << (randomLinearOutput - randomTestOutput).norm() << std::endl;
      return false;
    }
  }

  return true;
}

bool testWorldWrenchAssignment(std::shared_ptr<DynamicsInitialization> init)
{
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    for (int t = 0; t < init->poseTrials[trial].cols(); t++)
    {
      // Sum up all the force plate forces
      Eigen::Vector3s forcePlateSum = Eigen::Vector3s::Zero();
      for (int i = 0; i < init->forcePlateTrials[trial].size(); i++)
      {
        Eigen::Vector3s f = init->forcePlateTrials[trial][i].forces[t];
        Eigen::Vector3s moment = init->forcePlateTrials[trial][i].moments[t];
        (void)moment;
        Eigen::Vector3s cop
            = init->forcePlateTrials[trial][i].centersOfPressure[t];
        (void)cop;
        forcePlateSum += f;
      }

      // Sum up all the wrenches
      Eigen::Vector3s wrenchForceSum = Eigen::Vector3s::Zero();
      for (int i = 0; i < init->grfBodyNodes.size(); i++)
      {
        Eigen::Vector6s wrench
            = init->grfTrials[trial].col(t).segment<6>(i * 6);
        wrenchForceSum += wrench.tail<3>();
      }

      if (!equals(forcePlateSum, wrenchForceSum))
      {
        std::cout << "Got mismatched forces at time " << t << std::endl;
        Eigen::Matrix3s compare = Eigen::Matrix3s::Zero();
        compare.col(0) = forcePlateSum;
        compare.col(1) = wrenchForceSum;
        compare.col(2) = forcePlateSum - wrenchForceSum;
        std::cout << "Force plates - Wrenches - Diff" << std::endl
                  << compare << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool testRelationshipBetweenResidualAndLinear(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::shared_ptr<DynamicsInitialization> init)
{
  skel->setGravity(Eigen::Vector3s(0, -9.81, 0));
  skel->clearExternalForces();

  DynamicsFitter fitter(skel, init->grfBodyNodes, init->trackingMarkers);
  ResidualForceHelper residualHelper(skel, init->grfBodyIndices);
  SpatialNewtonHelper newtonHelper(skel);

  DynamicsFitProblemConfig config(skel);
  config.setIncludePoses(true);
  config.setIncludeMasses(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeBodyScales(true);
  // config.setIncludeMarkerOffsets(true);
  config.setResidualWeight(1.0);
  config.setResidualTorqueMultiple(1.0);
  config.setResidualUseL1(true);

  DynamicsFitProblem problem(init, skel, init->trackingMarkers, config);

  int totalAccTimesteps = 0;
  for (auto& block : problem.mBlocks)
  {
    for (int t = 0; t < block.len; t++)
    {
      int realT = block.start + t;
      if (realT > 0 && realT < init->poseTrials[block.trial].cols() - 1)
      {
        // Add force residual RMS errors to all the middle timesteps
        if (!init->probablyMissingGRF[block.trial][realT])
        {
          totalAccTimesteps++;
        }
      }
    }
  }

  s_t residualNorm = 0.0;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    std::vector<Eigen::Vector3s> comAccs = fitter.comAccelerations(init, trial);
    std::vector<Eigen::Vector3s> impliedForces
        = fitter.impliedCOMForces(init, trial, skel->getGravity());
    /*
    std::vector<Eigen::Vector3s> measuredForces
        = fitter.measuredGRFForces(init, trial);
    */

    for (int t = 1; t < init->poseTrials[trial].cols() - 1; t++)
    {
      if (init->probablyMissingGRF[trial][t])
      {
        continue;
      }

      s_t dt = init->trialTimesteps[trial];
      Eigen::VectorXs q = init->poseTrials[trial].col(t);
      Eigen::VectorXs dq = (init->poseTrials[trial].col(t)
                            - init->poseTrials[trial].col(t - 1))
                           / dt;
      Eigen::VectorXs ddq = (init->poseTrials[trial].col(t + 1)
                             - 2 * init->poseTrials[trial].col(t)
                             + init->poseTrials[trial].col(t - 1))
                            / (dt * dt);
      Eigen::VectorXs grf = init->grfTrials[trial].col(t);

      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);
      Eigen::Vector3s comAcc = skel->getCOMLinearAcceleration();
      Eigen::Vector3s impliedCOMAcc = comAccs[t];
      (void)impliedCOMAcc;
      /*
      if (!equals(comAcc, impliedCOMAcc, 1e-8))
      {
        std::cout << "Got mismatched COM acc's at time " << t << std::endl;
        Eigen::Matrix3s compare = Eigen::Matrix3s::Zero();
        compare.col(0) = comAcc;
        compare.col(1) = impliedCOMAcc;
        compare.col(2) = comAcc - impliedCOMAcc;
        std::cout << "Measured - Implied - Diff" << std::endl
                  << compare << std::endl;
        // return false;
      }
      */

      Eigen::Vector3s totalExternalForce
          = (comAcc - skel->getGravity()) * skel->getMass();

      Eigen::Vector6s residual
          = residualHelper.calculateResidual(q, dq, ddq, grf);
      // std::cout << "Test residual t=" << t << ": " << std::endl
      //           << residual << std::endl;

      s_t otherLoss = residualHelper.calculateResidualNorm(
          q,
          dq,
          ddq,
          grf,
          config.mResidualTorqueMultiple,
          config.mResidualUseL1);
      if (config.mResidualUseL1)
      {
        s_t loss = 0.0;
        loss += residual.head<3>().norm();
        loss += residual.tail<3>().norm();

        if (abs(loss - otherLoss) > 1e-10)
        {
          std::cout << "Different at time " << t << ": " << loss << " vs "
                    << otherLoss << " (" << (loss - otherLoss) << ")"
                    << std::endl;
        }

        residualNorm += loss / totalAccTimesteps;
      }
      else
      {
        residualNorm += residual.squaredNorm();
      }
      Eigen::Vector3s residualForce = residual.tail<3>();

      Eigen::Vector3s totalWithResidual = residualForce;
      for (int i = 0; i < grf.size() / 6; i++)
      {
        totalWithResidual += grf.segment<3>(i * 6 + 3);
      }

      if (!equals(totalWithResidual, totalExternalForce, 1e-8))
      {
        std::cout << "Got mismatched forces at time " << t << std::endl;
        Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 5);
        compare.col(0) = totalWithResidual;
        compare.col(1) = totalExternalForce;
        compare.col(2) = totalWithResidual - totalExternalForce;
        compare.col(3) = impliedForces[t];
        compare.col(4) = totalWithResidual - impliedForces[t];

        std::cout << "Fs + Residual : M*(a-g) : Diff : Implied : Diff"
                  << std::endl
                  << compare << std::endl;
        return false;
      }

      Eigen::Vector3s linearForceGap
          = newtonHelper.calculateLinearForceGap(q, dq, ddq, grf);
      Eigen::Vector3s totalWithLinearForceGap = linearForceGap;
      for (int i = 0; i < grf.size() / 6; i++)
      {
        totalWithLinearForceGap += grf.segment<3>(i * 6 + 3);
      }

      if (!equals(totalWithLinearForceGap, totalExternalForce, 1e-8))
      {
        std::cout << "Got mismatched forces at time " << t << std::endl;
        Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 5);
        compare.col(0) = totalWithLinearForceGap;
        compare.col(1) = totalExternalForce;
        compare.col(2) = totalWithLinearForceGap - totalExternalForce;
        compare.col(3) = impliedForces[t];
        compare.col(4) = totalWithResidual - impliedForces[t];

        std::cout << "Fs + Linear Gap : M*(a-g) : Diff : Implied : Diff"
                  << std::endl
                  << compare << std::endl;
        return false;
      }
    }
  }

  for (int blockIdx = 0; blockIdx < problem.mBlocks.size(); blockIdx++)
  {
    auto& block = problem.mBlocks[blockIdx];
    for (int t = 0; t < block.len; t++)
    {
      int realT = block.start + t;
      Eigen::VectorXs q = init->poseTrials[block.trial].col(realT);
      Eigen::VectorXs grf = init->grfTrials[block.trial].col(realT);
      if (!equals(q, Eigen::VectorXs(block.pos.col(t)), 1e-10))
      {
        std::cout
            << "Before unflatten(flatten()): Block pos doesn't match at t="
            << realT << " (block=" << blockIdx << ", t_in_block=" << t << ")"
            << std::endl;
        Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
        compare.col(0) = q;
        compare.col(1) = block.pos.col(t);
        compare.col(2) = q - block.pos.col(t);
        std::cout << "Init - Block - Diff" << std::endl << compare << std::endl;
        return false;
      }

      if (!equals(grf, Eigen::VectorXs(block.grf.col(t)), 1e-10))
      {
        std::cout
            << "Before unflatten(flatten()): Block grf doesn't match at t="
            << realT << " (block=" << blockIdx << ", t_in_block=" << t << ")"
            << std::endl;
        Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
        compare.col(0) = grf;
        compare.col(1) = block.grf.col(t);
        compare.col(2) = grf - block.grf.col(t);
        std::cout << "Init - Block - Diff" << std::endl << compare << std::endl;
        return false;
      }

      if (realT > 0 && realT < init->poseTrials[block.trial].cols() - 1)
      {
        s_t dt = init->trialTimesteps[block.trial];
        Eigen::VectorXs dq = (init->poseTrials[block.trial].col(realT)
                              - init->poseTrials[block.trial].col(realT - 1))
                             / dt;
        Eigen::VectorXs ddq = (init->poseTrials[block.trial].col(realT + 1)
                               - 2 * init->poseTrials[block.trial].col(realT)
                               + init->poseTrials[block.trial].col(realT - 1))
                              / (dt * dt);

        if (!equals(dq, Eigen::VectorXs(block.vel.col(t)), 1e-10))
        {
          std::cout
              << "Before unflatten(flatten()): Block vel doesn't match at t="
              << realT << " (block=" << blockIdx << ", t_in_block=" << t << ")"
              << std::endl;
          Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
          compare.col(0) = dq;
          compare.col(1) = block.vel.col(t);
          compare.col(2) = dq - block.vel.col(t);
          std::cout << "Init - Block - Diff" << std::endl
                    << compare << std::endl;
          return false;
        }

        if (!equals(ddq, Eigen::VectorXs(block.acc.col(t)), 1e-10))
        {
          std::cout
              << "Before unflatten(flatten()): Block acc doesn't match at t="
              << realT << " (block=" << blockIdx << ", t_in_block=" << t << ")"
              << std::endl;
          Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
          compare.col(0) = ddq;
          compare.col(1) = block.acc.col(t);
          compare.col(2) = ddq - block.acc.col(t);
          std::cout << "Init - Block - Diff" << std::endl
                    << compare << std::endl;
          return false;
        }
      }
    }
  }

  std::cout << "Residual norm: " << residualNorm << std::endl;

  Eigen::VectorXs originalMasses = skel->getLinkMasses();
  Eigen::VectorXs originalCOMs = skel->getGroupCOMs();
  Eigen::VectorXs originalInertias = skel->getGroupInertias();
  Eigen::VectorXs originalScales = skel->getBodyScales();

  problem.unflatten(problem.flatten());
  if (!equals(originalMasses, skel->getLinkMasses(), 1e-16))
  {
    std::cout << "Masses not preserved across flatten/unflatten!" << std::endl;
    return false;
  }
  if (!equals(originalCOMs, skel->getGroupCOMs(), 1e-16))
  {
    std::cout << "COMs not preserved across flatten/unflatten!" << std::endl;
    return false;
  }
  if (!equals(originalInertias, skel->getGroupInertias(), 1e-16))
  {
    std::cout << "Inertias not preserved across flatten/unflatten!"
              << std::endl;
    return false;
  }
  if (!equals(originalScales, skel->getBodyScales(), 1e-16))
  {
    std::cout << "Scales not preserved across flatten/unflatten!" << std::endl;
    return false;
  }
  for (int blockIdx = 0; blockIdx < problem.mBlocks.size(); blockIdx++)
  {
    auto& block = problem.mBlocks[blockIdx];
    for (int t = 0; t < block.len; t++)
    {
      int realT = block.start + t;
      Eigen::VectorXs q = init->poseTrials[block.trial].col(realT);
      Eigen::VectorXs grf = init->grfTrials[block.trial].col(realT);
      if (!equals(q, Eigen::VectorXs(block.pos.col(t)), 1e-10))
      {
        std::cout << "After unflatten(flatten()): Block pos doesn't match at t="
                  << realT << " (block=" << blockIdx << ", t_in_block=" << t
                  << ")" << std::endl;
        Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
        compare.col(0) = q;
        compare.col(1) = block.pos.col(t);
        compare.col(2) = q - block.pos.col(t);
        std::cout << "Init - Block - Diff" << std::endl << compare << std::endl;
        return false;
      }

      if (!equals(grf, Eigen::VectorXs(block.grf.col(t)), 1e-10))
      {
        std::cout << "After unflatten(flatten()): Block grf doesn't match at t="
                  << realT << " (block=" << blockIdx << ", t_in_block=" << t
                  << ")" << std::endl;
        Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
        compare.col(0) = grf;
        compare.col(1) = block.grf.col(t);
        compare.col(2) = grf - block.grf.col(t);
        std::cout << "Init - Block - Diff" << std::endl << compare << std::endl;
        return false;
      }

      if (realT > 0 && realT < init->poseTrials[block.trial].cols() - 1)
      {
        s_t dt = init->trialTimesteps[block.trial];
        Eigen::VectorXs dq = (init->poseTrials[block.trial].col(realT)
                              - init->poseTrials[block.trial].col(realT - 1))
                             / dt;
        Eigen::VectorXs ddq = (init->poseTrials[block.trial].col(realT + 1)
                               - 2 * init->poseTrials[block.trial].col(realT)
                               + init->poseTrials[block.trial].col(realT - 1))
                              / (dt * dt);

        if (!equals(dq, Eigen::VectorXs(block.vel.col(t)), 1e-10))
        {
          std::cout
              << "After unflatten(flatten()): Block vel doesn't match at t="
              << realT << " (block=" << blockIdx << ", t_in_block=" << t << ")"
              << std::endl;
          Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
          compare.col(0) = dq;
          compare.col(1) = block.vel.col(t);
          compare.col(2) = dq - block.vel.col(t);
          std::cout << "Init - Block - Diff" << std::endl
                    << compare << std::endl;
          return false;
        }

        if (!equals(ddq, Eigen::VectorXs(block.acc.col(t)), 1e-10))
        {
          std::cout
              << "After unflatten(flatten()): Block acc doesn't match at t="
              << realT << " (block=" << blockIdx << ", t_in_block=" << t << ")"
              << std::endl;
          Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(q.size(), 3);
          compare.col(0) = ddq;
          compare.col(1) = block.acc.col(t);
          compare.col(2) = ddq - block.acc.col(t);
          std::cout << "Init - Block - Diff" << std::endl
                    << compare << std::endl;
          return false;
        }
      }
    }
  }

  s_t problemLoss = problem.computeLoss(problem.flatten(), true);
  std::cout << "Problem loss: " << problemLoss << std::endl;

  // TODO: this doesn't seem to work on the CompleteHumanModel, and we don't
  // know why...
  if (abs(residualNorm - problemLoss) > 1e-10)
  {
    std::cout << "Problem loss: " << problemLoss << std::endl;
    std::cout << "abs(Residual norm - problemLoss): "
              << abs(residualNorm - problemLoss) << std::endl;
    std::cout << "abs(Residual norm - problemLoss) / abs(residualNorm): "
              << abs(residualNorm - problemLoss) / abs(residualNorm)
              << std::endl;
    EXPECT_EQ(residualNorm, problemLoss);
    return false;
  }

  return true;
}

std::shared_ptr<DynamicsInitialization> runEngine(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::shared_ptr<DynamicsInitialization> init,
    std::string modelPath,
    std::vector<std::string> trialNames,
    bool useReactionWheels = false,
    bool saveGUI = false,
    int maxTrialsToSolveMassOver = 4)
{
  // Have very loose bounds for scaling
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    skel->getBodyNode(i)->setScaleLowerBound(Eigen::Vector3s::Ones() * 0.3);
    skel->getBodyNode(i)->setScaleUpperBound(Eigen::Vector3s::Ones() * 3.0);
  }

  std::vector<Eigen::MatrixXs> originalTrajectories;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    originalTrajectories.push_back(init->poseTrials[trial]);
  }

  DynamicsFitter fitter(skel, init->grfBodyNodes, init->trackingMarkers);
  fitter.addJointBoundSlack(skel, 0.1);
  // fitter.boundPush(init);
  fitter.smoothAccelerations(init);

  // fitter.markMissingImpacts(init, 3, true);

  /*
  fitter.zeroLinearResidualsOnCOMTrajectory(init);
  fitter.multimassZeroLinearResidualsOnCOMTrajectory(init);

  s_t weightLinear = 1.0;
  s_t weightAngular = 1.0;
  s_t regularizeLinearResiduals = 0.1;
  s_t regularizeAngularResiduals = 0.1;
  s_t regularizeCopDriftCompensation = 1.0;
  int maxBuckets = 100;
  bool detectUnmeasuredTorque = false;
  int iterations = useReactionWheels ? 1 : 30;

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    Eigen::MatrixXs originalPoses = init->poseTrials[trial];
    for (int i = 0; i < iterations; i++)
    {
      bool commitCopDriftCompensation = i == iterations - 1;
      // this holds the mass constant, and re-jigs the trajectory to try to
      // make angular ACC's match more closely what was actually observed
      bool success = fitter.zeroLinearResidualsAndOptimizeAngular(
          init,
          trial,
          originalPoses,
          weightLinear,
          weightAngular,
          regularizeLinearResiduals,
          regularizeAngularResiduals,
          regularizeCopDriftCompensation,
          maxBuckets,
          useReactionWheels,
          commitCopDriftCompensation,
          detectUnmeasuredTorque);
      if (!success)
        break;
    }
    // Adjust the regularization target to match our newly solved trajectory, so
    // we're not trying to pull the root away from the solved trajectory
    init->regularizePosesTo[trial] = init->poseTrials[trial];

    fitter.recalibrateForcePlatesOffset(init, trial);
  }
  */

  bool shiftGRF = false;
  int maxShiftGRF = 4;
  int iterationsPerShift = 20;
  fitter.timeSyncAndInitializePipeline(
      init,
      useReactionWheels,
      shiftGRF,
      maxShiftGRF,
      iterationsPerShift,
      maxTrialsToSolveMassOver);

  /*
  // fitter.zeroLinearResidualsOnCOMTrajectory(init);
  fitter.multimassZeroLinearResidualsOnCOMTrajectory(init);

  // Regularize masses around the original values
  init->regularizeGroupMassesTo = skel->getGroupMasses();

  // if (!fitter.verifyLinearForceConsistency(init))
  // {
  //   std::cout << "Failed linear consistency check! Exiting early." <<
  //   std::endl; return init;
  // }
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    fitter.timeSyncTrialGRF(init, trial);
  }

  // Get rid of the rest of the angular residuals
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    Eigen::MatrixXs originalTrajectory = originalTrajectories[trial];
    for (int i = 0; i < 100; i++)
    {
      // this holds the mass constant, and re-jigs the trajectory to try to
      // make angular ACC's match more closely what was actually observed
      fitter.zeroLinearResidualsAndOptimizeAngular(
          init, trial, originalTrajectory, 1.0, 0.5, 0.1, 0.1, 150);
    }
    fitter.recalibrateForcePlates(init, trial);
  }

  // Recompute the marker offsets to minimize error
  fitter.optimizeMarkerOffsets(init);
  */

  auto secondPair = fitter.computeAverageRealForce(init);
  std::cout << "Avg GRF Force: " << secondPair.first << " N" << std::endl;
  std::cout << "Avg GRF Torque: " << secondPair.second << " Nm" << std::endl;

  /*
  if (!testRelationshipBetweenResidualAndLinear(skel, init))
  {
    std::cout << "The residual norm doesn't map!" << std::endl;
    return init;
  }
  */

  // skel->setGroupInertias(skel->getGroupInertias());

  // fitter.setCheckDerivatives(true);
  // fitter.setIterationLimit(100);
  // fitter.runOptimization(init, 0, 1, false, false, false, true, false,
  // false);

  // Just optimize the inertia regularizer
  // Seems to converge in 50-100 iters
  /*
  fitter.setIterationLimit(100);
  fitter.runSGDOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setResidualWeight(1.0)
          .setResidualUseL1(true)
          // .setLinearNewtonWeight(1.0)
          // .setLinearNewtonUseL1(true)
          .setIncludePoses(true)
      // .setVelAccImplicit(true)
  );
  */

  /*
  fitter.runIPOPTOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          // .setVelAccImplicit(true)
          .setMarkerWeight(1e4)
          .setMarkerUseL1(false)
          .setResidualWeight(1e-5)
          .setResidualUseL1(false)
          .setLinearNewtonWeight(1e-7)
          .setLinearNewtonUseL1(false)
          .setRegularizeSpatialAcc(1e-8)
          .setRegularizeSpatialAccUseL1(false)
          .setRegularizeCOMs(1e3)
          .setIncludePoses(true)
          .setIncludeCOMs(true));
  */

  /*
        s_t residualWeight,
      s_t markerWeight,
      bool includeMasses,
      bool includeCOMs,
      bool includeInertias,
      bool includeBodyScales,
      bool includePoses,
      bool includeMarkerOffsets,
      bool implicitVelAcc);
  */

  /*
  fitter.setIterationLimit(500);
  fitter.runSGDOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setRegularizeSpatialAcc(1e-5)
          .setIncludePoses(true));
  */

  /*
  fitter.runSGDOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setLinearNewtonWeight(1e-2)
          .setRegularizeSpatialAcc(1e-5)
          .setMarkerWeight(5.0)
          .setIncludeMasses(true)
          .setIncludePoses(true));
  */

  // fitter.setIterationLimit(150);
  // fitter.runIPOPTOptimization(
  //     init,
  //     DynamicsFitProblemConfig(skel).setDefaults(false).setIncludePoses(true));
  // // Re-optimize when we finish
  // fitter.optimizeSpatialResidualsOnCOMTrajectory(init);

  // // Run as L2 fitter.setIterationLimit(200);
  // fitter.runIPOPTOptimization(
  //     init,
  //     DynamicsFitProblemConfig(skel)
  //         .setDefaults()
  //         .setIncludeMasses(true)
  //         .setIncludeInertias(true)
  //         .setIncludePoses(true));

  int maxNumTrials = 3;
  (void)maxNumTrials;

  // // Re - run as L1
  fitter.setIterationLimit(150);
  // // fitter.runIPOPTOptimization(
  // //     init,
  // //     DynamicsFitProblemConfig(skel)
  // //         .setDefaults(true)
  // //         .setMaxNumTrials(maxNumTrials)
  // //         .setIncludeMarkerOffsets(true)
  // //         .setIncludePoses(true));

  fitter.setLBFGSHistoryLength(20);
  fitter.setIterationLimit(200);
  fitter.runIPOPTOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setDefaults(true)
          // Add extra slack to all the bounds
          .setMaxNumTrials(maxNumTrials)
          .setMaxNumBlocksPerTrial(20)
          .setIncludeMasses(true)
          // .setIncludeCOMs(true)
          // .setIncludeInertias(true)
          // .setPoseSubsetLen(6)
          // .setPoseSubsetStartIndex(0)
          // .setIncludeBodyScales(true)
          .setIncludeMarkerOffsets(true)
          .setIncludePoses(true));

  for (int i = 0; i < init->poseTrials.size(); i++)
  {
    if (init->includeTrialsInDynamicsFit[i])
    {
      fitter.runIPOPTOptimization(
          init,
          DynamicsFitProblemConfig(skel)
              .setDefaults(true)
              .setOnlyOneTrial(i)
              .setIncludePoses(true));
    }
  }

  // // Do a final polishing pass on the marker offsets
  // fitter.optimizeMarkerOffsets(init);

  // fitter.runIPOPTOptimization(
  //     init,
  //     DynamicsFitProblemConfig(skel)
  //         .setDefaults(true)
  //         .setConstrainResidualsZero(successOnAllResiduals)
  //         // .setVelAccImplicit(true)
  //         // .setIncludeMasses(true)
  //         // .setIncludeCOMs(true)
  //         // .setIncludeInertias(true)
  //         // .setIncludeBodyScales(true)
  //         // .setIncludeMarkerOffsets(true)
  //         .setIncludePoses(true));

  // fitter.setIterationLimit(50);
  // fitter.runSGDOptimization(
  //     init,
  //     DynamicsFitProblemConfig(skel)
  //         .setDefaults(true)
  //         .setIncludeMasses(true)
  //         .setIncludeCOMs(true)
  //         .setIncludeInertias(true)
  //         .setIncludeBodyScales(true)
  //         .setIncludeMarkerOffsets(true)
  //         .setIncludePoses(true));

  // // Reset force plates to 0-ish residuals
  // for (int trial = 0; trial < init->poseTrials.size(); trial++)
  // {
  //   bool successOnResiduals
  //       = fitter.optimizeSpatialResidualsOnCOMTrajectory(init, trial);
  //   if (successOnResiduals)
  //   {
  //     fitter.recalibrateForcePlates(init, trial);
  //   }
  // }

  /*
  fitter.setIterationLimit(50);
  fitter.runSGDOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setDefaults(true)
          .setIncludeMasses(true)
          .setIncludeCOMs(true)
          .setIncludeInertias(true)
          .setIncludeBodyScales(true)
          .setIncludePoses(true));
  fitter.zeroLinearResidualsOnCOMTrajectory(init);
  fitter.runSGDOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setDefaults(true)
          .setIncludeMasses(true)
          .setIncludeCOMs(true)
          .setIncludeInertias(true)
          .setIncludeBodyScales(true)
          .setIncludeMarkerOffsets(true)
          .setIncludePoses(true));
  fitter.zeroLinearResidualsOnCOMTrajectory(init);
  */

  /*
  fitter.setIterationLimit(200);
  fitter.runSGDOptimization(
      init, 2e-2, 50, true, true, true, true, false, true);

  fitter.setIterationLimit(50);
  fitter.runSGDOptimization(init, 2e-2, 50, true, true, true, true, true, true);
  */

  // RESTORE
  // for (int trial = 0; trial < init->originalPoses.size(); trial++)
  // {
  //   bool successOnResiduals = fitter.optimizeSpatialResidualsOnCOMTrajectory(
  //       init, trial, 5e-7); // 5e-9 is the practical limit
  //   if (successOnResiduals)
  //   {
  //     // For now, do nothing
  //     fitter.recalibrateForcePlates(init, trial);
  //   }
  //   else
  //   {
  //     successOnAllResiduals = false;
  //   }
  // }

  fitter.computePerfectGRFs(init);

  // bool consistent = fitter.checkPhysicalConsistency(init);
  // if (!consistent)
  // {
  //   std::cout << "ERROR: Physical consistency failed" << std::endl;
  //   // return init;
  // }

  // Try an explicit optimization at the end
  // fitter.setIterationLimit(100);
  // fitter.runIPOPTOptimization(
  //     init, 2e-2, 50, true, true, true, true, true, true, true);

  /*
  // Fine tune the positions, body scales, and marker offsets
  fitter.setIterationLimit(100);
  fitter.runOptimization(
      init, 2e-2, 100, false, false, false, true, true, true);
    */

  // fitter.optimizeSpatialResidualsOnCOMTrajectory(init);

  std::cout << "Post optimization mass: " << init->bodyMasses.sum() << " kg"
            << std::endl;
  std::cout << "Post optimization average ~GRF (Mass * 9.8): "
            << init->bodyMasses.sum() * 9.8 << " N" << std::endl;

  skel->setGroupMasses(init->initialGroupMasses);
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    auto* bodyNode = skel->getBodyNode(i);
    std::cout << "  \"" << bodyNode->getName() << "\": " << init->bodyMasses(i)
              << " kg (" << (init->bodyMasses(i) / bodyNode->getMass()) * 100
              << "% of original " << bodyNode->getMass() << " kg)" << std::endl;
    Eigen::Vector6s m = bodyNode->getInertia().getDimsAndEulerVector();
    std::cout << "      -> box dims (" << m(0) << "," << m(1) << "," << m(2)
              << ") and eulers (" << m(3) << "," << m(4) << "," << m(5) << ")"
              << std::endl;
  }
  skel->setLinkMasses(init->bodyMasses);

  std::cout << "Avg Marker RMSE: "
            << (fitter.computeAverageMarkerRMSE(init) * 100) << "cm"
            << std::endl;
  auto pair = fitter.computeAverageResidualForce(init);
  std::cout << "Avg Residual Force: " << pair.first << " N ("
            << (pair.first / secondPair.first) * 100 << "% of original "
            << secondPair.first << " N)" << std::endl;
  std::cout << "Avg Residual Torque: " << pair.second << " Nm ("
            << (pair.second / secondPair.second) * 100 << "% of original "
            << secondPair.second << " Nm)" << std::endl;
  std::cout << "Avg CoP movement in 'perfect' GRFs: "
            << fitter.computeAverageCOPChange(init) << " m" << std::endl;
  std::cout << "Avg force change in 'perfect' GRFs: "
            << fitter.computeAverageForceMagnitudeChange(init) << " N"
            << std::endl;

  for (int trial = 0; trial < init->probablyMissingGRF.size(); trial++)
  {
    int totalFrames = 0;
    for (int t = 0; t < init->probablyMissingGRF[trial].size(); t++)
    {
      if (init->probablyMissingGRF[trial][t])
      {
        totalFrames++;
      }
    }
    std::cout << "Trial " << trial << " missing GRF: " << totalFrames
              << std::endl;
  }

  if (saveGUI)
  {
    int trajectoryIndex = 0;

    std::cout << "Saving trajectory..." << std::endl;
    std::cout << "FPS: " << 1.0 / init->trialTimesteps[trajectoryIndex]
              << std::endl;

    // TODO: do we want to recreate the saving of a B3D file here?
    (void)modelPath;
    (void)trialNames;

    fitter.writeCSVData(
        "../../../javascript/src/data/movement2.csv",
        init,
        trajectoryIndex,
        false);
    fitter.saveDynamicsToGUI(
        "../../../javascript/src/data/movement2.bin",
        init,
        trajectoryIndex,
        (int)round(1.0 / init->trialTimesteps[trajectoryIndex]));
  }

  // OpenSimParser::saveOsimScalingXMLFile("subject", skel, skel->getMass(),
  // skel->getHeight(skel->getRestPositions()), "./unscaled_generic.osim", "",
  // "./scaled.osim", "../../../scaling.xml");
  // OpenSimParser::saveOsimInverseDynamicsProcessedForcesXMLFile("subject",
  // init->grfBodyNodes, "./grf_forces.mot", "../../../id_forces.xml");
  // std::vector<double> timesteps;
  // for (int i = 0; i < init->grfTrials[0].cols(); i++) {
  //   timesteps.push_back(init->trialTimesteps[0] * i);
  // }
  // OpenSimParser::saveMot(skel, "../../../mot.mot", timesteps,
  // init->poseTrials[0]);
  // OpenSimParser::saveProcessedGRFMot("../../../grf_forces.mot", timesteps,
  // init->grfBodyNodes, init->groundHeight[0], init->grfTrials[0]);

  const bool exportOsim = false;
  if (exportOsim)
  {
    std::vector<std::vector<s_t>> timestamps;
    for (int i = 0; i < init->poseTrials.size(); i++)
    {
      timestamps.emplace_back();
      for (int t = 0; t < init->poseTrials[i].cols(); t++)
      {
        timestamps[i].push_back((s_t)t * init->trialTimesteps[i]);
      }
    }
    for (int i = 0; i < init->poseTrials.size(); i++)
    {
      std::cout << "Saving IK Mot " << i << std::endl;
      OpenSimParser::saveMot(
          skel,
          "./_ik" + std::to_string(i) + ".mot",
          timestamps[i],
          init->poseTrials[i]);
      std::cout << "Saving GRF Mot " << i << std::endl;
      OpenSimParser::saveProcessedGRFMot(
          "./_grf" + std::to_string(i) + ".mot",
          timestamps[i],
          init->grfBodyNodes,
          skel,
          init->poseTrials[i],
          init->forcePlateTrials[i],
          init->grfTrials[i]);
    }
    for (int i = 0; i < init->poseTrials.size(); i++)
    {
      std::cout << "Saving OpenSim ID Forces " << i << " XML" << std::endl;
      OpenSimParser::saveOsimInverseDynamicsProcessedForcesXMLFile(
          "test_name",
          init->grfBodyNodes,
          "_grf" + std::to_string(i) + ".mot",
          "./_external_forces.xml");
      std::cout << "Saving OpenSim ID XML" << std::endl;
      OpenSimParser::saveOsimInverseDynamicsXMLFile(
          "trial",
          "Models/optimized_scale_and_markers.osim",
          "./_ik.mot",
          "./_external_forces.xml",
          "_id.sto",
          "_id_body_forces.sto",
          "./_id_setup.xml",
          0,
          init->trialTimesteps[i] * init->poseTrials[i].cols());
    }
  }

  // Attempt writing out the data
  // fitter.writeCSVData("../../../data/grf/Subject4/motion.csv", init, 0);
  // MJCFExporter exporter;
  // exporter.writeSkeleton("../../../data/grf/Subject4/model.mjcf", skel);

  return init;
}

std::shared_ptr<DynamicsInitialization> createInitialization(
    std::shared_ptr<dynamics::Skeleton> skel,
    MarkerMap markerMap,
    std::vector<std::string> trackingMarkers,
    std::vector<std::string> footNames,
    std::vector<std::string> motFiles,
    std::vector<std::string> c3dFiles,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    int limitTrialSizes = -1,
    int trialStartOffset = 0)
{
  std::vector<Eigen::MatrixXs> poseTrials;
  std::vector<C3D> c3ds;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<int> framesPerSecond;
  std::vector<std::vector<ForcePlate>> forcePlateTrials;

  for (int i = 0; i < motFiles.size(); i++)
  {
    OpenSimMot mot = OpenSimParser::loadMot(skel, motFiles[i]);
    poseTrials.push_back(mot.poses);
  }

  for (std::string& path : c3dFiles)
  {
    C3D c3d = C3DLoader::loadC3D(path);
    c3ds.push_back(c3d);
    markerObservationTrials.push_back(c3d.markerTimesteps);
    forcePlateTrials.push_back(c3d.forcePlates);
    framesPerSecond.push_back(c3d.framesPerSecond);
  }

  for (int i = 0; i < trcFiles.size(); i++)
  {
    OpenSimTRC trc = OpenSimParser::loadTRC(trcFiles[i]);
    framesPerSecond.push_back(trc.framesPerSecond);
    markerObservationTrials.push_back(trc.markerTimesteps);

    if (i < grfFiles.size())
    {
      std::vector<ForcePlate> grf
          = OpenSimParser::loadGRF(grfFiles[i], trc.timestamps);
      forcePlateTrials.push_back(grf);
    }
    else
    {
      forcePlateTrials.emplace_back();
    }
  }

  // This code trims all the timesteps down, if we asked for that
  if (limitTrialSizes > 0 || trialStartOffset > 0)
  {
    std::vector<Eigen::MatrixXs> trimmedPoseTrials;
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        trimmedMarkerObservationTrials;
    std::vector<std::vector<ForcePlate>> trimmedForcePlateTrials;

    for (int trial = 0; trial < poseTrials.size(); trial++)
    {
      // TODO: handle edge cases

      trimmedPoseTrials.push_back(poseTrials[trial].block(
          0, trialStartOffset, poseTrials[trial].rows(), limitTrialSizes));

      std::vector<std::map<std::string, Eigen::Vector3s>> markerSubset;
      for (int t = trialStartOffset; t < trialStartOffset + limitTrialSizes;
           t++)
      {
        markerSubset.push_back(markerObservationTrials[trial][t]);
      }
      trimmedMarkerObservationTrials.push_back(markerSubset);

      std::vector<ForcePlate> trimmedPlates;
      for (int i = 0; i < forcePlateTrials[trial].size(); i++)
      {
        ForcePlate& toCopy = forcePlateTrials[trial][i];
        ForcePlate trimmedPlate;

        trimmedPlate.corners = toCopy.corners;
        trimmedPlate.worldOrigin = toCopy.worldOrigin;

        for (int t = trialStartOffset; t < trialStartOffset + limitTrialSizes;
             t++)
        {
          trimmedPlate.centersOfPressure.push_back(toCopy.centersOfPressure[t]);
          trimmedPlate.forces.push_back(toCopy.forces[t]);
          trimmedPlate.moments.push_back(toCopy.moments[t]);
        }

        trimmedPlates.push_back(trimmedPlate);
      }
      trimmedForcePlateTrials.push_back(trimmedPlates);
    }

    poseTrials = trimmedPoseTrials;
    markerObservationTrials = trimmedMarkerObservationTrials;
    forcePlateTrials = trimmedForcePlateTrials;
  }

  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    auto report = MarkerFixer::generateDataErrorsReport(
        markerObservationTrials[i], 1.0 / (s_t)framesPerSecond[i]);
    markerObservationTrials[i] = report->markerObservationsAttemptedFixed;
  }

  std::vector<dynamics::BodyNode*> footNodes;
  for (std::string& name : footNames)
  {
    footNodes.push_back(skel->getBodyNode(name));
  }

  // Run the joints engine
  MarkerFitter fitter(skel, markerMap);

  std::vector<MarkerInitialization> kinematicInits;
  for (int trial = 0; trial < poseTrials.size(); trial++)
  {
    // 1. Find the initial scaling + IK
    MarkerInitialization fitterInit;
    fitterInit.poses = poseTrials[trial];
    fitterInit.groupScales = skel->getGroupScales();
    fitterInit.updatedMarkerMap = markerMap;

    std::vector<bool> newClip;
    for (int t = 0; t < poseTrials[trial].cols(); t++)
    {
      newClip.push_back(t == 0);
    }

    // 2. Find the joint centers
    // fitter.setJointSphereFitSGDIterations(50); // TODO comment out in Release
    // build
    fitter.setJointSphereFitSGDIterations(50);
    fitter.setJointAxisFitSGDIterations(50);
    fitter.findJointCenters(
        fitterInit, newClip, markerObservationTrials[trial]);
    fitter.findAllJointAxis(
        fitterInit, newClip, markerObservationTrials[trial]);
    fitter.computeJointConfidences(fitterInit, markerObservationTrials[trial]);

    kinematicInits.push_back(fitterInit);
  }

  std::shared_ptr<DynamicsInitialization> init
      = DynamicsFitter::createInitialization(
          skel,
          kinematicInits,
          trackingMarkers,
          footNodes,
          forcePlateTrials,
          framesPerSecond,
          markerObservationTrials);

  DynamicsFitter dynamicsFitter(
      skel, init->grfBodyNodes, init->trackingMarkers);
  // dynamicsFitter.estimateFootGroundContactsWithHeightHeuristic(init);
  dynamicsFitter.estimateFootGroundContactsWithStillness(init);

  return init;
}

std::shared_ptr<DynamicsInitialization> runEngine(
    std::string modelPath,
    std::vector<std::string> footNames,
    std::vector<std::string> motFiles,
    std::vector<std::string> c3dFiles,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    int limitTrialLength = -1,
    int trialStartOffset = 0,
    bool useReactionWheels = false,
    bool saveGUI = false,
    bool simplify = false,
    int maxTrialsToSolveMassOver = 4)
{
  OpenSimFile standard = OpenSimParser::parseOsim(modelPath);
  standard.skeleton->zeroTranslationInCustomFunctions();
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->autodetectScaleGroupAxisFlips(2);
  if (standard.skeleton->getBodyNode("hand_r") != nullptr)
  {
    standard.skeleton->setScaleGroupUniformScaling(
        standard.skeleton->getBodyNode("hand_r"));
  }
  standard.skeleton->autogroupSymmetricPrefixes("ulna", "radius");
  standard.skeleton->setPositionLowerLimit(0, -M_PI);
  standard.skeleton->setPositionUpperLimit(0, M_PI);
  standard.skeleton->setPositionLowerLimit(1, -M_PI);
  standard.skeleton->setPositionUpperLimit(1, M_PI);
  standard.skeleton->setPositionLowerLimit(2, -M_PI);
  standard.skeleton->setPositionUpperLimit(2, M_PI);
  // TODO: limit COM for the centerline bodies

  standard.skeleton->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // TODO: enable this if you need a specific starting mass for your subject for
  // a test s_t mass = standard.skeleton->getLinkMasses().sum(); s_t scaleBy
  // = 78.5 / mass; standard.skeleton->setLinkMasses(
  //     standard.skeleton->getLinkMasses() * scaleBy);

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      limitTrialLength,
      trialStartOffset);

  std::vector<std::string> trialNames;
  for (std::string& mot : motFiles)
  {
    trialNames.push_back(mot);
  }
  for (std::string& c3d : c3dFiles)
  {
    trialNames.push_back(c3d);
  }

  if (simplify)
  {
    std::map<std::string, std::string> mergeBodiesInto;
    std::shared_ptr<dynamics::Skeleton> simplified
        = standard.skeleton->simplifySkeleton("simplified", mergeBodiesInto);
    std::shared_ptr<DynamicsInitialization> simplifiedInit
        = DynamicsFitter::retargetInitialization(
            standard.skeleton, simplified, init);
    return runEngine(
        simplified,
        simplifiedInit,
        modelPath,
        trialNames,
        useReactionWheels,
        saveGUI,
        maxTrialsToSolveMassOver);
  }
  else
  {
    return runEngine(
        standard.skeleton,
        init,
        modelPath,
        trialNames,
        useReactionWheels,
        saveGUI,
        maxTrialsToSolveMassOver);
  }
}

std::pair<std::vector<MarkerInitialization>, OpenSimFile> runMarkerFitter(
    std::string modelPath,
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        markerObservationTrials,
    std::vector<int> framesPerSecond,
    std::vector<std::vector<ForcePlate>> forcePlates,
    s_t massKg,
    s_t heightM,
    std::string sex,
    bool saveGUI = false,
    bool runGUI = false)
{
  (void)forcePlates;
  (void)saveGUI;
  (void)runGUI;

  OpenSimFile standard = OpenSimParser::parseOsim(modelPath);
  standard.skeleton->zeroTranslationInCustomFunctions();
  standard.skeleton->autogroupSymmetricSuffixes();
  if (standard.skeleton->getBodyNode("hand_r") != nullptr)
  {
    standard.skeleton->setScaleGroupUniformScaling(
        standard.skeleton->getBodyNode("hand_r"));
  }
  standard.skeleton->autogroupSymmetricPrefixes("ulna", "radius");
  standard.skeleton->setPositionLowerLimit(0, -M_PI);
  standard.skeleton->setPositionUpperLimit(0, M_PI);
  standard.skeleton->setPositionLowerLimit(1, -M_PI);
  standard.skeleton->setPositionUpperLimit(1, M_PI);
  standard.skeleton->setPositionLowerLimit(2, -M_PI);
  standard.skeleton->setPositionUpperLimit(2, M_PI);

  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markerList;
  for (auto& pair : standard.markersMap)
  {
    markerList.push_back(pair.second);
  }
  // Do some sanity checks
  if (!verifySkeletonMarkerJacobians(standard.skeleton, markerList))
    return std::make_pair(std::vector<MarkerInitialization>(), standard);
  std::shared_ptr<dynamics::Skeleton> skelBallJoints
      = standard.skeleton->convertSkeletonToBallJoints();
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> ballMarkerList;
  for (auto& pair : standard.markersMap)
  {
    markerList.push_back(std::make_pair(
        skelBallJoints->getBodyNode(pair.second.first->getName()),
        pair.second.second));
  }
  if (!verifySkeletonMarkerJacobians(skelBallJoints, ballMarkerList))
    return std::make_pair(std::vector<MarkerInitialization>(), standard);

  // Create MarkerFitter
  MarkerFitter fitter(standard.skeleton, standard.markersMap);
  fitter.setInitialIKSatisfactoryLoss(0.005);
  fitter.setInitialIKMaxRestarts(50);
  fitter.setIterationLimit(500);
  if (standard.anatomicalMarkers.size() > 10)
  {
    // If there are at least 10 tracking markers
    std::cout << "Setting tracking markers based on OSIM model." << std::endl;
    fitter.setTrackingMarkers(standard.trackingMarkers);
  }
  else
  {
    // Set all the triads to be tracking markers, instead of anatomical
    std::cout << "WARNING!! Guessing tracking markers." << std::endl;
    fitter.setTriadsToTracking();
  }
  // This is 1.0x the values in the default code
  fitter.setRegularizeAnatomicalMarkerOffsets(10.0);
  // This is 1.0x the default value
  fitter.setRegularizeTrackingMarkerOffsets(0.05);
  // These are 2x the values in the default code
  // fitter.setMinSphereFitScore(3e-5 * 2)
  // fitter.setMinAxisFitScore(6e-5 * 2)
  fitter.setMinSphereFitScore(0.01);
  fitter.setMinAxisFitScore(0.001);
  // Default max joint weight is 0.5, so this is 2x the default value
  fitter.setMaxJointWeight(1.0);

  // Try with a joint force field at 10cm
  // fitter.setJointForceFieldThresholdDistance(0.1);
  // fitter.setJointForceFieldSoftness(10.0);

  // Create Anthropometric prior
  std::shared_ptr<Anthropometrics> anthropometrics
      = Anthropometrics::loadFromFile(
          "dart://sample/osim/ANSUR/ANSUR_metrics.xml");
  std::vector<std::string> cols = anthropometrics->getMetricNames();
  cols.push_back("Weightlbs");
  cols.push_back("Heightin");
  std::shared_ptr<MultivariateGaussian> gauss;
  if (sex == "male")
  {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_MALE_Public.csv",
        cols,
        0.001); // mm -> m
  }
  else if (sex == "female")
  {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_FEMALE_Public.csv",
        cols,
        0.001); // mm -> m
  }
  else
  {
    gauss = MultivariateGaussian::loadFromCSV(
        "dart://sample/osim/ANSUR/ANSUR_II_BOTH_Public.csv",
        cols,
        0.001); // mm -> m
  }
  std::map<std::string, s_t> observedValues;
  observedValues["Weightlbs"] = massKg * 2.204 * 0.001;
  observedValues["Heightin"] = heightM * 39.37 * 0.001;
  gauss = gauss->condition(observedValues);
  anthropometrics->setDistribution(gauss);
  fitter.setAnthropometricPrior(anthropometrics, 0.1);
  anthropometrics->getLogPDF(standard.skeleton);

  std::vector<std::shared_ptr<MarkersErrorReport>> reports;
  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    std::shared_ptr<MarkersErrorReport> report
        = fitter.generateDataErrorsReport(
            markerObservationTrials[i], 1.0 / (s_t)framesPerSecond[i]);
    for (std::string& warning : report->warnings)
    {
      std::cout << "DATA WARNING: " << warning << std::endl;
    }
    markerObservationTrials[i] = report->markerObservationsAttemptedFixed;
    reports.push_back(report);
  }

  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    if (!fitter.checkForEnoughMarkers(markerObservationTrials[i]))
    {
      std::cout << "Input files don't have enough markers that match the "
                   "OpenSim model! Aborting."
                << std::endl;
      return std::make_pair(std::vector<MarkerInitialization>(), standard);
    }
  }

  std::vector<MarkerInitialization> results
      = fitter.runMultiTrialKinematicsPipeline(
          markerObservationTrials,
          InitialMarkerFitParams()
              .setMaxTrialsToUseForMultiTrialScaling(5)
              .setMaxTimestepsToUseForMultiTrialScaling(4000),
          150);

  bool anySwapped = false;
  for (int i = 0; i < markerObservationTrials.size(); i++)
  {
    if (fitter.checkForFlippedMarkers(
            markerObservationTrials[i], results[i], reports[i]))
    {
      anySwapped = true;
      markerObservationTrials[i] = reports[i]->markerObservationsAttemptedFixed;
    }
  }
  if (anySwapped)
  {
    std::cout
        << "******** Unfortunately, it looks like some markers were swapped in "
           "the uploaded data, so we have to run the whole pipeline "
           "again with unswapped markers. ********"
        << std::endl;

    results = fitter.runMultiTrialKinematicsPipeline(
        markerObservationTrials,
        InitialMarkerFitParams()
            .setMaxTrialsToUseForMultiTrialScaling(5)
            .setMaxTimestepsToUseForMultiTrialScaling(4000),
        150);
  }

  for (int i = 0; i < reports.size(); i++)
  {
    std::cout << "Trial " << std::to_string(i) << std::endl;
    for (std::string& warning : reports[i]->warnings)
    {
      std::cout << "Warning: " << warning << std::endl;
    }
    for (std::string& info : reports[i]->info)
    {
      std::cout << "Info: " << info << std::endl;
    }
  }

  IKErrorReport finalKinematicsReport(
      standard.skeleton,
      results[0].updatedMarkerMap,
      results[0].poses,
      markerObservationTrials[0],
      anthropometrics);

  std::cout << "Final kinematic fit report:" << std::endl;
  finalKinematicsReport.printReport(5);

  std::cout << "Final marker locations: " << std::endl;
  for (auto& pair : results[0].updatedMarkerMap)
  {
    Eigen::Vector3s offset = pair.second.second;
    std::cout << pair.first << ": " << pair.second.first->getName() << ", "
              << offset(0) << " " << offset(1) << " " << offset(2) << std::endl;
  }

  /*
  std::cout << "Saving marker error report" << std::endl;
  finalKinematicsReport.saveCSVMarkerErrorReport(
      "./_ik_per_marker_error_report.csv");
  std::vector<std::vector<s_t>> timestamps;
  for (int i = 0; i < results.size(); i++)
  {
    timestamps.emplace_back();
    for (int t = 0; t < results[i].poses.cols(); t++)
    {
      timestamps[i].push_back((s_t)t * (1.0 / framesPerSecond[i]));
    }
  }
  for (int i = 0; i < results.size(); i++)
  {
    std::cout << "Saving IK Mot " << i << std::endl;
    OpenSimParser::saveMot(
        standard.skeleton,
        "./_ik" + std::to_string(i) + ".mot",
        timestamps[i],
        results[i].poses);
    std::cout << "Saving GRF Mot " << i << std::endl;
    OpenSimParser::saveGRFMot(
        "./_grf" + std::to_string(i) + ".mot", timestamps[i], forcePlates[i]);
    std::cout << "Saving TRC " << i << std::endl;
    std::cout << "timestamps[i]: " << timestamps[i].size() << std::endl;
    std::cout << "markerObservationTrials[i]: "
              << markerObservationTrials[i].size() << std::endl;
    OpenSimParser::saveTRC(
        "./_markers" + std::to_string(i) + ".trc",
        timestamps[i],
        markerObservationTrials[i]);
  }
  std::vector<std::string> markerNames;
  for (auto& pair : standard.markersMap)
  {
    markerNames.push_back(pair.first);
  }
  std::cout << "Saving OpenSim IK XML" << std::endl;
  OpenSimParser::saveOsimInverseKinematicsXMLFile(
      "trial",
      markerNames,
      "Models/optimized_scale_and_markers.osim",
      "./test.trc",
      "_ik_by_opensim.mot",
      "./_ik_setup.xml");
  // TODO: remove me
  for (int i = 0; i < results.size(); i++)
  {
    std::cout << "Saving OpenSim ID Forces " << i << " XML" << std::endl;
    OpenSimParser::saveOsimInverseDynamicsForcesXMLFile(
        "test_name",
        standard.skeleton,
        results[i].poses,
        forcePlates[i],
        "name_grf.mot",
        "./_external_forces.xml");
  }
  std::cout << "Saving OpenSim ID XML" << std::endl;
  OpenSimParser::saveOsimInverseDynamicsXMLFile(
      "trial",
      "Models/optimized_scale_and_markers.osim",
      "./_ik.mot",
      "./_external_forces.xml",
      "_id.sto",
      "_id_body_forces.sto",
      "./_id_setup.xml");

  if (saveGUI)
  {
    std::cout << "Saving trajectory..." << std::endl;
    std::cout << "FPS: " << framesPerSecond[0] << std::endl;
    std::cout << "Force plates len: " << forcePlates.size() << std::endl;
    fitter.saveTrajectoryAndMarkersToGUI(
        "../../../javascript/src/data/movement2.bin",
        results[0],
        markerObservationTrials[0],
        framesPerSecond[0],
        forcePlates[0]);
  }

  if (runGUI)
  {
    // Target markers
    std::shared_ptr<server::GUIWebsocketServer> server
        = std::make_shared<server::GUIWebsocketServer>();
    server->serve(8070);
    // , &scaled, goldPoses
    fitter.debugTrajectoryAndMarkersToGUI(
        server, results[0], markerObservationTrials[0], forcePlates[0]);
    server->blockWhileServing();
  }
  */

  return std::make_pair(results, standard);
}

std::shared_ptr<DynamicsInitialization> runEndToEnd(
    std::string modelPath,
    std::vector<std::string> footNames,
    std::vector<std::string> c3dFiles,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    int limitTrialSizes = -1,
    int trialStartOffset = 0,
    bool saveGUI = false,
    bool simplify = false,
    int maxTrialsToSolveMassOver = 4)
{
  std::vector<C3D> c3ds;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<int> framesPerSecond;
  std::vector<std::vector<ForcePlate>> forcePlateTrials;

  for (std::string& path : c3dFiles)
  {
    C3D c3d = C3DLoader::loadC3D(path);
    c3ds.push_back(c3d);
    markerObservationTrials.push_back(c3d.markerTimesteps);
    forcePlateTrials.push_back(c3d.forcePlates);
    framesPerSecond.push_back(c3d.framesPerSecond);
  }

  for (int i = 0; i < trcFiles.size(); i++)
  {
    OpenSimTRC trc = OpenSimParser::loadTRC(trcFiles[i]);
    framesPerSecond.push_back(trc.framesPerSecond);
    markerObservationTrials.push_back(trc.markerTimesteps);

    if (i < grfFiles.size())
    {
      std::vector<ForcePlate> grf
          = OpenSimParser::loadGRF(grfFiles[i], trc.timestamps);
      forcePlateTrials.push_back(grf);
    }
    else
    {
      forcePlateTrials.emplace_back();
    }
  }

  // This code trims all the timesteps down, if we asked for that
  if (limitTrialSizes > 0 || trialStartOffset > 0)
  {
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        trimmedMarkerObservationTrials;
    std::vector<std::vector<ForcePlate>> trimmedForcePlateTrials;

    for (int trial = 0; trial < markerObservationTrials.size(); trial++)
    {
      std::vector<std::map<std::string, Eigen::Vector3s>> markerSubset;
      for (int t = trialStartOffset; t < trialStartOffset + limitTrialSizes;
           t++)
      {
        markerSubset.push_back(markerObservationTrials[trial][t]);
      }
      trimmedMarkerObservationTrials.push_back(markerSubset);

      std::vector<ForcePlate> trimmedPlates;
      for (int i = 0; i < forcePlateTrials[trial].size(); i++)
      {
        ForcePlate& toCopy = forcePlateTrials[trial][i];
        ForcePlate trimmedPlate;

        trimmedPlate.corners = toCopy.corners;
        trimmedPlate.worldOrigin = toCopy.worldOrigin;

        for (int t = trialStartOffset; t < trialStartOffset + limitTrialSizes;
             t++)
        {
          trimmedPlate.centersOfPressure.push_back(toCopy.centersOfPressure[t]);
          trimmedPlate.forces.push_back(toCopy.forces[t]);
          trimmedPlate.moments.push_back(toCopy.moments[t]);
        }

        trimmedPlates.push_back(trimmedPlate);
      }
      trimmedForcePlateTrials.push_back(trimmedPlates);
    }

    markerObservationTrials = trimmedMarkerObservationTrials;
    forcePlateTrials = trimmedForcePlateTrials;
  }

  auto kinematicResults = runMarkerFitter(
      modelPath,
      markerObservationTrials,
      framesPerSecond,
      forcePlateTrials,
      // 62.6,
      // 1.68,
      // "unknown");
      72.16,
      1.83,
      "male");

  OpenSimFile standard = kinematicResults.second;
  standard.skeleton->setGroupScales(kinematicResults.first[0].groupScales);
  standard.skeleton->setGravity(Eigen::Vector3s(0, -9.81, 0));

  std::vector<dynamics::BodyNode*> footNodes;
  for (std::string& name : footNames)
  {
    footNodes.push_back(standard.skeleton->getBodyNode(name));
  }

  std::vector<MarkerInitialization> kinematicInits = kinematicResults.first;

  std::shared_ptr<DynamicsInitialization> init
      = DynamicsFitter::createInitialization(
          standard.skeleton,
          kinematicInits,
          standard.trackingMarkers,
          footNodes,
          forcePlateTrials,
          framesPerSecond,
          markerObservationTrials);

  DynamicsFitter dynamicsFitter(
      standard.skeleton, init->grfBodyNodes, init->trackingMarkers);
  // dynamicsFitter.estimateFootGroundContactsWithHeightHeuristic(init);
  dynamicsFitter.estimateFootGroundContactsWithStillness(init);

  std::vector<std::string> trialNames;
  for (std::string& trc : trcFiles)
  {
    trialNames.push_back(trc);
  }
  for (std::string& c3d : c3dFiles)
  {
    trialNames.push_back(c3d);
  }

  if (simplify)
  {
    std::map<std::string, std::string> mergeBodiesInto;
    std::shared_ptr<dynamics::Skeleton> simplified
        = standard.skeleton->simplifySkeleton("simplified", mergeBodiesInto);
    std::shared_ptr<DynamicsInitialization> simplifiedInit
        = DynamicsFitter::retargetInitialization(
            standard.skeleton, simplified, init);
    return runEngine(
        simplified,
        simplifiedInit,
        modelPath,
        trialNames,
        saveGUI,
        maxTrialsToSolveMassOver);
  }
  else
  {
    return runEngine(
        standard.skeleton,
        init,
        modelPath,
        trialNames,
        saveGUI,
        maxTrialsToSolveMassOver);
  }
}

bool verifyResidualElimination(
    std::string modelPath,
    std::vector<std::string> footNames,
    std::vector<std::string> motFiles,
    std::vector<std::string> c3dFiles,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    int limitTrialLength = -1,
    int trialStartOffset = 0)
{
  OpenSimFile standard = OpenSimParser::parseOsim(modelPath);
  standard.skeleton->zeroTranslationInCustomFunctions();
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->autodetectScaleGroupAxisFlips(2);
  if (standard.skeleton->getBodyNode("hand_r") != nullptr)
  {
    standard.skeleton->setScaleGroupUniformScaling(
        standard.skeleton->getBodyNode("hand_r"));
  }
  standard.skeleton->autogroupSymmetricPrefixes("ulna", "radius");
  standard.skeleton->setPositionLowerLimit(0, -M_PI);
  standard.skeleton->setPositionUpperLimit(0, M_PI);
  standard.skeleton->setPositionLowerLimit(1, -M_PI);
  standard.skeleton->setPositionUpperLimit(1, M_PI);
  standard.skeleton->setPositionLowerLimit(2, -M_PI);
  standard.skeleton->setPositionUpperLimit(2, M_PI);
  // TODO: limit COM for the centerline bodies

  standard.skeleton->setGravity(Eigen::Vector3s(0, -9.81, 0));

  std::shared_ptr<dynamics::Skeleton> skel = standard.skeleton;

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      limitTrialLength,
      trialStartOffset);

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    for (int t = 0; t < init->poseTrials[trial].cols(); t++)
    {
      init->poseTrials[trial].col(t)
          = standard.skeleton->convertPositionsToBallSpace(
              init->poseTrials[trial].col(t));
    }
  }

  DynamicsFitter fitter(skel, init->grfBodyNodes, init->trackingMarkers);
  fitter.smoothAccelerations(init);
  fitter.zeroLinearResidualsOnCOMTrajectory(init);

  ResidualForceHelper helper(skel, init->grfBodyIndices);

  std::vector<std::vector<std::vector<Eigen::Vector6s>>>
      originalTrialSpatialAccelerations;
  std::vector<std::vector<std::vector<Eigen::Vector3s>>>
      originalTrialWorldAccelerations;

  // Check all the COM spatial residuals are zero
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    s_t dt = init->trialTimesteps[trial];

    std::vector<std::vector<Eigen::Vector6s>> originalSpatialAccelerations;
    std::vector<std::vector<Eigen::Vector3s>> originalWorldAccelerations;
    for (int t = 1; t < init->poseTrials[trial].cols() - 1; t++)
    {
      Eigen::VectorXs q = init->poseTrials[trial].col(t);
      Eigen::VectorXs dq = skel->getPositionDifferences(
                               init->poseTrials[trial].col(t),
                               init->poseTrials[trial].col(t - 1))
                           / dt;
      Eigen::VectorXs ddq = (skel->getPositionDifferences(
                                 init->poseTrials[trial].col(t + 1),
                                 init->poseTrials[trial].col(t))
                             - skel->getPositionDifferences(
                                 init->poseTrials[trial].col(t),
                                 init->poseTrials[trial].col(t - 1)))
                            / (dt * dt);
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      std::vector<Eigen::Vector6s> bodyAccelerations;
      std::vector<Eigen::Vector3s> bodyWorldAccelerations;
      for (int i = 0; i < skel->getNumBodyNodes(); i++)
      {
        bodyAccelerations.push_back(
            skel->getBodyNode(i)->getCOMSpatialAcceleration());
        bodyWorldAccelerations.push_back(
            skel->getBodyNode(i)->getCOMLinearAcceleration());
      }
      originalSpatialAccelerations.push_back(bodyAccelerations);
      originalWorldAccelerations.push_back(bodyWorldAccelerations);

      Eigen::Vector6s spatialResidual = helper.calculateCOMSpatialResidual(
          q, dq, ddq, init->grfTrials[trial].col(t));
      if (spatialResidual.tail<3>().norm() > 1e-7)
      {
        std::cout << "Linear COM spatial residual non-zero at t=" << t << ":"
                  << std::endl
                  << spatialResidual << std::endl;
        return false;
      }
    }
    originalTrialSpatialAccelerations.push_back(originalSpatialAccelerations);
    originalTrialWorldAccelerations.push_back(originalWorldAccelerations);
  }

  // ZYX = 0,
  // XYZ = 1,
  // ZXY = 2,
  // XZY = 3,
  std::cout << "Axis order: "
            << (int)static_cast<dynamics::EulerFreeJoint*>(skel->getRootJoint())
                   ->getAxisOrder()
            << std::endl;

  // Rotate all the positions, maintaining the COM locations
  Eigen::Vector3s rotation = Eigen::Vector3s::Random() * 0.1;
  Eigen::Matrix3s R = math::expMapRot(rotation);
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    std::vector<Eigen::Vector3s> originalCOMs
        = fitter.comPositions(init, trial);
    for (int t = 0; t < init->poseTrials[trial].cols(); t++)
    {
      Eigen::VectorXs originalQ = init->poseTrials[trial].col(t);
      Eigen::VectorXs q = originalQ;
      // q.head<3>() += rotation;
      q.head<3>() = math::matrixToEulerZXY(R * eulerZXYToMatrix(q.head<3>()));
      skel->setPositions(q);
      Eigen::Vector3s newCOM = skel->getCOM();
      Eigen::Vector3s diff = newCOM - originalCOMs[t];
      q.segment<3>(3) -= diff;
      init->poseTrials[trial].col(t) = q;
    }
    std::vector<Eigen::Vector3s> newCOMs = fitter.comPositions(init, trial);
    for (int t = 0; t < originalCOMs.size(); t++)
    {
      if ((originalCOMs[t] - newCOMs[t]).norm() > 1e-8)
      {
        std::cout << "COM position changed after rotation at t=" << t << ":"
                  << std::endl;
        std::cout << "Original:" << std::endl << originalCOMs[t] << std::endl;
        std::cout << "New:" << std::endl << newCOMs[t] << std::endl;
      }
    }
  }

  // Check all the COM spatial residuals are still zero
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    s_t dt = init->trialTimesteps[trial];
    for (int t = 1; t < init->poseTrials[trial].cols() - 1; t++)
    {
      Eigen::VectorXs q = init->poseTrials[trial].col(t);
      Eigen::VectorXs dq = skel->getPositionDifferences(
                               init->poseTrials[trial].col(t),
                               init->poseTrials[trial].col(t - 1))
                           / dt;
      Eigen::VectorXs ddq = (skel->getPositionDifferences(
                                 init->poseTrials[trial].col(t + 1),
                                 init->poseTrials[trial].col(t))
                             - skel->getPositionDifferences(
                                 init->poseTrials[trial].col(t),
                                 init->poseTrials[trial].col(t - 1)))
                            / (dt * dt);
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      for (int i = 0; i < skel->getNumBodyNodes(); i++)
      {
        Eigen::Vector6s originalSpatial
            = originalTrialSpatialAccelerations[trial][t - 1][i];
        Eigen::Vector6s newSpatial
            = skel->getBodyNode(i)->getCOMSpatialAcceleration();
        newSpatial.tail<3>() = R.transpose() * newSpatial.tail<3>();
        if (!equals(originalSpatial, newSpatial, 1e-8))
        {
          Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(6, 3);
          compare.col(0) = originalSpatial;
          compare.col(1) = newSpatial;
          compare.col(2) = originalSpatial - newSpatial;
          std::cout << "Body-local \"" << skel->getBodyNode(i)->getName()
                    << "\" spatial acceleration on t=" << t
                    << " doesn't match original after rotation!" << std::endl;
          std::cout << "Original - Rotated - Diff" << std::endl
                    << compare << std::endl;
        }

        Eigen::Vector3s originalWorld
            = R * originalTrialWorldAccelerations[trial][t - 1][i];
        Eigen::Vector3s newWorld
            = skel->getBodyNode(i)->getCOMLinearAcceleration();
        if (!equals(originalWorld, newWorld, 1e-8))
        {
          Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 3);
          compare.col(0) = originalWorld;
          compare.col(1) = newWorld;
          compare.col(2) = originalWorld - newWorld;
          std::cout << "Body-local \"" << skel->getBodyNode(i)->getName()
                    << "\" linear acceleration on t=" << t
                    << " doesn't match original after rotation!" << std::endl;
          std::cout << "R * Original - New - Diff" << std::endl
                    << compare << std::endl;
          break;
        }
      }

      Eigen::Vector6s spatialResidual = helper.calculateCOMSpatialResidual(
          q, dq, ddq, init->grfTrials[trial].col(t));
      if (spatialResidual.tail<3>().norm() > 1e-7)
      {
        std::cout
            << "Linear COM spatial residual (after rotation) non-zero at t="
            << t << ":" << std::endl
            << spatialResidual << std::endl;
        return false;
      }
    }
  }

  return true;
}

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, ID_EQNS)
{
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(512);
#endif

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);
  file.skeleton->setPositions(file.skeleton->getRandomPose());
  file.skeleton->setVelocities(file.skeleton->getRandomVelocity());

  std::map<int, Eigen::Vector6s> worldForces;
  worldForces[file.skeleton->getBodyNode("calcn_r")->getIndexInSkeleton()]
      = Eigen::Vector6s::Random() * 1000;
  worldForces[file.skeleton->getBodyNode("calcn_l")->getIndexInSkeleton()]
      = Eigen::Vector6s::Random() * 1000;

  EXPECT_TRUE(testApplyWorldForces(file.skeleton));
  EXPECT_TRUE(testForwardDynamicsFormula(file.skeleton, worldForces));
  EXPECT_TRUE(testInverseDynamicsFormula(file.skeleton, worldForces));
  EXPECT_TRUE(testResidualAgainstID(file.skeleton, worldForces));

  for (int i = 0; i < file.skeleton->getNumJoints(); i++)
  {
    bool success = testBodyScaleJointJacobians(file.skeleton, i);
    if (!success)
    {
      EXPECT_TRUE(success);
      return;
    }
  }

  // If the downstream grad tests fail, uncomment these tests to diagnose root
  // issue:

  // EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::POSITION));
  // EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::GROUP_SCALES));
  // EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::GROUP_MASSES));
  // EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::GROUP_COMS));
  // EXPECT_TRUE(testMassJacobian(file.skeleton,
  // WithRespectTo::GROUP_INERTIAS));

  // EXPECT_TRUE(testCoriolisJacobian(file.skeleton, WithRespectTo::POSITION));
  // EXPECT_TRUE(testCoriolisJacobian(file.skeleton, WithRespectTo::VELOCITY));
  // EXPECT_TRUE(testCoriolisJacobian(file.skeleton,
  // WithRespectTo::GROUP_SCALES));
  // EXPECT_TRUE(testCoriolisJacobian(file.skeleton,
  // WithRespectTo::GROUP_MASSES));
  // EXPECT_TRUE(testCoriolisJacobian(file.skeleton,
  // WithRespectTo::GROUP_COMS)); EXPECT_TRUE(
  //     testCoriolisJacobian(file.skeleton, WithRespectTo::GROUP_INERTIAS));

  EXPECT_TRUE(
      testResidualJacWrt(file.skeleton, worldForces, WithRespectTo::POSITION));
  // EXPECT_TRUE(
  //     testResidualJacWrt(file.skeleton, worldForces,
  //     WithRespectTo::VELOCITY));
  // EXPECT_TRUE(testResidualJacWrt(
  //     file.skeleton, worldForces, WithRespectTo::ACCELERATION));
  // EXPECT_TRUE(testResidualJacWrt(
  //     file.skeleton, worldForces, WithRespectTo::GROUP_SCALES));
  // EXPECT_TRUE(testResidualJacWrt(
  //     file.skeleton, worldForces, WithRespectTo::GROUP_MASSES));
  EXPECT_TRUE(testResidualJacWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_COMS));
  EXPECT_TRUE(testResidualJacWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_INERTIAS));

  EXPECT_TRUE(
      testResidualGradWrt(file.skeleton, worldForces, WithRespectTo::POSITION));
  EXPECT_TRUE(
      testResidualGradWrt(file.skeleton, worldForces, WithRespectTo::VELOCITY));
  EXPECT_TRUE(testResidualGradWrt(
      file.skeleton, worldForces, WithRespectTo::ACCELERATION));
  EXPECT_TRUE(testResidualGradWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_SCALES));
  EXPECT_TRUE(testResidualGradWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_MASSES));
  EXPECT_TRUE(testResidualGradWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_COMS));
  EXPECT_TRUE(testResidualGradWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_INERTIAS));
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, ROOT_JACS)
{
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(512);
#endif

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);

  for (int i = 0; i < 5; i++)
  {
    std::vector<int> collisionBodies;
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_r")->getIndexInSkeleton());
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_l")->getIndexInSkeleton());
    bool success = testResidualTrajectoryTaylorExpansionWithRandomTrajectory(
        file.skeleton, collisionBodies, 10);
    if (!success)
    {
      EXPECT_TRUE(success);
      return;
    }
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, GROUP_INDICES)
{
  std::vector<int> indices;
  indices.push_back(10);
  indices.push_back(1001);
  indices.push_back(1002);
  indices.push_back(1003);
  indices.push_back(1004);
  indices.push_back(1005);
  indices.push_back(1006);
  indices.push_back(1007);
  indices.push_back(1008);

  std::vector<int> mapping = math::getConsolidatedMapping(indices, 2);

  EXPECT_EQ(mapping.size(), indices.size());

  int max = 0;
  for (int m : mapping)
  {
    if (m > max)
    {
      max = m;
    }
  }
  EXPECT_EQ(max, 1);

  EXPECT_NE(mapping[0], mapping[1]);
  EXPECT_EQ(mapping[1], mapping[2]);
  EXPECT_EQ(mapping[2], mapping[3]);
  EXPECT_EQ(mapping[3], mapping[4]);
  EXPECT_EQ(mapping[4], mapping[5]);
  EXPECT_EQ(mapping[5], mapping[6]);
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, LIN_JACS)
{
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(512);
#endif

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);

  for (int i = 0; i < 5; i++)
  {
    std::vector<int> collisionBodies;
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_r")->getIndexInSkeleton());
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_l")->getIndexInSkeleton());
    bool success = testLinearTrajectorLinearMapWithRandomTrajectory(
        file.skeleton, collisionBodies, 10, false);
    if (!success)
    {
      EXPECT_TRUE(success);
      return;
    }
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, LIN_JACS_REACTION_WHEELS)
{
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(512);
#endif

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);

  for (int i = 0; i < 5; i++)
  {
    std::vector<int> collisionBodies;
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_r")->getIndexInSkeleton());
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_l")->getIndexInSkeleton());
    bool success = testLinearTrajectorLinearMapWithRandomTrajectory(
        file.skeleton, collisionBodies, 10, true);
    if (!success)
    {
      EXPECT_TRUE(success);
      return;
    }
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, MULTIMASS_JACS)
{
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(512);
#endif

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);

  for (int i = 0; i < 5; i++)
  {
    std::vector<int> collisionBodies;
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_r")->getIndexInSkeleton());
    collisionBodies.push_back(
        file.skeleton->getBodyNode("calcn_l")->getIndexInSkeleton());
    bool success = testMultiMassLinearMapWithRandomTrajectory(
        file.skeleton, collisionBodies, 8);
    if (!success)
    {
      EXPECT_TRUE(success);
      return;
    }
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, SPATIAL_NEWTON_GRAD)
{
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(512);
#endif

  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);
  for (int i = 0; i < 10; i++)
  {
    file.skeleton->setPositions(file.skeleton->getRandomPose());
    file.skeleton->setVelocities(file.skeleton->getRandomVelocity());
    file.skeleton->setAccelerations(file.skeleton->getRandomVelocity());

    std::map<int, Eigen::Vector6s> worldForces;
    worldForces[file.skeleton->getBodyNode("calcn_r")->getIndexInSkeleton()]
        = Eigen::Vector6s::Random() * 1000;
    worldForces[file.skeleton->getBodyNode("calcn_l")->getIndexInSkeleton()]
        = Eigen::Vector6s::Random() * 1000;

    bool pos = testSpatialNewtonGrad(
        file.skeleton, worldForces, neural::WithRespectTo::POSITION);
    EXPECT_TRUE(pos);
    if (!pos)
      return;
    bool vel = testSpatialNewtonGrad(
        file.skeleton, worldForces, neural::WithRespectTo::VELOCITY);
    EXPECT_TRUE(vel);
    if (!vel)
      return;
    bool acc = testSpatialNewtonGrad(
        file.skeleton, worldForces, neural::WithRespectTo::ACCELERATION);
    EXPECT_TRUE(acc);
    if (!acc)
      return;
    bool mass = testSpatialNewtonGrad(
        file.skeleton, worldForces, neural::WithRespectTo::GROUP_MASSES);
    EXPECT_TRUE(mass);
    if (!mass)
      return;
    bool com = testSpatialNewtonGrad(
        file.skeleton, worldForces, neural::WithRespectTo::GROUP_COMS);
    EXPECT_TRUE(com);
    if (!com)
      return;
    bool scales = testSpatialNewtonGrad(
        file.skeleton, worldForces, neural::WithRespectTo::GROUP_SCALES);
    EXPECT_TRUE(scales);
    if (!scales)
      return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, IMPLIED_DENSITY_TEST)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/optimized_scale_and_markers.osim");
  srand(42);
  file.skeleton->setPositions(file.skeleton->getRandomPose());
  file.skeleton->setVelocities(file.skeleton->getRandomVelocity());

  std::cout << "For reference, water is 997 kg/m^3 (this is not an accident, "
               "due to definitions)"
            << std::endl;
  std::cout << "Average body density is 985 kg/m^3 (slightly less than water)"
            << std::endl;
  for (int i = 0; i < file.skeleton->getNumBodyNodes(); i++)
  {
    auto* bodyNode = file.skeleton->getBodyNode(i);
    std::cout << bodyNode->getName() << ": "
              << bodyNode->getInertia().getImpliedCubeDensity() << " kg/m^3"
              << std::endl;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_RESIDUALS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->autodetectScaleGroupAxisFlips(2);

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  /// The absolute value of the loss function can be very large, which leads to
  /// numerical precision issues when finite differencing over it.
  config.setResidualWeight(1e-4);
  config.setResidualUseL1(false);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  problem.unflatten(x);
  Eigen::VectorXs recovered = problem.flatten();

  if (!equals(x, recovered, 1e-9))
  {
    std::cout << "Flatten/unflatten not equal!" << std::endl;
    EXPECT_TRUE(equals(x, recovered, 1e-9));
    return;
  }

  for (auto& block : problem.mBlocks)
  {
    for (int t = 0; t < block.len; t++)
    {
      int realT = block.start + t;
      std::cout << "Testing timestamp " << realT << " / "
                << init->poseTrials[block.trial].cols() << std::endl;

      std::map<int, Eigen::Vector6s> forces;
      for (int j = 0; j < init->grfBodyIndices.size(); j++)
      {
        forces[init->grfBodyIndices[j]]
            = init->grfTrials[block.trial].col(realT).segment<6>(j * 6);
      }
      standard.skeleton->setPositions(block.pos.col(t));
      standard.skeleton->setVelocities(block.vel.col(t));
      standard.skeleton->setAccelerations(block.acc.col(t));
      bool pos = testResidualGradWrt(
          standard.skeleton, forces, WithRespectTo::POSITION, block.grf.col(t));
      if (!pos)
      {
        EXPECT_TRUE(pos);
        return;
      }
      bool vel = testResidualGradWrt(
          standard.skeleton, forces, WithRespectTo::VELOCITY, block.grf.col(t));
      if (!vel)
      {
        EXPECT_TRUE(vel);
        return;
      }
      bool acc = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::ACCELERATION,
          block.grf.col(t));
      if (!acc)
      {
        EXPECT_TRUE(acc);
        return;
      }
      bool scales = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_SCALES,
          block.grf.col(t));
      if (!scales)
      {
        EXPECT_TRUE(scales);
        return;
      }
      bool mass = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_MASSES,
          block.grf.col(t));
      if (!mass)
      {
        EXPECT_TRUE(mass);
        return;
      }
      bool com = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_COMS,
          block.grf.col(t));
      if (!com)
      {
        EXPECT_TRUE(com);
        return;
      }
      bool inertia = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_INERTIAS,
          block.grf.col(t));
      if (!inertia)
      {
        EXPECT_TRUE(inertia);
        return;
      }
    }
  }

  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  s_t tol = 1e-8;
  if (!equals(analytical, fd, tol))
  {
    std::cout
        << "Gradient of DynamicsFitProblem (only residual RMSE) not equal!"
        << std::endl;
    problem.debugErrors(fd, analytical, tol);
    EXPECT_TRUE(equals(analytical, fd, tol));

    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_RESIDUALS_SPRINTER)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Sprinter/IK/JA1Gait35_ik.mot");
  c3dFiles.push_back("dart://sample/grf/Sprinter/C3D/JA1Gait35.c3d");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->autodetectScaleGroupAxisFlips(2);

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20,
      87);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  /// The absolute value of the loss function can be very large, which leads to
  /// numerical precision issues when finite differencing over it.
  config.setResidualWeight(1e-6);
  config.setResidualUseL1(false);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  problem.unflatten(x);
  Eigen::VectorXs recovered = problem.flatten();

  if (!equals(x, recovered, 1e-9))
  {
    std::cout << "Flatten/unflatten not equal!" << std::endl;
    EXPECT_TRUE(equals(x, recovered, 1e-9));
    return;
  }

  for (auto& block : problem.mBlocks)
  {
    for (int t = 0; t < block.len; t++)
    {
      int realT = block.start + t;
      std::cout << "Testing timestamp " << realT << " / "
                << init->poseTrials[block.trial].cols() << std::endl;

      std::map<int, Eigen::Vector6s> forces;
      for (int j = 0; j < init->grfBodyIndices.size(); j++)
      {
        forces[init->grfBodyIndices[j]]
            = init->grfTrials[block.trial].col(realT).segment<6>(j * 6);
      }
      standard.skeleton->setPositions(block.pos.col(t));
      standard.skeleton->setVelocities(block.vel.col(t));
      standard.skeleton->setAccelerations(block.acc.col(t));
      bool pos = testResidualGradWrt(
          standard.skeleton, forces, WithRespectTo::POSITION, block.grf.col(t));
      if (!pos)
      {
        EXPECT_TRUE(pos);
        return;
      }
      bool vel = testResidualGradWrt(
          standard.skeleton, forces, WithRespectTo::VELOCITY, block.grf.col(t));
      if (!vel)
      {
        EXPECT_TRUE(vel);
        return;
      }
      bool acc = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::ACCELERATION,
          block.grf.col(t));
      if (!acc)
      {
        EXPECT_TRUE(acc);
        return;
      }
      bool scales = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_SCALES,
          block.grf.col(t));
      if (!scales)
      {
        EXPECT_TRUE(scales);
        return;
      }
      bool mass = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_MASSES,
          block.grf.col(t));
      if (!mass)
      {
        EXPECT_TRUE(mass);
        return;
      }
      bool com = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_COMS,
          block.grf.col(t));
      if (!com)
      {
        EXPECT_TRUE(com);
        return;
      }
      bool inertia = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_INERTIAS,
          block.grf.col(t));
      if (!inertia)
      {
        EXPECT_TRUE(inertia);
        return;
      }
    }
  }

  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  // TODO: tighten numerical bounds
  s_t tol = 1e-6;
  if (!equals(analytical, fd, tol))
  {
    std::cout
        << "Gradient of DynamicsFitProblem (only residual RMSE) not equal!"
        << std::endl;
    problem.debugErrors(fd, analytical, tol);
    EXPECT_TRUE(equals(analytical, fd, tol));

    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_LINEAR_NEWTON)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setLinearNewtonWeight(1.0);
  config.setLinearNewtonUseL1(true);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  bool result = problem.debugErrors(fd, analytical, 3e-8);
  if (result)
  {
    std::cout << "Gradient of DynamicsFitProblem linear Newton not equal!"
              << std::endl;
    EXPECT_FALSE(result);
    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_SPATIAL_ACC_REG)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setRegularizeSpatialAcc(1);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  bool result = problem.debugErrors(fd, analytical, 1e-7);
  if (result)
  {
    std::cout << "Gradient of DynamicsFitProblem linear Newton not equal!"
              << std::endl;
    EXPECT_FALSE(result);
    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_JOINT_ACC_REG)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setRegularizeJointAcc(1);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  bool result = problem.debugErrors(fd, analytical, 1e-6);
  if (result)
  {
    std::cout << "Gradient of DynamicsFitProblem linear Newton not equal!"
              << std::endl;
    EXPECT_FALSE(result);
    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_MARKERS_L2)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setMarkerWeight(1);
  config.setMarkerUseL1(false);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  bool result = problem.debugErrors(fd, analytical, 3e-8);
  if (result)
  {
    std::cout
        << "Gradient of DynamicsFitProblem (only L2 marker RMSE) not equal!"
        << std::endl;
    EXPECT_FALSE(result);
    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_DENSITY)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  bool anyBadJoints = false;
  for (int i = 0; i < standard.skeleton->getNumBodyNodes(); i++)
  {
    Eigen::Vector6s analytical
        = standard.skeleton->getBodyNode(i)
              ->getInertia()
              .getImpliedCubeDensityGradientWrtMomentVector();
    Eigen::Vector6s fd
        = standard.skeleton->getBodyNode(i)
              ->getInertia()
              .clone()
              .finiteDifferenceImpliedCubeDensityGradientWrtMomentVector();
    if ((analytical - fd).squaredNorm() > 1e-7)
    {
      std::cout << "Inertia grads equal on body \""
                << standard.skeleton->getBodyNode(i)->getName() << "\" (" << i
                << " at index [" << i * 6 << "-" << (i + 1) * 6
                << "]): " << std::endl;
      std::cout << "Analytical: " << std::endl << analytical << std::endl;
      std::cout << "FD: " << std::endl << fd << std::endl;
      std::cout << "Diff: " << std::endl << analytical - fd << std::endl;
      anyBadJoints = true;
    }
  }
  (void)anyBadJoints;
  /*
  if (anyBadJoints)
  {
    EXPECT_FALSE(anyBadJoints);
    return;
  }
  */

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setRegularizeImpliedDensity(1e-5);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  bool result = problem.debugErrors(fd, analytical, 3e-8);
  if (result)
  {
    std::cout
        << "Gradient of DynamicsFitProblem (only density gradients) not equal!"
        << std::endl;
    EXPECT_FALSE(result);
    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, L1_GRAD_NUMERICAL_STABILITY)
{
  Eigen::Vector3s target = Eigen::Vector3s::Random();
  Eigen::Vector3s x = target + 0.001 * Eigen::Vector3s::Random();

  // s_t value = (x - target).norm();
  Eigen::Vector3s analytical = (x - target).normalized();

  bool useRidders = false;

  Eigen::Vector3s result;
  math::finiteDifference<Eigen::Vector3s>(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::Vector3s perturbedX = x;
        perturbedX(dof) += eps;
        perturbed = (perturbedX - target).norm();
        return true;
      },
      result,
      useRidders ? 1e-3 : 1e-8,
      useRidders);

  if (!equals(result, analytical, 1e-8))
  {
    std::cout << "L1 grad Not equal!" << std::endl;
    Eigen::Matrix3s compare = Eigen::Matrix3s::Zero();
    compare.col(0) = analytical;
    compare.col(1) = result;
    compare.col(2) = analytical - result;
    std::cout << "Analytical - FD - Diff" << std::endl << compare << std::endl;
    EXPECT_TRUE(equals(result, analytical, 1e-8));
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_MARKERS_L1)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      25);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setMarkerUseL1(true);
  config.setMarkerWeight(100.0);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x, false);

  bool result = problem.debugErrors(fd, analytical, 2e-6);
  if (result)
  {
    std::cout
        << "Gradient of DynamicsFitProblem (only marker L1 RMSE) not equal!"
        << std::endl;
    EXPECT_FALSE(result);
    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_JOINTS_AND_AXIS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  /*
  init->joints = standard.skeleton->getJoints();
  init->jointWeights = Eigen::VectorXs::Random(init->joints.size());
  init->axisWeights = Eigen::VectorXs::Random(init->joints.size());
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    init->jointCenters.push_back(Eigen::MatrixXs::Random(
        init->joints.size() * 3, init->poseTrials[trial].cols()));
    init->jointAxis.push_back(Eigen::MatrixXs::Random(
        init->joints.size() * 6, init->poseTrials[trial].cols()));
  }
  */

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setJointWeight(1);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);
  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  if (!equals(analytical, fd, 1e-6))
  {
    std::cout << "Gradient of DynamicsFitProblem (only marker RMSE) not equal!"
              << std::endl;
    problem.debugErrors(fd, analytical, 1e-6);
    EXPECT_TRUE(equals(analytical, fd, 1e-6));

    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_MARKERS_L1_MATCHES_AVG_RMS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setMarkerWeight(1);
  config.setMarkerUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  s_t loss = problem.computeLoss(problem.flatten());

  DynamicsFitter fitter(
      standard.skeleton, init->grfBodyNodes, standard.trackingMarkers);
  s_t avgRMS = fitter.computeAverageMarkerRMSE(init);

  EXPECT_DOUBLE_EQ(loss, avgRMS);
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_RESIDUAL_L1_MATCHES_AVG_RMS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setResidualWeight(1.0);
  config.setResidualTorqueMultiple(1.0);
  config.setResidualUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  s_t loss = problem.computeLoss(problem.flatten());

  DynamicsFitter fitter(
      standard.skeleton, init->grfBodyNodes, standard.trackingMarkers);
  auto pairForces = fitter.computeAverageResidualForce(init);
  s_t sum = pairForces.first + pairForces.second;

  EXPECT_DOUBLE_EQ(loss, sum);
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_LINEAR_NEWTON_L1_MATCHES_AVG_RMS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setLinearNewtonWeight(1.0);
  config.setLinearNewtonUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  s_t loss = problem.computeLoss(problem.flatten());

  DynamicsFitter fitter(
      standard.skeleton, init->grfBodyNodes, standard.trackingMarkers);
  auto pairForces = fitter.computeAverageResidualForce(init);
  s_t sum = pairForces.first;

  std::cout << "Linear newton: " << loss << "N" << std::endl;
  std::cout << "Residual linear: " << sum << "N" << std::endl;

  EXPECT_DOUBLE_EQ(loss, sum);
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_LINEAR_NEWTON_L1_MATCHES_AVG_RMS_SPRINTER)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Sprinter/IK/JA1Gait35_ik.mot");
  c3dFiles.push_back("dart://sample/grf/Sprinter/C3D/JA1Gait35.c3d");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20,
      87);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setLinearNewtonWeight(1.0);
  config.setLinearNewtonUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);

  s_t loss = problem.computeLoss(problem.flatten());

  DynamicsFitter fitter(
      standard.skeleton, init->grfBodyNodes, standard.trackingMarkers);
  auto pairForces = fitter.computeAverageResidualForce(init);
  s_t sum = pairForces.first;

  std::cout << "Linear newton: " << loss << "N" << std::endl;
  std::cout << "Residual linear: " << sum << "N" << std::endl;

  EXPECT_DOUBLE_EQ(loss, sum);
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_REGULARIZATION)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setRegularizeMasses(1.5);
  config.setRegularizeCOMs(2.0);
  config.setRegularizeInertias(3.0);
  config.setRegularizeBodyScales(4.0);
  config.setRegularizePoses(5.0);
  config.setRegularizeImpliedDensity(0);
  config.setRegularizeTrackingMarkerOffsets(6.0);
  config.setRegularizeAnatomicalMarkerOffsets(7.0);

  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);
  srand(42);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  // Offset from zero to have some gradients
  x += Eigen::VectorXs::Random(x.size()) * 0.01;

  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);

  if (!equals(analytical, fd, 1e-7))
  {
    std::cout
        << "Gradient of DynamicsFitProblem (only regularization) not equal!"
        << std::endl;
    problem.debugErrors(fd, analytical, 1e-8);
    EXPECT_TRUE(equals(analytical, fd, 1e-7));
    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, FIT_PROBLEM_JAC)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      12);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setIncludeBodyScales(true);
  config.setIncludeCOMs(true);
  config.setIncludeInertias(true);
  config.setIncludeMarkerOffsets(true);
  config.setIncludeMasses(true);
  config.setIncludePoses(true);
  config.setConstrainResidualsZero(true);
  config.setMaxBlockSize(5);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, config);
  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;
  std::cout << "Constraint dim: " << problem.getConstraintSize() << std::endl;

  Eigen::MatrixXs analytical = problem.computeConstraintsJacobian();
  Eigen::MatrixXs fd = problem.finiteDifferenceConstraintsJacobian();

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Jacobian of constraints of DynamicsFitProblem not equal!"
              << std::endl;
    for (int i = 0; i < fd.rows(); i++)
    {
      std::cout << "Error on row " << i << std::endl;
      problem.debugErrors(fd.row(i), analytical.row(i), 1e-8);
      return;
    }
    EXPECT_TRUE(equals(analytical, fd, 1e-8));

    return;
  }
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, TEST_ZERO_RESIDUALS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/SprinterWithSpine/IK/JA1Gait35_ik.mot");
  trcFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/MarkerData/JA1Gait35.trc");
  grfFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/ID/JA1Gait35_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/SprinterWithSpine/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->setGravity(Eigen::Vector3s(0, -9.81, 0));

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitter fitter(
      standard.skeleton, init->grfBodyNodes, init->trackingMarkers);

  fitter.smoothAccelerations(init);
  Eigen::MatrixXs originalTrajectory = init->poseTrials[0];
  // this gets the mass of the skeleton
  fitter.zeroLinearResidualsOnCOMTrajectory(init);
  for (int i = 0; i < 20; i++)
  {
    // this holds the mass constant, and re-jigs the trajectory to try to get
    // the angular ACC's to match more closely what was actually observed
    fitter.zeroLinearResidualsAndOptimizeAngular(
        init, 0, originalTrajectory, 1.0, 5.0);
  }
  // fitter.zeroLinearResidualsAndOptimizeAngular(
  //     init, 0, originalTrajectory, 1.0, 0.0);

  // Recompute the mass of the system
  fitter.zeroLinearResidualsOnCOMTrajectory(init);

  // fitter.moveComsToMinimizeAngularResiduals(init);

  fitter.saveDynamicsToGUI(
      "../../../javascript/src/data/movement2.bin",
      init,
      0,
      (int)round(1.0 / init->trialTimesteps[0]));
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, TEST_ZERO_RESIDUALS_ABLATION)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/SprinterWithSpine/IK/JA1Gait35_ik.mot");
  trcFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/MarkerData/JA1Gait35.trc");
  grfFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/ID/JA1Gait35_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/SprinterWithSpine/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->setGravity(Eigen::Vector3s(0, -9.81, 0));

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitter fitter(
      standard.skeleton, init->grfBodyNodes, init->trackingMarkers);

  Eigen::MatrixXs originalTrajectory = init->poseTrials[0];
  // this gets the mass of the skeleton
  fitter.zeroLinearResidualsOnCOMTrajectoryAblation(init, 4);

  // fitter.moveComsToMinimizeAngularResiduals(init);

  fitter.saveDynamicsToGUI(
      "../../../javascript/src/data/movement2.bin",
      init,
      0,
      (int)round(1.0 / init->trialTimesteps[0]));
}
#endif

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, TEST_MULTIMASS_ZERO_RESIDUALS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/SprinterWithSpine/IK/JA1Gait35_ik.mot");
  trcFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/MarkerData/JA1Gait35.trc");
  grfFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/ID/JA1Gait35_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/SprinterWithSpine/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->setGravity(Eigen::Vector3s(0, -9.81, 0));

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitter fitter(
      standard.skeleton, init->grfBodyNodes, init->trackingMarkers);

  fitter.smoothAccelerations(init);
  // Eigen::MatrixXs originalTrajectory = init->poseTrials[0];
  fitter.multimassZeroLinearResidualsOnCOMTrajectory(init);

  std::cout << "Linear residual: "
            << fitter.computeAverageTrialResidualForce(init, 0).first
            << std::endl;

  fitter.saveDynamicsToGUI(
      "../../../javascript/src/data/movement2.bin",
      init,
      0,
      (int)round(1.0 / init->trialTimesteps[0]));
}
#endif

/*
#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, PROCESS_GRFS_CHECK)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Moore/unscaled_generic.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->autodetectScaleGroupAxisFlips(2);
  std::shared_ptr<dynamics::Skeleton> skel = standard.skeleton;

  std::vector<ForcePlate> forcePlates
      = OpenSimParser::loadGRF("dart://sample/grf/Moore/grf.mot");
  Eigen::MatrixXs poses
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), forcePlates[0].forces.size());

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::vector<int> footIndices;
  std::vector<dynamics::BodyNode*> footBodies;
  for (std::string footName : footNames)
  {
    dynamics::BodyNode* footBody = skel->getBodyNode(footName);
    footBodies.push_back(footBody);
    footIndices.push_back(footBody->getIndexInSkeleton());
  }
  std::vector<std::vector<int>> forcePlatesAssignedToContactBody;
  for (int i = 0; i < forcePlates.size(); i++)
  {
    forcePlates[i].autodetectNoiseThresholdAndClip();
    forcePlates[i].detectAndFixCopMomentConvention();
    forcePlatesAssignedToContactBody.emplace_back();
    for (int t = 0; t < poses.cols(); t++)
    {
      forcePlatesAssignedToContactBody
          [forcePlatesAssignedToContactBody.size() - 1]
              .push_back(0);
    }
  }
  Eigen::MatrixXs grfTrial
      = Eigen::MatrixXs::Zero(footBodies.size() * 6, poses.cols());

  DynamicsFitter::recomputeGRFs(
      forcePlates,
      poses,
      footBodies,
      std::vector<int>(),
      forcePlatesAssignedToContactBody,
      grfTrial,
      skel);

  s_t groundHeight = 0.0;
  for (int t = 0; t < poses.cols(); t++)
  {
    for (int i = 0; i < footIndices.size(); i++)
    {
      Eigen::Vector6s worldWrench = grfTrial.block<6, 1>(i * 6, t);
      Eigen::Vector9s copWrench
          = math::projectWrenchToCoP(worldWrench, groundHeight, 1);
      Eigen::Vector3s cop = copWrench.head<3>();

      s_t closestDistance = 1e9;
      int closestPlate = 0;
      for (int p = 0; p < forcePlates.size(); p++)
      {
        s_t dist = (forcePlates[p].centersOfPressure[t] - cop).norm();
        if (dist < closestDistance)
        {
          closestDistance = dist;
          closestPlate = p;
        }
      }
      if (closestDistance > 0.10)
      {
        std::cout << "Warning: cop " << cop.transpose() << " is "
                  << closestDistance << "m away from cop on original plate "
                  << closestPlate << std::endl;
        EXPECT_TRUE(closestDistance <= 0.10);
        return;
      }
    }
  }
}
#endif
*/

#ifdef ALL_TESTS
TEST(DynamicsFitter, WRENCH_ASSIGNMENT)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles);
  EXPECT_TRUE(testWorldWrenchAssignment(init));
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, RESIDUALS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      6);

  EXPECT_TRUE(
      testRelationshipBetweenResidualAndLinear(standard.skeleton, init));
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, RESIDUALS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      6);

  EXPECT_TRUE(
      testRelationshipBetweenResidualAndLinear(standard.skeleton, init));
}
#endif

#ifdef BLOCKING_GUI_TESTS
TEST(DynamicsFitter, GROUP_SYMMETRY)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  standard.skeleton->autodetectScaleGroupAxisFlips(2);
  int index = standard.skeleton->getScaleGroupIndex(
      standard.skeleton->getBodyNode("hand_l"));

  Eigen::VectorXs groupCOMs = standard.skeleton->getGroupCOMs();
  groupCOMs.segment<3>(index * 3) += Eigen::Vector3s::UnitZ() * 0.5;
  standard.skeleton->setGroupCOMs(groupCOMs);

  Eigen::VectorXs groupDimsAndEuler = standard.skeleton->getGroupInertias();
  Eigen::Vector6s dimsAndEulers = groupDimsAndEuler.segment<6>(index * 6);
  dimsAndEulers(3) += M_PI / 4; // Rotate X by 45 degrees
  groupDimsAndEuler.segment<6>(index * 6) = dimsAndEulers;
  standard.skeleton->setGroupInertias(groupDimsAndEuler);

  const BodyScaleGroup& group = standard.skeleton->getBodyScaleGroup(index);
  (void)group;

  GUIWebsocketServer server;
  server.renderBasis();
  server.renderSkeleton(standard.skeleton);
  server.renderSkeletonInertiaCubes(standard.skeleton);
  server.serve(8070);
  server.blockWhileServing();
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, RECOVER_X)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      standard.skeleton,
      standard.markersMap,
      standard.trackingMarkers,
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      20);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblem problem(
      init,
      standard.skeleton,
      init->updatedMarkerMap,
      standard.trackingMarkers,
      footNodes);
  problem.setVelAccImplicit(true);
  problem.setIncludePoses(false);

  Eigen::VectorXs x = problem.flatten();
  x += Eigen::VectorXs::Random(x.size()) * 0.01;
  problem.unflatten(x);
  s_t loss = problem.computeLoss(x);
  problem.intermediate_callback(
      Ipopt::AlgorithmMode::RegularMode,
      0,
      loss,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      nullptr,
      nullptr);
  problem.finalize_solution(
      Ipopt::SolverReturn::SUCCESS,
      0,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr);

  DynamicsFitProblem problem2(
      init,
      standard.skeleton,
      init->updatedMarkerMap,
      standard.trackingMarkers,
      footNodes);
  problem2.setVelAccImplicit(true);
  problem2.setIncludePoses(false);
  Eigen::VectorXs x2 = problem2.flatten();

  if (!equals(x, x2, 1e-10))
  {
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(x.size(), 3);
    compare.col(0) = x;
    compare.col(1) = x2;
    compare.col(2) = x - x2;
    std::cout << "x - recovered - diff" << std::endl << compare << std::endl;
    EXPECT_TRUE(equals(x, x2, 1e-10));

    problem.debugErrors(x2, x, 1e-10);
    return;
  }

  s_t loss2 = problem2.computeLoss(x2);
  if (abs(loss - loss2) > 1e-10)
  {
    std::cout << "Expected recovered loss to be equal (same settings): " << loss
              << " - " << loss2 << " = " << (loss - loss2) << std::endl;
    EXPECT_EQ(loss, loss2);
  }

  problem2.setIncludePoses(true);
  s_t loss3 = problem2.computeLoss(problem2.flatten());
  if (abs(loss - loss3) > 1e-10)
  {
    std::cout << "Expected recovered loss to be equal (including poses): "
              << loss << " - " << loss3 << " = " << (loss - loss3) << std::endl;
    EXPECT_EQ(loss, loss3);
  }
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, END_TO_END_SUBJECT4)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true,
      true);
}
#endif

/*
TEST(DynamicsFitter, DUMMY_1)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dynamics::EulerFreeJoint>();
  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  T.translation() = Eigen::Vector3s::UnitZ();
  pair.first->setTransformFromChildBodyNode(T);
  skel->setControlForce(0, 1.0);
  s_t dt = 1e-6;
  skel->setTimeStep(dt);
  skel->setGravity(Eigen::Vector3s::Zero());
  skel->computeForwardDynamics();
  std::cout << "Accelerations: " << std::endl
            << skel->getAccelerations() << std::endl;
  std::cout << "Inv mass matrix (tau -> ddq): " << std::endl
            << skel->getInvMassMatrix() << std::endl;
  std::cout << "Mass matrix (ddq -> tau): " << std::endl
            << skel->getMassMatrix() << std::endl;
  std::cout << "COM pos: " << std::endl << skel->getCOM() << std::endl;
  std::cout << "COM lin Acc: " << std::endl
            << skel->getCOMLinearAcceleration() << std::endl;
  std::cout << "COM spatial Acc: " << std::endl
            << skel->getCOMSpatialAcceleration() << std::endl;
  std::cout << "Integrating velocities" << std::endl;
  skel->integrateVelocities(dt);
  std::cout << "COM lin Acc: " << std::endl
            << skel->getCOMLinearAcceleration() << std::endl;
  std::cout << "COM spatial Acc: " << std::endl
            << skel->getCOMSpatialAcceleration() << std::endl;
  std::cout << "Integrating positions" << std::endl;
  skel->integratePositions(dt);
  std::cout << "COM pos: " << std::endl << skel->getCOM() << std::endl;
}
*/

/*
TEST(DynamicsFitter, DUMMY_2)
{
  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto pair = skel->createJointAndBodyNodePair<dynamics::EulerFreeJoint>();
  (void)pair;
  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  T.translation() = Eigen::Vector3s::UnitZ();
  pair.first->setTransformFromChildBodyNode(T);
  // skel->setControlForce(0, 1.0);
  s_t dt = 1e-6;
  skel->setTimeStep(dt);
  skel->setGravity(-9.81 * Eigen::Vector3s::UnitZ());
  skel->computeForwardDynamics();
  std::cout << "Accelerations with no vel: " << std::endl
            << skel->getAccelerations() << std::endl;
  std::cout << "COM spatial acc no vel: " << std::endl
            << skel->getCOMSpatialAcceleration() << std::endl;
  std::cout << "COM linear acc no vel: " << std::endl
            << skel->getCOMLinearAcceleration() << std::endl;

  skel->setVelocities(Eigen::Vector6s(0, 1, 0, 1, 0, 0));
  skel->computeForwardDynamics();
  std::cout << "Accelerations with vel: " << std::endl
            << skel->getAccelerations() << std::endl;
  std::cout << "COM spatial acc with vel: " << std::endl
            << skel->getCOMSpatialAcceleration() << std::endl;
  std::cout << "COM linear acc with vel: " << std::endl
            << skel->getCOMLinearAcceleration() << std::endl;
}
*/

// #ifdef JACOBIAN_TESTS
// This test doesn't pass, because rotating about the COM while preserving the
// COM trajectory doesn't actually work, due to differences between FD and
// analytical accelerations.
/*
TEST(DynamicsFitter, RESIDUAL_UNDER_ROTATION_SPRINTERS)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Sprinter2/IK/JA1Gait35_ik.mot");
  trcFiles.push_back("dart://sample/grf/Sprinter2/MarkerData/JA1Gait35.trc");
  grfFiles.push_back("dart://sample/grf/Sprinter2/ID/JA1Gait35_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  EXPECT_TRUE(verifyResidualElimination(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      200,
      87));
}
*/
// #endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, END_TO_END_SPRINTER)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Sprinter2/IK/JA1Gait35_ik.mot");
  trcFiles.push_back("dart://sample/grf/Sprinter2/MarkerData/JA1Gait35.trc");
  grfFiles.push_back("dart://sample/grf/Sprinter2/ID/JA1Gait35_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, END_TO_END_SPRINTER_WITH_SPINE)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/SprinterWithSpine/IK/JA1Gait35_ik.mot");
  trcFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/MarkerData/JA1Gait35.trc");
  grfFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/ID/JA1Gait35_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/SprinterWithSpine/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MARKERS_TO_DYNAMICS_SPRINTER_WITH_SPINE)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  trcFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/MarkerData/JA1Gait35.trc");
  grfFiles.push_back(
      "dart://sample/grf/SprinterWithSpine/ID/JA1Gait35_grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEndToEnd(
      "dart://sample/grf/SprinterWithSpine/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      c3dFiles,
      trcFiles,
      grfFiles,
      20,
      87,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, HAMNER_SUBJECT04)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  trcFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run200/markers.trc");
  trcFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run300/markers.trc");
  trcFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run400/markers.trc");
  trcFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run500/markers.trc");
  grfFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run200/grf.mot");
  grfFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run300/grf.mot");
  grfFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run400/grf.mot");
  grfFiles.push_back(
      "dart://sample/osim/HamnerSubject04/subject04/trials/run500/grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEndToEnd(
      "dart://sample/osim/HamnerSubject04/subject04/unscaled_generic.osim",
      footNames,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MARKERS_TO_DYNAMICS_SAM_DATA)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  trcFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run200/markers.trc");
  trcFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run300/markers.trc");
  trcFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run400/markers.trc");
  trcFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run500/markers.trc");
  grfFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run200/grf.mot");
  grfFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run300/grf.mot");
  grfFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run400/grf.mot");
  grfFiles.push_back(
      "dart://sample/grf/Hamner_subject17/trials/run500/grf.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEndToEnd(
      "dart://sample/grf/Hamner_subject17/"
      "unscaled_generic.osim",
      footNames,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MICHAEL_TEST_SCALING)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("S02DN101");
  trialNames.push_back("S02DN102");
  // trialNames.push_back("S02DN103");
  // trialNames.push_back("S02DN104");
  // trialNames.push_back("S02DN105");
  // trialNames.push_back("S02DN106");
  // trialNames.push_back("S02DN107");
  // trialNames.push_back("S02DN108");
  // trialNames.push_back("S02DN109");
  // trialNames.push_back("S02DN110");
  // trialNames.push_back("S02DN111");
  // trialNames.push_back("S02DN112");
  // trialNames.push_back("S02DN113");
  // trialNames.push_back("S02DN114");
  // trialNames.push_back("S02DN115");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    motFiles.push_back(
        "dart://sample/osim/MichaelTest4/IK/" + name + "_ik.mot");
    trcFiles.push_back(
        "dart://sample/osim/MichaelTest4/MarkerData/" + name + ".trc");
    grfFiles.push_back(
        "dart://sample/osim/MichaelTest4/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/osim/MichaelTest4/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, OPENCAP_SCALING)
{
  std::string subjectName = "Subject4";
  std::vector<std::string> trialNames;
  trialNames.push_back("DJ1");
  trialNames.push_back("walking1");
  trialNames.push_back("DJ2");
  trialNames.push_back("walking2");
  // trialNames.push_back("walking4");
  // trialNames.push_back("DJ3");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    motFiles.push_back(
        "dart://sample/osim/OpenCapTest/" + subjectName + "/IK/" + name
        + "_ik.mot");
    trcFiles.push_back(
        "dart://sample/osim/OpenCapTest/" + subjectName + "/MarkerData/" + name
        + ".trc");
    grfFiles.push_back(
        "dart://sample/osim/OpenCapTest/" + subjectName + "/ID/" + name
        + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/osim/OpenCapTest/" + subjectName + "/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, KirstenTest)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  std::string prefix = "dart://sample/osim/KirstenTest/";
  trcFiles.push_back(prefix + "DLS01.trc");
  grfFiles.push_back(prefix + "DLS01_grf.mot");
  motFiles.push_back(prefix + "DLS01_ik.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/osim/KirstenTest/final.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, HamnerMultipleTrials)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  std::string prefix = "dart://sample/osim/HamnerMultipleTrials/";
  trcFiles.push_back(prefix + "run200.trc");
  trcFiles.push_back(prefix + "run300.trc");
  grfFiles.push_back(prefix + "run200_grf.mot");
  grfFiles.push_back(prefix + "run300_grf.mot");
  motFiles.push_back(prefix + "run200_ik.mot");
  motFiles.push_back(prefix + "run300_ik.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  int maxTrialsToSolveMassOver = 1;
  runEngine(
      "dart://sample/osim/HamnerMultipleTrials/final.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      false,
      false,
      false,
      maxTrialsToSolveMassOver);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, GRFBlipTest)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  std::string prefix = "dart://sample/grf/GRFBlipTest1/";
  trcFiles.push_back(prefix + "MarkerData/markers.trc");
  grfFiles.push_back(prefix + "ID/markers_grf.mot");
  motFiles.push_back(prefix + "IK/markers_ik.mot");

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  int maxTrialsToSolveMassOver = 1;
  runEngine(
      "dart://sample/grf/GRFBlipTest1/Models/final.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      false,
      false,
      false,
      maxTrialsToSolveMassOver);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MARKERS_TO_DYNAMICS_OPENCAP)
{
  std::string subjectName = "Subject4";
  std::vector<std::string> trialNames;
  trialNames.push_back("DJ1");
  trialNames.push_back("walking1");
  trialNames.push_back("DJ2");
  trialNames.push_back("walking2");
  // trialNames.push_back("walking4");
  // trialNames.push_back("DJ3");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    motFiles.push_back(
        "dart://sample/osim/OpenCapTest/" + subjectName + "/IK/" + name
        + "_ik.mot");
    trcFiles.push_back(
        "dart://sample/osim/OpenCapTest/" + subjectName + "/MarkerData/" + name
        + ".trc");
    grfFiles.push_back(
        "dart://sample/osim/OpenCapTest/" + subjectName + "/ID/" + name
        + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEndToEnd(
      "dart://sample/osim/OpenCapTest/" + subjectName + "/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, OPENCAP_SCALING)
{
  std::vector<std::string> trialNames;
  // trialNames.push_back("DJ1");
  // trialNames.push_back("DJ4");
  // trialNames.push_back("DJ5");
  trialNames.push_back("walking2");
  // trialNames.push_back("walking3");
  // trialNames.push_back("walking4");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    motFiles.push_back(
        "dart://sample/grf/OpenCapUnfiltered/IK/" + name + "_ik.mot");
    trcFiles.push_back(
        "dart://sample/grf/OpenCapUnfiltered/MarkerData/" + name + ".trc");
    grfFiles.push_back(
        "dart://sample/grf/OpenCapUnfiltered/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/OpenCapUnfiltered/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      false,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, CARMAGO_TEST)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("treadmill_01_01");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    motFiles.push_back("dart://sample/grf/CarmagoTest/IK/" + name + "_ik.mot");
    trcFiles.push_back(
        "dart://sample/grf/CarmagoTest/MarkerData/" + name + ".trc");
    grfFiles.push_back("dart://sample/grf/CarmagoTest/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/CarmagoTest/Models/"
      "optimized_scale_and_markers.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      false,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, KIRSTEN_TEST)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("DLS01");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    trcFiles.push_back(
        "dart://sample/grf/kirsten_bug/MarkerData/" + name + ".trc");
    motFiles.push_back("dart://sample/grf/kirsten_bug/IK/" + name + "_ik.mot");
    grfFiles.push_back("dart://sample/grf/kirsten_bug/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/kirsten_bug/Models/"
      "final.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      false,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, OPENCAP_SCALE_TEST)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("DJ1");
  trialNames.push_back("DJ2");
  trialNames.push_back("DJ3");
  trialNames.push_back("DJAsym1");
  trialNames.push_back("DJAsym4");
  trialNames.push_back("DJAsym5");
  trialNames.push_back("squats1");
  trialNames.push_back("squatsAsym1");
  trialNames.push_back("static1");
  trialNames.push_back("STS1");
  trialNames.push_back("STSweakLegs1");
  trialNames.push_back("walking1");
  trialNames.push_back("walking2");
  trialNames.push_back("walking3");
  trialNames.push_back("walkingTS1");
  trialNames.push_back("walkingTS2");
  trialNames.push_back("walkingTS4");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    trcFiles.push_back(
        "dart://sample/grf/opencap_large/MarkerData/" + name + ".trc");
    motFiles.push_back(
        "dart://sample/grf/opencap_large/IK/" + name + "_ik.mot");
    grfFiles.push_back(
        "dart://sample/grf/opencap_large/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/opencap_large/Models/"
      "final.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      false,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MARKERS_TO_DYNAMICS_OPENCAP_UNFILTERED)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("DJ5");
  // trialNames.push_back("walking1");
  // trialNames.push_back("DJ2");
  // trialNames.push_back("walking2");
  // trialNames.push_back("walking4");
  // trialNames.push_back("DJ3");

  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    trcFiles.push_back(
        "dart://sample/grf/UnfiliteredOpencap/Marker/" + name + ".trc");
    grfFiles.push_back(
        "dart://sample/grf/UnfiliteredOpencap/Force/" + name
        + "_forces_filt999Hz.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEndToEnd(
      "dart://sample/osim/OpenCapTest/Subject4/Models/"
      "unscaled_generic.osim",
      // "dart://sample/grf/SprinterWithSpine/Models/"
      // "unscaled_generic.osim",
      footNames,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MARKERS_TO_DYNAMICS_MARILYN_BUG)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("markers_smpl");

  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    trcFiles.push_back(
        "dart://sample/osim/11_01_Marilyn_Bug/prod/MarkerData/" + name
        + ".trc");
    grfFiles.push_back(
        "dart://sample/osim/11_01_Marilyn_Bug/prod/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEndToEnd(
      "dart://sample/osim/11_01_Marilyn_Bug/prod/Models/"
      "unscaled_generic.osim",
      // "dart://sample/grf/SprinterWithSpine/Models/"
      // "unscaled_generic.osim",
      footNames,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, STAIRS_TEST)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("StairUp_1_segment_1");

  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    motFiles.push_back(
        "dart://sample/grf/StairsExample/IK/" + name + "_ik.mot");
    trcFiles.push_back(
        "dart://sample/grf/StairsExample/MarkerData/" + name + ".trc");
    grfFiles.push_back(
        "dart://sample/grf/StairsExample/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEngine(
      "dart://sample/grf/StairsExample/Models/"
      "match_markers_but_ignore_physics.osim",
      footNames,
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true,
      true);
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MARKERS_TO_DYNAMICS_STAIRS_TEST)
{
  std::vector<std::string> trialNames;
  trialNames.push_back("StairUp_1_segment_1");

  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  for (std::string& name : trialNames)
  {
    trcFiles.push_back(
        "dart://sample/grf/StairsExample/MarkerData/" + name + ".trc");
    grfFiles.push_back(
        "dart://sample/grf/StairsExample/ID/" + name + "_grf.mot");
  }

  std::vector<std::string> footNames;
  footNames.push_back("calcn_r");
  footNames.push_back("calcn_l");

  runEndToEnd(
      "dart://sample/grf/StairsExample/Models/"
      "match_markers_but_ignore_physics.osim",
      footNames,
      c3dFiles,
      trcFiles,
      grfFiles,
      -1,
      0,
      true);
}
#endif