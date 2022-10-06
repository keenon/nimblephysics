#include <cstdlib>
#include <iostream>
#include <memory>

#include <IpAlgTypes.hpp>
#include <gtest/gtest.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
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
  skel->integrateVelocities(skel->getTimeStep());
  Eigen::VectorXs nextVel = skel->getVelocities();

  // Reset
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  applyExternalForces(skel, worldForces);
  skel->setControlForces(Eigen::VectorXs::Zero(skel->getNumDofs()));

  // ID method
  Eigen::VectorXs taus = skel->getInverseDynamics(nextVel);

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
  Eigen::VectorXs acc = (nextVel - originalVel) / skel->getTimeStep();
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
  config.setResidualWeight(1.0);
  config.setResidualTorqueMultiple(1.0);
  config.setResidualUseL1(true);

  DynamicsFitProblem problem(
      init, skel, init->trackingMarkers, init->grfBodyNodes, config);

  s_t problemLoss = problem.computeLoss(problem.flatten(), true);
  std::cout << "Problem loss: " << problemLoss << std::endl;

  int totalAccTimesteps = 0;
  for (int trial = 0; trial < problem.mPoses.size(); trial++)
  {
    for (int t = 1; t < problem.mPoses[trial].cols() - 1; t++)
    {
      // Add force residual RMS errors to all the middle timesteps
      if (!init->probablyMissingGRF[trial][t])
      {
        totalAccTimesteps++;
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

      Eigen::VectorXs problemQ = problem.mPoses[trial].col(t);
      if (!equals(problemQ, q, 1e-9))
      {
        std::cout << "Poses not equal at time " << t << "!" << std::endl;
        return false;
      }
      Eigen::VectorXs problemDQ = problem.mVels[trial].col(t);
      if (!equals(problemDQ, dq, 1e-9))
      {
        std::cout << "Vels not equal at time " << t << "!" << std::endl;
        Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(dq.size(), 3);
        compare.col(0) = dq;
        compare.col(1) = problemDQ;
        compare.col(2) = dq - problemDQ;
        std::cout << "dq - problem - diff" << std::endl << compare << std::endl;
        return false;
      }
      Eigen::VectorXs problemDDQ = problem.mAccs[trial].col(t);
      if (!equals(problemDDQ, ddq, 1e-9))
      {
        std::cout << "Accs not equal at time " << t << "!" << std::endl;
        return false;
      }

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

  std::cout << "Residual norm: " << residualNorm << std::endl;
  if (abs(residualNorm - problemLoss) > 1e-8)
  {
    EXPECT_EQ(residualNorm, problemLoss);
    return false;
  }

  return true;
}

std::shared_ptr<DynamicsInitialization> runEngine(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::shared_ptr<DynamicsInitialization> init,
    bool saveGUI = false)
{

  DynamicsFitter fitter(skel, init->grfBodyNodes, init->trackingMarkers);
  fitter.smoothAccelerations(init);
  fitter.scaleLinkMassesFromGravity(init);
  // fitter.estimateLinkMassesFromAcceleration(init, 100);

  std::cout << "Initial mass: " << skel->getMass() << " kg" << std::endl;
  std::cout << "What we'd expect average ~GRF to be (Mass * 9.8): "
            << skel->getMass() * 9.8 << " N" << std::endl;
  auto secondPair = fitter.computeAverageRealForce(init);
  std::cout << "Avg Force: " << secondPair.first << " N" << std::endl;
  std::cout << "Avg Torque: " << secondPair.second << " Nm" << std::endl;

  if (!testRelationshipBetweenResidualAndLinear(skel, init))
  {
    std::cout << "The residual norm doesn't map!" << std::endl;
    return init;
  }

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

  // Run as L2
  fitter.setIterationLimit(200);
  fitter.runIPOPTOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setDefaults()
          .setIncludeMasses(true)
          .setIncludeInertias(true)
          .setIncludePoses(true));

  // Re-run as L1
  fitter.runIPOPTOptimization(
      init,
      DynamicsFitProblemConfig(skel)
          .setDefaults(true)
          .setIncludeMasses(true)
          .setIncludeInertias(true)
          .setIncludePoses(true));

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

  /*
  fitter.setIterationLimit(200);
  fitter.runSGDOptimization(
      init, 2e-2, 50, true, true, true, true, false, true);

  fitter.setIterationLimit(50);
  fitter.runSGDOptimization(init, 2e-2, 50, true, true, true, true, true, true);
  */

  // TODO: re-enable
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

  std::cout << "Post optimization mass: " << init->bodyMasses.sum() << " kg"
            << std::endl;
  std::cout << "Post optimization average ~GRF (Mass * 9.8): "
            << init->bodyMasses.sum() * 9.8 << " N" << std::endl;

  skel->setGroupMasses(init->originalGroupMasses);
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

  if (saveGUI)
  {
    std::cout << "Saving trajectory..." << std::endl;
    std::cout << "FPS: " << 1.0 / init->trialTimesteps[0] << std::endl;
    fitter.saveDynamicsToGUI(
        "../../../javascript/src/data/movement2.bin",
        init,
        0,
        (int)round(1.0 / init->trialTimesteps[0]));
  }

  // Attempt writing out the data
  fitter.writeCSVData("../../../data/grf/Subject4/motion.csv", init, 0);
  MJCFExporter exporter;
  exporter.writeSkeleton("../../../data/grf/Subject4/model.mjcf", skel);

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
          = OpenSimParser::loadGRF(grfFiles[i], trc.framesPerSecond);
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
  dynamicsFitter.estimateFootGroundContacts(init);

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
    bool saveGUI = false,
    bool simplify = false)
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

  if (simplify)
  {
    std::map<std::string, std::string> mergeBodiesInto;
    std::shared_ptr<dynamics::Skeleton> simplified
        = standard.skeleton->simplifySkeleton("simplified", mergeBodiesInto);
    std::shared_ptr<DynamicsInitialization> simplifiedInit
        = DynamicsFitter::retargetInitialization(
            standard.skeleton, simplified, init);
    return runEngine(simplified, simplifiedInit, saveGUI);
  }
  else
  {
    return runEngine(standard.skeleton, init, saveGUI);
  }
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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  /// The absolute value of the loss function can be very large, which leads to
  /// numerical precision issues when finite differencing over it.
  config.setResidualWeight(1e-4);
  config.setResidualUseL1(false);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  for (int trial = 0; trial < problem.mAccs.size(); trial++)
  {
    for (int t = 0; t < problem.mAccs[trial].cols(); t++)
    {
      std::cout << "Testing timestamp " << t << " / "
                << problem.mAccs[trial].cols() << std::endl;

      std::map<int, Eigen::Vector6s> forces;
      for (int j = 0; j < problem.mForceBodyIndices.size(); j++)
      {
        forces[problem.mForceBodyIndices[j]]
            = init->grfTrials[trial].col(t).segment<6>(j * 6);
      }
      standard.skeleton->setPositions(problem.mPoses[trial].col(t));
      standard.skeleton->setVelocities(problem.mVels[trial].col(t));
      standard.skeleton->setAccelerations(problem.mAccs[trial].col(t));
      bool pos = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::POSITION,
          init->grfTrials[trial].col(t));
      if (!pos)
      {
        EXPECT_TRUE(pos);
        return;
      }
      bool vel = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::VELOCITY,
          init->grfTrials[trial].col(t));
      if (!vel)
      {
        EXPECT_TRUE(vel);
        return;
      }
      bool acc = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::ACCELERATION,
          init->grfTrials[trial].col(t));
      if (!acc)
      {
        EXPECT_TRUE(acc);
        return;
      }
      bool scales = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_SCALES,
          init->grfTrials[trial].col(t));
      if (!scales)
      {
        EXPECT_TRUE(scales);
        return;
      }
      bool mass = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_MASSES,
          init->grfTrials[trial].col(t));
      if (!mass)
      {
        EXPECT_TRUE(mass);
        return;
      }
      bool com = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_COMS,
          init->grfTrials[trial].col(t));
      if (!com)
      {
        EXPECT_TRUE(com);
        return;
      }
      bool inertia = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_INERTIAS,
          init->grfTrials[trial].col(t));
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

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  for (int trial = 0; trial < problem.mAccs.size(); trial++)
  {
    for (int t = 0; t < problem.mAccs[trial].cols(); t++)
    {
      std::cout << "Testing timestamp " << t << " / "
                << problem.mAccs[trial].cols() << std::endl;

      std::map<int, Eigen::Vector6s> forces;
      for (int j = 0; j < problem.mForceBodyIndices.size(); j++)
      {
        forces[problem.mForceBodyIndices[j]]
            = init->grfTrials[trial].col(t).segment<6>(j * 6);
      }
      standard.skeleton->setPositions(problem.mPoses[trial].col(t));
      standard.skeleton->setVelocities(problem.mVels[trial].col(t));
      standard.skeleton->setAccelerations(problem.mAccs[trial].col(t));
      bool pos = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::POSITION,
          init->grfTrials[trial].col(t));
      if (!pos)
      {
        EXPECT_TRUE(pos);
        return;
      }
      bool vel = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::VELOCITY,
          init->grfTrials[trial].col(t));
      if (!vel)
      {
        EXPECT_TRUE(vel);
        return;
      }
      bool acc = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::ACCELERATION,
          init->grfTrials[trial].col(t));
      if (!acc)
      {
        EXPECT_TRUE(acc);
        return;
      }
      bool scales = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_SCALES,
          init->grfTrials[trial].col(t));
      if (!scales)
      {
        EXPECT_TRUE(scales);
        return;
      }
      bool mass = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_MASSES,
          init->grfTrials[trial].col(t));
      if (!mass)
      {
        EXPECT_TRUE(mass);
        return;
      }
      bool com = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_COMS,
          init->grfTrials[trial].col(t));
      if (!com)
      {
        EXPECT_TRUE(com);
        return;
      }
      bool inertia = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_INERTIAS,
          init->grfTrials[trial].col(t));
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
TEST(DynamicsFitter, FIT_PROBLEM_GRAD_RESIDUALS_IMPLICIT_VEL_POS)
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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  /// The absolute value of the loss function can be very large, which leads to
  /// numerical precision issues when finite differencing over it.
  config.setResidualWeight(1e-4);
  config.setResidualUseL1(false);
  config.setVelAccImplicit(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  problem.mConfig.setVelAccImplicit(false);
  Eigen::VectorXs explicitX = problem.flatten();
  Eigen::VectorXs constraints = problem.computeConstraints(explicitX);
  if (constraints.norm() >= 1e-10)
  {
    std::cout << "Constraints on explicit problem are not zero!" << std::endl;
    EXPECT_TRUE(constraints.norm() < 1e-10);
    return;
  }

  problem.mConfig.setVelAccImplicit(true);
  Eigen::VectorXs recovered2 = problem.flatten();
  if (!equals(x, recovered2, 1e-9))
  {
    std::cout << "Flatten/unflatten after toggling explicit not equal!"
              << std::endl;
    EXPECT_TRUE(equals(x, recovered2, 1e-9));
    return;
  }

  /*
  for (int trial = 0; trial < problem.mAccs.size(); trial++)
  {
    for (int t = 0; t < problem.mAccs[trial].cols(); t++)
    {
      std::cout << "Testing timestamp " << t << " / "
                << problem.mAccs[trial].cols() << std::endl;

      std::map<int, Eigen::Vector6s> forces;
      for (int j = 0; j < problem.mForceBodyIndices.size(); j++)
      {
        forces[problem.mForceBodyIndices[j]]
            = init->grfTrials[trial].col(t).segment<6>(j * 6);
      }
      standard.skeleton->setPositions(problem.mPoses[trial].col(t));
      standard.skeleton->setVelocities(problem.mVels[trial].col(t));
      standard.skeleton->setAccelerations(problem.mAccs[trial].col(t));
      bool pos = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::POSITION,
          init->grfTrials[trial].col(t));
      if (!pos)
      {
        EXPECT_TRUE(pos);
        return;
      }
      bool vel = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::VELOCITY,
          init->grfTrials[trial].col(t));
      if (!vel)
      {
        EXPECT_TRUE(vel);
        return;
      }
      bool acc = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::ACCELERATION,
          init->grfTrials[trial].col(t));
      if (!acc)
      {
        EXPECT_TRUE(acc);
        return;
      }
      bool scales = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_SCALES,
          init->grfTrials[trial].col(t));
      if (!scales)
      {
        EXPECT_TRUE(scales);
        return;
      }
      bool mass = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_MASSES,
          init->grfTrials[trial].col(t));
      if (!mass)
      {
        EXPECT_TRUE(mass);
        return;
      }
      bool com = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_COMS,
          init->grfTrials[trial].col(t));
      if (!com)
      {
        EXPECT_TRUE(com);
        return;
      }
      bool inertia = testResidualGradWrt(
          standard.skeleton,
          forces,
          WithRespectTo::GROUP_INERTIAS,
          init->grfTrials[trial].col(t));
      if (!inertia)
      {
        EXPECT_TRUE(inertia);
        return;
      }
    }
  }
  */

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

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setRegularizeSpatialAcc(1);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setMarkerWeight(1);
  config.setMarkerUseL1(false);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setRegularizeImpliedDensity(1e-5);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setMarkerUseL1(true);
  config.setMarkerWeight(100.0);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::VectorXs x = problem.flatten();
  Eigen::VectorXs analytical = problem.computeGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceGradient(x, false);

  bool result = problem.debugErrors(fd, analytical, 1e-6);
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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

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

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);
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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setMarkerWeight(1);
  config.setMarkerUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setResidualWeight(1.0);
  config.setResidualTorqueMultiple(1.0);
  config.setResidualUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setLinearNewtonWeight(1.0);
  config.setLinearNewtonUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setLinearNewtonWeight(1.0);
  config.setLinearNewtonUseL1(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);

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

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setRegularizeMasses(1.5);
  config.setRegularizeCOMs(2.0);
  config.setRegularizeInertias(3.0);
  config.setRegularizeBodyScales(4.0);
  config.setRegularizePoses(5.0);
  config.setRegularizeImpliedDensity(0);
  config.setRegularizeTrackingMarkerOffsets(6.0);
  config.setRegularizeAnatomicalMarkerOffsets(7.0);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);
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
      20);

  std::vector<dynamics::BodyNode*> footNodes;
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_r"));
  footNodes.push_back(standard.skeleton->getBodyNode("calcn_l"));

  DynamicsFitProblemConfig config(standard.skeleton);
  config.setIncludeInertias(true);
  config.setIncludePoses(true);

  DynamicsFitProblem problem(
      init, standard.skeleton, standard.trackingMarkers, footNodes, config);
  std::cout << "Problem dim: " << problem.getProblemSize() << std::endl;

  Eigen::MatrixXs analytical = problem.computeConstraintsJacobian();
  Eigen::MatrixXs fd = problem.finiteDifferenceConstraintsJacobian();

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Jacobian of constraints of DynamicsFitProblem not equal!"
              << std::endl;
    for (int i = 0; i < fd.rows(); i++)
    {
      problem.debugErrors(fd.row(i), analytical.row(i), 1e-8);
    }
    EXPECT_TRUE(equals(analytical, fd, 1e-8));

    return;
  }
}
#endif

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
      200,
      87,
      true);
}
#endif