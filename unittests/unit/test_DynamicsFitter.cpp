#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/DifferentiableExternalForce.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
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

std::shared_ptr<DynamicsInitialization> runEngine(
    OpenSimFile standard,
    std::vector<std::vector<ForcePlate>> forcePlateTrials,
    std::vector<Eigen::MatrixXs> poseTrials,
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        markerObservationTrials,
    std::vector<int> framesPerSecond,
    bool saveGUI = false)
{
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

  DynamicsFitter fitter(standard.skeleton, standard.markersMap);

  std::shared_ptr<DynamicsInitialization> init = fitter.createInitialization(
      forcePlateTrials, poseTrials, framesPerSecond, markerObservationTrials);
  fitter.scaleLinkMassesFromGravity(init);
  fitter.estimateLinkMassesFromAcceleration(init);

  if (saveGUI)
  {
    std::cout << "Saving trajectory..." << std::endl;
    std::cout << "FPS: " << framesPerSecond[0] << std::endl;
    fitter.saveDynamicsToGUI(
        "../../../javascript/src/data/movement2.bin",
        init,
        0,
        framesPerSecond[0]);
  }

  return init;
}

std::shared_ptr<DynamicsInitialization> runEngine(
    std::string modelPath,
    std::vector<std::string> motFiles,
    std::vector<std::string> c3dFiles,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    bool saveGUI = false)
{
  std::vector<Eigen::MatrixXs> poseTrials;
  std::vector<C3D> c3ds;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<int> framesPerSecond;
  std::vector<std::vector<ForcePlate>> forcePlates;

  OpenSimFile standard = OpenSimParser::parseOsim(modelPath);
  for (int i = 0; i < motFiles.size(); i++)
  {
    OpenSimMot mot = OpenSimParser::loadMot(standard.skeleton, motFiles[i]);
    poseTrials.push_back(mot.poses);
  }

  for (std::string& path : c3dFiles)
  {
    C3D c3d = C3DLoader::loadC3D(path);
    c3ds.push_back(c3d);
    markerObservationTrials.push_back(c3d.markerTimesteps);
    forcePlates.push_back(c3d.forcePlates);
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
      forcePlates.push_back(grf);
    }
    else
    {
      forcePlates.emplace_back();
    }
  }

  return runEngine(
      standard,
      forcePlates,
      poseTrials,
      markerObservationTrials,
      framesPerSecond,
      saveGUI);
}

#ifdef JACOBIAN_TESTS
TEST(DynamicsFitter, ID_EQNS)
{
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

  EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::POSITION));
  EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::GROUP_SCALES));
  EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::GROUP_MASSES));
  EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::GROUP_COMS));
  EXPECT_TRUE(testMassJacobian(file.skeleton, WithRespectTo::GROUP_INERTIAS));

  EXPECT_TRUE(testCoriolisJacobian(file.skeleton, WithRespectTo::POSITION));
  EXPECT_TRUE(testCoriolisJacobian(file.skeleton, WithRespectTo::VELOCITY));
  EXPECT_TRUE(testCoriolisJacobian(file.skeleton, WithRespectTo::GROUP_SCALES));
  EXPECT_TRUE(testCoriolisJacobian(file.skeleton, WithRespectTo::GROUP_MASSES));
  EXPECT_TRUE(testCoriolisJacobian(file.skeleton, WithRespectTo::GROUP_COMS));
  EXPECT_TRUE(
      testCoriolisJacobian(file.skeleton, WithRespectTo::GROUP_INERTIAS));

  EXPECT_TRUE(
      testResidualJacWrt(file.skeleton, worldForces, WithRespectTo::POSITION));
  EXPECT_TRUE(
      testResidualJacWrt(file.skeleton, worldForces, WithRespectTo::VELOCITY));
  EXPECT_TRUE(testResidualJacWrt(
      file.skeleton, worldForces, WithRespectTo::ACCELERATION));
  EXPECT_TRUE(testResidualJacWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_SCALES));
  EXPECT_TRUE(testResidualJacWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_MASSES));
  EXPECT_TRUE(testResidualJacWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_COMS));
  EXPECT_TRUE(testResidualJacWrt(
      file.skeleton, worldForces, WithRespectTo::GROUP_INERTIAS));
}
#endif

#ifdef ALL_TESTS
TEST(DynamicsFitter, MASS_INITIALIZATION)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  runEngine(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim",
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      true);
}
#endif