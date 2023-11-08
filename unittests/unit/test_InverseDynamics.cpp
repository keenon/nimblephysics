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

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

void applyExternalForces(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::Vector6s rootResidual,
    std::map<int, Eigen::Vector6s> worldForces)
{
  skel->clearExternalForces();
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* body = skel->getBodyNode(i);
    if (worldForces.count(i))
    {
      body->setExtWrench(math::dAdInvT(
          body->getWorldTransform().inverse(), worldForces.at(i)));
    }
  }
  skel->getRootBodyNode()->setExtWrench(rootResidual);
}

bool testInverseDynamicsFormula(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::Vector6s rootResidual,
    std::map<int, Eigen::Vector6s> worldForces)
{
  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalTau = skel->getRandomPose();
  skel->setControlForces(originalTau);
  applyExternalForces(skel, rootResidual, worldForces);
  skel->computeForwardDynamics();
  Eigen::VectorXs acc = skel->getAccelerations();

  // Reset
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  skel->clearExternalForces();
  skel->setControlForces(Eigen::VectorXs::Zero(skel->getNumDofs()));

  std::vector<dynamics::BodyNode*> bodyNodes;
  const Eigen::Isometry3s T_wr = skel->getRootBodyNode()->getWorldTransform();
  const Eigen::Isometry3s T_rw = T_wr.inverse();
  std::vector<Eigen::Vector6s> rootFrameWrenches;
  for (auto& pair : worldForces)
  {
    bodyNodes.push_back(skel->getBodyNode(pair.first));
    const Eigen::Vector6s rootBodyWrench = math::dAdInvT(T_rw, pair.second);
    rootFrameWrenches.push_back(rootBodyWrench);

#ifndef NDEBUG
    const Eigen::Vector6s worldWrench = math::dAdInvT(T_wr, rootBodyWrench);
    if (!equals(worldWrench, pair.second, 1e-8))
    {
      std::cout << "Failed to recover world wrench!" << std::endl;
      return false;
    }
#endif
  }

  // ID method
  Eigen::VectorXs taus = skel->getInverseDynamicsFromPredictions(
      acc, bodyNodes, rootFrameWrenches, rootResidual);

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

  return true;
}

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
  Eigen::Vector6s rootResidual = Eigen::Vector6s::Random() * 1000;

  EXPECT_TRUE(
      testInverseDynamicsFormula(file.skeleton, rootResidual, worldForces));
}