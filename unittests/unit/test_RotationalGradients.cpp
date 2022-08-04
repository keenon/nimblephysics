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
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;
using namespace utils;

bool verifyRotationVelocitySkewSymmetric(
    std::shared_ptr<dynamics::Skeleton> skel, int bodyIndex)
{
  dynamics::BodyNode* body = skel->getBodyNode(bodyIndex);
  Eigen::Vector6s vel = body->getCOMSpatialVelocity();
  Eigen::Matrix3s originalR = body->getWorldTransform().linear();

  Eigen::VectorXs originalPos = skel->getPositions();

  s_t eps = 1e-3;
  Eigen::Matrix3s fdR = Eigen::Matrix3s::Zero();
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        skel->setPositions(originalPos);
        skel->integratePositions(eps);
        perturbed = body->getWorldTransform().linear();
        return true;
      },
      fdR,
      eps,
      true);
  skel->setPositions(originalPos);

  // Verify that dR = R*[w]
  Eigen::Matrix3s skew = originalR * math::makeSkewSymmetric(vel.segment<3>(0));
  if (!equals(fdR, skew, 1e-8))
  {
    std::cout << "Problem verifying dR = R*[w]" << std::endl;
    std::cout << "Body " << body->getName() << " (" << bodyIndex << ")"
              << " not equal on skew symmetric velocity FD" << std::endl;
    std::cout << "dR:" << std::endl << fdR << std::endl;
    std::cout << "R*[w]:" << std::endl << skew << std::endl;
    std::cout << "diff:" << std::endl << (fdR - skew) << std::endl;
    return false;
  }

  // Verify that [w] = R^T*dR
  Eigen::Matrix3s rawSkew = math::makeSkewSymmetric(vel.segment<3>(0));
  Eigen::Matrix3s rotatedFdR = originalR.transpose() * fdR;
  if (!equals(rotatedFdR, rawSkew, 1e-8))
  {
    std::cout << "Problem verifying [w] = R^T*dR" << std::endl;
    std::cout << "Body " << body->getName() << " (" << bodyIndex << ")"
              << " not equal on skew symmetric velocity FD" << std::endl;
    std::cout << "R^T*dR:" << std::endl << rotatedFdR << std::endl;
    std::cout << "[w]:" << std::endl << rawSkew << std::endl;
    std::cout << "diff:" << std::endl << (rotatedFdR - rawSkew) << std::endl;
    return false;
  }
  return true;
}

bool verifyRotationAccelerationFormula(
    std::shared_ptr<dynamics::Skeleton> skel, int bodyIndex)
{
  dynamics::BodyNode* body = skel->getBodyNode(bodyIndex);
  Eigen::Matrix3s R0 = body->getWorldTransform().linear();
  Eigen::Vector3s acc = body->getSpatialAcceleration().head<3>();

  Eigen::VectorXs originalPos = skel->getPositions();
  Eigen::VectorXs originalVel = skel->getVelocities();
  Eigen::VectorXs originalAcc = skel->getAccelerations();

  s_t eps = 1e-3;
  Eigen::Matrix3s fdR = Eigen::Matrix3s::Zero();
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        skel->setPositions(originalPos);
        skel->setVelocities(originalVel);
        skel->setAccelerations(originalAcc);

        skel->integratePositions(eps);
        perturbed = body->getWorldTransform().linear();
        return true;
      },
      fdR,
      eps,
      true);

  Eigen::Matrix3s fddR = Eigen::Matrix3s::Zero();
  // Compute ddR in an outer loop
  math::finiteDifference<Eigen::Matrix3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix3s& perturbed) {
        skel->setPositions(originalPos);
        skel->setVelocities(originalVel);
        skel->setAccelerations(originalAcc);

        skel->integratePositions(eps);
        skel->integrateVelocities(eps);

        Eigen::VectorXs innerPos = skel->getPositions();
        Eigen::VectorXs innerVel = skel->getVelocities();

        // Compute dR inside
        Eigen::Matrix3s fdR_nested = Eigen::Matrix3s::Zero();
        math::finiteDifference<Eigen::Matrix3s>(
            [&](/* in*/ s_t eps,
                /*out*/ Eigen::Matrix3s& perturbed) {
              skel->setPositions(innerPos);
              skel->setVelocities(innerVel);
              skel->setAccelerations(originalAcc);

              skel->integratePositions(eps);
              perturbed = body->getWorldTransform().linear();
              return true;
            },
            fdR_nested,
            eps,
            true);

        perturbed = fdR_nested;
        return true;
      },
      fddR,
      eps,
      true);
  skel->setPositions(originalPos);
  skel->setVelocities(originalVel);
  skel->setAccelerations(originalAcc);

  // Verify that [dw] = dR^T*dR + R^T*ddR
  Eigen::Matrix3s rawSkew = math::makeSkewSymmetric(acc.segment<3>(0));
  Eigen::Matrix3s rightTerm = fdR.transpose() * fdR + R0.transpose() * fddR;
  if (!equals(rawSkew, rightTerm, 1e-7))
  {
    std::cout << "Problem verifying [dw] = dR^T*dR + R^T*ddR" << std::endl;
    std::cout << "Body " << body->getName() << " (" << bodyIndex << ")"
              << " not equal on skew symmetric acc FD" << std::endl;
    std::cout << "dR^T*dR + R^T*ddR:" << std::endl << rightTerm << std::endl;
    std::cout << "[dw]:" << std::endl << rawSkew << std::endl;
    std::cout << "diff:" << std::endl << (rightTerm - rawSkew) << std::endl;
    return false;
  }

  return true;
}

// #ifdef ALL_TESTS
TEST(RotationGradients, ROTATIONAL_VELOCITY_SKEW_SYMMETRIC_DEF_ATLAS)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::UniversalLoader::loadSkeleton(
          world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  srand(42);
  atlas->setPositions(atlas->getRandomPose());
  atlas->setVelocities(atlas->getRandomVelocity());
  atlas->setAccelerations(atlas->getRandomVelocity());

  for (int i = 0; i < atlas->getNumBodyNodes(); i++)
  {
    bool isVelocityValid = verifyRotationVelocitySkewSymmetric(atlas, i);
    if (!isVelocityValid)
    {
      ASSERT_TRUE(isVelocityValid);
      return;
    }
  }
}
// #endif

// #ifdef ALL_TESTS
TEST(RotationGradients, ROTATIONAL_ACC_SKEW_SYMMETRIC_DEF_ATLAS)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::UniversalLoader::loadSkeleton(
          world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  srand(42);
  atlas->setPositions(atlas->getRandomPose());
  atlas->setVelocities(atlas->getRandomVelocity());
  atlas->setAccelerations(atlas->getRandomVelocity());

  for (int i = 0; i < atlas->getNumBodyNodes(); i++)
  {
    bool isAccelerationValid = verifyRotationAccelerationFormula(atlas, i);
    if (!isAccelerationValid)
    {
      ASSERT_TRUE(isAccelerationValid);
      return;
    }
  }
}
// #endif