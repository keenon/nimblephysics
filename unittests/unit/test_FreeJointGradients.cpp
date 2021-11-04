#define _USE_MATH_DEFINES
#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/collision/dart/DARTCollide.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace realtime;

#define ALL_TESTS

Eigen::MatrixXs skelPosPosJacFD(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXs pos,
    Eigen::VectorXs vel,
    s_t dt)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(dofs, dofs);
  s_t EPS = 1e-7;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = pos;
    perturbed(i) += EPS;
    Eigen::VectorXs plus = skel->integratePositionsExplicit(perturbed, vel, dt);

    perturbed = pos;
    perturbed(i) -= EPS;
    Eigen::VectorXs minus
        = skel->integratePositionsExplicit(perturbed, vel, dt);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  return jac;
}

Eigen::MatrixXs skelVelPosJacFD(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXs pos,
    Eigen::VectorXs vel,
    s_t dt)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(dofs, dofs);
  s_t EPS = 1e-7;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXs perturbed = vel;
    perturbed(i) += EPS;
    Eigen::VectorXs plus = skel->integratePositionsExplicit(pos, perturbed, dt);

    perturbed = vel;
    perturbed(i) -= EPS;
    Eigen::VectorXs minus
        = skel->integratePositionsExplicit(pos, perturbed, dt);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  return jac;
}

//==============================================================================
#ifdef ALL_TESTS
TEST(FreeJointGradients, ATLAS_JACOBIANS)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::SdfParser::readSkeleton(
          "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  world->addSkeleton(atlas);

  int dofs = atlas->getNumDofs();
  s_t dt = 0.01;
  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs pos = Eigen::VectorXs::Random(dofs);
    Eigen::VectorXs vel = Eigen::VectorXs::Random(dofs);

    Eigen::MatrixXs posPosAnalytical = atlas->getPosPosJacobian(pos, vel, dt);
    Eigen::MatrixXs velPosAnalytical = atlas->getVelPosJacobian(pos, vel, dt);
    Eigen::MatrixXs posPosFD = skelPosPosJacFD(atlas, pos, vel, dt);
    Eigen::MatrixXs velPosFD = skelVelPosJacFD(atlas, pos, vel, dt);

    const s_t tol = 3e-9;

    if (!equals(posPosAnalytical, posPosFD, tol))
    {
      std::cout << "Pos-Pos Analytical (top-left 6x6): " << std::endl
                << posPosAnalytical.block<6, 6>(0, 0) << std::endl;
      std::cout << "Pos-Pos FD (top-left 6x6): " << std::endl
                << posPosFD.block<6, 6>(0, 0) << std::endl;
      std::cout << "Pos-Pos Diff (" << (posPosAnalytical - posPosFD).minCoeff()
                << " - " << (posPosAnalytical - posPosFD).maxCoeff()
                << ") (top-left 6x6): " << std::endl
                << (posPosAnalytical - posPosFD).block<6, 6>(0, 0) << std::endl;
      break;
    }
    EXPECT_TRUE(equals(posPosAnalytical, posPosFD, tol));

    if (!equals(velPosAnalytical, velPosFD, tol))
    {
      std::cout << "Vel-Pos Analytical (top-left 6x6): " << std::endl
                << velPosAnalytical.block<6, 6>(0, 0) << std::endl;
      std::cout << "Vel-Pos FD (top-left 6x6): " << std::endl
                << velPosFD.block<6, 6>(0, 0) << std::endl;
      std::cout << "Vel-Pos Diff (" << (velPosAnalytical - velPosFD).minCoeff()
                << " - " << (velPosAnalytical - velPosFD).maxCoeff()
                << ") (top-left 6x6): " << std::endl
                << (velPosAnalytical - velPosFD).block<6, 6>(0, 0) << std::endl;
      break;
    }
    EXPECT_TRUE(equals(velPosAnalytical, velPosFD, tol));

    world->step();
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(FreeJointGradients, INTEGRATE_POSITIONS_EXPLICIT)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::SdfParser::readSkeleton(
          "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  world->addSkeleton(atlas);

  int dofs = atlas->getNumDofs();
  s_t dt = 0.01;
  for (int i = 0; i < 1000; i++)
  {
    Eigen::VectorXs pos = Eigen::VectorXs::Random(dofs);
    Eigen::VectorXs vel = Eigen::VectorXs::Random(dofs);
    atlas->setPositions(pos);
    atlas->setVelocities(vel);
    atlas->integratePositions(dt);
    Eigen::VectorXs implicitNextPos = atlas->getPositions();

    // Scramble positions
    atlas->setPositions(Eigen::VectorXs::Random(dofs));
    atlas->setVelocities(Eigen::VectorXs::Random(dofs));
    Eigen::VectorXs explicitNextPos
        = atlas->integratePositionsExplicit(pos, vel, dt);

    EXPECT_TRUE(implicitNextPos.isApprox(explicitNextPos, 1e-10));
  }
}
#endif

//==============================================================================
Eigen::Vector6s integratePos(
    Eigen::Vector6s pos, Eigen::Vector6s vel, s_t dt)
{
  const Eigen::Isometry3s mQ = FreeJoint::convertToTransform(pos);
  const Eigen::Isometry3s Qnext = mQ * FreeJoint::convertToTransform(vel * dt);

  return FreeJoint::convertToPositions(Qnext);
}

Eigen::Vector6s integratePosByParts(
    Eigen::Vector6s pos, Eigen::Vector6s vel, s_t dt)
{
  const Eigen::Matrix3s mR = BallJoint::convertToRotation(pos.head<3>());
  const Eigen::Matrix3s Rnext
      = mR * BallJoint::convertToRotation(vel.head<3>() * dt);

  Eigen::Vector6s ret = Eigen::Vector6s::Zero();
  ret.head<3>() = BallJoint::convertToPositions(Rnext);
  ret.tail<3>() = pos.tail<3>() + (mR * vel.tail<3>() * dt);

  return ret;
}

//==============================================================================
#ifdef ALL_TESTS
TEST(FreeJointGradients, FREE_JOINT_BY_PARTS)
{
  s_t dt = 0.01;

  for (int i = 0; i < 1000; i++)
  {
    Eigen::Vector6s pos = Eigen::Vector6s::Random();
    Eigen::Vector6s vel = Eigen::Vector6s::Random();
    Eigen::Vector6s nextPos = integratePos(pos, vel, dt);
    Eigen::Vector6s nextPosByParts = integratePosByParts(pos, vel, dt);

    EXPECT_TRUE(nextPosByParts.isApprox(nextPos, 1e-9));
  }
}
#endif

Eigen::Vector3s rotateBall(Eigen::Vector3s pos, Eigen::Vector3s vel, s_t dt)
{
  const Eigen::Matrix3s mR = BallJoint::convertToRotation(pos.head<3>());
  const Eigen::Matrix3s Rnext
      = mR * BallJoint::convertToRotation(vel.head<3>() * dt);
  return BallJoint::convertToPositions(Rnext);
}

Eigen::Matrix3s rotatePosPosJacFD(
    Eigen::Vector3s pos, Eigen::Vector3s vel, s_t dt)
{
  Eigen::Matrix3s jac = Eigen::Matrix3s::Zero();

  const s_t EPS = 1e-7;
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector3s perturbed = pos;
    perturbed(i) += EPS;
    Eigen::Vector3s outPos = rotateBall(perturbed, vel, dt);

    perturbed = pos;
    perturbed(i) -= EPS;
    Eigen::Vector3s outNeg = rotateBall(perturbed, vel, dt);

    jac.col(i) = (outPos - outNeg) / (2 * EPS);
  }

  return jac;
}

Eigen::Matrix3s rotateVelPosJacFD(
    Eigen::Vector3s pos, Eigen::Vector3s vel, s_t dt)
{
  Eigen::Matrix3s jac = Eigen::Matrix3s::Zero();

  const s_t EPS = 1e-7;
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector3s perturbed = vel;
    perturbed(i) += EPS;
    Eigen::Vector3s outPos = rotateBall(pos, perturbed, dt);

    perturbed = vel;
    perturbed(i) -= EPS;
    Eigen::Vector3s outNeg = rotateBall(pos, perturbed, dt);

    jac.col(i) = (outPos - outNeg) / (2 * EPS);
  }

  return jac;
}

//==============================================================================
#ifdef ALL_TESTS
TEST(FreeJointGradients, ROTATION_JOINT_JAC)
{
  s_t dt = 0.01;
  Eigen::Vector3s pos = Eigen::Vector3s::UnitX();
  Eigen::Vector3s vel = Eigen::Vector3s::UnitY();

  // Just check these don't crash
  rotatePosPosJacFD(pos, vel, dt);
  rotateVelPosJacFD(pos, vel, dt);
}
#endif

/*
//==============================================================================
#ifdef ALL_TESTS
TEST(FreeJointGradients, GUI_EXPLORE)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3s(0.0, -9.81, 0));

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));

  std::shared_ptr<dynamics::Skeleton> box = dynamics::Skeleton::create("box");
  auto pair = box->createJointAndBodyNodePair<dynamics::FreeJoint>();
  pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  pair.second->setFrictionCoeff(0.0);

  std::shared_ptr<dynamics::Skeleton> groundBox
      = dynamics::Skeleton::create("groundBox");
  auto groundPair
      = groundBox->createJointAndBodyNodePair<dynamics::WeldJoint>();
  std::shared_ptr<BoxShape> groundShape(
      new BoxShape(Eigen::Vector3s(10.0, 1.0, 10.0)));
  groundPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      groundShape);
  groundPair.second->setFrictionCoeff(1.0);
  Eigen::Isometry3s groundTransform = Eigen::Isometry3s::Identity();
  groundTransform.translation()(1) = -0.999;
  groundPair.first->setTransformFromParentBodyNode(groundTransform);

  // world->addSkeleton(atlas);
  world->addSkeleton(box);
  world->addSkeleton(groundBox);

  // Disable the ground from casting its own shadows
  groundBox->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
      false);

  world->step();

  std::vector<Eigen::Vector3s> pointsX;
  pointsX.push_back(Eigen::Vector3s::Zero());
  pointsX.push_back(Eigen::Vector3s::UnitX() * 10);
  std::vector<Eigen::Vector3s> pointsY;
  pointsY.push_back(Eigen::Vector3s::Zero());
  pointsY.push_back(Eigen::Vector3s::UnitY() * 10);
  std::vector<Eigen::Vector3s> pointsZ;
  pointsZ.push_back(Eigen::Vector3s::Zero());
  pointsZ.push_back(Eigen::Vector3s::UnitZ() * 10);

  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.createLine("unitX", pointsX, Eigen::Vector3s::UnitX());
  server.createLine("unitY", pointsY, Eigen::Vector3s::UnitY());
  server.createLine("unitZ", pointsZ, Eigen::Vector3s::UnitZ());
  server.serve(8070);

  Ticker ticker(0.01);
  ticker.registerTickListener([&](long time) {
    s_t diff = sin(((s_t)time / 2000));
    diff = diff * diff;
    // atlas->setPosition(0, diff * dart::math::constantsd::pi());
    // s_t diff2 = sin(((s_t)time / 4000));
    // atlas->setPosition(4, diff2 * 1);

    Eigen::Isometry3s fromRoot = Eigen::Isometry3s::Identity();
    fromRoot.translation() = Eigen::Vector3s::UnitZ() * diff;
    pair.first->setTransformFromParentBodyNode(fromRoot);

    server.renderWorld(world);
  });
  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
  }
}
#endif
*/