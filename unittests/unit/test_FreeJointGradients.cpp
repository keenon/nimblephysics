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

Eigen::MatrixXd skelPosPosJacFD(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXd pos,
    Eigen::VectorXd vel,
    double dt)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(dofs, dofs);
  double EPS = 1e-7;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXd perturbed = pos;
    perturbed(i) += EPS;
    Eigen::VectorXd plus = skel->integratePositionsExplicit(perturbed, vel, dt);

    perturbed = pos;
    perturbed(i) -= EPS;
    Eigen::VectorXd minus
        = skel->integratePositionsExplicit(perturbed, vel, dt);

    jac.col(i) = (plus - minus) / (2 * EPS);
  }

  return jac;
}

Eigen::MatrixXd skelVelPosJacFD(
    std::shared_ptr<dynamics::Skeleton> skel,
    Eigen::VectorXd pos,
    Eigen::VectorXd vel,
    double dt)
{
  int dofs = skel->getNumDofs();
  Eigen::MatrixXd jac = Eigen::MatrixXd::Zero(dofs, dofs);
  double EPS = 1e-7;
  for (int i = 0; i < dofs; i++)
  {
    Eigen::VectorXd perturbed = vel;
    perturbed(i) += EPS;
    Eigen::VectorXd plus = skel->integratePositionsExplicit(pos, perturbed, dt);

    perturbed = vel;
    perturbed(i) -= EPS;
    Eigen::VectorXd minus
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
  double dt = 0.01;
  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXd pos = Eigen::VectorXd::Random(dofs);
    Eigen::VectorXd vel = Eigen::VectorXd::Random(dofs);

    Eigen::MatrixXd posPosAnalytical = atlas->getPosPosJac(pos, vel, dt);
    Eigen::MatrixXd velPosAnalytical = atlas->getVelPosJac(pos, vel, dt);
    Eigen::MatrixXd posPosFD = skelPosPosJacFD(atlas, pos, vel, dt);
    Eigen::MatrixXd velPosFD = skelVelPosJacFD(atlas, pos, vel, dt);

    const double tol = 3e-9;

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
  double dt = 0.01;
  for (int i = 0; i < 1000; i++)
  {
    Eigen::VectorXd pos = Eigen::VectorXd::Random(dofs);
    Eigen::VectorXd vel = Eigen::VectorXd::Random(dofs);
    atlas->setPositions(pos);
    atlas->setVelocities(vel);
    atlas->integratePositions(dt);
    Eigen::VectorXd implicitNextPos = atlas->getPositions();

    // Scramble positions
    atlas->setPositions(Eigen::VectorXd::Random(dofs));
    atlas->setVelocities(Eigen::VectorXd::Random(dofs));
    Eigen::VectorXd explicitNextPos
        = atlas->integratePositionsExplicit(pos, vel, dt);

    EXPECT_TRUE(implicitNextPos.isApprox(explicitNextPos, 1e-10));
  }
}
#endif

//==============================================================================
Eigen::Vector6d integratePos(
    Eigen::Vector6d pos, Eigen::Vector6d vel, double dt)
{
  const Eigen::Isometry3d mQ = FreeJoint::convertToTransform(pos);
  const Eigen::Isometry3d Qnext = mQ * FreeJoint::convertToTransform(vel * dt);

  return FreeJoint::convertToPositions(Qnext);
}

Eigen::Vector6d integratePosByParts(
    Eigen::Vector6d pos, Eigen::Vector6d vel, double dt)
{
  const Eigen::Matrix3d mR = BallJoint::convertToRotation(pos.head<3>());
  const Eigen::Matrix3d Rnext
      = mR * BallJoint::convertToRotation(vel.head<3>() * dt);

  Eigen::Vector6d ret = Eigen::Vector6d::Zero();
  ret.head<3>() = BallJoint::convertToPositions(Rnext);
  ret.tail<3>() = pos.tail<3>() + (mR * vel.tail<3>() * dt);

  return ret;
}

//==============================================================================
#ifdef ALL_TESTS
TEST(FreeJointGradients, FREE_JOINT_BY_PARTS)
{
  double dt = 0.01;

  for (int i = 0; i < 1000; i++)
  {
    Eigen::Vector6d pos = Eigen::Vector6d::Random();
    Eigen::Vector6d vel = Eigen::Vector6d::Random();
    Eigen::Vector6d nextPos = integratePos(pos, vel, dt);
    Eigen::Vector6d nextPosByParts = integratePosByParts(pos, vel, dt);

    EXPECT_TRUE(nextPosByParts.isApprox(nextPos, 1e-9));
  }
}
#endif

Eigen::Vector3d rotateBall(Eigen::Vector3d pos, Eigen::Vector3d vel, double dt)
{
  const Eigen::Matrix3d mR = BallJoint::convertToRotation(pos.head<3>());
  const Eigen::Matrix3d Rnext
      = mR * BallJoint::convertToRotation(vel.head<3>() * dt);
  return BallJoint::convertToPositions(Rnext);
}

Eigen::Matrix3d rotatePosPosJacFD(
    Eigen::Vector3d pos, Eigen::Vector3d vel, double dt)
{
  Eigen::Matrix3d jac = Eigen::Matrix3d::Zero();

  const double EPS = 1e-7;
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector3d perturbed = pos;
    perturbed(i) += EPS;
    Eigen::Vector3d outPos = rotateBall(perturbed, vel, dt);

    perturbed = pos;
    perturbed(i) -= EPS;
    Eigen::Vector3d outNeg = rotateBall(perturbed, vel, dt);

    jac.col(i) = (outPos - outNeg) / (2 * EPS);
  }

  return jac;
}

Eigen::Matrix3d rotateVelPosJacFD(
    Eigen::Vector3d pos, Eigen::Vector3d vel, double dt)
{
  Eigen::Matrix3d jac = Eigen::Matrix3d::Zero();

  const double EPS = 1e-7;
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector3d perturbed = vel;
    perturbed(i) += EPS;
    Eigen::Vector3d outPos = rotateBall(pos, perturbed, dt);

    perturbed = vel;
    perturbed(i) -= EPS;
    Eigen::Vector3d outNeg = rotateBall(pos, perturbed, dt);

    jac.col(i) = (outPos - outNeg) / (2 * EPS);
  }

  return jac;
}

//==============================================================================
#ifdef ALL_TESTS
TEST(FreeJointGradients, ROTATION_JOINT_JAC)
{
  double dt = 0.01;
  Eigen::Vector3d pos = Eigen::Vector3d::UnitX();
  Eigen::Vector3d vel = Eigen::Vector3d::UnitY();

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
  world->setConstraintForceMixingEnabled(true);
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3d(0.0, -9.81, 0));

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));

  std::shared_ptr<dynamics::Skeleton> box = dynamics::Skeleton::create("box");
  auto pair = box->createJointAndBodyNodePair<dynamics::FreeJoint>();
  pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  pair.second->setFrictionCoeff(0.0);

  std::shared_ptr<dynamics::Skeleton> groundBox
      = dynamics::Skeleton::create("groundBox");
  auto groundPair
      = groundBox->createJointAndBodyNodePair<dynamics::WeldJoint>();
  std::shared_ptr<BoxShape> groundShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  groundPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      groundShape);
  groundPair.second->setFrictionCoeff(1.0);
  Eigen::Isometry3d groundTransform = Eigen::Isometry3d::Identity();
  groundTransform.translation()(1) = -0.999;
  groundPair.first->setTransformFromParentBodyNode(groundTransform);

  // world->addSkeleton(atlas);
  world->addSkeleton(box);
  world->addSkeleton(groundBox);

  // Disable the ground from casting its own shadows
  groundBox->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
      false);

  world->step();

  std::vector<Eigen::Vector3d> pointsX;
  pointsX.push_back(Eigen::Vector3d::Zero());
  pointsX.push_back(Eigen::Vector3d::UnitX() * 10);
  std::vector<Eigen::Vector3d> pointsY;
  pointsY.push_back(Eigen::Vector3d::Zero());
  pointsY.push_back(Eigen::Vector3d::UnitY() * 10);
  std::vector<Eigen::Vector3d> pointsZ;
  pointsZ.push_back(Eigen::Vector3d::Zero());
  pointsZ.push_back(Eigen::Vector3d::UnitZ() * 10);

  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.createLine("unitX", pointsX, Eigen::Vector3d::UnitX());
  server.createLine("unitY", pointsY, Eigen::Vector3d::UnitY());
  server.createLine("unitZ", pointsZ, Eigen::Vector3d::UnitZ());
  server.serve(8070);

  Ticker ticker(0.01);
  ticker.registerTickListener([&](long time) {
    double diff = sin(((double)time / 2000));
    diff = diff * diff;
    // atlas->setPosition(0, diff * dart::math::constantsd::pi());
    // double diff2 = sin(((double)time / 4000));
    // atlas->setPosition(4, diff2 * 1);

    Eigen::Isometry3d fromRoot = Eigen::Isometry3d::Identity();
    fromRoot.translation() = Eigen::Vector3d::UnitZ() * diff;
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