#include <iostream>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "dart/constraint/BoxedLcpConstraintSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/simulation/World.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace dynamics;
using namespace simulation;

/*
TEST(ScrewGeometry, EXP_JAC)
{
  srand(42);
  s_t EPS = 0.001;

  Eigen::Vector3s axis = Eigen::Vector3s::UnitX();
  Eigen::Matrix3s originalRotation = math::expMapRot(axis);

  Eigen::Vector3s perturb = Eigen::Vector3s::UnitY();
  Eigen::Matrix3s perturbedRotation = math::expMapRot(axis + perturb * EPS);
  Eigen::Matrix3s perturbation
      = originalRotation.transpose() * perturbedRotation;
  Eigen::Vector3s perturbPlus = math::logMap(perturbation);

  std::cout << "Perturb: " << std::endl << perturb << std::endl;
  std::cout << "+Perturb Recovered: " << std::endl << perturbPlus << std::endl;

  perturbedRotation = math::expMapRot(axis - perturb * EPS);
  perturbation = originalRotation.transpose() * perturbedRotation;
  Eigen::Vector3s perturbMinus = math::logMap(perturbation);

  std::cout << "-Perturb Recovered: " << std::endl << perturbMinus << std::endl;

  Eigen::Vector3s perturbGrad = (perturbPlus - perturbMinus) / (2 * EPS);

  std::cout << "Perturb Grad: " << std::endl << perturbGrad << std::endl;

  Eigen::Matrix3s perturbationRecovered = math::expMapRot(perturbGrad * EPS);

  std::cout << "Perturbation: " << std::endl << perturbation << std::endl;
  std::cout << "Perturbation Recovered: " << std::endl
            << perturbationRecovered << std::endl;
}
*/

/******************************************************************************

This test sets up a configuration that looks like this:

          Force
            |
            v
          +---+
Force --> |   |
          +---+
      -------------
            ^
       Fixed ground

There's a box with six DOFs, a full FreeJoint, with a force driving it into the
ground. The ground has configurable friction in this setup.

*/
void testFreeBlockWithFrictionCoeff(
    s_t frictionCoeff, s_t mass, bool freeJoint)
{
  // World
  WorldPtr world = World::create();

  // Set up the LCP solver to be super super accurate, so our
  // finite-differencing tests don't fail due to LCP errors. This isn't
  // necessary during a real forward pass, but is helpful to make the
  // mathematical invarients in the tests more reliable.
  static_cast<constraint::BoxedLcpConstraintSolver*>(
      world->getConstraintSolver())
      ->makeHyperAccurateAndVerySlow();

  ///////////////////////////////////////////////
  // Create the box
  ///////////////////////////////////////////////

  SkeletonPtr box = Skeleton::create("box");

  std::pair<Joint*, BodyNode*> pair;
  if (freeJoint)
  {
    pair = box->createJointAndBodyNodePair<FreeJoint>(nullptr);
  }
  else
  {
    pair = box->createJointAndBodyNodePair<BallJoint>(nullptr);
  }
  Joint* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  Eigen::Isometry3s fromParent = Eigen::Isometry3s::Identity();
  fromParent.translation() = Eigen::Vector3s::UnitX() * 2;
  boxJoint->setTransformFromParentBodyNode(fromParent);

  Eigen::Isometry3s fromChild = Eigen::Isometry3s::Identity();
  fromChild.translation() = Eigen::Vector3s::UnitX();
  boxJoint->setTransformFromChildBodyNode(fromChild);

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3s(1, -1, 0));
  // Prevent the mass matrix from being Identity
  boxBody->setMass(mass);

  // Move off the origin to X=1, rotate 90deg on the Y axis
  if (freeJoint)
  {
    box->setPosition(3, 1.0);
  }
  box->setPosition(1, 90 * M_PI / 180);

  world->addSkeleton(box);

  /*
  Eigen::Matrix4d realTransform = boxBody->getWorldTransform().matrix();
  Eigen::Matrix4d expTransform = math::expMapDart(box->getPositions()).matrix();
  if (realTransform != expTransform)
  {
    std::cout << "Transforms don't match!" << std::endl;
    std::cout << "Real: " << std::endl << realTransform << std::endl;
    std::cout << "Exp: " << std::endl << expTransform << std::endl;
    std::cout << "Diff: " << std::endl
              << (realTransform - expTransform) << std::endl;
    return;
  }
  */

  ///////////////////////////////////////////////
  // Create the floor
  ///////////////////////////////////////////////

  SkeletonPtr floor = Skeleton::create("floor");

  std::pair<WeldJoint*, BodyNode*> floorPair
      = floor->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* floorJoint = floorPair.first;
  BodyNode* floorBody = floorPair.second;

  Eigen::Isometry3s floorPosition = Eigen::Isometry3s::Identity();
  floorPosition.translation() = Eigen::Vector3s(0, -(1.0 - 1e-2), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3s::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3s(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(frictionCoeff);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  Eigen::VectorXs timestepVel = box->getVelocities();
  Eigen::VectorXs timestepWorldVel = world->getVelocities();

  // world->step();

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.serve(8070);

  while (server.isServing())
  {
  }
  */

  Eigen::VectorXs worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(verifyPerturbedScrewAxisForForce(world));
  /*
  EXPECT_TRUE(testScrews(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
  */
}

// #ifdef ALL_TESTS
TEST(GRADIENTS, FREE_BLOCK)
{
  testFreeBlockWithFrictionCoeff(1e7, 1, true);
}
// #endif

#ifdef ALL_TESTS
TEST(GRADIENTS, BALL_BLOCK)
{
  testFreeBlockWithFrictionCoeff(1e7, 1, false);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, FREE_VELOCITY_INTEGRATION)
{
  // World
  WorldPtr world = World::create();
  world->setGravity(Eigen::Vector3s::UnitY());

  SkeletonPtr box = Skeleton::create("box");

  std::pair<Joint*, BodyNode*> pair
      = box->createJointAndBodyNodePair<FreeJoint>(nullptr);
  Joint* boxJoint = pair.first;
  BodyNode* boxBody = pair.second;

  // boxBody->addExtForce(Eigen::Vector3s(1, -1, 0));

  world->addSkeleton(box);

  std::cout << world->getMassMatrix() << std::endl;

  Eigen::Vector6s vel = Eigen::Vector6s::Zero();
  vel(0) = 1.0;
  vel(4) = 1.0;
  world->setVelocities(vel);

  world->step();
  world->step();
  BackpropSnapshotPtr snapshot = neural::forwardPass(world, true);
  Eigen::MatrixXs jacC
      = snapshot->getJacobianOfC(world, WithRespectTo::POSITION);

  // verifyF_c(world);
  // runVelocityTest(world);
  std::cout << jacC << std::endl;
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, NORMALIZED_SCREW_GRADIENT_ROTATE_X)
{
  Eigen::Vector6s screwX = Eigen::Vector6s::Zero();
  screwX(0) = 1.0;
  Eigen::Vector3s point = Eigen::Vector3s::UnitY();
  // if we rotate point by screwX, it should move in the negative Z direction
  s_t theta = -90 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3s rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3s expectedPoint = -Eigen::Vector3s::UnitZ();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPos = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3s perturbedNeg = math::expMap(screwX * (theta - EPS)) * point;
  Eigen::Vector3s bruteForceGradient
      = (perturbedPos - perturbedNeg) / (2 * EPS);
  Eigen::Vector3s expectedGradient = Eigen::Vector3s::UnitY();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-9));

  Eigen::Vector3s analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-9));
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, NORMALIZED_SCREW_GRADIENT_RANDOM_THETA_ZERO)
{
  for (int i = 0; i < 20; i++)
  {
    Eigen::Vector6s screwRand = Eigen::Vector6s::Random();
    screwRand.head<3>() = screwRand.head<3>().normalized();

    Eigen::Vector3s point = Eigen::Vector3s::UnitY();
    s_t theta = 0;

    s_t EPS = 1e-7;
    Eigen::Vector3s perturbedPos
        = math::expMap(screwRand * (theta + EPS)) * point;
    Eigen::Vector3s perturbedNeg
        = math::expMap(screwRand * (theta - EPS)) * point;
    Eigen::Vector3s bruteForceGradient
        = (perturbedPos - perturbedNeg) / (2 * EPS);

    Eigen::Vector3s analyticalGradient
        = math::gradientWrtTheta(screwRand, point, theta);
    EXPECT_TRUE(equals(analyticalGradient, bruteForceGradient, 1e-9));
  }
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, UNNORMALIZED_SCREW_GRADIENT_RANDOM_THETA_ZERO)
{
  for (int i = 0; i < 20; i++)
  {
    Eigen::Vector6s screwRand = Eigen::Vector6s::Random();
    // screwRand.head<3>() = screwRand.head<3>().normalized();

    Eigen::Vector3s point = Eigen::Vector3s::UnitY();
    s_t theta = 0;

    s_t EPS = 1e-7;
    Eigen::Vector3s perturbedPos
        = math::expMap(screwRand * (theta + EPS)) * point;
    Eigen::Vector3s perturbedNeg
        = math::expMap(screwRand * (theta - EPS)) * point;
    Eigen::Vector3s bruteForceGradient
        = (perturbedPos - perturbedNeg) / (2 * EPS);

    Eigen::Vector3s analyticalGradient
        = math::gradientWrtTheta(screwRand, point, theta);
    EXPECT_TRUE(equals(analyticalGradient, bruteForceGradient, 1e-9));
  }
}
#endif

/*
TEST(GRADIENTS, NORMALIZED_SCREW_GRADIENT_RANDOM_THETA_RANDOM)
{
  for (int i = 0; i < 20; i++)
  {
    Eigen::Vector6s screwRand = Eigen::Vector6s::Random();
    screwRand.head<3>() = screwRand.head<3>().normalized();

    Eigen::Vector3s point = Eigen::Vector3s::UnitY();
    s_t theta = rand() * 2 * 3.1415926535;

    s_t EPS = 1e-7;
    Eigen::Vector3s perturbedPos
        = math::expMap(screwRand * (theta + EPS)) * point;
    Eigen::Vector3s perturbedNeg
        = math::expMap(screwRand * (theta - EPS)) * point;
    Eigen::Vector3s bruteForceGradient
        = (perturbedPos - perturbedNeg) / (2 * EPS);

    Eigen::Vector3s analyticalGradient
        = math::gradientWrtTheta(screwRand, point, theta);
    EXPECT_TRUE(equals(analyticalGradient, bruteForceGradient, 1e-9));
  }
}
*/