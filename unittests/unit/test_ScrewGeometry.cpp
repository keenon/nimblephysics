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

#include "TestHelpers.hpp"

using namespace dart;
using namespace dynamics;
using namespace simulation;

/*
TEST(ScrewGeometry, EXP_JAC)
{
  srand(42);
  double EPS = 0.001;

  Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
  Eigen::Matrix3d originalRotation = math::expMapRot(axis);

  Eigen::Vector3d perturb = Eigen::Vector3d::UnitY();
  Eigen::Matrix3d perturbedRotation = math::expMapRot(axis + perturb * EPS);
  Eigen::Matrix3d perturbation
      = originalRotation.transpose() * perturbedRotation;
  Eigen::Vector3d perturbPlus = math::logMap(perturbation);

  std::cout << "Perturb: " << std::endl << perturb << std::endl;
  std::cout << "+Perturb Recovered: " << std::endl << perturbPlus << std::endl;

  perturbedRotation = math::expMapRot(axis - perturb * EPS);
  perturbation = originalRotation.transpose() * perturbedRotation;
  Eigen::Vector3d perturbMinus = math::logMap(perturbation);

  std::cout << "-Perturb Recovered: " << std::endl << perturbMinus << std::endl;

  Eigen::Vector3d perturbGrad = (perturbPlus - perturbMinus) / (2 * EPS);

  std::cout << "Perturb Grad: " << std::endl << perturbGrad << std::endl;

  Eigen::Matrix3d perturbationRecovered = math::expMapRot(perturbGrad * EPS);

  std::cout << "Perturbation: " << std::endl << perturbation << std::endl;
  std::cout << "Perturbation Recovered: " << std::endl
            << perturbationRecovered << std::endl;
}
*/

bool testScrews(WorldPtr world, SkeletonPtr skel)
{
  double EPS = 1e-4;

  Eigen::VectorXd original = skel->getPositions();
  std::vector<dynamics::DegreeOfFreedom*> dofs = skel->getDofs();
  for (int dofIndex = 0; dofIndex < dofs.size(); dofIndex++)
  {
    dynamics::DegreeOfFreedom* dof = dofs[dofIndex];

    dynamics::BodyNode* node = dof->getChildBodyNode();
    Eigen::Isometry3d originalTransform = node->getWorldTransform();

    // get world twist
    int jointIndex = dof->getIndexInJoint();
    math::Jacobian relativeJac = dof->getJoint()->getRelativeJacobian();
    Eigen::Vector3d translation
        = dof->getJoint()->getRelativeTransform().translation();
    dynamics::BodyNode* childNode = dof->getChildBodyNode();
    Eigen::Isometry3d transform = childNode->getWorldTransform();
    Eigen::Vector6d worldTwist = dof->getJoint()->getWorldAxisScrew(jointIndex);
    // Eigen::Vector6d worldTwist = math::AdT(transform, localTwist);

    double pos = dof->getPosition();
    dof->setPosition(pos + EPS);

    Eigen::Matrix4d analyticalPerturbRotation
        = (math::expMap(worldTwist * EPS) * transform).matrix();
    Eigen::Matrix4d realPerturbRotation
        = childNode->getWorldTransform().matrix();

    // Reset
    dof->setPosition(pos);

    if (!equals(analyticalPerturbRotation, realPerturbRotation, 1e-8))
    {
      std::cout << "Axis: " << dofIndex << std::endl;
      std::cout << "Analytical perturbations" << std::endl
                << analyticalPerturbRotation << std::endl;
      std::cout << "Real perturbations" << std::endl
                << realPerturbRotation << std::endl;
      Eigen::Matrix4d diff = analyticalPerturbRotation - realPerturbRotation;
      std::cout << "Diff" << std::endl << diff << std::endl;

      return false;
    }
  }

  return true;
}

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
    double frictionCoeff, double mass, bool freeJoint)
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

  Eigen::Isometry3d fromParent = Eigen::Isometry3d::Identity();
  fromParent.translation() = Eigen::Vector3d::UnitX() * 2;
  boxJoint->setTransformFromParentBodyNode(fromParent);

  Eigen::Isometry3d fromChild = Eigen::Isometry3d::Identity();
  fromChild.translation() = Eigen::Vector3d::UnitX();
  boxJoint->setTransformFromChildBodyNode(fromChild);

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  boxBody->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);
  boxBody->setFrictionCoeff(frictionCoeff);

  // Add a force driving the box down into the floor, and to the left
  boxBody->addExtForce(Eigen::Vector3d(1, -1, 0));
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

  Eigen::Isometry3d floorPosition = Eigen::Isometry3d::Identity();
  floorPosition.translation() = Eigen::Vector3d(0, -(1.0 - 1e-2), 0);
  floorJoint->setTransformFromParentBodyNode(floorPosition);
  floorJoint->setTransformFromChildBodyNode(Eigen::Isometry3d::Identity());

  std::shared_ptr<BoxShape> floorShape(
      new BoxShape(Eigen::Vector3d(10.0, 1.0, 10.0)));
  floorBody->createShapeNodeWith<VisualAspect, CollisionAspect>(floorShape);
  floorBody->setFrictionCoeff(frictionCoeff);

  world->addSkeleton(floor);

  ///////////////////////////////////////////////
  // Run the tests
  ///////////////////////////////////////////////

  box->computeForwardDynamics();
  box->integrateVelocities(world->getTimeStep());
  Eigen::VectorXd timestepVel = box->getVelocities();
  Eigen::VectorXd timestepWorldVel = world->getVelocities();

  world->step();

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.serve(8070);

  while (server.isServing())
  {
  }
  */

  Eigen::VectorXd worldVel = world->getVelocities();
  // Test the classic formulation
  EXPECT_TRUE(testScrews(world, box));
  /*
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  EXPECT_TRUE(verifyAnalyticalBackprop(world));
  EXPECT_TRUE(verifyWrtMass(world));
  */
}

TEST(GRADIENTS, FREE_BLOCK)
{
  testFreeBlockWithFrictionCoeff(1e7, 1, true);
}

TEST(GRADIENTS, BALL_BLOCK)
{
  testFreeBlockWithFrictionCoeff(1e7, 1, false);
}