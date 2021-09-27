#define _USE_MATH_DEFINES
#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/collision/CollisionResult.hpp"
#include "dart/collision/dart/DARTCollide.hpp"
#include "dart/constraint/BoxedLcpConstraintSolver.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace realtime;

// #define ALL_TESTS

#ifdef ALL_TESTS
TEST(DARTCollide, ATLAS_5_HIP_FOOT)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  world->setPenetrationCorrectionEnabled(false);
  world->setGravity(Eigen::Vector3s(0.0, -9.81, 0));

  // Set up the LCP solver to be super super accurate, so our
  // finite-differencing tests don't fail due to LCP errors. This isn't
  // necessary during a real forward pass, but is helpful to make the
  // mathematical invarients in the tests more reliable.
  static_cast<constraint::BoxedLcpConstraintSolver*>(
      world->getConstraintSolver())
      ->makeHyperAccurateAndVerySlow();

  // Load the meshes

  auto retriever = std::make_shared<utils::CompositeResourceRetriever>();
  retriever->addSchemaRetriever(
      "file", std::make_shared<common::LocalResourceRetriever>());
  retriever->addSchemaRetriever("dart", utils::DartResourceRetriever::create());
  std::string leftFootPath = "dart://sample/sdf/atlas/l_foot.dae";
  dynamics::ShapePtr leftFootMesh = std::make_shared<dynamics::MeshShape>(
      Eigen::Vector3s::Ones(),
      dynamics::MeshShape::loadMesh(leftFootPath, retriever),
      leftFootPath,
      retriever);
  std::string rightFootPath = "dart://sample/sdf/atlas/r_foot.dae";
  dynamics::ShapePtr rightFootMesh = std::make_shared<dynamics::MeshShape>(
      Eigen::Vector3s::Ones(),
      dynamics::MeshShape::loadMesh(rightFootPath, retriever),
      rightFootPath,
      retriever);
  std::shared_ptr<BoxShape> rootShape(
      new BoxShape(Eigen::Vector3s::Ones() * 0.5));

  // Set up matrices

  Eigen::Matrix4d l_leg_hpx_WorldTransform;
  // clang-format off
  l_leg_hpx_WorldTransform << 1,           0,           0,           0,
                              0, 1.11022e-16,           1,       -0.01,
                              0,          -1, 1.11022e-16,      -0.089,
                              0,           0,           0,           1;
  // clang-format on
  Eigen::Vector3s l_leg_hpx_Axis = Eigen::Vector3s::UnitX();
  Eigen::Matrix4d r_leg_hpx_WorldTransform;
  // clang-format off
  r_leg_hpx_WorldTransform << 1,           0,           0,           0,
                              0, 1.11022e-16,           1,       -0.01,
                              0,          -1, 1.11022e-16,       0.089,
                              0,           0,           0,           1;
  // clang-format on
  Eigen::Vector3s r_leg_hpx_Axis = Eigen::Vector3s::UnitX();
  Eigen::Matrix4d l_foot_WorldTransform;
  // clang-format off
  l_foot_WorldTransform << 1,           0,           0,           0,
                           0, 1.11022e-16,           1,      -0.856,
                           0,          -1, 1.11022e-16,      -0.089,
                           0,           0,           0,           1;
  // clang-format on
  Eigen::Matrix4d r_foot_WorldTransform;
  // clang-format off
  r_foot_WorldTransform << 1,           0,           0,           0,
                           0, 1.11022e-16,           1,      -0.856,
                           0,          -1, 1.11022e-16,       0.089,
                           0,           0,           0,           1;
  // clang-format on

  std::shared_ptr<dynamics::Skeleton> singleJointAtlas
      = dynamics::Skeleton::create();

  Eigen::Isometry3s leftHipPos = Eigen::Isometry3s(l_leg_hpx_WorldTransform);
  Eigen::Isometry3s rightHipPos = Eigen::Isometry3s(r_leg_hpx_WorldTransform);
  Eigen::Isometry3s leftFootPos = Eigen::Isometry3s(l_foot_WorldTransform);
  Eigen::Isometry3s rightFootPos = Eigen::Isometry3s(r_foot_WorldTransform);
  Eigen::Isometry3s leftHipToLeftFoot = leftHipPos.inverse() * leftFootPos;
  Eigen::Isometry3s rightHipToRightFoot = rightHipPos.inverse() * rightFootPos;

  // Create root box

  std::pair<dynamics::FreeJoint*, dynamics::BodyNode*> rootPair
      = singleJointAtlas->createJointAndBodyNodePair<dynamics::FreeJoint>();
  dynamics::FreeJoint* rootJoint = rootPair.first;
  dynamics::BodyNode* rootBody = rootPair.second;
  rootBody->createShapeNodeWith<VisualAspect, CollisionAspect>(rootShape);

  // Create left leg

  std::pair<dynamics::RevoluteJoint*, dynamics::BodyNode*> leftHipPair
      = rootBody->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>();
  dynamics::RevoluteJoint* leftHip = leftHipPair.first;
  leftHip->setTransformFromParentBodyNode(leftHipPos);
  leftHip->setTransformFromChildBodyNode(leftHipToLeftFoot.inverse());
  leftHip->setAxis(l_leg_hpx_Axis);
  dynamics::BodyNode* leftFoot = leftHipPair.second;
  leftFoot->createShapeNodeWith<VisualAspect, CollisionAspect>(leftFootMesh);
  assert(leftFoot->getWorldTransform().matrix() == leftFootPos.matrix());

  // Create right leg

  std::pair<dynamics::RevoluteJoint*, dynamics::BodyNode*> rightHipPair
      = rootBody->createChildJointAndBodyNodePair<dynamics::RevoluteJoint>();
  dynamics::RevoluteJoint* rightHip = rightHipPair.first;
  rightHip->setTransformFromParentBodyNode(rightHipPos);
  rightHip->setTransformFromChildBodyNode(rightHipToRightFoot.inverse());
  rightHip->setAxis(r_leg_hpx_Axis);
  dynamics::BodyNode* rightFoot = rightHipPair.second;
  rightFoot->createShapeNodeWith<VisualAspect, CollisionAspect>(rightFootMesh);
  assert(rightFoot->getWorldTransform().matrix() == rightFootPos.matrix());

  world->addSkeleton(singleJointAtlas);

  // Load ground
  dart::utils::DartLoader urdfLoader;
  std::shared_ptr<dynamics::Skeleton> ground
      = urdfLoader.parseSkeleton("dart://sample/sdf/atlas/ground.urdf");
  world->addSkeleton(ground);
  // Disable the ground from casting its own shadows
  ground->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
      false);

  // Set up the world

  // give velocity parallel to the ground plane, to move friction away from
  // non-differentiable points
  // singleJointAtlas->setVelocity(0, 0.01);
  // singleJointAtlas->setVelocity(2, 0.01);

  world->step();
  Eigen::VectorXs worldVel = world->getVelocities();

  // Run the tests

  // EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // std::cout << "Passed Analytical Jacobians" << std::endl;
  // EXPECT_TRUE(verifyScratch(world, WithRespectTo::FORCE));
  // EXPECT_TRUE(verifyF_c(world));
  // std::cout << "Passed F_c" << std::endl;
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  std::cout << "Passed Vel Gradients" << std::endl;
  EXPECT_TRUE(verifyPosGradients(world, 1, 1e-8));
  std::cout << "Passed Pos Gradients" << std::endl;
  EXPECT_TRUE(verifyWrtMass(world));
  std::cout << "Passed Mass Gradients" << std::endl;

  /*

  // Display everything out on a GUI

  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.serve(8070);

  Ticker ticker(0.01);
  ticker.registerTickListener([&](long time) {
    s_t diff = sin(((s_t)time / 2000));
    singleJointAtlas->setPosition(6, diff);
    singleJointAtlas->setPosition(7, -diff);
    // world->step();
    server.renderWorld(world);
  });
  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
  }
  */
}
#endif

void testAtlas(bool withGroundContact)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  world->setPenetrationCorrectionEnabled(false);
  world->setGravity(Eigen::Vector3s(0.0, -9.81, 0));

  // Set up the LCP solver to be super super accurate, so our
  // finite-differencing tests don't fail due to LCP errors. This isn't
  // necessary during a real forward pass, but is helpful to make the
  // mathematical invarients in the tests more reliable.
  static_cast<constraint::BoxedLcpConstraintSolver*>(
      world->getConstraintSolver())
      ->makeHyperAccurateAndVerySlow();

  // Load ground and Atlas robot and add them to the world
  dart::utils::DartLoader urdfLoader;

  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::SdfParser::readSkeleton(
          "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  world->addSkeleton(atlas);

  if (withGroundContact)
  {
    std::shared_ptr<dynamics::Skeleton> ground
        = urdfLoader.parseSkeleton("dart://sample/sdf/atlas/ground.urdf");
    world->addSkeleton(ground);
  }

  // Set initial configuration for Atlas robot
  atlas->setPosition(0, -0.5 * dart::math::constantsd::pi());
  atlas->setPosition(4, -0.01);
  Eigen::VectorXs originalPos = atlas->getPositions();
  Eigen::VectorXs worldVel = world->getVelocities();

  atlas->setVelocities(Eigen::VectorXs::Zero(atlas->getNumDofs()));
  // Give a tiny bit of lateral motion, to keep the friction away from
  // non-differentiable points that can have lots of valid gradients
  // atlas->setVelocity(3, 0.01);
  // atlas->setVelocity(5, 0.01);

  /*
  world->step();
  auto& result = world->getLastCollisionResult();
  for (int i = 0; i < result.getNumContacts(); i++)
  {
    std::cout << "Depth[" << i << "]: " << result.getContact(i).penetrationDepth
              << std::endl;
  }
  return;
  */

  // EXPECT_TRUE(verifyAnalyticalJacobians(world));

  // EXPECT_TRUE(verifyPosVelJacobian(world, worldVel));

  EXPECT_TRUE(verifyF_c(world));
  std::cout << "Passed F_c" << std::endl;
  EXPECT_TRUE(verifyVelGradients(world, worldVel));
  std::cout << "Passed vel" << std::endl;
  EXPECT_TRUE(verifyPosGradients(world, 1, 1e-8));
  std::cout << "Passed pos" << std::endl;
  EXPECT_TRUE(verifyWrtMass(world));
  std::cout << "Passed mass" << std::endl;
  std::cout << "Passed everything except backprop" << std::endl;
  // This is outrageously slow
  // EXPECT_TRUE(verifyAnalyticalBackprop(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, ATLAS_FLOATING)
{
  testAtlas(false);
}
#endif

// #ifdef ALL_TESTS
TEST(GRADIENTS, ATLAS_GROUND)
{
  testAtlas(true);
}
// #endif