#include <iostream>

#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/constraint/ContactConstraint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace constraint;

/*
TEST(GRADIENTS, ODE_GRADIENTS)
{
  Eigen::Vector3s grad = Eigen::Vector3s::Random();
  Eigen::Vector3s normal = Eigen::Vector3s::Random();
  Eigen::Vector3s firstFrictionDirection = Eigen::Vector3s::UnitZ();

  ContactConstraint::TangentBasisMatrix ode
      = ContactConstraint::getTangentBasisMatrixODE(
          normal, firstFrictionDirection);

  ContactConstraint::TangentBasisMatrix odeGrad
      = ContactConstraint::getTangentBasisMatrixODEGradient(
          normal, grad, firstFrictionDirection);
}
*/

/**
 * This sets up two boxes colliding with each other. Each can rotate and
 * translate freely. We set it up so that the one box's corner is contacting
 * the other box's face.
 *
 *  /\ +---+
 * /  \|   |
 * \  /|   |
 *  \/ +---+
 */
void testVertexFaceCollision(bool isSelfCollision)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box1 = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> box1Pair
      = box1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> box1Shape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  // ShapeNode* box1Node =
  box1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      box1Shape);

  // This box is rotated by 45 degrees on the X and Y axis, so that it's
  // sqrt(3) along the X axis.
  SkeletonPtr box2 = Skeleton::create("vertex box");
  std::pair<FreeJoint*, BodyNode*> box2Pair;

  if (isSelfCollision)
  {
    box1->enableSelfCollision(true);
    box2Pair = box1Pair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    box2Pair = box2->createJointAndBodyNodePair<FreeJoint>();
  }

  s_t box2EdgeSize = sqrt(1.0 / 3);
  std::shared_ptr<BoxShape> box2Shape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0) * box2EdgeSize));
  // ShapeNode* box2Node =
  box2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      box2Shape);
  FreeJoint* box2Joint = box2Pair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  Eigen::Isometry3s box2Position = Eigen::Isometry3s::Identity();
  box2Position.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3s(0, 45, -45) * 3.1415 / 180);
  box2Position.translation()
      = box2Position.linear() * Eigen::Vector3s(1.0 - 2e-2, 0, 0);
  box2Joint->setTransformFromChildBodyNode(box2Position);

  world->addSkeleton(box1);
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  // Set the vel of the X translation of the 2nd box
  vels(9) = 0.1;
  world->setVelocities(vels);

  // renderWorld(world);
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));

  //////////////////////////////////////////////////
  // Try it in reverse skeleton order, to flip collision type enums
  //////////////////////////////////////////////////

  world->removeAllSkeletons();
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }
  world->addSkeleton(box1);

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, VERTEX_FACE_COLLISION)
{
  testVertexFaceCollision(false);
}

TEST(GRADIENTS, VERTEX_FACE_SELF_COLLISION)
{
  testVertexFaceCollision(true);
}
#endif

/**
 * This sets up two boxes colliding with each other. Each can rotate and
 * translate freely. We set it up so that the one box's edge is contacting the
 * other box's edge
 *
 *  /\
 * /  \
 * \  /+---+
 *  \/ |   |
 *     +---+
 */
void testEdgeEdgeCollision(bool isSelfCollision, bool useMesh)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box1 = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> box1Pair
      = box1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> box1Shape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  Eigen::Vector3s size = Eigen::Vector3s(1.0, 1.0, 1.0);
  if (useMesh)
  {
    aiScene* boxMesh = createBoxMeshUnsafe();
    std::shared_ptr<SharedMeshWrapper> boxMeshHolder
        = std::make_shared<SharedMeshWrapper>(boxMesh);
    std::shared_ptr<MeshShape> boxShape(
        new MeshShape(size, boxMeshHolder, "", nullptr, true));
    // ShapeNode* sphereNode =
    box1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
        boxShape);
  }
  else
  {
    std::shared_ptr<BoxShape> boxShape(new BoxShape(size));
    // ShapeNode* sphereNode =
    box1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
        boxShape);
  }

  // This box is rotated by 45 degrees on the X and Y axis, so that it's
  // sqrt(3) along the X axis.
  SkeletonPtr box2 = Skeleton::create("vertex box");
  std::pair<FreeJoint*, BodyNode*> box2Pair;

  if (isSelfCollision)
  {
    box1->enableSelfCollision(true);
    box2Pair = box1Pair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    box2Pair = box2->createJointAndBodyNodePair<FreeJoint>();
  }

  std::shared_ptr<BoxShape> box2Shape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  // ShapeNode* box2Node =
  if (useMesh)
  {
    aiScene* boxMesh = createBoxMeshUnsafe();
    std::shared_ptr<SharedMeshWrapper> boxMeshHolder
        = std::make_shared<SharedMeshWrapper>(boxMesh);
    std::shared_ptr<MeshShape> boxShape(
        new MeshShape(size, boxMeshHolder, "", nullptr, true));
    // ShapeNode* sphereNode =
    box2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
        boxShape);
  }
  else
  {
    std::shared_ptr<BoxShape> boxShape(new BoxShape(size));
    // ShapeNode* sphereNode =
    box2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
        boxShape);
  }

  FreeJoint* box2Joint = box2Pair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  Eigen::Isometry3s box2Position = Eigen::Isometry3s::Identity();
  box2Position.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3s(0, 45, 45) * 3.1415 / 180);
  box2Position.translation() = box2Position.linear() * Eigen::Vector3s(1, -1, 0)
                               * ((2 * sqrt(0.5) / sqrt(2)) - 0.01);
  box2Joint->setTransformFromChildBodyNode(box2Position);

  world->addSkeleton(box1);
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  vels(9) += 0.1;  // +x
  vels(10) -= 0.1; // -y
  world->setVelocities(vels);

  /*
  RestorableSnapshot snapshot(world);
  world->step();
  snapshot.restore();

  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.renderBasis();

  const collision::CollisionResult& result = world->getLastCollisionResult();
  const collision::Contact& contact = result.getContact(0);
  std::cout << contact.penetrationDepth << std::endl;

  std::vector<Eigen::Vector3s> edgeLineA;
  edgeLineA.push_back(contact.edgeAFixedPoint);
  edgeLineA.push_back(contact.edgeAFixedPoint + contact.edgeADir);
  server.createLine("edge_a", edgeLineA, Eigen::Vector3s(0,1,0));

  std::vector<Eigen::Vector3s> edgeLineB;
  edgeLineB.push_back(contact.edgeBFixedPoint);
  edgeLineB.push_back(contact.edgeBFixedPoint + contact.edgeBDir);
  server.createLine("edge_b", edgeLineB, Eigen::Vector3s(1,0,0));

  server.serve(8070);

  while (server.isServing())
  {
  }
  */

  /*
  std::shared_ptr<neural::BackpropSnapshot> snapshot
      = neural::forwardPass(world, true);
  snapshot->diagnoseSubJacobianErrors(world, WithRespectTo::POSITION);
  return;
  */

  // renderWorld(world);
  // EXPECT_TRUE(verifyPerturbedContactEdges(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));

  //////////////////////////////////////////////////
  // Try it in reverse skeleton order, to flip collision type enums
  //////////////////////////////////////////////////

  world->removeAllSkeletons();
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }
  world->addSkeleton(box1);

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, EDGE_EDGE_BOX_COLLISION)
{
  testEdgeEdgeCollision(false, false);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, EDGE_EDGE_BOX_SELF_COLLISION)
{
  testEdgeEdgeCollision(true, false);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, EDGE_EDGE_MESH_COLLISION)
{
  testEdgeEdgeCollision(false, true);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, EDGE_EDGE_MESH_SELF_COLLISION)
{
  testEdgeEdgeCollision(true, true);
}
#endif

/**
 * This sets up a sphere colliding with a box.
 *
 *     +---+
 *     |   |
 *    O|   |
 *     +---+
 */
void testSphereBoxCollision(bool isSelfCollision, int numFaces)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> boxPair
      = box->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3s(1.0, 1.0, 1.0)));
  // ShapeNode* boxNode =
  boxPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);

  SkeletonPtr sphere = Skeleton::create("sphere");
  std::pair<FreeJoint*, BodyNode*> spherePair;

  if (isSelfCollision)
  {
    box->enableSelfCollision(true);
    spherePair = boxPair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    spherePair = sphere->createJointAndBodyNodePair<FreeJoint>();
  }

  std::shared_ptr<SphereShape> sphereShape(new SphereShape(0.5));
  // ShapeNode* sphereNode =
  spherePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      sphereShape);
  FreeJoint* sphereJoint = spherePair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  Eigen::Isometry3s spherePosition = Eigen::Isometry3s::Identity();
  if (numFaces == 1)
  {
    spherePosition.translation() = Eigen::Vector3s(1.0 - 2e-2, 0, 0);
  }
  else if (numFaces == 2)
  {
    spherePosition.translation() = Eigen::Vector3s(
        ((0.5 - 2e-2) / sqrt(2)) + 0.5, ((0.5 - 2e-2) / sqrt(2)) + 0.5, 0);
  }
  else if (numFaces == 3)
  {
    spherePosition.translation() = Eigen::Vector3s(
        ((0.5 - 2e-2) / sqrt(3)) + 0.5,
        ((0.5 - 2e-2) / sqrt(3)) + 0.5,
        ((0.5 - 2e-2) / sqrt(3)) + 0.5);
  }
  else if (numFaces == 4)
  {
    spherePosition.translation() = Eigen::Vector3s(0.1, 0.0, 0.0);
  }
  sphereJoint->setTransformFromChildBodyNode(spherePosition);

  world->addSkeleton(box);
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  // Set the vel of the X translation of the 2nd box
  vels(9) = 0.1;
  world->setVelocities(vels);

  // renderWorld(world);
  EXPECT_TRUE(verifyAnalyticalJacobians(world, numFaces == 4));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));

  //////////////////////////////////////////////////
  // Try it in reverse skeleton order, to flip collision type enums
  //////////////////////////////////////////////////

  world->removeAllSkeletons();
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }
  world->addSkeleton(box);

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world, numFaces == 4));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, SPHERE_BOX_COLLISION_1_FACE)
{
  testSphereBoxCollision(false, 1);
}

TEST(GRADIENTS, SPHERE_BOX_COLLISION_2_FACE)
{
  testSphereBoxCollision(false, 2);
}

TEST(GRADIENTS, SPHERE_BOX_COLLISION_3_FACE)
{
  testSphereBoxCollision(false, 3);
}

TEST(GRADIENTS, SPHERE_BOX_COLLISION_4_FACE)
{
  testSphereBoxCollision(false, 4);
}

TEST(GRADIENTS, SPHERE_BOX_SELF_COLLISION_1_FACE)
{
  testSphereBoxCollision(true, 1);
}
#endif

/**
 * This sets up two spheres with asymmetric radii colliding with each other
 */
void testSphereSphereCollision(bool isSelfCollision, s_t radius1, s_t radius2)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr sphere1 = Skeleton::create("sphere 1");
  std::pair<FreeJoint*, BodyNode*> sphere1Pair
      = sphere1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<SphereShape> sphereShape1(new SphereShape(radius1));
  // ShapeNode* sphere1Node =
  sphere1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      sphereShape1);

  SkeletonPtr sphere2 = Skeleton::create("sphere 2");
  std::pair<FreeJoint*, BodyNode*> sphere2Pair;

  if (isSelfCollision)
  {
    sphere1->enableSelfCollision(true);
    sphere2Pair
        = sphere1Pair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    sphere2Pair = sphere2->createJointAndBodyNodePair<FreeJoint>();
  }

  std::shared_ptr<SphereShape> sphere2Shape(new SphereShape(radius2));
  // ShapeNode* sphereNode =
  sphere2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      sphere2Shape);
  FreeJoint* sphere2Joint = sphere2Pair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  Eigen::Isometry3s sphere2Position = Eigen::Isometry3s::Identity();
  sphere2Position.translation()
      = Eigen::Vector3s(radius1 + radius2 - 2e-2, 0, 0);
  sphere2Joint->setTransformFromChildBodyNode(sphere2Position);

  world->addSkeleton(sphere1);
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere2);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  // Set the vel of the X translation of the 2nd box
  vels(9) = 0.1;
  world->setVelocities(vels);

  // renderWorld(world);
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, SPHERE_SPHERE_COLLISION)
{
  testSphereSphereCollision(false, 0.5, 0.7);
}
#endif

/**
 * This sets up a sphere colliding with a mesh box.
 *
 *     +---+
 *     |   |
 *    O|   |
 *     +---+
 */
void testSphereMeshCollision(bool isSelfCollision, int numFaces)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> boxPair
      = box->createJointAndBodyNodePair<FreeJoint>();

  aiScene* boxMesh = createBoxMeshUnsafe();
  std::shared_ptr<SharedMeshWrapper> boxMeshHolder
      = std::make_shared<SharedMeshWrapper>(boxMesh);
  std::shared_ptr<MeshShape> boxShape(new MeshShape(
      Eigen::Vector3s(1.0, 1.0, 1.0), boxMeshHolder, "", nullptr, true));

  // ShapeNode* boxNode =
  boxPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(boxShape);

  SkeletonPtr sphere = Skeleton::create("sphere");
  std::pair<FreeJoint*, BodyNode*> spherePair;

  if (isSelfCollision)
  {
    box->enableSelfCollision(true);
    spherePair = boxPair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    spherePair = sphere->createJointAndBodyNodePair<FreeJoint>();
  }

  std::shared_ptr<SphereShape> sphereShape(new SphereShape(0.5));
  // ShapeNode* sphereNode =
  spherePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      sphereShape);
  FreeJoint* sphereJoint = spherePair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  Eigen::Isometry3s spherePosition = Eigen::Isometry3s::Identity();
  s_t penetrationDepth = 2e-3;
  if (numFaces == 1)
  {
    spherePosition.translation()
        = Eigen::Vector3s(1.0 - penetrationDepth, 0, 0);
  }
  else if (numFaces == 2)
  {
    spherePosition.translation() = Eigen::Vector3s(
        ((0.5 - penetrationDepth) * sqrt(1.0 / 2)) + 0.5,
        ((0.5 - penetrationDepth) * sqrt(1.0 / 2)) + 0.5,
        0);
  }
  else if (numFaces == 3)
  {
    spherePosition.translation() = Eigen::Vector3s(
        ((0.5 - penetrationDepth) * sqrt(1.0 / 3)) + 0.5,
        ((0.5 - penetrationDepth) * sqrt(1.0 / 3)) + 0.5,
        ((0.5 - penetrationDepth) * sqrt(1.0 / 3)) + 0.5);
  }
  else if (numFaces == 4)
  {
    spherePosition.translation() = Eigen::Vector3s(0.1, 0.0, 0.0);
  }
  sphereJoint->setTransformFromChildBodyNode(spherePosition);

  world->addSkeleton(box);
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  if (numFaces == 1 || numFaces == 2)
  {
    // Set the vel of the X translation of the 2nd box
    vels(9) = 0.1;
  }
  else if (numFaces == 3 || numFaces == 4)
  {
    // Set the vel of the X, Y, Z translation of the 2nd box
    vels(9) = 0.1;
    vels(10) = 0.1;
    vels(11) = 0.1;

    /*
    server::GUIWebsocketServer server;
    server.renderWorld(world);
    server.renderBasis();
    server.serve(8070);

    while (server.isServing())
    {
    }
    */
  }
  world->setVelocities(vels);

  // renderWorld(world);
  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  // EXPECT_TRUE(verifyPerturbedContactEdges(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world, numFaces == 4));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));

  //////////////////////////////////////////////////
  // Try it in reverse skeleton order, to flip collision type enums
  //////////////////////////////////////////////////

  world->removeAllSkeletons();
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }
  world->addSkeleton(box);

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world, numFaces == 4));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, SPHERE_MESH_COLLISION_1_FACE)
{
  testSphereMeshCollision(false, 1);
}

TEST(GRADIENTS, SPHERE_MESH_COLLISION_2_FACE)
{
  testSphereMeshCollision(false, 2);
}

TEST(GRADIENTS, SPHERE_MESH_COLLISION_3_FACE)
{
  testSphereMeshCollision(false, 3);
}

TEST(GRADIENTS, SPHERE_MESH_SELF_COLLISION_1_FACE)
{
  testSphereMeshCollision(true, 1);
}
#endif

/**
 * This sets up a sphere colliding with a capsule, either on the end or at the
 * mid-section.
 *
 *     C===D CD
 *
 *     or
 *
 *       CD
 *     C===D
 */
void testSphereCapsuleCollision(bool isSelfCollision, int type)
{
  s_t height = 1.0;
  s_t radius1 = 0.4;
  s_t radius2 = 0.3;

  Eigen::Isometry3s T1 = Eigen::Isometry3s::Identity();
  Eigen::Isometry3s T2 = Eigen::Isometry3s::Identity();
  if (type == 1)
  {
    T2.translation()(2) = height / 2 + sqrt(0.5) * (radius1 + radius2 - 0.01);
    // Move slightly off exact Z axis, to avoid issues where we're finite
    // differencing over a discontinuity in the friction cone matrix.
    T2.translation()(0) = sqrt(0.5) * (radius1 + radius2 - 0.01);
  }
  else if (type == 2)
  {
    T2.translation()(0) = radius1 + radius2 - 0.01;
  }

  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This capsule is centered at (0,0,0), and extends in the Z direction
  SkeletonPtr capsule = Skeleton::create("capsule");
  std::pair<FreeJoint*, BodyNode*> capsulePair
      = capsule->createJointAndBodyNodePair<FreeJoint>();
  capsulePair.first->setTransformFromParentBodyNode(T1);

  std::shared_ptr<CapsuleShape> boxShape(new CapsuleShape(radius1, height));

  // ShapeNode* capsuleNode =
  capsulePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      boxShape);

  SkeletonPtr sphere = Skeleton::create("sphere");
  std::pair<FreeJoint*, BodyNode*> spherePair;

  if (isSelfCollision)
  {
    capsule->enableSelfCollision(true);
    spherePair
        = capsulePair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    spherePair = sphere->createJointAndBodyNodePair<FreeJoint>();
  }

  std::shared_ptr<SphereShape> sphereShape(new SphereShape(radius2));
  // ShapeNode* sphereNode =
  spherePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      sphereShape);
  FreeJoint* sphereJoint = spherePair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  sphereJoint->setTransformFromParentBodyNode(T2);

  world->addSkeleton(capsule);
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  if (type == 1)
  {
    // Set the vel of the Z translation of the sphere
    vels(11) = -0.01;
  }
  else if (type == 2)
  {
    // Set the vel of the X translation of the sphere
    vels(9) = -0.01;
  }
  world->setVelocities(vels);

  // renderWorld(world);
  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  // EXPECT_TRUE(verifyPerturbedContactEdges(world));
  // EXPECT_TRUE(verifyPerturbedContactForceDirections(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.renderBasis();
  server.serve(8070);

  while (server.isServing())
  {
  }
  */

  //////////////////////////////////////////////////
  // Try it in reverse skeleton order, to flip collision type enums
  //////////////////////////////////////////////////

  world->removeAllSkeletons();
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }
  world->addSkeleton(capsule);

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, SPHERE_CAPSULE_END_COLLISION)
{
  testSphereCapsuleCollision(false, 1);
}

TEST(GRADIENTS, SPHERE_CAPSULE_SIDE_COLLISION)
{
  testSphereCapsuleCollision(false, 2);
}

TEST(GRADIENTS, SPHERE_CAPSULE_SIDE_SELF_COLLISION)
{
  testSphereCapsuleCollision(true, 2);
}
#endif

/**
 * This sets up a pair of capsules colliding
 */
void testCapsuleCapsuleCollision(bool isSelfCollision, int type)
{
  s_t height = 1.0;
  s_t radius1 = 0.4;
  s_t radius2 = 0.3;

  Eigen::Isometry3s T1 = Eigen::Isometry3s::Identity();

  Eigen::Isometry3s T2 = Eigen::Isometry3s::Identity();
  if (type == 1)
  {
    // T shaped
    T2.translation()(0) = radius1 + radius2 + (height / 2) - 0.01;
    T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3s(0, M_PI_2, 0));
  }
  else if (type == 2)
  {
    // X shaped
    T2.translation()(1) = radius1 + radius2 - 0.01;
    T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3s(0, M_PI_2, 0));
  }
  else if (type == 3)
  {
    // L shaped
    T2.translation()(0) = sqrt(0.5) * ((height / 2) + radius1 + radius2 - 0.01);
    T2.translation()(2)
        = (height / 2)
          + (sqrt(0.5) * ((height / 2) + radius1 + radius2 - 0.01));
    T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3s(0, M_PI_4, 0));
  }

  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This capsule is centered at (0,0,0), and extends in the Z direction
  SkeletonPtr capsuleA = Skeleton::create("capsule_A");
  std::pair<FreeJoint*, BodyNode*> capsulePair
      = capsuleA->createJointAndBodyNodePair<FreeJoint>();
  capsulePair.first->setTransformFromParentBodyNode(T1);

  std::shared_ptr<CapsuleShape> capsuleShapeA(
      new CapsuleShape(radius1, height));

  // ShapeNode* capsuleNodeA =
  capsulePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      capsuleShapeA);

  SkeletonPtr capsuleB = Skeleton::create("capsuleB");
  std::pair<FreeJoint*, BodyNode*> capsuleBPair;

  if (isSelfCollision)
  {
    capsuleA->enableSelfCollision(true);
    capsuleBPair
        = capsulePair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    capsuleBPair = capsuleB->createJointAndBodyNodePair<FreeJoint>();
  }

  std::shared_ptr<CapsuleShape> capsuleShapeB(
      new CapsuleShape(radius2, height));
  // ShapeNode* sphereNode =
  capsuleBPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      capsuleShapeB);
  FreeJoint* sphereJoint = capsuleBPair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  sphereJoint->setTransformFromParentBodyNode(T2);

  world->addSkeleton(capsuleA);
  if (!isSelfCollision)
  {
    world->addSkeleton(capsuleB);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  if (type == 1)
  {
    // Set the vel of the X translation of capsule B
    vels(9) = -0.01;
  }
  else if (type == 2)
  {
    // Set the vel of the Y translation of capsule B
    vels(10) = -0.01;
  }
  else if (type == 3)
  {
    // Set the vel of the X and Z translation of capsule B
    vels(9) = -0.01;
    vels(11) = -0.01;
  }
  // translate velocity of capsule B to local space
  vels.segment<3>(9) = T2.linear().transpose() * vels.segment<3>(9);

  world->setVelocities(vels);

  // renderWorld(world);
  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  // EXPECT_TRUE(verifyPerturbedContactEdges(world));
  // EXPECT_TRUE(verifyPerturbedContactForceDirections(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.renderBasis();
  server.serve(8070);

  while (server.isServing())
  {
  }
  */

  //////////////////////////////////////////////////
  // Try it in reverse skeleton order, to flip collision type enums
  //////////////////////////////////////////////////

  world->removeAllSkeletons();
  if (!isSelfCollision)
  {
    world->addSkeleton(capsuleB);
  }
  world->addSkeleton(capsuleA);

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, CAPSULE_CAPSULE_T_SHAPE)
{
  testCapsuleCapsuleCollision(false, 1);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, CAPSULE_CAPSULE_X_SHAPE)
{
  testCapsuleCapsuleCollision(false, 2);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, CAPSULE_CAPSULE_L_SHAPE)
{
  testCapsuleCapsuleCollision(false, 3);
}
#endif

/**
 * This sets up a pair of capsules colliding
 */
void testBoxCapsuleCollision(bool isSelfCollision, bool useMesh, int type)
{
  Eigen::Vector3s size = Eigen::Vector3s(1, 1, 1);
  s_t height = 1.0;
  s_t radius = 0.5;

  Eigen::Isometry3s T1 = Eigen::Isometry3s::Identity();

  Eigen::Isometry3s T2 = Eigen::Isometry3s::Identity();
  if (type == 1)
  {
    // capsule-edge collision
    T2.translation()(1) = 0.5 + sqrt(0.5) * (radius - 0.01);
    T2.translation()(2) = 0.5 + sqrt(0.5) * (radius - 0.01);
    T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3s(M_PI_4, 0, 0));
  }
  else if (type == 2)
  {
    // sphere-face and pipe-edge
    size = Eigen::Vector3s(2, 1, 2);
    T2.translation()(1) = 1.0 - 0.01;
    T2.translation()(2) = 1.0;
  }
  else if (type == 3)
  {
    // pipe-face collision
    size = Eigen::Vector3s(10, 1, 10);
    T2.translation()(1) = 1.0 - 0.01;
  }
  else if (type == 4)
  {
    // end-sphere -> face collision, rotated 45 deg to avoid +Z
    T1.linear() = math::eulerXYZToMatrix(Eigen::Vector3s(0, M_PI_4, 0));
    T2.translation()(0)
        = sqrt(0.5) * (size(0) / 2 + height / 2 + radius - 0.01);
    T2.translation()(2)
        = sqrt(0.5) * (size(0) / 2 + height / 2 + radius - 0.01);
    T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3s(0, M_PI_4, 0));
  }
  else if (type == 5)
  {
    // vertex - pipe collision
    T2.translation()(0)
        = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
    T2.translation()(1)
        = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
    T2.translation()(2)
        = 0.5 + sqrt(radius * radius / 3) - sqrt(0.01 * 0.01 / 3);
    T2.linear() = math::eulerXYZToMatrix(Eigen::Vector3s(M_PI_4, 0, 0));
  }

  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setPenetrationCorrectionEnabled(false);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  // This capsule is centered at (0,0,0), and extends in the Z direction
  SkeletonPtr capsule = Skeleton::create("capsule_A");
  std::pair<FreeJoint*, BodyNode*> capsulePair
      = capsule->createJointAndBodyNodePair<FreeJoint>();
  capsulePair.first->setTransformFromParentBodyNode(T1);

  std::shared_ptr<CapsuleShape> capsuleShape(new CapsuleShape(radius, height));

  // ShapeNode* capsuleNode =
  capsulePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
      capsuleShape);

  SkeletonPtr box = Skeleton::create("box");
  std::pair<FreeJoint*, BodyNode*> boxPair;

  if (isSelfCollision)
  {
    capsule->enableSelfCollision(true);
    boxPair = capsulePair.second->createChildJointAndBodyNodePair<FreeJoint>();
  }
  else
  {
    boxPair = box->createJointAndBodyNodePair<FreeJoint>();
  }

  if (useMesh)
  {
    aiScene* boxMesh = createBoxMeshUnsafe();
    std::shared_ptr<SharedMeshWrapper> boxMeshHolder
        = std::make_shared<SharedMeshWrapper>(boxMesh);
    std::shared_ptr<MeshShape> boxShape(
        new MeshShape(size, boxMeshHolder, "", nullptr, true));
    // ShapeNode* sphereNode =
    boxPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
        boxShape);
  }
  else
  {
    std::shared_ptr<BoxShape> boxShape(new BoxShape(size));
    // ShapeNode* sphereNode =
    boxPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
        boxShape);
  }

  FreeJoint* boxJoint = boxPair.first;
  FreeJoint* capsuleJoint = capsulePair.first;
  // Eigen::Matrix3s rotation = Eigen::Matrix3s::Identity();

  boxJoint->setTransformFromParentBodyNode(T1);
  capsuleJoint->setTransformFromParentBodyNode(T2);

  world->addSkeleton(capsule);
  if (!isSelfCollision)
  {
    world->addSkeleton(box);
  }

  Eigen::VectorXs vels = Eigen::VectorXs::Zero(world->getNumDofs());
  if (type == 1 || type == 2 || type == 3)
  {
    // Set the vel of the Y translation of the capsule
    vels(4) = -0.01;
  }
  else if (type == 4 || type == 5)
  {
    // Set the vel of the X translation of capsule
    vels(3) = -0.01;
  }
  // translate velocity of capsule B to local space
  vels.segment<3>(3) = T2.linear().transpose() * vels.segment<3>(3);

  world->setVelocities(vels);

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.renderBasis();
  server.serve(8070);

  server.blockWhileServing();
  */

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyJacobianOfClampingConstraints(world));
  // EXPECT_TRUE(verifyVelGradients(world, vels));
  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // return;

  // renderWorld(world);
  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  // EXPECT_TRUE(verifyPerturbedContactEdges(world));
  // EXPECT_TRUE(verifyPerturbedContactForceDirections(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));

  /*
  server::GUIWebsocketServer server;
  server.renderWorld(world);
  server.renderBasis();
  server.serve(8070);

  server.blockWhileServing();
  */

  //////////////////////////////////////////////////
  // Try it in reverse skeleton order, to flip collision type enums
  //////////////////////////////////////////////////

  world->removeAllSkeletons();
  if (!isSelfCollision)
  {
    world->addSkeleton(box);
  }
  world->addSkeleton(capsule);

  // EXPECT_TRUE(verifyPerturbedContactPositions(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, PIPE_EDGE_BOX)
{
  testBoxCapsuleCollision(false, false, 1);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, PIPE_FACE_AND_EDGE_BOX)
{
  testBoxCapsuleCollision(false, false, 2);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, PIPE_VERTEX_BOX)
{
  testBoxCapsuleCollision(false, false, 5);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, PIPE_FACE_BOX)
{
  testBoxCapsuleCollision(false, false, 3);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, END_SPHERE_FACE_BOX)
{
  testBoxCapsuleCollision(false, false, 4);
}
#endif

#ifdef ALL_TESTS
TEST(GRADIENTS, PIPE_EDGE_MESH)
{
  testBoxCapsuleCollision(false, true, 1);
}

TEST(GRADIENTS, PIPE_FACE_AND_EDGE_MESH)
{
  testBoxCapsuleCollision(false, true, 2);
}

TEST(GRADIENTS, PIPE_VERTEX_MESH)
{
  testBoxCapsuleCollision(false, true, 5);
}

TEST(GRADIENTS, PIPE_FACE_MESH)
{
  testBoxCapsuleCollision(false, true, 3);
}

TEST(GRADIENTS, END_SPHERE_FACE_MESH)
{
  testBoxCapsuleCollision(false, true, 4);
}
#endif