#include <iostream>

#include <dart/gui/gui.hpp>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
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
#include "dart/simulation/World.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

// #define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;

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
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box1 = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> box1Pair
      = box1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> box1Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* box1Node
      = box1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
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

  double box2EdgeSize = sqrt(1.0 / 3);
  std::shared_ptr<BoxShape> box2Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0) * box2EdgeSize));
  ShapeNode* box2Node
      = box2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          box2Shape);
  FreeJoint* box2Joint = box2Pair.first;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

  Eigen::Isometry3d box2Position = Eigen::Isometry3d::Identity();
  box2Position.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3d(0, 45, -45) * 3.1415 / 180);
  box2Position.translation()
      = box2Position.linear() * Eigen::Vector3d(1.0 - 2e-2, 0, 0);
  box2Joint->setTransformFromChildBodyNode(box2Position);

  world->addSkeleton(box1);
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }

  Eigen::VectorXd vels = Eigen::VectorXd::Zero(world->getNumDofs());
  // Set the vel of the X translation of the 2nd box
  vels(9) = 0.1;
  world->setVelocities(vels);

  // renderWorld(world);
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
void testEdgeEdgeCollision(bool isSelfCollision)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box1 = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> box1Pair
      = box1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> box1Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* box1Node
      = box1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
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

  std::shared_ptr<BoxShape> box2Shape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* box2Node
      = box2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          box2Shape);
  FreeJoint* box2Joint = box2Pair.first;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

  Eigen::Isometry3d box2Position = Eigen::Isometry3d::Identity();
  box2Position.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3d(0, 45, 45) * 3.1415 / 180);
  box2Position.translation() = box2Position.linear() * Eigen::Vector3d(1, -1, 0)
                               * ((2 * sqrt(0.5) / sqrt(2)) - 2e-2);
  box2Joint->setTransformFromChildBodyNode(box2Position);

  world->addSkeleton(box1);
  if (!isSelfCollision)
  {
    world->addSkeleton(box2);
  }

  Eigen::VectorXd vels = Eigen::VectorXd::Zero(world->getNumDofs());
  vels(9) += 0.1;  // +x
  vels(10) -= 0.1; // -y
  world->setVelocities(vels);

  // renderWorld(world);
  // EXPECT_TRUE(verifyPerturbedContactEdges(world));
  // EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
}

#ifdef ALL_TESTS
TEST(GRADIENTS, EDGE_EDGE_COLLISION)
{
  testEdgeEdgeCollision(false);
}

TEST(GRADIENTS, EDGE_EDGE_SELF_COLLISION)
{
  testEdgeEdgeCollision(true);
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
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> boxPair
      = box->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));
  ShapeNode* boxNode
      = boxPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          boxShape);

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
  ShapeNode* sphereNode
      = spherePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          sphereShape);
  FreeJoint* sphereJoint = spherePair.first;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

  Eigen::Isometry3d spherePosition = Eigen::Isometry3d::Identity();
  if (numFaces == 1)
  {
    spherePosition.translation() = Eigen::Vector3d(1.0 - 2e-2, 0, 0);
  }
  else if (numFaces == 2)
  {
    spherePosition.translation() = Eigen::Vector3d(
        (0.5 / sqrt(2)) + 0.5 - 2e-2, (0.5 / sqrt(2)) + 0.5 - 2e-2, 0);
  }
  else if (numFaces == 3)
  {
    spherePosition.translation() = Eigen::Vector3d(
        (0.5 / sqrt(3)) + 0.5 - 2e-2,
        (0.5 / sqrt(3)) + 0.5 - 2e-2,
        (0.5 / sqrt(3)) + 0.5 - 2e-2);
  }
  else if (numFaces == 4)
  {
    spherePosition.translation() = Eigen::Vector3d(0.1, 0.0, 0.0);
  }
  sphereJoint->setTransformFromChildBodyNode(spherePosition);

  world->addSkeleton(box);
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }

  Eigen::VectorXd vels = Eigen::VectorXd::Zero(world->getNumDofs());
  // Set the vel of the X translation of the 2nd box
  vels(9) = 0.1;
  world->setVelocities(vels);

  // renderWorld(world);
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
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
void testSphereSphereCollision(
    bool isSelfCollision, double radius1, double radius2)
{
  // World
  WorldPtr world = World::create();
  auto collision_detector
      = collision::CollisionDetector::getFactory()->create("dart");
  world->getConstraintSolver()->setCollisionDetector(collision_detector);
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr sphere1 = Skeleton::create("sphere 1");
  std::pair<FreeJoint*, BodyNode*> sphere1Pair
      = sphere1->createJointAndBodyNodePair<FreeJoint>();
  std::shared_ptr<SphereShape> sphereShape1(new SphereShape(radius1));
  ShapeNode* sphere1Node
      = sphere1Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
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
  ShapeNode* sphereNode
      = sphere2Pair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          sphere2Shape);
  FreeJoint* sphere2Joint = sphere2Pair.first;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

  Eigen::Isometry3d sphere2Position = Eigen::Isometry3d::Identity();
  sphere2Position.translation()
      = Eigen::Vector3d(radius1 + radius2 - 2e-2, 0, 0);
  sphere2Joint->setTransformFromChildBodyNode(sphere2Position);

  world->addSkeleton(sphere1);
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere2);
  }

  Eigen::VectorXd vels = Eigen::VectorXd::Zero(world->getNumDofs());
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
  world->setGravity(Eigen::Vector3d(0, -9.81, 0));

  // This box is centered at (0,0,0), and extends to [-0.5, 0.5] on every axis
  SkeletonPtr box = Skeleton::create("face box");
  std::pair<FreeJoint*, BodyNode*> boxPair
      = box->createJointAndBodyNodePair<FreeJoint>();

  aiScene* boxMesh = createBoxMeshUnsafe();
  std::shared_ptr<MeshShape> boxShape(
      new MeshShape(Eigen::Vector3d(1.0, 1.0, 1.0), boxMesh));

  ShapeNode* boxNode
      = boxPair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          boxShape);

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
  ShapeNode* sphereNode
      = spherePair.second->createShapeNodeWith<VisualAspect, CollisionAspect>(
          sphereShape);
  FreeJoint* sphereJoint = spherePair.first;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();

  Eigen::Isometry3d spherePosition = Eigen::Isometry3d::Identity();
  if (numFaces == 1)
  {
    spherePosition.translation() = Eigen::Vector3d(1.0 - 2e-2, 0, 0);
  }
  else if (numFaces == 2)
  {
    spherePosition.translation() = Eigen::Vector3d(
        (0.5 / sqrt(2)) + 0.5 - 2e-2, (0.5 / sqrt(2)) + 0.5 - 2e-2, 0);
  }
  else if (numFaces == 3)
  {
    spherePosition.translation() = Eigen::Vector3d(
        (0.5 / sqrt(3)) + 0.5 - 2e-2,
        (0.5 / sqrt(3)) + 0.5 - 2e-2,
        (0.5 / sqrt(3)) + 0.5 - 2e-2);
  }
  else if (numFaces == 4)
  {
    spherePosition.translation() = Eigen::Vector3d(0.1, 0.0, 0.0);
  }
  sphereJoint->setTransformFromChildBodyNode(spherePosition);

  world->addSkeleton(box);
  if (!isSelfCollision)
  {
    world->addSkeleton(sphere);
  }

  Eigen::VectorXd vels = Eigen::VectorXd::Zero(world->getNumDofs());
  // Set the vel of the X translation of the 2nd box
  vels(9) = 0.1;
  world->setVelocities(vels);

  // renderWorld(world);
  EXPECT_TRUE(verifyPerturbedContactPositions(world));
  /*
  EXPECT_TRUE(verifyPerturbedContactNormals(world));
  EXPECT_TRUE(verifyPerturbedContactEdges(world));
  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  EXPECT_TRUE(verifyVelGradients(world, vels));
  EXPECT_TRUE(verifyWrtMass(world));
  */
}

TEST(GRADIENTS, SPHERE_MESH_COLLISION_1_FACE)
{
  testSphereMeshCollision(false, 1);
}

#ifdef ALL_TESTS
TEST(GRADIENTS, SPHERE_MESH_COLLISION_2_FACE)
{
  testSphereMeshCollision(false, 2);
}

TEST(GRADIENTS, SPHERE_MESH_COLLISION_3_FACE)
{
  testSphereMeshCollision(false, 3);
}

TEST(GRADIENTS, SPHERE_MESH_COLLISION_4_FACE)
{
  testSphereMeshCollision(false, 4);
}

TEST(GRADIENTS, SPHERE_MESH_SELF_COLLISION_1_FACE)
{
  testSphereMeshCollision(true, 1);
}
#endif