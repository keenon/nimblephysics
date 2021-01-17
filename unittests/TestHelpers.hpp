/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file TestHelper.h
 * @author Can Erdogan
 * @date Feb 03, 2013
 * @brief Contains the helper functions for the tests.
 */

#ifndef DART_UNITTESTS_TEST_HELPERS_H
#define DART_UNITTESTS_TEST_HELPERS_H

#include <vector>

#include <Eigen/Dense>
#include <assimp/scene.h>

#include "dart/collision/CollisionDetector.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/dynamics.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/simulation/World.hpp"

using namespace Eigen;
using namespace dart::math;
using namespace dart::collision;
using namespace dart::dynamics;
using namespace dart::simulation;

/// Function headers
enum TypeOfDOF
{
  DOF_X,
  DOF_Y,
  DOF_Z,
  DOF_ROLL,
  DOF_PITCH,
  DOF_YAW
};

//==============================================================================
/// Returns true if the two matrices are equal within the given bound
template <class MATRIX>
bool equals(
    const Eigen::DenseBase<MATRIX>& _expected,
    const Eigen::DenseBase<MATRIX>& _actual,
    double tol = 1e-5)
{
  // Get the matrix sizes and sanity check the call
  const size_t n1 = _expected.cols(), m1 = _expected.rows();
  const size_t n2 = _actual.cols(), m2 = _actual.rows();
  if (m1 != m2 || n1 != n2)
    return false;

  // Check each index
  for (size_t i = 0; i < m1; i++)
  {
    for (size_t j = 0; j < n1; j++)
    {
      if (std::isnan(_expected(i, j)) ^ std::isnan(_actual(i, j)))
        return false;
      else if (fabs(_expected(i, j)) > 1)
      {
        // Test relative error for values that are larger than 1
        if (fabs((_expected(i, j) - _actual(i, j)) / _expected(i, j)) > tol)
          return false;
      }
      else if (fabs(_expected(i, j) - _actual(i, j)) > tol)
        return false;
    }
  }

  // If no problems, the two matrices are equal
  return true;
}

//==============================================================================
bool equals(
    const Eigen::Isometry3d& tf1,
    const Eigen::Isometry3d& tf2,
    double tol = 1e-5)
{
  auto se3 = dart::math::logMap(tf1.inverse() * tf2);
  auto norm = se3.norm();

  return (norm < tol);
}

//==============================================================================
/// Add an end-effector to the last link of the given robot
void addEndEffector(SkeletonPtr robot, BodyNode* parent_node, Vector3d dim)
{
  // Create the end-effector node with a random dimension
  BodyNode::Properties node(BodyNode::AspectProperties("ee"));
  std::shared_ptr<Shape> shape(new BoxShape(Vector3d(0.2, 0.2, 0.2)));

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translate(Eigen::Vector3d(0.0, 0.0, dim(2)));
  Joint::Properties joint("eeJoint", T);

  auto pair
      = robot->createJointAndBodyNodePair<WeldJoint>(parent_node, joint, node);
  auto bodyNode = pair.second;
  bodyNode->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
      shape);
}

//==============================================================================
std::pair<Joint*, BodyNode*> add1DofJoint(
    SkeletonPtr skel,
    BodyNode* parent,
    const BodyNode::Properties& node,
    const std::string& name,
    double val,
    double min,
    double max,
    int type)
{
  GenericJoint<R1Space>::Properties properties(name);
  properties.mPositionLowerLimits[0] = min;
  properties.mPositionUpperLimits[0] = max;
  std::pair<Joint*, BodyNode*> newComponent;
  if (DOF_X == type)
    newComponent = skel->createJointAndBodyNodePair<PrismaticJoint>(
        parent,
        PrismaticJoint::Properties(properties, Vector3d(1.0, 0.0, 0.0)),
        node);
  else if (DOF_Y == type)
    newComponent = skel->createJointAndBodyNodePair<PrismaticJoint>(
        parent,
        PrismaticJoint::Properties(properties, Vector3d(0.0, 1.0, 0.0)),
        node);
  else if (DOF_Z == type)
    newComponent = skel->createJointAndBodyNodePair<PrismaticJoint>(
        parent,
        PrismaticJoint::Properties(properties, Vector3d(0.0, 0.0, 1.0)),
        node);
  else if (DOF_YAW == type)
    newComponent = skel->createJointAndBodyNodePair<RevoluteJoint>(
        parent,
        RevoluteJoint::Properties(properties, Vector3d(0.0, 0.0, 1.0)),
        node);
  else if (DOF_PITCH == type)
    newComponent = skel->createJointAndBodyNodePair<RevoluteJoint>(
        parent,
        RevoluteJoint::Properties(properties, Vector3d(0.0, 1.0, 0.0)),
        node);
  else if (DOF_ROLL == type)
    newComponent = skel->createJointAndBodyNodePair<RevoluteJoint>(
        parent,
        RevoluteJoint::Properties(properties, Vector3d(1.0, 0.0, 0.0)),
        node);

  newComponent.first->setPosition(0, val);
  return newComponent;
}

SkeletonPtr createCartpole()
{
  SkeletonPtr cartpole = Skeleton::create("cartpole");

  std::pair<PrismaticJoint*, BodyNode*> sledPair
      = cartpole->createJointAndBodyNodePair<PrismaticJoint>(nullptr);
  sledPair.first->setAxis(Eigen::Vector3d(1, 0, 0));
  std::shared_ptr<BoxShape> sledShapeBox(
      new BoxShape(Eigen::Vector3d(0.05, 0.25, 0.05)));
  ShapeNode* sledShape
      = sledPair.second->createShapeNodeWith<VisualAspect>(sledShapeBox);

  std::pair<RevoluteJoint*, BodyNode*> armPair
      = cartpole->createJointAndBodyNodePair<RevoluteJoint>(sledPair.second);
  armPair.first->setAxis(Eigen::Vector3d(0, 0, 1));
  std::shared_ptr<BoxShape> armShapeBox(
      new BoxShape(Eigen::Vector3d(0.05, 0.25, 0.05)));
  ShapeNode* armShape
      = armPair.second->createShapeNodeWith<VisualAspect>(armShapeBox);

  Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
  armOffset.translation() = Eigen::Vector3d(0, -0.5, 0);
  armPair.first->setTransformFromChildBodyNode(armOffset);

  return cartpole;
}

SkeletonPtr createMultiarmRobot(
    int numLinks, double radiansPerJoint, int attachPoint = -1)
{
  SkeletonPtr arm = Skeleton::create("arm");
  BodyNode* parent = nullptr;

  std::shared_ptr<BoxShape> boxShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0)));

  std::vector<BodyNode*> nodes;

  for (std::size_t i = 0; i < numLinks; i++)
  {
    RevoluteJoint::Properties jointProps;
    jointProps.mName = "revolute_" + std::to_string(i);
    BodyNode::Properties bodyProps;
    bodyProps.mName = "arm_" + std::to_string(i);
    std::pair<RevoluteJoint*, BodyNode*> jointPair
        = arm->createJointAndBodyNodePair<RevoluteJoint>(
            parent, jointProps, bodyProps);
    nodes.push_back(jointPair.second);
    if (parent != nullptr)
    {
      Eigen::Isometry3d armOffset = Eigen::Isometry3d::Identity();
      armOffset.translation() = Eigen::Vector3d(0, 1.0, 0);
      jointPair.first->setTransformFromParentBodyNode(armOffset);
    }
    jointPair.second->setMass(1.0);
    parent = jointPair.second;
    if ((attachPoint == -1 && i < numLinks - 1) || i != attachPoint)
    {
      ShapeNode* visual = parent->createShapeNodeWith<VisualAspect>(boxShape);
    }
  }

  if (attachPoint != -1)
  {
    parent = nodes[attachPoint];
  }

  std::shared_ptr<BoxShape> endShape(
      new BoxShape(Eigen::Vector3d(1.0, 1.0, 1.0) * sqrt(1.0 / 3.0)));
  ShapeNode* endNode
      = parent->createShapeNodeWith<VisualAspect, CollisionAspect>(endShape);
  parent->setFrictionCoeff(1);
  return arm;
}

//==============================================================================
/// Creates an arbitrary three-link robot consisting of Single-DOF joints
SkeletonPtr createThreeLinkRobot(
    Vector3d dim1,
    TypeOfDOF type1,
    Vector3d dim2,
    TypeOfDOF type2,
    Vector3d dim3,
    TypeOfDOF type3,
    bool finished = false,
    bool collisionShape = true,
    size_t stopAfter = 3)
{
  SkeletonPtr robot = Skeleton::create();

  Vector3d dimEE = dim1;

  // Create the first link
  BodyNode::Properties node(BodyNode::AspectProperties("link1"));
  node.mInertia.setLocalCOM(Vector3d(0.0, 0.0, dim1(2) / 2.0));
  std::shared_ptr<Shape> shape(new BoxShape(dim1));

  std::pair<Joint*, BodyNode*> pair1 = add1DofJoint(
      robot,
      nullptr,
      node,
      "joint1",
      0.0,
      -constantsd::pi(),
      constantsd::pi(),
      type1);
  auto current_node = pair1.second;
  auto shapeNode = current_node->createShapeNodeWith<VisualAspect>(shape);
  if (collisionShape)
  {
    shapeNode->createCollisionAspect();
    shapeNode->createDynamicsAspect();
  }

  BodyNode* parent_node = current_node;

  if (stopAfter > 1)
  {
    // Create the second link
    node = BodyNode::Properties(BodyNode::AspectProperties("link2"));
    node.mInertia.setLocalCOM(Vector3d(0.0, 0.0, dim2(2) / 2.0));
    shape = std::shared_ptr<Shape>(new BoxShape(dim2));

    std::pair<Joint*, BodyNode*> pair2 = add1DofJoint(
        robot,
        parent_node,
        node,
        "joint2",
        0.0,
        -constantsd::pi(),
        constantsd::pi(),
        type2);
    Joint* joint = pair2.first;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translate(Eigen::Vector3d(0.0, 0.0, dim1(2)));
    joint->setTransformFromParentBodyNode(T);

    auto current_node = pair2.second;
    auto shapeNode = current_node->createShapeNodeWith<VisualAspect>(shape);
    if (collisionShape)
    {
      shapeNode->createCollisionAspect();
      shapeNode->createDynamicsAspect();
    }

    parent_node = pair2.second;
    dimEE = dim2;
  }

  if (stopAfter > 2)
  {
    // Create the third link
    node = BodyNode::Properties(BodyNode::AspectProperties("link3"));
    node.mInertia.setLocalCOM(Vector3d(0.0, 0.0, dim3(2) / 2.0));
    shape = std::shared_ptr<Shape>(new BoxShape(dim3));
    std::pair<Joint*, BodyNode*> pair3 = add1DofJoint(
        robot,
        parent_node,
        node,
        "joint3",
        0.0,
        -constantsd::pi(),
        constantsd::pi(),
        type3);

    Joint* joint = pair3.first;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translate(Eigen::Vector3d(0.0, 0.0, dim2(2)));
    joint->setTransformFromParentBodyNode(T);

    auto current_node = pair3.second;
    auto shapeNode = current_node->createShapeNodeWith<VisualAspect>(shape);
    if (collisionShape)
    {
      shapeNode->createCollisionAspect();
      shapeNode->createDynamicsAspect();
    }

    parent_node = pair3.second;
    dimEE = dim3;
  }

  // If finished, add an end effector
  if (finished)
    addEndEffector(robot, parent_node, dimEE);

  return robot;
}

//==============================================================================
/// Creates an arbitrary two-link robot consisting of Single-DOF joints
SkeletonPtr createTwoLinkRobot(
    Vector3d dim1,
    TypeOfDOF type1,
    Vector3d dim2,
    TypeOfDOF type2,
    bool finished = true)
{
  return createThreeLinkRobot(
      dim1,
      type1,
      dim2,
      type2,
      Eigen::Vector3d::Zero(),
      DOF_X,
      finished,
      true,
      2);
}

//==============================================================================
/// Creates a N link manipulator with the given dimensions where each joint is
/// the specified type
SkeletonPtr createNLinkRobot(
    int _n, Vector3d dim, TypeOfDOF type, bool finished = false)
{
  assert(_n > 0);

  SkeletonPtr robot = Skeleton::create();
  robot->disableSelfCollisionCheck();

  // Create the first link, the joint with the ground and its shape
  BodyNode::Properties node(BodyNode::AspectProperties("link1"));
  node.mInertia.setLocalCOM(Vector3d(0.0, 0.0, dim(2) / 2.0));
  std::shared_ptr<Shape> shape(new BoxShape(dim));

  std::pair<Joint*, BodyNode*> pair1 = add1DofJoint(
      robot,
      nullptr,
      node,
      "joint1",
      0.0,
      -constantsd::pi(),
      constantsd::pi(),
      type);

  Joint* joint = pair1.first;
  joint->setDampingCoefficient(0, 0.01);

  auto current_node = pair1.second;
  current_node
      ->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
          shape);

  BodyNode* parent_node = current_node;

  // Create links iteratively
  for (int i = 1; i < _n; ++i)
  {
    std::ostringstream ssLink;
    std::ostringstream ssJoint;
    ssLink << "link" << i;
    ssJoint << "joint" << i;

    node = BodyNode::Properties(BodyNode::AspectProperties(ssLink.str()));
    node.mInertia.setLocalCOM(Vector3d(0.0, 0.0, dim(2) / 2.0));
    shape = std::shared_ptr<Shape>(new BoxShape(dim));

    std::pair<Joint*, BodyNode*> newPair = add1DofJoint(
        robot,
        parent_node,
        node,
        ssJoint.str(),
        0.0,
        -constantsd::pi(),
        constantsd::pi(),
        type);

    Joint* joint = newPair.first;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translate(Eigen::Vector3d(0.0, 0.0, dim(2)));
    joint->setTransformFromParentBodyNode(T);
    joint->setDampingCoefficient(0, 0.01);

    auto current_node = newPair.second;
    current_node
        ->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
            shape);

    parent_node = current_node;
  }

  // If finished, initialize the skeleton
  if (finished)
    addEndEffector(robot, parent_node, dim);

  return robot;
}

//==============================================================================
/// Creates a N link pendulum with the given dimensions where each joint is
/// the specified type. The each offset from the joint position to the child
/// body is specified.
SkeletonPtr createNLinkPendulum(
    size_t numBodyNodes,
    const Vector3d& dim,
    TypeOfDOF type,
    const Vector3d& offset,
    bool finished = false)
{
  assert(numBodyNodes > 0);

  SkeletonPtr robot = Skeleton::create();
  robot->disableSelfCollisionCheck();

  // Create the first link, the joint with the ground and its shape
  BodyNode::Properties node(BodyNode::AspectProperties("link1"));
  node.mInertia.setLocalCOM(Vector3d(0.0, 0.0, dim(2) / 2.0));
  std::shared_ptr<Shape> shape(new BoxShape(dim));

  std::pair<Joint*, BodyNode*> pair1 = add1DofJoint(
      robot,
      nullptr,
      node,
      "joint1",
      0.0,
      -constantsd::pi(),
      constantsd::pi(),
      type);

  Joint* joint = pair1.first;
  Eigen::Isometry3d T = joint->getTransformFromChildBodyNode();
  T.translation() = offset;
  joint->setTransformFromChildBodyNode(T);
  joint->setDampingCoefficient(0, 0.01);

  auto current_node = pair1.second;
  current_node
      ->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
          shape);

  BodyNode* parent_node = current_node;

  // Create links iteratively
  for (size_t i = 1; i < numBodyNodes; ++i)
  {
    std::ostringstream ssLink;
    std::ostringstream ssJoint;
    ssLink << "link" << i;
    ssJoint << "joint" << i;

    node = BodyNode::Properties(BodyNode::AspectProperties(ssLink.str()));
    node.mInertia.setLocalCOM(Vector3d(0.0, 0.0, dim(2) / 2.0));
    shape = std::shared_ptr<Shape>(new BoxShape(dim));

    std::pair<Joint*, BodyNode*> newPair = add1DofJoint(
        robot,
        parent_node,
        node,
        ssJoint.str(),
        0.0,
        -constantsd::pi(),
        constantsd::pi(),
        type);

    Joint* joint = newPair.first;
    Eigen::Isometry3d T = joint->getTransformFromChildBodyNode();
    T.translation() = offset;
    joint->setTransformFromChildBodyNode(T);
    joint->setDampingCoefficient(0, 0.01);

    auto current_node = newPair.second;
    current_node
        ->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
            shape);

    parent_node = current_node;
  }

  // If finished, initialize the skeleton
  if (finished)
    addEndEffector(robot, parent_node, dim);

  return robot;
}

//==============================================================================
SkeletonPtr createGround(
    const Eigen::Vector3d& _size,
    const Eigen::Vector3d& _position = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& _orientation = Eigen::Vector3d::Zero())
{
  double mass = 1.0;

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = _position;
  T.linear() = eulerXYZToMatrix(_orientation);
  Joint::Properties joint("joint1", T);

  BodyNode::Properties node(BodyNode::AspectProperties(std::string("link")));
  std::shared_ptr<Shape> shape(new BoxShape(_size));
  node.mInertia.setMass(mass);

  SkeletonPtr skeleton = Skeleton::create();
  auto pair
      = skeleton->createJointAndBodyNodePair<WeldJoint>(nullptr, joint, node);

  auto body_node = pair.second;
  body_node->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
      shape);

  return skeleton;
}

//==============================================================================
SkeletonPtr createObject(
    const Eigen::Vector3d& _position = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& _orientation = Eigen::Vector3d::Zero())
{
  double mass = 1.0;

  GenericJoint<SE3Space>::Properties joint(std::string("joint1"));

  BodyNode::Properties node(BodyNode::AspectProperties(std::string("link1")));
  node.mInertia.setMass(mass);

  SkeletonPtr skeleton = Skeleton::create();
  skeleton->createJointAndBodyNodePair<FreeJoint>(nullptr, joint, node);

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = _position;
  T.linear() = eulerXYZToMatrix(_orientation);
  skeleton->getJoint(0)->setPositions(FreeJoint::convertToPositions(T));

  return skeleton;
}

//==============================================================================
SkeletonPtr createSphere(
    const double _radius,
    const Eigen::Vector3d& _position = Eigen::Vector3d::Zero())
{
  SkeletonPtr sphere = createObject(_position);

  BodyNode* bn = sphere->getBodyNode(0);
  std::shared_ptr<EllipsoidShape> ellipShape(
      new EllipsoidShape(Vector3d::Constant(_radius * 2.0)));
  bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
      ellipShape);

  return sphere;
}

//==============================================================================
SkeletonPtr createBox(
    const Eigen::Vector3d& _size,
    const Eigen::Vector3d& _position = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& _orientation = Eigen::Vector3d::Zero())
{
  SkeletonPtr box = createObject(_position, _orientation);

  BodyNode* bn = box->getBodyNode(0);
  std::shared_ptr<Shape> boxShape(new BoxShape(_size));
  bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(
      boxShape);

  return box;
}

//==============================================================================
struct TestResource : public dart::common::Resource
{
  size_t getSize() override
  {
    return 0;
  }

  size_t tell() override
  {
    return 0;
  }

  bool seek(ptrdiff_t /*_offset*/, SeekType /*_origin*/) override
  {
    return false;
  }

  size_t read(void* /*_buffer*/, size_t /*_size*/, size_t /*_count*/) override
  {
    return 0;
  }
};

//==============================================================================
struct PresentResourceRetriever : public dart::common::ResourceRetriever
{
  bool exists(const dart::common::Uri& _uri) override
  {
    mExists.push_back(_uri.toString());
    return true;
  }

  std::string getFilePath(const dart::common::Uri& _uri) override
  {
    mGetFilePath.push_back(_uri.toString());
    return _uri.toString();
  }

  dart::common::ResourcePtr retrieve(const dart::common::Uri& _uri) override
  {
    mRetrieve.push_back(_uri.toString());
    return std::make_shared<TestResource>();
  }

  std::vector<std::string> mExists;
  std::vector<std::string> mGetFilePath;
  std::vector<std::string> mRetrieve;
};

//==============================================================================
struct AbsentResourceRetriever : public dart::common::ResourceRetriever
{
  bool exists(const dart::common::Uri& _uri) override
  {
    mExists.push_back(_uri.toString());
    return false;
  }

  std::string getFilePath(const dart::common::Uri& _uri) override
  {
    mGetFilePath.push_back(_uri.toString());
    return "";
  }

  dart::common::ResourcePtr retrieve(const dart::common::Uri& _uri) override
  {
    mRetrieve.push_back(_uri.toString());
    return nullptr;
  }

  std::vector<std::string> mExists;
  std::vector<std::string> mGetFilePath;
  std::vector<std::string> mRetrieve;
};

aiScene* createBoxMeshUnsafe()
{
  aiScene* scene = (aiScene*)malloc(sizeof(aiScene));
  scene->mNumMeshes = 1;
  aiMesh* mesh = (aiMesh*)malloc(sizeof(aiMesh));
  aiMesh** arr = (aiMesh**)malloc(sizeof(void*) * scene->mNumMeshes);
  scene->mMeshes = arr;
  scene->mMeshes[0] = mesh;
  scene->mMaterials = nullptr;

  mesh->mNormals = nullptr;
  mesh->mNumUVComponents[0] = 0;

  // Create vertices

  mesh->mNumVertices = 8;
  mesh->mVertices = (aiVector3D*)malloc(sizeof(aiVector3D) * 10);

  // Bottom face

  mesh->mVertices[0].x = -.5;
  mesh->mVertices[0].y = -.5;
  mesh->mVertices[0].z = -.5;

  mesh->mVertices[1].x = .5;
  mesh->mVertices[1].y = -.5;
  mesh->mVertices[1].z = -.5;

  mesh->mVertices[2].x = .5;
  mesh->mVertices[2].y = .5;
  mesh->mVertices[2].z = -.5;

  mesh->mVertices[3].x = -.5;
  mesh->mVertices[3].y = .5;
  mesh->mVertices[3].z = -.5;

  // Top face

  mesh->mVertices[4].x = -.5;
  mesh->mVertices[4].y = -.5;
  mesh->mVertices[4].z = .5;

  mesh->mVertices[5].x = .5;
  mesh->mVertices[5].y = -.5;
  mesh->mVertices[5].z = .5;

  mesh->mVertices[6].x = .5;
  mesh->mVertices[6].y = .5;
  mesh->mVertices[6].z = .5;

  mesh->mVertices[7].x = -.5;
  mesh->mVertices[7].y = .5;
  mesh->mVertices[7].z = .5;

  // Create faces

  mesh->mNumFaces = 12;
  mesh->mFaces = (aiFace*)malloc(sizeof(aiFace) * mesh->mNumFaces);
  for (int i = 0; i < mesh->mNumFaces; i++)
  {
    mesh->mFaces[i].mIndices = (unsigned int*)malloc(sizeof(unsigned int) * 3);
    mesh->mFaces[i].mNumIndices = 3;
  }

  // Bottom face

  mesh->mFaces[0].mIndices[0] = 0;
  mesh->mFaces[0].mIndices[1] = 1;
  mesh->mFaces[0].mIndices[2] = 2;

  mesh->mFaces[1].mIndices[0] = 0;
  mesh->mFaces[1].mIndices[1] = 2;
  mesh->mFaces[1].mIndices[2] = 3;

  // Top face

  mesh->mFaces[2].mIndices[0] = 4;
  mesh->mFaces[2].mIndices[1] = 5;
  mesh->mFaces[2].mIndices[2] = 6;

  mesh->mFaces[3].mIndices[0] = 4;
  mesh->mFaces[3].mIndices[1] = 6;
  mesh->mFaces[3].mIndices[2] = 7;

  // Left face

  mesh->mFaces[4].mIndices[0] = 0;
  mesh->mFaces[4].mIndices[1] = 1;
  mesh->mFaces[4].mIndices[2] = 5;

  mesh->mFaces[5].mIndices[0] = 0;
  mesh->mFaces[5].mIndices[1] = 4;
  mesh->mFaces[5].mIndices[2] = 5;

  // Right face

  mesh->mFaces[6].mIndices[0] = 2;
  mesh->mFaces[6].mIndices[1] = 3;
  mesh->mFaces[6].mIndices[2] = 6;

  mesh->mFaces[7].mIndices[0] = 3;
  mesh->mFaces[7].mIndices[1] = 6;
  mesh->mFaces[7].mIndices[2] = 7;

  // Back face

  mesh->mFaces[8].mIndices[0] = 1;
  mesh->mFaces[8].mIndices[1] = 2;
  mesh->mFaces[8].mIndices[2] = 5;

  mesh->mFaces[9].mIndices[0] = 2;
  mesh->mFaces[9].mIndices[1] = 5;
  mesh->mFaces[9].mIndices[2] = 6;

  // Front face

  mesh->mFaces[10].mIndices[0] = 0;
  mesh->mFaces[10].mIndices[1] = 3;
  mesh->mFaces[10].mIndices[2] = 4;

  mesh->mFaces[11].mIndices[0] = 3;
  mesh->mFaces[11].mIndices[1] = 4;
  mesh->mFaces[11].mIndices[2] = 7;

  mesh->mNumFaces = 12;

  return scene;
}

#endif // #ifndef DART_UNITTESTS_TEST_HELPERS_H
