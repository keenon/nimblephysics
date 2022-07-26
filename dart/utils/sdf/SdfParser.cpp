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

#include "dart/utils/sdf/SdfParser.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <tinyxml2.h>

#include "dart/common/Console.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CustomJoint.hpp"
#include "dart/dynamics/CylinderShape.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ScrewJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SoftBodyNode.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/SkelParser.hpp"
#include "dart/utils/XmlHelpers.hpp"

namespace dart {
namespace utils {

namespace SdfParser {

namespace {

using BodyPropPtr = std::shared_ptr<dynamics::BodyNode::Properties>;

struct SDFBodyNode
{
  BodyPropPtr properties;
  Eigen::Isometry3s initTransform;
  std::string type;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using JointPropPtr = std::shared_ptr<dynamics::Joint::Properties>;

struct SDFJoint
{
  JointPropPtr properties;
  std::string parentName;
  std::string childName;
  std::string type;
};

// Maps the name of a BodyNode to its properties
using BodyMap = common::aligned_map<std::string, SDFBodyNode>;

// Maps a child BodyNode to the properties of its parent Joint
using JointMap = std::map<std::string, SDFJoint>;

simulation::WorldPtr readWorld(
    tinyxml2::XMLElement* worldElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

void readPhysics(
    tinyxml2::XMLElement* physicsElement, simulation::WorldPtr world);

dynamics::SkeletonPtr readSkeleton(
    tinyxml2::XMLElement* skeletonElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

bool createPair(
    dynamics::SkeletonPtr skeleton,
    dynamics::BodyNode* parent,
    const SDFJoint& newJoint,
    const SDFBodyNode& newBody);

enum NextResult
{
  VALID,
  CONTINUE,
  BREAK,
  CREATE_FREEJOINT_ROOT
};

NextResult getNextJointAndNodePair(
    BodyMap::iterator& body,
    JointMap::const_iterator& parentJoint,
    dynamics::BodyNode*& parentBody,
    const dynamics::SkeletonPtr skeleton,
    BodyMap& sdfBodyNodes,
    const JointMap& sdfJoints);

dynamics::SkeletonPtr makeSkeleton(
    tinyxml2::XMLElement* skeletonElement, Eigen::Isometry3s& skeletonFrame);

template <class NodeType>
std::pair<dynamics::Joint*, dynamics::BodyNode*> createJointAndNodePair(
    dynamics::SkeletonPtr skeleton,
    dynamics::BodyNode* parent,
    const SDFJoint& joint,
    const SDFBodyNode& node);

BodyMap readAllBodyNodes(
    tinyxml2::XMLElement* skeletonElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever,
    const Eigen::Isometry3s& skeletonFrame);

SDFBodyNode readBodyNode(
    tinyxml2::XMLElement* bodyNodeElement,
    const Eigen::Isometry3s& skeletonFrame,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

dynamics::SoftBodyNode::UniqueProperties readSoftBodyProperties(
    tinyxml2::XMLElement* softBodyNodeElement);

dynamics::ShapePtr readShape(
    tinyxml2::XMLElement* shapelement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

dynamics::ShapeNode* readShapeNode(
    dynamics::BodyNode* bodyNode,
    tinyxml2::XMLElement* shapeNodeEle,
    const std::string& shapeNodeName,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

void readMaterial(
    tinyxml2::XMLElement* materialEle, dynamics::ShapeNode* shapeNode);

void readVisualizationShapeNode(
    dynamics::BodyNode* bodyNode,
    tinyxml2::XMLElement* vizShapeNodeEle,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

void readCollisionShapeNode(
    dynamics::BodyNode* bodyNode,
    tinyxml2::XMLElement* collShapeNodeEle,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

void readAspects(
    const dynamics::SkeletonPtr& skeleton,
    tinyxml2::XMLElement* skeletonElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever);

JointMap readAllJoints(
    tinyxml2::XMLElement* skeletonElement,
    const Eigen::Isometry3s& skeletonFrame,
    const BodyMap& sdfBodyNodes);

SDFJoint readJoint(
    tinyxml2::XMLElement* jointElement,
    const BodyMap& bodies,
    const Eigen::Isometry3s& skeletonFrame);

dart::dynamics::WeldJoint::Properties readWeldJoint(
    tinyxml2::XMLElement* jointElement,
    const Eigen::Isometry3s& parentModelFrame,
    const std::string& name);

dynamics::RevoluteJoint::Properties readRevoluteJoint(
    tinyxml2::XMLElement* revoluteJointElement,
    const Eigen::Isometry3s& parentModelFrame,
    const std::string& name);

dynamics::PrismaticJoint::Properties readPrismaticJoint(
    tinyxml2::XMLElement* jointElement,
    const Eigen::Isometry3s& parentModelFrame,
    const std::string& name);

dynamics::ScrewJoint::Properties readScrewJoint(
    tinyxml2::XMLElement* jointElement,
    const Eigen::Isometry3s& parentModelFrame,
    const std::string& name);

dynamics::UniversalJoint::Properties readUniversalJoint(
    tinyxml2::XMLElement* jointElement,
    const Eigen::Isometry3s& parentModelFrame,
    const std::string& name);

dynamics::BallJoint::Properties readBallJoint(
    tinyxml2::XMLElement* jointElement,
    const Eigen::Isometry3s& parentModelFrame,
    const std::string& name);

common::ResourceRetrieverPtr getRetriever(
    const common::ResourceRetrieverPtr& retriever);

} // anonymous namespace

//==============================================================================
simulation::WorldPtr readSdfFile(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  return readWorld(uri, nullOrRetriever);
}

//==============================================================================
simulation::WorldPtr readWorld(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const auto retriever = getRetriever(nullOrRetriever);

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument sdfFile;
  try
  {
    openXMLFile(sdfFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    dtwarn << "[SdfParser::readSdfFile] Loading file [" << uri.toString()
           << "] failed: " << e.what() << "\n";
    return nullptr;
  }

  //--------------------------------------------------------------------------
  // Load DART
  tinyxml2::XMLElement* sdfElement = nullptr;
  sdfElement = sdfFile.FirstChildElement("sdf");
  if (sdfElement == nullptr)
    return nullptr;

  //--------------------------------------------------------------------------
  // version attribute
  std::string version = getAttributeString(sdfElement, "version");
  // TODO: We need version aware SDF parser (see #264)
  // We support 1.4 only for now.
  if (version != "1.4" && version != "1.5")
  {
    dtwarn << "[SdfParser::readSdfFile] The file format of [" << uri.toString()
           << "] was found to be [" << version << "], but we only support SDF "
           << "1.4 and 1.5!\n";
    return nullptr;
  }

  //--------------------------------------------------------------------------
  // Load World
  tinyxml2::XMLElement* worldElement = nullptr;
  worldElement = sdfElement->FirstChildElement("world");
  if (worldElement == nullptr)
    return nullptr;

  return readWorld(worldElement, uri, retriever);
}

//==============================================================================
dynamics::SkeletonPtr readSkeleton(
    const common::Uri& uri, const common::ResourceRetrieverPtr& nullOrRetriever)
{
  const auto retriever = getRetriever(nullOrRetriever);

  //--------------------------------------------------------------------------
  // Load xml and create Document
  tinyxml2::XMLDocument _dartFile;
  try
  {
    openXMLFile(_dartFile, uri, retriever);
  }
  catch (std::exception const& e)
  {
    dtwarn << "[SdfParser::readSkeleton] Loading file [" << uri.toString()
           << "] failed: " << e.what() << "\n";
    return nullptr;
  }

  //--------------------------------------------------------------------------
  // Load sdf
  tinyxml2::XMLElement* sdfElement = nullptr;
  sdfElement = _dartFile.FirstChildElement("sdf");
  if (sdfElement == nullptr)
    return nullptr;

  //--------------------------------------------------------------------------
  // version attribute
  std::string version = getAttributeString(sdfElement, "version");
  // TODO: We need version aware SDF parser (see #264)
  // We support 1.4 only for now.
  if (version != "1.4" && version != "1.5")
  {
    dtwarn << "[SdfParser::readSdfFile] The file format of [" << uri.toString()
           << "] was found to be [" << version
           << "], but we only support SDF 1.4 and 1.5!\n";
    return nullptr;
  }
  //--------------------------------------------------------------------------
  // Load skeleton
  tinyxml2::XMLElement* skelElement = nullptr;
  skelElement = sdfElement->FirstChildElement("model");
  if (skelElement == nullptr)
    return nullptr;

  dynamics::SkeletonPtr newSkeleton = readSkeleton(skelElement, uri, retriever);

  return newSkeleton;
}

//==============================================================================
std::string writeVec3(Eigen::Vector3s vec)
{
  return std::to_string(vec(0)) + " " + std::to_string(vec(1)) + " "
         + std::to_string(vec(2));
}

//==============================================================================
void appendShapeNodeToSDF(
    tinyxml2::XMLDocument* xmlDoc,
    tinyxml2::XMLElement* link,
    dynamics::BodyNode* body,
    dynamics::ShapeNode* shapeNode,
    Eigen::Isometry3s offsetT)
{
  using namespace tinyxml2;
  XMLElement* visual = nullptr;
  XMLElement* visualGeometry = nullptr;

  XMLElement* collision = nullptr;
  XMLElement* collisionGeometry = nullptr;

  dynamics::Shape* shape = shapeNode->getShape().get();

  if (shapeNode->hasVisualAspect())
  {
    visual = xmlDoc->NewElement("visual");
    visual->SetAttribute("name", (body->getName() + "_visual").c_str());
    link->InsertEndChild(visual);
    visualGeometry = xmlDoc->NewElement("geometry");
    visual->InsertEndChild(visualGeometry);
    XMLElement* visualPose = xmlDoc->NewElement("pose");
    visual->InsertEndChild(visualPose);
    visualPose->SetText(
        (writeVec3(offsetT * shapeNode->getRelativeTranslation()) + " "
         + writeVec3(math::matrixToEulerXYZ(
             offsetT * shapeNode->getRelativeRotation())))
            .c_str());
  }
  if (shapeNode->hasCollisionAspect())
  {
    collision = xmlDoc->NewElement("collision");
    collision->SetAttribute("name", (body->getName() + "_collision").c_str());
    link->InsertEndChild(collision);
    collisionGeometry = xmlDoc->NewElement("geometry");
    collision->InsertEndChild(collisionGeometry);
    XMLElement* collisionPose = xmlDoc->NewElement("pose");
    collision->InsertEndChild(collisionPose);
    collisionPose->SetText(
        (writeVec3(offsetT * shapeNode->getRelativeTranslation()) + " "
         + writeVec3(math::matrixToEulerXYZ(
             offsetT * shapeNode->getRelativeRotation())))
            .c_str());
  }

  // Create the object from scratch
  if (shape->getType() == "BoxShape")
  {
    dynamics::BoxShape* boxShape = dynamic_cast<dynamics::BoxShape*>(shape);

    if (shapeNode->hasVisualAspect())
    {
      XMLElement* visualBox = xmlDoc->NewElement("box");
      XMLElement* scale = xmlDoc->NewElement("scale");
      scale->SetText(writeVec3(boxShape->getSize()).c_str());
      visualBox->InsertEndChild(scale);
      visualGeometry->InsertEndChild(visualBox);
    }
    if (shapeNode->hasCollisionAspect())
    {
      XMLElement* collisionBox = xmlDoc->NewElement("box");
      XMLElement* scale = xmlDoc->NewElement("scale");
      scale->SetText(writeVec3(boxShape->getSize()).c_str());
      collisionBox->InsertEndChild(scale);
      collisionGeometry->InsertEndChild(collisionBox);
    }
  }
  else if (shape->getType() == "MeshShape")
  {
    dynamics::MeshShape* meshShape = dynamic_cast<dynamics::MeshShape*>(shape);
    std::string meshAbsolutePath = meshShape->getMeshPath();
    int geometryStart = meshAbsolutePath.find("Geometry");
    if (geometryStart != std::string::npos)
    {
      meshAbsolutePath = meshAbsolutePath.substr(geometryStart);
    }

    if (shapeNode->hasVisualAspect())
    {
      XMLElement* visualMesh = xmlDoc->NewElement("mesh");
      XMLElement* scale = xmlDoc->NewElement("scale");
      scale->SetText(writeVec3(meshShape->getScale()).c_str());
      visualMesh->InsertEndChild(scale);
      XMLElement* uri = xmlDoc->NewElement("uri");
      uri->SetText(meshAbsolutePath.c_str());
      visualMesh->InsertEndChild(uri);
      visualGeometry->InsertEndChild(visualMesh);
    }
    if (shapeNode->hasCollisionAspect())
    {
      XMLElement* collisionMesh = xmlDoc->NewElement("mesh");
      XMLElement* scale = xmlDoc->NewElement("scale");
      scale->SetText(writeVec3(meshShape->getScale()).c_str());
      collisionMesh->InsertEndChild(scale);
      XMLElement* uri = xmlDoc->NewElement("uri");
      uri->SetText(meshAbsolutePath.c_str());
      collisionMesh->InsertEndChild(uri);
      collisionGeometry->InsertEndChild(collisionMesh);
    }
  }
  else if (shape->getType() == "SphereShape")
  {
    dynamics::SphereShape* sphereShape
        = dynamic_cast<dynamics::SphereShape*>(shape);

    if (shapeNode->hasVisualAspect())
    {
      XMLElement* visualSphere = xmlDoc->NewElement("sphere");
      visualSphere->SetAttribute("radius", sphereShape->getRadius());
      XMLElement* radius = xmlDoc->NewElement("radius");
      radius->SetText(std::to_string(sphereShape->getRadius()).c_str());
      visualSphere->InsertEndChild(radius);
      visualGeometry->InsertEndChild(visualSphere);
    }
    if (shapeNode->hasCollisionAspect())
    {
      XMLElement* collisionSphere = xmlDoc->NewElement("sphere");
      XMLElement* radius = xmlDoc->NewElement("radius");
      radius->SetText(std::to_string(sphereShape->getRadius()).c_str());
      collisionSphere->InsertEndChild(radius);
      collisionGeometry->InsertEndChild(collisionSphere);
    }
  }
  else if (shape->getType() == "CapsuleShape")
  {
    // Ignore
  }
  else if (
      shape->getType() == "EllipsoidShape"
      && dynamic_cast<dynamics::EllipsoidShape*>(shape)->isSphere())
  {
    // Ignore
  }
  else
  {
    // Ignore
  }
}

//==============================================================================
void writeSkeleton(const std::string& path, dynamics::SkeletonPtr skel)
{
  using namespace tinyxml2;

  tinyxml2::XMLDocument xmlDoc;

  XMLElement* sdf = xmlDoc.NewElement("sdf");
  sdf->SetAttribute("version", "1.4");
  xmlDoc.InsertFirstChild(sdf);

  XMLElement* model = xmlDoc.NewElement("model");
  model->SetAttribute("name", skel->getName().c_str());
  sdf->InsertFirstChild(model);

  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    auto* body = skel->getBodyNode(i);

    XMLElement* link = xmlDoc.NewElement("link");
    link->SetAttribute("name", body->getName().c_str());

    XMLElement* linkPose = xmlDoc.NewElement("pose");
    linkPose->SetText((writeVec3(body->getWorldTransform().translation()) + " "
                       + writeVec3(math::matrixToEulerZYX(
                           body->getWorldTransform().linear())))
                          .c_str());
    link->InsertEndChild(linkPose);

    for (int j = 0; j < body->getNumShapeNodes(); j++)
    {
      dynamics::ShapeNode* shapeNode = body->getShapeNode(j);
      appendShapeNodeToSDF(
          &xmlDoc, link, body, shapeNode, Eigen::Isometry3s::Identity());
    }

    XMLElement* inertial = xmlDoc.NewElement("inertial");
    link->InsertEndChild(inertial);

    s_t totalMass = body->getMass();
    Eigen::Vector3s COM = body->getInertia().getLocalCOM();
    // dart::dynamics::Inertia inertia = body->getInertia();

    XMLElement* inertialMass = xmlDoc.NewElement("mass");
    inertialMass->SetText(std::to_string(totalMass).c_str());
    inertial->InsertEndChild(inertialMass);
    XMLElement* inertialPose = xmlDoc.NewElement("pose");
    inertialPose->SetText(
        (writeVec3(COM) + " " + writeVec3(Eigen::Vector3s::Zero())).c_str());
    inertial->InsertEndChild(inertialPose);

    XMLElement* inertialInertia = xmlDoc.NewElement("inertia");
    s_t i_xx = 0;
    s_t i_xy = 0;
    s_t i_xz = 0;
    s_t i_yy = 0;
    s_t i_yz = 0;
    s_t i_zz = 0;
    body->getMomentOfInertia(i_xx, i_yy, i_zz, i_xy, i_xz, i_yz);
    XMLElement* inertiaXX = xmlDoc.NewElement("ixx");
    inertiaXX->SetText(std::to_string(i_xx).c_str());
    inertialInertia->InsertEndChild(inertiaXX);

    XMLElement* inertiaXY = xmlDoc.NewElement("ixy");
    inertiaXY->SetText(std::to_string(i_xy).c_str());
    inertialInertia->InsertEndChild(inertiaXY);

    XMLElement* inertiaXZ = xmlDoc.NewElement("ixz");
    inertiaXZ->SetText(std::to_string(i_xz).c_str());
    inertialInertia->InsertEndChild(inertiaXZ);

    XMLElement* inertiaYY = xmlDoc.NewElement("iyy");
    inertiaYY->SetText(std::to_string(i_yy).c_str());
    inertialInertia->InsertEndChild(inertiaYY);

    XMLElement* inertiaYZ = xmlDoc.NewElement("iyz");
    inertiaYZ->SetText(std::to_string(i_yz).c_str());
    inertialInertia->InsertEndChild(inertiaYZ);

    XMLElement* inertiaZZ = xmlDoc.NewElement("izz");
    inertiaZZ->SetText(std::to_string(i_zz).c_str());
    inertialInertia->InsertEndChild(inertiaZZ);

    inertial->InsertEndChild(inertialInertia);

    model->InsertEndChild(link);
  }

  for (int i = 0; i < skel->getNumJoints(); i++)
  {
    auto* joint = skel->getJoint(i);
    if (joint->getParentBodyNode() == nullptr)
      continue;

    XMLElement* jointXml = xmlDoc.NewElement("joint");
    jointXml->SetAttribute("name", joint->getName().c_str());
    model->InsertEndChild(jointXml);

    /*
      childToJoint
      = getValueIsometry3sWithExtrinsicRotation(_jointElement, "pose");
    */
    XMLElement* jointPose = xmlDoc.NewElement("pose");
    Eigen::Isometry3s T = joint->getTransformFromChildBodyNode().inverse();
    jointPose->SetText((writeVec3(T.translation()) + " "
                        + writeVec3(math::matrixToEulerZYX(T.linear())))
                           .c_str());
    jointXml->InsertEndChild(jointPose);

    if (joint->getType() == dynamics::RevoluteJoint::getStaticType())
    {
      dynamics::RevoluteJoint* revolute
          = static_cast<dynamics::RevoluteJoint*>(joint);

      jointXml->SetAttribute("type", "revolute");
      XMLElement* axis = xmlDoc.NewElement("axis");
      XMLElement* xyz = xmlDoc.NewElement("xyz");
      xyz->SetText(writeVec3(revolute->getAxis()).c_str());
      axis->InsertEndChild(xyz);

      XMLElement* limit = xmlDoc.NewElement("limit");
      XMLElement* lowerLimit = xmlDoc.NewElement("lower");
      limit->InsertEndChild(lowerLimit);
      lowerLimit->SetText(
          std::to_string(revolute->getPositionLowerLimit(0)).c_str());
      XMLElement* upperLimit = xmlDoc.NewElement("upper");
      limit->InsertEndChild(upperLimit);
      upperLimit->SetText(
          std::to_string(revolute->getPositionUpperLimit(0)).c_str());
      axis->InsertEndChild(limit);

      jointXml->InsertEndChild(axis);
    }
    else if (joint->getType() == dynamics::UniversalJoint::getStaticType())
    {
      dynamics::UniversalJoint* universal
          = static_cast<dynamics::UniversalJoint*>(joint);

      jointXml->SetAttribute("type", "revolute2");
      XMLElement* axis1 = xmlDoc.NewElement("axis");
      XMLElement* xyz1 = xmlDoc.NewElement("xyz");
      xyz1->SetText(writeVec3(universal->getAxis1()).c_str());
      axis1->InsertEndChild(xyz1);
      jointXml->InsertEndChild(axis1);

      XMLElement* axis2 = xmlDoc.NewElement("axis2");
      XMLElement* xyz2 = xmlDoc.NewElement("xyz");
      xyz2->SetText(writeVec3(universal->getAxis2()).c_str());
      axis2->InsertEndChild(xyz2);
      jointXml->InsertEndChild(axis2);
    }
    else if (joint->getType() == dynamics::EulerJoint::getStaticType())
    {
      dynamics::EulerJoint* euler = static_cast<dynamics::EulerJoint*>(joint);
      (void)euler;
      jointXml->SetAttribute("type", "ball");
    }
    else if (joint->getType() == dynamics::BallJoint::getStaticType())
    {
      dynamics::BallJoint* ball = static_cast<dynamics::BallJoint*>(joint);
      (void)ball;
      jointXml->SetAttribute("type", "ball");
    }
    else if (joint->getType() == dynamics::WeldJoint::getStaticType())
    {
      dynamics::WeldJoint* weld = static_cast<dynamics::WeldJoint*>(joint);
      (void)weld;
      jointXml->SetAttribute("type", "fixed");
    }
    else if (joint->getType() == dynamics::CustomJoint<1>::getStaticType())
    {
      std::cout
          << "SDF Does not support <CustomJoint> types, so this joint will be "
             "ignored when writing. Use Skeleton::simplifySkeleton() first to "
             "approximate CustomJoints with simpler joints."
          << std::endl;
      continue;

      /*
      dynamics::CustomJoint<1>* custom
          = static_cast<dynamics::CustomJoint<1>*>(joint);
      (void)custom;
      jointXml->SetAttribute("type", "revolute");

      Eigen::Vector3s axisDir = Eigen::Vector3s::UnitX();

      // Figure out the axis properly
      s_t oldPos = custom->getPosition(0);
      custom->setPosition(0, 0);
      Eigen::Isometry3s T_zero = custom->getRelativeTransform();
      custom->setPosition(0, custom->getPositionLowerLimit(0));
      Eigen::Isometry3s T_lower = custom->getRelativeTransform();
      custom->setPosition(0, custom->getPositionUpperLimit(0));
      Eigen::Isometry3s T_upper = custom->getRelativeTransform();
      custom->setPosition(0, oldPos);
      Eigen::Isometry3s relative = (T_lower.inverse() * T_upper);
      axisDir = (T.inverse().linear() * math::logMap(relative.linear()))
                    .normalized();
      Eigen::Isometry3s T_upperLimit = (T_zero.inverse() * T_upper);
      Eigen::Vector3s upperLimitRot
          = (T.inverse().linear() * math::logMap(T_upperLimit.linear()));
      s_t upperLimitVal = axisDir.dot(upperLimitRot);
      Eigen::Isometry3s T_lowerLimit = (T_zero.inverse() * T_lower);
      Eigen::Vector3s lowerLimitRot
          = (T.inverse().linear() * math::logMap(T_lowerLimit.linear()));
      s_t lowerLimitVal = axisDir.dot(lowerLimitRot);

      XMLElement* axis = xmlDoc.NewElement("axis");
      XMLElement* xyz = xmlDoc.NewElement("xyz");
      xyz->SetText(writeVec3(axisDir).c_str());
      axis->InsertEndChild(xyz);

      XMLElement* limit = xmlDoc.NewElement("limit");
      XMLElement* lowerLimit = xmlDoc.NewElement("lower");
      limit->InsertEndChild(lowerLimit);
      lowerLimit->SetText(std::to_string(lowerLimitVal).c_str());
      XMLElement* upperLimit = xmlDoc.NewElement("lower");
      limit->InsertEndChild(upperLimit);
      upperLimit->SetText(std::to_string(upperLimitVal).c_str());
      axis->InsertEndChild(limit);

      jointXml->InsertEndChild(axis);
      */
    }
    else
    {
      std::cout << "Unsupported joint type! " << joint->getType()
                << " on joint " << joint->getName() << std::endl;
      std::cout << "Joint " << joint->getName() << " parent body is "
                << joint->getParentBodyNode()->getName() << std::endl;
      std::cout << "Defaulting to a fixed joint!" << std::endl;
      jointXml->SetAttribute("type", "fixed");
    }

    XMLElement* parentXml = xmlDoc.NewElement("parent");
    parentXml->SetText(joint->getParentBodyNode()->getName().c_str());
    jointXml->InsertEndChild(parentXml);

    XMLElement* childXml = xmlDoc.NewElement("child");
    childXml->SetText(joint->getChildBodyNode()->getName().c_str());
    jointXml->InsertEndChild(childXml);

    joint->getType();
  }

  std::cout << "Saving SDF file to " << path << std::endl;
  xmlDoc.SaveFile(path.c_str());
}

namespace {

//==============================================================================
simulation::WorldPtr readWorld(
    tinyxml2::XMLElement* worldElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever)
{
  assert(worldElement != nullptr);

  // Create a world
  simulation::WorldPtr newWorld = simulation::World::create();

  //--------------------------------------------------------------------------
  // Name attribute
  std::string name = getAttributeString(worldElement, "name");
  newWorld->setName(name);

  //--------------------------------------------------------------------------
  // Load physics
  if (hasElement(worldElement, "physics"))
  {
    tinyxml2::XMLElement* physicsElement
        = worldElement->FirstChildElement("physics");
    readPhysics(physicsElement, newWorld);
  }

  //--------------------------------------------------------------------------
  // Load skeletons
  ElementEnumerator skeletonElements(worldElement, "model");
  while (skeletonElements.next())
  {
    dynamics::SkeletonPtr newSkeleton
        = readSkeleton(skeletonElements.get(), baseUri, retriever);

    newWorld->addSkeleton(newSkeleton);
  }

  return newWorld;
}

//==============================================================================
void readPhysics(
    tinyxml2::XMLElement* physicsElement, simulation::WorldPtr world)
{
  // Type attribute
  // std::string physicsEngineName = getAttribute(_physicsElement, "type");

  // Time step
  if (hasElement(physicsElement, "max_step_size"))
  {
    s_t timeStep = getValueDouble(physicsElement, "max_step_size");
    world->setTimeStep(timeStep);
  }

  // Number of max contacts
  // if (hasElement(_physicsElement, "max_contacts"))
  // {
  //   int timeStep = getValueInt(_physicsElement, "max_contacts");
  //   _world->setMaxNumContacts(timeStep);
  // }

  // Gravity
  if (hasElement(physicsElement, "gravity"))
  {
    Eigen::Vector3s gravity = getValueVector3s(physicsElement, "gravity");
    world->setGravity(gravity);
  }
}

//==============================================================================
dynamics::SkeletonPtr readSkeleton(
    tinyxml2::XMLElement* skeletonElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever)
{
  assert(skeletonElement != nullptr);

  Eigen::Isometry3s skeletonFrame = Eigen::Isometry3s::Identity();
  dynamics::SkeletonPtr newSkeleton
      = makeSkeleton(skeletonElement, skeletonFrame);

  //--------------------------------------------------------------------------
  // Bodies
  BodyMap sdfBodyNodes
      = readAllBodyNodes(skeletonElement, baseUri, retriever, skeletonFrame);

  //--------------------------------------------------------------------------
  // Joints
  JointMap sdfJoints
      = readAllJoints(skeletonElement, skeletonFrame, sdfBodyNodes);

  // Iterate through the collected properties and construct the Skeleton from
  // the root nodes downward
  BodyMap::iterator body = sdfBodyNodes.begin();
  JointMap::const_iterator parentJoint;
  dynamics::BodyNode* parentBody{nullptr};
  while (body != sdfBodyNodes.end())
  {
    NextResult result = getNextJointAndNodePair(
        body, parentJoint, parentBody, newSkeleton, sdfBodyNodes, sdfJoints);

    if (BREAK == result)
      break;
    else if (CONTINUE == result)
      continue;
    else if (CREATE_FREEJOINT_ROOT == result)
    {
      // If a root FreeJoint is needed for the parent of the current joint, then
      // create it
      SDFJoint rootJoint;
      rootJoint.properties = dynamics::FreeJoint::Properties::createShared(
          dynamics::Joint::Properties("root", body->second.initTransform));
      rootJoint.type = "free";

      if (!createPair(newSkeleton, nullptr, rootJoint, body->second))
        break;

      sdfBodyNodes.erase(body);
      body = sdfBodyNodes.begin();

      continue;
    }

    if (!createPair(newSkeleton, parentBody, parentJoint->second, body->second))
      break;

    sdfBodyNodes.erase(body);
    body = sdfBodyNodes.begin();
  }

  // Read aspects here since aspects cannot be added if the BodyNodes haven't
  // created yet.
  readAspects(newSkeleton, skeletonElement, baseUri, retriever);

  // Set positions to their initial values
  newSkeleton->resetPositions();

  return newSkeleton;
}

//==============================================================================
bool createPair(
    dynamics::SkeletonPtr skeleton,
    dynamics::BodyNode* parent,
    const SDFJoint& newJoint,
    const SDFBodyNode& newBody)
{
  std::pair<dynamics::Joint*, dynamics::BodyNode*> pair;

  if (newBody.type.empty())
  {
    pair = createJointAndNodePair<dynamics::BodyNode>(
        skeleton, parent, newJoint, newBody);
  }
  else if (std::string("soft") == newBody.type)
  {
    pair = createJointAndNodePair<dynamics::SoftBodyNode>(
        skeleton, parent, newJoint, newBody);
  }
  else
  {
    dterr << "[SdfParser::createPair] Unsupported Link type: " << newBody.type
          << "\n";
    return false;
  }

  if (!pair.first || !pair.second)
    return false;

  return true;
}

//==============================================================================
NextResult getNextJointAndNodePair(
    BodyMap::iterator& body,
    JointMap::const_iterator& parentJoint,
    dynamics::BodyNode*& parentBody,
    const dynamics::SkeletonPtr skeleton,
    BodyMap& sdfBodyNodes,
    const JointMap& sdfJoints)
{
  parentJoint = sdfJoints.find(body->first);
  if (parentJoint == sdfJoints.end())
  {
    return CREATE_FREEJOINT_ROOT;
  }

  const std::string& parentBodyName = parentJoint->second.parentName;
  const std::string& parentJointName = parentJoint->second.properties->mName;

  // Check if the parent Body is created yet
  parentBody = skeleton->getBodyNode(parentBodyName);
  if (nullptr == parentBody && parentBodyName != "world"
      && !parentBodyName.empty())
  {
    // Find the properties of the parent Joint of the current Joint, because it
    // does not seem to be created yet.
    BodyMap::iterator check_parent_body = sdfBodyNodes.find(parentBodyName);

    if (check_parent_body == sdfBodyNodes.end())
    {
      // The Body does not exist in the file
      dterr << "[SdfParser::getNextJointAndNodePair] Could not find Link "
            << "named [" << parentBodyName << "] requested as parent of "
            << "Joint [" << parentJointName << "]. We will now quit "
            << "parsing.\n";
      return BREAK;
    }
    else
    {
      body = check_parent_body;
      return CONTINUE; // Create the parent before creating the current Joint
    }
  }

  return VALID;
}

dynamics::SkeletonPtr makeSkeleton(
    tinyxml2::XMLElement* _skeletonElement, Eigen::Isometry3s& skeletonFrame)
{
  assert(_skeletonElement != nullptr);

  dynamics::SkeletonPtr newSkeleton = dynamics::Skeleton::create();

  //--------------------------------------------------------------------------
  // Name attribute
  std::string name = getAttributeString(_skeletonElement, "name");
  newSkeleton->setName(name);

  //--------------------------------------------------------------------------
  // immobile attribute
  if (hasElement(_skeletonElement, "static"))
  {
    bool isStatic = getValueBool(_skeletonElement, "static");
    newSkeleton->setMobile(!isStatic);
  }

  //--------------------------------------------------------------------------
  // transformation
  if (hasElement(_skeletonElement, "pose"))
  {
    Eigen::Isometry3s W
        = getValueIsometry3sWithExtrinsicRotation(_skeletonElement, "pose");
    skeletonFrame = W;
  }

  return newSkeleton;
}

//==============================================================================
template <class NodeType>
std::pair<dynamics::Joint*, dynamics::BodyNode*> createJointAndNodePair(
    dynamics::SkeletonPtr skeleton,
    dynamics::BodyNode* parent,
    const SDFJoint& joint,
    const SDFBodyNode& node)
{
  const std::string& type = joint.type;

  if (std::string("prismatic") == type)
    return skeleton->createJointAndBodyNodePair<dynamics::PrismaticJoint>(
        parent,
        static_cast<const dynamics::PrismaticJoint::Properties&>(
            *joint.properties),
        static_cast<const typename NodeType::Properties&>(*node.properties));
  else if (std::string("revolute") == type)
    return skeleton->createJointAndBodyNodePair<dynamics::RevoluteJoint>(
        parent,
        static_cast<const dynamics::RevoluteJoint::Properties&>(
            *joint.properties),
        static_cast<const typename NodeType::Properties&>(*node.properties));
  else if (std::string("screw") == type)
    return skeleton->createJointAndBodyNodePair<dynamics::ScrewJoint>(
        parent,
        static_cast<const dynamics::ScrewJoint::Properties&>(*joint.properties),
        static_cast<const typename NodeType::Properties&>(*node.properties));
  else if (std::string("revolute2") == type)
    return skeleton->createJointAndBodyNodePair<dynamics::UniversalJoint>(
        parent,
        static_cast<const dynamics::UniversalJoint::Properties&>(
            *joint.properties),
        static_cast<const typename NodeType::Properties&>(*node.properties));
  else if (std::string("ball") == type)
    return skeleton->createJointAndBodyNodePair<dynamics::BallJoint>(
        parent,
        static_cast<const dynamics::BallJoint::Properties&>(*joint.properties),
        static_cast<const typename NodeType::Properties&>(*node.properties));
  else if (std::string("fixed") == type)
    return skeleton->createJointAndBodyNodePair<dynamics::WeldJoint>(
        parent,
        static_cast<const dynamics::WeldJoint::Properties&>(*joint.properties),
        static_cast<const typename NodeType::Properties&>(*node.properties));
  else if (std::string("free") == type)
    return skeleton->createJointAndBodyNodePair<dynamics::FreeJoint>(
        parent,
        static_cast<const dynamics::FreeJoint::Properties&>(*joint.properties),
        static_cast<const typename NodeType::Properties&>(*node.properties));

  dterr << "[SdfParser::createJointAndNodePair] Unsupported Joint type "
           "encountered: "
        << type << ". Please report this as a bug! We will now quit parsing.\n";
  return std::pair<dynamics::Joint*, dynamics::BodyNode*>(nullptr, nullptr);
}

//==============================================================================
BodyMap readAllBodyNodes(
    tinyxml2::XMLElement* skeletonElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever,
    const Eigen::Isometry3s& skeletonFrame)
{
  ElementEnumerator bodies(skeletonElement, "link");
  BodyMap sdfBodyNodes;
  while (bodies.next())
  {
    SDFBodyNode body
        = readBodyNode(bodies.get(), skeletonFrame, baseUri, retriever);

    BodyMap::iterator it = sdfBodyNodes.find(body.properties->mName);
    if (it != sdfBodyNodes.end())
    {
      dtwarn << "[SdfParser::readAllBodyNodes] Duplicate name in file: "
             << body.properties->mName << "\n"
             << "Every Link must have a unique name!\n";
      continue;
    }

    sdfBodyNodes[body.properties->mName] = body;
  }

  return sdfBodyNodes;
}

//===============================================================================
SDFBodyNode readBodyNode(
    tinyxml2::XMLElement* bodyNodeElement,
    const Eigen::Isometry3s& skeletonFrame,
    const common::Uri& /*baseUri*/,
    const common::ResourceRetrieverPtr& /*retriever*/)
{
  assert(bodyNodeElement != nullptr);

  dynamics::BodyNode::Properties properties;
  Eigen::Isometry3s initTransform = Eigen::Isometry3s::Identity();

  // Name attribute
  std::string name = getAttributeString(bodyNodeElement, "name");
  properties.mName = name;

  //--------------------------------------------------------------------------
  // gravity
  if (hasElement(bodyNodeElement, "gravity"))
  {
    bool gravityMode = getValueBool(bodyNodeElement, "gravity");
    properties.mGravityMode = gravityMode;
  }

  //--------------------------------------------------------------------------
  // self_collide
  //    if (hasElement(_bodyElement, "self_collide"))
  //    {
  //        bool gravityMode = getValueBool(_bodyElement, "self_collide");
  //    }

  //--------------------------------------------------------------------------
  // transformation
  if (hasElement(bodyNodeElement, "pose"))
  {
    Eigen::Isometry3s W
        = getValueIsometry3sWithExtrinsicRotation(bodyNodeElement, "pose");
    initTransform = skeletonFrame * W;
  }
  else
  {
    initTransform = skeletonFrame;
  }

  //--------------------------------------------------------------------------
  // inertia
  if (hasElement(bodyNodeElement, "inertial"))
  {
    tinyxml2::XMLElement* inertiaElement
        = getElement(bodyNodeElement, "inertial");

    // mass
    if (hasElement(inertiaElement, "mass"))
    {
      s_t mass = getValueDouble(inertiaElement, "mass");
      properties.mInertia.setMass(mass);
    }

    // offset
    if (hasElement(inertiaElement, "pose"))
    {
      Eigen::Isometry3s T
          = getValueIsometry3sWithExtrinsicRotation(inertiaElement, "pose");
      properties.mInertia.setLocalCOM(T.translation());
    }

    // inertia
    if (hasElement(inertiaElement, "inertia"))
    {
      tinyxml2::XMLElement* moiElement = getElement(inertiaElement, "inertia");

      s_t ixx = getValueDouble(moiElement, "ixx");
      s_t iyy = getValueDouble(moiElement, "iyy");
      s_t izz = getValueDouble(moiElement, "izz");

      s_t ixy = getValueDouble(moiElement, "ixy");
      s_t ixz = getValueDouble(moiElement, "ixz");
      s_t iyz = getValueDouble(moiElement, "iyz");

      properties.mInertia.setMoment(ixx, iyy, izz, ixy, ixz, iyz);
    }
  }

  SDFBodyNode sdfBodyNode;
  sdfBodyNode.initTransform = initTransform;
  if (hasElement(bodyNodeElement, "soft_shape"))
  {
    auto softProperties = readSoftBodyProperties(bodyNodeElement);

    sdfBodyNode.properties = dynamics::SoftBodyNode::Properties::createShared(
        properties, softProperties);
    sdfBodyNode.type = "soft";
  }
  else
  {
    sdfBodyNode.properties
        = dynamics::BodyNode::Properties::createShared(properties);
    sdfBodyNode.type = "";
  }

  return sdfBodyNode;
}

//==============================================================================
dynamics::SoftBodyNode::UniqueProperties readSoftBodyProperties(
    tinyxml2::XMLElement* softBodyNodeElement)
{
  //---------------------------------- Note ------------------------------------
  // SoftBodyNode is created if _softBodyNodeElement has <soft_shape>.
  // Otherwise, BodyNode is created.

  //----------------------------------------------------------------------------
  assert(softBodyNodeElement != nullptr);

  dynamics::SoftBodyNode::UniqueProperties softProperties;

  //----------------------------------------------------------------------------
  // Soft properties
  if (hasElement(softBodyNodeElement, "soft_shape"))
  {
    tinyxml2::XMLElement* softShapeEle
        = getElement(softBodyNodeElement, "soft_shape");

    // mass
    s_t totalMass = getValueDouble(softShapeEle, "total_mass");

    // pose
    Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
    if (hasElement(softShapeEle, "pose"))
      T = getValueIsometry3sWithExtrinsicRotation(softShapeEle, "pose");

    // geometry
    tinyxml2::XMLElement* geometryEle = getElement(softShapeEle, "geometry");
    if (hasElement(geometryEle, "sphere"))
    {
      tinyxml2::XMLElement* sphereEle = getElement(geometryEle, "sphere");
      const auto radius = getValueDouble(sphereEle, "radius");
      const auto nSlices = getValueUInt(sphereEle, "num_slices");
      const auto nStacks = getValueUInt(sphereEle, "num_stacks");
      softProperties = dynamics::SoftBodyNodeHelper::makeSphereProperties(
          radius, nSlices, nStacks, totalMass);
    }
    else if (hasElement(geometryEle, "box"))
    {
      tinyxml2::XMLElement* boxEle = getElement(geometryEle, "box");
      Eigen::Vector3s size = getValueVector3s(boxEle, "size");
      Eigen::Vector3i frags = getValueVector3i(boxEle, "frags");
      softProperties = dynamics::SoftBodyNodeHelper::makeBoxProperties(
          size, T, frags, totalMass);
    }
    else if (hasElement(geometryEle, "ellipsoid"))
    {
      tinyxml2::XMLElement* ellipsoidEle = getElement(geometryEle, "ellipsoid");
      Eigen::Vector3s size = getValueVector3s(ellipsoidEle, "size");
      const auto nSlices = getValueUInt(ellipsoidEle, "num_slices");
      const auto nStacks = getValueUInt(ellipsoidEle, "num_stacks");
      softProperties = dynamics::SoftBodyNodeHelper::makeEllipsoidProperties(
          size, nSlices, nStacks, totalMass);
    }
    else if (hasElement(geometryEle, "cylinder"))
    {
      tinyxml2::XMLElement* ellipsoidEle = getElement(geometryEle, "cylinder");
      s_t radius = getValueDouble(ellipsoidEle, "radius");
      s_t height = getValueDouble(ellipsoidEle, "height");
      s_t nSlices = getValueDouble(ellipsoidEle, "num_slices");
      s_t nStacks = getValueDouble(ellipsoidEle, "num_stacks");
      s_t nRings = getValueDouble(ellipsoidEle, "num_rings");
      softProperties = dynamics::SoftBodyNodeHelper::makeCylinderProperties(
          radius,
          height,
          static_cast<int>(nSlices),
          static_cast<int>(nStacks),
          static_cast<int>(nRings),
          totalMass);
    }
    else
    {
      dterr << "Unknown soft shape.\n";
    }

    // kv
    if (hasElement(softShapeEle, "kv"))
    {
      softProperties.mKv = getValueDouble(softShapeEle, "kv");
    }

    // ke
    if (hasElement(softShapeEle, "ke"))
    {
      softProperties.mKe = getValueDouble(softShapeEle, "ke");
    }

    // damp
    if (hasElement(softShapeEle, "damp"))
    {
      softProperties.mDampCoeff = getValueDouble(softShapeEle, "damp");
    }
  }

  return softProperties;
}

//==============================================================================
dynamics::ShapePtr readShape(
    tinyxml2::XMLElement* _shapelement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& _retriever)
{
  dynamics::ShapePtr newShape;

  // type
  assert(hasElement(_shapelement, "geometry"));
  tinyxml2::XMLElement* geometryElement = getElement(_shapelement, "geometry");

  if (hasElement(geometryElement, "sphere"))
  {
    tinyxml2::XMLElement* sphereElement = getElement(geometryElement, "sphere");

    const auto radius = getValueDouble(sphereElement, "radius");

    newShape = dynamics::ShapePtr(new dynamics::SphereShape(radius));
  }
  else if (hasElement(geometryElement, "box"))
  {
    tinyxml2::XMLElement* boxElement = getElement(geometryElement, "box");

    Eigen::Vector3s size = getValueVector3s(boxElement, "size");

    newShape = dynamics::ShapePtr(new dynamics::BoxShape(size));
  }
  else if (hasElement(geometryElement, "cylinder"))
  {
    tinyxml2::XMLElement* cylinderElement
        = getElement(geometryElement, "cylinder");

    s_t radius = getValueDouble(cylinderElement, "radius");
    s_t height = getValueDouble(cylinderElement, "length");

    newShape = dynamics::ShapePtr(new dynamics::CylinderShape(radius, height));
  }
  else if (hasElement(geometryElement, "plane"))
  {
    // TODO: Don't support plane shape yet.
    tinyxml2::XMLElement* planeElement = getElement(geometryElement, "plane");

    Eigen::Vector2s visSize = getValueVector2s(planeElement, "size");
    // TODO: Need to use normal for correct orientation of the plane
    // Eigen::Vector3s normal = getValueVector3s(planeElement, "normal");

    Eigen::Vector3s size(visSize(0), visSize(1), 0.001);

    newShape = dynamics::ShapePtr(new dynamics::BoxShape(size));
  }
  else if (hasElement(geometryElement, "mesh"))
  {
    tinyxml2::XMLElement* meshEle = getElement(geometryElement, "mesh");
    // TODO(JS): We assume that uri is just file name for the mesh
    if (!hasElement(meshEle, "uri"))
    {
      // TODO(MXG): Figure out how to report the file name and line number of
      dtwarn << "[SdfParser::readShape] Mesh is missing a URI, which is "
             << "required in order to load it\n";
      return nullptr;
    }
    std::string uri = getValueString(meshEle, "uri");

    Eigen::Vector3s scale = hasElement(meshEle, "scale")
                                ? getValueVector3s(meshEle, "scale")
                                : Eigen::Vector3s::Ones();

    const std::string meshUri = common::Uri::getRelativeUri(baseUri, uri);
    std::shared_ptr<dynamics::SharedMeshWrapper> model
        = dynamics::MeshShape::loadMesh(meshUri, _retriever);

    if (model)
      newShape = std::make_shared<dynamics::MeshShape>(
          scale, model, meshUri, _retriever);
    else
    {
      dtwarn << "[SdfParser::readShape] Failed to load mesh model [" << meshUri
             << "].\n";
      return nullptr;
    }
  }
  else
  {
    std::cout << "Invalid shape type." << std::endl;
    return nullptr;
  }

  return newShape;
}

//==============================================================================
dynamics::ShapeNode* readShapeNode(
    dynamics::BodyNode* bodyNode,
    tinyxml2::XMLElement* shapeNodeEle,
    const std::string& shapeNodeName,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever)
{
  assert(bodyNode);

  auto shape = readShape(shapeNodeEle, baseUri, retriever);
  auto shapeNode = bodyNode->createShapeNode(shape, shapeNodeName);

  // Transformation
  if (hasElement(shapeNodeEle, "pose"))
  {
    const Eigen::Isometry3s W
        = getValueIsometry3sWithExtrinsicRotation(shapeNodeEle, "pose");
    shapeNode->setRelativeTransform(W);
  }

  return shapeNode;
}

//==============================================================================
void readMaterial(
    tinyxml2::XMLElement* materialEle, dynamics::ShapeNode* shapeNode)
{

  auto visualAspect = shapeNode->getVisualAspect();
  if (hasElement(materialEle, "diffuse"))
  {
    Eigen::VectorXs color = getValueVectorXs(materialEle, "diffuse");
    if (color.size() == 3)
    {
      Eigen::Vector3s color3d = color;
      visualAspect->setColor(color3d);
    }
    else if (color.size() == 4)
    {
      Eigen::Vector4s color4d = color;
      visualAspect->setColor(color4d);
    }
    else
    {
      dterr << "[SdfParse::readMaterial] Unsupported color vector size: "
            << color.size() << "\n";
    }
  }
}

//==============================================================================
void readVisualizationShapeNode(
    dynamics::BodyNode* bodyNode,
    tinyxml2::XMLElement* vizShapeNodeEle,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever)
{
  dynamics::ShapeNode* newShapeNode = readShapeNode(
      bodyNode,
      vizShapeNodeEle,
      bodyNode->getName() + " - visual shape",
      baseUri,
      retriever);

  newShapeNode->createVisualAspect();

  // Material
  if (hasElement(vizShapeNodeEle, "material"))
  {
    tinyxml2::XMLElement* materialEle = getElement(vizShapeNodeEle, "material");
    readMaterial(materialEle, newShapeNode);
  }
}

//==============================================================================
void readCollisionShapeNode(
    dynamics::BodyNode* bodyNode,
    tinyxml2::XMLElement* collShapeNodeEle,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever)
{
  dynamics::ShapeNode* newShapeNode = readShapeNode(
      bodyNode,
      collShapeNodeEle,
      bodyNode->getName() + " - collision shape",
      baseUri,
      retriever);

  newShapeNode->createCollisionAspect();
}

//==============================================================================
void readAspects(
    const dynamics::SkeletonPtr& skeleton,
    tinyxml2::XMLElement* skeletonElement,
    const common::Uri& baseUri,
    const common::ResourceRetrieverPtr& retriever)
{
  ElementEnumerator xmlBodies(skeletonElement, "link");
  while (xmlBodies.next())
  {
    auto bodyElement = xmlBodies.get();
    auto bodyNodeName = getAttributeString(bodyElement, "name");
    auto bodyNode = skeleton->getBodyNode(bodyNodeName);

    // visualization_shape
    ElementEnumerator vizShapes(bodyElement, "visual");
    while (vizShapes.next())
    {
      readVisualizationShapeNode(bodyNode, vizShapes.get(), baseUri, retriever);
    }

    // collision_shape
    ElementEnumerator collShapes(bodyElement, "collision");
    while (collShapes.next())
      readCollisionShapeNode(bodyNode, collShapes.get(), baseUri, retriever);
  }
}

//==============================================================================
JointMap readAllJoints(
    tinyxml2::XMLElement* _skeletonElement,
    const Eigen::Isometry3s& skeletonFrame,
    const BodyMap& sdfBodyNodes)
{
  JointMap sdfJoints;
  ElementEnumerator joints(_skeletonElement, "joint");
  while (joints.next())
  {
    SDFJoint joint = readJoint(joints.get(), sdfBodyNodes, skeletonFrame);

    if (joint.childName.empty())
    {
      dterr << "[SdfParser::readAllJoints] Joint named ["
            << joint.properties->mName << "] does not have a valid child "
            << "Link, so it will not be added to the Skeleton\n";
      continue;
    }

    JointMap::iterator it = sdfJoints.find(joint.childName);
    if (it != sdfJoints.end())
    {
      dterr << "[SdfParser::readAllJoints] Joint named ["
            << joint.properties->mName << "] is claiming Link ["
            << joint.childName << "] as its child, but that is already "
            << "claimed by Joint [" << it->second.properties->mName
            << "]. Joint [" << joint.properties->mName
            << "] will be discarded\n";
      continue;
    }

    sdfJoints[joint.childName] = joint;
  }

  return sdfJoints;
}

SDFJoint readJoint(
    tinyxml2::XMLElement* _jointElement,
    const BodyMap& _sdfBodyNodes,
    const Eigen::Isometry3s& _skeletonFrame)
{
  assert(_jointElement != nullptr);

  //--------------------------------------------------------------------------
  // Type attribute
  std::string type = getAttributeString(_jointElement, "type");
  assert(!type.empty());

  //--------------------------------------------------------------------------
  // Name attribute
  std::string name = getAttributeString(_jointElement, "name");

  //--------------------------------------------------------------------------
  // parent
  BodyMap::const_iterator parent_it = _sdfBodyNodes.end();

  if (hasElement(_jointElement, "parent"))
  {
    std::string strParent = getValueString(_jointElement, "parent");

    if (strParent != std::string("world"))
    {
      parent_it = _sdfBodyNodes.find(strParent);

      if (parent_it == _sdfBodyNodes.end())
      {
        dterr << "[SdfParser::readJoint] Cannot find a Link named ["
              << strParent << "] requested as the parent of the Joint named ["
              << name << "]\n";
        assert(0);
      }
    }
  }
  else
  {
    dterr << "[SdfParser::readJoint] You must set parent link for "
          << "the Joint [" << name << "]!\n";
    assert(0);
  }

  //--------------------------------------------------------------------------
  // child
  BodyMap::const_iterator child_it = _sdfBodyNodes.end();

  if (hasElement(_jointElement, "child"))
  {
    std::string strChild = getValueString(_jointElement, "child");

    child_it = _sdfBodyNodes.find(strChild);

    if (child_it == _sdfBodyNodes.end())
    {
      dterr << "[SdfParser::readJoint] Cannot find a Link named [" << strChild
            << "] requested as the child of the Joint named [" << name << "]\n";
      assert(0);
    }
  }
  else
  {
    dterr << "[SdfParser::readJoint] You must set the child link for the Joint "
          << "[" << name << "]!\n";
    assert(0);
  }

  SDFJoint newJoint;
  newJoint.parentName
      = (parent_it == _sdfBodyNodes.end()) ? "" : parent_it->first;
  newJoint.childName = (child_it == _sdfBodyNodes.end()) ? "" : child_it->first;

  //--------------------------------------------------------------------------
  // transformation
  Eigen::Isometry3s parentWorld = Eigen::Isometry3s::Identity();
  Eigen::Isometry3s childToJoint = Eigen::Isometry3s::Identity();
  Eigen::Isometry3s childWorld = Eigen::Isometry3s::Identity();

  if (parent_it != _sdfBodyNodes.end())
    parentWorld = parent_it->second.initTransform;
  if (child_it != _sdfBodyNodes.end())
    childWorld = child_it->second.initTransform;
  if (hasElement(_jointElement, "pose"))
    childToJoint
        = getValueIsometry3sWithExtrinsicRotation(_jointElement, "pose");

  Eigen::Isometry3s parentToJoint
      = parentWorld.inverse() * childWorld * childToJoint;

  // TODO: Workaround!!
  Eigen::Isometry3s parentModelFrame
      = (childWorld * childToJoint).inverse() * _skeletonFrame;

  if (type == std::string("fixed"))
    newJoint.properties = dynamics::WeldJoint::Properties::createShared(
        readWeldJoint(_jointElement, parentModelFrame, name));
  if (type == std::string("prismatic"))
    newJoint.properties = dynamics::PrismaticJoint::Properties::createShared(
        readPrismaticJoint(_jointElement, parentModelFrame, name));
  if (type == std::string("revolute"))
    newJoint.properties = dynamics::RevoluteJoint::Properties::createShared(
        readRevoluteJoint(_jointElement, parentModelFrame, name));
  if (type == std::string("screw"))
    newJoint.properties = dynamics::ScrewJoint::Properties::createShared(
        readScrewJoint(_jointElement, parentModelFrame, name));
  if (type == std::string("revolute2"))
    newJoint.properties = dynamics::UniversalJoint::Properties::createShared(
        readUniversalJoint(_jointElement, parentModelFrame, name));
  if (type == std::string("ball"))
    newJoint.properties = dynamics::BallJoint::Properties::createShared(
        readBallJoint(_jointElement, parentModelFrame, name));

  newJoint.type = type;

  newJoint.properties->mName = name;

  newJoint.properties->mT_ChildBodyToJoint = childToJoint;
  newJoint.properties->mT_ParentBodyToJoint = parentToJoint;

  return newJoint;
}

static void reportMissingElement(
    const std::string& functionName,
    const std::string& elementName,
    const std::string& objectType,
    const std::string& objectName)
{
  dterr << "[SdfParser::" << functionName << "] Missing element " << elementName
        << " for " << objectType << " named " << objectName << "\n";
  assert(0);
}

static void readAxisElement(
    tinyxml2::XMLElement* axisElement,
    const Eigen::Isometry3s& _parentModelFrame,
    Eigen::Vector3s& axis,
    s_t& lower,
    s_t& upper,
    s_t& initial,
    s_t& rest,
    s_t& damping)
{
  // use_parent_model_frame
  bool useParentModelFrame = false;
  if (hasElement(axisElement, "use_parent_model_frame"))
    useParentModelFrame = getValueBool(axisElement, "use_parent_model_frame");

  // xyz
  Eigen::Vector3s xyz = getValueVector3s(axisElement, "xyz");
  if (useParentModelFrame)
  {
    xyz = _parentModelFrame.rotation() * xyz;
  }
  axis = xyz;

  // dynamics
  if (hasElement(axisElement, "dynamics"))
  {
    tinyxml2::XMLElement* dynamicsElement = getElement(axisElement, "dynamics");

    // damping
    if (hasElement(dynamicsElement, "damping"))
    {
      damping = getValueDouble(dynamicsElement, "damping");
    }
  }

  // limit
  if (hasElement(axisElement, "limit"))
  {
    tinyxml2::XMLElement* limitElement = getElement(axisElement, "limit");

    // lower
    if (hasElement(limitElement, "lower"))
    {
      lower = getValueDouble(limitElement, "lower");
    }

    // upper
    if (hasElement(limitElement, "upper"))
    {
      upper = getValueDouble(limitElement, "upper");
    }
  }

  // If the zero position is out of our limits, we should change the initial
  // position instead of assuming zero
  if (0.0 < lower || upper < 0.0)
  {
    if (isfinite(lower) && isfinite(upper))
      initial = (lower + upper) / 2.0;
    else if (isfinite(lower))
      initial = lower;
    else if (isfinite(upper))
      initial = upper;

    // Any other case means the limits are both +inf, both -inf, or one is a NaN

    // Apply the same logic to the rest position.
    rest = initial;
  }
}

dart::dynamics::WeldJoint::Properties readWeldJoint(
    tinyxml2::XMLElement* /*_jointElement*/,
    const Eigen::Isometry3s&,
    const std::string&)
{
  return dynamics::WeldJoint::Properties();
}

dynamics::RevoluteJoint::Properties readRevoluteJoint(
    tinyxml2::XMLElement* _revoluteJointElement,
    const Eigen::Isometry3s& _parentModelFrame,
    const std::string& _name)
{
  assert(_revoluteJointElement != nullptr);

  dynamics::RevoluteJoint::Properties newRevoluteJoint;

  //--------------------------------------------------------------------------
  // axis
  if (hasElement(_revoluteJointElement, "axis"))
  {
    tinyxml2::XMLElement* axisElement
        = getElement(_revoluteJointElement, "axis");

    readAxisElement(
        axisElement,
        _parentModelFrame,
        newRevoluteJoint.mAxis,
        newRevoluteJoint.mPositionLowerLimits[0],
        newRevoluteJoint.mPositionUpperLimits[0],
        newRevoluteJoint.mInitialPositions[0],
        newRevoluteJoint.mRestPositions[0],
        newRevoluteJoint.mDampingCoefficients[0]);
  }
  else
  {
    reportMissingElement("readRevoluteJoint", "axis", "joint", _name);
  }

  return newRevoluteJoint;
}

dynamics::PrismaticJoint::Properties readPrismaticJoint(
    tinyxml2::XMLElement* _jointElement,
    const Eigen::Isometry3s& _parentModelFrame,
    const std::string& _name)
{
  assert(_jointElement != nullptr);

  dynamics::PrismaticJoint::Properties newPrismaticJoint;

  //--------------------------------------------------------------------------
  // axis
  if (hasElement(_jointElement, "axis"))
  {
    tinyxml2::XMLElement* axisElement = getElement(_jointElement, "axis");

    readAxisElement(
        axisElement,
        _parentModelFrame,
        newPrismaticJoint.mAxis,
        newPrismaticJoint.mPositionLowerLimits[0],
        newPrismaticJoint.mPositionUpperLimits[0],
        newPrismaticJoint.mInitialPositions[0],
        newPrismaticJoint.mRestPositions[0],
        newPrismaticJoint.mDampingCoefficients[0]);
  }
  else
  {
    reportMissingElement("readPrismaticJoint", "axis", "joint", _name);
  }

  return newPrismaticJoint;
}

dynamics::ScrewJoint::Properties readScrewJoint(
    tinyxml2::XMLElement* _jointElement,
    const Eigen::Isometry3s& _parentModelFrame,
    const std::string& _name)
{
  assert(_jointElement != nullptr);

  dynamics::ScrewJoint::Properties newScrewJoint;

  //--------------------------------------------------------------------------
  // axis
  if (hasElement(_jointElement, "axis"))
  {
    tinyxml2::XMLElement* axisElement = getElement(_jointElement, "axis");

    readAxisElement(
        axisElement,
        _parentModelFrame,
        newScrewJoint.mAxis,
        newScrewJoint.mPositionLowerLimits[0],
        newScrewJoint.mPositionUpperLimits[0],
        newScrewJoint.mInitialPositions[0],
        newScrewJoint.mRestPositions[0],
        newScrewJoint.mDampingCoefficients[0]);
  }
  else
  {
    reportMissingElement("readScrewJoint", "axis", "joint", _name);
  }

  // pitch
  if (hasElement(_jointElement, "thread_pitch"))
  {
    s_t pitch = getValueDouble(_jointElement, "thread_pitch");
    newScrewJoint.mPitch = pitch;
  }

  return newScrewJoint;
}

dynamics::UniversalJoint::Properties readUniversalJoint(
    tinyxml2::XMLElement* _jointElement,
    const Eigen::Isometry3s& _parentModelFrame,
    const std::string& _name)
{
  assert(_jointElement != nullptr);

  dynamics::UniversalJoint::Properties newUniversalJoint;

  //--------------------------------------------------------------------------
  // axis
  if (hasElement(_jointElement, "axis"))
  {
    tinyxml2::XMLElement* axisElement = getElement(_jointElement, "axis");

    readAxisElement(
        axisElement,
        _parentModelFrame,
        newUniversalJoint.mAxis[0],
        newUniversalJoint.mPositionLowerLimits[0],
        newUniversalJoint.mPositionUpperLimits[0],
        newUniversalJoint.mInitialPositions[0],
        newUniversalJoint.mRestPositions[0],
        newUniversalJoint.mDampingCoefficients[0]);
  }
  else
  {
    reportMissingElement("readUniversalJoint", "axis", "joint", _name);
  }

  //--------------------------------------------------------------------------
  // axis2
  if (hasElement(_jointElement, "axis2"))
  {
    tinyxml2::XMLElement* axis2Element = getElement(_jointElement, "axis2");

    readAxisElement(
        axis2Element,
        _parentModelFrame,
        newUniversalJoint.mAxis[1],
        newUniversalJoint.mPositionLowerLimits[1],
        newUniversalJoint.mPositionUpperLimits[1],
        newUniversalJoint.mInitialPositions[1],
        newUniversalJoint.mRestPositions[1],
        newUniversalJoint.mDampingCoefficients[1]);
  }
  else
  {
    reportMissingElement("readUniversalJoint", "axis2", "joint", _name);
  }

  return newUniversalJoint;
}

dynamics::BallJoint::Properties readBallJoint(
    tinyxml2::XMLElement* /*_jointElement*/,
    const Eigen::Isometry3s&,
    const std::string&)
{
  return dynamics::BallJoint::Properties();
}

//==============================================================================
common::ResourceRetrieverPtr getRetriever(
    const common::ResourceRetrieverPtr& retriever)
{
  if (retriever)
  {
    return retriever;
  }
  else
  {
    auto newRetriever = std::make_shared<utils::CompositeResourceRetriever>();
    newRetriever->addSchemaRetriever(
        "file", std::make_shared<common::LocalResourceRetriever>());
    newRetriever->addSchemaRetriever("dart", DartResourceRetriever::create());

    return newRetriever;
  }
}

} // anonymous namespace

} // namespace SdfParser

} // namespace utils
} // namespace dart
