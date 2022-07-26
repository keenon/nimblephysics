#include "dart/utils/MJCFExporter.hpp"

#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <tinyxml2.h>

#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CapsuleShape.hpp"
#include "dart/dynamics/EllipsoidShape.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/simulation/Recording.hpp"

namespace dart {
namespace utils {

//==============================================================================
std::string writeVec3(Eigen::Vector3s vec)
{
  return std::to_string(vec(0)) + " " + std::to_string(vec(1)) + " "
         + std::to_string(vec(2));
}

//==============================================================================
void recursivelyWriteJointAndBody(
    tinyxml2::XMLDocument& xmlDoc,
    tinyxml2::XMLElement* assetXml,
    tinyxml2::XMLElement* actuatorsXml,
    tinyxml2::XMLElement* elem,
    dynamics::Joint* joint,
    bool isRoot)
{
  using namespace tinyxml2;
  (void)elem;
  (void)joint;

  dynamics::BodyNode* body = joint->getChildBodyNode();

  XMLElement* bodyXml = xmlDoc.NewElement("body");
  bodyXml->SetAttribute("name", body->getName().c_str());
  bodyXml->SetAttribute(
      "pos", writeVec3(joint->getRelativeTransform().translation()).c_str());
  Eigen::Matrix3s R = Eigen::Matrix3s::Identity();
  if (isRoot)
  {
    R = math::eulerXYZToMatrix(Eigen::Vector3s::UnitX() * M_PI / 2);
  }
  bodyXml->SetAttribute(
      "euler",
      writeVec3(
          math::matrixToEulerXYZ(R * joint->getRelativeTransform().linear()))
          .c_str());

  XMLElement* inertiaXml = xmlDoc.NewElement("inertial");
  inertiaXml->SetAttribute("pos", writeVec3(body->getCOM()).c_str());
  inertiaXml->SetAttribute("mass", std::to_string(body->getMass()).c_str());
  // dart::dynamics::Inertia inertia = body->getInertia();
  s_t i_xx = 0;
  s_t i_xy = 0;
  s_t i_xz = 0;
  s_t i_yy = 0;
  s_t i_yz = 0;
  s_t i_zz = 0;
  body->getMomentOfInertia(i_xx, i_yy, i_zz, i_xy, i_xz, i_yz);
  // M(1,1), M(2,2), M(3,3), M(1,2), M(1,3), M(2,3)
  inertiaXml->SetAttribute(
      "fullinertia",
      (std::to_string(i_xx) + " " + std::to_string(i_yy) + " "
       + std::to_string(i_zz) + " " + std::to_string(i_xy) + " "
       + std::to_string(i_xz) + " " + std::to_string(i_yz))
          .c_str());
  bodyXml->InsertEndChild(inertiaXml);

  if (isRoot)
  {
    /*
    <camera name="side" pos="0 -4 2" xyaxes="1 0 0 0 1 2" mode="trackcom"/>
    <camera name="back" pos="-4 0 2" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>
    */
    XMLElement* sideCamera = xmlDoc.NewElement("camera");
    sideCamera->SetAttribute("name", "side");
    sideCamera->SetAttribute("pos", "0 -.1 4.7");
    sideCamera->SetAttribute("euler", "0 0 0");
    sideCamera->SetAttribute("mode", "trackcom");
    bodyXml->InsertEndChild(sideCamera);

    XMLElement* backCamera = xmlDoc.NewElement("camera");
    backCamera->SetAttribute("name", "back");
    backCamera->SetAttribute("pos", "-4.7 -.1 0");
    backCamera->SetAttribute("euler", "0 -1.570796325 0");
    backCamera->SetAttribute("mode", "trackcom");
    bodyXml->InsertEndChild(backCamera);
  }

  if (joint->getType() == dynamics::RevoluteJoint::getStaticType())
  {
    dynamics::RevoluteJoint* revolute
        = static_cast<dynamics::RevoluteJoint*>(joint);
    (void)revolute;

    XMLElement* motor = xmlDoc.NewElement("motor");
    actuatorsXml->InsertEndChild(motor);
    motor->SetAttribute("gear", "100");
    motor->SetAttribute("joint", joint->getDofName(0).c_str());
    motor->SetAttribute("name", joint->getDofName(0).c_str());

    XMLElement* jointXml = xmlDoc.NewElement("joint");
    bodyXml->InsertEndChild(jointXml);
    jointXml->SetAttribute(
        "pos",
        writeVec3(joint->getTransformFromChildBodyNode().translation())
            .c_str());
    jointXml->SetAttribute("name", joint->getDofName(0).c_str());
    jointXml->SetAttribute("type", "hinge");
    jointXml->SetAttribute(
        "axis",
        writeVec3(
            joint->getTransformFromChildBodyNode().linear()
            * revolute->getAxis())
            .c_str());
    if (revolute->getPositionUpperLimit(0) > revolute->getPositionLowerLimit(0))
    {
      jointXml->SetAttribute("limited", "true");
      jointXml->SetAttribute(
          "range",
          (std::to_string(revolute->getPositionLowerLimit(0)) + " "
           + std::to_string(revolute->getPositionUpperLimit(0)))
              .c_str());
    }
    else
    {
      std::cout << "Joint " << revolute->getName()
                << " had backwards joints limits: "
                << revolute->getPositionLowerLimit(0) << " "
                << revolute->getPositionUpperLimit(0) << std::endl;
      jointXml->SetAttribute(
          "range",
          (std::to_string(revolute->getPositionUpperLimit(0)) + " "
           + std::to_string(revolute->getPositionLowerLimit(0)))
              .c_str());
    }
  }
  else if (joint->getType() == dynamics::UniversalJoint::getStaticType())
  {
    dynamics::UniversalJoint* universal
        = static_cast<dynamics::UniversalJoint*>(joint);
    (void)universal;

    for (int i = 0; i < 1; i++)
    {
      XMLElement* motor = xmlDoc.NewElement("motor");
      actuatorsXml->InsertEndChild(motor);
      motor->SetAttribute("gear", "100");
      motor->SetAttribute("joint", joint->getDofName(i).c_str());
      motor->SetAttribute("name", joint->getDofName(i).c_str());

      XMLElement* jointXml = xmlDoc.NewElement("joint");
      bodyXml->InsertEndChild(jointXml);
      jointXml->SetAttribute(
          "pos",
          writeVec3(joint->getTransformFromChildBodyNode().translation())
              .c_str());
      jointXml->SetAttribute("name", joint->getDofName(0).c_str());
      jointXml->SetAttribute("type", "hinge");
      jointXml->SetAttribute(
          "axis",
          writeVec3(
              joint->getTransformFromChildBodyNode().linear()
              * (i == 0 ? universal->getAxis1() : universal->getAxis2()))
              .c_str());
      if (universal->getPositionUpperLimit(i)
          > universal->getPositionLowerLimit(i))
      {
        jointXml->SetAttribute("limited", "true");
        jointXml->SetAttribute(
            "range",
            (std::to_string(universal->getPositionLowerLimit(i)) + " "
             + std::to_string(universal->getPositionUpperLimit(i)))
                .c_str());
      }
    }
  }
  else if (joint->getType() == dynamics::BallJoint::getStaticType())
  {
    dynamics::BallJoint* ball = static_cast<dynamics::BallJoint*>(joint);
    (void)ball;
    XMLElement* jointXml = xmlDoc.NewElement("joint");
    bodyXml->InsertEndChild(jointXml);
    jointXml->SetAttribute(
        "pos",
        writeVec3(joint->getTransformFromChildBodyNode().translation())
            .c_str());
    jointXml->SetAttribute("name", joint->getName().c_str());
    jointXml->SetAttribute("type", "ball");
  }
  else if (joint->getType() == dynamics::EulerJoint::getStaticType())
  {
    dynamics::EulerJoint* euler = static_cast<dynamics::EulerJoint*>(joint);
    (void)euler;

    for (int i = 0; i < 3; i++)
    {
      XMLElement* motor = xmlDoc.NewElement("motor");
      actuatorsXml->InsertEndChild(motor);
      motor->SetAttribute("gear", "100");
      motor->SetAttribute("joint", joint->getDofName(i).c_str());
      motor->SetAttribute("name", joint->getDofName(i).c_str());

      XMLElement* jointX = xmlDoc.NewElement("joint");
      if (isRoot)
      {
        bodyXml->InsertFirstChild(jointX);
      }
      else
      {
        bodyXml->InsertEndChild(jointX);
      }
      jointX->SetAttribute(
          "pos",
          writeVec3(joint->getTransformFromChildBodyNode().translation())
              .c_str());
      jointX->SetAttribute("name", joint->getDofName(i).c_str());
      jointX->SetAttribute("type", "hinge");
      jointX->SetAttribute(
          "axis",
          writeVec3(
              joint->getTransformFromChildBodyNode().linear()
              * euler->getAxis(i))
              .c_str());
      if (euler->getPositionUpperLimit(i) > euler->getPositionLowerLimit(i))
      {
        jointX->SetAttribute("limited", "true");
        jointX->SetAttribute(
            "range",
            (std::to_string(euler->getPositionLowerLimit(i)) + " "
             + std::to_string(euler->getPositionUpperLimit(i)))
                .c_str());
      }
    }
  }
  else if (joint->getType() == dynamics::FreeJoint::getStaticType())
  {
    dynamics::FreeJoint* freeJoint = static_cast<dynamics::FreeJoint*>(joint);
    (void)freeJoint;
    XMLElement* jointXml = xmlDoc.NewElement("joint");
    bodyXml->InsertEndChild(jointXml);
    jointXml->SetAttribute(
        "pos",
        writeVec3(joint->getTransformFromChildBodyNode().translation())
            .c_str());
    jointXml->SetAttribute("name", joint->getName().c_str());
    jointXml->SetAttribute("type", "free");
    jointXml->SetAttribute("stiffness", "0");
    jointXml->SetAttribute("damping", "0");
    jointXml->SetAttribute("frictionloss", "0");
    jointXml->SetAttribute("armature", "0");
    // damping="0" stiffness="0" armature="0"
    if (isRoot)
    {
      jointXml->SetAttribute("damping", "0");
      jointXml->SetAttribute("stiffness", "0");
      jointXml->SetAttribute("armature", "0");
    }
  }
  else if (joint->getType() == dynamics::EulerFreeJoint::getStaticType())
  {
    dynamics::EulerFreeJoint* eulerFreeJoint
        = static_cast<dynamics::EulerFreeJoint*>(joint);
    (void)eulerFreeJoint;
    for (int i = 0; i < 6; i++)
    {
      if (!isRoot)
      {
        XMLElement* motor = xmlDoc.NewElement("motor");
        actuatorsXml->InsertEndChild(motor);
        motor->SetAttribute("gear", "100");
        motor->SetAttribute("joint", joint->getDofName(i).c_str());
        motor->SetAttribute("name", joint->getDofName(i).c_str());
      }

      XMLElement* jointX = xmlDoc.NewElement("joint");
      if (isRoot)
      {
        bodyXml->InsertFirstChild(jointX);
      }
      else
      {
        bodyXml->InsertEndChild(jointX);
      }
      jointX->SetAttribute(
          "pos",
          writeVec3(joint->getTransformFromChildBodyNode().translation())
              .c_str());
      jointX->SetAttribute("name", joint->getDofName(i).c_str());
      jointX->SetAttribute("type", i < 3 ? "hinge" : "slide");
      jointX->SetAttribute(
          "axis",
          writeVec3(
              joint->getTransformFromChildBodyNode().linear()
              * eulerFreeJoint->getAxis(i))
              .c_str());
      if (eulerFreeJoint->getPositionUpperLimit(i)
          > eulerFreeJoint->getPositionLowerLimit(i))
      {
        jointX->SetAttribute("limited", "true");
        jointX->SetAttribute(
            "range",
            (std::to_string(eulerFreeJoint->getPositionLowerLimit(i)) + " "
             + std::to_string(eulerFreeJoint->getPositionUpperLimit(i)))
                .c_str());
      }
      // damping="0" stiffness="0" armature="0"
      if (isRoot)
      {
        jointX->SetAttribute("damping", "0");
        jointX->SetAttribute("stiffness", "0");
        jointX->SetAttribute("armature", "0");
      }
    }
  }
  else
  {
    std::cout << "Unregnized joint type! " << joint->getType() << std::endl;
  }

  for (int i = 0; i < body->getNumShapeNodes(); i++)
  {
    dynamics::ShapeNode* shapeNode = body->getShapeNode(i);
    dynamics::Shape* shape = shapeNode->getShape().get();

    XMLElement* geomXml = xmlDoc.NewElement("geom");
    bodyXml->InsertEndChild(geomXml);
    geomXml->SetAttribute("name", shapeNode->getName().c_str());
    geomXml->SetAttribute(
        "pos", writeVec3(shapeNode->getRelativeTranslation()).c_str());
    geomXml->SetAttribute(
        "euler",
        writeVec3(math::matrixToEulerXYZ(shapeNode->getRelativeRotation()))
            .c_str());

    // Create the object from scratch
    if (shape->getType() == "BoxShape")
    {
      dynamics::BoxShape* boxShape = dynamic_cast<dynamics::BoxShape*>(shape);

      geomXml->SetAttribute("type", "box");
      geomXml->SetAttribute("size", writeVec3(boxShape->getSize()).c_str());
    }
    else if (shape->getType() == "MeshShape")
    {
      dynamics::MeshShape* meshShape
          = dynamic_cast<dynamics::MeshShape*>(shape);
      std::string meshAbsolutePath = meshShape->getMeshPath();
      int geometryStart = meshAbsolutePath.find("Geometry/");
      if (geometryStart != std::string::npos)
      {
        meshAbsolutePath
            = meshAbsolutePath.substr(geometryStart + strlen("Geometry/"));
      }

      std::string meshName = meshAbsolutePath;
      if (meshName.find(".") != std::string::npos)
      {
        meshName = meshName.substr(0, meshName.find("."));
      }

      geomXml->SetAttribute("type", "mesh");
      geomXml->SetAttribute("mesh", meshName.c_str());

      XMLElement* meshXml = xmlDoc.NewElement("mesh");
      meshXml->SetAttribute("name", meshName.c_str());
      meshXml->SetAttribute("file", (meshName + ".vtp.ply.stl").c_str());
      meshXml->SetAttribute("scale", writeVec3(meshShape->getScale()).c_str());
      assetXml->InsertEndChild(meshXml);
    }
    else if (shape->getType() == "SphereShape")
    {
      dynamics::SphereShape* sphereShape
          = dynamic_cast<dynamics::SphereShape*>(shape);

      geomXml->SetAttribute("type", "sphere");
      geomXml->SetAttribute("size", sphereShape->getRadius());
    }
    else if (shape->getType() == "CapsuleShape")
    {
      dynamics::CapsuleShape* capsuleShape
          = dynamic_cast<dynamics::CapsuleShape*>(shape);

      geomXml->SetAttribute("type", "capsule");
      geomXml->SetAttribute(
          "size",
          (std::to_string(capsuleShape->getRadius()) + " "
           + std::to_string(capsuleShape->getHeight()))
              .c_str());
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

  for (int i = 0; i < body->getNumChildJoints(); i++)
  {
    recursivelyWriteJointAndBody(
        xmlDoc, assetXml, actuatorsXml, bodyXml, body->getChildJoint(i), false);
  }

  elem->InsertEndChild(bodyXml);
}

//==============================================================================
void MJCFExporter::writeSkeleton(
    const std::string& path, std::shared_ptr<dynamics::Skeleton> skel)
{
  (void)path;
  (void)skel;

  using namespace tinyxml2;

  tinyxml2::XMLDocument xmlDoc;

  XMLElement* mujoco = xmlDoc.NewElement("mujoco");
  xmlDoc.InsertFirstChild(mujoco);

  // Set some global values

  XMLElement* compiler = xmlDoc.NewElement("compiler");
  compiler->SetAttribute("angle", "radian");
  compiler->SetAttribute("coordinate", "local");
  compiler->SetAttribute("meshdir", "Geometry/");
  compiler->SetAttribute("inertiafromgeom", "auto");
  compiler->SetAttribute("balanceinertia", "true");
  compiler->SetAttribute("boundmass", "0.001");
  compiler->SetAttribute("boundinertia", "0.001");
  mujoco->InsertFirstChild(compiler);

  XMLElement* defaultXml = xmlDoc.NewElement("default");
  mujoco->InsertEndChild(defaultXml);

  XMLElement* defaultGeom = xmlDoc.NewElement("geom");
  defaultGeom->SetAttribute(
      "conaffinity", "0"); // Disable all collisions between meshes
  defaultGeom->SetAttribute("rgba", "0.7 0.5 .3 1");
  defaultGeom->SetAttribute("margin", "0.001");
  defaultXml->InsertEndChild(defaultGeom);

  XMLElement* defaultSite = xmlDoc.NewElement("site");
  defaultSite->SetAttribute("rgba", "0.7 0.5 0.3 1");
  defaultXml->InsertEndChild(defaultSite);

  XMLElement* defaultJoint = xmlDoc.NewElement("joint");
  defaultJoint->SetAttribute("limited", "true");
  defaultJoint->SetAttribute("damping", "0.5");
  defaultJoint->SetAttribute("armature", "0.1");
  defaultJoint->SetAttribute("stiffness", "2");
  defaultXml->InsertEndChild(defaultJoint);

  XMLElement* defaultMotor = xmlDoc.NewElement("motor");
  defaultMotor->SetAttribute("ctrllimited", "true");
  defaultMotor->SetAttribute("ctrlrange", "-1 1");
  defaultXml->InsertEndChild(defaultMotor);

  XMLElement* option = xmlDoc.NewElement("option");
  option->SetAttribute("timestep", "0.01");
  mujoco->InsertEndChild(option);

  XMLElement* size = xmlDoc.NewElement("size");
  size->SetAttribute("njmax", "1000");
  size->SetAttribute("nconmax", "400");
  size->SetAttribute("nuser_jnt", "1");
  mujoco->InsertEndChild(size);

  // <option timestep="0.01"/>
  // <size njmax="1000" nconmax="400" nuser_jnt="1"/>

  XMLElement* assetXml = xmlDoc.NewElement("asset");
  mujoco->InsertEndChild(assetXml);
  // clang-format off
  /*
  <asset>
        <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
        <material name="MatPlane" reflectance="0.2" texrepeat="1 1" texuniform="true" texture="grid"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
  */
  // clang-format on
  XMLElement* skyboxTexture = xmlDoc.NewElement("texture");
  skyboxTexture->SetAttribute("name", "skybox");
  skyboxTexture->SetAttribute("builtin", "gradient");
  skyboxTexture->SetAttribute("height", "100");
  skyboxTexture->SetAttribute("rgb1", ".4 .5 .6");
  skyboxTexture->SetAttribute("rgb2", "0 0 0");
  skyboxTexture->SetAttribute("type", "skybox");
  skyboxTexture->SetAttribute("width", "100");
  assetXml->InsertEndChild(skyboxTexture);
  XMLElement* flatTexture = xmlDoc.NewElement("texture");
  flatTexture->SetAttribute("name", "flat");
  flatTexture->SetAttribute("builtin", "flat");
  flatTexture->SetAttribute("height", "1278");
  flatTexture->SetAttribute("mark", "cross");
  flatTexture->SetAttribute("markrgb", "1 1 1");
  flatTexture->SetAttribute("name", "texgeom");
  flatTexture->SetAttribute("random", "0.01");
  flatTexture->SetAttribute("rgb1", "0.8 0.6 0.4");
  flatTexture->SetAttribute("rgb2", "0.8 0.6 0.4");
  flatTexture->SetAttribute("type", "cube");
  flatTexture->SetAttribute("width", "127");
  assetXml->InsertEndChild(flatTexture);
  XMLElement* gridTexture = xmlDoc.NewElement("texture");
  gridTexture->SetAttribute("name", "grid");
  gridTexture->SetAttribute("type", "2d");
  gridTexture->SetAttribute("builtin", "checker");
  gridTexture->SetAttribute("rgb1", ".1 .2 .3");
  gridTexture->SetAttribute("rgb2", ".1 .2 .3");
  gridTexture->SetAttribute("width", "300");
  gridTexture->SetAttribute("height", "300");
  gridTexture->SetAttribute("mark", "edge");
  gridTexture->SetAttribute("markrgb", ".2 .3 .4");
  assetXml->InsertEndChild(gridTexture);
  XMLElement* gridMaterial = xmlDoc.NewElement("material");
  gridMaterial->SetAttribute("name", "MatPlane");
  gridMaterial->SetAttribute("reflectance", "0.2");
  gridMaterial->SetAttribute("texrepeat", "1 1");
  gridMaterial->SetAttribute("texuniform", "true");
  gridMaterial->SetAttribute("texture", "grid");
  assetXml->InsertEndChild(gridMaterial);
  XMLElement* geomMaterial = xmlDoc.NewElement("material");
  geomMaterial->SetAttribute("name", "geom");
  geomMaterial->SetAttribute("texture", "texgeom");
  geomMaterial->SetAttribute("texuniform", "true");
  assetXml->InsertEndChild(geomMaterial);

  XMLElement* worldbody = xmlDoc.NewElement("worldbody");
  mujoco->InsertEndChild(worldbody);

  /*
  <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0
  0" rgba="0.8 0.9 0.8 1" size="50 50 0.2" type="plane"/> <light cutoff="100"
  diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3"
  specular=".1 .1 .1"/>
  */

  worldbody->InsertNewComment(
      "\n"
      "        <body name=\"treadmill\" pos=\"0 0 0\">\n"
      "            <geom pos=\"0 0 .2\" friction=\"1 .1 .1\"  "
      "material=\"MatPlane\" name=\"treadmill\"  "
      "rgba=\".4 .5 .4 1\" mass=\"20000\" size=\"50 1.5 0.1\" type=\"box\" "
      "condim=\"3\" />\n"
      "            <joint name=\"treadmill\" axis = \"1 0 0 \" pos=\"0 "
      "0 0\" range=\"-100 100\" type=\"slide\"/>\n"
      "        </body>\n"
      "        ");

  XMLElement* floor = xmlDoc.NewElement("geom");
  floor->SetAttribute("condim", "3");
  floor->SetAttribute("friction", "1 .1 .1");
  floor->SetAttribute("material", "MatPlane");
  floor->SetAttribute("name", "floor");
  floor->SetAttribute("pos", "0 0 0");
  floor->SetAttribute("rgba", "0.8 0.9 0.8 1");
  floor->SetAttribute("size", "50 50 0.2");
  floor->SetAttribute("type", "plane");
  worldbody->InsertEndChild(floor);

  XMLElement* light = xmlDoc.NewElement("light");
  light->SetAttribute("cutoff", "100");
  light->SetAttribute("diffuse", "1 1 1");
  light->SetAttribute("dir", "0 0 -1.3");
  light->SetAttribute("directional", "true");
  light->SetAttribute("exponent", "1");
  light->SetAttribute("pos", "0 0 1.3");
  light->SetAttribute("specular", ".1 .1 .1");
  worldbody->InsertEndChild(light);

  XMLElement* actuators = xmlDoc.NewElement("actuator");
  mujoco->InsertEndChild(actuators);

  recursivelyWriteJointAndBody(
      xmlDoc, assetXml, actuators, worldbody, skel->getRootJoint(), true);

  std::cout << "Saving MJCF file to " << path << std::endl;
  xmlDoc.SaveFile(path.c_str());
};

} // namespace utils
} // namespace dart