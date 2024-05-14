#include "dart/server/GUIStateMachine.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#include <assimp/scene.h>
#include <boost/filesystem.hpp>
// #include <urdf_sensor/sensor.h>

#include "dart/collision/CollisionResult.hpp"
#include "dart/common/Aspect.hpp"
#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CapsuleShape.hpp"
#include "dart/dynamics/ConstantCurveIncompressibleJoint.hpp"
#include "dart/dynamics/Inertia.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/ShapeFrame.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/proto/GUI.pb.h"
#include "dart/server/RawJsonUtils.hpp"
#include "dart/server/external/base64/base64.h"
#include "dart/simulation/World.hpp"

namespace dart {
namespace server {

GUIStateMachine::GUIStateMachine() : mMessagesQueued(0)
{
}

GUIStateMachine::~GUIStateMachine()
{
}

std::string GUIStateMachine::getCurrentStateAsJson()
{
  proto::CommandList list;
  for (auto pair : mLayers)
  {
    encodeCreateLayer(list, pair.second);
  }
  for (auto pair : mBoxes)
  {
    encodeCreateBox(list, pair.second);
  }
  for (auto pair : mTextures)
  {
    encodeCreateTexture(list, pair.second);
  }
  for (auto pair : mMeshes)
  {
    encodeCreateMesh(list, pair.second);
  }
  for (auto pair : mSpheres)
  {
    encodeCreateSphere(list, pair.second);
  }
  for (auto pair : mCapsules)
  {
    encodeCreateCapsule(list, pair.second);
  }
  for (auto pair : mCones)
  {
    encodeCreateCone(list, pair.second);
  }
  for (auto pair : mCylinders)
  {
    encodeCreateCylinder(list, pair.second);
  }
  for (auto pair : mLines)
  {
    encodeCreateLine(list, pair.second);
  }
  for (auto pair : mTooltips)
  {
    encodeSetTooltip(list, pair.second);
  }
  for (auto pair : mObjectWarnings)
  {
    encodeSetObjectWarning(list, pair.second);
  }
  for (auto pair : mSpanWarnings)
  {
    encodeSetSpanWarning(list, pair.second);
  }
  for (auto pair : mText)
  {
    encodeCreateText(list, pair.second);
  }
  for (auto pair : mButtons)
  {
    encodeCreateButton(list, pair.second);
  }
  for (auto pair : mSliders)
  {
    encodeCreateSlider(list, pair.second);
  }
  for (auto pair : mPlots)
  {
    encodeCreatePlot(list, pair.second);
  }
  for (auto pair : mRichPlots)
  {
    encodeCreateRichPlot(list, pair.second);
    for (auto dataPair : pair.second.data)
    {
      encodeSetRichPlotData(list, pair.second.key, dataPair.second);
    }
  }
  for (auto key : mDragEnabled)
  {
    encodeEnableDrag(list, key);
  }
  for (auto key : mTooltipEditable)
  {
    encodeEnableEditTooltip(list, key);
  }

  return list.SerializeAsString();
}

/// This formats the latest set of commands as JSON, and clears the buffer
std::string GUIStateMachine::flushJson()
{
  const std::lock_guard<std::recursive_mutex> lock(mProtoMutex);

  mCommandListOutputBuffer.clear();
  mCommandList.SerializeToString(&mCommandListOutputBuffer);

  // Reset
  mMessagesQueued = 0;
  mCommandList.Clear();

  return mCommandListOutputBuffer;
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
void GUIStateMachine::renderWorld(
    const std::shared_ptr<simulation::World>& world,
    const std::string& prefix,
    bool renderForces,
    bool renderForceMagnitudes,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    renderSkeleton(
        world->getSkeletonRef(i), prefix, Eigen::Vector4s::Ones() * -1, layer);
  }

  const collision::CollisionResult& result = world->getLastCollisionResult();
  deleteObjectsByPrefix(prefix + "__contact_");
  if (renderForces)
  {
    for (int i = 0; i < result.getNumContacts(); i++)
    {
      const collision::Contact& contact = result.getContact(i);
      s_t scale = renderForceMagnitudes ? contact.lcpResult * 10 : 2;
      std::vector<Eigen::Vector3s> points;
      points.push_back(contact.point);
      points.push_back(contact.point + (contact.normal * scale));
      createLine(
          prefix + "__contact_" + std::to_string(i) + "_a",
          points,
          Eigen::Vector4s(1.0, 0.5, 0.5, 1.0),
          layer);
      std::vector<Eigen::Vector3s> pointsB;
      pointsB.push_back(contact.point);
      pointsB.push_back(contact.point - (contact.normal * scale));
      createLine(
          prefix + "__contact_" + std::to_string(i) + "_b",
          pointsB,
          Eigen::Vector4s(0, 1, 0, 1.0),
          layer);
    }
  }
}

/// This is a high-level command that creates a basis
void GUIStateMachine::renderBasis(
    s_t scale,
    const std::string& prefix,
    const Eigen::Vector3s pos,
    const Eigen::Vector3s euler,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  T.translation() = pos;
  T.linear() = math::eulerXYZToMatrix(euler);

  std::vector<Eigen::Vector3s> pointsX;
  pointsX.push_back(T * Eigen::Vector3s::Zero());
  pointsX.push_back(T * (Eigen::Vector3s::UnitX() * scale));
  std::vector<Eigen::Vector3s> pointsY;
  pointsY.push_back(T * Eigen::Vector3s::Zero());
  pointsY.push_back(T * (Eigen::Vector3s::UnitY() * scale));
  std::vector<Eigen::Vector3s> pointsZ;
  pointsZ.push_back(T * Eigen::Vector3s::Zero());
  pointsZ.push_back(T * (Eigen::Vector3s::UnitZ() * scale));

  deleteObjectsByPrefix(prefix + "__basis_");
  std::vector<s_t> width;
  width.push_back(1.0);
  width.push_back(0.3);
  createLine(
      prefix + "__basis_unitX",
      pointsX,
      Eigen::Vector4s(1.0, 0.0, 0.0, 1.0),
      layer,
      width);
  createLine(
      prefix + "__basis_unitY",
      pointsY,
      Eigen::Vector4s(0.0, 1.0, 0.0, 1.0),
      layer,
      width);
  createLine(
      prefix + "__basis_unitZ",
      pointsZ,
      Eigen::Vector4s(0.0, 0.0, 1.0, 1.0),
      layer,
      width);
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
void GUIStateMachine::renderSkeleton(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::string& prefix,
    Eigen::Vector4s overrideColor,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  bool useOriginalColor = overrideColor == -1 * Eigen::Vector4s::Ones();

  for (int j = 0; j < skel->getNumJoints(); j++)
  {
    dynamics::Joint* joint = skel->getJoint(j);
    if (joint->getType()
        == dynamics::ConstantCurveIncompressibleJoint::getStaticType())
    {
      dynamics::ConstantCurveIncompressibleJoint* curveJoint
          = static_cast<dynamics::ConstantCurveIncompressibleJoint*>(joint);
      Eigen::Isometry3s parentT
          = curveJoint->getParentBodyNode() != nullptr
                ? curveJoint->getParentBodyNode()->getWorldTransform()
                : Eigen::Isometry3s::Identity();

      Eigen::Vector3s neutralPos = curveJoint->getNeutralPos();
      Eigen::Vector3s totalPos = neutralPos + curveJoint->getPositionsStatic();
      curveJoint->setNeutralPos(Eigen::Vector3s::Zero());
      std::vector<Eigen::Vector3s> points;
      int numPoints = 10;
      for (int i = 0; i <= numPoints; i++)
      {
        s_t frac = ((s_t)(i) / (s_t)numPoints);
        Eigen::Isometry3s localT
            = parentT
              * curveJoint->getRelativeTransformAt(
                  frac * totalPos, frac * curveJoint->getLength());
        points.push_back(localT.translation());
      }
      curveJoint->setNeutralPos(neutralPos);
      Eigen::Isometry3s T = curveJoint->getChildBodyNode()->getWorldTransform();
      renderBasis(
          0.05,
          curveJoint->getName(),
          T.translation(),
          math::matrixToEulerXYZ(T.linear()));

      std::stringstream jointNameStream;
      jointNameStream << prefix << "_";
      jointNameStream << skel->getName();
      jointNameStream << "_";
      jointNameStream << joint->getName();
      std::string jointName = jointNameStream.str();

      createLine(
          jointName,
          points,
          useOriginalColor ? Eigen::Vector4s(1, 0, 0, 1) : overrideColor,
          layer);
    }
  }
  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    dynamics::BodyNode* node = skel->getBodyNode(j);
    if (node == nullptr)
    {
      std::cout << "ERROR! GUIStateMachine found a null body node! This "
                   "isn't supposed to be possible. Proceeding anyways."
                << std::endl;
      continue;
    }

    for (int k = 0; k < node->getNumShapeNodes(); k++)
    {
      dynamics::ShapeNode* shapeNode = node->getShapeNode(k);
      dynamics::Shape* shape = shapeNode->getShape().get();

      std::stringstream shapeNameStream;
      shapeNameStream << prefix << "_";
      shapeNameStream << skel->getName();
      shapeNameStream << "_";
      shapeNameStream << node->getName();
      shapeNameStream << "_";
      shapeNameStream << k;
      std::string shapeName = shapeNameStream.str();

      if (!shapeNode->hasVisualAspect())
        continue;

      if (shape == nullptr)
      {
        dtwarn << "Found a ShapeNode with no attached Shape object. This can "
                  "sometimes happen if you're trying to load models with "
                  "corrupted mesh files. Ignoring: "
               << shapeName << std::endl;
        continue;
      }

      dynamics::VisualAspect* visual = shapeNode->getVisualAspect(true);
      if (visual == nullptr)
        continue;

      if (!hasObject(shapeName))
      {
        if (!visual->isHidden())
        {
          // Create the object from scratch
          if (shape->getType() == "BoxShape")
          {
            dynamics::BoxShape* boxShape
                = dynamic_cast<dynamics::BoxShape*>(shape);
            createBox(
                shapeName,
                boxShape->getSize(),
                shapeNode->getWorldTransform().translation(),
                math::matrixToEulerXYZ(shapeNode->getWorldTransform().linear()),
                useOriginalColor ? visual->getRGBA() : overrideColor,
                layer,
                visual->getCastShadows(),
                visual->getReceiveShadows());
          }
          else if (shape->getType() == "MeshShape")
          {
            dynamics::MeshShape* meshShape
                = dynamic_cast<dynamics::MeshShape*>(shape);
            createMeshASSIMP(
                shapeName,
                meshShape->getMesh(),
                meshShape->getMeshPath(),
                shapeNode->getWorldTransform().translation(),
                math::matrixToEulerXYZ(shapeNode->getWorldTransform().linear()),
                meshShape->getScale(),
                useOriginalColor ? visual->getRGBA() : overrideColor,
                layer,
                visual->getCastShadows(),
                visual->getReceiveShadows());
          }
          else if (shape->getType() == "SphereShape")
          {
            dynamics::SphereShape* sphereShape
                = dynamic_cast<dynamics::SphereShape*>(shape);
            createSphere(
                shapeName,
                sphereShape->getRadius(),
                shapeNode->getWorldTransform().translation(),
                useOriginalColor ? visual->getRGBA() : overrideColor,
                layer,
                visual->getCastShadows(),
                visual->getReceiveShadows());
          }
          else if (shape->getType() == "CapsuleShape")
          {
            dynamics::CapsuleShape* capsuleShape
                = dynamic_cast<dynamics::CapsuleShape*>(shape);
            createCapsule(
                shapeName,
                capsuleShape->getRadius(),
                capsuleShape->getHeight(),
                shapeNode->getWorldTransform().translation(),
                math::matrixToEulerXYZ(shapeNode->getWorldTransform().linear()),
                useOriginalColor ? visual->getRGBA() : overrideColor,
                layer,
                visual->getCastShadows(),
                visual->getReceiveShadows());
          }
          else if (
              shape->getType() == "EllipsoidShape"
              && dynamic_cast<dynamics::EllipsoidShape*>(shape)->isSphere())
          {
            dynamics::EllipsoidShape* sphereShape
                = dynamic_cast<dynamics::EllipsoidShape*>(shape);
            createSphere(
                shapeName,
                sphereShape->getRadii()[0],
                shapeNode->getWorldTransform().translation(),
                useOriginalColor ? visual->getRGBA() : overrideColor,
                layer,
                visual->getCastShadows(),
                visual->getReceiveShadows());
          }
          else
          {
            dterr << "[GUIStateMachine.renderSkeleton()] Attempting to render "
                     "a shape type ["
                  << shape->getType() << "] that is not supported "
                  << "by the web GUI. Currently, only BoxShape and "
                  << "EllipsoidShape (only when all the radii are equal) and "
                     "SphereShape "
                     "and MeshShape and CapsuleShape are "
                  << "supported. This shape will be invisible in the GUI.\n";
          }
          setObjectTooltip(shapeName, node->getName());
        }
      }
      else
      {
        // Otherwise, we just need to send updates for anything that changed
        if (visual->isHidden())
        {
          deleteObject(shapeName);
        }
        else
        {
          Eigen::Vector3s pos = shapeNode->getWorldTransform().translation();
          Eigen::Vector3s euler
              = math::matrixToEulerXYZ(shapeNode->getWorldTransform().linear());
          Eigen::Vector4s color
              = useOriginalColor ? visual->getRGBA() : overrideColor;
          // std::cout << "Color " << shapeName << ":" << color << std::endl;
          Eigen::Vector3s scale = Eigen::Vector3s::Zero();
          if (shape->getType() == "BoxShape")
          {
            dynamics::BoxShape* boxShape
                = dynamic_cast<dynamics::BoxShape*>(shape);
            scale = boxShape->getSize();
          }
          else if (shape->getType() == "MeshShape")
          {
            dynamics::MeshShape* meshShape
                = dynamic_cast<dynamics::MeshShape*>(shape);
            scale = meshShape->getScale();
          }
          else if (shape->getType() == "SphereShape")
          {
            dynamics::SphereShape* sphereShape
                = dynamic_cast<dynamics::SphereShape*>(shape);
            scale = Eigen::Vector3s::Ones() * sphereShape->getRadius();
          }
          else if (shape->getType() == "CapsuleShape")
          {
            dynamics::CapsuleShape* capsuleShape
                = dynamic_cast<dynamics::CapsuleShape*>(shape);
            scale = Eigen::Vector3s(
                capsuleShape->getRadius(),
                capsuleShape->getRadius(),
                capsuleShape->getHeight());
          }
          else if (
              shape->getType() == "EllipsoidShape"
              && dynamic_cast<dynamics::EllipsoidShape*>(shape)->isSphere())
          {
            dynamics::EllipsoidShape* sphereShape
                = dynamic_cast<dynamics::EllipsoidShape*>(shape);
            scale = Eigen::Vector3s::Ones() * sphereShape->getRadii()[0];
          }

          if (getObjectScale(shapeName) != scale)
            setObjectScale(shapeName, scale);
          if (getObjectPosition(shapeName) != pos)
            setObjectPosition(shapeName, pos);
          if (getObjectRotation(shapeName) != euler)
            setObjectRotation(shapeName, euler);
          if (getObjectColor(shapeName) != color && !useOriginalColor)
            setObjectColor(shapeName, color);
        }
      }
    }
  }
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
void GUIStateMachine::renderSkeletonInertiaCubes(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::string& prefix,
    Eigen::Vector4s color,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    dynamics::BodyNode* node = skel->getBodyNode(j);
    if (node == nullptr)
    {
      std::cout << "ERROR! GUIStateMachine found a null body node! This "
                   "isn't supposed to be possible. Proceeding anyways."
                << std::endl;
      continue;
    }

    Eigen::Vector3s com = node->getCOM();
    Eigen::Vector6s dimsAndEuler = node->getInertia().getDimsAndEulerVector();
    Eigen::Vector3s dims = dimsAndEuler.head<3>();
    Eigen::Vector3s euler = dimsAndEuler.tail<3>();
    Eigen::Matrix3s R = math::eulerXYZToMatrix(euler);
    std::string name = prefix + node->getName();

    createBox(
        name,
        dims,
        com,
        math::matrixToEulerXYZ(node->getWorldTransform().linear() * R),
        color,
        layer,
        false,
        false);
    setObjectTooltip(name, node->getName() + " Inertia");
  }
}

/// This either creates or moves an arrow to have the new start and end points
void GUIStateMachine::renderArrow(
    Eigen::Vector3s start,
    Eigen::Vector3s end,
    s_t bodyRadius,
    s_t tipRadius,
    Eigen::Vector4s color,
    const std::string& prefix,
    const std::string& layer)
{
  std::string cylinderKey = prefix + "_cylinder";
  std::string coneKey = prefix + "_cone";

  s_t length = (end - start).norm();
  s_t headRatio = 0.5;
  s_t headLength = length * headRatio;
  s_t bodyLength = length * (1.0 - headRatio);

  Eigen::Vector3s bodyCenter
      = start + (end - start).normalized() * bodyLength * 0.5;
  Eigen::Vector3s headCenter
      = end + (start - end).normalized() * headLength * 0.5;

  Eigen::Matrix3s R = Eigen::Matrix3s::Zero();
  R.col(1) = (end - start).normalized();
  Eigen::Vector3s cross = Eigen::Vector3s::UnitX();
  if ((R.col(1) - cross).norm() < 1e-8)
  {
    cross = Eigen::Vector3s::UnitZ();
  }
  R.col(0) = R.col(1).cross(cross).normalized();
  R.col(2) = R.col(1).cross(R.col(0)).normalized();
  Eigen::Vector3s euler = math::matrixToEulerXYZ(R);

  // If this arrow alread exists, just update it
  if (mCones.find(coneKey) != mCones.end()
      && mCylinders.find(cylinderKey) != mCylinders.end())
  {
    setObjectPosition(cylinderKey, bodyCenter);
    setObjectRotation(cylinderKey, euler);
    setObjectScale(
        cylinderKey, Eigen::Vector3s(bodyRadius, bodyLength, bodyRadius));
    setObjectColor(cylinderKey, color);

    setObjectPosition(coneKey, headCenter);
    setObjectRotation(coneKey, euler);
    setObjectScale(coneKey, Eigen::Vector3s(tipRadius, headLength, tipRadius));
    setObjectColor(coneKey, color);
  }
  // If it doesn't yet exist, we've got to create the objects
  else
  {
    createCylinder(cylinderKey, 1.0, 1.0, bodyCenter, euler, color, layer);
    setObjectScale(
        cylinderKey, Eigen::Vector3s(bodyRadius, bodyLength, bodyRadius));

    createCone(coneKey, 1.0, 1.0, headCenter, euler, color, layer);
    setObjectScale(coneKey, Eigen::Vector3s(tipRadius, headLength, tipRadius));
  }
}

/// This is a high-level command that renders a given trajectory as a bunch of
/// lines in the world, one per body
void GUIStateMachine::renderTrajectoryLines(
    std::shared_ptr<simulation::World> originalWorld,
    Eigen::MatrixXs positions,
    std::string prefix,
    const std::string& layer)
{
  // Just clone the world, to avoid contention for the world, since this method
  // can spend a long time setting the world into different states.
  std::shared_ptr<simulation::World> world = originalWorld->clone();

  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  assert(positions.rows() == world->getNumDofs());

  std::unordered_map<std::string, std::vector<Eigen::Vector3s>> paths;
  std::unordered_map<std::string, Eigen::Vector4s> colors;

  neural::RestorableSnapshot snapshot(world);
  for (int t = 0; t < positions.cols(); t++)
  {
    world->setPositions(positions.col(t));
    for (int i = 0; i < world->getNumSkeletons(); i++)
    {
      std::shared_ptr<dynamics::Skeleton> skel = world->getSkeleton(i);
      for (int j = 0; j < skel->getNumBodyNodes(); j++)
      {
        dynamics::BodyNode* node = skel->getBodyNode(j);
        std::vector<dynamics::ShapeNode*> shapeNodes
            = node->getShapeNodesWith<dynamics::VisualAspect>();
        for (int k = 0; k < shapeNodes.size(); k++)
        {
          dynamics::ShapeNode* node = shapeNodes[k];
          dynamics::VisualAspect* visual = node->getVisualAspect();

          std::stringstream shapeNameStream;
          shapeNameStream << prefix << "_" << skel->getName() << "_"
                          << node->getName() << "_" << k;
          std::string shapeName = shapeNameStream.str();
          paths[shapeName].push_back(node->getWorldTransform().translation());
          colors[shapeName] = visual->getRGBA();
        }
      }
    }
  }
  snapshot.restore();

  for (auto pair : paths)
  {
    // This command will automatically overwrite any lines with the same key
    createLine(pair.first, pair.second, colors[pair.first], layer);
  }
}

/// This is a high-level command that renders a wrench on a body node
void GUIStateMachine::renderBodyWrench(
    const dynamics::BodyNode* body,
    Eigen::Vector6s wrench,
    s_t scaleFactor,
    std::string prefix,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Eigen::Isometry3s T = body->getWorldTransform();

  Eigen::Vector3s tau = wrench.head<3>();
  Eigen::Vector3s f = wrench.tail<3>();
  Eigen::Matrix3s skew = math::makeSkewSymmetric(f);

  Eigen::Vector3s residual = f.dot(tau) * (f / f.squaredNorm());
  Eigen::Vector3s r = -skew.completeOrthogonalDecomposition().solve(tau);

  Eigen::Vector3s recoveredTau = r.cross(f) + residual;
  Eigen::Vector3s diff = tau - recoveredTau;
  if (diff.squaredNorm() > 1e-8)
  {
    std::cout << "Error in renderBodyWrench()! Got diff: " << diff.squaredNorm()
              << std::endl;
  }

  std::vector<Eigen::Vector3s> torqueLine;
  torqueLine.push_back(T * (r * scaleFactor));
  torqueLine.push_back(T * ((r + residual) * scaleFactor));
  std::vector<Eigen::Vector3s> forceLine;
  forceLine.push_back(T * (r * scaleFactor));
  forceLine.push_back(T * ((r + f) * scaleFactor));

  createLine(
      prefix + "_" + body->getName() + "_torque",
      torqueLine,
      Eigen::Vector4s(0.8, 0.8, 0.8, 1.0),
      layer);
  createLine(
      prefix + "_" + body->getName() + "_force",
      forceLine,
      Eigen::Vector4s(1.0, 0.0, 0.0, 1.0),
      layer);
}

/// This renders little velocity lines starting at every vertex in the passed
/// in body
void GUIStateMachine::renderMovingBodyNodeVertices(
    const dynamics::BodyNode* body,
    s_t scaleFactor,
    std::string prefix,
    const std::string& layer)
{
  std::vector<dynamics::BodyNode::MovingVertex> verts
      = body->getMovingVerticesInWorldSpace();

  for (int i = 0; i < verts.size(); i++)
  {
    std::vector<Eigen::Vector3s> line;
    line.push_back(verts[i].pos);
    line.push_back(verts[i].pos + verts[i].vel * scaleFactor);
    createLine(
        prefix + "_" + body->getName() + "_" + std::to_string(i),
        line,
        Eigen::Vector4s(1.0, 0.0, 0.0, 1.0),
        layer);
  }
}

/// This is a high-level command that removes the lines rendering a wrench on
/// a body node
void GUIStateMachine::clearBodyWrench(
    const dynamics::BodyNode* body, std::string prefix)
{
  deleteObject(prefix + "_" + body->getName() + "_torque");
  deleteObject(prefix + "_" + body->getName() + "_force");
}

/// This completely resets the web GUI, deleting all objects, UI elements, and
/// listeners
void GUIStateMachine::clear()
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_clear_all()->set_dummy(true);
  });

  mBoxes.clear();
  mSpheres.clear();
  mCylinders.clear();
  mCones.clear();
  mCapsules.clear();
  mLines.clear();
  mMeshes.clear();
  mText.clear();
  mButtons.clear();
  mSliders.clear();
  mPlots.clear();
}

/// Set frames per second
void GUIStateMachine::setFramesPerSecond(int framesPerSecond)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  queueCommand([this, framesPerSecond](proto::CommandList& json) {
    encodeSetFramesPerSecond(json, framesPerSecond);
  });
}

/// This creates a layer in the web GUI
void GUIStateMachine::createLayer(
    std::string key, const Eigen::Vector4s& color, bool defaultShow)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Layer& layer = mLayers[key];
  layer.key = key;
  layer.color = color;
  layer.defaultShow = defaultShow;

  queueCommand([this, key](proto::CommandList& json) {
    encodeCreateLayer(json, mLayers[key]);
  });
}

/// This creates a box in the web GUI under a specified key
void GUIStateMachine::createBox(
    std::string key,
    const Eigen::Vector3s& size,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Box& box = mBoxes[key];
  box.key = key;
  box.size = size;
  box.pos = pos;
  box.euler = euler;
  box.color = color;
  box.layer = layer;
  box.castShadows = castShadows;
  box.receiveShadows = receiveShadows;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateBox(list, mBoxes[key]);
  });
}

/// This creates a sphere in the web GUI under a specified key
void GUIStateMachine::createSphere(
    std::string key,
    s_t radius,
    const Eigen::Vector3s& pos,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  Eigen::Vector3s radii = Eigen::Vector3s::Constant(radius);
  createSphere(key, radii, pos, color, layer, castShadows, receiveShadows);
}

/// This creates a sphere in the web GUI under a specified key
void GUIStateMachine::createSphere(
    std::string key,
    Eigen::Vector3s radii,
    const Eigen::Vector3s& pos,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Sphere& sphere = mSpheres[key];
  sphere.key = key;
  sphere.radii = radii;
  sphere.pos = pos;
  sphere.color = color;
  sphere.layer = layer;
  sphere.castShadows = castShadows;
  sphere.receiveShadows = receiveShadows;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateSphere(list, mSpheres.at(key));
  });
}

/// This creates a cone in the web GUI under a specified key
void GUIStateMachine::createCone(
    std::string key,
    s_t radius,
    s_t height,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Cone& cone = mCones[key];
  cone.key = key;
  cone.radius = radius;
  cone.height = height;
  cone.pos = pos;
  cone.euler = euler;
  cone.color = color;
  cone.layer = layer;
  cone.castShadows = castShadows;
  cone.receiveShadows = receiveShadows;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateCone(list, mCones.at(key));
  });
}

/// This creates a cylinder in the web GUI under a specified key
void GUIStateMachine::createCylinder(
    std::string key,
    s_t radius,
    s_t height,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  Cylinder& cylinder = mCylinders[key];
  cylinder.key = key;
  cylinder.radius = radius;
  cylinder.height = height;
  cylinder.pos = pos;
  cylinder.euler = euler;
  cylinder.color = color;
  cylinder.layer = layer;
  cylinder.castShadows = castShadows;
  cylinder.receiveShadows = receiveShadows;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateCylinder(list, mCylinders.at(key));
  });
}

/// This creates a capsule in the web GUI under a specified key
void GUIStateMachine::createCapsule(
    std::string key,
    s_t radius,
    s_t height,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Capsule& capsule = mCapsules[key];
  capsule.key = key;
  capsule.radius = radius;
  capsule.height = height;
  capsule.pos = pos;
  capsule.euler = euler;
  capsule.color = color;
  capsule.layer = layer;
  capsule.castShadows = castShadows;
  capsule.receiveShadows = receiveShadows;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateCapsule(list, mCapsules.at(key));
  });
}

/// This creates a line in the web GUI under a specified key
void GUIStateMachine::createLine(
    std::string key,
    const std::vector<Eigen::Vector3s>& points,
    const Eigen::Vector4s& color,
    const std::string& layer,
    const std::vector<s_t>& width)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Line& line = mLines[key];
  line.key = key;
  line.points = points;
  line.color = color;
  line.layer = layer;
  line.width.clear();
  for (int i = 0; i < line.points.size(); i++)
  {
    if (i < width.size())
    {
      line.width.push_back(width[i]);
    }
    else
    {
      line.width.push_back(1.0);
    }
  }

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateLine(list, mLines.at(key));
  });
}

/// This creates a mesh in the web GUI under a specified key, using raw shape
/// data
void GUIStateMachine::createMesh(
    std::string key,
    const std::vector<Eigen::Vector3s>& vertices,
    const std::vector<Eigen::Vector3s>& vertexNormals,
    const std::vector<Eigen::Vector3i>& faces,
    const std::vector<Eigen::Vector2s>& uv,
    const std::vector<std::string>& textures,
    const std::vector<int>& textureStartIndices,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector3s& scale,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Mesh& mesh = mMeshes[key];
  mesh.key = key;
  mesh.vertices = vertices;
  mesh.vertexNormals = vertexNormals;
  mesh.faces = faces;
  mesh.uv = uv;
  mesh.textures = textures;
  mesh.textureStartIndices = textureStartIndices;
  mesh.pos = pos;
  mesh.euler = euler;
  mesh.scale = scale;
  mesh.color = color;
  mesh.layer = layer;
  mesh.castShadows = castShadows;
  mesh.receiveShadows = receiveShadows;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateMesh(list, mMeshes.at(key));
  });
}

/// This creates a mesh in the web GUI under a specified key, from the ASSIMP
/// mesh
void GUIStateMachine::createMeshASSIMP(
    const std::string& key,
    const aiScene* mesh,
    const std::string& meshPath,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector3s& scale,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  std::vector<Eigen::Vector3s> vertices;
  std::vector<Eigen::Vector3s> vertexNormals;
  std::vector<Eigen::Vector3i> faces;
  std::vector<Eigen::Vector2s> uv;
  std::vector<std::string> textures;
  std::vector<int> textureStartIndices;

  std::string currentTexturePath = "";

  for (int i = 0; i < mesh->mNumMeshes; i++)
  {
    aiMesh* m = mesh->mMeshes[i];
    aiMaterial* mtl = nullptr;
    if (mesh->mMaterials != nullptr)
    {
      mtl = mesh->mMaterials[m->mMaterialIndex];
    }
    aiString path;
    if (mtl != nullptr
        && aiReturn_SUCCESS
               == aiGetMaterialTexture(mtl, aiTextureType_DIFFUSE, 0, &path))
    {
      std::string newTexturePath = std::string(path.C_Str());
      if (newTexturePath != currentTexturePath)
      {
        currentTexturePath = newTexturePath;
        textures.push_back(newTexturePath);
        textureStartIndices.push_back(vertices.size());
        if (mTextures.find(newTexturePath) == mTextures.end())
        {
          boost::filesystem::path fullPath = boost::filesystem::canonical(
              boost::filesystem::path(currentTexturePath),
              boost::filesystem::path(
                  meshPath.substr(0, meshPath.find_last_of("/"))));

          createTextureFromFile(newTexturePath, std::string(fullPath.c_str()));
        }
      }
    }

    for (int j = 0; j < m->mNumVertices; j++)
    {
      vertices.emplace_back(
          m->mVertices[j][0], m->mVertices[j][1], m->mVertices[j][2]);
      if (m->mNormals != nullptr)
      {
        vertexNormals.emplace_back(
            m->mNormals[j][0], m->mNormals[j][1], m->mNormals[j][2]);
      }
      if (m->mNumUVComponents[0] >= 2)
      {
        uv.emplace_back(m->mTextureCoords[0][j][0], m->mTextureCoords[0][j][1]);
      }
      /*
      if (m->mNumUVComponents[0] == 2 && m->mTextureCoords[0][j][0])
      {
      }
      */
    }
    for (int k = 0; k < m->mNumFaces; k++)
    {
      assert(m->mFaces[k].mNumIndices == 3);
      faces.emplace_back(
          m->mFaces[k].mIndices[0],
          m->mFaces[k].mIndices[1],
          m->mFaces[k].mIndices[2]);
    }
  }

  createMesh(
      key,
      vertices,
      vertexNormals,
      faces,
      uv,
      textures,
      textureStartIndices,
      pos,
      euler,
      scale,
      color,
      layer,
      castShadows,
      receiveShadows);
}

void GUIStateMachine::createMeshFromShape(
    const std::string& key,
    const std::shared_ptr<dynamics::MeshShape> mesh,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector3s& scale,
    const Eigen::Vector4s& color,
    const std::string& layer,
    bool castShadows,
    bool receiveShadows)
{
  createMeshASSIMP(
      key,
      mesh->getMesh(),
      mesh->getMeshPath(),
      pos,
      euler,
      scale,
      color,
      layer,
      castShadows,
      receiveShadows);
}

/// This creates a texture object, to be sent to the web frontend
void GUIStateMachine::createTexture(
    const std::string& key, const std::string& base64)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Texture tex;
  tex.key = key;
  tex.base64 = base64;

  mTextures[key] = tex;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateTexture(list, mTextures[key]);
  });
}

/// This creates a texture object by loading it from a file
void GUIStateMachine::createTextureFromFile(
    const std::string& key, const std::string& path)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  std::ifstream in(path);
  std::ostringstream sstr;
  sstr << in.rdbuf();

  std::string suffix = path.substr(path.find_last_of(".") + 1);
  std::string base64
      = "data:image/" + suffix + ";base64, " + ::base64_encode(sstr.str());
  createTexture(key, base64);
}

/// This returns true if we've already got an object with the key "key"
bool GUIStateMachine::hasObject(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return true;
  if (mSpheres.find(key) != mSpheres.end())
    return true;
  if (mCapsules.find(key) != mCapsules.end())
    return true;
  if (mCylinders.find(key) != mCylinders.end())
    return true;
  if (mCones.find(key) != mCones.end())
    return true;
  if (mLines.find(key) != mLines.end())
    return true;
  if (mMeshes.find(key) != mMeshes.end())
    return true;
  return false;
}

/// This returns the position of an object, if we've got it (and it's not a
/// line). Otherwise it returns Vector3s::Zero().
Eigen::Vector3s GUIStateMachine::getObjectPosition(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes.at(key).pos;
  if (mSpheres.find(key) != mSpheres.end())
    return mSpheres.at(key).pos;
  if (mCapsules.find(key) != mCapsules.end())
    return mCapsules.at(key).pos;
  if (mCones.find(key) != mCones.end())
    return mCones.at(key).pos;
  if (mCylinders.find(key) != mCylinders.end())
    return mCylinders.at(key).pos;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes.at(key).pos;
  return Eigen::Vector3s::Zero();
}

/// This returns the rotation of an object, if we've got it (and it's not a
/// line or a sphere). Otherwise it returns Vector3s::Zero().
Eigen::Vector3s GUIStateMachine::getObjectRotation(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes.at(key).euler;
  if (mCapsules.find(key) != mCapsules.end())
    return mCapsules.at(key).euler;
  if (mCones.find(key) != mCones.end())
    return mCones.at(key).euler;
  if (mCylinders.find(key) != mCylinders.end())
    return mCylinders.at(key).euler;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes.at(key).euler;
  return Eigen::Vector3s::Zero();
}

/// This returns the color of an object, if we've got it. Otherwise it returns
/// Vector3s::Zero().
Eigen::Vector4s GUIStateMachine::getObjectColor(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes.at(key).color;
  if (mSpheres.find(key) != mSpheres.end())
    return mSpheres.at(key).color;
  if (mCapsules.find(key) != mCapsules.end())
    return mCapsules.at(key).color;
  if (mCones.find(key) != mCones.end())
    return mCones.at(key).color;
  if (mCylinders.find(key) != mCylinders.end())
    return mCylinders.at(key).color;
  if (mLines.find(key) != mLines.end())
    return mLines.at(key).color;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes.at(key).color;
  return Eigen::Vector4s::Zero();
}

/// This returns the size of a box, scale of a mesh, 3vec of [radius, radius,
/// radius] for a sphere, and [radius, radius, height] for a capsule.
Eigen::Vector3s GUIStateMachine::getObjectScale(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes.at(key).size;
  if (mSpheres.find(key) != mSpheres.end())
    return mSpheres.at(key).radii;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes.at(key).scale;
  return Eigen::Vector3s::Zero();
}

/// This moves an object (e.g. box, sphere, line) to a specified position
void GUIStateMachine::setObjectPosition(
    const std::string& key, const Eigen::Vector3s& pos)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
  {
    mBoxes.at(key).pos = pos;
  }
  if (mSpheres.find(key) != mSpheres.end())
  {
    mSpheres.at(key).pos = pos;
  }
  if (mCapsules.find(key) != mCapsules.end())
  {
    mCapsules.at(key).pos = pos;
  }
  if (mCones.find(key) != mCones.end())
  {
    mCones.at(key).pos = pos;
  }
  if (mCylinders.find(key) != mCylinders.end())
  {
    mCylinders.at(key).pos = pos;
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes.at(key).pos = pos;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_position()->set_key(getStringCode(key));
    command->mutable_set_object_position()->add_data((double)pos(0));
    command->mutable_set_object_position()->add_data((double)pos(1));
    command->mutable_set_object_position()->add_data((double)pos(2));
  });
}

/// This moves an object (e.g. box, sphere, line) to a specified orientation
void GUIStateMachine::setObjectRotation(
    const std::string& key, const Eigen::Vector3s& euler)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
  {
    mBoxes.at(key).euler = euler;
  }
  if (mCapsules.find(key) != mCapsules.end())
  {
    mCapsules.at(key).euler = euler;
  }
  if (mCylinders.find(key) != mCylinders.end())
  {
    mCylinders.at(key).euler = euler;
  }
  if (mCones.find(key) != mCones.end())
  {
    mCones.at(key).euler = euler;
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes.at(key).euler = euler;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_rotation()->set_key(getStringCode(key));
    command->mutable_set_object_rotation()->add_data((double)euler(0));
    command->mutable_set_object_rotation()->add_data((double)euler(1));
    command->mutable_set_object_rotation()->add_data((double)euler(2));
  });
}

/// This changes an object (e.g. box, sphere, line) color
void GUIStateMachine::setObjectColor(
    const std::string& key, const Eigen::Vector4s& color)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
  {
    mBoxes.at(key).color = color;
  }
  if (mSpheres.find(key) != mSpheres.end())
  {
    mSpheres.at(key).color = color;
  }
  if (mLines.find(key) != mLines.end())
  {
    mLines.at(key).color = color;
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes.at(key).color = color;
  }
  if (mCapsules.find(key) != mCapsules.end())
  {
    mCapsules.at(key).color = color;
  }
  if (mCylinders.find(key) != mCylinders.end())
  {
    mCylinders.at(key).color = color;
  }
  if (mCones.find(key) != mCones.end())
  {
    mCones.at(key).color = color;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_color()->set_key(getStringCode(key));
    command->mutable_set_object_color()->add_data((double)color(0));
    command->mutable_set_object_color()->add_data((double)color(1));
    command->mutable_set_object_color()->add_data((double)color(2));
    command->mutable_set_object_color()->add_data((double)color(3));
  });
}

/// This changes an object (e.g. box, sphere, mesh) size. Has no effect on
/// lines.
void GUIStateMachine::setObjectScale(
    const std::string& key, const Eigen::Vector3s& scale)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
  {
    mBoxes.at(key).size = scale;
  }
  if (mSpheres.find(key) != mSpheres.end())
  {
    mSpheres.at(key).radii = scale;
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes.at(key).scale = scale;
  }
  if (mCapsules.find(key) != mCapsules.end())
  {
    mCapsules.at(key).height = scale(1);
    mCapsules.at(key).radius = scale(0);
  }
  if (mCylinders.find(key) != mCylinders.end())
  {
    mCylinders.at(key).height = scale(1);
    mCylinders.at(key).radius = scale(0);
  }
  if (mCones.find(key) != mCones.end())
  {
    mCones.at(key).height = scale(1);
    mCones.at(key).radius = scale(0);
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_scale()->set_key(getStringCode(key));
    command->mutable_set_object_scale()->add_data((double)scale(0));
    command->mutable_set_object_scale()->add_data((double)scale(1));
    command->mutable_set_object_scale()->add_data((double)scale(2));
  });
}

/// This sets a tooltip for the object at key.
void GUIStateMachine::setObjectTooltip(
    const std::string& key, const std::string& tooltip)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Tooltip& t = mTooltips[key];
  t.key = key;
  t.tooltip = tooltip;

  queueCommand([this, key](proto::CommandList& list) {
    encodeSetTooltip(list, mTooltips[key]);
  });
}

/// This removes a tooltip for the object at key.
void GUIStateMachine::deleteObjectTooltip(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mTooltips.erase(key);

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_delete_object_tooltip()->set_key(getStringCode(key));
  });
}

/// This sets a warning on a span of timesteps - only has an effect on the
/// replay viewer, not on a live view
void GUIStateMachine::setSpanWarning(
    int startTimestep,
    int endTimestep,
    const std::string& warningKey,
    const std::string& warning,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  SpanWarning& w = mSpanWarnings[warningKey];
  w.warningKey = warningKey;
  w.warning = warning;
  w.startTimestep = startTimestep;
  w.endTimestep = endTimestep;
  w.layer = layer;

  queueCommand([this, warningKey](proto::CommandList& list) {
    encodeSetSpanWarning(list, mSpanWarnings[warningKey]);
  });
}

/// This sets a warning for the object at key.
void GUIStateMachine::setObjectWarning(
    const std::string& key,
    const std::string& warningKey,
    const std::string& warning,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  ObjectWarning& w = mObjectWarnings[warningKey];
  w.key = key;
  w.warningKey = warningKey;
  w.warning = warning;
  w.layer = layer;

  queueCommand([this, warningKey](proto::CommandList& list) {
    encodeSetObjectWarning(list, mObjectWarnings[warningKey]);
  });
}

/// This deletes a warning for the object at key.
void GUIStateMachine::deleteObjectWarning(
    const std::string& key, const std::string& warningKey)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mObjectWarnings.erase(key);

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_delete_object_warning()->set_key(getStringCode(key));
    command->mutable_delete_object_warning()->set_warning_key(
        getStringCode(warningKey));
  });
}

/// This sets an object to allow dragging around by the mouse on the GUI
void GUIStateMachine::setObjectDragEnabled(const std::string& key)
{
  mDragEnabled.emplace(key);
  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_enable_drag()->set_key(getStringCode(key));
  });
}

/// This sets an object to allow editing the tooltip by double-clicking on the
/// object
void GUIStateMachine::setObjectTooltipEditable(const std::string& key)
{
  mTooltipEditable.emplace(key);
  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_enable_edit_tooltip()->set_key(getStringCode(key));
  });
}

/// This deletes an object by key
void GUIStateMachine::deleteObject(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  // We actually want to delete objects even if they don't currently exist,
  // because people may skip frames in the visualizer and we want to properly
  // clean up. if (!hasObject(key))
  // {
  //   return;
  // }

  mBoxes.erase(key);
  mSpheres.erase(key);
  mLines.erase(key);
  mMeshes.erase(key);
  mCapsules.erase(key);
  mCones.erase(key);
  mCylinders.erase(key);
  mTooltips.erase(key);

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_delete_object()->set_key(getStringCode(key));
  });
}

/// This deletes all the objects that match a given prefix
void GUIStateMachine::deleteObjectsByPrefix(const std::string& prefix)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  std::vector<std::string> toDelete;
  for (auto& pair : mBoxes)
  {
    if (prefix.size() <= pair.first.size())
    {
      auto res
          = std::mismatch(prefix.begin(), prefix.end(), pair.first.begin());
      if (res.first == prefix.end())
      {
        // We've got a match!
        toDelete.push_back(pair.first);
      }
    }
  }

  for (auto& pair : mSpheres)
  {
    if (prefix.size() <= pair.first.size())
    {
      auto res
          = std::mismatch(prefix.begin(), prefix.end(), pair.first.begin());
      if (res.first == prefix.end())
      {
        // We've got a match!
        toDelete.push_back(pair.first);
      }
    }
  }

  for (auto& pair : mLines)
  {
    if (prefix.size() <= pair.first.size())
    {
      auto res
          = std::mismatch(prefix.begin(), prefix.end(), pair.first.begin());
      if (res.first == prefix.end())
      {
        // We've got a match!
        toDelete.push_back(pair.first);
      }
    }
  }

  for (auto& pair : mMeshes)
  {
    if (prefix.size() <= pair.first.size())
    {
      auto res
          = std::mismatch(prefix.begin(), prefix.end(), pair.first.begin());
      if (res.first == prefix.end())
      {
        // We've got a match!
        toDelete.push_back(pair.first);
      }
    }
  }

  for (auto& pair : mCapsules)
  {
    if (prefix.size() <= pair.first.size())
    {
      auto res
          = std::mismatch(prefix.begin(), prefix.end(), pair.first.begin());
      if (res.first == prefix.end())
      {
        // We've got a match!
        toDelete.push_back(pair.first);
      }
    }
  }

  for (auto& pair : mCylinders)
  {
    if (prefix.size() <= pair.first.size())
    {
      auto res
          = std::mismatch(prefix.begin(), prefix.end(), pair.first.begin());
      if (res.first == prefix.end())
      {
        // We've got a match!
        toDelete.push_back(pair.first);
      }
    }
  }

  for (auto& pair : mCones)
  {
    if (prefix.size() <= pair.first.size())
    {
      auto res
          = std::mismatch(prefix.begin(), prefix.end(), pair.first.begin());
      if (res.first == prefix.end())
      {
        // We've got a match!
        toDelete.push_back(pair.first);
      }
    }
  }

  for (std::string key : toDelete)
  {
    deleteObject(key);
  }
}

/// This places some text on the screen at the specified coordinates
void GUIStateMachine::createText(
    const std::string& key,
    const std::string& contents,
    const Eigen::Vector2i& fromTopLeft,
    const Eigen::Vector2i& size,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Text text;
  text.key = key;
  text.contents = contents;
  text.fromTopLeft = fromTopLeft;
  text.size = size;
  text.layer = layer;

  mText[key] = text;

  queueCommand([&](proto::CommandList& list) { encodeCreateText(list, text); });
}

/// This changes the contents of text on the screen
void GUIStateMachine::setTextContents(
    const std::string& key, const std::string& newContents)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mText.find(key) != mText.end())
  {
    mText[key].contents = newContents;

    queueCommand([&](proto::CommandList& list) {
      proto::Command* command = list.add_command();
      command->mutable_set_text_contents()->set_key(getStringCode(key));
      command->mutable_set_text_contents()->set_contents(newContents);
    });
  }
  else
  {
    std::cout
        << "Tried to setTextContents() for a key (" << key
        << ") that doesn't exist as a Text object. Call createText() first."
        << std::endl;
  }
}

/// This places a clickable button on the screen at the specified coordinates
void GUIStateMachine::createButton(
    const std::string& key,
    const std::string& label,
    const Eigen::Vector2i& fromTopLeft,
    const Eigen::Vector2i& size,
    std::function<void()> onClick,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Button button;
  button.key = key;
  button.label = label;
  button.fromTopLeft = fromTopLeft;
  button.size = size;
  button.onClick = onClick;
  button.layer = layer;

  mButtons[key] = button;

  queueCommand(
      [&](proto::CommandList& list) { encodeCreateButton(list, button); });
}

/// This changes the contents of text on the screen
void GUIStateMachine::setButtonLabel(
    const std::string& key, const std::string& newLabel)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mButtons.find(key) != mButtons.end())
  {
    mButtons[key].label = newLabel;

    queueCommand([&](proto::CommandList& list) {
      proto::Command* command = list.add_command();
      command->mutable_set_button_label()->set_key(getStringCode(key));
      command->mutable_set_button_label()->set_label(newLabel);
    });
  }
  else
  {
    std::cout
        << "Tried to setButtonLabel() for a key (" << key
        << ") that doesn't exist as a Button object. Call createButton() first."
        << std::endl;
  }
}

/// This creates a slider
void GUIStateMachine::createSlider(
    const std::string& key,
    const Eigen::Vector2i& fromTopLeft,
    const Eigen::Vector2i& size,
    s_t min,
    s_t max,
    s_t value,
    bool onlyInts,
    bool horizontal,
    std::function<void(s_t)> onChange,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Slider slider;
  slider.key = key;
  slider.fromTopLeft = fromTopLeft;
  slider.size = size;
  slider.min = min;
  slider.max = max;
  slider.value = value;
  slider.onlyInts = onlyInts;
  slider.horizontal = horizontal;
  slider.onChange = onChange;
  slider.layer = layer;

  mSliders[key] = slider;

  queueCommand(
      [&](proto::CommandList& list) { encodeCreateSlider(list, slider); });
}

/// This changes the contents of text on the screen
void GUIStateMachine::setSliderValue(const std::string& key, s_t value)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].value = value;

    queueCommand([&](proto::CommandList& list) {
      proto::Command* command = list.add_command();
      command->mutable_set_slider_value()->set_key(getStringCode(key));
      command->mutable_set_slider_value()->set_value((double)value);
    });
  }
  else
  {
    std::cout
        << "Tried to setSliderValue() for a key (" << key
        << ") that doesn't exist as a Slider object. Call createSlider() first."
        << std::endl;
  }
}

/// This changes the contents of text on the screen
void GUIStateMachine::setSliderMin(const std::string& key, s_t min)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].min = min;

    queueCommand([&](proto::CommandList& list) {
      proto::Command* command = list.add_command();
      command->mutable_set_slider_min()->set_key(getStringCode(key));
      command->mutable_set_slider_min()->set_value((double)min);
    });
  }
  else
  {
    std::cout
        << "Tried to setSliderMin() for a key (" << key
        << ") that doesn't exist as a Slider object. Call createSlider() first."
        << std::endl;
  }
}

/// This changes the contents of text on the screen
void GUIStateMachine::setSliderMax(const std::string& key, s_t max)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].max = max;

    queueCommand([&](proto::CommandList& list) {
      proto::Command* command = list.add_command();
      command->mutable_set_slider_max()->set_key(getStringCode(key));
      command->mutable_set_slider_max()->set_value((double)max);
    });
  }
  else
  {
    std::cout
        << "Tried to setSliderMax() for a key (" << key
        << ") that doesn't exist as a Slider object. Call createSlider() first."
        << std::endl;
  }
}

/// This creates a plot to display data on the GUI
void GUIStateMachine::createPlot(
    const std::string& key,
    const Eigen::Vector2i& fromTopLeft,
    const Eigen::Vector2i& size,
    const std::vector<s_t>& xs,
    s_t minX,
    s_t maxX,
    const std::vector<s_t>& ys,
    s_t minY,
    s_t maxY,
    const std::string& type,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Plot plot;
  plot.key = key;
  plot.fromTopLeft = fromTopLeft;
  plot.size = size;
  plot.xs = xs;
  plot.minX = minX;
  plot.maxX = maxX;
  plot.ys = ys;
  plot.minY = minY;
  plot.maxY = maxY;
  plot.type = type;
  plot.layer = layer;

  mPlots[key] = plot;

  queueCommand([&](proto::CommandList& list) { encodeCreatePlot(list, plot); });
}

/// This changes the contents of a plot, along with its display limits
void GUIStateMachine::setPlotData(
    const std::string& key,
    const std::vector<s_t>& xs,
    s_t minX,
    s_t maxX,
    const std::vector<s_t>& ys,
    s_t minY,
    s_t maxY)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mPlots.find(key) != mPlots.end())
  {
    mPlots[key].xs = xs;
    mPlots[key].minX = minX;
    mPlots[key].maxX = maxX;
    mPlots[key].ys = ys;
    mPlots[key].minY = minY;
    mPlots[key].maxY = maxY;

    queueCommand([&](proto::CommandList& list) {
      proto::Command* command = list.add_command();
      command->mutable_set_plot_data()->set_key(getStringCode(key));
      command->mutable_set_plot_data()->add_bounds((double)minX);
      command->mutable_set_plot_data()->add_bounds((double)maxX);
      command->mutable_set_plot_data()->add_bounds((double)minY);
      command->mutable_set_plot_data()->add_bounds((double)maxY);
      for (s_t x : xs)
      {
        command->mutable_set_plot_data()->add_xs((double)x);
      }
      for (s_t y : ys)
      {
        command->mutable_set_plot_data()->add_ys((double)y);
      }
    });
  }
  else
  {
    std::cout
        << "Tried to setPlotData() for a key (" << key
        << ") that doesn't exist as a Plot object. Call createPlot() first."
        << std::endl;
  }
}

/// This creates a rich plot with axis labels, a title, tickmarks, and
/// multiple simultaneous lines
void GUIStateMachine::createRichPlot(
    const std::string& key,
    const Eigen::Vector2i& fromTopLeft,
    const Eigen::Vector2i& size,
    s_t minX,
    s_t maxX,
    s_t minY,
    s_t maxY,
    const std::string& title,
    const std::string& xAxisLabel,
    const std::string& yAxisLabel,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  RichPlot plot;
  plot.key = key;
  plot.fromTopLeft = fromTopLeft;
  plot.size = size;
  plot.minX = minX;
  plot.maxX = maxX;
  plot.minY = minY;
  plot.maxY = maxY;
  plot.title = title;
  plot.xAxisLabel = xAxisLabel;
  plot.yAxisLabel = yAxisLabel;
  plot.layer = layer;

  mRichPlots[key] = plot;

  queueCommand(
      [&](proto::CommandList& list) { encodeCreateRichPlot(list, plot); });
}

/// This sets a single data stream for a rich plot. `name` should be human
/// readable and unique. You can overwrite data by using the same `name` with
/// multiple calls to `setRichPlotData`.
void GUIStateMachine::setRichPlotData(
    const std::string& key,
    const std::string& name,
    const std::string& color,
    const std::string& type,
    const std::vector<s_t>& xs,
    const std::vector<s_t>& ys)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mRichPlots.find(key) != mRichPlots.end())
  {
    RichPlotData data;
    data.name = name;
    data.color = color;
    data.type = type;
    data.xs = xs;
    data.ys = ys;
    mRichPlots[key].data[name] = data;

    queueCommand([this, key, data](proto::CommandList& list) {
      encodeSetRichPlotData(list, key, data);
    });
  }
  else
  {
    std::cout << "Tried to setRichPlotData() for a key (" << key
              << ") that doesn't exist as a RichPlot object. Call "
                 "createRichPlot() first."
              << std::endl;
  }
}

/// This sets the plot bounds for a rich plot object
void GUIStateMachine::setRichPlotBounds(
    const std::string& key, s_t minX, s_t maxX, s_t minY, s_t maxY)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mRichPlots.find(key) != mRichPlots.end())
  {
    mRichPlots[key].minX = minX;
    mRichPlots[key].maxX = maxX;
    mRichPlots[key].minY = minY;
    mRichPlots[key].maxY = maxY;

    queueCommand([&](proto::CommandList& list) {
      proto::Command* command = list.add_command();
      command->mutable_set_rich_plot_bounds()->set_key(getStringCode(key));
      command->mutable_set_rich_plot_bounds()->add_bounds((double)minX);
      command->mutable_set_rich_plot_bounds()->add_bounds((double)maxX);
      command->mutable_set_rich_plot_bounds()->add_bounds((double)minY);
      command->mutable_set_rich_plot_bounds()->add_bounds((double)maxY);
    });
  }
  else
  {
    std::cout << "Tried to setRichPlotBounds() for a key (" << key
              << ") that doesn't exist as a RichPlot object. Call "
                 "createRichPlot() first."
              << std::endl;
  }
}

/// This moves a UI element on the screen
void GUIStateMachine::setUIElementPosition(
    const std::string& key, const Eigen::Vector2i& fromTopLeft)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mText.find(key) != mText.end())
  {
    mText[key].fromTopLeft = fromTopLeft;
  }
  if (mButtons.find(key) != mButtons.end())
  {
    mButtons[key].fromTopLeft = fromTopLeft;
  }
  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].fromTopLeft = fromTopLeft;
  }
  if (mPlots.find(key) != mPlots.end())
  {
    mPlots[key].fromTopLeft = fromTopLeft;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_ui_elem_pos()->set_key(getStringCode(key));
    command->mutable_set_ui_elem_pos()->add_fromtopleft(fromTopLeft(0));
    command->mutable_set_ui_elem_pos()->add_fromtopleft(fromTopLeft(1));
  });
}

/// This changes the size of a UI element
void GUIStateMachine::setUIElementSize(
    const std::string& key, const Eigen::Vector2i& size)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mText.find(key) != mText.end())
  {
    mText[key].size = size;
  }
  if (mButtons.find(key) != mButtons.end())
  {
    mButtons[key].size = size;
  }
  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].size = size;
  }
  if (mPlots.find(key) != mPlots.end())
  {
    mPlots[key].size = size;
  }
  if (mRichPlots.find(key) != mRichPlots.end())
  {
    mRichPlots[key].size = size;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_ui_elem_size()->set_key(getStringCode(key));
    command->mutable_set_ui_elem_size()->add_size(size(0));
    command->mutable_set_ui_elem_size()->add_size(size(1));
  });
}

/// This deletes a UI element by key
void GUIStateMachine::deleteUIElement(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mText.erase(key);
  mButtons.erase(key);
  mSliders.erase(key);
  mPlots.erase(key);
  mRichPlots.erase(key);

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_delete_ui_elem()->set_key(getStringCode(key));
  });
}

/// This gets an integer code for a string
int GUIStateMachine::getStringCode(const std::string& key)
{
  if (mStringCodes.count(key) == 0)
  {
    int code = mStringCodes.size() + 1;
    mStringCodes[key] = code;
    mCodeStrings[code] = key;
  }
  return mStringCodes[key];
}

/// This gets a string code for an integer
std::string GUIStateMachine::getCodeString(int code)
{
  if (mCodeStrings.count(code) != 0)
  {
    return mCodeStrings[code];
  }
  else
  {
    return "";
  }
}

void GUIStateMachine::queueCommand(
    std::function<void(proto::CommandList&)> writeCommand)
{
  const std::lock_guard<std::recursive_mutex> lock(mProtoMutex);

  writeCommand(mCommandList);
  mMessagesQueued++;
}

void GUIStateMachine::encodeSetFramesPerSecond(
    proto::CommandList& list, int framesPerSecond)
{
  proto::Command* command = list.add_command();
  command->mutable_set_frames_per_second()->set_framespersecond(
      framesPerSecond);
}

void GUIStateMachine::encodeCreateLayer(proto::CommandList& list, Layer& layer)
{
  proto::Command* command = list.add_command();
  command->mutable_layer()->set_key(getStringCode(layer.key));
  command->mutable_layer()->set_name(layer.key);
  command->mutable_layer()->add_color((double)layer.color(0));
  command->mutable_layer()->add_color((double)layer.color(1));
  command->mutable_layer()->add_color((double)layer.color(2));
  command->mutable_layer()->add_color((double)layer.color(3));
  command->mutable_layer()->set_default_show(layer.defaultShow);
}

void GUIStateMachine::encodeCreateBox(proto::CommandList& list, Box& box)
{
  proto::Command* command = list.add_command();
  command->mutable_box()->set_key(getStringCode(box.key));
  command->mutable_box()->set_layer(getStringCode(box.layer));
  command->mutable_box()->add_data((double)box.size(0));
  command->mutable_box()->add_data((double)box.size(1));
  command->mutable_box()->add_data((double)box.size(2));
  command->mutable_box()->add_data((double)box.pos(0));
  command->mutable_box()->add_data((double)box.pos(1));
  command->mutable_box()->add_data((double)box.pos(2));
  command->mutable_box()->add_data((double)box.euler(0));
  command->mutable_box()->add_data((double)box.euler(1));
  command->mutable_box()->add_data((double)box.euler(2));
  command->mutable_box()->add_data((double)box.color(0));
  command->mutable_box()->add_data((double)box.color(1));
  command->mutable_box()->add_data((double)box.color(2));
  command->mutable_box()->add_data((double)box.color(3));
  command->mutable_box()->set_cast_shadows(box.castShadows);
  command->mutable_box()->set_receive_shadows(box.receiveShadows);
}

void GUIStateMachine::encodeCreateSphere(
    proto::CommandList& list, Sphere& sphere)
{
  proto::Command* command = list.add_command();
  command->mutable_sphere()->set_key(getStringCode(sphere.key));
  command->mutable_sphere()->set_layer(getStringCode(sphere.layer));
  command->mutable_sphere()->set_cast_shadows(sphere.castShadows);
  command->mutable_sphere()->set_receive_shadows(sphere.receiveShadows);
  command->mutable_sphere()->add_data((double)sphere.radii(0));
  command->mutable_sphere()->add_data((double)sphere.radii(1));
  command->mutable_sphere()->add_data((double)sphere.radii(2));
  command->mutable_sphere()->add_data((double)sphere.pos(0));
  command->mutable_sphere()->add_data((double)sphere.pos(1));
  command->mutable_sphere()->add_data((double)sphere.pos(2));
  command->mutable_sphere()->add_data((double)sphere.color(0));
  command->mutable_sphere()->add_data((double)sphere.color(1));
  command->mutable_sphere()->add_data((double)sphere.color(2));
  command->mutable_sphere()->add_data((double)sphere.color(3));

  /*
  json << "{ \"type\": \"create_sphere\", \"key\": \"" << sphere.key
       << "\", \"radius\": " << numberToJson(sphere.radius);
  json << ", \"pos\": ";
  vec3ToJson(json, sphere.pos);
  json << ", \"color\": ";
  vec4ToJson(json, sphere.color);
  json << ", \"layer\": \"" << sphere.layer << "\"";
  json << ", \"cast_shadows\": " << (sphere.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": "
       << (sphere.receiveShadows ? "true" : "false");
  json << "}";
  */
}

void GUIStateMachine::encodeCreateCone(proto::CommandList& list, Cone& cone)
{
  proto::Command* command = list.add_command();
  command->mutable_cone()->set_key(getStringCode(cone.key));
  command->mutable_cone()->set_layer(getStringCode(cone.layer));
  command->mutable_cone()->set_cast_shadows(cone.castShadows);
  command->mutable_cone()->set_receive_shadows(cone.receiveShadows);
  command->mutable_cone()->add_data((double)cone.radius);
  command->mutable_cone()->add_data((double)cone.height);
  command->mutable_cone()->add_data((double)cone.pos(0));
  command->mutable_cone()->add_data((double)cone.pos(1));
  command->mutable_cone()->add_data((double)cone.pos(2));
  command->mutable_cone()->add_data((double)cone.euler(0));
  command->mutable_cone()->add_data((double)cone.euler(1));
  command->mutable_cone()->add_data((double)cone.euler(2));
  command->mutable_cone()->add_data((double)cone.color(0));
  command->mutable_cone()->add_data((double)cone.color(1));
  command->mutable_cone()->add_data((double)cone.color(2));
  command->mutable_cone()->add_data((double)cone.color(3));
}

void GUIStateMachine::encodeCreateCylinder(
    proto::CommandList& list, Cylinder& cylinder)
{
  proto::Command* command = list.add_command();
  command->mutable_cylinder()->set_key(getStringCode(cylinder.key));
  command->mutable_cylinder()->set_layer(getStringCode(cylinder.layer));
  command->mutable_cylinder()->set_cast_shadows(cylinder.castShadows);
  command->mutable_cylinder()->set_receive_shadows(cylinder.receiveShadows);
  command->mutable_cylinder()->add_data((double)cylinder.radius);
  command->mutable_cylinder()->add_data((double)cylinder.height);
  command->mutable_cylinder()->add_data((double)cylinder.pos(0));
  command->mutable_cylinder()->add_data((double)cylinder.pos(1));
  command->mutable_cylinder()->add_data((double)cylinder.pos(2));
  command->mutable_cylinder()->add_data((double)cylinder.euler(0));
  command->mutable_cylinder()->add_data((double)cylinder.euler(1));
  command->mutable_cylinder()->add_data((double)cylinder.euler(2));
  command->mutable_cylinder()->add_data((double)cylinder.color(0));
  command->mutable_cylinder()->add_data((double)cylinder.color(1));
  command->mutable_cylinder()->add_data((double)cylinder.color(2));
  command->mutable_cylinder()->add_data((double)cylinder.color(3));
}

void GUIStateMachine::encodeCreateCapsule(
    proto::CommandList& list, Capsule& capsule)
{
  proto::Command* command = list.add_command();
  command->mutable_capsule()->set_key(getStringCode(capsule.key));
  command->mutable_capsule()->set_layer(getStringCode(capsule.layer));
  command->mutable_capsule()->set_cast_shadows(capsule.castShadows);
  command->mutable_capsule()->set_receive_shadows(capsule.receiveShadows);
  command->mutable_capsule()->add_data((double)capsule.radius);
  command->mutable_capsule()->add_data((double)capsule.height);
  command->mutable_capsule()->add_data((double)capsule.pos(0));
  command->mutable_capsule()->add_data((double)capsule.pos(1));
  command->mutable_capsule()->add_data((double)capsule.pos(2));
  command->mutable_capsule()->add_data((double)capsule.euler(0));
  command->mutable_capsule()->add_data((double)capsule.euler(1));
  command->mutable_capsule()->add_data((double)capsule.euler(2));
  command->mutable_capsule()->add_data((double)capsule.color(0));
  command->mutable_capsule()->add_data((double)capsule.color(1));
  command->mutable_capsule()->add_data((double)capsule.color(2));
  command->mutable_capsule()->add_data((double)capsule.color(3));
}

void GUIStateMachine::encodeCreateLine(proto::CommandList& list, Line& line)
{
  proto::Command* command = list.add_command();
  command->mutable_line()->set_key(getStringCode(line.key));
  command->mutable_line()->set_layer(getStringCode(line.layer));
  command->mutable_line()->add_color((double)line.color(0));
  command->mutable_line()->add_color((double)line.color(1));
  command->mutable_line()->add_color((double)line.color(2));
  command->mutable_line()->add_color((double)line.color(3));
  for (Eigen::Vector3s& point : line.points)
  {
    command->mutable_line()->add_points((double)point(0));
    command->mutable_line()->add_points((double)point(1));
    command->mutable_line()->add_points((double)point(2));
  }
  for (s_t width : line.width)
  {
    command->mutable_line()->add_width(width);
  }
}

void GUIStateMachine::encodeCreateMesh(proto::CommandList& list, Mesh& mesh)
{
  proto::Command* command = list.add_command();
  command->mutable_mesh()->set_key(getStringCode(mesh.key));
  command->mutable_mesh()->set_layer(getStringCode(mesh.layer));
  for (Eigen::Vector3s& vertex : mesh.vertices)
  {
    command->mutable_mesh()->add_vertex((double)vertex(0));
    command->mutable_mesh()->add_vertex((double)vertex(1));
    command->mutable_mesh()->add_vertex((double)vertex(2));
  }
  for (Eigen::Vector3s& normal : mesh.vertexNormals)
  {
    command->mutable_mesh()->add_vertex_normal((double)normal(0));
    command->mutable_mesh()->add_vertex_normal((double)normal(1));
    command->mutable_mesh()->add_vertex_normal((double)normal(2));
  }
  for (Eigen::Vector3i& face : mesh.faces)
  {
    command->mutable_mesh()->add_face(face(0));
    command->mutable_mesh()->add_face(face(1));
    command->mutable_mesh()->add_face(face(2));
  }
  for (Eigen::Vector2s& uv : mesh.uv)
  {
    command->mutable_mesh()->add_uv((double)uv(0));
    command->mutable_mesh()->add_uv((double)uv(1));
  }
  for (int i = 0; i < mesh.textures.size(); i++)
  {
    command->mutable_mesh()->add_texture(getStringCode(mesh.textures[i]));
    command->mutable_mesh()->add_texture_start(mesh.textureStartIndices[i]);
  }
  command->mutable_mesh()->add_data((double)mesh.scale(0));
  command->mutable_mesh()->add_data((double)mesh.scale(1));
  command->mutable_mesh()->add_data((double)mesh.scale(2));
  command->mutable_mesh()->add_data((double)mesh.pos(0));
  command->mutable_mesh()->add_data((double)mesh.pos(1));
  command->mutable_mesh()->add_data((double)mesh.pos(2));
  command->mutable_mesh()->add_data((double)mesh.euler(0));
  command->mutable_mesh()->add_data((double)mesh.euler(1));
  command->mutable_mesh()->add_data((double)mesh.euler(2));
  command->mutable_mesh()->add_data((double)mesh.color(0));
  command->mutable_mesh()->add_data((double)mesh.color(1));
  command->mutable_mesh()->add_data((double)mesh.color(2));
  command->mutable_mesh()->add_data((double)mesh.color(3));
  command->mutable_mesh()->set_cast_shadows(mesh.castShadows);
  command->mutable_mesh()->set_receive_shadows(mesh.receiveShadows);
}

void GUIStateMachine::encodeSetTooltip(
    proto::CommandList& list, Tooltip& tooltip)
{
  proto::Command* command = list.add_command();
  command->mutable_set_object_tooltip()->set_key(getStringCode(tooltip.key));
  command->mutable_set_object_tooltip()->set_tooltip(tooltip.tooltip);
}

void GUIStateMachine::encodeSetObjectWarning(
    proto::CommandList& list, ObjectWarning& warning)
{
  proto::Command* command = list.add_command();
  command->mutable_set_object_warning()->set_key(getStringCode(warning.key));
  command->mutable_set_object_warning()->set_warning_key(
      getStringCode(warning.warningKey));
  command->mutable_set_object_warning()->set_warning(warning.warning);
  command->mutable_set_object_warning()->set_layer(
      getStringCode(warning.layer));
}

void GUIStateMachine::encodeSetSpanWarning(
    proto::CommandList& list, SpanWarning& warning)
{
  proto::Command* command = list.add_command();
  command->mutable_set_span_warning()->set_warning_key(
      getStringCode(warning.warningKey));
  command->mutable_set_span_warning()->set_warning(warning.warning);
  command->mutable_set_span_warning()->set_start_timestep(
      warning.startTimestep);
  command->mutable_set_span_warning()->set_end_timestep(warning.endTimestep);
  command->mutable_set_span_warning()->set_layer(getStringCode(warning.layer));
}

void GUIStateMachine::encodeCreateTexture(
    proto::CommandList& list, Texture& texture)
{
  proto::Command* command = list.add_command();
  command->mutable_texture()->set_key(getStringCode(texture.key));
  command->mutable_texture()->set_base64(texture.base64);
}

void GUIStateMachine::encodeEnableDrag(
    proto::CommandList& list, const std::string& key)
{
  proto::Command* command = list.add_command();
  command->mutable_enable_drag()->set_key(getStringCode(key));
}

void GUIStateMachine::encodeEnableEditTooltip(
    proto::CommandList& list, const std::string& key)
{
  proto::Command* command = list.add_command();
  command->mutable_enable_edit_tooltip()->set_key(getStringCode(key));
}

void GUIStateMachine::encodeCreateText(proto::CommandList& list, Text& text)
{
  proto::Command* command = list.add_command();
  command->mutable_text()->set_key(getStringCode(text.key));
  command->mutable_text()->set_layer(getStringCode(text.layer));
  command->mutable_text()->add_pos(text.fromTopLeft(0));
  command->mutable_text()->add_pos(text.fromTopLeft(1));
  command->mutable_text()->add_pos(text.size(0));
  command->mutable_text()->add_pos(text.size(1));
  command->mutable_text()->set_contents(text.contents);
}

void GUIStateMachine::encodeCreateButton(
    proto::CommandList& list, Button& button)
{
  proto::Command* command = list.add_command();
  command->mutable_button()->set_key(getStringCode(button.key));
  command->mutable_button()->set_layer(getStringCode(button.layer));
  command->mutable_button()->add_pos(button.fromTopLeft(0));
  command->mutable_button()->add_pos(button.fromTopLeft(1));
  command->mutable_button()->add_pos(button.size(0));
  command->mutable_button()->add_pos(button.size(1));
  command->mutable_button()->set_label(button.label);
}

void GUIStateMachine::encodeCreateSlider(
    proto::CommandList& list, Slider& slider)
{
  proto::Command* command = list.add_command();
  command->mutable_slider()->set_key(getStringCode(slider.key));
  command->mutable_slider()->set_layer(getStringCode(slider.layer));
  command->mutable_slider()->add_pos(slider.fromTopLeft(0));
  command->mutable_slider()->add_pos(slider.fromTopLeft(1));
  command->mutable_slider()->add_pos(slider.size(0));
  command->mutable_slider()->add_pos(slider.size(1));
  command->mutable_slider()->add_data((double)slider.min);
  command->mutable_slider()->add_data((double)slider.max);
  command->mutable_slider()->add_data((double)slider.value);
  command->mutable_slider()->set_only_ints(slider.onlyInts);
  command->mutable_slider()->set_horizontal(slider.horizontal);
}

void GUIStateMachine::encodeCreatePlot(proto::CommandList& list, Plot& plot)
{
  proto::Command* command = list.add_command();
  command->mutable_plot()->set_key(getStringCode(plot.key));
  command->mutable_plot()->set_layer(getStringCode(plot.layer));
  command->mutable_plot()->set_plot_type(plot.type);
  command->mutable_plot()->add_pos(plot.fromTopLeft(0));
  command->mutable_plot()->add_pos(plot.fromTopLeft(1));
  command->mutable_plot()->add_pos(plot.size(0));
  command->mutable_plot()->add_pos(plot.size(1));
  command->mutable_plot()->add_bounds((double)plot.minX);
  command->mutable_plot()->add_bounds((double)plot.maxX);
  command->mutable_plot()->add_bounds((double)plot.minY);
  command->mutable_plot()->add_bounds((double)plot.maxY);
  for (s_t x : plot.xs)
  {
    command->mutable_plot()->add_xs((double)x);
  }
  for (s_t y : plot.ys)
  {
    command->mutable_plot()->add_ys((double)y);
  }
}

void GUIStateMachine::encodeCreateRichPlot(
    proto::CommandList& list, RichPlot& plot)
{
  proto::Command* command = list.add_command();
  command->mutable_rich_plot()->set_key(getStringCode(plot.key));
  command->mutable_rich_plot()->set_layer(getStringCode(plot.layer));
  command->mutable_rich_plot()->add_pos(plot.fromTopLeft(0));
  command->mutable_rich_plot()->add_pos(plot.fromTopLeft(1));
  command->mutable_rich_plot()->add_pos(plot.size(0));
  command->mutable_rich_plot()->add_pos(plot.size(1));
  command->mutable_rich_plot()->add_bounds((double)plot.minX);
  command->mutable_rich_plot()->add_bounds((double)plot.maxX);
  command->mutable_rich_plot()->add_bounds((double)plot.minY);
  command->mutable_rich_plot()->add_bounds((double)plot.maxY);
  command->mutable_rich_plot()->set_title(plot.title);
  command->mutable_rich_plot()->set_x_axis_label(plot.xAxisLabel);
  command->mutable_rich_plot()->set_y_axis_label(plot.yAxisLabel);
}

/*
export type SetRichPlotData = {
  type: "set_rich_plot_data";
  key: string;
  name: string;
  color: string;
  xs: number[];
  ys: number[];
  plot_type: "line" | "scatter";
};
*/
void GUIStateMachine::encodeSetRichPlotData(
    proto::CommandList& list,
    const std::string& plotKey,
    const RichPlotData& data)
{
  proto::Command* command = list.add_command();
  command->mutable_set_rich_plot_data()->set_key(getStringCode(plotKey));
  command->mutable_set_rich_plot_data()->set_name(data.name);
  command->mutable_set_rich_plot_data()->set_color(data.color);
  command->mutable_set_rich_plot_data()->set_plot_type(data.type);
  for (s_t x : data.xs)
  {
    command->mutable_set_rich_plot_data()->add_xs((double)x);
  }
  for (s_t y : data.ys)
  {
    command->mutable_set_rich_plot_data()->add_ys((double)y);
  }
}

} // namespace server
} // namespace dart