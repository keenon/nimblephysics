#include "dart/server/GUIStateMachine.hpp"

#include <chrono>
#include <fstream>
#include <sstream>

#include <assimp/scene.h>
#include <boost/filesystem.hpp>
#include <urdf_sensor/sensor.h>

#include "dart/collision/CollisionResult.hpp"
#include "dart/common/Aspect.hpp"
#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CapsuleShape.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/ShapeFrame.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/math/Geometry.hpp"
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
  for (auto pair : mLines)
  {
    encodeCreateLine(list, pair.second);
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
  for (auto key : mMouseInteractionEnabled)
  {
    encodeEnableMouseInteraction(list, key);
  }

  return list.SerializeAsString();
}

/// This formats the latest set of commands as JSON, and clears the buffer
std::string GUIStateMachine::flushJson()
{
  const std::lock_guard<std::recursive_mutex> lock(mProtoMutex);

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
  createLine(
      prefix + "__basis_unitX",
      pointsX,
      Eigen::Vector4s(1.0, 0.0, 0.0, 1.0),
      layer);
  createLine(
      prefix + "__basis_unitY",
      pointsY,
      Eigen::Vector4s(0.0, 1.0, 0.0, 1.0),
      layer);
  createLine(
      prefix + "__basis_unitZ",
      pointsZ,
      Eigen::Vector4s(0.0, 0.0, 1.0, 1.0),
      layer);
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
          if (getObjectColor(shapeName) != color)
            setObjectColor(shapeName, color);
        }
      }
    }
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
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Sphere& sphere = mSpheres[key];
  sphere.key = key;
  sphere.radius = radius;
  sphere.pos = pos;
  sphere.color = color;
  sphere.layer = layer;
  sphere.castShadows = castShadows;
  sphere.receiveShadows = receiveShadows;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateSphere(list, mSpheres[key]);
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
    encodeCreateCapsule(list, mCapsules[key]);
  });
}

/// This creates a line in the web GUI under a specified key
void GUIStateMachine::createLine(
    std::string key,
    const std::vector<Eigen::Vector3s>& points,
    const Eigen::Vector4s& color,
    const std::string& layer)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Line& line = mLines[key];
  line.key = key;
  line.points = points;
  line.color = color;
  line.layer = layer;

  queueCommand([this, key](proto::CommandList& list) {
    encodeCreateLine(list, mLines[key]);
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
    encodeCreateMesh(list, mMeshes[key]);
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
    return mBoxes[key].pos;
  if (mSpheres.find(key) != mSpheres.end())
    return mSpheres[key].pos;
  if (mCapsules.find(key) != mCapsules.end())
    return mCapsules[key].pos;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes[key].pos;
  return Eigen::Vector3s::Zero();
}

/// This returns the rotation of an object, if we've got it (and it's not a
/// line or a sphere). Otherwise it returns Vector3s::Zero().
Eigen::Vector3s GUIStateMachine::getObjectRotation(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes[key].euler;
  if (mCapsules.find(key) != mCapsules.end())
    return mCapsules[key].euler;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes[key].euler;
  return Eigen::Vector3s::Zero();
}

/// This returns the color of an object, if we've got it. Otherwise it returns
/// Vector3s::Zero().
Eigen::Vector4s GUIStateMachine::getObjectColor(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes[key].color;
  if (mSpheres.find(key) != mSpheres.end())
    return mSpheres[key].color;
  if (mCapsules.find(key) != mCapsules.end())
    return mCapsules[key].color;
  if (mLines.find(key) != mLines.end())
    return mLines[key].color;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes[key].color;
  return Eigen::Vector4s::Zero();
}

/// This returns the size of a box, scale of a mesh, 3vec of [radius, radius,
/// radius] for a sphere, and [radius, radius, height] for a capsule.
Eigen::Vector3s GUIStateMachine::getObjectScale(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes[key].size;
  if (mSpheres.find(key) != mSpheres.end())
    return Eigen::Vector3s::Ones() * mSpheres[key].radius;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes[key].scale;
  return Eigen::Vector3s::Zero();
}

/// This moves an object (e.g. box, sphere, line) to a specified position
void GUIStateMachine::setObjectPosition(
    const std::string& key, const Eigen::Vector3s& pos)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
  {
    mBoxes[key].pos = pos;
  }
  if (mSpheres.find(key) != mSpheres.end())
  {
    mSpheres[key].pos = pos;
  }
  if (mCapsules.find(key) != mCapsules.end())
  {
    mCapsules[key].pos = pos;
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes[key].pos = pos;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_position()->set_key(getStringCode(key));
    command->mutable_set_object_position()->add_data(pos(0));
    command->mutable_set_object_position()->add_data(pos(1));
    command->mutable_set_object_position()->add_data(pos(2));
  });
}

/// This moves an object (e.g. box, sphere, line) to a specified orientation
void GUIStateMachine::setObjectRotation(
    const std::string& key, const Eigen::Vector3s& euler)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
  {
    mBoxes[key].euler = euler;
  }
  if (mCapsules.find(key) != mCapsules.end())
  {
    mCapsules[key].euler = euler;
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes[key].euler = euler;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_rotation()->set_key(getStringCode(key));
    command->mutable_set_object_rotation()->add_data(euler(0));
    command->mutable_set_object_rotation()->add_data(euler(1));
    command->mutable_set_object_rotation()->add_data(euler(2));
  });
}

/// This changes an object (e.g. box, sphere, line) color
void GUIStateMachine::setObjectColor(
    const std::string& key, const Eigen::Vector4s& color)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
  {
    mBoxes[key].color = color;
  }
  if (mSpheres.find(key) != mSpheres.end())
  {
    mSpheres[key].color = color;
  }
  if (mLines.find(key) != mLines.end())
  {
    mLines[key].color = color;
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes[key].color = color;
  }
  if (mCapsules.find(key) != mCapsules.end())
  {
    mCapsules[key].color = color;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_color()->set_key(getStringCode(key));
    command->mutable_set_object_color()->add_data(color(0));
    command->mutable_set_object_color()->add_data(color(1));
    command->mutable_set_object_color()->add_data(color(2));
    command->mutable_set_object_color()->add_data(color(3));
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
    mBoxes[key].size = scale;
  }
  if (mSpheres.find(key) != mSpheres.end())
  {
    mSpheres[key].radius = scale(0);
  }
  if (mMeshes.find(key) != mMeshes.end())
  {
    mMeshes[key].scale = scale;
  }

  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_set_object_scale()->set_key(getStringCode(key));
    command->mutable_set_object_scale()->add_data(scale(0));
    command->mutable_set_object_scale()->add_data(scale(1));
    command->mutable_set_object_scale()->add_data(scale(2));
  });
}

/// This sets an object to allow mouse interaction on the GUI
void GUIStateMachine::setObjectMouseInteractionEnabled(const std::string& key)
{
  mMouseInteractionEnabled.emplace(key);
  queueCommand([&](proto::CommandList& list) {
    proto::Command* command = list.add_command();
    command->mutable_enable_mouse_interaction()->set_key(getStringCode(key));
  });
}

/// This deletes an object by key
void GUIStateMachine::deleteObject(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mBoxes.erase(key);
  mSpheres.erase(key);
  mLines.erase(key);
  mMeshes.erase(key);
  mCapsules.erase(key);

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
      command->mutable_set_slider_value()->set_value(value);
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
      command->mutable_set_slider_min()->set_value(min);
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
      command->mutable_set_slider_max()->set_value(max);
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
      command->mutable_set_plot_data()->add_bounds(minX);
      command->mutable_set_plot_data()->add_bounds(maxX);
      command->mutable_set_plot_data()->add_bounds(minY);
      command->mutable_set_plot_data()->add_bounds(maxY);
      for (s_t x : xs)
      {
        command->mutable_set_plot_data()->add_xs(x);
      }
      for (s_t y : ys)
      {
        command->mutable_set_plot_data()->add_ys(y);
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
      command->mutable_set_rich_plot_bounds()->add_bounds(minX);
      command->mutable_set_rich_plot_bounds()->add_bounds(maxX);
      command->mutable_set_rich_plot_bounds()->add_bounds(minY);
      command->mutable_set_rich_plot_bounds()->add_bounds(maxY);
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
  command->mutable_layer()->add_color(layer.color(0));
  command->mutable_layer()->add_color(layer.color(1));
  command->mutable_layer()->add_color(layer.color(2));
  command->mutable_layer()->add_color(layer.color(3));
  command->mutable_layer()->set_default_show(layer.defaultShow);
}

void GUIStateMachine::encodeCreateBox(proto::CommandList& list, Box& box)
{
  proto::Command* command = list.add_command();
  command->mutable_box()->set_key(getStringCode(box.key));
  command->mutable_box()->set_layer(getStringCode(box.layer));
  command->mutable_box()->add_data(box.size(0));
  command->mutable_box()->add_data(box.size(1));
  command->mutable_box()->add_data(box.size(2));
  command->mutable_box()->add_data(box.pos(0));
  command->mutable_box()->add_data(box.pos(1));
  command->mutable_box()->add_data(box.pos(2));
  command->mutable_box()->add_data(box.euler(0));
  command->mutable_box()->add_data(box.euler(1));
  command->mutable_box()->add_data(box.euler(2));
  command->mutable_box()->add_data(box.color(0));
  command->mutable_box()->add_data(box.color(1));
  command->mutable_box()->add_data(box.color(2));
  command->mutable_box()->add_data(box.color(3));
  command->mutable_box()->set_cast_shadows(box.receiveShadows);
  command->mutable_box()->set_receive_shadows(box.receiveShadows);
}

void GUIStateMachine::encodeCreateSphere(
    proto::CommandList& list, Sphere& sphere)
{
  proto::Command* command = list.add_command();
  command->mutable_sphere()->set_key(getStringCode(sphere.key));
  command->mutable_sphere()->set_layer(getStringCode(sphere.layer));
  command->mutable_sphere()->set_cast_shadows(sphere.receiveShadows);
  command->mutable_sphere()->set_receive_shadows(sphere.receiveShadows);
  command->mutable_sphere()->add_data(sphere.radius);
  command->mutable_sphere()->add_data(sphere.pos(0));
  command->mutable_sphere()->add_data(sphere.pos(1));
  command->mutable_sphere()->add_data(sphere.pos(2));
  command->mutable_sphere()->add_data(sphere.color(0));
  command->mutable_sphere()->add_data(sphere.color(1));
  command->mutable_sphere()->add_data(sphere.color(2));
  command->mutable_sphere()->add_data(sphere.color(3));

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

void GUIStateMachine::encodeCreateCapsule(
    proto::CommandList& list, Capsule& capsule)
{
  proto::Command* command = list.add_command();
  command->mutable_capsule()->set_key(getStringCode(capsule.key));
  command->mutable_capsule()->set_layer(getStringCode(capsule.layer));
  command->mutable_capsule()->set_cast_shadows(capsule.receiveShadows);
  command->mutable_capsule()->set_receive_shadows(capsule.receiveShadows);
  command->mutable_capsule()->add_data(capsule.radius);
  command->mutable_capsule()->add_data(capsule.height);
  command->mutable_capsule()->add_data(capsule.pos(0));
  command->mutable_capsule()->add_data(capsule.pos(1));
  command->mutable_capsule()->add_data(capsule.pos(2));
  command->mutable_capsule()->add_data(capsule.euler(0));
  command->mutable_capsule()->add_data(capsule.euler(1));
  command->mutable_capsule()->add_data(capsule.euler(2));
  command->mutable_capsule()->add_data(capsule.color(0));
  command->mutable_capsule()->add_data(capsule.color(1));
  command->mutable_capsule()->add_data(capsule.color(2));
  command->mutable_capsule()->add_data(capsule.color(3));
}

void GUIStateMachine::encodeCreateLine(proto::CommandList& list, Line& line)
{
  proto::Command* command = list.add_command();
  command->mutable_line()->set_key(getStringCode(line.key));
  command->mutable_line()->set_layer(getStringCode(line.layer));
  command->mutable_line()->add_color(line.color(0));
  command->mutable_line()->add_color(line.color(1));
  command->mutable_line()->add_color(line.color(2));
  command->mutable_line()->add_color(line.color(3));
  for (Eigen::Vector3s& point : line.points)
  {
    command->mutable_line()->add_points(point(0));
    command->mutable_line()->add_points(point(1));
    command->mutable_line()->add_points(point(2));
  }
}

void GUIStateMachine::encodeCreateMesh(proto::CommandList& list, Mesh& mesh)
{
  proto::Command* command = list.add_command();
  command->mutable_mesh()->set_key(getStringCode(mesh.key));
  command->mutable_mesh()->set_layer(getStringCode(mesh.layer));
  for (Eigen::Vector3s& vertex : mesh.vertices)
  {
    command->mutable_mesh()->add_vertex(vertex(0));
    command->mutable_mesh()->add_vertex(vertex(1));
    command->mutable_mesh()->add_vertex(vertex(2));
  }
  for (Eigen::Vector3s& normal : mesh.vertexNormals)
  {
    command->mutable_mesh()->add_vertex_normal(normal(0));
    command->mutable_mesh()->add_vertex_normal(normal(1));
    command->mutable_mesh()->add_vertex_normal(normal(2));
  }
  for (Eigen::Vector3i& face : mesh.faces)
  {
    command->mutable_mesh()->add_face(face(0));
    command->mutable_mesh()->add_face(face(1));
    command->mutable_mesh()->add_face(face(2));
  }
  for (Eigen::Vector2s& uv : mesh.uv)
  {
    command->mutable_mesh()->add_uv(uv(0));
    command->mutable_mesh()->add_uv(uv(1));
  }
  for (int i = 0; i < mesh.textures.size(); i++)
  {
    command->mutable_mesh()->add_texture(getStringCode(mesh.textures[i]));
    command->mutable_mesh()->add_texture_start(mesh.textureStartIndices[i]);
  }
  command->mutable_mesh()->add_data(mesh.scale(0));
  command->mutable_mesh()->add_data(mesh.scale(1));
  command->mutable_mesh()->add_data(mesh.scale(2));
  command->mutable_mesh()->add_data(mesh.pos(0));
  command->mutable_mesh()->add_data(mesh.pos(1));
  command->mutable_mesh()->add_data(mesh.pos(2));
  command->mutable_mesh()->add_data(mesh.euler(0));
  command->mutable_mesh()->add_data(mesh.euler(1));
  command->mutable_mesh()->add_data(mesh.euler(2));
  command->mutable_mesh()->add_data(mesh.color(0));
  command->mutable_mesh()->add_data(mesh.color(1));
  command->mutable_mesh()->add_data(mesh.color(2));
  command->mutable_mesh()->add_data(mesh.color(3));
  command->mutable_mesh()->set_cast_shadows(mesh.receiveShadows);
  command->mutable_mesh()->set_receive_shadows(mesh.receiveShadows);
}

void GUIStateMachine::encodeCreateTexture(
    proto::CommandList& list, Texture& texture)
{
  proto::Command* command = list.add_command();
  command->mutable_texture()->set_key(getStringCode(texture.key));
  command->mutable_texture()->set_base64(texture.base64);
}

void GUIStateMachine::encodeEnableMouseInteraction(
    proto::CommandList& list, const std::string& key)
{
  proto::Command* command = list.add_command();
  command->mutable_enable_mouse_interaction()->set_key(getStringCode(key));
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
  command->mutable_slider()->add_data(slider.min);
  command->mutable_slider()->add_data(slider.max);
  command->mutable_slider()->add_data(slider.value);
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
  command->mutable_plot()->add_bounds(plot.minX);
  command->mutable_plot()->add_bounds(plot.maxX);
  command->mutable_plot()->add_bounds(plot.minY);
  command->mutable_plot()->add_bounds(plot.maxY);
  for (s_t x : plot.xs)
  {
    command->mutable_plot()->add_xs(x);
  }
  for (s_t y : plot.ys)
  {
    command->mutable_plot()->add_ys(y);
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
  command->mutable_rich_plot()->add_bounds(plot.minX);
  command->mutable_rich_plot()->add_bounds(plot.maxX);
  command->mutable_rich_plot()->add_bounds(plot.minY);
  command->mutable_rich_plot()->add_bounds(plot.maxY);
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
    command->mutable_set_rich_plot_data()->add_xs(x);
  }
  for (s_t y : data.ys)
  {
    command->mutable_set_rich_plot_data()->add_ys(y);
  }
}

} // namespace server
} // namespace dart