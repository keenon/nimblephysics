#include "dart/server/GUIStateMachine.hpp"

#include <chrono>
#include <fstream>
#include <sstream>

#include <assimp/scene.h>
#include <boost/filesystem.hpp>

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
#include "dart/server/RawJsonUtils.hpp"
#include "dart/server/external/base64/base64.h"
#include "dart/simulation/World.hpp"

namespace dart {
namespace server {

GUIStateMachine::GUIStateMachine() : mMessagesQueued(0)
{
  mJson << "[";
}

GUIStateMachine::~GUIStateMachine()
{
}

std::string GUIStateMachine::getCurrentStateAsJson()
{
  std::stringstream json;
  json << "[";
  bool isFirst = true;
  for (auto pair : mBoxes)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateBox(json, pair.second);
  }
  for (auto pair : mSpheres)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateSphere(json, pair.second);
  }
  for (auto pair : mCapsules)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateCapsule(json, pair.second);
  }
  for (auto pair : mLines)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateLine(json, pair.second);
  }
  for (auto pair : mTextures)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateTexture(json, pair.second);
  }
  for (auto pair : mMeshes)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateMesh(json, pair.second);
  }
  for (auto pair : mText)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateText(json, pair.second);
  }
  for (auto pair : mButtons)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateButton(json, pair.second);
  }
  for (auto pair : mSliders)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateSlider(json, pair.second);
  }
  for (auto pair : mPlots)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreatePlot(json, pair.second);
  }
  for (auto pair : mRichPlots)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeCreateRichPlot(json, pair.second);
    for (auto dataPair : pair.second.data)
    {
      json << ",";
      encodeSetRichPlotData(json, pair.second.key, dataPair.second);
    }
  }
  for (auto key : mMouseInteractionEnabled)
  {
    if (isFirst)
      isFirst = false;
    else
      json << ",";
    encodeEnableMouseInteraction(json, key);
  }

  json << "]";

  return json.str();
}

/// This formats the latest set of commands as JSON, and clears the buffer
std::string GUIStateMachine::flushJson()
{
  const std::lock_guard<std::recursive_mutex> lock(mJsonMutex);

  mJson << "]";
  std::string json = mJson.str();

  // Reset
  mMessagesQueued = 0;
  mJson = std::stringstream();
  mJson << "[";

  return json;
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
void GUIStateMachine::renderWorld(
    const std::shared_ptr<simulation::World>& world,
    const std::string& prefix,
    bool renderForces,
    bool renderForceMagnitudes)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    renderSkeleton(world->getSkeletonRef(i), prefix);
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
      createLine(prefix + "__contact_" + std::to_string(i) + "_a", points);
      std::vector<Eigen::Vector3s> pointsB;
      pointsB.push_back(contact.point);
      pointsB.push_back(contact.point - (contact.normal * scale));
      createLine(
          prefix + "__contact_" + std::to_string(i) + "_b",
          pointsB,
          Eigen::Vector4s(0, 1, 0, 1.0));
    }
  }
}

/// This is a high-level command that creates a basis
void GUIStateMachine::renderBasis(
    s_t scale,
    const std::string& prefix,
    const Eigen::Vector3s pos,
    const Eigen::Vector3s euler)
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
      prefix + "__basis_unitX", pointsX, Eigen::Vector4s(1.0, 0.0, 0.0, 1.0));
  createLine(
      prefix + "__basis_unitY", pointsY, Eigen::Vector4s(0.0, 1.0, 0.0, 1.0));
  createLine(
      prefix + "__basis_unitZ", pointsZ, Eigen::Vector4s(0.0, 0.0, 1.0, 1.0));
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
void GUIStateMachine::renderSkeleton(
    const std::shared_ptr<dynamics::Skeleton>& skel,
    const std::string& prefix,
    Eigen::Vector4s overrideColor)
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
    std::string prefix)
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
    createLine(pair.first, pair.second, colors[pair.first]);
  }
}

/// This is a high-level command that renders a wrench on a body node
void GUIStateMachine::renderBodyWrench(
    const dynamics::BodyNode* body,
    Eigen::Vector6s wrench,
    s_t scaleFactor,
    std::string prefix)
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
      Eigen::Vector4s(0.8, 0.8, 0.8, 1.0));
  createLine(
      prefix + "_" + body->getName() + "_force",
      forceLine,
      Eigen::Vector4s(1.0, 0.0, 0.0, 1.0));
}

/// This renders little velocity lines starting at every vertex in the passed
/// in body
void GUIStateMachine::renderMovingBodyNodeVertices(
    const dynamics::BodyNode* body, s_t scaleFactor, std::string prefix)
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
        Eigen::Vector4s(1.0, 0.0, 0.0, 1.0));
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

  queueCommand(
      [&](std::stringstream& json) { json << "{ \"type\": \"clear_all\" }"; });
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

/// This creates a box in the web GUI under a specified key
void GUIStateMachine::createBox(
    std::string key,
    const Eigen::Vector3s& size,
    const Eigen::Vector3s& pos,
    const Eigen::Vector3s& euler,
    const Eigen::Vector4s& color,
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
  box.castShadows = castShadows;
  box.receiveShadows = receiveShadows;

  queueCommand([this, key](std::stringstream& json) {
    encodeCreateBox(json, mBoxes[key]);
  });
}

/// This creates a sphere in the web GUI under a specified key
void GUIStateMachine::createSphere(
    std::string key,
    s_t radius,
    const Eigen::Vector3s& pos,
    const Eigen::Vector4s& color,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Sphere& sphere = mSpheres[key];
  sphere.key = key;
  sphere.radius = radius;
  sphere.pos = pos;
  sphere.color = color;
  sphere.castShadows = castShadows;
  sphere.receiveShadows = receiveShadows;

  queueCommand([this, key](std::stringstream& json) {
    encodeCreateSphere(json, mSpheres[key]);
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
  capsule.castShadows = castShadows;
  capsule.receiveShadows = receiveShadows;

  queueCommand([this, key](std::stringstream& json) {
    encodeCreateCapsule(json, mCapsules[key]);
  });
}

/// This creates a line in the web GUI under a specified key
void GUIStateMachine::createLine(
    std::string key,
    const std::vector<Eigen::Vector3s>& points,
    const Eigen::Vector4s& color)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Line& line = mLines[key];
  line.key = key;
  line.points = points;
  line.color = color;

  queueCommand([this, key](std::stringstream& json) {
    encodeCreateLine(json, mLines[key]);
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
  mesh.castShadows = castShadows;
  mesh.receiveShadows = receiveShadows;

  queueCommand([this, key](std::stringstream& json) {
    encodeCreateMesh(json, mMeshes[key]);
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

  queueCommand(
      [&](std::stringstream& json) { encodeCreateTexture(json, tex); });
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_object_pos\", \"key\": \"" << key
         << "\", \"pos\": ";
    vec3ToJson(json, pos);
    json << "}";
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_object_rotation\", \"key\": \"" << key
         << "\", \"euler\": ";
    vec3ToJson(json, euler);
    json << "}";
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_object_color\", \"key\": \"" << key
         << "\", \"color\": ";
    vec4ToJson(json, color);
    json << "}";
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_object_scale\", \"key\": \"" << key
         << "\", \"scale\": ";
    vec3ToJson(json, scale);
    json << "}";
  });
}

/// This sets an object to allow mouse interaction on the GUI
void GUIStateMachine::setObjectMouseInteractionEnabled(const std::string& key)
{
  mMouseInteractionEnabled.emplace(key);
  queueCommand([&](std::stringstream& json) {
    encodeEnableMouseInteraction(json, key);
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"delete_object\", \"key\": \"" << key << "\" }";
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
    const Eigen::Vector2i& size)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Text text;
  text.key = key;
  text.contents = contents;
  text.fromTopLeft = fromTopLeft;
  text.size = size;

  mText[key] = text;

  queueCommand([&](std::stringstream& json) { encodeCreateText(json, text); });
}

/// This changes the contents of text on the screen
void GUIStateMachine::setTextContents(
    const std::string& key, const std::string& newContents)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mText.find(key) != mText.end())
  {
    mText[key].contents = newContents;

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_text_contents\", \"key\": " << key
           << "\", \"label\": \"" << escapeJson(newContents) << "\" }";
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
    std::function<void()> onClick)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Button button;
  button.key = key;
  button.label = label;
  button.fromTopLeft = fromTopLeft;
  button.size = size;
  button.onClick = onClick;

  mButtons[key] = button;

  queueCommand(
      [&](std::stringstream& json) { encodeCreateButton(json, button); });
}

/// This changes the contents of text on the screen
void GUIStateMachine::setButtonLabel(
    const std::string& key, const std::string& newLabel)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mButtons.find(key) != mButtons.end())
  {
    mButtons[key].label = newLabel;

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_button_label\", \"key\": " << key
           << "\", \"label\": \"" << escapeJson(newLabel) << "\" }";
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
    std::function<void(s_t)> onChange)
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

  mSliders[key] = slider;

  queueCommand(
      [&](std::stringstream& json) { encodeCreateSlider(json, slider); });
}

/// This changes the contents of text on the screen
void GUIStateMachine::setSliderValue(const std::string& key, s_t value)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].value = value;

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_slider_value\", \"key\": " << key
           << "\", \"value\": " << numberToJson(value) << " }";
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

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_slider_min\", \"key\": " << key
           << "\", \"value\": " << numberToJson(min) << " }";
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

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_slider_max\", \"key\": " << key
           << "\", \"value\": " << numberToJson(max) << " }";
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
    const std::string& type)
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

  mPlots[key] = plot;

  queueCommand([&](std::stringstream& json) { encodeCreatePlot(json, plot); });
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

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_plot_data\", \"key\": " << key
           << "\", \"xs\": ";
      vecToJson(json, xs);
      json << ", \"ys\": ";
      vecToJson(json, ys);
      json << ", \"min_x\": " << numberToJson(minX);
      json << ", \"max_x\": " << numberToJson(maxX);
      json << ", \"min_y\": " << numberToJson(minY);
      json << ", \"max_y\": " << numberToJson(maxY);
      json << " }";
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
    const std::string& yAxisLabel)
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

  mRichPlots[key] = plot;

  queueCommand(
      [&](std::stringstream& json) { encodeCreateRichPlot(json, plot); });
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

    queueCommand([this, key, data](std::stringstream& json) {
      encodeSetRichPlotData(json, key, data);
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

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_rich_plot_bounds\", \"key\": " << key
           << "\", \"xs\": ";
      json << ", \"min_x\": " << numberToJson(minX);
      json << ", \"max_x\": " << numberToJson(maxX);
      json << ", \"min_y\": " << numberToJson(minY);
      json << ", \"max_y\": " << numberToJson(maxY);
      json << " }";
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_ui_elem_pos\", \"key\": " << key
         << "\", \"from_top_left\": ";
    vec2iToJson(json, fromTopLeft);
    json << " }";
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_ui_elem_size\", \"key\": " << key
         << "\", \"size\": ";
    vec2iToJson(json, size);
    json << " }";
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"delete_ui_elem\", \"key\": \"" << key << "\" }";
  });
}

void GUIStateMachine::queueCommand(
    std::function<void(std::stringstream&)> writeCommand)
{
  const std::lock_guard<std::recursive_mutex> lock(mJsonMutex);

  if (mMessagesQueued > 0)
  {
    mJson << ",";
  }
  mMessagesQueued++;
  writeCommand(mJson);
}

void GUIStateMachine::encodeCreateBox(std::stringstream& json, Box& box)
{
  json << "{ \"type\": \"create_box\", \"key\": \"" << box.key
       << "\", \"size\": ";
  vec3ToJson(json, box.size);
  json << ", \"pos\": ";
  vec3ToJson(json, box.pos);
  json << ", \"euler\": ";
  vec3ToJson(json, box.euler);
  json << ", \"color\": ";
  vec4ToJson(json, box.color);
  json << ", \"cast_shadows\": " << (box.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": " << (box.receiveShadows ? "true" : "false");
  json << "}";
}

void GUIStateMachine::encodeCreateSphere(
    std::stringstream& json, Sphere& sphere)
{
  json << "{ \"type\": \"create_sphere\", \"key\": \"" << sphere.key
       << "\", \"radius\": " << numberToJson(sphere.radius);
  json << ", \"pos\": ";
  vec3ToJson(json, sphere.pos);
  json << ", \"color\": ";
  vec4ToJson(json, sphere.color);
  json << ", \"cast_shadows\": " << (sphere.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": "
       << (sphere.receiveShadows ? "true" : "false");
  json << "}";
}

void GUIStateMachine::encodeCreateCapsule(
    std::stringstream& json, Capsule& capsule)
{
  json << "{ \"type\": \"create_capsule\", \"key\": \"" << capsule.key
       << "\", \"radius\": " << numberToJson(capsule.radius)
       << ", \"height\": " << numberToJson(capsule.height);
  json << ", \"pos\": ";
  vec3ToJson(json, capsule.pos);
  json << ", \"euler\": ";
  vec3ToJson(json, capsule.euler);
  json << ", \"color\": ";
  vec4ToJson(json, capsule.color);
  json << ", \"cast_shadows\": " << (capsule.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": "
       << (capsule.receiveShadows ? "true" : "false");
  json << "}";
}

void GUIStateMachine::encodeCreateLine(std::stringstream& json, Line& line)
{
  json << "{ \"type\": \"create_line\", \"key\": \"" << line.key;
  json << "\", \"points\": [";
  bool firstPoint = true;
  for (Eigen::Vector3s& point : line.points)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec3ToJson(json, point);
  }
  json << "], \"color\": ";
  vec4ToJson(json, line.color);
  json << "}";
}

void GUIStateMachine::encodeCreateMesh(std::stringstream& json, Mesh& mesh)
{
  json << "{ \"type\": \"create_mesh\", \"key\": \"" << mesh.key;
  json << "\", \"vertices\": [";
  bool firstPoint = true;
  for (Eigen::Vector3s& vertex : mesh.vertices)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec3ToJson(json, vertex);
  }
  json << "], \"vertex_normals\": [";
  firstPoint = true;
  for (Eigen::Vector3s& normal : mesh.vertexNormals)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec3ToJson(json, normal);
  }
  json << "], \"faces\": [";
  firstPoint = true;
  for (Eigen::Vector3i& face : mesh.faces)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec3iToJson(json, face);
  }
  json << "], \"uv\": [";
  firstPoint = true;
  for (Eigen::Vector2s& uv : mesh.uv)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec2dToJson(json, uv);
  }
  json << "], \"texture_starts\": [";
  firstPoint = true;
  for (int i = 0; i < mesh.textures.size(); i++)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    json << "{ \"key\": \"" << mesh.textures[i]
         << "\", \"start\": " << mesh.textureStartIndices[i] << "}";
  }
  json << "], \"color\": ";
  vec4ToJson(json, mesh.color);
  json << ", \"pos\": ";
  vec3ToJson(json, mesh.pos);
  json << ", \"euler\": ";
  vec3ToJson(json, mesh.euler);
  json << ", \"scale\": ";
  vec3ToJson(json, mesh.scale);
  json << ", \"cast_shadows\": " << (mesh.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": " << (mesh.receiveShadows ? "true" : "false");
  json << "}";
}

void GUIStateMachine::encodeCreateTexture(
    std::stringstream& json, Texture& texture)
{
  json << "{ \"type\": \"create_texture\", \"key\": \"" << texture.key;
  json << "\", \"base64\": \"" << texture.base64 << "\" }";
}

void GUIStateMachine::encodeEnableMouseInteraction(
    std::stringstream& json, const std::string& key)
{
  json << "{ \"type\": \"enable_mouse\", \"key\": \"" << key << "\" }";
}

void GUIStateMachine::encodeCreateText(std::stringstream& json, Text& text)
{
  json << "{ \"type\": \"create_text\", \"key\": \"" << text.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, text.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, text.size);
  json << ", \"contents\": \"" << escapeJson(text.contents);
  json << "\" }";
}

void GUIStateMachine::encodeCreateButton(
    std::stringstream& json, Button& button)
{
  json << "{ \"type\": \"create_button\", \"key\": \"" << button.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, button.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, button.size);
  json << ", \"label\": \"" << escapeJson(button.label);
  json << "\" }";
}

void GUIStateMachine::encodeCreateSlider(
    std::stringstream& json, Slider& slider)
{
  json << "{ \"type\": \"create_slider\", \"key\": \"" << slider.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, slider.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, slider.size);
  json << ", \"max\": " << numberToJson(slider.max);
  json << ", \"min\": " << numberToJson(slider.min);
  json << ", \"value\": " << numberToJson(slider.value);
  json << ", \"only_ints\": " << (slider.onlyInts ? "true" : "false");
  json << ", \"horizontal\": " << (slider.horizontal ? "true" : "false");
  json << "}";
}

void GUIStateMachine::encodeCreatePlot(std::stringstream& json, Plot& plot)
{
  json << "{ \"type\": \"create_plot\", \"key\": \"" << plot.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, plot.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, plot.size);
  json << ", \"max_x\": " << numberToJson(plot.maxX);
  json << ", \"min_x\": " << numberToJson(plot.minX);
  json << ", \"max_y\": " << numberToJson(plot.maxY);
  json << ", \"min_y\": " << numberToJson(plot.minY);
  json << ", \"xs\": ";
  vecToJson(json, plot.xs);
  json << ", \"ys\": ";
  vecToJson(json, plot.ys);
  json << ", \"plot_type\": \"" << plot.type;
  json << "\" }";
}

void GUIStateMachine::encodeCreateRichPlot(
    std::stringstream& json, RichPlot& plot)
{
  json << "{ \"type\": \"create_rich_plot\", \"key\": \"" << plot.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, plot.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, plot.size);
  json << ", \"max_x\": " << numberToJson(plot.maxX);
  json << ", \"min_x\": " << numberToJson(plot.minX);
  json << ", \"max_y\": " << numberToJson(plot.maxY);
  json << ", \"min_y\": " << numberToJson(plot.minY);
  json << ", \"title\": \"";
  json << plot.title;
  json << "\", \"x_axis_label\": \"";
  json << plot.xAxisLabel;
  json << "\", \"y_axis_label\": \"";
  json << plot.yAxisLabel;
  json << "\" }";
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
    std::stringstream& json,
    const std::string& plotKey,
    const RichPlotData& data)
{
  json << "{ \"type\": \"set_rich_plot_data\", \"key\": \"" << plotKey
       << "\", \"name\": \"";
  json << data.name;
  json << "\", \"color\": \"";
  json << data.color;
  json << "\", \"plot_type\": \"";
  json << data.type;
  json << "\", \"xs\": ";
  vecToJson(json, data.xs);
  json << ", \"ys\": ";
  vecToJson(json, data.ys);
  json << "}";
}

} // namespace server
} // namespace dart