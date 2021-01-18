#include "dart/server/GUIWebsocketServer.hpp"

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

GUIWebsocketServer::GUIWebsocketServer()
  : mServing(false),
    mMessagesQueued(0),
    mAutoflush(true),
    mScreenSize(Eigen::Vector2i(680, 420)),
    mPort(-1)
{
  mJson << "[";
}

GUIWebsocketServer::~GUIWebsocketServer()
{
  if (mServing)
  {
    dterr
        << "GUIWebsocketServer is being deallocated while it's still "
           "serving! The server will now terminate, and attempt to clean up. "
           "If this was not intended "
           "behavior, please keep a reference to the GUIWebsocketServer to "
           "keep the server alive. If this was intended behavior, please call "
           "stopServing() on "
           "the server before deallocating it."
        << std::endl;
    stopServing();
  }
}

/// This is a non-blocking call to start a websocket server on a given port
void GUIWebsocketServer::serve(int port)
{
  mPort = port;
  // Register signal and signal handler
  if (mServing)
  {
    std::cout << "Errer in GUIWebsocketServer::serve()! Already serving. "
                 "Ignoring request."
              << std::endl;
    return;
  }
  mServing = true;
  mServer = new WebsocketServer();

  // Register our network callbacks, ensuring the logic is run on the main
  // thread's event loop
  mServer->connect([this](ClientConnection conn) {
    // We don't need high throughput, so run everything through a global mutex
    // to avoid data races
    const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

    // Send a hello message to the client
    // mServer->send(conn) seems to break, cause conn appears to get cleaned
    // up in race conditions (it's a weak pointer)
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
    for (auto key : mMouseInteractionEnabled)
    {
      if (isFirst)
        isFirst = false;
      else
        json << ",";
      encodeEnableMouseInteraction(json, key);
    }

    json << "]";

    // std::cout << json.str() << std::endl;

    std::string jsonStr = json.str();
    try
    {
      mServer->send(conn, jsonStr);
    }
    catch (...)
    {
      dterr << "GUIWebsocketServer caught an error broadcasting message \""
            << jsonStr << "\"" << std::endl;
    }
    // mServer->broadcast("{\"type\": 1}");
    /*
    mServer->broadcast(
        "{\"type\": \"init\", \"world\": " + mWorld->toJson() + "}");
    */

    for (auto listener : mConnectionListeners)
    {
      listener();
    }
  });

  mServer->disconnect([this](ClientConnection /* conn */) {
    std::clog << "Connection closed." << std::endl;
    std::clog << "There are now " << mServer->numConnections()
              << " open connections." << std::endl;
  });
  mServer->message([this](
                       ClientConnection /* conn */, const Json::Value& args) {
    if (args["type"].asString() == "keydown")
    {
      std::string key = args["key"].asString();
      {
        const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);
        this->mKeysDown.insert(key);
      }
      for (auto listener : this->mKeydownListeners)
      {
        listener(key);
      }
    }
    else if (args["type"].asString() == "keyup")
    {
      std::string key = args["key"].asString();
      {
        const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);
        this->mKeysDown.erase(key);
      }
      for (auto listener : this->mKeyupListeners)
      {
        listener(key);
      }
    }
    else if (args["type"].asString() == "button_click")
    {
      std::string key = args["key"].asString();
      if (mButtons.find(key) != mButtons.end())
      {
        mButtons[key].onClick();
      }
    }
    else if (args["type"].asString() == "slider_set_value")
    {
      std::string key = args["key"].asString();
      double value = args["value"].asDouble();
      if (mSliders.find(key) != mSliders.end())
      {
        mSliders[key].value = value;
        mSliders[key].onChange(value);
      }
    }
    else if (args["type"].asString() == "screen_resize")
    {
      Eigen::Vector2i size
          = Eigen::Vector2i(args["size"][0].asInt(), args["size"][1].asInt());
      mScreenSize = size;

      for (auto handler : mScreenResizeListeners)
      {
        handler(size);
      }
    }
    else if (args["type"].asString() == "drag")
    {
      std::string key = args["key"].asString();
      Eigen::Vector3d pos = Eigen::Vector3d(
          args["pos"][0].asDouble(),
          args["pos"][1].asDouble(),
          args["pos"][2].asDouble());

      for (auto handler : mDragListeners[key])
      {
        handler(pos);
      }
    }
  });

  // unblock signals in this thread
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_UNBLOCK, &sigset, nullptr);

  /*
  // The signal set is used to register termination notifications
  mSignalSet = new asio::signal_set(mServerEventLoop, SIGINT, SIGTERM);

  // register the handle_stop callback
  mSignalSet->async_wait([&](asio::error_code const& error, int signal_number) {
    if (error == asio::error::operation_aborted)
    {
      std::cout << "Signal listener was terminated by asio" << std::endl;
    }
    else if (error)
    {
      std::cout << "Got an error registering termination signals: " << error
                << std::endl;
    }
    else if (
        signal_number == SIGINT || signal_number == SIGTERM
        || signal_number == SIGQUIT)
    {
      std::cout << "Shutting down the server..." << std::endl;
      stopServing();
      mServerEventLoop.stop();
      exit(signal_number);
    }
  });
  */

  // Start the networking thread
  mServerThread = new std::thread([this, port]() {
    /*
    // block signals in this thread and subsequently
    // spawned threads so they're guaranteed to go to the main thread
    sigset_t sigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigset, nullptr);
    */

    std::cout << "GUIWebsocketServer will start serving a WebSocket server on "
                 "ws://localhost:"
              << port << std::endl;
    bool success = mServer->run(port);
    if (!success)
    {
      // This means we failed to bind to the port
      stopServing();
    }
  });
}

/// This kills the server, if one was running
void GUIWebsocketServer::stopServing()
{
  if (!mServing)
    return;
  std::cout << "GUIWebsocketServer is shutting down the WebSocket server on "
               "ws://localhost:"
            << mPort << std::endl;
  mServer->stop();
  mServerThread->join();
  delete mServer;
  delete mServerThread;
  mServing = false;
}

/// Returns true if we're serving
bool GUIWebsocketServer::isServing()
{
  return mServing;
}

/// This adds a listener that will get called when someone connects to the
/// server
void GUIWebsocketServer::registerConnectionListener(
    std::function<void()> listener)
{
  mConnectionListeners.push_back(listener);
}

/// This adds a listener that will get called when ctrl+C is pressed
void GUIWebsocketServer::registerShutdownListener(
    std::function<void()> listener)
{
  mShutdownListeners.push_back(listener);
}

/// This adds a listener that will get called when there is a key-down event
/// on the web client
void GUIWebsocketServer::registerKeydownListener(
    std::function<void(std::string)> listener)
{
  mKeydownListeners.push_back(listener);
}

/// This adds a listener that will get called when there is a key-up event
/// on the web client
void GUIWebsocketServer::registerKeyupListener(
    std::function<void(std::string)> listener)
{
  mKeyupListeners.push_back(listener);
}

/// Gets the set of all the keys currently being pressed
const std::unordered_set<std::string>& GUIWebsocketServer::getKeysDown() const
{
  return mKeysDown;
}

/// This tells us whether or not to automatically flush after each command
void GUIWebsocketServer::setAutoflush(bool autoflush)
{
  mAutoflush = autoflush;
}

/// This sends the current list of commands to the web GUI
void GUIWebsocketServer::flush()
{
  const std::lock_guard<std::recursive_mutex> lock(mJsonMutex);

  mJson << "]";
  std::string json = mJson.str();
  if (mServing)
  {
    try
    {
      mServer->broadcast(json);
    }
    catch (...)
    {
      dterr << "GUIWebsocketServer caught an error broadcasting message \""
            << json << "\"" << std::endl;
    }
  }

  // Reset
  mMessagesQueued = 0;
  /*
  mJson.str(std::string());
  mJson.clear();
  */
  mJson = std::stringstream();
  mJson << "[";
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
GUIWebsocketServer& GUIWebsocketServer::renderWorld(
    const std::shared_ptr<simulation::World>& world,
    const std::string& prefix,
    bool renderForces,
    bool renderForceMagnitudes)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  bool oldAutoflush = mAutoflush;
  mAutoflush = false;

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
      double scale = renderForceMagnitudes ? contact.lcpResult * 10 : 2;
      std::vector<Eigen::Vector3d> points;
      points.push_back(contact.point);
      points.push_back(contact.point + (contact.normal * scale));
      createLine(prefix + "__contact_" + std::to_string(i) + "_a", points);
      std::vector<Eigen::Vector3d> pointsB;
      pointsB.push_back(contact.point);
      pointsB.push_back(contact.point - (contact.normal * scale));
      createLine(
          prefix + "__contact_" + std::to_string(i) + "_b",
          pointsB,
          Eigen::Vector3d(0, 1, 0));
    }
  }

  mAutoflush = oldAutoflush;
  if (mAutoflush)
  {
    flush();
  }
  return *this;
}

/// This is a high-level command that creates a basis
GUIWebsocketServer& GUIWebsocketServer::renderBasis(
    double scale,
    const std::string& prefix,
    const Eigen::Vector3d pos,
    const Eigen::Vector3d euler)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = pos;
  T.linear() = math::eulerXYZToMatrix(euler);

  std::vector<Eigen::Vector3d> pointsX;
  pointsX.push_back(T * Eigen::Vector3d::Zero());
  pointsX.push_back(T * (Eigen::Vector3d::UnitX() * scale));
  std::vector<Eigen::Vector3d> pointsY;
  pointsY.push_back(T * Eigen::Vector3d::Zero());
  pointsY.push_back(T * (Eigen::Vector3d::UnitY() * scale));
  std::vector<Eigen::Vector3d> pointsZ;
  pointsZ.push_back(T * Eigen::Vector3d::Zero());
  pointsZ.push_back(T * (Eigen::Vector3d::UnitZ() * scale));

  deleteObjectsByPrefix(prefix + "__basis_");
  createLine(prefix + "__basis_unitX", pointsX, Eigen::Vector3d::UnitX());
  createLine(prefix + "__basis_unitY", pointsY, Eigen::Vector3d::UnitY());
  createLine(prefix + "__basis_unitZ", pointsZ, Eigen::Vector3d::UnitZ());
}

/// This is a high-level command that creates/updates all the shapes in a
/// world by calling the lower-level commands
GUIWebsocketServer& GUIWebsocketServer::renderSkeleton(
    const std::shared_ptr<dynamics::Skeleton>& skel, const std::string& prefix)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  bool oldAutoflush = mAutoflush;
  mAutoflush = false;

  for (int j = 0; j < skel->getNumBodyNodes(); j++)
  {
    dynamics::BodyNode* node = skel->getBodyNode(j);
    if (node == nullptr)
    {
      std::cout << "ERROR! GUIWebsocketServer found a null body node! This "
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
                visual->getColor(),
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
                visual->getColor(),
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
                visual->getColor(),
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
                visual->getColor(),
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
                visual->getColor(),
                visual->getCastShadows(),
                visual->getReceiveShadows());
          }
          else
          {
            dterr
                << "[GUIWebsocketServer.renderSkeleton()] Attempting to render "
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
          Eigen::Vector3d pos = shapeNode->getWorldTransform().translation();
          Eigen::Vector3d euler
              = math::matrixToEulerXYZ(shapeNode->getWorldTransform().linear());
          Eigen::Vector3d color = visual->getColor();
          // std::cout << "Color " << shapeName << ":" << color << std::endl;

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

  mAutoflush = oldAutoflush;
  if (mAutoflush)
  {
    flush();
  }

  return *this;
}

/// This is a high-level command that renders a given trajectory as a bunch of
/// lines in the world, one per body
GUIWebsocketServer& GUIWebsocketServer::renderTrajectoryLines(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXd positions,
    std::string prefix)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  assert(positions.rows() == world->getNumDofs());

  bool oldAutoflush = mAutoflush;
  mAutoflush = false;

  std::unordered_map<std::string, std::vector<Eigen::Vector3d>> paths;
  std::unordered_map<std::string, Eigen::Vector3d> colors;

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
          colors[shapeName] = visual->getColor();
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

  flush();
  mAutoflush = oldAutoflush;
  return *this;
}

/// This completely resets the web GUI, deleting all objects, UI elements, and
/// listeners
GUIWebsocketServer& GUIWebsocketServer::clear()
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
  mScreenResizeListeners.clear();
  mKeydownListeners.clear();
  mShutdownListeners.clear();
  return *this;
}

/// This creates a box in the web GUI under a specified key
GUIWebsocketServer& GUIWebsocketServer::createBox(
    std::string key,
    const Eigen::Vector3d& size,
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& euler,
    const Eigen::Vector3d& color,
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

  return *this;
}

/// This creates a sphere in the web GUI under a specified key
GUIWebsocketServer& GUIWebsocketServer::createSphere(
    std::string key,
    double radius,
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& color,
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

  return *this;
}

/// This creates a capsule in the web GUI under a specified key
GUIWebsocketServer& GUIWebsocketServer::createCapsule(
    std::string key,
    double radius,
    double height,
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& euler,
    const Eigen::Vector3d& color,
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

  return *this;
}

/// This creates a line in the web GUI under a specified key
GUIWebsocketServer& GUIWebsocketServer::createLine(
    std::string key,
    const std::vector<Eigen::Vector3d>& points,
    const Eigen::Vector3d& color)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Line& line = mLines[key];
  line.key = key;
  line.points = points;
  line.color = color;

  queueCommand([this, key](std::stringstream& json) {
    encodeCreateLine(json, mLines[key]);
  });

  return *this;
}

/// This creates a mesh in the web GUI under a specified key, using raw shape
/// data
GUIWebsocketServer& GUIWebsocketServer::createMesh(
    std::string key,
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3d>& vertexNormals,
    const std::vector<Eigen::Vector3i>& faces,
    const std::vector<Eigen::Vector2d>& uv,
    const std::vector<std::string>& textures,
    const std::vector<int>& textureStartIndices,
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& euler,
    const Eigen::Vector3d& scale,
    const Eigen::Vector3d& color,
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

  return *this;
}

/// This creates a mesh in the web GUI under a specified key, from the ASSIMP
/// mesh
GUIWebsocketServer& GUIWebsocketServer::createMeshASSIMP(
    const std::string& key,
    const aiScene* mesh,
    const std::string& meshPath,
    const Eigen::Vector3d& pos,
    const Eigen::Vector3d& euler,
    const Eigen::Vector3d& scale,
    const Eigen::Vector3d& color,
    bool castShadows,
    bool receiveShadows)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  std::vector<Eigen::Vector3d> vertices;
  std::vector<Eigen::Vector3d> vertexNormals;
  std::vector<Eigen::Vector3i> faces;
  std::vector<Eigen::Vector2d> uv;
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

  return *this;
}

/// This creates a texture object, to be sent to the web frontend
GUIWebsocketServer& GUIWebsocketServer::createTexture(
    const std::string& key, const std::string& base64)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  Texture tex;
  tex.key = key;
  tex.base64 = base64;

  mTextures[key] = tex;

  queueCommand(
      [&](std::stringstream& json) { encodeCreateTexture(json, tex); });

  return *this;
}

/// This creates a texture object by loading it from a file
GUIWebsocketServer& GUIWebsocketServer::createTextureFromFile(
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
  return *this;
}

/// This returns true if we've already got an object with the key "key"
bool GUIWebsocketServer::hasObject(const std::string& key)
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
/// line). Otherwise it returns Vector3d::Zero().
Eigen::Vector3d GUIWebsocketServer::getObjectPosition(const std::string& key)
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
  return Eigen::Vector3d::Zero();
}

/// This returns the rotation of an object, if we've got it (and it's not a
/// line or a sphere). Otherwise it returns Vector3d::Zero().
Eigen::Vector3d GUIWebsocketServer::getObjectRotation(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mBoxes.find(key) != mBoxes.end())
    return mBoxes[key].euler;
  if (mCapsules.find(key) != mCapsules.end())
    return mCapsules[key].euler;
  if (mMeshes.find(key) != mMeshes.end())
    return mMeshes[key].euler;
  return Eigen::Vector3d::Zero();
}

/// This returns the color of an object, if we've got it. Otherwise it returns
/// Vector3d::Zero().
Eigen::Vector3d GUIWebsocketServer::getObjectColor(const std::string& key)
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
  return Eigen::Vector3d::Zero();
}

/// This moves an object (e.g. box, sphere, line) to a specified position
GUIWebsocketServer& GUIWebsocketServer::setObjectPosition(
    const std::string& key, const Eigen::Vector3d& pos)
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

  return *this;
}

/// This moves an object (e.g. box, sphere, line) to a specified orientation
GUIWebsocketServer& GUIWebsocketServer::setObjectRotation(
    const std::string& key, const Eigen::Vector3d& euler)
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

  return *this;
}

/// This changes an object (e.g. box, sphere, line) color
GUIWebsocketServer& GUIWebsocketServer::setObjectColor(
    const std::string& key, const Eigen::Vector3d& color)
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
    vec3ToJson(json, color);
    json << "}";
  });

  return *this;
}

/// This enables mouse events on an object (if they're not already), and calls
/// "listener" whenever the object is dragged with the desired drag
/// coordinates
GUIWebsocketServer& GUIWebsocketServer::registerDragListener(
    const std::string& key, std::function<void(Eigen::Vector3d)> listener)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mMouseInteractionEnabled.emplace(key);
  queueCommand([&](std::stringstream& json) {
    encodeEnableMouseInteraction(json, key);
  });
  mDragListeners[key].push_back(listener);
  return *this;
}

/// This deletes an object by key
GUIWebsocketServer& GUIWebsocketServer::deleteObject(const std::string& key)
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

  return *this;
}

/// This deletes all the objects that match a given prefix
GUIWebsocketServer& GUIWebsocketServer::deleteObjectsByPrefix(
    const std::string& prefix)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  bool oldAutoflush = mAutoflush;
  mAutoflush = false;

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

  mAutoflush = oldAutoflush;
  if (mAutoflush)
    flush();

  return *this;
}

/// This gets the current screen size
Eigen::Vector2i GUIWebsocketServer::getScreenSize()
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  return mScreenSize;
}

/// This registers a callback to get called whenever the screen size changes.
void GUIWebsocketServer::registerScreenResizeListener(
    std::function<void(Eigen::Vector2i)> listener)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mScreenResizeListeners.push_back(listener);
}

/// This places some text on the screen at the specified coordinates
GUIWebsocketServer& GUIWebsocketServer::createText(
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

  return *this;
}

/// This changes the contents of text on the screen
GUIWebsocketServer& GUIWebsocketServer::setTextContents(
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

  return *this;
}

/// This places a clickable button on the screen at the specified coordinates
GUIWebsocketServer& GUIWebsocketServer::createButton(
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

  return *this;
}

/// This changes the contents of text on the screen
GUIWebsocketServer& GUIWebsocketServer::setButtonLabel(
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

  return *this;
}

/// This creates a slider
GUIWebsocketServer& GUIWebsocketServer::createSlider(
    const std::string& key,
    const Eigen::Vector2i& fromTopLeft,
    const Eigen::Vector2i& size,
    double min,
    double max,
    double value,
    bool onlyInts,
    bool horizontal,
    std::function<void(double)> onChange)
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

  return *this;
}

/// This changes the contents of text on the screen
GUIWebsocketServer& GUIWebsocketServer::setSliderValue(
    const std::string& key, double value)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].value = value;

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_slider_value\", \"key\": " << key
           << "\", \"value\": " << value << " }";
    });
  }
  else
  {
    std::cout
        << "Tried to setSliderValue() for a key (" << key
        << ") that doesn't exist as a Slider object. Call createSlider() first."
        << std::endl;
  }

  return *this;
}

/// This changes the contents of text on the screen
GUIWebsocketServer& GUIWebsocketServer::setSliderMin(
    const std::string& key, double min)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].min = min;

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_slider_min\", \"key\": " << key
           << "\", \"value\": " << min << " }";
    });
  }
  else
  {
    std::cout
        << "Tried to setSliderMin() for a key (" << key
        << ") that doesn't exist as a Slider object. Call createSlider() first."
        << std::endl;
  }

  return *this;
}

/// This changes the contents of text on the screen
GUIWebsocketServer& GUIWebsocketServer::setSliderMax(
    const std::string& key, double max)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  if (mSliders.find(key) != mSliders.end())
  {
    mSliders[key].max = max;

    queueCommand([&](std::stringstream& json) {
      json << "{ \"type\": \"set_slider_max\", \"key\": " << key
           << "\", \"value\": " << max << " }";
    });
  }
  else
  {
    std::cout
        << "Tried to setSliderMax() for a key (" << key
        << ") that doesn't exist as a Slider object. Call createSlider() first."
        << std::endl;
  }

  return *this;
}

/// This creates a plot to display data on the GUI
GUIWebsocketServer& GUIWebsocketServer::createPlot(
    const std::string& key,
    const Eigen::Vector2i& fromTopLeft,
    const Eigen::Vector2i& size,
    const std::vector<double>& xs,
    double minX,
    double maxX,
    const std::vector<double>& ys,
    double minY,
    double maxY,
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

  return *this;
}

/// This changes the contents of a plot, along with its display limits
GUIWebsocketServer& GUIWebsocketServer::setPlotData(
    const std::string& key,
    const std::vector<double>& xs,
    double minX,
    double maxX,
    const std::vector<double>& ys,
    double minY,
    double maxY)
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
      json << ", \"min_x\": " << minX;
      json << ", \"max_x\": " << maxX;
      json << ", \"min_y\": " << minY;
      json << ", \"max_y\": " << maxY;
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

/// This moves a UI element on the screen
GUIWebsocketServer& GUIWebsocketServer::setUIElementPosition(
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
  return *this;
}

/// This changes the size of a UI element
GUIWebsocketServer& GUIWebsocketServer::setUIElementSize(
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

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"set_ui_elem_size\", \"key\": " << key
         << "\", \"size\": ";
    vec2iToJson(json, size);
    json << " }";
  });
  return *this;
}

/// This deletes a UI element by key
GUIWebsocketServer& GUIWebsocketServer::deleteUIElement(const std::string& key)
{
  const std::lock_guard<std::recursive_mutex> lock(this->globalMutex);

  mText.erase(key);
  mButtons.erase(key);
  mSliders.erase(key);
  mPlots.erase(key);

  queueCommand([&](std::stringstream& json) {
    json << "{ \"type\": \"delete_ui_elem\", \"key\": \"" << key << "\" }";
  });

  return *this;
}

void GUIWebsocketServer::queueCommand(
    std::function<void(std::stringstream&)> writeCommand)
{
  const std::lock_guard<std::recursive_mutex> lock(mJsonMutex);

  if (mMessagesQueued > 0)
  {
    mJson << ",";
  }
  mMessagesQueued++;
  writeCommand(mJson);

  if (mAutoflush)
  {
    // Our lock is re-entrant, so it'll be allowed to be acquired again during
    // flushing
    flush();
  }
}

void GUIWebsocketServer::encodeCreateBox(std::stringstream& json, Box& box)
{
  json << "{ \"type\": \"create_box\", \"key\": \"" << box.key
       << "\", \"size\": ";
  vec3ToJson(json, box.size);
  json << ", \"pos\": ";
  vec3ToJson(json, box.pos);
  json << ", \"euler\": ";
  vec3ToJson(json, box.euler);
  json << ", \"color\": ";
  vec3ToJson(json, box.color);
  json << ", \"cast_shadows\": " << (box.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": " << (box.receiveShadows ? "true" : "false");
  json << "}";
}

void GUIWebsocketServer::encodeCreateSphere(
    std::stringstream& json, Sphere& sphere)
{
  json << "{ \"type\": \"create_sphere\", \"key\": \"" << sphere.key
       << "\", \"radius\": " << sphere.radius;
  json << ", \"pos\": ";
  vec3ToJson(json, sphere.pos);
  json << ", \"color\": ";
  vec3ToJson(json, sphere.color);
  json << ", \"cast_shadows\": " << (sphere.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": "
       << (sphere.receiveShadows ? "true" : "false");
  json << "}";
}

void GUIWebsocketServer::encodeCreateCapsule(
    std::stringstream& json, Capsule& capsule)
{
  json << "{ \"type\": \"create_capsule\", \"key\": \"" << capsule.key
       << "\", \"radius\": " << capsule.radius
       << ", \"height\": " << capsule.height;
  json << ", \"pos\": ";
  vec3ToJson(json, capsule.pos);
  json << ", \"euler\": ";
  vec3ToJson(json, capsule.euler);
  json << ", \"color\": ";
  vec3ToJson(json, capsule.color);
  json << ", \"cast_shadows\": " << (capsule.castShadows ? "true" : "false");
  json << ", \"receive_shadows\": "
       << (capsule.receiveShadows ? "true" : "false");
  json << "}";
}

void GUIWebsocketServer::encodeCreateLine(std::stringstream& json, Line& line)
{
  json << "{ \"type\": \"create_line\", \"key\": \"" << line.key;
  json << "\", \"points\": [";
  bool firstPoint = true;
  for (Eigen::Vector3d& point : line.points)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec3ToJson(json, point);
  }
  json << "], \"color\": ";
  vec3ToJson(json, line.color);
  json << "}";
}

void GUIWebsocketServer::encodeCreateMesh(std::stringstream& json, Mesh& mesh)
{
  json << "{ \"type\": \"create_mesh\", \"key\": \"" << mesh.key;
  json << "\", \"vertices\": [";
  bool firstPoint = true;
  for (Eigen::Vector3d& vertex : mesh.vertices)
  {
    if (firstPoint)
      firstPoint = false;
    else
      json << ", ";
    vec3ToJson(json, vertex);
  }
  json << "], \"vertex_normals\": [";
  firstPoint = true;
  for (Eigen::Vector3d& normal : mesh.vertexNormals)
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
  for (Eigen::Vector2d& uv : mesh.uv)
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
  vec3ToJson(json, mesh.color);
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

void GUIWebsocketServer::encodeCreateTexture(
    std::stringstream& json, Texture& texture)
{
  json << "{ \"type\": \"create_texture\", \"key\": \"" << texture.key;
  json << "\", \"base64\": \"" << texture.base64 << "\" }";
}

void GUIWebsocketServer::encodeEnableMouseInteraction(
    std::stringstream& json, const std::string& key)
{
  json << "{ \"type\": \"enable_mouse\", \"key\": \"" << key << "\" }";
}

void GUIWebsocketServer::encodeCreateText(std::stringstream& json, Text& text)
{
  json << "{ \"type\": \"create_text\", \"key\": \"" << text.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, text.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, text.size);
  json << ", \"contents\": \"" << escapeJson(text.contents);
  json << "\" }";
}

void GUIWebsocketServer::encodeCreateButton(
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

void GUIWebsocketServer::encodeCreateSlider(
    std::stringstream& json, Slider& slider)
{
  json << "{ \"type\": \"create_slider\", \"key\": \"" << slider.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, slider.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, slider.size);
  json << ", \"max\": " << slider.max;
  json << ", \"min\": " << slider.min;
  json << ", \"value\": " << slider.value;
  json << ", \"only_ints\": " << slider.onlyInts ? "true" : "false";
  json << ", \"horizontal\": " << slider.horizontal ? "true" : "false";
  json << "}";
}

void GUIWebsocketServer::encodeCreatePlot(std::stringstream& json, Plot& plot)
{
  json << "{ \"type\": \"create_plot\", \"key\": \"" << plot.key
       << "\", \"from_top_left\": ";
  vec2iToJson(json, plot.fromTopLeft);
  json << ", \"size\": ";
  vec2iToJson(json, plot.size);
  json << ", \"max_x\": " << plot.maxX;
  json << ", \"min_x\": " << plot.minX;
  json << ", \"max_y\": " << plot.maxY;
  json << ", \"min_y\": " << plot.minY;
  json << ", \"xs\": ";
  vecToJson(json, plot.xs);
  json << ", \"ys\": ";
  vecToJson(json, plot.ys);
  json << ", \"plot_type\": \"" << plot.type;
  json << "\" }";
}

} // namespace server
} // namespace dart