#include "dart/realtime/RealtimeWorld.hpp"

#include <chrono>
#include <cstdlib>

#include <json/json.h>

#include "dart/simulation/World.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

namespace dart {
namespace realtime {

RealtimeWorld::RealtimeWorld(
    std::shared_ptr<simulation::World> world,
    std::function<Eigen::VectorXd()> getForces,
    std::function<void(Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd)>
        recordState)
  : mWorld(world),
    mGetForces(getForces),
    mRecordState(recordState),
    mRunning(false),
    mIterCount(0),
    mServing(false),
    mMPCPlanWorld(world->clone())
{
}

RealtimeWorld::~RealtimeWorld()
{
  stopServing();
  stop();
}

void RealtimeWorld::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mMainThread = new std::thread(&RealtimeWorld::mainLoop, this);
}

void RealtimeWorld::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mMainThread->join();
  delete mMainThread;
}

/// This runs a local websocket server that you can connect to to receive live
/// updates
void RealtimeWorld::serve(int port)
{
  // Register signal and signal handler
  if (mServing)
  {
    std::cout << "Already serving!" << std::endl;
    return;
  }
  mServing = true;
  mServer = new WebsocketServer();
  mServerEventLoop = new asio::io_service();

  // Register our network callbacks, ensuring the logic is run on the main
  // thread's event loop
  mServer->connect([&](ClientConnection conn) {
    mServerEventLoop->post([&]() {
      std::clog << "Connection opened." << std::endl;
      std::clog << "There are now " << mServer->numConnections()
                << " open connections." << std::endl;
      // Send a hello message to the client
      mServer->send(
          conn, "{\"type\": \"init\", \"world\": " + mWorld->toJson() + "}");

      for (auto listener : mConnectionListeners)
      {
        listener();
      }
    });
  });
  mServer->disconnect([&](ClientConnection /* conn */) {
    mServerEventLoop->post([&]() {
      std::clog << "Connection closed." << std::endl;
      std::clog << "There are now " << mServer->numConnections()
                << " open connections." << std::endl;
    });
  });
  mServer->message([&](ClientConnection /* conn */, const Json::Value& args) {
    mServerEventLoop->post([args, this]() {
      if (args["type"].asString() == "keydown")
      {
        std::string key = args["key"].asString();
        for (auto listener : this->mKeydownListeners)
        {
          listener(key);
        }
        this->mKeysDown.insert(key);
      }
      if (args["type"].asString() == "keyup")
      {
        std::string key = args["key"].asString();
        for (auto listener : this->mKeyupListeners)
        {
          listener(key);
        }
        this->mKeysDown.erase(key);
      }
    });
  });

  // Start the networking thread
  mServerThread = new std::thread([&]() { mServer->run(port); });

  // Start the event loop for the main thread
  mServerWork = new asio::io_service::work(*mServerEventLoop);
  mServerMainThread = new std::thread([&]() { mServerEventLoop->run(); });
}

/// This kills the server, if one was running
void RealtimeWorld::stopServing()
{
  if (!mServing)
    return;
  mServing = false;
  mServerWork->get_io_service().stop();
  mServer->stop();
  mServerThread->join();
  mServerMainThread->join();
  delete mServer;
  delete mServerThread;
  delete mServerEventLoop;
  delete mServerWork;
  delete mServerMainThread;
}

/// This adds a listener that will get called when someone connects to the
/// server
void RealtimeWorld::registerConnectionListener(std::function<void()> listener)
{
  mConnectionListeners.push_back(listener);
}

/// This adds a listener that will get called when there is a key-down event
/// on the web client
void RealtimeWorld::registerKeydownListener(
    std::function<void(std::string)> listener)
{
  mKeydownListeners.push_back(listener);
}

/// This adds a listener that will get called when there is a key-up event
/// on the web client
void RealtimeWorld::registerKeyupListener(
    std::function<void(std::string)> listener)
{
  mKeyupListeners.push_back(listener);
}

/// This adds a listener that will get called right before every step(), that
/// can do whatever it likes to the world
void RealtimeWorld::registerPreStepListener(
    std::function<void(
        int,
        std::shared_ptr<simulation::World>,
        std::unordered_set<std::string>)> listener)
{
  mPreStepListeners.push_back(listener);
}

/// This sends a plan out to any web clients that may be watching, so that
/// they can render the MPC plan as it evolves.
void RealtimeWorld::displayMPCPlan(const trajectory::TrajectoryRollout* rollout)
{
  if (mServing)
  {
    mServer->broadcast(
        "{\"type\": \"new_plan\", \"plan\": " + rollout->toJson(mMPCPlanWorld)
        + "}");
  }
}

void RealtimeWorld::mainLoop()
{
  while (mRunning)
  {
    mIterCount++;

    int interval = (int)(mWorld->getTimeStep() * 1000);
    auto x = std::chrono::steady_clock::now()
             + std::chrono::milliseconds(interval);

    mWorld->setForces(mGetForces());

    for (auto listener : mPreStepListeners)
    {
      listener(mIterCount, mWorld, mKeysDown);
    }

    mWorld->step();

    mRecordState(
        mWorld->getPositions(), mWorld->getVelocities(), mWorld->getMasses());

    if (mServing)
    {
      mServer->broadcast(
          "{\"type\": \"update\", \"timestep\": " + std::to_string(mIterCount)
          + ", \"positions\": " + mWorld->positionsToJson()
          + ", \"colors\": " + mWorld->colorsToJson() + "}");
    }

    std::this_thread::sleep_until(x);
  }
}

} // namespace realtime
} // namespace dart