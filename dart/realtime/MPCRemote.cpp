#include "dart/realtime/MPCRemote.hpp"

#include <grpcpp/grpcpp.h>
#include <sys/types.h>
#include <unistd.h>

#include "dart/proto/SerializeEigen.hpp"
#include "dart/realtime/MPCLocal.hpp"
#include "dart/realtime/Millis.hpp"
#include "dart/simulation/World.hpp"

namespace dart {
namespace realtime {

// RealTimeControlBuffer(int forceDim, int steps, int millisPerStep);

/// This connects to an MPC remote server
MPCRemote::MPCRemote(
    const std::string& host, int port, int dofs, int steps, int millisPerStep)
  : mChannel(grpc::CreateChannel(
      host + ":" + std::to_string(port), grpc::InsecureChannelCredentials())),
    mStub(proto::MPCService::NewStub(mChannel)),
    mBuffer(dofs, steps, millisPerStep),
    mRunning(false)
{
}

/// This forks the process, starts a server on another process, and connects
/// to it
MPCRemote::MPCRemote(MPCLocal& local)
  : mStub(nullptr),
    mBuffer(RealTimeControlBuffer(
        local.mWorld->getNumDofs(), local.mSteps, local.mMillisPerStep)),
    mRunning(false)
{
  int port = (rand() % 2000) + 2000;

  int original_id = getpid();
  int child_id = fork();
  // We're in the child process, boot a server to listen
  if (child_id == 0)
  {
    // Start a thread to periodically check if our parent has died, and if so
    // commit suicide
    std::thread parent_liveness_poll([&]() {
      while (true)
      {
        // Only poll once-per-second
        std::this_thread::sleep_for(std::chrono::seconds(1));
        int parent_id = getppid();
        // This means the parent is dead
        if (parent_id != original_id)
        {
          exit(0);
        }
      }
    });
    // Start a server on this thread
    local.serve(port);
    // When we're done serving, kill this process
    exit(0);
  }
  // We're in the parent process
  else if (child_id > 0)
  {
    std::cout << "(MPC fork process id = " << child_id << ")" << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    mChannel = grpc::CreateChannel(
        "localhost:" + std::to_string(port),
        grpc::InsecureChannelCredentials());

    mStub = proto::MPCService::NewStub(mChannel);
  }
}

/// This gets the force to apply to the world at this instant. If we haven't
/// computed anything for this instant yet, this just returns 0s.
Eigen::VectorXd MPCRemote::getForce(long now)
{
  return mBuffer.getPlannedForce(now);
}

/// This returns how many millis we have left until we've run out of plan.
/// This can be a negative number, if we've run past our plan.
long MPCRemote::getRemainingPlanBufferMillis()
{
  return mBuffer.getPlanBufferMillisAfter(timeSinceEpochMillis());
}

/// This records the current state of the world based on some external sensing
/// and inference. This resets the error in our model just assuming the world
/// is exactly following our simulation.
void MPCRemote::recordGroundTruthState(
    long time, Eigen::VectorXd pos, Eigen::VectorXd vel, Eigen::VectorXd mass)
{
  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  grpc::ClientContext context;

  proto::MPCRecordGroundTruthStateRequest request;
  request.set_time(time);
  proto::serializeVector(*request.mutable_pos(), pos);
  proto::serializeVector(*request.mutable_vel(), vel);
  proto::serializeVector(*request.mutable_mass(), mass);

  proto::MPCRecordGroundTruthStateReply reply;

  // The actual RPC.
  grpc::Status status
      = mStub->RecordGroundTruthState(&context, request, &reply);

  // Act upon its status.
  if (!status.ok())
  {
    std::cout << "gRPC got error: " << status.error_code() << ": "
              << status.error_message() << std::endl;
  }
}

/// This starts our main thread and begins running optimizations
void MPCRemote::start()
{
  if (mRunning)
    return;
  mRunning = true;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  grpc::ClientContext context;

  proto::MPCStartRequest request;
  request.set_clientclock(timeSinceEpochMillis());

  proto::MPCStartReply reply;

  // The actual RPC.
  grpc::Status status = mStub->Start(&context, request, &reply);

  // Act upon its status.
  if (!status.ok())
  {
    std::cout << "gRPC got error: " << status.error_code() << ": "
              << status.error_message() << std::endl;
  }

  // Start a thread to listen for updates
  mUpdateListenerThread = std::thread([&]() {
    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    grpc::ClientContext context;

    proto::MPCListenForUpdatesRequest request;

    // The actual RPC.
    std::unique_ptr<grpc::ClientReader<proto::MPCListenForUpdatesReply>> stream
        = mStub->ListenForUpdates(&context, request);

    proto::MPCListenForUpdatesReply reply;
    while (mRunning && stream->Read(&reply))
    {
      trajectory::TrajectoryRolloutReal rollout
          = trajectory::TrajectoryRollout::deserialize(reply.rollout());

      mBuffer.setForcePlan(
          reply.starttime(), timeSinceEpochMillis(), rollout.getForcesConst());

      for (auto listener : mReplannedListeners)
      {
        listener(reply.starttime(), &rollout, reply.replandurationmillis());
      }
    }
  });
}

/// This stops our main thread, waits for it to finish, and then returns
void MPCRemote::stop()
{
  if (!mRunning)
    return;
  mRunning = false;

  // Context for the client. It could be used to convey extra information to
  // the server and/or tweak certain RPC behaviors.
  grpc::ClientContext context;

  proto::MPCStopRequest request;
  request.set_clientclock(timeSinceEpochMillis());

  proto::MPCStopReply reply;

  // The actual RPC.
  grpc::Status status = mStub->Stop(&context, request, &reply);

  // Act upon its status.
  if (!status.ok())
  {
    std::cout << "gRPC got error: " << status.error_code() << ": "
              << status.error_message() << std::endl;
  }
}

/// This registers a listener to get called when we finish replanning
void MPCRemote::registerReplanningListener(
    std::function<void(long, const trajectory::TrajectoryRollout*, long)>
        replanListener)
{
  mReplannedListeners.push_back(replanListener);
}

} // namespace realtime
} // namespace dart