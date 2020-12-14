#ifndef DART_MPC_REMOTE
#define DART_MPC_REMOTE

#include <functional>
#include <memory>
#include <vector>

#include "dart/proto/MPC.grpc.pb.h"
#include "dart/realtime/MPC.hpp"
#include "dart/realtime/MPCLocal.hpp"
#include "dart/realtime/RealTimeControlBuffer.hpp"

namespace grpc {
class Channel;
}

namespace dart {

namespace realtime {

class MPCRemote : public MPC
{
public:
  /// This connects to an MPC remote server
  MPCRemote(
      const std::string& host,
      int port,
      int dofs,
      int steps,
      int millisPerStep);

  /// This forks the process, starts a server on another process, and connects
  /// to it
  MPCRemote(MPCLocal& local);

  /// This gets the force to apply to the world at this instant. If we haven't
  /// computed anything for this instant yet, this just returns 0s.
  Eigen::VectorXd getForce(long now) override;

  /// This returns how many millis we have left until we've run out of plan.
  /// This can be a negative number, if we've run past our plan.
  long getRemainingPlanBufferMillis() override;

  /// This records the current state of the world based on some external sensing
  /// and inference. This resets the error in our model just assuming the world
  /// is exactly following our simulation.
  void recordGroundTruthState(
      long time,
      Eigen::VectorXd pos,
      Eigen::VectorXd vel,
      Eigen::VectorXd mass) override;

  /// This starts our main thread and begins running optimizations
  void start() override;

  /// This stops our main thread, waits for it to finish, and then returns
  void stop() override;

  /// This registers a listener to get called when we finish replanning
  void registerReplanningListener(
      std::function<void(long, const trajectory::TrajectoryRollout*, long)>
          replanListener) override;

protected:
  bool mRunning;
  std::shared_ptr<grpc::Channel> mChannel;
  std::unique_ptr<proto::MPCService::Stub> mStub;
  RealTimeControlBuffer mBuffer;
  std::thread mUpdateListenerThread;

  // These are listeners that get called when we finish replanning
  std::vector<
      std::function<void(long, const trajectory::TrajectoryRollout*, long)>>
      mReplannedListeners;
};

} // namespace realtime
} // namespace dart

#endif