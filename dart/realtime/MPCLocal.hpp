#ifndef DART_REALTIME_MPCLocal
#define DART_REALTIME_MPCLocal

#include <memory>
#include <thread>

#include <Eigen/Dense>

#include "dart/proto/MPC.grpc.pb.h"
#include "dart/realtime/MPC.hpp"
#include "dart/realtime/RealTimeControlBuffer.hpp"

using namespace grpc;

namespace dart {
namespace simulation {
class World;
}

namespace trajectory {
class LossFn;
class Solution;
class Problem;
class Optimizer;
class TrajectoryRollout;
} // namespace trajectory

namespace realtime {

class MPCLocal final : public MPC
{

  friend class MPCRemote;

public:
  MPCLocal(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<trajectory::LossFn> loss,
      int planningHorizonMillis);

  /// Copy constructor
  MPCLocal(const MPCLocal& mpc);

  /// This updates the loss function that we're going to move in real time to
  /// minimize. This can happen quite frequently, for example if our loss
  /// function is to track a mouse pointer in a simulated environment, we may
  /// reset the loss function every time the mouse moves.
  void setLoss(std::shared_ptr<trajectory::LossFn> loss);

  /// This sets the optimizer that MPCLocal will use. This will override the
  /// default optimizer. This should be called before start().
  void setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer);

  /// This returns the current optimizer that MPCLocal is using
  std::shared_ptr<trajectory::Optimizer> getOptimizer();

  /// This sets the problem that MPCLocal will use. This will override the
  /// default problem. This should be called before start().
  void setProblem(std::shared_ptr<trajectory::Problem> problem);

  /// This returns the current problem definition that MPCLocal is using
  std::shared_ptr<trajectory::Problem> getProblem();

  /// This gets the force to apply to the world at this instant. If we haven't
  /// computed anything for this instant yet, this just returns 0s.
  Eigen::VectorXd getForce(long now) override;

  /// This returns how many millis we have left until we've run out of plan.
  /// This can be a negative number, if we've run past our plan.
  long getRemainingPlanBufferMillis() override;

  /// This can completely silence log output
  void setSilent(bool silent);

  /// This enables linesearch on the IPOPT sub-problems. Defaults to true. This
  /// increases the stability of solutions, but can lead to spikes in solution
  /// times.
  void setEnableLineSearch(bool enabled);

  /// This enables "guards" on the IPOPT sub-problems. Defaults to false. This
  /// means that every IPOPT sub-problem always returns the best explored
  /// trajectory, even if it subsequently explored other states. This increases
  /// the stability of solutions, but can lead to getting stuck in local minima.
  void setEnableOptimizationGuards(bool enabled);

  /// Defaults to false. This records every iteration of IPOPT in the log, so we
  /// can debug it. This should only be used on MPCLocal that's running for a
  /// short time. Otherwise the log will grow without bound.
  void setRecordIterations(bool enabled);

  /// This gets the current maximum number of iterations that IPOPT will be
  /// allowed to run during an optimization.
  int getMaxIterations();

  /// This sets the current maximum number of iterations that IPOPT will be
  /// allowed to run during an optimization. MPCLocal reserves the right to
  /// change this value during runtime depending on timing and performance
  /// values observed during running.
  void setMaxIterations(int maxIters);

  /// This records the current state of the world based on some external sensing
  /// and inference. This resets the error in our model just assuming the world
  /// is exactly following our simulation.
  void recordGroundTruthState(
      long time,
      Eigen::VectorXd pos,
      Eigen::VectorXd vel,
      Eigen::VectorXd mass) override;

  /// This optimizes a block of the plan, starting at `startTime`
  void optimizePlan(long startTime);

  /// This adjusts parameters to make sure we're keeping up with real time. We
  /// can compute how many (ms / step) it takes us to optimize plans. Sometimes
  /// we can decrease (ms / step) by increasing the length of the optimization
  /// and increasing the parallelism. We can also change the step size in the
  /// physics engine to produce less accurate results, but keep up with the
  /// world in fewer steps.
  void adjustPerformance(long lastOptimizeTimeMillis);

  /// This starts our main thread and begins running optimizations
  void start() override;

  /// This stops our main thread, waits for it to finish, and then returns
  void stop() override;

  /// This returns the main record we've been keeping of our optimization up to
  /// this point
  std::shared_ptr<trajectory::Solution> getCurrentSolution();

  /// This registers a listener to get called when we finish replanning
  void registerReplanningListener(
      std::function<void(long, const trajectory::TrajectoryRollout*, long)>
          replanListener) override;

  /// This launches a server on the specified port. This call blocks
  /// indefinitely, until the program is killed with Ctrl+C
  void serve(int port);

protected:
  /// This is the function for the optimization thread to run when we're live
  void optimizationThreadLoop();

  bool mRunning;
  std::shared_ptr<simulation::World> mWorld;
  std::shared_ptr<trajectory::LossFn> mLoss;
  ObservationLog mObservationLog;

  // Meta config
  bool mEnableLinesearch;
  bool mEnableOptimizationGuards;
  bool mRecordIterations;

  int mPlanningHorizonMillis;
  int mMillisPerStep;
  int mSteps;
  int mShotLength;
  int mMaxIterations;
  int mMillisInAdvanceToPlan;
  long mLastOptimizedTime;
  RealTimeControlBuffer mBuffer;
  std::thread mOptimizationThread;
  bool mSilent;

  std::shared_ptr<trajectory::Optimizer> mOptimizer;
  std::shared_ptr<trajectory::Solution> mSolution;
  std::shared_ptr<trajectory::Problem> mProblem;

  // These are listeners that get called when we finish replanning
  std::vector<
      std::function<void(long, const trajectory::TrajectoryRollout*, long)>>
      mReplannedListeners;

  friend class RPCWrapperMPCLocal;
};

class RPCWrapperMPCLocal : public proto::MPCService::Service
{
public:
  RPCWrapperMPCLocal(MPCLocal& local);

  /// Remotely start the compute running
  grpc::Status Start(
      grpc::ServerContext* context,
      const proto::MPCStartRequest* request,
      proto::MPCStartReply* response) override;

  /// Remotely stop the compute running
  grpc::Status Stop(
      grpc::ServerContext* context,
      const proto::MPCStopRequest* request,
      proto::MPCStopReply* response) override;

  /// Remotely listen for replanning updates
  grpc::Status ListenForUpdates(
      grpc::ServerContext* context,
      const proto::MPCListenForUpdatesRequest* request,
      grpc::ServerWriter<proto::MPCListenForUpdatesReply>* writer) override;

  /// Remotely listen for replanning updates
  grpc::Status RecordGroundTruthState(
      grpc::ServerContext* context,
      const proto::MPCRecordGroundTruthStateRequest* request,
      proto::MPCRecordGroundTruthStateReply* reply) override;

  /// Remotely listen for replanning updates
  grpc::Status ObserveForce(
      grpc::ServerContext* context,
      const proto::MPCObserveForceRequest* request,
      proto::MPCObserveForceReply* reply) override;

protected:
  MPCLocal& mLocal;
};

} // namespace realtime
} // namespace dart

#endif