#include "dart/realtime/MPCLocal.hpp"

#include <google/protobuf/arena_impl.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "dart/performance/PerformanceLog.hpp"
#include "dart/proto/SerializeEigen.hpp"
#include "dart/realtime/Millis.hpp"
#include "dart/realtime/RealTimeControlBuffer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Solution.hpp"

#include "signal.h"

namespace dart {

using namespace trajectory;

namespace realtime {

MPCLocal::MPCLocal(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<trajectory::LossFn> loss,
    int planningHorizonMillis)
  : mRunning(false),
    mWorld(world),
    mLoss(loss),
    mObservationLog(
        timeSinceEpochMillis(),
        world->getPositions(),
        world->getVelocities(),
        world->getMasses()),
    mEnableLinesearch(true),
    mEnableOptimizationGuards(false),
    mRecordIterations(false),
    mPlanningHorizonMillis(planningHorizonMillis),
    mMillisPerStep(1000 * world->getTimeStep()),
    mSteps((int)ceil((s_t)planningHorizonMillis / mMillisPerStep)),
    mShotLength(50),
    mMaxIterations(5),
    mMillisInAdvanceToPlan(0),
    mLastOptimizedTime(0L),
    mBuffer(RealTimeControlBuffer(world->getNumDofs(), mSteps, mMillisPerStep)),
    mSilent(false)
{
}

/// Copy constructor
MPCLocal::MPCLocal(const MPCLocal& mpc)
  : mRunning(mpc.mRunning),
    mWorld(mpc.mWorld),
    mLoss(mpc.mLoss),
    mObservationLog(mpc.mObservationLog),
    mEnableLinesearch(mpc.mEnableLinesearch),
    mEnableOptimizationGuards(mpc.mEnableOptimizationGuards),
    mRecordIterations(mpc.mRecordIterations),
    mPlanningHorizonMillis(mpc.mPlanningHorizonMillis),
    mMillisPerStep(mpc.mMillisPerStep),
    mSteps(mpc.mSteps),
    mShotLength(mpc.mShotLength),
    mMaxIterations(mpc.mMaxIterations),
    mMillisInAdvanceToPlan(mpc.mMillisInAdvanceToPlan),
    mLastOptimizedTime(mpc.mLastOptimizedTime),
    mBuffer(mpc.mBuffer),
    mSilent(mpc.mSilent)
{
}

/// This updates the loss function that we're going to move in real time to
/// minimize. This can happen quite frequently, for example if our loss
/// function is to track a mouse pointer in a simulated environment, we may
/// reset the loss function every time the mouse moves.
void MPCLocal::setLoss(std::shared_ptr<trajectory::LossFn> loss)
{
  mLoss = loss;
}

/// This sets the optimizer that MPCLocal will use. This will override the
/// default optimizer. This should be called before start().
void MPCLocal::setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer)
{
  mOptimizer = optimizer;
}

/// This returns the current optimizer that MPCLocal is using
std::shared_ptr<trajectory::Optimizer> MPCLocal::getOptimizer()
{
  return mOptimizer;
}

/// This sets the problem that MPCLocal will use. This will override the default
/// problem. This should be called before start().
void MPCLocal::setProblem(std::shared_ptr<trajectory::Problem> problem)
{
  mProblem = problem;
}

/// This returns the current problem definition that MPCLocal is using
std::shared_ptr<trajectory::Problem> MPCLocal::getProblem()
{
  return mProblem;
}

/// This gets the force to apply to the world at this instant. If we haven't
/// computed anything for this instant yet, this just returns 0s.
Eigen::VectorXs MPCLocal::getControlForce(long now)
{
  return mBuffer.getPlannedForce(now);
}

/// This returns how many millis we have left until we've run out of plan.
/// This can be a negative number, if we've run past our plan.
long MPCLocal::getRemainingPlanBufferMillis()
{
  return mBuffer.getPlanBufferMillisAfter(timeSinceEpochMillis());
}

/// This can completely silence log output
void MPCLocal::setSilent(bool silent)
{
  mSilent = silent;
}

/// This enables linesearch on the IPOPT sub-problems. Defaults to true. This
/// increases the stability of solutions, but can lead to spikes in solution
/// times.
void MPCLocal::setEnableLineSearch(bool enabled)
{
  mEnableLinesearch = enabled;
}

/// This enables "guards" on the IPOPT sub-problems. Defaults to false. This
/// means that every IPOPT sub-problem always returns the best explored
/// trajectory, even if it subsequently explored other states. This increases
/// the stability of solutions, but can lead to getting stuck in local minima.
void MPCLocal::setEnableOptimizationGuards(bool enabled)
{
  mEnableOptimizationGuards = enabled;
}

/// Defaults to false. This records every iteration of IPOPT in the log, so we
/// can debug it. This should only be used on MPCLocal that's running for a
/// short time. Otherwise the log will grow without bound.
void MPCLocal::setRecordIterations(bool enabled)
{
  mRecordIterations = enabled;
}

/// This gets the current maximum number of iterations that IPOPT will be
/// allowed to run during an optimization.
int MPCLocal::getMaxIterations()
{
  return mMaxIterations;
}

/// This sets the current maximum number of iterations that IPOPT will be
/// allowed to run during an optimization. MPCLocal reserves the right to change
/// this value during runtime depending on timing and performance values
/// observed during running.
void MPCLocal::setMaxIterations(int maxIters)
{
  mMaxIterations = maxIters;
}

/// This records the current state of the world based on some external sensing
/// and inference. This resets the error in our model just assuming the world
/// is exactly following our simulation.
void MPCLocal::recordGroundTruthState(
    long time, Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mass)
{
  mObservationLog.observe(time, pos, vel, mass);
}

/// This optimizes a block of the plan, starting at `startTime`
void MPCLocal::optimizePlan(long startTime)
{
  // We don't allow time to go backwards, because that leads to all sorts of
  // issues. We can get called for a time before a time we already optimized
  // for, because of dilating buffers in front of our current time. If that
  // happens, just pretent like we were asked for the latest time we were
  // optimizing for.
  if (startTime < mLastOptimizedTime)
  {
    startTime = mLastOptimizedTime;
  }

  if (mSolution == nullptr)
  {
    PerformanceLog::initialize();
    PerformanceLog* log = PerformanceLog::startRoot("MPCLocal loop");

    std::shared_ptr<simulation::World> worldClone = mWorld->clone();
    PerformanceLog* estimateState = log->startRun("Estimate State");

    mBuffer.estimateWorldStateAt(worldClone, &mObservationLog, startTime);
    estimateState->end();

    if (!mOptimizer)
    {
      PerformanceLog* createOpt = log->startRun("Create Default IPOPT");

      std::shared_ptr<IPOptOptimizer> ipoptOptimizer
          = std::make_shared<IPOptOptimizer>();
      ipoptOptimizer->setCheckDerivatives(false);
      ipoptOptimizer->setSuppressOutput(true);
      ipoptOptimizer->setRecoverBest(mEnableOptimizationGuards);
      ipoptOptimizer->setTolerance(1e-3);
      ipoptOptimizer->setIterationLimit(mMaxIterations);
      ipoptOptimizer->setDisableLinesearch(!mEnableLinesearch);
      ipoptOptimizer->setRecordFullDebugInfo(false);
      ipoptOptimizer->setRecordIterations(false);
      if (mSilent)
      {
        ipoptOptimizer->setSilenceOutput(true);
      }
      mOptimizer = ipoptOptimizer;

      createOpt->end();
    }

    if (!mProblem)
    {
      std::shared_ptr<MultiShot> multishot = std::make_shared<MultiShot>(
          worldClone, *mLoss.get(), mSteps, mShotLength, false);
      multishot->setParallelOperationsEnabled(true);
      mProblem = multishot;
    }

    PerformanceLog* optimizeTrack = log->startRun("Optimize");

    mSolution = mOptimizer->optimize(mProblem.get());
    optimizeTrack->end();

    mLastOptimizedTime = startTime;

    mBuffer.setControlForcePlan(
        startTime,
        timeSinceEpochMillis(),
        mProblem->getRolloutCache(worldClone)->getControlForcesConst());

    log->end();

    std::cout << PerformanceLog::finalize()["MPCLocal loop"]->prettyPrint()
              << std::endl;
  }
  else
  {
    std::shared_ptr<simulation::World> worldClone = mWorld->clone();

    int diff = startTime - mLastOptimizedTime;
    int steps
        = static_cast<int>(floor(static_cast<s_t>(diff) / mMillisPerStep));
    int roundedDiff = steps * mMillisPerStep;
    long roundedStartTime = mLastOptimizedTime + roundedDiff;
    long totalPlanTime = mSteps * mMillisPerStep;
    s_t percentage = (s_t)roundedDiff * 100.0 / totalPlanTime;

    if (!mSilent)
    {
      std::cout << "Advancing plan by " << roundedDiff << "ms = " << steps
                << " steps, " << (percentage) << "% of total " << totalPlanTime
                << "ms plan time" << std::endl;
    }

    long startComputeWallTime = timeSinceEpochMillis();

    mBuffer.estimateWorldStateAt(
        worldClone, &mObservationLog, roundedStartTime);

    mProblem->advanceSteps(
        worldClone,
        worldClone->getPositions(),
        worldClone->getVelocities(),
        steps);

    mSolution->reoptimize();

    // std::cout << "MPCLocal::optimizePlan() mBuffer.setControlForcePlan()" <<
    // std::endl;

    mBuffer.setControlForcePlan(
        startTime,
        timeSinceEpochMillis(),
        mProblem->getRolloutCache(worldClone)->getControlForcesConst());

    long computeDurationWallTime
        = timeSinceEpochMillis() - startComputeWallTime;

    // Call any listeners that might be waiting on us
    for (auto listener : mReplannedListeners)
    {
      listener(
          startTime,
          mProblem->getRolloutCache(worldClone),
          computeDurationWallTime);
    }

    if (!mSilent)
    {
      s_t factorOfSafety = 0.5;
      std::cout << " -> We were allowed "
                << (int)floor(roundedDiff * factorOfSafety)
                << "ms to solve this problem (" << roundedDiff
                << "ms new planning * " << factorOfSafety
                << " factor of safety), and it took us "
                << computeDurationWallTime << "ms" << std::endl;
    }

    mLastOptimizedTime = roundedStartTime;
  }
}

/// This adjusts parameters to make sure we're keeping up with real time. We
/// can compute how many (ms / step) it takes us to optimize plans. Sometimes
/// we can decrease (ms / step) by increasing the length of the optimization
/// and increasing the parallelism. We can also change the step size in the
/// physics engine to produce less accurate results, but keep up with the
/// world in fewer steps.
void MPCLocal::adjustPerformance(long lastOptimizeTimeMillis)
{
  // This ensures that we don't "optimize our way out of sync", by letting the
  // optimizer change forces that already happened by the time the optimization
  // finishes, leading to us getting out of sync. Better to make our plans start
  // into the future.
  mMillisInAdvanceToPlan = 1.2 * lastOptimizeTimeMillis;
  // Don't go more than 200ms into the future, cause then errors have a chance
  // to propagate
  if (mMillisInAdvanceToPlan > 200)
    mMillisInAdvanceToPlan = 200;

  /*
  s_t millisToComputeEachStep = (s_t)lastOptimizeTimeMillis / mSteps;
  // Our safety margin is 3x, we want to be at least 3 times as fast as real
  // time
  long desiredMillisPerStep = 3 * millisToComputeEachStep;

  // This means our simulation step is too small, and we risk overflowing our
  // buffer before optimization finishes
  if (desiredMillisPerStep > mMillisPerStep)
  {
    std::cout << "Detected we're going too slow! Increasing timestep size from "
              << mMillisPerStep << "ms -> " << desiredMillisPerStep << "ms"
              << std::endl;

    mBuffer.setMillisPerStep(mMillisPerStep);
    mMillisPerStep = desiredMillisPerStep;
  }
  */
}

/// This starts our main thread and begins running optimizations
void MPCLocal::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mOptimizationThread = std::thread(&MPCLocal::optimizationThreadLoop, this);
}

/// This stops our main thread, waits for it to finish, and then returns
void MPCLocal::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mOptimizationThread.join();
}

/// This returns the main record we've been keeping of our optimization up to
/// this point
std::shared_ptr<trajectory::Solution> MPCLocal::getCurrentSolution()
{
  return mSolution;
}

/// This registers a listener to get called when we finish replanning
void MPCLocal::registerReplanningListener(
    std::function<void(long, const trajectory::TrajectoryRollout*, long)>
        replanListener)
{
  mReplannedListeners.push_back(replanListener);
}

/// This launches a server on the specified port
void MPCLocal::serve(int port)
{
  std::string server_address("0.0.0.0:" + std::to_string(port));

  grpc::EnableDefaultHealthCheckService(true);
  // grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  RPCWrapperMPCLocal wrapper(*this);

  builder.RegisterService(&wrapper);
  // Finally assemble the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

///////////////////////////////////////////////////////////////////////
/// Implements the gRPC API
///////////////////////////////////////////////////////////////////////

RPCWrapperMPCLocal::RPCWrapperMPCLocal(MPCLocal& local) : mLocal(local)
{
}

/// Remotely start the compute running
grpc::Status RPCWrapperMPCLocal::Start(
    grpc::ServerContext* /* context */,
    const proto::MPCStartRequest* /* request */,
    proto::MPCStartReply* /* response */)
{
  mLocal.start();
  return grpc::Status::OK;
}

/// Remotely stop the compute running
grpc::Status RPCWrapperMPCLocal::Stop(
    grpc::ServerContext* /* context */,
    const proto::MPCStopRequest* /* request */,
    proto::MPCStopReply* /* response */)
{
  mLocal.stop();
  return grpc::Status::OK;
}

/// Remotely listen for replanning updates
grpc::Status RPCWrapperMPCLocal::ListenForUpdates(
    grpc::ServerContext* /* context */,
    const proto::MPCListenForUpdatesRequest* /* request */,
    grpc::ServerWriter<proto::MPCListenForUpdatesReply>* writer)
{
  proto::MPCListenForUpdatesReply reply;
  mLocal.registerReplanningListener(
      [&](long startTime,
          const trajectory::TrajectoryRollout* rollout,
          long duration) {
        reply.mutable_rollout()->Clear();
        rollout->serialize(*reply.mutable_rollout());
        reply.set_starttime(startTime);
        reply.set_replandurationmillis(duration);
        writer->Write(reply);
      });

  while (true)
  {
    // spin
  }
}

/// Remotely listen for replanning updates
grpc::Status RPCWrapperMPCLocal::RecordGroundTruthState(
    grpc::ServerContext* /* context */,
    const proto::MPCRecordGroundTruthStateRequest* request,
    proto::MPCRecordGroundTruthStateReply* /* reply */)
{
  // std::cout << "gRPC server: RecordGroundTruthState" << std::endl;
  mLocal.recordGroundTruthState(
      request->time(),
      deserializeVector(request->pos()),
      deserializeVector(request->vel()),
      deserializeVector(request->mass()));
  return grpc::Status::OK;
}

/// Remotely listen for replanning updates
grpc::Status RPCWrapperMPCLocal::ObserveForce(
    grpc::ServerContext* /* context */,
    const proto::MPCObserveForceRequest* request,
    proto::MPCObserveForceReply* /* reply */)
{
  // std::cout << "gRPC server: ObserveForce" << std::endl;
  mLocal.mBuffer.manuallyRecordObservedForce(
      request->time(), deserializeVector(request->force()));
  return grpc::Status::OK;
}

/// This is the function for the optimization thread to run when we're live
void MPCLocal::optimizationThreadLoop()
{
  // block signals in this thread and subsequently
  // spawned threads, so they're guaranteed to go to the server thread
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

  while (mRunning)
  {
    long startTime = timeSinceEpochMillis();
    optimizePlan(startTime + mMillisInAdvanceToPlan);
    long endTime = timeSinceEpochMillis();
    adjustPerformance(endTime - startTime);
  }
}

} // namespace realtime
} // namespace dart