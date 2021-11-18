#include "dart/realtime/iLQRLocal.hpp"

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
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/BackpropSnapshot.hpp"

#include "signal.h"

namespace dart {

using namespace trajectory;

namespace realtime {

LQRBuffer::LQRBuffer(
  int steps, 
  size_t nDofs, 
  size_t nControls, 
  Extrapolate_Method extrapolate)
{
  std::cout << "LQRBuffer Initializing ..." <<steps<<" "<<nDofs<<" "<<nControls<< std::endl;
  nsteps = steps;
  state_dim = nDofs * 2;
  control_dim = nControls;
  ext = extrapolate;

  for(size_t i = 0; i < nsteps - 1; i++)
  {
    X.push_back(Eigen::VectorXs::Zero(state_dim));
    Xnew.push_back(Eigen::VectorXs::Zero(state_dim));
    U.push_back(Eigen::VectorXs::Zero(control_dim));
    Unew.push_back(Eigen::VectorXs::Zero(control_dim));
    K.push_back(Eigen::MatrixXs::Zero(control_dim, state_dim));
    k.push_back(Eigen::VectorXs::Zero(control_dim));
        
    // Jacobians
    Fx.push_back(Eigen::MatrixXs::Zero(state_dim, state_dim));
    Fu.push_back(Eigen::MatrixXs::Zero(state_dim, control_dim));

    // Gradients
    Lx.push_back(Eigen::VectorXs::Zero(state_dim));
    Lu.push_back(Eigen::VectorXs::Zero(control_dim));

    // Hessians
    Lxx.push_back(Eigen::MatrixXs::Zero(state_dim, state_dim));
    Lux.push_back(Eigen::MatrixXs::Zero(state_dim, control_dim));
    Luu.push_back(Eigen::MatrixXs::Zero(control_dim, control_dim));
  }
  X.push_back(Eigen::VectorXs::Zero(state_dim));
  Xnew.push_back(Eigen::VectorXs::Zero(state_dim));
  Lx.push_back(Eigen::VectorXs::Zero(state_dim));
  Lxx.push_back(Eigen::MatrixXs::Zero(state_dim, state_dim));
  std::cout << "LQRBuffer Initialized" << std::endl;
}

void LQRBuffer::resetXUNew()
{
  Xnew.assign(X.begin(), X.end());
  Unew.assign(U.begin(), U.end());
}

void LQRBuffer::updateXUOld()
{
  X = Xnew;
  U = Unew;
}

// From Buffer Read Plan for execution as well as control law starting from timestamp
// Which should have equivalent effectiveness of advancePlan
// TODO: What is the difference between set plan and 
void LQRBuffer::readNewActionPlan(long timestamp, RealTimeControlBuffer buffer)
{
  Eigen::MatrixXs existForce = Eigen::MatrixXs::Zero(control_dim, nsteps);
  Eigen::MatrixXs existk = Eigen::MatrixXs::Zero(control_dim, nsteps);
  std::vector<Eigen::MatrixXs> existK;
  for(int i = 0; i < nsteps; i++)
  {
    existK.push_back(Eigen::MatrixXs::Zero(control_dim, state_dim));
  }
  buffer.getPlannedForcesStartingAt(timestamp, existForce);
  buffer.getPlannedkStartingAt(timestamp, existk);
  buffer.getPlannedKStartingAt(timestamp, existK);
  size_t existSteps = buffer.getRemainSteps(timestamp);
  size_t i = 0;
  for(;i < existSteps; i++)
  {
    U[i] = existForce.col(i);
    k[i] = existk.col(i);
    K[i] = existK[i];
  }
  size_t last = i;
  switch(ext)
  {
    case ZERO:
      for(; i < nsteps; i++)
      {
        U[i] = Eigen::VectorXs::Zero(control_dim);
        k[i] = Eigen::VectorXs::Zero(control_dim);
        K[i] = Eigen::MatrixXs::Zero(control_dim, state_dim);
      }
      break;
    case LAST:
      for(; i < nsteps; i++)
      {
        U[i].segment(0, control_dim) = U[last];
        k[i].segment(0, control_dim) = k[last];
        K[i].block(0, 0, control_dim, state_dim) = K[last];
      }
      break;
    case RANDOM:
      for(; i < nsteps; i++)
      {
        U[i] = Eigen::VectorXs::Random(control_dim);
        k[i] = Eigen::VectorXs::Random(control_dim);
        K[i] = Eigen::MatrixXs::Random(control_dim, state_dim);
      }
      break;
    default:
      assert(false && "Should not reach here");
  }
}

void LQRBuffer::updateL(std::vector<Eigen::VectorXs> Lx_new, std::vector<Eigen::VectorXs> Lu_new,
               std::vector<Eigen::MatrixXs> Lxx_new, std::vector<Eigen::MatrixXs> Luu_new,
               std::vector<Eigen::MatrixXs> Lux_new)
{
  Lx = Lx_new;
  Lu = Lu_new;
  Lxx = Lxx_new;
  Luu = Luu_new;
  Lux = Lux_new;
}

void LQRBuffer::setNewActionPlan(long timestamp, RealTimeControlBuffer* buffer)
{
  // Need to compute how many step in U has been invalid
  long current_time = timeSinceEpochMillis();
  Eigen::MatrixXs control_force = Eigen::MatrixXs::Zero(control_dim, nsteps);
  for(int i = 0; i < nsteps; i++)
  {
    control_force.col(i) = U[i];
  }
  // This assume the vector indicate plan right after timestamp
  buffer->setControlForcePlan(
        timestamp,
        current_time,
        control_force);
}

void LQRBuffer::setNewControlLaw(long timestamp, RealTimeControlBuffer* buffer)
{
  // Need to compute how many step in U has been invalid
  long current_time = timeSinceEpochMillis();

  // This assume the vector indicate plan right after timestamp
  buffer->setControlLawPlan(
        timestamp,
        current_time,
        k,
        K,
        X);
}

void LQRBuffer::updateF(std::vector<Eigen::MatrixXs> Fx_new, std::vector<Eigen::MatrixXs> Fu_new)
{
  Fx = Fx_new;
  Fu = Fu_new;
}

bool LQRBuffer::validateXnew()
{
  for(size_t i=0; i < Xnew.size(); i++)
  {
    if(Xnew[i].hasNaN())
      return false;
  }
  return true;
}

iLQRLocal::iLQRLocal(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<TargetReachingCost> costFn,
    Eigen::VectorXi actuatedJoint,
    int planningHorizonMillis,
    s_t scale)
  : mRunning(false),
    mWorld(world),
    mLoss(costFn->getLossFn()),
    mObservationLog(
        timeSinceEpochMillis(),
        world->getPositions(),
        world->getVelocities(),
        world->getMasses()),
    mEnableLinesearch(true),
    mEnableOptimizationGuards(false),
    mRecordIterations(false),
    mPlanningHorizonMillis(planningHorizonMillis),
    mMillisPerStep(scale*1000 * world->getTimeStep()),
    mSteps((int)ceil((s_t)planningHorizonMillis / mMillisPerStep)),
    mShotLength(50),
    mMaxIterations(5),
    mMillisInAdvanceToPlan(0),
    mLastOptimizedTime(0L),
    mBuffer(RealTimeControlBuffer(world->getNumDofs(), mSteps, mMillisPerStep, world->getNumDofs() * 2)),
    mSilent(false),
    // Below are iLQR related information
    mActuatedJoint(actuatedJoint),
    mAlpha_reset_value(1.0),
    mAlpha(1.0),
    mPatience_reset_value(8),
    mPatience(8),
    mDelta0(2.0),
    mDelta(2.0),
    mMU_MIN(1e-6),
    mMU(100.*1e-6),
    mMU_reset_value(100.*1e-6),
    mCost(0.0),
    last_loss(std::numeric_limits<s_t>::max()),
    mlqrBuffer(LQRBuffer(mSteps, world->getNumDofs(), actuatedJoint.size(), Extrapolate_Method::LAST))
{
  size_t nControls = actuatedJoint.size();
  mLast_U = Eigen::VectorXs::Zero(nControls);
}


/// This sets the optimizer that iLQRLocal will use. This will override the
/// default optimizer. This should be called before start().
void iLQRLocal::setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer)
{
  mOptimizer = optimizer;
}

/// This returns the current optimizer that iLQRLocal is using
std::shared_ptr<trajectory::Optimizer> iLQRLocal::getOptimizer()
{
  return mOptimizer;
}

/// This sets the problem that iLQRLocal will use. This will override the default
/// problem. This should be called before start().
void iLQRLocal::setProblem(std::shared_ptr<trajectory::Problem> problem)
{
  mProblem = problem;
}

/// This returns the current problem definition that iLQRLocal is using
std::shared_ptr<trajectory::Problem> iLQRLocal::getProblem()
{
  return mProblem;
}

/// This gets the force to apply to the world at this instant. If we haven't
/// computed anything for this instant yet, this just returns 0s.
Eigen::VectorXs iLQRLocal::getControlForce(long now)
{
  return mBuffer.getPlannedForce(now);
}

/// This gets the force from LQR control law and current state
Eigen::VectorXs iLQRLocal::computeForce(Eigen::VectorXs state, long now)
{
  Eigen::MatrixXs K = getControlK(now);
  Eigen::VectorXs k = getControlk(now);
  Eigen::VectorXs x = mBuffer.getPlannedState(now);

  Eigen::VectorXs force = mLast_U + mAlpha * k 
                         + K * (state - x);
  return force;

}

/// This returns how many millis we have left until we've run out of plan.
/// This can be a negative number, if we've run past our plan.
long iLQRLocal::getRemainingPlanBufferMillis()
{
  return mBuffer.getPlanBufferMillisAfter(timeSinceEpochMillis());
}

/// This can completely silence log output
void iLQRLocal::setSilent(bool silent)
{
  mSilent = silent;
}

/// This enables linesearch on the IPOPT sub-problems. Defaults to true. This
/// increases the stability of solutions, but can lead to spikes in solution
/// times.
void iLQRLocal::setEnableLineSearch(bool enabled)
{
  mEnableLinesearch = enabled;
}

/// This enables "guards" on the IPOPT sub-problems. Defaults to false. This
/// means that every IPOPT sub-problem always returns the best explored
/// trajectory, even if it subsequently explored other states. This increases
/// the stability of solutions, but can lead to getting stuck in local minima.
void iLQRLocal::setEnableOptimizationGuards(bool enabled)
{
  mEnableOptimizationGuards = enabled;
}

/// Defaults to false. This records every iteration of IPOPT in the log, so we
/// can debug it. This should only be used on iLQRLocal that's running for a
/// short time. Otherwise the log will grow without bound.
void iLQRLocal::setRecordIterations(bool enabled)
{
  mRecordIterations = enabled;
}

/// This gets the current maximum number of iterations that IPOPT will be
/// allowed to run during an optimization.
int iLQRLocal::getMaxIterations()
{
  return mMaxIterations;
}

/// This sets the current maximum number of iterations that IPOPT will be
/// allowed to run during an optimization. iLQRLocal reserves the right to change
/// this value during runtime depending on timing and performance values
/// observed during running.
void iLQRLocal::setMaxIterations(int maxIters)
{
  mMaxIterations = maxIters;
}

/// This records the current state of the world based on some external sensing
/// and inference. This resets the error in our model just assuming the world
/// is exactly following our simulation.
void iLQRLocal::recordGroundTruthState(
    long time, Eigen::VectorXs pos, Eigen::VectorXs vel, Eigen::VectorXs mass)
{
  mObservationLog.observe(time, pos, vel, mass);
}

/// This optimizes a block of the plan, starting at `startTime`
/// startTime = current + millisAdvanceToPlan
void iLQRLocal::optimizePlan(long startTime)
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

  if (mSolution == nullptr || variableChange())
  {
    PerformanceLog::initialize();
    PerformanceLog* log = PerformanceLog::startRoot("TrajOptLocal loop");

    std::shared_ptr<simulation::World> worldClone = mWorld->clone();
    PerformanceLog* estimateState = log->startRun("Estimate State");

    mBuffer.estimateWorldStateAt(worldClone, &mObservationLog, startTime);
    estimateState->end();
    std::cout<<"Optimization Stage"<<std::endl;
    if (!mOptimizer)
    {
      PerformanceLog* createOpt = log->startRun("Create Default IPOPT");

      std::shared_ptr<IPOptOptimizer> ipoptOptimizer
          = std::make_shared<IPOptOptimizer>();
      ipoptOptimizer->setCheckDerivatives(false);
      ipoptOptimizer->setSuppressOutput(true);
      ipoptOptimizer->setRecoverBest(mEnableOptimizationGuards);
      ipoptOptimizer->setTolerance(1e-3); //1e-3
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

    if (!mProblem || variableChange())
    {
      std::shared_ptr<MultiShot> multishot = std::make_shared<MultiShot>(
          worldClone, *mLoss.get(), mSteps, mShotLength, false);
      multishot->setParallelOperationsEnabled(true);
      mProblem = multishot;
      mVarchange = false;
    }

    PerformanceLog* optimizeTrack = log->startRun("Optimize");
    //std::cout<<"MPC Optimization Start"<<std::endl;
    mSolution = mOptimizer->optimize(mProblem.get());
    //std::cout<<"MPC Optimization end"<<std::endl;
    optimizeTrack->end();

    mLastOptimizedTime = startTime;

    mBuffer.setControlForcePlan(
        startTime,
        timeSinceEpochMillis(),
        mProblem->getRolloutCache(worldClone)->getControlForcesConst());

    log->end();

    // std::cout << PerformanceLog::finalize()["iLQRLocal loop"]->prettyPrint()
    //           << std::endl;
  }
  else
  {
    std::shared_ptr<simulation::World> worldClone = mWorld->clone();
    std::cout<<"Re-optimization stage "<<startTime<<std::endl;
    int diff = startTime - mLastOptimizedTime;
    // How many steps taken for last run of optimization in real world
    int steps
        = static_cast<int>(floor(static_cast<s_t>(diff) / mMillisPerStep));
    int roundedDiff = steps * mMillisPerStep;
    long roundedStartTime = mLastOptimizedTime + roundedDiff;
    long totalPlanTime = mSteps * mMillisPerStep;
    // How many percent of the planned force has been executed
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

    // Make sure that in the force buffer, the pointer of starting point is pointed to current
    mProblem->advanceSteps(
        worldClone,
        worldClone->getPositions(),
        worldClone->getVelocities(),
        steps);

    // Optimizer based partially on initial guess of where force supposed to be
    mSolution->reoptimize();

    // std::cout << "iLQRLocal::optimizePlan() mBuffer.setControlForcePlan()" <<
    // std::endl;

    // Update the control force plan from current time stamp.
    // The new optimization problem is started at start time, however since the replanning
    // Process take an non trivial amount of time, the force to be set is from now.
    mBuffer.setControlForcePlan(
        startTime,
        timeSinceEpochMillis(),
        mProblem->getRolloutCache(worldClone)->getControlForcesConst());

    long computeDurationWallTime
        = timeSinceEpochMillis() - startComputeWallTime;

    // Call any listeners that might be waiting on us
    // Currently the Listeners are trajectory plotter
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
    // Here is when optimization finished, supposingly this time should equals to now
    // And the real world can immediately get forced that has been planned
    mLastOptimizedTime = roundedStartTime;
  }
}

/// This adjusts parameters to make sure we're keeping up with real time. We
/// can compute how many (ms / step) it takes us to optimize plans. Sometimes
/// we can decrease (ms / step) by increasing the length of the optimization
/// and increasing the parallelism. We can also change the step size in the
/// physics engine to produce less accurate results, but keep up with the
/// world in fewer steps.
/// TODO: (Eric) THIS FUNCITON IS HARD CODED!!! For diffenrent environment
/// it may cause serious problems accordingly
void iLQRLocal::adjustPerformance(long lastOptimizeTimeMillis)
{
  mMillisInAdvanceToPlan = 1.2 * lastOptimizeTimeMillis;
  if (mMillisInAdvanceToPlan > 200)
    mMillisInAdvanceToPlan = 200;
}

/// This starts our main thread and begins running optimizations
void iLQRLocal::start()
{
  if (mRunning)
    return;
  mRunning = true;
  mOptimizationThread = std::thread(&iLQRLocal::optimizationThreadLoop, this);
}

/// This starts our main thread and begins running iLQR optimizations
void iLQRLocal::ilqrstart()
{
  if (mRunning)
    return;
  mRunning = true;
  mOptimizationThread = std::thread(&iLQRLocal::iLQRoptimizationThreadLoop, this);
}

/// This stops our main thread, waits for it to finish, and then returns
void iLQRLocal::stop()
{
  if (!mRunning)
    return;
  mRunning = false;
  mOptimizationThread.join();
}

/// This stops our main thread, waits for it to finish, and then returns
void iLQRLocal::ilqrstop()
{
  if(!mRunning)
    return;
  mRunning = false;
  mOptimizationThread.join();
}

/// This returns the main record we've been keeping of our optimization up to
/// this point
std::shared_ptr<trajectory::Solution> iLQRLocal::getCurrentSolution()
{
  return mSolution;
}

/// ==========================================================================
/// ============ Here are functions for iLQR =================================
/// This will run forward pass from start time and record info such as Jacobian
/// of each timestep and gradient as well as hessian from cost function
bool iLQRLocal::ilqrForward(long startTime, simulation::WorldPtr world)
{
  bool nan_flag = false;
  mBuffer.estimateWorldStateAt(world, &mObservationLog, startTime);
  Eigen::VectorXs pos = world->getPositions();
  Eigen::VectorXs vel = world->getVelocities();
  Eigen::VectorXs mass = world->getMasses();

  std::vector<Eigen::VectorXs> Lx;
  std::vector<Eigen::VectorXs> Lu;
  std::vector<Eigen::MatrixXs> Lxx;
  std::vector<Eigen::MatrixXs> Luu;
  std::vector<Eigen::MatrixXs> Lux;


  while(mPatience > 0)
  {
    // Reset Control Force
    mlqrBuffer.resetXUNew();
    
    // set control force and initial condition
    world->setPositions(pos);
    world->setVelocities(vel);
    world->setMasses(mass);

    // Rollout Trajectory
    Eigen::MatrixXs pos = Eigen::MatrixXs::Zero(world->getNumDofs(), mSteps);
    std::unordered_map<std::string, Eigen::MatrixXs> pos_map;
    pos_map.emplace("identity", pos);
    Eigen::MatrixXs vel = Eigen::MatrixXs::Zero(world->getNumDofs(), mSteps);
    std::unordered_map<std::string, Eigen::MatrixXs> vel_map;
    vel_map.emplace("identity", vel);
    Eigen::MatrixXs forces = Eigen::MatrixXs::Zero(world->getNumDofs(), mSteps);
    std::unordered_map<std::string, Eigen::MatrixXs> force_map;
    force_map.emplace("identity", forces);
    Eigen::VectorXs mass = Eigen::VectorXs::Zero(world->getMassDims());
    std::unordered_map<std::string, Eigen::MatrixXs> meta;
    TrajectoryRolloutReal rollout = TrajectoryRolloutReal(pos_map, vel_map, force_map, mass, meta);
    s_t loss = 0;
    s_t dt = world->getTimeStep();

    // Executing the trajectory according to control law
    getTrajectory(world, &rollout, &mlqrBuffer);

    // Require states be stored in rollout
    // Assume loss except the final loss are multiplied by dt
    Lx       = mCostFn->ilqrGradientEstimator(&rollout, loss, WRTFLAG::X, dt);
    Lu       = mCostFn->ilqrGradientEstimator(&rollout, loss, WRTFLAG::U, dt);
    Lxx      = mCostFn->ilqrHessianEstimator(&rollout, WRTFLAG::XX, dt);
    Luu      = mCostFn->ilqrHessianEstimator(&rollout, WRTFLAG::UU, dt);
    Lux      = mCostFn->ilqrHessianEstimator(&rollout, WRTFLAG::UX, dt);
    
    

    // Sanity Check
    if(!mlqrBuffer.validateXnew())
    {
      mAlpha *= 0.5;
      mPatience--;
      nan_flag = true;
      continue;
    }
    else if(!((last_loss - loss) >= 0))
    {
      mAlpha *= 0.5;
      mPatience--;
      nan_flag = false;
      continue;
    }
    else // Good to leave
    {
      mCost = loss;
      mPatience = mPatience_reset_value;
      mAlpha = mAlpha_reset_value;
      nan_flag = false;
      break;
    }
  }
  if(mPatience == 0)
  {
    std::cout << "Forward Pass Run Out of Patience, exiting ..." << std::endl;
    // The out of patience is due to nan
    if(nan_flag)
    {
      return false;
    }
    else // The out of patience is due to loss not decrease
    {
      // 
      // Record new gradient and Hessian
      mlqrBuffer.updateXUOld();
      mlqrBuffer.updateL(Lx, Lu, Lxx, Luu, Lux);
      return true;
    }
  }
  // Record new gradient and Hessian
  mlqrBuffer.updateXUOld();
  mlqrBuffer.updateL(Lx, Lu, Lxx, Luu, Lux);
  return true;
}

bool iLQRLocal::ilqrBackward(simulation::WorldPtr world)
{
  s_t dt = world->getTimeStep();
  bool done = false;
  bool early_termination = false;
  // The loop should exit either done or early terminate
  while(!early_termination && !done)
  {
    Eigen::VectorXs Vx = mlqrBuffer.Lx[mSteps-1] / dt;
    Eigen::MatrixXs Vxx = mlqrBuffer.Lxx[mSteps-1] / dt;    
    for(size_t cursor = mSteps - 2; cursor > 0; cursor--)
    {
      // Build up Q matrices
      Eigen::VectorXs Qx = mlqrBuffer.Lx[cursor] +  mlqrBuffer.Fx[cursor].transpose() * Vx;
      Eigen::VectorXs Qu = mlqrBuffer.Lu[cursor] +  mlqrBuffer.Fu[cursor].transpose() * Vx;
      Eigen::MatrixXs Qxx = mlqrBuffer.Lxx[cursor] 
                            + mlqrBuffer.Fx[cursor].transpose() * Vxx * mlqrBuffer.Fx[cursor];
      Eigen::MatrixXs Qux = mlqrBuffer.Lux[cursor].transpose()
                            + mlqrBuffer.Fu[cursor].transpose() * Vxx * mlqrBuffer.Fx[cursor];
      Eigen::MatrixXs Quu = mlqrBuffer.Luu[cursor] 
                            + mlqrBuffer.Fu[cursor].transpose() * Vxx * mlqrBuffer.Fu[cursor];
      Eigen::MatrixXs I = Eigen::MatrixXs::Identity(Vxx.rows(),Vxx.cols());
      Eigen::MatrixXs Quubar = mlqrBuffer.Luu[cursor] 
                            + mlqrBuffer.Fu[cursor].transpose() * (Vxx + mMU * I) * mlqrBuffer.Fu[cursor];
      Eigen::MatrixXs Quxbar = mlqrBuffer.Luu[cursor] 
                            + (mlqrBuffer.Fu[cursor].transpose() * (Vxx + mMU * I) * mlqrBuffer.Fx[cursor]).transpose();
      if (abs(Quubar.determinant()) < 1e-5)
      {
        if(mPatience == 0)
        {
          early_termination = true;
          std::cout << "Regularize Patience limit met, exiting... " << std::endl;
          break;
        }
        else
        {
          std::cout << "Warning Singular Quu, iteration: "<< cursor <<"- repeating backward pass with increased mu."<< std::endl;
          // Increase mu
          if(mDelta * mDelta0 > mDelta0)
          {
            mDelta *= mDelta0;
          }
          else
          {
            mDelta = mDelta0;
          }
          if(mMU * mDelta > mMU_MIN)
          {
            mMU *= mDelta;
          }
          else
          {
            mMU = mMU_MIN;
          }
          mPatience -= 1;
          break;
        }
      }
      else
      {
        // Compute K matrix
        Eigen::MatrixXs Quubar_inv = Quubar.inverse();
        mlqrBuffer.K[cursor] = -Quubar_inv * Quxbar.transpose();
        mlqrBuffer.k[cursor] = -Quubar_inv * Qu;
        // Update Vxx and Vx
        Vx = Qx + mlqrBuffer.K[cursor].transpose() * Quu * mlqrBuffer.k[cursor] 
                + mlqrBuffer.K[cursor].transpose() * Qu 
                + Qux.transpose() * mlqrBuffer.k[cursor];
        Vxx = Qxx + mlqrBuffer.K[cursor].transpose() * Quu * mlqrBuffer.k[cursor] 
                  + mlqrBuffer.K[cursor].transpose() * Qux 
                  + Qux.transpose() * mlqrBuffer.K[cursor];
        if(cursor == 0)
          done = true;
      }
    }
  }
  mPatience = mPatience_reset_value;
  if(early_termination)
  {
    return false;
  }
  // Progress Delta for next iteration
  // Decrease mu
  if(1.0/mDelta0 > mDelta / mDelta0)
  {
    mDelta /= mDelta0;
  }
  else
  {
    mDelta = 1.0/mDelta0;
  }

  if(mMU * mDelta > mMU_MIN)
  {
    mMU = mMU * mDelta;
  }
  else
  {
    mMU = 0.0;
  }
  return true;
}

/// This function rollout the trajectory and save all the Jacobians to Buffer
/// as well as record the loss
/// This function automatically update the trajectory
void iLQRLocal::getTrajectory(simulation::WorldPtr world,
                              TrajectoryRollout* rollout,
                              LQRBuffer* lqrBuffer)
{
  // Need to store the state and actions in rollout
  for(int i = 0; i < mSteps - 1; i++)
  {
    // Compute New actions
    lqrBuffer->Unew[i] = lqrBuffer->Unew[i] + mAlpha * lqrBuffer->k[i] 
                         + lqrBuffer->K[i] * (lqrBuffer->Xnew[i] - lqrBuffer->X[i]);
    // Record state and action on rollout
    rollout->getPoses().col(i) = world->getPositions();
    rollout->getVels().col(i) = world->getVelocities();
    for(int j = 0; j < mActuatedJoint.size(); j++)
    {
      rollout->getControlForces()(mActuatedJoint(j),i) = lqrBuffer->Unew[i](j);
    }
    

    // Set the control force just computed from control law
    Eigen::VectorXs action = rollout->getControlForces().col(i);
    world->setControlForces(action);
    // Step the world
    std::shared_ptr<neural::BackpropSnapshot> snapshot = neural::forwardPass(world, false);
    // set new state
    lqrBuffer->Xnew[i+1].segment(0, world->getNumDofs()) = snapshot->getPostStepPosition();
    // set new Jacobian for linearization
    lqrBuffer->Fx[i] = snapshot->getStateJacobian(world);
    // Need to reassemble the matrix by actuated dofs
    lqrBuffer->Fu[i] = assembleJacobianMatrix(snapshot->getActionJacobian(world));
    
  }
}

bool iLQRLocal::ilqroptimizePlan(long startTime)
{
  if (startTime < mLastOptimizedTime)
  {
    startTime = mLastOptimizedTime;
  }
  // Get action from mBuffer according to time
  mlqrBuffer.readNewActionPlan(startTime, mBuffer);
  
  // Copy the world
  std::shared_ptr<simulation::World> worldClone = mWorld->clone();

  // Run some iterations of forward and backward
  bool forward_flag = true;
  bool backward_flag = true;
  s_t last_cost = mCost;
  // This Process takes non trivial amount of time
  // By the time is finished or failed the real world has taken lots of steps

  for(int iter = 0; iter < mMaxIterations; iter++)
  {
    forward_flag = ilqrForward(startTime, worldClone);
    if(forward_flag)
    {
      backward_flag = ilqrBackward(worldClone);
    }
    if(!forward_flag || !backward_flag)
    {
      std::cout << "Unable to solve the trajectory using DDP" << std::endl;
      forward_flag = forward_flag && backward_flag;
      break;
    }
    if(abs(last_cost - mCost) <= mTolerence)
    {
      break;
    }
  }

  if(forward_flag)
  {
    // Update the mBuffer according to current time
    mlqrBuffer.setNewActionPlan(startTime, &mBuffer);
    mlqrBuffer.setNewControlLaw(startTime, &mBuffer);

    // Reset Parameters
    mPatience = mPatience_reset_value;
    mMU = mMU_reset_value;
    mDelta = mDelta0;
    last_loss = std::numeric_limits<s_t>::max();
    mAlpha = mAlpha_reset_value;
    return true;
  }
  return false;
}

void iLQRLocal::setPatience(int patience)
{
  mPatience = patience;
  mPatience_reset_value = patience;
}

void iLQRLocal::setTolerence(s_t tolerence)
{
  mTolerence = tolerence;
}

int iLQRLocal::getPatience()
{
  return mTolerence;
}

void iLQRLocal::setAlpha(s_t alpha)
{
  mAlpha = alpha;
  mAlpha_reset_value = alpha;
}

Eigen::VectorXs iLQRLocal::getControlk(long now)
{
  return mBuffer.getPlannedk(now);
}

Eigen::MatrixXs iLQRLocal::getControlK(long now)
{
  return mBuffer.getPlannedK(now);
}

Eigen::VectorXs iLQRLocal::getControlState(long now)
{
  return mBuffer.getPlannedState(now);
}

void iLQRLocal::setTargetReachingCostFn(std::shared_ptr<TargetReachingCost> costFn)
{
  mCostFn = costFn;
  mLoss = costFn->getLossFn();
}

Eigen::MatrixXs iLQRLocal::assembleJacobianMatrix(Eigen::MatrixXs B)
{
  size_t ndim = mActuatedJoint.size();
  Eigen::MatrixXs Bnew = Eigen::MatrixXs::Zero(ndim, ndim);
  for(size_t i = 0; i < Bnew.cols(); i++)
  {
    for(size_t j = 0; j < Bnew.cols(); j++)
    {
      Bnew(i,j) = B((size_t)mActuatedJoint(i),(size_t)mActuatedJoint(j));
    }
  }
  return Bnew;
}

/// This registers a listener to get called when we finish replanning
void iLQRLocal::registerReplanningListener(
    std::function<void(long, const trajectory::TrajectoryRollout*, long)>
        replanListener)
{
  mReplannedListeners.push_back(replanListener);
}

/// This launches a server on the specified port
void iLQRLocal::serve(int port)
{
  std::string server_address("0.0.0.0:" + std::to_string(port));

  grpc::EnableDefaultHealthCheckService(true);
  // grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  RPCWrapperiLQRLocal wrapper(*this);

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

RPCWrapperiLQRLocal::RPCWrapperiLQRLocal(iLQRLocal& local) : mLocal(local)
{
}

/// Remotely start the compute running
grpc::Status RPCWrapperiLQRLocal::Start(
    grpc::ServerContext* /* context */,
    const proto::MPCStartRequest* /* request */,
    proto::MPCStartReply* /* response */)
{
  mLocal.start();
  return grpc::Status::OK;
}

/// Remotely stop the compute running
grpc::Status RPCWrapperiLQRLocal::Stop(
    grpc::ServerContext* /* context */,
    const proto::MPCStopRequest* /* request */,
    proto::MPCStopReply* /* response */)
{
  mLocal.stop();
  return grpc::Status::OK;
}

/// Remotely listen for replanning updates
grpc::Status RPCWrapperiLQRLocal::ListenForUpdates(
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
grpc::Status RPCWrapperiLQRLocal::RecordGroundTruthState(
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
grpc::Status RPCWrapperiLQRLocal::ObserveForce(
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
void iLQRLocal::optimizationThreadLoop()
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

/// This is the function which use iLQR as trajectory optimization
void iLQRLocal::iLQRoptimizationThreadLoop()
{
  // block signals in this thread and subsequentially
  // spawned threads, so they're guaranteed to go to the server thread
  sigset_t sigset;
  sigemptyset(&sigset);
  sigaddset(&sigset, SIGINT);
  sigaddset(&sigset, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigset, nullptr);

  while(mRunning)
  {
    long startTime = timeSinceEpochMillis();
    bool status = ilqroptimizePlan(startTime + mMillisInAdvanceToPlan);
    if(!status)
    {
      std::cout << "iLQR Fail to solve the problem !!" << std::endl;
      mRunning = false;
      break;
    }
    long endTime = timeSinceEpochMillis();
    adjustPerformance(endTime - startTime);
  }
}

bool iLQRLocal::variableChange()
{
  return mVarchange;
}

void iLQRLocal::setMasschange(s_t mass)
{
  if(abs(mass-pre_mass)>0.001)
  {
    mVarchange = true;
  }
  pre_mass = mass;
}

void iLQRLocal::setCOMchange(Eigen::Vector3s com)
{
  if((com-pre_com).norm()>0.001)
  {
    mVarchange = true;
  }
  pre_com = com;
}

void iLQRLocal::setMOIchange(Eigen::Vector6s moi)
{
  if((moi - pre_moi).norm()>0.001)
  {
    mVarchange = true;
  }
  pre_moi = moi;
}

void iLQRLocal::setMUchange(s_t mu)
{
  if(abs(mu-pre_mu) > 0.001)
  {
    mVarchange = true;
  }
  pre_mu = mu;
}

} // namespace realtime
} // namespace dart