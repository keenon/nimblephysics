#ifndef DART_REALTIME_iLQRLocal
#define DART_REALTIME_iLQRLocal

#include <memory>
#include <thread>

#include <Eigen/Dense>

#include "dart/proto/MPC.grpc.pb.h"
#include "dart/realtime/MPC.hpp"
#include "dart/realtime/RealTimeControlBuffer.hpp"
#include "dart/realtime/ObservationLog.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/TargetReachingCost.hpp"
#include "dart/simulation/SmartPointer.hpp"
#include "dart/neural/BackpropSnapshot.hpp"

using namespace grpc;

namespace dart {

namespace simulation {
class World;
}

namespace neural {
  class BackpropSnapShot;
}

namespace trajectory {
class LossFn;
class Solution;
class Problem;
class Optimizer;
class TrajectoryRollout;
class TrajectoryRolloutReal;
}

namespace realtime {

enum Extrapolate_Method {
  ZERO,
  LAST,
  RANDOM};

class LQRBuffer
{
public:
  LQRBuffer(
    int steps, 
    size_t nDofs, 
    size_t nControls, 
    Extrapolate_Method extrapolate);

  void resetXUNew();

  void updateXUOld();

  void readNewActionPlan(long timestamp, RealTimeControlBuffer buffer);

  // Write New Action to Buffer
  void setNewActionPlan(long timestamp, RealTimeControlBuffer *buffer);

  void setNewControlLaw(long timestamp, RealTimeControlBuffer *buffer);

  void updateL(std::vector<Eigen::VectorXs> Lx_new, std::vector<Eigen::VectorXs> Lu_new,
               std::vector<Eigen::MatrixXs> Lxx_new, std::vector<Eigen::MatrixXs> Luu_new,
               std::vector<Eigen::MatrixXs> Lux_new);

  void updateF(std::vector<Eigen::MatrixXs> Fx_new, std::vector<Eigen::MatrixXs> Fu_new);

  bool validateXnew();


  // Parameters
  size_t nsteps;
  size_t control_dim;
  size_t state_dim;
  Extrapolate_Method ext; 

  // system related
  std::vector<Eigen::VectorXs> X;
  std::vector<Eigen::VectorXs> Xnew;
  std::vector<Eigen::VectorXs> U;
  std::vector<Eigen::VectorXs> Unew;
  std::vector<Eigen::MatrixXs> K;
  std::vector<Eigen::VectorXs> k;
  // jacobians
  std::vector<Eigen::MatrixXs> Fx;
  std::vector<Eigen::MatrixXs> Fu;

  // Gradients
  std::vector<Eigen::VectorXs> Lx;
  std::vector<Eigen::VectorXs> Lu;

  // Hessians
  std::vector<Eigen::MatrixXs> Lxx;
  std::vector<Eigen::MatrixXs> Lux;
  std::vector<Eigen::MatrixXs> Luu;
};


class iLQRLocal final : public MPC
{
public:
  iLQRLocal(
      std::shared_ptr<simulation::World> world,
      std::shared_ptr<TargetReachingCost> costFn,
      size_t nControls,
      int planningHorizonMillis,
      s_t scale);

  /// This sets the optimizer for trajectory optimization
  void setOptimizer(std::shared_ptr<trajectory::Optimizer> optimizer);

  /// This returns the current optimizer that MPC is using
  std::shared_ptr<trajectory::Optimizer> getOptimizer();

  /// This sets the problem for MPC
  void setProblem(std::shared_ptr<trajectory::Problem> problem);

  /// This returns the problem
  std::shared_ptr<trajectory::Problem> getProblem();

  /// This get Control Force from current timestep
  Eigen::VectorXs getControlForce(long now) override;

  /// This get feed forward term k from current timestep
  Eigen::VectorXs getControlk(long now);

  /// This get State based on time step
  Eigen::VectorXs getControlState(long now);

  /// This get feedback matrix from current timestep
  Eigen::MatrixXs getControlK(long now);

  Eigen::VectorXs computeForce(Eigen::VectorXs state, long now);

  /// This returns how many millis we have left until we've run out of plan
  long getRemainingPlanBufferMillis() override;

  /// This can completely silence log output
  void setSilent(bool silent);

  /// This enable linesearch on the IPOPT sub-problem or iLQR. Defaults to true
  void setEnableLineSearch(bool enabled);

  /// This enable optimization guards of trajectory optimization.
  void setEnableOptimizationGuards(bool enabled);

  /// Record iterations
  int getMaxIterations();

  /// This sets the current maximum number of iterations for trajopt or iLQR
  void setMaxIterations(int maxIters);

  /// This set record iteration
  void setRecordIterations(bool enabled);

  /// This record the current state
  void recordGroundTruthState(
      long time,
      Eigen::VectorXs pos,
      Eigen::VectorXs vel,
      Eigen::VectorXs mass) override;

  /// This optimize the plan starting at starttime
  void optimizePlan(long startTime);

  /// This will adjust parameters to make sure we are upto date
  void adjustPerformance(long lastOptimizeTimeMillis);

  std::shared_ptr<trajectory::Solution> getCurrentSolution();

  /// =========================================================
  /// ==== Here are funcitons for iLQR ========================
  /// This will run forward pass from starttime and record info
  /// such as Jacobian of each step and gradient as well as
  /// hessian from cost function
  bool ilqrForward(simulation::WorldPtr world);

  bool ilqrBackward();

  bool ilqroptimizePlan(long startTime);

  void setTargetReachingCostFn(std::shared_ptr<TargetReachingCost> costFn);

  /// Get the jacobian for linearization
  void getTrajectory(simulation::WorldPtr world,
                     trajectory::TrajectoryRollout* rollout,
                     LQRBuffer* lqrBuffer);

  void setPatience(int patience);   

  void setTolerence(s_t tolerence);

  int getPatience();

  /// set initial action learning rate, prevent numerical error
  void setAlpha(s_t alpha);

  s_t getAlpha();

  void setMU(s_t mu);

  s_t getMU();

  /// ==========================================================

  void registerReplanningListener(
      std::function<void(long, const trajectory::TrajectoryRollout*, long)>
        replanListener) override;
  
  void serve(int port);

  bool variableChange();

  void setMasschange(s_t mass);

  void setCOMchange(Eigen::Vector3s com);

  void setMOIchange(Eigen::Vector6s moi);

  void setMUchange(s_t mu);

  void start() override;

  void ilqrstart();

  void stop() override;

  void ilqrstop();

  // For Debug
  std::vector<Eigen::VectorXs> getStatesFromiLQRBuffer();

  std::vector<Eigen::VectorXs> getActionsFromiLQRBuffer();

  void setCurrentCost(s_t cost);

  s_t getCurrentCost();

  trajectory::TrajectoryRolloutReal createRollout(size_t steps, size_t dofs, size_t mass_dim);

  void setActionBound(s_t actionBound);

protected:
    void optimizationThreadLoop();

    void iLQRoptimizationThreadLoop();

    Eigen::MatrixXs assembleJacobianMatrix(Eigen::MatrixXs B);

    bool mRunning;
    std::shared_ptr<simulation::World> mWorld;
    std::shared_ptr<trajectory::LossFn> mLoss;
    ObservationLog mObservationLog;

    // Meta Config
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
    // Cost function instance
    std::shared_ptr<TargetReachingCost> mCostFn;
    

    RealTimeControlBuffer mBuffer;
    LQRBuffer mlqrBuffer;
    s_t mCost;
    // Some Parameters of ilqr
    Eigen::VectorXi mActuatedJoint;
    s_t mAlpha_reset_value;
    s_t mAlpha;
    int mPatience_reset_value;
    int mPatience;
    int mDelta0;
    int mDelta;
    s_t mMU_MIN;
    s_t mMU;
    s_t mMU_reset_value;
    s_t mTolerence;
    int mActionDim;
    int mStateDim;
    s_t mActionBound = 1000; // Default 1000 or numerical issue will occur


    std::thread mOptimizationThread;
    bool mSilent;
    // Some history track for compare and set in SSID
    s_t pre_mass;
    s_t pre_mu;
    Eigen::Vector3s pre_com;
    Eigen::Vector6s pre_moi;
    bool mVarchange = false;

    Eigen::VectorXs mLast_U;

    /// For Trajectory Optimization
    std::shared_ptr<trajectory::Optimizer> mOptimizer;
    std::shared_ptr<trajectory::Solution> mSolution;
    std::shared_ptr<trajectory::Problem> mProblem;

    std::vector<std::function<void(long, const trajectory::TrajectoryRollout*, long)>>
        mReplannedListeners;

    friend class RPCWrapperiLQRLocal;
};

class RPCWrapperiLQRLocal : public proto::MPCService::Service
{
public:
  RPCWrapperiLQRLocal(iLQRLocal& local);
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
  iLQRLocal& mLocal;
};

} // namespace realtime
} // namespace dart

#endif