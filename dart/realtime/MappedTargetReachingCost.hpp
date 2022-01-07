#ifndef DART_REALTIME_MappedTargetReachingCost
#define DART_REALTIME_MappedTargetReachingCost

#include <memory>
#include <Eigen/Dense>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/simulation/World.hpp"
#include "dart/simulation/SmartPointer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/IKMapping.hpp"
#include "dart/realtime/TargetReachingCost.hpp"

namespace dart{

namespace trajectory {
class LossFn;
class TrajectoryRollout;
class TrajectoryRolloutReal;
}

namespace neural{
class IKMapping;
}

namespace realtime {

class MappedTargetReachingCost
{
public:
  // Constructor using weight of different states
  MappedTargetReachingCost(
      Eigen::VectorXs runningStateWeight,
      Eigen::VectorXs runningActionWeight,
      Eigen::VectorXs finalStateWeight,
      std::shared_ptr<simulation::World> world);

  // API for iLQR Gradient and Loss compute
  // It should call protected function
  std::vector<Eigen::VectorXs> ilqrGradientEstimator(const trajectory::TrajectoryRollout* rollout,
                                                     s_t& total_cost,
                                                     WRTFLAG wrt);

  // API for iLQR Hessian compute
  // It should call protected function
  std::vector<Eigen::MatrixXs> ilqrHessianEstimator(const trajectory::TrajectoryRollout* rollout,
                                                    WRTFLAG wrt);

  // API for IP-OPT LossFn
  // It should call protected function
  s_t loss(const trajectory::TrajectoryRollout* rollout);

  // API for IP-OPT LossAndGradienFn
  // It should call protected function
  s_t lossGrad(const trajectory::TrajectoryRollout* rollout,
                  trajectory::TrajectoryRollout* gradWrtRollout);

  // API to get Loss Function for IP-OPT
  std::shared_ptr<trajectory::LossFn> getLossFn();

  // set target
  void setTarget(Eigen::VectorXs target);

  void setTimeStep(s_t timestep);

  void setSSIDNodeIndex(std::vector<size_t> ssid_index);

  void setMapping(neural::IKMapping mapping);

  void enableSSIDLoss(s_t weight);

  /// This is dedicated for Whip
  void setLinkLength(Eigen::VectorXi lengths);

  s_t computeLoss(const trajectory::TrajectoryRollout* rollout);

  // Compute Loss and Gradient
  void computeGradX(const trajectory::TrajectoryRollout* rollout, Eigen::Ref<Eigen::MatrixXs> grads);

  void computeGradU(const trajectory::TrajectoryRollout* rollout, Eigen::Ref<Eigen::MatrixXs> grads);

  void computeGradForce(const trajectory::TrajectoryRollout* rollout, Eigen::Ref<Eigen::MatrixXs> grads);
  // Compute Hessian from trajectory
  void computeHessXX(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);

  void computeHessUU(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);

  void computeHessUX(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);

  void computeHessXU(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);
  
protected:

  Eigen::VectorXs getCartesianPos(Eigen::VectorXs q_pos);

  Eigen::VectorXs getCartesianVel(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel);

  Eigen::MatrixXs getStateJacobian(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel);

  Eigen::MatrixXs getXHessian(Eigen::VectorXs q_pos);

  Eigen::MatrixXs getYHessian(Eigen::VectorXs q_pos);

  Eigen::MatrixXs getVxHessian(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel);

  Eigen::MatrixXs getVyHessian(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel);

  // Internal information
  Eigen::VectorXs mRunningStateWeight;

  Eigen::VectorXs mRunningActionWeight;

  Eigen::VectorXs mRunningActionWeightFull;

  Eigen::VectorXs mFinalStateWeight;

  Eigen::VectorXi mLinkLength;

  neural::IKMapping mMapping;

  std::shared_ptr<simulation::World> mWorld;

  // Target

  Eigen::VectorXs mTarget;

  int mStateDim;

  int mMappedStateDim = 6;

  int mActionDim;

  s_t dt = 1.0;

  // SSID Heuristic
  bool mUseSSIDHeuristic = false;

  s_t mSSIDHeuristicWeight;

  std::vector<size_t> mSSIDNodeIndex;

  std::vector<std::vector<Eigen::MatrixXs>> mAks;

};

}
}



#endif