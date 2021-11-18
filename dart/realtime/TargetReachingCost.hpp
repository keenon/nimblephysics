#ifndef DART_REALTIME_TargetReachingCost
#define DART_REALTIME_TargetReachingCost

#include <memory>
#include <Eigen/Dense>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart{

namespace trajectory {
class LossFn;
class TrajectoryRollout;
class TrajectoryRolloutReal;
}

namespace realtime {

enum WRTFLAG {
  X,
  U,
  XX,
  UX,
  UU,
  XU
};

class TargetReachingCost
{
public:
  // Constructor using weight of different states
  TargetReachingCost(
      Eigen::VectorXs runningStateWeight,
      Eigen::VectorXs runningActionWeight,
      Eigen::VectorXs finalStateWeight,
      Eigen::VectorXi actuatedJoint);

  // API for iLQR Gradient and Loss compute
  // It should call protected function
  std::vector<Eigen::VectorXs> ilqrGradientEstimator(const trajectory::TrajectoryRollout* rollout,
                                                     s_t& total_cost,
                                                     WRTFLAG wrt,
                                                     s_t dt);

  // API for iLQR Hessian compute
  // It should call protected function
  std::vector<Eigen::MatrixXs> ilqrHessianEstimator(const trajectory::TrajectoryRollout* rollout,
                                                    WRTFLAG wrt,
                                                    s_t dt);

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

protected:
  // Compute Loss and Gradient
  s_t computeLoss(const trajectory::TrajectoryRollout* rollout);

  void computeGradX(const trajectory::TrajectoryRollout* rollout, Eigen::Ref<Eigen::MatrixXs> grads);

  void computeGradU(const trajectory::TrajectoryRollout* rollout, Eigen::Ref<Eigen::MatrixXs> grads);

  // Compute Hessian from trajectory
  void computeHessXX(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);

  void computeHessUU(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);

  void computeHessUX(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);

  void computeHessXU(const trajectory::TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess);

  // Internal information
  Eigen::VectorXs mRunningStateWeight;

  Eigen::VectorXs mRunningActionWeight;

  Eigen::VectorXs mRunningActionWeightFull;

  Eigen::VectorXs mFinalStateWeight;

  Eigen::VectorXi mActuatedJoint;

  // Target

  Eigen::VectorXs mTarget;

  int mStateDim;

  int mActionDim;

  bool mUseFullAction;

};

}
}



#endif