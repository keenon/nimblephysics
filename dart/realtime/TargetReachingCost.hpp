#ifndef DART_REALTIME_TargetReachingCost
#define DART_REALTIME_TargetReachingCost

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

  // TODO: Change the name
  void setSSIDMassNodeIndex(Eigen::VectorXi indices);

  // TODO: Implement it
  void setSSIDCOMNodeIndex(Eigen::VectorXi indices);

  // void setSSIDMOINodeIndex(Eigen::VectorXi indices);

  void setSSIDDampJointIndex(Eigen::VectorXi indices);

  void setSSIDSpringJointIndex(Eigen::VectorXi indices);

  void enableSSIDLoss(s_t weight);

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

  // Related to Heuristic Gradient and weight computation
  s_t computeSSIDMassLoss();

  s_t computeSSIDCOMLoss();

  s_t computeSSIDDampCoeffLoss();

  s_t computeSSIDSpringStiffLoss();

  Eigen::VectorXs computeSSIDMassGrad(Eigen::VectorXs state, Eigen::VectorXs prev_state);

  Eigen::VectorXs computeSSIDCOMGrad(Eigen::VectorXs state, Eigen::VectorXs prev_state);

  Eigen::VectorXs computeSSIDDampCoeffGrad(Eigen::VectorXs state);

  Eigen::VectorXs computeSSIDSpringCoeffGrad(Eigen::VectorXs state);

  Eigen::MatrixXs computeSSIDMassHess(Eigen::VectorXs state, Eigen::VectorXs prev_state);

  Eigen::MatrixXs computeSSIDCOMHess(Eigen::VectorXs state, Eigen::VectorXs prev_state);

  Eigen::MatrixXs computeSSIDDampCoeffHess(Eigen::VectorXs state);

  Eigen::MatrixXs computeSSIDSpringCoeffHess(Eigen::VectorXs state);

  // Internal information
  Eigen::VectorXs mRunningStateWeight;

  Eigen::VectorXs mRunningActionWeight;

  Eigen::VectorXs mRunningActionWeightFull;

  Eigen::VectorXs mFinalStateWeight;

  std::shared_ptr<simulation::World> mWorld;

  // Target

  Eigen::VectorXs mTarget;

  int mStateDim;

  int mActionDim;

  s_t dt = 1.0;

  // SSID Heuristic
  bool mUseSSIDHeuristic = false;

  s_t mSSIDHeuristicWeight;

  Eigen::VectorXi mass_indices;

  Eigen::VectorXi com_indices;

  Eigen::VectorXi damp_indices;

  Eigen::VectorXi spring_indices;

  std::vector<std::vector<Eigen::MatrixXs>> mAks;

};

}
}



#endif