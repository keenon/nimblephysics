#include <Eigen/Dense>
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/TargetReachingCost.hpp"

namespace dart {
using namespace trajectory;

namespace realtime {

TargetReachingCost::TargetReachingCost(
  Eigen::VectorXs runningStateWeight,
  Eigen::VectorXs runningActionWeight,
  Eigen::VectorXs finalStateWeight,
  std::shared_ptr<simulation::World> world):
  mRunningStateWeight(runningStateWeight),
  mRunningActionWeight(runningActionWeight),
  mFinalStateWeight(finalStateWeight),
  mStateDim(runningStateWeight.size()),
  mActionDim(runningActionWeight.size()),
  mWorld(world)
{
  std::cout << "State and Action Dim in Cost Fn: " << mStateDim << " " << mActionDim << std::endl;
  mRunningActionWeight = runningActionWeight;
}

std::vector<Eigen::VectorXs> TargetReachingCost::ilqrGradientEstimator(
  const TrajectoryRollout* rollout,
  s_t& total_cost,
  WRTFLAG wrt)
{
  total_cost = computeLoss(rollout);
  int steps = rollout->getPosesConst().cols();
  Eigen::MatrixXs grad;
  if(wrt == WRTFLAG::X)
  {
    grad = Eigen::MatrixXs::Zero(mStateDim, steps);
    computeGradX(rollout, grad);
  }
  else if(wrt == WRTFLAG::U)
  {
    grad = Eigen::MatrixXs::Zero(mActionDim, steps-1);
    computeGradU(rollout, grad);
  }
  else
    assert(false && "Shouldn't reach here");
  std::vector<Eigen::VectorXs> output;

  for(int i = 0; i < grad.cols(); i++)
  {
    Eigen::VectorXs col = grad.col(i);
    output.push_back(col);
  }
  return output;
}

std::vector<Eigen::MatrixXs> TargetReachingCost::ilqrHessianEstimator(
  const TrajectoryRollout* rollout,
  WRTFLAG wrt)
{
  std::vector<Eigen::MatrixXs> hess;
  switch (wrt)
  {
  case WRTFLAG::U:
  case WRTFLAG::X:
    assert(false && "Should not use grad flag");
    break;
  case WRTFLAG::XX:
    computeHessXX(rollout, hess);
    break;
  case WRTFLAG::UU:
    computeHessUU(rollout, hess);
    break;
  case WRTFLAG::UX:
    computeHessUX(rollout, hess);
    break;
  case WRTFLAG::XU:
    computeHessXU(rollout, hess);
    break;
  }
  return hess;
}

s_t TargetReachingCost::loss(const TrajectoryRollout* rollout)
{
  return computeLoss(rollout);
}

s_t TargetReachingCost::lossGrad(const TrajectoryRollout* rollout, TrajectoryRollout* gradWrtRollout)
{
  int steps = rollout->getPosesConst().cols();
  Eigen::MatrixXs grad_x = Eigen::MatrixXs::Zero(mStateDim, steps);
  Eigen::MatrixXs grad_u = Eigen::MatrixXs::Zero((int)(mStateDim/2), steps);
  computeGradX(rollout, grad_x);
  computeGradForce(rollout, grad_u.block(0, 0, (int)(mStateDim/2), steps-1));
  gradWrtRollout->getPoses().setZero();
  gradWrtRollout->getVels().setZero();
  gradWrtRollout->getControlForces().setZero();
  gradWrtRollout->getPoses().block(0, 0, (int)(mStateDim/2), steps)
    = grad_x.block(0, 0, (int)(mStateDim/2), steps);
  gradWrtRollout->getVels().block(0, 0, (int)(mStateDim/2), steps)
    = grad_x.block((int)(mStateDim/2), 0, (int)(mStateDim/2), steps);
  gradWrtRollout->getControlForces().block(0, 0, (int)(mStateDim/2), steps)
    = grad_u;
  return computeLoss(rollout);
}

std::shared_ptr<LossFn> TargetReachingCost::getLossFn()
{
  boost::function<s_t(const TrajectoryRollout*)> 
         loss_(boost::bind(&TargetReachingCost::loss, this, _1));
  boost::function<s_t(const TrajectoryRollout*, TrajectoryRollout*)>
         lossGrad_(boost::bind(&TargetReachingCost::lossGrad, this, _1, _2));
  return std::make_shared<LossFn>(loss_, lossGrad_);
}

void TargetReachingCost::setTarget(Eigen::VectorXs target)
{
  assert(target.size() == mStateDim);
  mTarget = target;
}

void TargetReachingCost::setTimeStep(s_t timestep)
{
  assert(timestep > 0.001);
  dt = timestep;
}

/// Must be set
void TargetReachingCost::setSSIDMassNodeIndex(Eigen::VectorXi mass_indices)
{
  mMass_indices = mass_indices;
  for(int i = 0; i < mass_indices.size(); i++)
  {
    mAks.push_back(std::vector<Eigen::MatrixXs>());
  }
}

void TargetReachingCost::setSSIDSpringJointIndex(Eigen::VectorXi spring_indices)
{
  mSpring_indices = spring_indices;
}

void TargetReachingCost::enableSSIDLoss(s_t weight)
{
  mUseSSIDHeuristic = true;
  mSSIDHeuristicWeight = weight;
}

void TargetReachingCost::setSSIDHeuristicWeight(s_t weight)
{
  mSSIDHeuristicWeight = weight;
}

// For the Velocity we can use loss or not use
s_t TargetReachingCost::computeLoss(const TrajectoryRollout* rollout)
{
  s_t loss = 0;
  int steps = rollout->getPosesConst().cols();
  // Compute Running State and Loss from target reaching
  Eigen::VectorXs init_state = mWorld->getState();
  for(int i = 0; i < mAks.size(); i++)
  {
    mAks[i].clear();
  }
  for(int i = 0; i < steps - 1; i++)
  {
    Eigen::VectorXs state = Eigen::VectorXs::Zero(mStateDim);
    Eigen::VectorXs action = mWorld->mapToActionSpaceVector(rollout->getControlForcesConst().col(i));

    state.segment(0, (int)(mStateDim/2)) = rollout->getPosesConst().col(i);
    state.segment((int)(mStateDim/2), (int)(mStateDim/2)) = rollout->getVelsConst().col(i);
    
    loss += (mRunningStateWeight.asDiagonal() * ((state - mTarget).cwiseAbs2())).sum() * dt;
    loss += (mRunningActionWeight.asDiagonal() * (action.cwiseAbs2())).sum() * dt;
    if(mUseSSIDHeuristic)
    {
      // if(i>=1)
      // {
      //   Eigen::VectorXs prev_state = Eigen::VectorXs::Zero(mStateDim);
      //   prev_state.segment(0, (int)(mStateDim/2)) = rollout->getPosesConst().col(i-1);
      //   prev_state.segment((int)(mStateDim/2), (int)(mStateDim/2)) = rollout->getVelsConst().col(i-1);
      //   loss += computeSSIDMassLoss(state, prev_state);
      // }
      // TODO : Add other heuristics
      loss = (1-mSSIDHeuristicWeight) * loss + mSSIDHeuristicWeight * computeSSIDSpringStiffLoss(state);
      // std::cout<< "Loss: " << computeSSIDSpringStiffLoss(state) << std::endl;
    }
    
  }
  mWorld->setState(init_state);
  // std::cout<< "None Final Loss: " << loss << std::endl;
  // Add Final State Error
  Eigen::VectorXs finalState = Eigen::VectorXs::Zero(mStateDim);
  finalState.segment(0, (int)(mStateDim/2)) = rollout->getPosesConst().col(steps - 1);
  finalState.segment((int)(mStateDim/2), (int)(mStateDim/2)) = rollout->getVelsConst().col(steps - 1);
  s_t final_loss = (mFinalStateWeight.asDiagonal() * ((finalState - mTarget).cwiseAbs2())).sum();
  loss += (1 - mSSIDHeuristicWeight) * final_loss;
  return loss;
}

void TargetReachingCost::computeGradX(const TrajectoryRollout* rollout, Eigen::Ref<Eigen::MatrixXs> grads)
{
  int steps = rollout->getPosesConst().cols();
  assert(grads.cols() == steps && grads.rows() == mStateDim);
  // Compute Grad for running state
  for(int i = 0; i < steps - 1; i++)
  {
    Eigen::VectorXs state = Eigen::VectorXs::Zero(mStateDim);
    //std::cout<< "State Dim: " << mStateDim << "Half Dim: " << (int)(mStateDim/2) << std::endl;
    state.segment(0, (int)(mStateDim/2)) = rollout->getPosesConst().col(i);
    state.segment((int)(mStateDim/2), (int)(mStateDim/2)) = rollout->getVelsConst().col(i);

    grads.col(i) = 2*mRunningStateWeight.asDiagonal() * (state - mTarget) * dt;
    if(mUseSSIDHeuristic)
    {
      // if(i>= 1)
      // {
      //   Eigen::VectorXs prev_state = Eigen::VectorXs::Zero(mStateDim); 
      //   prev_state.segment(0, (int)(mStateDim/2)) = rollout->getPosesConst().col(i-1);
      //   prev_state.segment((int)(mStateDim/2), (int)(mStateDim/2)) = rollout->getVelsConst().col(i-1);
      //   grads.col(i) += computeSSIDMassGrad(state, prev_state, i);
      // }
      grads.col(i) =(1-mSSIDHeuristicWeight)*grads.col(i) + mSSIDHeuristicWeight * computeSSIDSpringStiffGrad(state);
      // std::cout<< "Grad: " << computeSSIDSpringStiffGrad(state) << std::endl;

    }
  }
  // Compute Grad for final state
  Eigen::VectorXs state = Eigen::VectorXs::Zero(mStateDim);
  state.segment(0, (int)(mStateDim/2)) = rollout->getPosesConst().col(steps - 1);
  state.segment((int)(mStateDim/2), (int)(mStateDim/2)) = rollout->getVelsConst().col(steps - 1);
  grads.col(steps - 1) = (1 - mSSIDHeuristicWeight) * 2 * mFinalStateWeight.asDiagonal() * (state - mTarget);
}

void TargetReachingCost::computeGradU(const TrajectoryRollout* rollout,
                                    Eigen::Ref<Eigen::MatrixXs> grads)
{
  int steps = rollout->getPosesConst().cols();
  assert(grads.cols() == steps - 1 && grads.rows() == mActionDim);
  
  // Compute Grad for Running action
  for(int i = 0; i < steps - 1; i++)
  {
    Eigen::VectorXs action;    
    action = mWorld->mapToActionSpaceVector(rollout->getControlForcesConst().col(i));
    grads.col(i) = 2 * mRunningActionWeight.asDiagonal() * action * dt;
    
  }
}

// Here compute gradients of actions which will not have final steps
void TargetReachingCost::computeGradForce(const TrajectoryRollout* rollout, 
                                      Eigen::Ref<Eigen::MatrixXs> grads)
{
  int steps = rollout->getPosesConst().cols();
  assert(grads.cols() == steps - 1 && grads.rows() == (int)(mStateDim/2));
  
  // Compute Grad for Running action
  for(int i = 0; i < steps - 1; i++)
  {
    Eigen::VectorXs force;    
    force = rollout->getControlForcesConst().col(i);
    grads.col(i) = 2 * mWorld->mapToForceSpaceVector(mRunningActionWeight).asDiagonal() * force * dt;
    
  }
}

// Compute Hessian of XX which should have length of steps
// Assume hess is an empty vector
void TargetReachingCost::computeHessXX(
    const TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess)
{
  assert(hess.size() == 0);
  int steps = rollout->getPosesConst().cols();
  Eigen::MatrixXs hess_xx;
  for(int i = 0; i < steps - 1; i++)
  {
    hess_xx = 2*mRunningStateWeight.asDiagonal() * dt;
    if(mUseSSIDHeuristic)
    {
      // if(i>=1)
      // {
      //   hess_xx += computeSSIDMassHess(i);
      // }
      hess_xx = hess_xx*(1-mSSIDHeuristicWeight) + mSSIDHeuristicWeight * computeSSIDSpringStiffHess();
      // std::cout<< "Hess: " << computeSSIDSpringStiffHess() << std::endl;
    }
    hess.push_back(hess_xx);
  }
  Eigen::MatrixXs final_hess_xx =(1-mSSIDHeuristicWeight) * 2*mFinalStateWeight.asDiagonal();
  hess.push_back(final_hess_xx);
}

// There is no cross terms between x and u hence the hessian should be all zero
void TargetReachingCost::computeHessXU(
    const TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess)
{
  assert(hess.size() == 0);
  int steps = rollout->getPosesConst().cols();

  for(int i = 0; i < steps - 1; i++)
  {
    hess.push_back(Eigen::MatrixXs::Zero(mActionDim, mStateDim) * dt);
  }
}

// There is no cross terms between u and x hence the hessian should be all zero
void TargetReachingCost::computeHessUX(
  const TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess)
{
  assert(hess.size() == 0);
  int steps = rollout->getPosesConst().cols();
  for(int i = 0; i < steps - 1; i++)
  {
    hess.push_back(Eigen::MatrixXs::Zero(mStateDim,mActionDim) * dt);
  }
}

void TargetReachingCost::computeHessUU(
  const TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess)
{
  assert(hess.size() == 0);
  int steps = rollout->getPosesConst().cols();
  Eigen::MatrixXs hess_uu;

  for(int i = 0; i < steps - 1; i++)
  {
    hess_uu = 2*mRunningActionWeight.asDiagonal() * dt;
    hess.push_back(hess_uu);
  }
}

s_t TargetReachingCost::computeSSIDMassLoss(Eigen::VectorXs state, Eigen::VectorXs prev_state)
{
  s_t loss = 0;
  int dofs = (int)(state.size() / 2);
  mWorld->setState(state);
  for(int j = 0; j < mMass_indices.size(); j++)
  {
    Eigen::MatrixXs Ak = mWorld->getLinkAkMatrixIndex(mMass_indices(j));
    mAks[j].push_back(Ak);
    loss += (Ak * (state.segment(dofs, dofs) - prev_state.segment(dofs, dofs))).cwiseAbs2().sum() * dt;
  }
  
  return loss;
}

// TODO: Eric After deadline implement such function on all parameters
// s_t computeSSIDDampCoeffLoss()

s_t TargetReachingCost::computeSSIDSpringStiffLoss(Eigen::VectorXs state)
{
  s_t loss = 0;
  Eigen::VectorXs err = Eigen::VectorXs::Zero(mSpring_indices.size());
  for(int j = 0; j < mSpring_indices.size(); j++)
  {
    s_t rest_pose = mWorld->getRestPositionIndex(mSpring_indices(j));
    err(j) = state(mSpring_indices(j)) - rest_pose;
  }
  loss -= err.norm();
  return loss;
}

Eigen::VectorXs TargetReachingCost::computeSSIDMassGrad(Eigen::VectorXs state, Eigen::VectorXs prev_state, int cur)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(state.size());
  int dofs = (int)(state.size() / 2);
  for(int j = 0; j < mMass_indices.size(); j++)
  {
    Eigen::VectorXs new_grad = 2 * (mAks[j][cur-1].transpose() * mAks[j][cur-1]) * (state.segment(dofs, dofs) - prev_state.segment(dofs, dofs)) * dt;
    grad += new_grad;
  }
  return grad;
}

Eigen::VectorXs TargetReachingCost::computeSSIDSpringStiffGrad(Eigen::VectorXs state)
{
  Eigen::VectorXs grad = Eigen::VectorXs::Zero(state.size());
  for(int j = 0; j < mSpring_indices.size(); j++)
  {
    grad(mSpring_indices(j)) = 2 * (state(mSpring_indices(j)) - mWorld->getRestPositionIndex(mSpring_indices(j)));
  }
  return -grad;
}

Eigen::MatrixXs TargetReachingCost::computeSSIDMassHess(int cur)
{
  Eigen::MatrixXs hess = Eigen::MatrixXs::Zero(mStateDim, mStateDim);
  for(int j = 0; j < mMass_indices.size(); j++)
  {
    hess.block((int)(mStateDim/2), (int)(mStateDim/2), (int)(mStateDim/2), (int)(mStateDim/2))
        +=  mAks[j][cur-1].transpose() * mAks[j][cur-1] * dt;
  }
  return hess;
}

Eigen::MatrixXs TargetReachingCost::computeSSIDSpringStiffHess()
{
  Eigen::MatrixXs hess = Eigen::MatrixXs::Zero(mStateDim, mStateDim);
  for(int j = 0; j < mSpring_indices.size(); j++)
  {
    hess(mSpring_indices(j), mSpring_indices(j)) = 1;
  }
  return -hess;
}

}
}