#include <Eigen/Dense>
#include <math.h>
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/MappedTargetReachingCost.hpp"

namespace dart {
using namespace trajectory;

namespace realtime {

MappedTargetReachingCost::MappedTargetReachingCost(
  Eigen::VectorXs runningStateWeight,
  Eigen::VectorXs runningActionWeight,
  Eigen::VectorXs finalStateWeight,
  std::shared_ptr<simulation::World> world):
  mRunningStateWeight(runningStateWeight),
  mRunningActionWeight(runningActionWeight),
  mFinalStateWeight(finalStateWeight),
  mStateDim(world->getNumDofs() * 2),
  mActionDim(runningActionWeight.size()),
  mWorld(world),
  mMapping(world)
{
  std::cout << "State and Action Dim in Cost Fn: " << mStateDim << " " << mActionDim << std::endl;
  mRunningActionWeight = runningActionWeight;
}

std::vector<Eigen::VectorXs> MappedTargetReachingCost::ilqrGradientEstimator(
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

std::vector<Eigen::MatrixXs> MappedTargetReachingCost::ilqrHessianEstimator(
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

s_t MappedTargetReachingCost::loss(const TrajectoryRollout* rollout)
{
  return computeLoss(rollout);
}

s_t MappedTargetReachingCost::lossGrad(const TrajectoryRollout* rollout, TrajectoryRollout* gradWrtRollout)
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

std::shared_ptr<LossFn> MappedTargetReachingCost::getLossFn()
{
  boost::function<s_t(const TrajectoryRollout*)> 
         loss_(boost::bind(&MappedTargetReachingCost::loss, this, _1));
  boost::function<s_t(const TrajectoryRollout*, TrajectoryRollout*)>
         lossGrad_(boost::bind(&MappedTargetReachingCost::lossGrad, this, _1, _2));
  return std::make_shared<LossFn>(loss_, lossGrad_);
}

void MappedTargetReachingCost::setTarget(Eigen::VectorXs target)
{
  assert(target.size() == mStateDim);
  mTarget = target;
}

void MappedTargetReachingCost::setTimeStep(s_t timestep)
{
  assert(timestep > 0.001);
  dt = timestep;
}

/// Must be set
void MappedTargetReachingCost::setSSIDNodeIndex(std::vector<size_t> ssid_index)
{
  mSSIDNodeIndex = ssid_index;
  for(int i = 0; i < ssid_index.size(); i++)
  {
    mAks.push_back(std::vector<Eigen::MatrixXs>());
  }
}

void MappedTargetReachingCost::setMapping(neural::IKMapping mapping)
{
  mMapping = mapping;
}

void MappedTargetReachingCost::setLinkLength(Eigen::VectorXi lengths)
{
  mLinkLength = lengths;
}

void MappedTargetReachingCost::enableSSIDLoss(s_t weight)
{
  mUseSSIDHeuristic = true;
  mSSIDHeuristicWeight = weight;
}

// For the Velocity we can use loss or not use
// TODO: May need to be modified
s_t MappedTargetReachingCost::computeLoss(const TrajectoryRollout* rollout)
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
    Eigen::VectorXs state = Eigen::VectorXs::Zero(mMappedStateDim);
    Eigen::VectorXs action = mWorld->mapToActionSpaceVector(rollout->getControlForcesConst().col(i));

    state.segment(0, (int)(mMappedStateDim/2)) = getCartesianPos(rollout->getPosesConst().col(i));
    state.segment((int)(mMappedStateDim/2), (int)(mMappedStateDim/2)) 
      = getCartesianVel(rollout->getPosesConst().col(i) ,rollout->getVelsConst().col(i));

    loss += (mRunningStateWeight.asDiagonal() * ((state - mTarget).cwiseAbs2())).sum() * dt;
    loss += (mRunningActionWeight.asDiagonal() * (action.cwiseAbs2())).sum() * dt;
    if(mUseSSIDHeuristic && i >= 1)
    {
      mWorld->setState(state);
      for(int j = 0; j < mSSIDNodeIndex.size(); j++)
      {
        Eigen::MatrixXs Ak = mWorld->getLinkAkMatrixIndex(mSSIDNodeIndex[j]);
        mAks[j].push_back(Ak);
        loss += mSSIDHeuristicWeight * (Ak * (rollout->getVelsConst().col(i) - rollout->getVelsConst().col(i-1))).cwiseAbs2().sum() * dt;
      }
    }
  }
  mWorld->setState(init_state);
  // std::cout<< "None Final Loss: " << loss << std::endl;
  // Add Final State Error
  Eigen::VectorXs finalState = Eigen::VectorXs::Zero(mStateDim);
  finalState.segment(0, (int)(mMappedStateDim/2)) = rollout->getPosesConst().col(steps - 1);
  finalState.segment((int)(mMappedStateDim/2), (int)(mStateDim/2)) = rollout->getVelsConst().col(steps - 1);
  s_t final_loss = (mFinalStateWeight.asDiagonal() * ((finalState - mTarget).cwiseAbs2())).sum();
  // std::cout << "Final Loss:\n " << final_loss << std::endl;
  loss += final_loss;
  return loss;
}

// TODO: Need to be modified need derivation
void MappedTargetReachingCost::computeGradX(const TrajectoryRollout* rollout, Eigen::Ref<Eigen::MatrixXs> grads)
{
  int steps = rollout->getPosesConst().cols();
  assert(grads.cols() == steps && grads.rows() == mStateDim);
  // Compute Grad for running state
  for(int i = 0; i < steps - 1; i++)
  {
    Eigen::VectorXs state = Eigen::VectorXs::Zero(mMappedStateDim);
    Eigen::VectorXs Jac = getStateJacobian(rollout->getPosesConst().col(i),rollout->getVelsConst().col(i));
    //std::cout<< "State Dim: " << mStateDim << "Half Dim: " << (int)(mStateDim/2) << std::endl;
    state.segment(0, (int)(mMappedStateDim/2)) = getCartesianPos(rollout->getPosesConst().col(i));
    state.segment((int)(mMappedStateDim/2), (int)(mMappedStateDim/2)) 
      = getCartesianVel(rollout->getPosesConst().col(i), rollout->getVelsConst().col(i));

    grads.col(i) = Jac * (2*mRunningStateWeight.asDiagonal() * (state - mTarget) * dt);
    if(mUseSSIDHeuristic && i >= 1)
    {
      for(int j = 0; j < mSSIDNodeIndex.size(); j++)
      {
        Eigen::VectorXs new_grad = 2 * mSSIDHeuristicWeight * (mAks[j][i-1].transpose() * mAks[j][i-1]) * (rollout->getVelsConst().col(i) - rollout->getVelsConst().col(i-1)) * dt;
        // std::cout << new_grad.cols() << " " << grads.block((int)(mStateDim/2), i, (int)(mStateDim/2), 1) << std::endl;
        grads.block((int)(mStateDim/2), i, (int)(mStateDim/2), 1) += new_grad;
      }
    }
  }
  // Compute Grad for final state
  Eigen::VectorXs state = Eigen::VectorXs::Zero(mMappedStateDim);
  Eigen::VectorXs Jac = getStateJacobian(rollout->getPosesConst().col(steps-1),rollout->getVelsConst().col(steps-1));
  state.segment(0, (int)(mMappedStateDim/2)) = getCartesianPos(rollout->getPosesConst().col(steps - 1));
  state.segment((int)(mMappedStateDim/2), (int)(mMappedStateDim/2)) 
      = getCartesianVel(rollout->getPosesConst().col(steps-1) ,rollout->getVelsConst().col(steps - 1));
  grads.col(steps - 1) = 2 * mFinalStateWeight.asDiagonal() * (state - mTarget);
}

void MappedTargetReachingCost::computeGradU(const TrajectoryRollout* rollout,
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
void MappedTargetReachingCost::computeGradForce(const TrajectoryRollout* rollout, 
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
// TODO: Need to be modified
void MappedTargetReachingCost::computeHessXX(
    const TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess)
{
  assert(hess.size() == 0);
  int steps = rollout->getPosesConst().cols();
  Eigen::MatrixXs hess_xx;
  for(int i = 0; i < steps - 1; i++)
  {
    Eigen::MatrixXs jac 
        = getStateJacobian(rollout->getPosesConst().col(i),rollout->getVelsConst().col(i));
    Eigen::VectorXs mapped_state = Eigen::VectorXs::Zero(mMappedStateDim);
    Eigen::VectorXs mapped_pos = getCartesianPos(rollout->getPosesConst().col(i));
    Eigen::VectorXs mapped_vel 
        = getCartesianVel(rollout->getPosesConst().col(i), rollout->getVelsConst().col(i));
    mapped_state.segment(0, (int)(mMappedStateDim/2)) 
        = mapped_pos;
    mapped_state.segment((int)(mMappedStateDim/2), (int)(mMappedStateDim/2)) 
        = mapped_vel;
    // hess: mStateDim , mMappedStateDim
    // jac: mMappStateDim, mStateDim
    Eigen::VectorXs diff = mapped_state - mTarget;
    hess_xx = 2 * dt * (mRunningStateWeight(0) * diff(0) * getXHessian(mapped_pos)
                + mRunningStateWeight(1) * diff(1) * getYHessian(mapped_pos)
                + mRunningStateWeight(3) * diff(3) * getVxHessian(mapped_pos, mapped_vel)
                + mRunningStateWeight(4) * diff(4) * getVyHessian(mapped_pos, mapped_vel)
                + jac.transpose() * mRunningStateWeight.asDiagonal() * jac);
    if(mUseSSIDHeuristic && i >= 1)
    {
      for(int j = 0; j < mSSIDNodeIndex.size(); j++)
      {
        hess_xx.block((int)(mStateDim/2), (int)(mStateDim/2), (int)(mStateDim/2), (int)(mStateDim/2))
           += mSSIDHeuristicWeight * mAks[j][i-1].transpose() * mAks[j][i-1] * dt;
      }
    }
    hess.push_back(hess_xx);
  }


  Eigen::MatrixXs jac 
      = getStateJacobian(rollout->getPosesConst().col(steps-1),rollout->getVelsConst().col(steps-1));
  Eigen::VectorXs mapped_state = Eigen::VectorXs::Zero(mMappedStateDim);
  Eigen::VectorXs mapped_pos = getCartesianPos(rollout->getPosesConst().col(steps-1));
  Eigen::VectorXs mapped_vel 
      = getCartesianVel(rollout->getPosesConst().col(steps-1) ,rollout->getVelsConst().col(steps-1));
  mapped_state.segment(0, (int)(mMappedStateDim/2)) 
      = mapped_pos;
  mapped_state.segment((int)(mMappedStateDim/2), (int)(mMappedStateDim/2)) 
      = mapped_vel;
  Eigen::VectorXs diff = mapped_state - mTarget;
  Eigen::MatrixXs final_hess_xx = 2 * dt * (mFinalStateWeight(0) * diff(0) * getXHessian(mapped_pos)
                + mFinalStateWeight(1) * diff(1) * getYHessian(mapped_pos)
                + mFinalStateWeight(3) * diff(3) * getVxHessian(mapped_pos, mapped_vel)
                + mFinalStateWeight(4) * diff(4) * getVyHessian(mapped_pos, mapped_vel) 
              + jac.transpose() * mFinalStateWeight.asDiagonal() * jac); // TODO: One jac need transpose
  hess.push_back(final_hess_xx);
}

// There is no cross terms between x and u hence the hessian should be all zero
void MappedTargetReachingCost::computeHessXU(
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
void MappedTargetReachingCost::computeHessUX(
  const TrajectoryRollout* rollout, std::vector<Eigen::MatrixXs> &hess)
{
  assert(hess.size() == 0);
  int steps = rollout->getPosesConst().cols();
  for(int i = 0; i < steps - 1; i++)
  {
    hess.push_back(Eigen::MatrixXs::Zero(mStateDim,mActionDim) * dt);
  }
}

void MappedTargetReachingCost::computeHessUU(
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

Eigen::VectorXs MappedTargetReachingCost::getCartesianPos(Eigen::VectorXs q_pos)
{
  Eigen::VectorXs start_pos = mWorld->getPositions();
  mWorld->setPositions(q_pos);
  Eigen::VectorXs cart_pos = mMapping.getPositions(mWorld);
  mWorld->setPositions(start_pos);
  return cart_pos;
}

Eigen::VectorXs MappedTargetReachingCost::getCartesianVel(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel)
{
  Eigen::VectorXs start_state = mWorld->getState();
  mWorld->setPositions(q_pos);
  mWorld->setVelocities(q_vel);
  Eigen::VectorXs cart_vel = mMapping.getVelocities(mWorld);
  mWorld->setState(start_state);
  return cart_vel;
}

// This function can directly use Nimble's api
Eigen::MatrixXs MappedTargetReachingCost::getStateJacobian(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel)
{
  Eigen::MatrixXs jac = Eigen::MatrixXs::Zero(mMappedStateDim, mStateDim);
  int mapped_dofs = (int)(mMappedStateDim/2);
  int dofs = (int)(mStateDim/2);
  Eigen::VectorXs start_state = mWorld->getState();
  mWorld->setPositions(q_pos);
  mWorld->setVelocities(q_vel);
  jac.block(0, 0, mapped_dofs, dofs) = mMapping.getRealPosToMappedPosJac(mWorld);
  jac.block(mapped_dofs, dofs, mapped_dofs, dofs) = mMapping.getRealVelToMappedVelJac(mWorld);
  jac.block(mapped_dofs, 0, mapped_dofs, dofs) = mMapping.getRealPosToMappedVelJac(mWorld);
  jac.block(0, dofs, mapped_dofs, dofs) = mMapping.getRealVelToMappedPosJac(mWorld);
  mWorld->setState(start_state);
  return jac;
}

Eigen::MatrixXs MappedTargetReachingCost::getXHessian(Eigen::VectorXs q_pos)
{
  Eigen::MatrixXs hess = Eigen::MatrixXs::Zero(mStateDim, mStateDim);
  int dofs = (int)(mStateDim/2);
  for(int i = 0; i < dofs; i++)
  {
    for(int j = i; j < dofs; j++)
    {
      if(i==0 || j==0)
      {
        hess(i,j) = 0;
      }
      else
      {
        s_t cum_q = 0;
        for(int cur = 1; cur < j+1;cur++)
        {
          cum_q += q_pos(cur);
        }
        for(int k = j; k < dofs;k++)
        {
          hess(i,j) += sin(cum_q) * mLinkLength(k-1);
          hess(j,i) += sin(cum_q) * mLinkLength(k-1);
          cum_q += q_pos(k);
        }
      }
    }
  }
  return hess;
}

Eigen::MatrixXs MappedTargetReachingCost::getYHessian(Eigen::VectorXs q_pos)
{
  Eigen::MatrixXs hess = Eigen::MatrixXs::Zero(mStateDim, mStateDim);
  int dofs = (int)(mStateDim/2);
  for(int i = 0; i < dofs; i++)
  {
    for(int j = i; j < dofs; j++)
    {
      if(i==0 || j==0)
      {
        hess(i,j) = 0;
      }
      else
      {
        s_t cum_q = 0;
        for(int cur = 1; cur < j+1;cur++)
        {
          cum_q += q_pos(cur);
        }
        for(int k = j; k < dofs;k++)
        {
          hess(i,j) -= cos(cum_q) * mLinkLength(k-1);
          hess(j,i) -= cos(cum_q) * mLinkLength(k-1);
          cum_q += q_pos(k);
        }
      }
    }
  }
  return hess;
}

Eigen::MatrixXs MappedTargetReachingCost::getVxHessian(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel)
{
  Eigen::MatrixXs hess = Eigen::MatrixXs::Zero(mStateDim, mStateDim);
  int dofs = (int)(mStateDim/2);
  for(int i = 0; i < dofs; i++)
  {
    s_t global_cum_q = 0;
    s_t global_cum_dq = 0;
    for(int j = i; j < dofs; j++)
    {
      if(i==0 || j==0)
      {
        hess(i,j) = 0;
      }
      else
      {
        s_t cum_q = global_cum_q;
        s_t cum_dq = global_cum_dq;
        for(int k = j; k < dofs;k++)
        {
          cum_q += q_pos(k);
          cum_dq += q_vel(k);
          hess(i,j) += cos(cum_q) * cum_dq * mLinkLength(k-1);
          hess(j,i) += cos(cum_q) * cum_dq * mLinkLength(k-1);
          hess(i, j+dofs) += sin(cum_q) * mLinkLength(k-1);
          hess(j+dofs, i) += sin(cum_q) * mLinkLength(k-1);
          hess(i+dofs, j) += sin(cum_q) * mLinkLength(k-1);
          hess(j, i+dofs) += sin(cum_q) * mLinkLength(k-1);
        }
        global_cum_q += q_pos(j);
        global_cum_dq += q_vel(j);
      }
    }
  }
  return hess;
}

Eigen::MatrixXs MappedTargetReachingCost::getVyHessian(Eigen::VectorXs q_pos, Eigen::VectorXs q_vel)
{
    Eigen::MatrixXs hess = Eigen::MatrixXs::Zero(mStateDim, mStateDim);
  int dofs = (int)(mStateDim/2);
  for(int i = 0; i < dofs; i++)
  {
    s_t global_cum_q = 0;
    s_t global_cum_dq = 0;
    for(int j = i; j < dofs; j++)
    {
      if(i==0 || j==0)
      {
        hess(i,j) = 0;
      }
      else
      {
        s_t cum_q = global_cum_q;
        s_t cum_dq = global_cum_dq;
        for(int cur = 1; cur < j;cur++)
        {
          cum_q += q_pos(cur);
          cum_dq += q_vel(cur);
        }
        for(int k = j; k < dofs;k++)
        {
          cum_q += q_pos(k);
          cum_dq += q_vel(k);
          hess(i,j) += sin(cum_q) * cum_dq * mLinkLength(k-1);
          hess(j,i) += sin(cum_q) * cum_dq * mLinkLength(k-1);
          hess(i, j+dofs) -= cos(cum_q) * mLinkLength(k-1);
          hess(j+dofs, i) -= cos(cum_q) * mLinkLength(k-1);
          hess(i+dofs, j) -= cos(cum_q) * mLinkLength(k-1);
          hess(j, i+dofs) -= cos(cum_q) * mLinkLength(k-1);
        }
        global_cum_q += q_pos(j);
        global_cum_dq += q_vel(j);
      }
    }
  }
  return hess;
}

}
}