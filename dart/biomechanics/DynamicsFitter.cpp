#include "dart/biomechanics/DynamicsFitter.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <utility>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpSolveStatistics.hpp>
#include <coin/IpTNLP.hpp>

#include "dart/biomechanics/C3DForcePlatforms.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/MarkerLabeller.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/AssignmentMatcher.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/AccelerationSmoother.hpp"

namespace dart {
namespace biomechanics {

using namespace Ipopt;

//==============================================================================
ResidualForceHelper::ResidualForceHelper(
    std::shared_ptr<dynamics::Skeleton> skeleton, std::vector<int> forceBodies)
  : mSkel(skeleton)
{
  for (int i : forceBodies)
  {
    mForces.emplace_back(skeleton, i);
  }
}

//==============================================================================
// Computes the residual for a specific timestep
Eigen::Vector6s ResidualForceHelper::calculateResidual(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat)
{
  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  // TODO: there is certainly a more efficient way to do this, since we only
  // care about the first 6 values anyways
  Eigen::MatrixXs M = mSkel->getMassMatrix();
  Eigen::VectorXs C = mSkel->getCoriolisAndGravityForces();
  Eigen::VectorXs Fs = Eigen::VectorXs::Zero(mSkel->getNumDofs());
  for (int i = 0; i < mForces.size(); i++)
  {
    Eigen::VectorXs fTaus
        = mForces[i].computeTau(forcesConcat.segment<6>(i * 6));
    Fs += fTaus;
  }
  Eigen::VectorXs manualTau = M * ddq + C - Fs;

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);

  return manualTau.head<6>();
}

//==============================================================================
// Computes the residual norm for a specific timestep
s_t ResidualForceHelper::calculateResidualNorm(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    s_t torquesMultiple,
    bool useL1)
{
  Eigen::Vector6s residual = calculateResidual(q, dq, ddq, forcesConcat);
  if (useL1)
  {
    return residual.head<3>().norm() * torquesMultiple
           + residual.tail<3>().norm();
  }
  else
  {
    return residual.squaredNorm();
  }
}

//==============================================================================
// Computes the Jacobian of the residual with respect to the first position
Eigen::MatrixXs ResidualForceHelper::calculateResidualJacobianWrt(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    neural::WithRespectTo* wrt)
{
  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getVelocities();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  // Eigen::VectorXs manualTau = M * acc + C - Fs;
  if (wrt == neural::WithRespectTo::POSITION
      || wrt == neural::WithRespectTo::GROUP_SCALES)
  {
    Eigen::MatrixXs dM = mSkel->getJacobianOfM(ddq, wrt);
    Eigen::MatrixXs dC = mSkel->getJacobianOfC(wrt);
    Eigen::MatrixXs dFs
        = Eigen::MatrixXs::Zero(mSkel->getNumDofs(), wrt->dim(mSkel.get()));
    for (int i = 0; i < mForces.size(); i++)
    {
      Eigen::MatrixXs dfTaus
          = mForces[i].getJacobianOfTauWrt(forcesConcat.segment<6>(i * 6), wrt);
      dFs += dfTaus;
    }
    Eigen::MatrixXs jac = dM + dC - dFs;

    mSkel->setPositions(originalPos);
    mSkel->setVelocities(originalVel);
    mSkel->setAccelerations(originalAcc);

    // Only take the first 6 rows
    return jac.block(0, 0, 6, jac.cols());
  }
  else if (
      wrt == neural::WithRespectTo::GROUP_MASSES
      || wrt == neural::WithRespectTo::GROUP_COMS
      || wrt == neural::WithRespectTo::GROUP_INERTIAS)
  {
    Eigen::MatrixXs dM = mSkel->getJacobianOfM(ddq, wrt);
    Eigen::MatrixXs dC = mSkel->getJacobianOfC(wrt);
    Eigen::MatrixXs jac = dM + dC;

    mSkel->setPositions(originalPos);
    mSkel->setVelocities(originalVel);
    mSkel->setAccelerations(originalAcc);

    // Only take the first 6 rows
    return jac.block(0, 0, 6, jac.cols());
  }
  else if (wrt == neural::WithRespectTo::VELOCITY)
  {
    Eigen::MatrixXs dC = mSkel->getJacobianOfC(neural::WithRespectTo::VELOCITY);

    mSkel->setPositions(originalPos);
    mSkel->setVelocities(originalVel);
    mSkel->setAccelerations(originalAcc);

    // Only take the first 6 rows
    return dC.block(0, 0, 6, dC.cols());
  }
  else if (wrt == neural::WithRespectTo::ACCELERATION)
  {
    Eigen::MatrixXs M = mSkel->getMassMatrix();

    mSkel->setPositions(originalPos);
    mSkel->setVelocities(originalVel);
    mSkel->setAccelerations(originalAcc);

    // Only take the first 6 rows
    return M.block(0, 0, 6, M.cols());
  }
  else
  {
    Eigen::MatrixXs J
        = finiteDifferenceResidualJacobianWrt(q, dq, ddq, forcesConcat, wrt);

    mSkel->setPositions(originalPos);
    mSkel->setVelocities(originalVel);
    mSkel->setAccelerations(originalAcc);

    return J;
  }
}

//==============================================================================
// Computes the Jacobian of the residual with respect to the first position
Eigen::MatrixXs ResidualForceHelper::finiteDifferenceResidualJacobianWrt(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    neural::WithRespectTo* wrt)
{
  Eigen::MatrixXs result(6, wrt->dim(mSkel.get()));

  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  Eigen::VectorXs originalWrt = wrt->get(mSkel.get());

  bool useRidders = true;
  s_t eps = useRidders ? 1e-2 : 1e-5;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs newWrt = originalWrt;
        newWrt(dof) += eps;
        wrt->set(mSkel.get(), newWrt);
        perturbed = calculateResidual(
            mSkel->getPositions(),
            mSkel->getVelocities(),
            mSkel->getAccelerations(),
            forcesConcat);
        return true;
      },
      result,
      eps,
      useRidders);

  wrt->set(mSkel.get(), originalWrt);

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);
  return result;
}

//==============================================================================
// Computes the gradient of the residual norm with respect to `wrt`
Eigen::VectorXs ResidualForceHelper::calculateResidualNormGradientWrt(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    neural::WithRespectTo* wrt,
    s_t torquesMultiple,
    bool useL1)
{
  Eigen::Vector6s res = calculateResidual(q, dq, ddq, forcesConcat);
  Eigen::MatrixXs jac
      = calculateResidualJacobianWrt(q, dq, ddq, forcesConcat, wrt);
  if (useL1)
  {
    res.head<3>().normalize();
    res.head<3>() *= torquesMultiple;
    res.tail<3>().normalize();
    return jac.transpose() * res;
  }
  else
  {
    return jac.transpose() * 2 * res;
  }
}

//==============================================================================
// Computes the gradient of the residual norm with respect to `wrt`
Eigen::VectorXs ResidualForceHelper::finiteDifferenceResidualNormGradientWrt(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    neural::WithRespectTo* wrt,
    s_t torquesMultiple,
    bool useL1)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(wrt->dim(mSkel.get()));

  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  Eigen::VectorXs originalWrt = wrt->get(mSkel.get());

  s_t stepSize = 1e-7;
  if (wrt == neural::WithRespectTo::ACCELERATION)
  {
    stepSize = 5e-4;
  }
  if (wrt == neural::WithRespectTo::GROUP_MASSES
      || wrt == neural::WithRespectTo::GROUP_INERTIAS)
  {
    stepSize = 1e-6;
  }

  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::VectorXs newWrt = originalWrt;
        newWrt(dof) += eps;
        wrt->set(mSkel.get(), newWrt);
        perturbed = calculateResidualNorm(
            mSkel->getPositions(),
            mSkel->getVelocities(),
            mSkel->getAccelerations(),
            forcesConcat,
            torquesMultiple,
            useL1);
        return true;
      },
      result,
      // 5e-4,
      stepSize,
      true);

  wrt->set(mSkel.get(), originalWrt);

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);
  return result;
}

//==============================================================================
SpatialNewtonHelper::SpatialNewtonHelper(
    std::shared_ptr<dynamics::Skeleton> skeleton)
  : mSkel(skeleton)
{
}

//==============================================================================
// Computes the f=m*a (in linear components only) difference from observed
// forces, and presents that in 3-axis form (X,Y,Z world coordinates) in
// Newtons.
Eigen::Vector3s SpatialNewtonHelper::calculateLinearForceGap(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat)
{
  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  Eigen::Vector3s f = Eigen::Vector3s::Zero();
  for (int i = 0; i < forcesConcat.size() / 6; i++)
  {
    f += forcesConcat.segment<3>(i * 6 + 3);
  }

  Eigen::Vector3s linearF = Eigen::Vector3s::Zero();
  Eigen::VectorXs worldAccs = mSkel->getCOMWorldLinearAccelerations();
  for (int i = 0; i < worldAccs.size() / 3; i++)
  {
    Eigen::Vector3s a = worldAccs.segment<3>(i * 3);
    a -= mSkel->getGravity();
    s_t m = mSkel->getBodyNode(i)->getMass();
    linearF += m * a;
  }

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);

  return linearF - f;
}

//==============================================================================
// Computes the f=m*a (in linear components only) difference from observed
// forces, and presents that in 3-axis form (X,Y,Z world coordinates) in
// Newtons.
s_t SpatialNewtonHelper::calculateLinearForceGapNorm(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    bool useL1)
{
  Eigen::Vector3s diff = calculateLinearForceGap(q, dq, ddq, forcesConcat);
  if (useL1)
  {
    return diff.norm();
  }
  else
  {
    return diff.squaredNorm();
  }
}

//==============================================================================
// Computes the gradient of gap norm with respect to `wrt`
Eigen::VectorXs SpatialNewtonHelper::calculateLinearForceGapNormGradientWrt(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    neural::WithRespectTo* wrt,
    bool useL1)
{
  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  Eigen::Vector3s diff = calculateLinearForceGap(q, dq, ddq, forcesConcat);

  Eigen::Vector3s gradDiff;
  if (useL1)
  {
    gradDiff = diff.normalized();
  }
  else
  {
    gradDiff = diff * 2;
  }

  Eigen::VectorXs worldAccs;
  Eigen::MatrixXs jac;
  if (wrt == neural::WithRespectTo::GROUP_MASSES)
  {
    worldAccs = mSkel->getCOMWorldLinearAccelerations();
  }
  else
  {
    jac = mSkel->getCOMWorldLinearAccelerationsJacobian(wrt);
  }

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);

  if (wrt == neural::WithRespectTo::GROUP_MASSES)
  {
    Eigen::VectorXs grad = Eigen::VectorXs::Zero(mSkel->getNumScaleGroups());

    for (int i = 0; i < mSkel->getNumScaleGroups(); i++)
    {
      dynamics::BodyScaleGroup group = mSkel->getBodyScaleGroup(i);
      for (int j = 0; j < group.nodes.size(); j++)
      {
        int b = group.nodes[j]->getIndexInSkeleton();
        Eigen::Vector3s a = worldAccs.segment<3>(b * 3);
        a -= mSkel->getGravity();
        grad(i) += a.dot(gradDiff);
      }
    }

    return grad;
  }
  else
  {
    Eigen::MatrixXs condensedJac = Eigen::MatrixXs::Zero(3, jac.cols());
    for (int i = 0; i < mSkel->getNumBodyNodes(); i++)
    {
      condensedJac += mSkel->getBodyNode(i)->getMass()
                      * jac.block(i * 3, 0, 3, jac.cols());
    }
    return condensedJac.transpose() * gradDiff;
  }
}

//==============================================================================
// Computes the gradient of gap norm with respect to `wrt`
Eigen::VectorXs
SpatialNewtonHelper::finiteDifferenceLinearForceGapNormGradientWrt(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    neural::WithRespectTo* wrt,
    bool useL1)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(wrt->dim(mSkel.get()));

  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  Eigen::VectorXs originalWrt = wrt->get(mSkel.get());

  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::VectorXs newWrt = originalWrt;
        newWrt(dof) += eps;
        wrt->set(mSkel.get(), newWrt);
        perturbed = calculateLinearForceGapNorm(
            mSkel->getPositions(),
            mSkel->getVelocities(),
            mSkel->getAccelerations(),
            forcesConcat,
            useL1);
        return true;
      },
      result,
      5e-4,
      true);

  wrt->set(mSkel.get(), originalWrt);

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);
  return result;
}

//==============================================================================
// Computes the norm of the spatial acceleration vector for each body
s_t SpatialNewtonHelper::calculateAccelerationNorm(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs weightBodies,
    bool useL1)
{
  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  s_t sum = 0.0;
  Eigen::VectorXs worldAccs = mSkel->getCOMWorldLinearAccelerations();
  for (int i = 0; i < worldAccs.size() / 3; i++)
  {
    Eigen::Vector3s a = worldAccs.segment<3>(i * 3);
    if (useL1)
    {
      sum += weightBodies(i) * a.norm();
    }
    else
    {
      sum += weightBodies(i) * a.squaredNorm();
    }
  }

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);

  return sum;
}

//==============================================================================
// Computes the gradient of the norm of the spatial acceleration vector for
// each body
Eigen::VectorXs SpatialNewtonHelper::calculateAccelerationNormGradient(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs weightBodies,
    neural::WithRespectTo* wrt,
    bool useL1)
{
  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  Eigen::VectorXs worldAccs = mSkel->getCOMWorldLinearAccelerations();
  for (int i = 0; i < worldAccs.size() / 3; i++)
  {
    if (useL1)
    {
      worldAccs.segment<3>(i * 3)
          = weightBodies(i) * worldAccs.segment<3>(i * 3).normalized();
    }
    else
    {
      worldAccs.segment<3>(i * 3) *= 2 * weightBodies(i);
    }
  }
  Eigen::MatrixXs jac = mSkel->getCOMWorldLinearAccelerationsJacobian(wrt);

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);

  return jac.transpose() * worldAccs;
}

//==============================================================================
// Computes the gradient of the norm of the spatial acceleration vector for
// each body
Eigen::VectorXs SpatialNewtonHelper::finiteDifferenceAccelerationNormGradient(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs weightBodies,
    neural::WithRespectTo* wrt,
    bool useL1)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(wrt->dim(mSkel.get()));

  Eigen::VectorXs originalPos = mSkel->getPositions();
  Eigen::VectorXs originalVel = mSkel->getVelocities();
  Eigen::VectorXs originalAcc = mSkel->getAccelerations();

  mSkel->setPositions(q);
  mSkel->setVelocities(dq);
  mSkel->setAccelerations(ddq);

  Eigen::VectorXs originalWrt = wrt->get(mSkel.get());

  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::VectorXs newWrt = originalWrt;
        newWrt(dof) += eps;
        wrt->set(mSkel.get(), newWrt);
        perturbed = calculateAccelerationNorm(
            mSkel->getPositions(),
            mSkel->getVelocities(),
            mSkel->getAccelerations(),
            weightBodies,
            useL1);
        return true;
      },
      result,
      5e-5,
      true);

  wrt->set(mSkel.get(), originalWrt);

  mSkel->setPositions(originalPos);
  mSkel->setVelocities(originalVel);
  mSkel->setAccelerations(originalAcc);
  return result;
}

//==============================================================================
DynamicsFitProblem::DynamicsFitProblem(
    std::shared_ptr<DynamicsInitialization> init,
    std::shared_ptr<dynamics::Skeleton> skeleton,
    std::vector<std::string> trackingMarkers,
    std::vector<dynamics::BodyNode*> footNodes,
    DynamicsFitProblemConfig config)
  : mInit(init),
    mSkeleton(skeleton),
    mFootNodes(footNodes),
    mConfig(config),
    mBestObjectiveValue(std::numeric_limits<s_t>::infinity())
{
  // 1. Set up the markers

  for (auto& pair : init->updatedMarkerMap)
  {
    mMarkerNames.push_back(pair.first);
    mMarkers.push_back(pair.second);
    if (std::find(trackingMarkers.begin(), trackingMarkers.end(), pair.first)
        != trackingMarkers.end())
    {
      mMarkerIsTracking.push_back(true);
    }
    else
    {
      mMarkerIsTracking.push_back(false);
    }
  }

  // 2. Set up the q, dq, ddq, and GRF

  int dofs = skeleton->getNumDofs();
  for (int i = 0; i < init->poseTrials.size(); i++)
  {
    s_t dt = init->trialTimesteps[i];
    Eigen::MatrixXs& inputPoses = init->poseTrials[i];
    std::cout << "Trial " << i << ": " << inputPoses.cols() << std::endl;
    Eigen::MatrixXs poses = Eigen::MatrixXs::Zero(dofs, inputPoses.cols());
    Eigen::MatrixXs vels = Eigen::MatrixXs::Zero(dofs, inputPoses.cols());
    Eigen::MatrixXs accs = Eigen::MatrixXs::Zero(dofs, inputPoses.cols());
    for (int j = 0; j < inputPoses.cols(); j++)
    {
      poses.col(j) = inputPoses.col(j);
    }
    for (int j = 1; j < inputPoses.cols(); j++)
    {
      vels.col(j) = (inputPoses.col(j) - inputPoses.col(j - 1)) / dt;
    }
    for (int j = 1; j < inputPoses.cols() - 1; j++)
    {
      accs.col(j) = (inputPoses.col(j + 1) - 2 * inputPoses.col(j)
                     + inputPoses.col(j - 1))
                    / (dt * dt);
    }
    mPoses.push_back(poses);
    mVels.push_back(vels);
    mAccs.push_back(accs);
  }

  for (auto* node : footNodes)
  {
    mForceBodyIndices.push_back(node->getIndexInSkeleton());
  }

  mResidualHelper
      = std::make_shared<ResidualForceHelper>(mSkeleton, mForceBodyIndices);
  mSpatialNewtonHelper = std::make_shared<SpatialNewtonHelper>(mSkeleton);
}

//==============================================================================
// This returns the dimension of the decision variables (the length of the
// flatten() vector), which depends on which variables we choose to include in
// the optimization problem.
int DynamicsFitProblem::getProblemSize()
{
  int size = 0;
  if (mConfig.mIncludeMasses)
  {
    size += mSkeleton->getNumScaleGroups();
  }
  if (mConfig.mIncludeCOMs)
  {
    size += mSkeleton->getNumScaleGroups() * 3;
  }
  if (mConfig.mIncludeInertias)
  {
    size += mSkeleton->getNumScaleGroups() * 6;
  }
  if (mConfig.mIncludeBodyScales)
  {
    size += mSkeleton->getGroupScaleDim();
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    size += mMarkers.size() * 3;
  }
  if (mConfig.mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    if (mConfig.mVelAccImplicit)
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        size += mPoses[trial].cols() * dofs;
      }
    }
    else
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        // Add first q
        size += dofs;
        // Add pos + vel + acc
        size += (mPoses[trial].cols() - 2) * dofs * 3;
        // Add last q, dq
        size += dofs * 2;
      }
    }
  }
  return size;
}

//==============================================================================
// This writes the problem state into a flat vector
Eigen::VectorXs DynamicsFitProblem::flatten()
{
  Eigen::VectorXs flat = Eigen::VectorXs::Zero(getProblemSize());
  int cursor = 0;
  if (mConfig.mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    flat.segment(cursor, dim) = mSkeleton->getGroupMasses();
    cursor += dim;
  }
  if (mConfig.mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    flat.segment(cursor, dim) = mSkeleton->getGroupCOMs();
    cursor += dim;
  }
  if (mConfig.mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    flat.segment(cursor, dim) = mSkeleton->getGroupInertias();
    cursor += dim;
  }
  if (mConfig.mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    flat.segment(cursor, dim) = mSkeleton->getGroupScales();
    cursor += dim;
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      flat.segment(cursor, 3) = mMarkers[i].second;
      cursor += 3;
    }
  }
  if (mConfig.mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    if (mConfig.mVelAccImplicit)
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        for (int t = 0; t < mPoses[trial].cols(); t++)
        {
          flat.segment(cursor, dofs) = mPoses[trial].col(t);
          cursor += dofs;
        }
      }
    }
    else
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        // First q
        flat.segment(cursor, dofs) = mPoses[trial].col(0);
        cursor += dofs;
        // All the middle q, dq, ddq's
        for (int t = 1; t < mPoses[trial].cols() - 1; t++)
        {
          flat.segment(cursor, dofs) = mPoses[trial].col(t);
          cursor += dofs;
          flat.segment(cursor, dofs) = mVels[trial].col(t);
          cursor += dofs;
          flat.segment(cursor, dofs) = mAccs[trial].col(t);
          cursor += dofs;
        }
        // Get the last q and dq
        int lastT = mPoses[trial].cols() - 1;
        flat.segment(cursor, dofs) = mPoses[trial].col(lastT);
        cursor += dofs;
        flat.segment(cursor, dofs) = mVels[trial].col(lastT);
        cursor += dofs;
      }
    }
  }

  assert(cursor == flat.size());

  return flat;
}

//==============================================================================
// This writes the upper bounds into a flat vector
Eigen::VectorXs DynamicsFitProblem::flattenUpperBound()
{
  Eigen::VectorXs flat = Eigen::VectorXs::Zero(getProblemSize());
  int cursor = 0;
  if (mConfig.mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    flat.segment(cursor, dim) = mSkeleton->getGroupMassesUpperBound();
    cursor += dim;
  }
  if (mConfig.mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    flat.segment(cursor, dim) = mSkeleton->getGroupCOMUpperBound();
    cursor += dim;
  }
  if (mConfig.mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    flat.segment(cursor, dim) = mSkeleton->getGroupInertiasUpperBound();
    cursor += dim;
  }
  if (mConfig.mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    flat.segment(cursor, dim) = mSkeleton->getGroupScalesUpperBound();
    cursor += dim;
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      flat.segment(cursor, 3) = Eigen::Vector3s::Ones() * 5;
      cursor += 3;
    }
  }
  if (mConfig.mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    if (mConfig.mVelAccImplicit)
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        for (int t = 0; t < mPoses[trial].cols(); t++)
        {
          flat.segment(cursor, dofs) = mSkeleton->getPositionUpperLimits();
          cursor += dofs;
        }
      }
    }
    else
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        // First q
        flat.segment(cursor, dofs) = mSkeleton->getPositionUpperLimits();
        cursor += dofs;
        // All the middle q, dq, ddq's
        for (int t = 1; t < mPoses[trial].cols() - 1; t++)
        {
          flat.segment(cursor, dofs) = mSkeleton->getPositionUpperLimits();
          cursor += dofs;
          flat.segment(cursor, dofs) = mSkeleton->getVelocityUpperLimits();
          cursor += dofs;
          flat.segment(cursor, dofs) = mSkeleton->getAccelerationUpperLimits();
          cursor += dofs;
        }
        // Get the last q and dq
        flat.segment(cursor, dofs) = mSkeleton->getPositionUpperLimits();
        cursor += dofs;
        flat.segment(cursor, dofs) = mSkeleton->getVelocityUpperLimits();
        cursor += dofs;
      }
    }
  }

  assert(cursor == flat.size());

  return flat;
}

//==============================================================================
// This writes the upper bounds into a flat vector
Eigen::VectorXs DynamicsFitProblem::flattenLowerBound()
{
  Eigen::VectorXs flat = Eigen::VectorXs::Zero(getProblemSize());
  int cursor = 0;
  if (mConfig.mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    flat.segment(cursor, dim) = mSkeleton->getGroupMassesLowerBound();
    cursor += dim;
  }
  if (mConfig.mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    flat.segment(cursor, dim) = mSkeleton->getGroupCOMLowerBound();
    cursor += dim;
  }
  if (mConfig.mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    flat.segment(cursor, dim) = mSkeleton->getGroupInertiasLowerBound();
    cursor += dim;
  }
  if (mConfig.mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    flat.segment(cursor, dim) = mSkeleton->getGroupScalesLowerBound();
    cursor += dim;
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      flat.segment(cursor, 3) = Eigen::Vector3s::Ones() * -5;
      cursor += 3;
    }
  }
  if (mConfig.mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    if (mConfig.mVelAccImplicit)
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        for (int t = 0; t < mPoses[trial].cols(); t++)
        {
          flat.segment(cursor, dofs) = mSkeleton->getPositionLowerLimits();
          cursor += dofs;
        }
      }
    }
    else
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        // First q
        flat.segment(cursor, dofs) = mSkeleton->getPositionLowerLimits();
        cursor += dofs;
        // All the middle q, dq, ddq's
        for (int t = 1; t < mPoses[trial].cols() - 1; t++)
        {
          flat.segment(cursor, dofs) = mSkeleton->getPositionLowerLimits();
          cursor += dofs;
          flat.segment(cursor, dofs) = mSkeleton->getVelocityLowerLimits();
          cursor += dofs;
          flat.segment(cursor, dofs) = mSkeleton->getAccelerationLowerLimits();
          cursor += dofs;
        }
        // Get the last q and dq
        flat.segment(cursor, dofs) = mSkeleton->getPositionLowerLimits();
        cursor += dofs;
        flat.segment(cursor, dofs) = mSkeleton->getVelocityLowerLimits();
        cursor += dofs;
      }
    }
  }

  assert(cursor == flat.size());

  return flat;
}

//==============================================================================
// This reads the problem state out of a flat vector, and into the init object
void DynamicsFitProblem::unflatten(Eigen::VectorXs x)
{
  if (x.size() == mLastX.size() && x == mLastX)
  {
    // No need to overwrite the same update twice
    return;
  }
  mLastX = x;

  int cursor = 0;
  if (mConfig.mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    mSkeleton->setGroupMasses(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mConfig.mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    mSkeleton->setGroupCOMs(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mConfig.mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    mSkeleton->setGroupInertias(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mConfig.mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    mSkeleton->setGroupScales(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      mMarkers[i].second = x.segment(cursor, 3);
      cursor += 3;
    }
  }
  if (mConfig.mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();
    if (mConfig.mVelAccImplicit)
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        s_t dt = mInit->trialTimesteps[trial];

        for (int t = 0; t < mPoses[trial].cols(); t++)
        {
          mPoses[trial].col(t) = x.segment(cursor, dofs);
          cursor += dofs;
        }
        for (int t = 1; t < mPoses[trial].cols(); t++)
        {
          mVels[trial].col(t)
              = (mPoses[trial].col(t) - mPoses[trial].col(t - 1)) / dt;
        }
        for (int t = 1; t < mPoses[trial].cols() - 1; t++)
        {
          mAccs[trial].col(t)
              = (mPoses[trial].col(t + 1) - 2 * mPoses[trial].col(t)
                 + mPoses[trial].col(t - 1))
                / (dt * dt);
        }
      }
    }
    else
    {
      for (int trial = 0; trial < mPoses.size(); trial++)
      {
        // First q
        mPoses[trial].col(0) = x.segment(cursor, dofs);
        cursor += dofs;
        // All the middle q, dq, ddq's
        for (int t = 1; t < mPoses[trial].cols() - 1; t++)
        {
          mPoses[trial].col(t) = x.segment(cursor, dofs);
          cursor += dofs;
          mVels[trial].col(t) = x.segment(cursor, dofs);
          cursor += dofs;
          mAccs[trial].col(t) = x.segment(cursor, dofs);
          cursor += dofs;
        }
        // Get the last q and dq
        int lastT = mPoses[trial].cols() - 1;
        mPoses[trial].col(lastT) = x.segment(cursor, dofs);
        cursor += dofs;
        mVels[trial].col(lastT) = x.segment(cursor, dofs);
        cursor += dofs;
      }
    }
  }

  assert(cursor == x.size());
}

//==============================================================================
// This gets the value of the loss function, as a weighted sum of the
// discrepancy between measured and expected GRF data and other regularization
// terms.
s_t DynamicsFitProblem::computeLoss(Eigen::VectorXs x, bool logExplanation)
{
  unflatten(x);

  s_t sum = 0.0;

  /*
  if (logExplanation)
  {
    Eigen::MatrixXs compare
        = Eigen::MatrixXs::Zero(mSkeleton->getNumScaleGroups(), 5);
    compare.col(0) = mSkeleton->getGroupMasses();
    compare.col(1) = mInit->originalGroupMasses;
    compare.col(2) = mSkeleton->getGroupMassesUpperBound();
    compare.col(3) = mSkeleton->getGroupMassesLowerBound();
    compare.col(4) = (mSkeleton->getGroupMasses() - mInit->originalGroupMasses);
    std::cout << "masses - orig - upper - lower - diff" << std::endl
              << compare << std::endl;
  }
  */

  if (mInit->probablyMissingGRF.size() < mInit->poseTrials.size())
  {
    std::cout << "Don't ask for loss before you've called "
                 "DynamicsFitter::estimateFootGroundContacts() with this init "
                 "object! Killing the process with exit 1."
              << std::endl;
    exit(1);
  }

  s_t massRegularization
      = mConfig.mRegularizeMasses * (1.0 / mSkeleton->getNumScaleGroups())
        * (mSkeleton->getGroupMasses() - mInit->originalGroupMasses)
              .squaredNorm();
  sum += massRegularization;
  assert(!isnan(sum));
  s_t comRegularization
      = mConfig.mRegularizeCOMs * (1.0 / mSkeleton->getNumScaleGroups())
        * (mSkeleton->getGroupCOMs() - mInit->originalGroupCOMs).squaredNorm();
  sum += comRegularization;
  assert(!isnan(sum));
  s_t inertiaRegularization
      = mConfig.mRegularizeInertias * (1.0 / mSkeleton->getNumScaleGroups())
        * (mSkeleton->getGroupInertias() - mInit->originalGroupInertias)
              .squaredNorm();
  sum += inertiaRegularization;
  assert(!isnan(sum));
  s_t scaleRegularization
      = mConfig.mRegularizeBodyScales * (1.0 / mSkeleton->getNumScaleGroups())
        * (mSkeleton->getGroupScales() - mInit->originalGroupScales)
              .squaredNorm();
  sum += scaleRegularization;
  assert(!isnan(sum));
  s_t markerRegularization = 0.0;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (mInit->originalMarkerOffsets.count(mMarkerNames.at(i)))
    {
      markerRegularization
          += (mMarkerIsTracking[i] ? mConfig.mRegularizeTrackingMarkerOffsets
                                   : mConfig.mRegularizeAnatomicalMarkerOffsets)
             * (1.0 / mMarkerNames.size())
             * (mMarkers[i].second
                - mInit->originalMarkerOffsets.at(mMarkerNames.at(i)))
                   .squaredNorm();
    }
    assert(!isnan(markerRegularization));
  }
  sum += markerRegularization;

  s_t densityRegularization = 0.0;
  Eigen::VectorXs masses = mSkeleton->getGroupMasses();
  Eigen::VectorXs inertias = mSkeleton->getGroupInertias();
  for (int i = 0; i < mSkeleton->getNumScaleGroups(); i++)
  {
    s_t mass = masses(i);
    Eigen::Vector3s dims = inertias.segment<3>(i * 6);
    s_t volume = dims(0) * dims(1) * dims(2);
    s_t density = mass / volume;
    s_t error = HUMAN_DENSITY_KG_M3 - density;
    densityRegularization += mConfig.mRegularizeImpliedDensity * error * error;
  }
  sum += densityRegularization;

  Eigen::VectorXs originalPos = mSkeleton->getPositions();
  mSkeleton->clearExternalForces();

  int totalTimesteps = 0;
  int totalAccTimesteps = 0;
  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    totalTimesteps += mPoses[trial].cols();
    for (int t = 1; t < mPoses[trial].cols() - 1; t++)
    {
      // Add force residual RMS errors to all the middle timesteps
      if (!mInit->probablyMissingGRF[trial][t])
      {
        totalAccTimesteps++;
      }
    }
  }

  s_t linearNewtonError = 0.0;
  s_t residualRMS = 0.0;
  s_t markerRMS = 0.0;
  s_t poseRegularization = 0.0;
  s_t accRegularization = 0.0;
  s_t jointRMS = 0.0;
  s_t axisRMS = 0.0;
  int markerCount = 0;

  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    for (int t = 0; t < mPoses[trial].cols(); t++)
    {
      mSkeleton->setPositions(mPoses[trial].col(t));

      // Add force residual RMS errors to all the middle timesteps
      if (t > 0 && t < mPoses[trial].cols() - 1
          && !mInit->probablyMissingGRF[trial][t])
      {
        if (mConfig.mLinearNewtonWeight > 0)
        {
          s_t cost = mConfig.mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateLinearForceGapNorm(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mInit->grfTrials[trial].col(t),
                         mConfig.mLinearNewtonUseL1);
          linearNewtonError += cost;
          assert(!isnan(linearNewtonError));
        }
        if (mConfig.mResidualWeight > 0)
        {
          s_t cost = mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                     * mResidualHelper->calculateResidualNorm(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mInit->grfTrials[trial].col(t),
                         mConfig.mResidualTorqueMultiple,
                         mConfig.mResidualUseL1);
          residualRMS += cost;
          assert(!isnan(residualRMS));
        }
        if (mConfig.mRegularizeAcc > 0)
        {
          s_t cost = mConfig.mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNorm(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mConfig.mRegularizeAccBodyWeights,
                         mConfig.mRegularizeAccUseL1);
          accRegularization += cost;
          assert(!isnan(accRegularization));
        }
      }

      // Add marker RMS errors to every timestep
      auto markerPoses = mSkeleton->getMarkerWorldPositions(mMarkers);
      auto observedMarkerPoses = mInit->markerObservationTrials[trial][t];
      for (int i = 0; i < mMarkerNames.size(); i++)
      {
        Eigen::Vector3s marker = markerPoses.segment<3>(i * 3);
        if (observedMarkerPoses.count(mMarkerNames[i]))
        {
          Eigen::Vector3s diff
              = observedMarkerPoses.at(mMarkerNames[i]) - marker;
          s_t thisMarkerCost;
          if (mConfig.mMarkerUseL1)
          {
            thisMarkerCost = diff.norm();
          }
          else
          {
            thisMarkerCost = diff.squaredNorm();
          }
          markerRMS += thisMarkerCost;
          markerCount++;
          assert(!isnan(markerRMS));
        }
      }

      // Add joints
      Eigen::VectorXs jointPoses
          = mSkeleton->getJointWorldPositions(mInit->joints);
      Eigen::VectorXs jointCenters = mInit->jointCenters[trial].col(t);
      Eigen::VectorXs jointAxis = mInit->jointAxis[trial].col(t);
      Eigen::VectorXs jointDiff = jointPoses - jointCenters;
      for (int i = 0; i < mInit->jointWeights.size(); i++)
      {
        jointRMS
            += (jointPoses.segment<3>(i * 3) - jointCenters.segment<3>(i * 3))
                   .squaredNorm()
               * mInit->jointWeights(i);
      }
      for (int i = 0; i < mInit->axisWeights.size(); i++)
      {
        Eigen::Vector3s axisCenter = jointAxis.segment<3>(i * 6);
        Eigen::Vector3s axisDir = jointAxis.segment<3>(i * 6 + 3).normalized();
        Eigen::Vector3s actualJointPos = jointPoses.segment<3>(i * 3);
        // Subtract out any component parallel to the axis
        Eigen::Vector3s jointDiff = actualJointPos - axisCenter;
        jointDiff -= jointDiff.dot(axisDir) * axisDir;
        axisRMS += jointDiff.squaredNorm() * mInit->axisWeights(i);
      }

      // Add regularization
      poseRegularization
          += mConfig.mRegularizePoses * (1.0 / totalTimesteps)
             * (mPoses[trial].col(t) - mInit->originalPoses[trial].col(t))
                   .squaredNorm();
      assert(!isnan(poseRegularization));
    }
  }
  sum += linearNewtonError;
  sum += residualRMS;
  sum += accRegularization;
  markerRMS *= mConfig.mMarkerWeight;
  if (markerCount > 0)
  {
    markerRMS /= markerCount;
  }
  sum += markerRMS;
  jointRMS *= mConfig.mJointWeight;
  sum += jointRMS;
  axisRMS *= mConfig.mJointWeight;
  sum += axisRMS;
  sum += poseRegularization;
  assert(!isnan(sum));

  if (logExplanation)
  {
    std::cout << "["
              << "massR=" << massRegularization << ",comR=" << comRegularization
              << ",inR=" << inertiaRegularization
              << ",dnsR=" << densityRegularization
              << ",scR=" << scaleRegularization
              << ",mkrR=" << markerRegularization
              << ",accR=" << accRegularization << ",jntRMS=" << jointRMS
              << ",axisRMS=" << axisRMS << ",qR=" << poseRegularization
              << ",fRMS=" << residualRMS << ",linF=" << linearNewtonError
              << ",mkRMS=" << markerRMS << "]" << std::endl;
  }

  return sum;
}

//==============================================================================
// This gets the gradient of the loss function
Eigen::VectorXs DynamicsFitProblem::computeGradient(Eigen::VectorXs x)
{
  unflatten(x);

  Eigen::VectorXs grad = Eigen::VectorXs::Zero(getProblemSize());
  const int dofs = mSkeleton->getNumDofs();

  /*
  //////////////////////
  // From flatten():
  //////////////////////

  int cursor = 0;
  if (mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    mSkeleton->setGroupMasses(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    mSkeleton->setGroupCOMs(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    mSkeleton->setGroupInertias(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    mSkeleton->setGroupScales(x.segment(cursor, dim));
    cursor += dim;
  }
  if (mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      mMarkers[i].second = x.segment(cursor, 3);
      cursor += 3;
    }
  }
  if (mIncludePoses)
  {
  */
  if (mInit->probablyMissingGRF.size() < mInit->poseTrials.size())
  {
    std::cout << "Don't ask for gradients before you've called "
                 "DynamicsFitter::estimateFootGroundContacts() with this init "
                 "object! Killing the process with exit 1."
              << std::endl;
    exit(1);
  }

  int posesCursor = 0;
  if (mConfig.mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    grad.segment(posesCursor, dim)
        += mConfig.mRegularizeMasses * 2
           * (1.0 / mSkeleton->getNumScaleGroups())
           * (mSkeleton->getGroupMasses() - mInit->originalGroupMasses);

    Eigen::VectorXs masses = mSkeleton->getGroupMasses();
    Eigen::VectorXs inertias = mSkeleton->getGroupInertias();
    for (int i = 0; i < mSkeleton->getNumScaleGroups(); i++)
    {
      s_t mass = masses(i);
      Eigen::Vector3s dims = inertias.segment<3>(i * 6);
      s_t volume = dims(0) * dims(1) * dims(2);
      grad(posesCursor + i) += mConfig.mRegularizeImpliedDensity
                               * (2 * mass - 2 * volume * HUMAN_DENSITY_KG_M3)
                               / (volume * volume);
    }

    posesCursor += dim;
  }
  if (mConfig.mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    grad.segment(posesCursor, dim)
        += mConfig.mRegularizeCOMs * 2 * (1.0 / mSkeleton->getNumScaleGroups())
           * (mSkeleton->getGroupCOMs() - mInit->originalGroupCOMs);
    posesCursor += dim;
  }
  if (mConfig.mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    grad.segment(posesCursor, dim)
        += mConfig.mRegularizeInertias * 2
           * (1.0 / mSkeleton->getNumScaleGroups())
           * (mSkeleton->getGroupInertias() - mInit->originalGroupInertias);

    Eigen::VectorXs masses = mSkeleton->getGroupMasses();
    Eigen::VectorXs inertias = mSkeleton->getGroupInertias();
    for (int i = 0; i < mSkeleton->getNumScaleGroups(); i++)
    {
      s_t mass = masses(i);
      Eigen::Vector3s dims = inertias.segment<3>(i * 6);
      s_t volume = dims(0) * dims(1) * dims(2);
      s_t constant
          = mConfig.mRegularizeImpliedDensity
            * (2 * mass * HUMAN_DENSITY_KG_M3 * volume - 2 * mass * mass)
            / (volume * volume);
      grad(posesCursor + i * 6 + 0) += constant / dims(0);
      grad(posesCursor + i * 6 + 1) += constant / dims(1);
      grad(posesCursor + i * 6 + 2) += constant / dims(2);
    }

    posesCursor += dim;
  }
  if (mConfig.mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    grad.segment(posesCursor, dim)
        += mConfig.mRegularizeBodyScales * 2
           * (1.0 / mSkeleton->getNumScaleGroups())
           * (mSkeleton->getGroupScales() - mInit->originalGroupScales);
    posesCursor += dim;
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      grad.segment<3>(posesCursor)
          += 2
             * (mMarkerIsTracking[i]
                    ? mConfig.mRegularizeTrackingMarkerOffsets
                    : mConfig.mRegularizeAnatomicalMarkerOffsets)
             * (1.0 / mMarkerNames.size())
             * (mMarkers[i].second
                - mInit->originalMarkerOffsets[mMarkerNames[i]]);
      posesCursor += 3;
    }
  }

  int totalTimesteps = 0;
  int totalAccTimesteps = 0;
  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    totalTimesteps += mPoses[trial].cols();
    for (int t = 1; t < mPoses[trial].cols() - 1; t++)
    {
      // Add force residual RMS errors to all the middle timesteps
      if (!mInit->probablyMissingGRF[trial][t])
      {
        totalAccTimesteps++;
      }
    }
  }
  int markerCount = 0;
  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    for (int t = 0; t < mPoses[trial].cols(); t++)
    {
      auto& markerObservations = mInit->markerObservationTrials[trial][t];
      for (int i = 0; i < mMarkers.size(); i++)
      {
        if (markerObservations.count(mMarkerNames[i]))
        {
          markerCount++;
        }
      }
    }
  }

  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    s_t dt = mInit->trialTimesteps[trial];
    for (int t = 0; t < mPoses[trial].cols(); t++)
    {
      mSkeleton->setPositions(mPoses[trial].col(t));
      Eigen::VectorXs lossGradWrtMarkerError
          = Eigen::VectorXs::Zero(mMarkers.size() * 3);
      auto& markerObservations = mInit->markerObservationTrials[trial][t];
      auto markerPoses = mSkeleton->getMarkerWorldPositions(mMarkers);
      for (int i = 0; i < mMarkers.size(); i++)
      {
        if (markerObservations.count(mMarkerNames[i]))
        {
          Eigen::Vector3s markerOffset
              = markerPoses.segment<3>(i * 3)
                - markerObservations.at(mMarkerNames[i]);
          if (mConfig.mMarkerUseL1)
          {
            markerOffset.normalize();
          }
          else
          {
            markerOffset *= 2;
          }
          lossGradWrtMarkerError.segment<3>(i * 3)
              = (mConfig.mMarkerWeight / markerCount) * markerOffset;
        }
      }

      Eigen::VectorXs jointGrad
          = Eigen::VectorXs::Zero(mInit->joints.size() * 3);
      Eigen::VectorXs worldJoints
          = mSkeleton->getJointWorldPositions(mInit->joints);
      Eigen::VectorXs targetJoints = mInit->jointCenters[trial].col(t);
      Eigen::VectorXs targetAxis = mInit->jointAxis[trial].col(t);
      for (int i = 0; i < mInit->joints.size(); i++)
      {
        Eigen::Vector3s worldDiff
            = worldJoints.segment<3>(i * 3) - targetJoints.segment<3>(i * 3);
        jointGrad.segment<3>(i * 3) += 2 * worldDiff * mInit->jointWeights(i);

        Eigen::Vector3s axisDiff
            = worldJoints.segment<3>(i * 3) - targetAxis.segment<3>(i * 6);
        Eigen::Vector3s axis = targetAxis.segment<3>(i * 6 + 3).normalized();
        axisDiff -= axisDiff.dot(axis) * axis;
        jointGrad.segment<3>(i * 3) += 2 * axisDiff * mInit->axisWeights(i);
      }
      jointGrad *= mConfig.mJointWeight;

      // We only compute the residual on middle t's, since we can't finite
      // difference acceleration at the edges of the clip
      if (t > 0 && t < mPoses[trial].cols() - 1)
      {
        int cursor = 0;
        if (mConfig.mIncludeMasses)
        {
          int dim = mSkeleton->getNumScaleGroups();
          if (!mInit->probablyMissingGRF[trial][t])
          {
            if (mConfig.mResidualWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                     * mResidualHelper->calculateResidualNormGradientWrt(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mInit->grfTrials[trial].col(t),
                         neural::WithRespectTo::GROUP_MASSES,
                         mConfig.mResidualTorqueMultiple,
                         mConfig.mResidualUseL1);
            }
            if (mConfig.mLinearNewtonWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper
                           ->calculateLinearForceGapNormGradientWrt(
                               mPoses[trial].col(t),
                               mVels[trial].col(t),
                               mAccs[trial].col(t),
                               mInit->grfTrials[trial].col(t),
                               neural::WithRespectTo::GROUP_MASSES,
                               mConfig.mLinearNewtonUseL1);
            }
            /*
            // This should always be 0, and therefore not necessary
            if (mRegularizeAcc > 0)
            {
              grad.segment(cursor, dim)
                  += mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNormGradient(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mRegularizeAccBodyWeights,
                         neural::WithRespectTo::GROUP_MASSES,
                         mRegularizeAccUseL1);
            }
            */
          }
          cursor += dim;
        }
        if (mConfig.mIncludeCOMs)
        {
          int dim = mSkeleton->getNumScaleGroups() * 3;
          if (!mInit->probablyMissingGRF[trial][t])
          {
            if (mConfig.mResidualWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                     * mResidualHelper->calculateResidualNormGradientWrt(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mInit->grfTrials[trial].col(t),
                         neural::WithRespectTo::GROUP_COMS,
                         mConfig.mResidualTorqueMultiple,
                         mConfig.mResidualUseL1);
            }
            if (mConfig.mLinearNewtonWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper
                           ->calculateLinearForceGapNormGradientWrt(
                               mPoses[trial].col(t),
                               mVels[trial].col(t),
                               mAccs[trial].col(t),
                               mInit->grfTrials[trial].col(t),
                               neural::WithRespectTo::GROUP_COMS,
                               mConfig.mLinearNewtonUseL1);
            }
            if (mConfig.mRegularizeAcc > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNormGradient(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mConfig.mRegularizeAccBodyWeights,
                         neural::WithRespectTo::GROUP_COMS,
                         mConfig.mRegularizeAccUseL1);
            }
          }
          cursor += dim;
        }
        if (mConfig.mIncludeInertias)
        {
          int dim = mSkeleton->getNumScaleGroups() * 6;
          if (!mInit->probablyMissingGRF[trial][t])
          {
            if (mConfig.mResidualWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                     * mResidualHelper->calculateResidualNormGradientWrt(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mInit->grfTrials[trial].col(t),
                         neural::WithRespectTo::GROUP_INERTIAS,
                         mConfig.mResidualTorqueMultiple,
                         mConfig.mResidualUseL1);
            }
            /*
            // This should always be 0, and therefore not necessary

            if (mLinearNewtonWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper
                           ->calculateLinearForceGapNormGradientWrt(
                               mPoses[trial].col(t),
                               mVels[trial].col(t),
                               mAccs[trial].col(t),
                               mInit->grfTrials[trial].col(t),
                               neural::WithRespectTo::GROUP_INERTIAS,
                               mLinearNewtonUseL1);
            }
            if (mRegularizeAcc > 0)
            {
              grad.segment(cursor, dim)
                  += mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNormGradient(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mRegularizeAccBodyWeights,
                         neural::WithRespectTo::GROUP_INERTIAS,
                         mRegularizeAccUseL1);
            }
            */
          }
          cursor += dim;
        }
        if (mConfig.mIncludeBodyScales)
        {
          int dim = mSkeleton->getGroupScaleDim();
          if (!mInit->probablyMissingGRF[trial][t])
          {
            if (mConfig.mResidualWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                     * mResidualHelper->calculateResidualNormGradientWrt(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mInit->grfTrials[trial].col(t),
                         neural::WithRespectTo::GROUP_SCALES,
                         mConfig.mResidualTorqueMultiple,
                         mConfig.mResidualUseL1);
            }
            if (mConfig.mLinearNewtonWeight > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper
                           ->calculateLinearForceGapNormGradientWrt(
                               mPoses[trial].col(t),
                               mVels[trial].col(t),
                               mAccs[trial].col(t),
                               mInit->grfTrials[trial].col(t),
                               neural::WithRespectTo::GROUP_SCALES,
                               mConfig.mLinearNewtonUseL1);
            }
            if (mConfig.mRegularizeAcc > 0)
            {
              grad.segment(cursor, dim)
                  += mConfig.mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNormGradient(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mConfig.mRegularizeAccBodyWeights,
                         neural::WithRespectTo::GROUP_SCALES,
                         mConfig.mRegularizeAccUseL1);
            }
          }

          // Record marker gradients
          grad.segment(cursor, dim)
              += MarkerFitter::getMarkerLossGradientWrtGroupScales(
                  mSkeleton, mMarkers, lossGradWrtMarkerError);

          // Record joint gradients
          grad.segment(cursor, dim)
              += mSkeleton
                     ->getJointWorldPositionsJacobianWrtGroupScales(
                         mInit->joints)
                     .transpose()
                 * jointGrad;

          cursor += dim;
        }
        if (mConfig.mIncludeMarkerOffsets)
        {
          int dim = mMarkers.size() * 3;
          grad.segment(cursor, dim)
              += MarkerFitter::getMarkerLossGradientWrtMarkerOffsets(
                  mSkeleton, mMarkers, lossGradWrtMarkerError);
          cursor += dim;
        }

        if (mConfig.mIncludePoses)
        {
          Eigen::VectorXs posGrad = Eigen::VectorXs::Zero(dofs);
          Eigen::VectorXs velGrad = Eigen::VectorXs::Zero(dofs);
          Eigen::VectorXs accGrad = Eigen::VectorXs::Zero(dofs);
          if (!mInit->probablyMissingGRF[trial][t])
          {
            if (mConfig.mResidualWeight > 0)
            {
              posGrad += mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                         * mResidualHelper->calculateResidualNormGradientWrt(
                             mPoses[trial].col(t),
                             mVels[trial].col(t),
                             mAccs[trial].col(t),
                             mInit->grfTrials[trial].col(t),
                             neural::WithRespectTo::POSITION,
                             mConfig.mResidualTorqueMultiple,
                             mConfig.mResidualUseL1);
              velGrad += mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                         * mResidualHelper->calculateResidualNormGradientWrt(
                             mPoses[trial].col(t),
                             mVels[trial].col(t),
                             mAccs[trial].col(t),
                             mInit->grfTrials[trial].col(t),
                             neural::WithRespectTo::VELOCITY,
                             mConfig.mResidualTorqueMultiple,
                             mConfig.mResidualUseL1);
              accGrad += mConfig.mResidualWeight * (1.0 / totalAccTimesteps)
                         * mResidualHelper->calculateResidualNormGradientWrt(
                             mPoses[trial].col(t),
                             mVels[trial].col(t),
                             mAccs[trial].col(t),
                             mInit->grfTrials[trial].col(t),
                             neural::WithRespectTo::ACCELERATION,
                             mConfig.mResidualTorqueMultiple,
                             mConfig.mResidualUseL1);
            }
            if (mConfig.mLinearNewtonWeight > 0)
            {
              posGrad += mConfig.mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                         * mSpatialNewtonHelper
                               ->calculateLinearForceGapNormGradientWrt(
                                   mPoses[trial].col(t),
                                   mVels[trial].col(t),
                                   mAccs[trial].col(t),
                                   mInit->grfTrials[trial].col(t),
                                   neural::WithRespectTo::POSITION,
                                   mConfig.mLinearNewtonUseL1);
              velGrad += mConfig.mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                         * mSpatialNewtonHelper
                               ->calculateLinearForceGapNormGradientWrt(
                                   mPoses[trial].col(t),
                                   mVels[trial].col(t),
                                   mAccs[trial].col(t),
                                   mInit->grfTrials[trial].col(t),
                                   neural::WithRespectTo::VELOCITY,
                                   mConfig.mLinearNewtonUseL1);
              accGrad += mConfig.mLinearNewtonWeight * (1.0 / totalAccTimesteps)
                         * mSpatialNewtonHelper
                               ->calculateLinearForceGapNormGradientWrt(
                                   mPoses[trial].col(t),
                                   mVels[trial].col(t),
                                   mAccs[trial].col(t),
                                   mInit->grfTrials[trial].col(t),
                                   neural::WithRespectTo::ACCELERATION,
                                   mConfig.mLinearNewtonUseL1);
            }
            if (mConfig.mRegularizeAcc > 0)
            {
              posGrad
                  += mConfig.mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNormGradient(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mConfig.mRegularizeAccBodyWeights,
                         neural::WithRespectTo::POSITION,
                         mConfig.mRegularizeAccUseL1);
              velGrad
                  += mConfig.mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNormGradient(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mConfig.mRegularizeAccBodyWeights,
                         neural::WithRespectTo::VELOCITY,
                         mConfig.mRegularizeAccUseL1);
              accGrad
                  += mConfig.mRegularizeAcc * (1.0 / totalAccTimesteps)
                     * mSpatialNewtonHelper->calculateAccelerationNormGradient(
                         mPoses[trial].col(t),
                         mVels[trial].col(t),
                         mAccs[trial].col(t),
                         mConfig.mRegularizeAccBodyWeights,
                         neural::WithRespectTo::ACCELERATION,
                         mConfig.mRegularizeAccUseL1);
            }
          }

          grad.segment(posesCursor, dofs) += posGrad;

          // Record marker gradients
          grad.segment(posesCursor, dofs)
              += MarkerFitter::getMarkerLossGradientWrtJoints(
                  mSkeleton, mMarkers, lossGradWrtMarkerError);

          // Record regularization
          grad.segment(posesCursor, dofs)
              += mConfig.mRegularizePoses * 2 * (1.0 / totalTimesteps)
                 * (mPoses[trial].col(t) - mInit->originalPoses[trial].col(t));

          // Record joint gradients
          grad.segment(posesCursor, dofs)
              += mSkeleton
                     ->getJointWorldPositionsJacobianWrtJointPositions(
                         mInit->joints)
                     .transpose()
                 * jointGrad;

          if (mConfig.mVelAccImplicit)
          {
            // v = (t - t-1) / dt
            // dv/dt = I / dt
            // dv/dt-1 = -I / dt
            grad.segment(posesCursor, dofs) += velGrad / dt;
            assert(t > 0);
            grad.segment(posesCursor - dofs, dofs) -= velGrad / dt;

            // a = ((t+1) - 2*t + (t-1)) / dt*dt
            grad.segment(posesCursor, dofs) -= 2 * accGrad / (dt * dt);
            assert(t < mPoses[trial].cols());
            grad.segment(posesCursor + dofs, dofs) += accGrad / (dt * dt);
            assert(t > 0);
            grad.segment(posesCursor - dofs, dofs) += accGrad / (dt * dt);
          }
          else
          {
            // Record Vel gradients wrt pos
            posesCursor += dofs;
            grad.segment(posesCursor, dofs) += velGrad;

            // Record Acc gradients wrt pos
            posesCursor += dofs;
            grad.segment(posesCursor, dofs) += accGrad;
          }

          posesCursor += dofs;
        }
      }
      else
      {
        int cursor = 0;
        if (mConfig.mIncludeMasses)
        {
          int dim = mSkeleton->getNumScaleGroups();
          cursor += dim;
        }
        if (mConfig.mIncludeCOMs)
        {
          int dim = mSkeleton->getNumScaleGroups() * 3;
          cursor += dim;
        }
        if (mConfig.mIncludeInertias)
        {
          int dim = mSkeleton->getNumScaleGroups() * 6;
          cursor += dim;
        }
        if (mConfig.mIncludeBodyScales)
        {
          int dim = mSkeleton->getGroupScaleDim();
          // Record marker gradients
          grad.segment(cursor, dim)
              += MarkerFitter::getMarkerLossGradientWrtGroupScales(
                  mSkeleton, mMarkers, lossGradWrtMarkerError);
          // Record joint gradients
          grad.segment(cursor, dim)
              += mSkeleton
                     ->getJointWorldPositionsJacobianWrtGroupScales(
                         mInit->joints)
                     .transpose()
                 * jointGrad;

          cursor += dim;
        }
        if (mConfig.mIncludeMarkerOffsets)
        {
          int dim = mMarkers.size() * 3;
          grad.segment(cursor, dim)
              += MarkerFitter::getMarkerLossGradientWrtMarkerOffsets(
                  mSkeleton, mMarkers, lossGradWrtMarkerError);
          cursor += dim;
        }
        if (mConfig.mIncludePoses)
        {
          // Record marker gradients
          grad.segment(posesCursor, dofs)
              += MarkerFitter::getMarkerLossGradientWrtJoints(
                  mSkeleton, mMarkers, lossGradWrtMarkerError);
          // Record regularization
          grad.segment(posesCursor, dofs)
              += mConfig.mRegularizePoses * 2 * (1.0 / totalTimesteps)
                 * (mPoses[trial].col(t) - mInit->originalPoses[trial].col(t));
          // Record joint gradients
          grad.segment(posesCursor, dofs)
              += mSkeleton
                     ->getJointWorldPositionsJacobianWrtJointPositions(
                         mInit->joints)
                     .transpose()
                 * jointGrad;

          posesCursor += dofs;

          if (!mConfig.mVelAccImplicit && t == mPoses[trial].cols() - 1)
          {
            // Skip over the velocities at this last timestep where they're
            // available, they don't do anything
            posesCursor += dofs;
          }
        }
      }
    }
  }

  assert(posesCursor == grad.size());

  return grad;
}

//==============================================================================
// This gets the gradient of the loss function
Eigen::VectorXs DynamicsFitProblem::finiteDifferenceGradient(
    Eigen::VectorXs x, bool useRidders)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getProblemSize());

  math::finiteDifference(
      [this, x](
          /* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::VectorXs perturbedX = x;
        perturbedX(dof) += eps;
        perturbed = computeLoss(perturbedX);
        return true;
      },
      result,
      useRidders ? 1e-3 : 1e-8,
      useRidders);

  return result;
}

// This gets the number of constraints that the problem requires
int DynamicsFitProblem::getConstraintSize()
{
  if (mConfig.mIncludePoses && !mConfig.mVelAccImplicit)
  {
    int numConstraints = 0;
    int dofs = mSkeleton->getNumDofs();
    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      if (mPoses[trial].cols() > 2)
      {
        // Each ddq and dq need a constraint
        numConstraints += (mPoses[trial].cols() - 2) * dofs * 2;
      }
    }
    // Last dq needs a constraint as well
    numConstraints += dofs;
    return numConstraints;
  }

  return 0;
}

// This gets the value of the constraints vector. These constraints are only
// active when we're including positions in the decision variables, and they
// just enforce that finite differencing is valid to relate velocity,
// acceleration, and position.
Eigen::VectorXs DynamicsFitProblem::computeConstraints(Eigen::VectorXs x)
{
  if (mConfig.mIncludePoses && !mConfig.mVelAccImplicit)
  {
    unflatten(x);
    int dim = getConstraintSize();
    Eigen::VectorXs constraints = Eigen::VectorXs::Zero(dim);
    int dofs = mSkeleton->getNumDofs();

    int cursor = 0;
    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      s_t dt = mInit->trialTimesteps[trial];
      for (int t = 1; t < mPoses[trial].cols() - 1; t++)
      {
        for (int i = 0; i < dofs; i++)
        {
          s_t fd = mPoses[trial](i, t) - mPoses[trial](i, t - 1);
          constraints(cursor) = (mVels[trial](i, t) * dt) - fd;
          cursor++;
        }
        for (int i = 0; i < dofs; i++)
        {
          s_t fd = mVels[trial](i, t + 1) - mVels[trial](i, t);
          constraints(cursor) = (mAccs[trial](i, t) * dt) - fd;
          cursor++;
        }
      }
      if (mPoses[trial].cols() > 1)
      {
        // Last dq needs a constraint as well
        int lastT = mPoses[trial].cols() - 1;
        for (int i = 0; i < dofs; i++)
        {
          s_t fd = mPoses[trial](i, lastT) - mPoses[trial](i, lastT - 1);
          constraints(cursor) = (mVels[trial](i, lastT) * dt) - fd;
          cursor++;
        }
      }
    }
    assert(cursor == constraints.size());
    return constraints;
  }
  return Eigen::VectorXs::Zero(0);
}

// This gets the sparse version of the constraints jacobian, returning objects
// with (row,col,value).
std::vector<std::tuple<int, int, s_t>>
DynamicsFitProblem::computeSparseConstraintsJacobian()
{
  int colCursor = 0;
  if (mConfig.mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    colCursor += dim;
  }
  if (mConfig.mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    colCursor += dim;
  }
  if (mConfig.mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    colCursor += dim;
  }
  if (mConfig.mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    colCursor += dim;
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    colCursor += 3 * mMarkers.size();
  }

#ifndef NDEBUG
  int cols = getProblemSize();
#endif

  std::vector<std::tuple<int, int, s_t>> result;
  if (mConfig.mIncludePoses && !mConfig.mVelAccImplicit)
  {
    int dofs = mSkeleton->getNumDofs();

    // Add space for the first set of poses
    colCursor += dofs;

    int rowCursor = 0;
    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      s_t dt = mInit->trialTimesteps[trial];
      for (int t = 1; t < mPoses[trial].cols() - 1; t++)
      {
        for (int i = 0; i < dofs; i++)
        {
          // s_t fd = mPoses[trial](i, t) - mPoses[trial](i, t - 1);
          // constraints(rowCursor) = (mVels[trial](i, t) * dt) - fd;
          int q_i_t_minus_1 = colCursor - (dofs * (t == 1 ? 1 : 3)) + i;
          int q_i_t0 = colCursor + i;
          int dq_i_t0 = colCursor + dofs + i;
          result.emplace_back(rowCursor, q_i_t_minus_1, 1);
          result.emplace_back(rowCursor, q_i_t0, -1);
          result.emplace_back(rowCursor, dq_i_t0, dt);
          rowCursor++;
        }
        for (int i = 0; i < dofs; i++)
        {
          // s_t fd = mVels[trial](i, t + 1) - mVels[trial](i, t);
          // constraints(rowCursor) = (mAccs[trial](i, t) * dt) - fd;
          int dq_i_t0 = colCursor + dofs + i;
          int dq_i_t1 = colCursor + (dofs * 3) + dofs + i;
          int ddq_i_t0 = colCursor + (dofs * 2) + i;
          result.emplace_back(rowCursor, dq_i_t0, 1);
          result.emplace_back(rowCursor, dq_i_t1, -1);
          result.emplace_back(rowCursor, ddq_i_t0, dt);
          rowCursor++;
        }

        // add space for this t's [q, dq, ddq]
        colCursor += dofs * 3;
#ifndef NDEBUG
        assert(colCursor < cols);
#endif
      }
      if (mPoses[trial].cols() > 1)
      {
        // Do the last dq constraint
        for (int i = 0; i < dofs; i++)
        {
          // s_t fd
          //     = mPoses[trial](i, lastT) - mPoses[trial](i, lastT - 1);
          // constraints(cursor) = (mVels[trial](i, lastAccT + 1) * dt) - fd;
          int q_i_t0 = colCursor + i;
          int q_i_t_minus_1
              = colCursor - (dofs * (mPoses[trial].cols() > 2 ? 3 : 1)) + i;
          int dq_i_t0 = colCursor + dofs + i;
          result.emplace_back(rowCursor, q_i_t0, -1);
          result.emplace_back(rowCursor, q_i_t_minus_1, 1);
          result.emplace_back(rowCursor, dq_i_t0, dt);
          rowCursor++;
        }
        // add space for the last [q, dq]
        colCursor += dofs * 2;
      }
      assert(colCursor <= cols);
    }
  }

  assert(colCursor == cols);
  return result;
}

// This gets the jacobian of the constraints vector with respect to x
Eigen::MatrixXs DynamicsFitProblem::computeConstraintsJacobian()
{
  if (!mConfig.mIncludePoses || mConfig.mVelAccImplicit)
  {
    return Eigen::MatrixXs::Zero(0, 0);
  }
  int problemDim = getProblemSize();
  int constraintDim = getConstraintSize();
  auto sparseConstraints = computeSparseConstraintsJacobian();
  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(constraintDim, problemDim);
  for (auto& tuple : sparseConstraints)
  {
    J(std::get<0>(tuple), std::get<1>(tuple)) = std::get<2>(tuple);
  }
  return J;
}

// This gets the jacobian of the constraints vector with respect to x
Eigen::MatrixXs DynamicsFitProblem::finiteDifferenceConstraintsJacobian()
{
  int problemDim = getProblemSize();
  int constraintDim = getConstraintSize();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(constraintDim, problemDim);
  Eigen::VectorXs original = flatten();

  const bool useRidders = false;
  s_t eps = useRidders ? 1e-3 : 1e-6;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::VectorXs tweaked = original;
        tweaked(dof) += eps;
        perturbed = computeConstraints(tweaked);
        return true;
      },
      result,
      eps,
      useRidders);

  return result;
}

//==============================================================================
bool debugVector(
    Eigen::VectorXs fd, Eigen::VectorXs analytical, std::string name, s_t tol)
{
  bool anyError = false;
  for (int i = 0; i < fd.size(); i++)
  {
    bool isError = false;
    s_t error = 0.0;
    if (fabs(fd(i)) > 1)
    {
      // Test relative error for values that are larger than 1
      if (fabs((fd(i) - analytical(i)) / fd(i)) > tol)
      {
        error = fabs((fd(i) - analytical(i)) / fd(i));
        isError = true;
      }
    }
    else if (fabs(fd(i) - analytical(i)) > tol)
    {
      error = fabs(fd(i) - analytical(i));
      isError = true;
    }
    if (isError)
    {
      std::cout << "Error on " << name << "[" << i << "]: " << fd(i) << " - "
                << analytical(i) << " = " << error << std::endl;
      anyError = true;
    }
  }
  return anyError;
}

//==============================================================================
// Print out the errors in a gradient vector in human readable form
bool DynamicsFitProblem::debugErrors(
    Eigen::VectorXs fd, Eigen::VectorXs analytical, s_t tol)
{
  int cursor = 0;
  bool anyError = false;
  if (mConfig.mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    anyError |= debugVector(
        fd.segment(cursor, dim), analytical.segment(cursor, dim), "mass", tol);
    cursor += dim;
  }
  if (mConfig.mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    anyError |= debugVector(
        fd.segment(cursor, dim), analytical.segment(cursor, dim), "COM", tol);
    cursor += dim;
  }
  if (mConfig.mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    anyError |= debugVector(
        fd.segment(cursor, dim),
        analytical.segment(cursor, dim),
        "inertia",
        tol);
    cursor += dim;
  }
  if (mConfig.mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    anyError |= debugVector(
        fd.segment(cursor, dim),
        analytical.segment(cursor, dim),
        "bodyScales",
        tol);
    cursor += dim;
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      anyError |= debugVector(
          fd.segment(cursor, 3),
          analytical.segment(cursor, 3),
          "marker_" + std::to_string(i),
          tol);
      cursor += 3;
    }
  }
  if (mConfig.mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      if (mConfig.mVelAccImplicit)
      {
        for (int t = 0; t < mPoses[trial].cols(); t++)
        {
          anyError |= debugVector(
              fd.segment(cursor, dofs),
              analytical.segment(cursor, dofs),
              "poses@t=" + std::to_string(t),
              tol);
          cursor += dofs;
        }
      }
      else
      {
        anyError |= debugVector(
            fd.segment(cursor, dofs),
            analytical.segment(cursor, dofs),
            "poses@t=0",
            tol);
        cursor += dofs;

        for (int t = 1; t < mPoses[trial].cols() - 1; t++)
        {
          anyError |= debugVector(
              fd.segment(cursor, dofs),
              analytical.segment(cursor, dofs),
              "poses@t=" + std::to_string(t),
              tol);
          cursor += dofs;
          anyError |= debugVector(
              fd.segment(cursor, dofs),
              analytical.segment(cursor, dofs),
              "vels@t=" + std::to_string(t),
              tol);
          cursor += dofs;
          anyError |= debugVector(
              fd.segment(cursor, dofs),
              analytical.segment(cursor, dofs),
              "accs@t=" + std::to_string(t),
              tol);
          cursor += dofs;
        }

        int finalT = mPoses[trial].cols() - 1;
        anyError |= debugVector(
            fd.segment(cursor, dofs),
            analytical.segment(cursor, dofs),
            "poses@t=" + std::to_string(finalT),
            tol);
        cursor += dofs;
        anyError |= debugVector(
            fd.segment(cursor, dofs),
            analytical.segment(cursor, dofs),
            "vels@t=" + std::to_string(finalT),
            tol);
        cursor += dofs;
      }
    }
  }
  return anyError;
}

//==============================================================================
// This attempts to perfect the physical consistency of the data, and writes
// them back to the problem
void DynamicsFitProblem::computePerfectGRFs()
{
  mInit->perfectGrfTrials.clear();
  mInit->perfectForcePlateTrials.clear();
  mInit->perfectTorques.clear();
  mInit->perfectGrfAsCopTorqueForces.clear();
  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    mSkeleton->setTimeStep(mInit->trialTimesteps[trial]);
    s_t groundHeight = mInit->groundHeight[trial];

    Eigen::MatrixXs perfectGrfTrial = Eigen::MatrixXs::Zero(
        mInit->grfTrials[trial].rows(), mInit->grfTrials[trial].cols());
    Eigen::MatrixXs perfectTorques = Eigen::MatrixXs::Zero(
        mSkeleton->getNumDofs(), mInit->grfTrials[trial].cols());
    Eigen::MatrixXs perfectGrfAsCopTorqueForce = Eigen::MatrixXs::Zero(
        mFootNodes.size() * 9, mInit->grfTrials[trial].cols());
    std::vector<ForcePlate> perfectForcePlates;
    for (int i = 0; i < mInit->forcePlateTrials[trial].size(); i++)
    {
      ForcePlate& originalPlate = mInit->forcePlateTrials[trial][i];
      perfectForcePlates.emplace_back();
      ForcePlate& perfectPlate
          = perfectForcePlates[perfectForcePlates.size() - 1];

      perfectPlate.worldOrigin = originalPlate.worldOrigin;
      perfectPlate.corners = originalPlate.corners;
    }
    for (int t = 1; t < mPoses[trial].cols() - 1; t++)
    {
      mSkeleton->setPositions(mPoses[trial].col(t));
      mSkeleton->setVelocities(mVels[trial].col(t));
      mSkeleton->setAccelerations(mAccs[trial].col(t));

      int activeFootIndex = -1;
      bool onlyOneActive = false;
      for (int i = 0; i < mInit->grfBodyForceActive[trial][t].size(); i++)
      {
        bool active = mInit->grfBodyForceActive[trial][t][i];
        if (active)
        {
          if (activeFootIndex == -1)
          {
            activeFootIndex = i;
            onlyOneActive = true;
          }
          else
          {
            onlyOneActive = false;
          }
        }
      }

      int activeForcePlateIndex = -1;
      bool onlyOneForcePlateActive = false;
      for (int i = 0; i < mInit->forcePlateTrials[trial].size(); i++)
      {
        if (mInit->forcePlateTrials[trial][i].forces[t].norm() > 1e-3)
        {
          if (activeForcePlateIndex == -1)
          {
            activeForcePlateIndex = i;
            onlyOneForcePlateActive = true;
          }
          else
          {
            onlyOneForcePlateActive = false;
          }
        }
      }

      if (onlyOneActive && onlyOneForcePlateActive)
      {
        auto result = mSkeleton->getContactInverseDynamics(
            mVels[trial].col(t + 1), mFootNodes[activeFootIndex]);
        perfectTorques.col(t) = result.jointTorques;
        Eigen::Vector6s worldWrench = math::dAdInvT(
            mFootNodes[activeFootIndex]->getWorldTransform(),
            result.contactWrench);
        perfectGrfTrial.block<6, 1>(activeFootIndex * 6, t) = worldWrench;

#ifndef NDEBUG
        Eigen::Vector6s residual = mResidualHelper->calculateResidual(
            mPoses[trial].col(t),
            mVels[trial].col(t),
            mAccs[trial].col(t),
            perfectGrfTrial.col(t));
        s_t norm = residual.squaredNorm();
        // std::cout << "Residual norm t=" << t << ": " << norm << std::endl;
        assert(norm < 1e-10);
#endif

        Eigen::Vector9s copWrench
            = math::projectWrenchToCoP(worldWrench, groundHeight, 1);
        perfectGrfAsCopTorqueForce.block<9, 1>(9 * activeFootIndex, t)
            = copWrench;
        // add to a force plate
        for (int i = 0; i < perfectForcePlates.size(); i++)
        {
          if (i == activeForcePlateIndex)
          {
            perfectForcePlates[i].centersOfPressure.push_back(
                copWrench.head<3>());
            perfectForcePlates[i].moments.push_back(copWrench.segment<3>(3));
            perfectForcePlates[i].forces.push_back(copWrench.segment<3>(6));
          }
          else
          {
            perfectForcePlates[i].centersOfPressure.push_back(
                Eigen::Vector3s::Zero());
            perfectForcePlates[i].forces.push_back(Eigen::Vector3s::Zero());
            perfectForcePlates[i].moments.push_back(Eigen::Vector3s::Zero());
          }
        }
      }
      else
      {
        Eigen::VectorXs sensorWorldGRF = mInit->grfTrials[trial].col(t);
        std::vector<Eigen::Vector6s> localWrenches;
        std::vector<const dynamics::BodyNode*> constFootNodes;
        for (int i = 0; i < mFootNodes.size(); i++)
        {
          constFootNodes.push_back(mFootNodes[i]);
          localWrenches.push_back(math::dAdT(
              mFootNodes[i]->getWorldTransform(),
              sensorWorldGRF.segment<6>(i * 6)));
        }
        auto resultCops = mSkeleton->getMultipleContactInverseDynamicsNearCoP(
            mVels[trial].col(t + 1),
            constFootNodes,
            localWrenches,
            mInit->groundHeight[trial],
            1,
            0.1,
            false);

        perfectTorques.col(t) = resultCops.jointTorques;

        std::vector<Eigen::Vector6s> worldWrenches;
        Eigen::VectorXs perfectGRF
            = Eigen::VectorXs::Zero(mFootNodes.size() * 6);
        for (int i = 0; i < mFootNodes.size(); i++)
        {
          Eigen::Vector6s worldWrench = math::dAdInvT(
              mFootNodes[i]->getWorldTransform(),
              resultCops.contactWrenches[i]);
          worldWrenches.push_back(worldWrench);
          perfectGRF.segment<6>(i * 6) = worldWrench;
        }
        perfectGrfTrial.col(t) = perfectGRF;

#ifndef NDEBUG
        Eigen::Vector6s residual = mResidualHelper->calculateResidual(
            mPoses[trial].col(t),
            mVels[trial].col(t),
            mAccs[trial].col(t),
            perfectGrfTrial.col(t));
        s_t norm = residual.squaredNorm();
        std::cout << "Residual norm t=" << t << ": " << norm << std::endl;
        assert(norm < 1e-10);
#endif

        std::vector<Eigen::Vector3s> forces;
        std::vector<Eigen::Vector3s> cops;
        std::vector<Eigen::Vector3s> taus;

        Eigen::MatrixXs platesToFeet = Eigen::MatrixXs::Zero(
            mInit->forcePlateTrials[trial].size(), mFootNodes.size());
        for (int i = 0; i < mFootNodes.size(); i++)
        {
          Eigen::Vector6s worldWrench = worldWrenches[i];
          Eigen::Vector9s copWrench
              = math::projectWrenchToCoP(worldWrench, groundHeight, 1);
          perfectGrfAsCopTorqueForce.block<9, 1>(9 * i, t) = copWrench;

          cops.push_back(copWrench.segment<3>(0));
          taus.push_back(copWrench.segment<3>(3));
          forces.push_back(copWrench.segment<3>(6));

          /*
          // Find the center of pressure, using the copy-pasted method from
          // above in the single-footed case
          // TODO: factor out into a utility method
          Eigen::Vector3s worldTau = worldWrench.head<3>();
          Eigen::Vector3s worldF = worldWrench.tail<3>();
          Eigen::Matrix3s crossF = math::makeSkewSymmetric(worldF);
          Eigen::Vector3s rightSide = worldTau - crossF.col(1) * groundHeight;
          Eigen::Matrix3s leftSide = -crossF;
          leftSide.col(1) = worldF;
          Eigen::Vector3s p
              = leftSide.completeOrthogonalDecomposition().solve(rightSide);
          s_t k = p(1);
          p(1) = 0;
          Eigen::Vector3s expectedTau = worldF * k;
          Eigen::Vector3s cop = p;
          cop(1) = groundHeight;

          forces.push_back(worldF);
          cops.push_back(cop);
          taus.push_back(expectedTau);

          perfectGrfAsCopTorqueForce.block<3, 1>(9 * i, t) = cop;
          perfectGrfAsCopTorqueForce.block<3, 1>(9 * i + 3, t) = expectedTau;
          perfectGrfAsCopTorqueForce.block<3, 1>(9 * i + 6, t) = worldF;
          */

          Eigen::Vector3s cop = copWrench.segment<3>(0);
          for (int j = 0; j < mInit->forcePlateTrials[trial].size(); j++)
          {
            platesToFeet(j, i)
                = 1.0
                  / (cop
                     - mInit->forcePlateTrials[trial][j].centersOfPressure[t])
                        .norm();
          }
        }
        Eigen::VectorXi platesToFeetAssignment
            = math::AssignmentMatcher::assignRowsToColumns(platesToFeet);
        for (int i = 0; i < perfectForcePlates.size(); i++)
        {
          if (platesToFeetAssignment(i) == -1)
          {
            perfectForcePlates[i].centersOfPressure.push_back(
                Eigen::Vector3s::Zero());
            perfectForcePlates[i].forces.push_back(Eigen::Vector3s::Zero());
            perfectForcePlates[i].moments.push_back(Eigen::Vector3s::Zero());
          }
          else if (perfectForcePlates[i].centersOfPressure.size() <= t)
          {
            perfectForcePlates[i].centersOfPressure.push_back(
                cops[platesToFeetAssignment(i)]);
            perfectForcePlates[i].forces.push_back(
                forces[platesToFeetAssignment(i)]);
            perfectForcePlates[i].moments.push_back(
                taus[platesToFeetAssignment(i)]);
          }
        }
      }
    }
    mInit->perfectGrfTrials.push_back(perfectGrfTrial);
    mInit->perfectTorques.push_back(perfectTorques);
    mInit->perfectGrfAsCopTorqueForces.push_back(perfectGrfAsCopTorqueForce);
    mInit->perfectForcePlateTrials.push_back(perfectForcePlates);
  }
}

//==============================================================================
DynamicsFitProblemConfig::DynamicsFitProblemConfig(
    std::shared_ptr<dynamics::Skeleton> skeleton)
  : mIncludeMasses(false),
    mIncludeCOMs(false),
    mIncludeInertias(false),
    mIncludeBodyScales(false),
    mIncludePoses(false),
    mIncludeMarkerOffsets(false),
    mResidualWeight(0),
    mLinearNewtonWeight(0),
    mMarkerWeight(0.0),
    mJointWeight(0.0),
    mLinearNewtonUseL1(true),
    mResidualUseL1(true),
    mMarkerUseL1(true),
    mResidualTorqueMultiple(0.0),
    mRegularizeAcc(0.0),
    mRegularizeAccBodyWeights(
        Eigen::VectorXs::Ones(skeleton->getNumBodyNodes())),
    mRegularizeAccUseL1(false),
    mRegularizeMasses(0.0),
    mRegularizeCOMs(0.0),
    mRegularizeInertias(0.0),
    mRegularizeTrackingMarkerOffsets(0.0),
    mRegularizeAnatomicalMarkerOffsets(0.0),
    mRegularizeImpliedDensity(0),
    mRegularizeBodyScales(0.0),
    mRegularizePoses(0.0),
    mVelAccImplicit(false)
// mResidualWeight(0.1),
// mLinearNewtonWeight(0.1),
// mMarkerWeight(1.0),
// mJointWeight(1.0),
// mLinearNewtonUseL1(true),
// mResidualUseL1(true),
// mMarkerUseL1(true),
// mResidualTorqueMultiple(3.0),
// mRegularizeAcc(0.01),
// mRegularizeAccBodyWeights(
//     Eigen::VectorXs::Ones(skeleton->getNumBodyNodes())),
// mRegularizeAccUseL1(false),
// mRegularizeMasses(10.0),
// mRegularizeCOMs(20.0),
// mRegularizeInertias(1.0),
// mRegularizeTrackingMarkerOffsets(0.05),
// mRegularizeAnatomicalMarkerOffsets(10.0),
// mRegularizeImpliedDensity(3e-8),
// mRegularizeBodyScales(1.0),
// mRegularizePoses(0.0),
// mVelAccImplicit(false)
{
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setDefaults(bool l1)
{
  // mResidualWeight = 2e-2;
  mResidualWeight = 1e-2;
  mMarkerWeight = 50;
  mLinearNewtonWeight = 1e-2;
  mJointWeight = 1.0;
  mLinearNewtonUseL1 = true;
  mResidualUseL1 = true;
  mMarkerUseL1 = true;
  mResidualTorqueMultiple = 3.0;
  mRegularizeAcc = 0;
  mRegularizeAccUseL1 = false;
  mRegularizeMasses = 10.0;
  mRegularizeCOMs = 20.0;
  mRegularizeInertias = 1.0;
  mRegularizeTrackingMarkerOffsets = 0.05;
  mRegularizeAnatomicalMarkerOffsets = 10.0;
  mRegularizeImpliedDensity = 3e-8;
  mRegularizeBodyScales = 1.0;
  mRegularizePoses = 0.0;
  mVelAccImplicit = false;

  if (!l1)
  {
    // mMarkerWeight = 500;
    mMarkerWeight = 2500;
    mResidualWeight = 1e-5;
    mLinearNewtonWeight = 5e-4;
    mResidualUseL1 = false;
    mLinearNewtonUseL1 = false;
    mMarkerUseL1 = false;
  }

  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setIncludeMasses(bool value)
{
  mIncludeMasses = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setIncludeCOMs(bool value)
{
  mIncludeCOMs = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setIncludeInertias(
    bool value)
{
  mIncludeInertias = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setIncludePoses(bool value)
{
  mIncludePoses = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setIncludeMarkerOffsets(
    bool value)
{
  mIncludeMarkerOffsets = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setIncludeBodyScales(
    bool value)
{
  mIncludeBodyScales = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setLinearNewtonWeight(
    s_t weight)
{
  mLinearNewtonWeight = weight;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setResidualWeight(
    s_t weight)
{
  mResidualWeight = weight;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setResidualTorqueMultiple(
    s_t value)
{
  mResidualTorqueMultiple = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setMarkerWeight(s_t weight)
{
  mMarkerWeight = weight;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setJointWeight(s_t weight)
{
  mJointWeight = weight;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setLinearNewtonUseL1(
    bool l1)
{
  mLinearNewtonUseL1 = l1;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setResidualUseL1(bool l1)
{
  mResidualUseL1 = l1;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setMarkerUseL1(bool l1)
{
  mMarkerUseL1 = l1;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setRegularizeSpatialAcc(
    s_t value)
{
  mRegularizeAcc = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig&
DynamicsFitProblemConfig::setRegularizeSpatialAccBodyWeights(
    Eigen::VectorXs bodyWeights)
{
  mRegularizeAccBodyWeights = bodyWeights;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig&
DynamicsFitProblemConfig::setRegularizeSpatialAccUseL1(bool l1)
{
  mRegularizeAccUseL1 = l1;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setRegularizeMasses(
    s_t value)
{
  mRegularizeMasses = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setRegularizeCOMs(s_t value)
{
  mRegularizeCOMs = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setRegularizeInertias(
    s_t value)
{
  mRegularizeInertias = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setRegularizeBodyScales(
    s_t value)
{
  mRegularizeBodyScales = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setRegularizePoses(
    s_t value)
{
  mRegularizePoses = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig&
DynamicsFitProblemConfig::setRegularizeTrackingMarkerOffsets(s_t value)
{
  mRegularizeTrackingMarkerOffsets = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig&
DynamicsFitProblemConfig::setRegularizeAnatomicalMarkerOffsets(s_t value)
{
  mRegularizeAnatomicalMarkerOffsets = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setRegularizeImpliedDensity(
    s_t value)
{
  mRegularizeImpliedDensity = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblemConfig& DynamicsFitProblemConfig::setVelAccImplicit(
    bool implicit)
{
  mVelAccImplicit = implicit;
  return *(this);
}

//------------------------- Ipopt::TNLP --------------------------------------

//==============================================================================
/// \brief Method to return some info about the nlp
bool DynamicsFitProblem::get_nlp_info(
    Ipopt::Index& n,
    Ipopt::Index& m,
    Ipopt::Index& nnz_jac_g,
    Ipopt::Index& nnz_h_lag,
    Ipopt::TNLP::IndexStyleEnum& index_style)
{
  // Set the number of decision variables
  n = getProblemSize();

  // Set the total number of constraints
  m = getConstraintSize();

  // Set the number of entries in the constraint Jacobian
  nnz_jac_g = computeSparseConstraintsJacobian().size();

  // Set the number of entries in the Hessian
  nnz_h_lag = n * n;

  // use the C style indexing (0-based)
  index_style = Ipopt::TNLP::C_STYLE;

  return true;
}

//==============================================================================
/// \brief Method to return the bounds for my problem
bool DynamicsFitProblem::get_bounds_info(
    Ipopt::Index n,
    Ipopt::Number* x_l,
    Ipopt::Number* x_u,
    Ipopt::Index m,
    Ipopt::Number* g_l,
    Ipopt::Number* g_u)
{
  // Lower and upper bounds on X
  Eigen::Map<Eigen::VectorXd> upperBounds(x_u, n);
  upperBounds.setConstant(std::numeric_limits<double>::infinity());
  Eigen::Map<Eigen::VectorXd> lowerBounds(x_l, n);
  lowerBounds.setConstant(-1 * std::numeric_limits<double>::infinity());

  upperBounds = flattenUpperBound().cast<double>();
  lowerBounds = flattenLowerBound().cast<double>();

  // Our constraint function has to be 0
  Eigen::Map<Eigen::VectorXd> constraintUpperBounds(g_u, m);
  constraintUpperBounds.setZero();
  Eigen::Map<Eigen::VectorXd> constraintLowerBounds(g_l, m);
  constraintLowerBounds.setZero();

  return true;
}

//==============================================================================
/// \brief Method to return the starting point for the algorithm
bool DynamicsFitProblem::get_starting_point(
    Ipopt::Index n,
    bool init_x,
    Ipopt::Number* _x,
    bool init_z,
    Ipopt::Number* z_L,
    Ipopt::Number* z_U,
    Ipopt::Index m,
    bool init_lambda,
    Ipopt::Number* lambda)
{
  // Here, we assume we only have starting values for x
  (void)init_x;
  assert(init_x == true);
  (void)init_z;
  assert(init_z == false);
  (void)init_lambda;
  assert(init_lambda == false);
  // We don't set the lagrange multipliers
  (void)z_L;
  (void)z_U;
  (void)m;
  (void)lambda;

  if (init_x)
  {
    Eigen::Map<Eigen::VectorXd> x(_x, n);
    x = flatten().cast<double>();
  }

  return true;
}

//==============================================================================
/// \brief Method to return the objective value
bool DynamicsFitProblem::eval_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number& _obj_value)
{
  (void)_new_x;
  Eigen::Map<const Eigen::VectorXd> x(_x, _n);

  _obj_value = (double)computeLoss(x.cast<s_t>(), true);

  return true;
}

//==============================================================================
/// \brief Method to return the gradient of the objective
bool DynamicsFitProblem::eval_grad_f(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number* _grad_f)
{
  (void)_new_x;
  Eigen::Map<const Eigen::VectorXd> x(_x, _n);
  Eigen::Map<Eigen::VectorXd> grad(_grad_f, _n);

  grad = computeGradient(x.cast<s_t>()).cast<double>();

  return true;
}

//==============================================================================
/// \brief Method to return the constraint residuals
bool DynamicsFitProblem::eval_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Number* _g)
{
  (void)_new_x;
  Eigen::Map<const Eigen::VectorXd> x(_x, _n);
  Eigen::Map<Eigen::VectorXd> g(_g, _m);

  g = computeConstraints(x.cast<s_t>()).cast<double>();

  return true;
}

//==============================================================================
/// \brief Method to return:
///        1) The structure of the jacobian (if "values" is nullptr)
///        2) The values of the jacobian (if "values" is not nullptr)
bool DynamicsFitProblem::eval_jac_g(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Index _m,
    Ipopt::Index _nnzj,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  (void)_new_x;
  (void)_m;

  // If the iRow and jCol arguments are not nullptr, then IPOPT wants you to
  // fill in the sparsity structure of the Jacobian (the row and column
  // indices only). At this time, the x argument and the values argument will
  // be nullptr.

  std::vector<std::tuple<int, int, s_t>> sparse
      = computeSparseConstraintsJacobian();

  if (nullptr == _x)
  {
    Eigen::Map<Eigen::VectorXi> rows(_iRow, _nnzj);
    Eigen::Map<Eigen::VectorXi> cols(_jCol, _nnzj);
    for (int i = 0; i < sparse.size(); i++)
    {
      rows(i) = (double)std::get<0>(sparse[i]);
      cols(i) = (double)std::get<1>(sparse[i]);
    }
  }
  else
  {
    // Return the concatenated gradient of everything
    Eigen::Map<const Eigen::VectorXd> x(_x, _n);
    Eigen::Map<Eigen::VectorXd> vals(_values, _nnzj);

    for (int i = 0; i < sparse.size(); i++)
    {
      vals(i) = (double)std::get<2>(sparse[i]);
    }
  }

  return true;
}

//==============================================================================
/// \brief Method to return:
///        1) The structure of the hessian of the lagrangian (if "values" is
///           nullptr)
///        2) The values of the hessian of the lagrangian (if "values" is not
///           nullptr)
bool DynamicsFitProblem::eval_h(
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    bool _new_x,
    Ipopt::Number _obj_factor,
    Ipopt::Index _m,
    const Ipopt::Number* _lambda,
    bool _new_lambda,
    Ipopt::Index _nele_hess,
    Ipopt::Index* _iRow,
    Ipopt::Index* _jCol,
    Ipopt::Number* _values)
{
  (void)_n;
  (void)_x;
  (void)_new_x;
  (void)_obj_factor;
  (void)_m;
  (void)_lambda;
  (void)_new_lambda;
  (void)_nele_hess;
  (void)_iRow;
  (void)_jCol;
  (void)_values;
  return false;
}

//==============================================================================
/// \brief This method is called when the algorithm is complete so the TNLP
///        can store/write the solution
void DynamicsFitProblem::finalize_solution(
    Ipopt::SolverReturn _status,
    Ipopt::Index _n,
    const Ipopt::Number* _x,
    const Ipopt::Number* _z_L,
    const Ipopt::Number* _z_U,
    Ipopt::Index _m,
    const Ipopt::Number* _g,
    const Ipopt::Number* _lambda,
    Ipopt::Number _obj_value,
    const Ipopt::IpoptData* _ip_data,
    Ipopt::IpoptCalculatedQuantities* _ip_cq)
{
  (void)_status;
  (void)_n;
  (void)_x;
  (void)_z_L;
  (void)_z_U;
  (void)_m;
  (void)_g;
  (void)_lambda;
  (void)_obj_value;
  (void)_ip_data;
  (void)_ip_cq;
  // Eigen::Map<const Eigen::VectorXd> x(_x, _n);
  std::cout << "Recovering state with best loss: iteration "
            << mBestObjectiveValueIteration << " with " << mBestObjectiveValue
            << std::endl;
  Eigen::VectorXs x = mBestObjectiveValueState;

  unflatten(x);

  if (mConfig.mIncludeMasses)
  {
    mInit->groupMasses = mSkeleton->getGroupMasses();
    mInit->bodyMasses = mSkeleton->getLinkMasses();
  }
  if (mConfig.mIncludeCOMs)
  {
    mInit->bodyCom.resize(3, mSkeleton->getNumBodyNodes());
    for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
    {
      mInit->bodyCom.col(i)
          = mSkeleton->getBodyNode(i)->getInertia().getLocalCOM();
    }
  }
  if (mConfig.mIncludeInertias)
  {
    mInit->groupInertias = mSkeleton->getGroupInertias();
    mInit->bodyInertia.resize(6, mSkeleton->getNumBodyNodes());
    for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
    {
      mInit->bodyInertia.col(i)
          = mSkeleton->getBodyNode(i)->getInertia().getDimsAndEulerVector();
    }
  }
  if (mConfig.mIncludeBodyScales)
  {
    mInit->groupScales = mSkeleton->getGroupScales();
  }
  if (mConfig.mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      mInit->markerOffsets[mMarkerNames[i]] = mMarkers[i].second;
      mInit->updatedMarkerMap[mMarkerNames[i]] = mMarkers[i];
    }
  }
  if (mConfig.mIncludePoses)
  {
    mInit->poseTrials = mPoses;
  }
}

//==============================================================================
bool DynamicsFitProblem::intermediate_callback(
    Ipopt::AlgorithmMode mode,
    Ipopt::Index iter,
    Ipopt::Number obj_value,
    Ipopt::Number inf_pr,
    Ipopt::Number inf_du,
    Ipopt::Number mu,
    Ipopt::Number d_norm,
    Ipopt::Number regularization_size,
    Ipopt::Number alpha_du,
    Ipopt::Number alpha_pr,
    Ipopt::Index ls_trials,
    const Ipopt::IpoptData* ip_data,
    Ipopt::IpoptCalculatedQuantities* ip_cq)
{
  (void)mode;
  (void)iter;
  (void)obj_value;
  (void)inf_pr;
  (void)inf_du;
  (void)mu;
  (void)d_norm;
  (void)regularization_size;
  (void)alpha_du;
  (void)alpha_pr;
  (void)ls_trials;
  (void)ip_data;
  (void)ip_cq;

  if (obj_value < mBestObjectiveValue && abs(inf_pr) < 1.0)
  {
    mBestObjectiveValueIteration = iter;
    mBestObjectiveValue = obj_value;
    mBestObjectiveValueState = mLastX;
  }

  return true;
}

//==============================================================================
DynamicsFitter::DynamicsFitter(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    std::vector<dynamics::BodyNode*> footNodes,
    std::vector<std::string> trackingMarkers)
  : mSkeleton(skeleton),
    mFootNodes(footNodes),
    mTrackingMarkers(trackingMarkers),
    mTolerance(1e-8),
    mIterationLimit(500),
    mLBFGSHistoryLength(8),
    mCheckDerivatives(false),
    mPrintFrequency(1),
    mSilenceOutput(false),
    mDisableLinesearch(false){

    };

//==============================================================================
// This bundles together the objects we need in order to track a dynamics
// problem around through multiple steps of optimization
std::shared_ptr<DynamicsInitialization> DynamicsFitter::createInitialization(
    std::shared_ptr<dynamics::Skeleton> skel,
    dynamics::MarkerMap markerMap,
    std::vector<std::string> trackingMarkers,
    std::vector<dynamics::BodyNode*> grfNodes,
    std::vector<std::vector<ForcePlate>> forcePlateTrials,
    std::vector<Eigen::MatrixXs> poseTrials,
    std::vector<int> framesPerSecond,
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        markerObservationTrials)
{
  std::shared_ptr<DynamicsInitialization> init
      = std::make_shared<DynamicsInitialization>();
  init->forcePlateTrials = forcePlateTrials;
  init->originalPoses = poseTrials;
  init->markerObservationTrials = markerObservationTrials;
  init->trackingMarkers = trackingMarkers;
  init->updatedMarkerMap = markerMap;
  for (auto& pair : markerMap)
  {
    init->markerOffsets[pair.first] = pair.second.second;
  }
  init->bodyMasses = skel->getLinkMasses();
  init->groupMasses = skel->getGroupMasses();
  init->groupScales = skel->getGroupScales();
  init->bodyCom.resize(3, skel->getNumBodyNodes());
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    init->bodyCom.col(i) = skel->getBodyNode(i)->getInertia().getLocalCOM();
  }
  init->groupInertias = skel->getGroupInertias();
  init->bodyInertia.resize(6, skel->getNumBodyNodes());
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    init->bodyInertia.col(i)
        = skel->getBodyNode(i)->getInertia().getDimsAndEulerVector();
  }
  init->bodyMasses.resize(skel->getNumBodyNodes());
  for (int i = 0; i < skel->getNumBodyNodes(); i++)
  {
    assert(
        skel->getBodyNode(i)->getInertia().getMass()
        == skel->getBodyNode(i)->getMass());
    init->bodyMasses(i) = skel->getBodyNode(i)->getInertia().getMass();
  }

  // Initially smooth the accelerations just a little bit

  for (int i = 0; i < init->originalPoses.size(); i++)
  {
    init->poseTrials.push_back(init->originalPoses[i]);
    init->trialTimesteps.push_back(1.0 / framesPerSecond[i]);
  }

  // Match force plates to the feet

  Eigen::VectorXs originalPose = skel->getPositions();

  init->grfBodyNodes = grfNodes;
  for (int i = 0; i < grfNodes.size(); i++)
  {
    init->grfBodyIndices.push_back(grfNodes[i]->getIndexInSkeleton());
  }

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    std::vector<ForcePlate> forcePlates = init->forcePlateTrials[trial];

    Eigen::MatrixXs& poses = init->poseTrials[trial];

    Eigen::MatrixXs GRF
        = Eigen::MatrixXs::Zero(grfNodes.size() * 6, poses.cols());

    for (int t = 0; t < poses.cols(); t++)
    {
      skel->setPositions(poses.col(t));

      for (int i = 0; i < forcePlates.size(); i++)
      {
        Eigen::Vector3s cop = forcePlates[i].centersOfPressure[t];
        Eigen::Vector3s force = forcePlates[i].forces[t];
        Eigen::Vector3s moments = forcePlates[i].moments[t];
        // Ignore timesteps where the force plate has a 0 force, those don't
        // need to be assigned to anything
        if (force.norm() == 0 || force.hasNaN() || cop.hasNaN()
            || moments.hasNaN())
        {
          continue;
        }
        Eigen::Vector6s wrench = Eigen::Vector6s::Zero();
        wrench.head<3>() = moments;
        wrench.tail<3>() = force;
        Eigen::Isometry3s wrenchT = Eigen::Isometry3s::Identity();
        wrenchT.translation() = cop;
        Eigen::Vector6s worldWrench = math::dAdInvT(wrenchT, wrench);

        // Every force from force plates must be accounted for somewhere. Simply
        // assign it to the nearest foot
        int closestFoot = -1;
        s_t minDist = (s_t)std::numeric_limits<double>::infinity();
        for (int i = 0; i < grfNodes.size(); i++)
        {
          Eigen::Vector3s footLoc
              = grfNodes[i]->getWorldTransform().translation();
          s_t dist = (footLoc - cop).norm();
          if (dist < minDist)
          {
            minDist = dist;
            closestFoot = i;
          }
        }
        assert(closestFoot != -1);

        // If multiple force plates assign to the same foot, sum up the forces
        GRF.block<6, 1>(closestFoot * 6, t) += worldWrench;
      }
      std::cout << "Trial " << trial << " t=" << t << ": GRF norm "
                << GRF.col(t).norm() << std::endl;
    }
    init->grfTrials.push_back(GRF);
  }

  // Make copies of data to use for regularization
  init->originalGroupMasses = skel->getGroupMasses();
  init->originalGroupCOMs = skel->getGroupCOMs();
  init->originalGroupInertias = skel->getGroupInertias();
  init->originalGroupScales = skel->getGroupScales();
  for (auto& pair : init->markerOffsets)
  {
    init->originalMarkerOffsets[pair.first] = pair.second;
  }

  return init;
}

//==============================================================================
// This creates an optimization problem from a kinematics initialization
std::shared_ptr<DynamicsInitialization> DynamicsFitter::createInitialization(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<MarkerInitialization> kinematicInit,
    std::vector<std::string> trackingMarkers,
    std::vector<dynamics::BodyNode*> grfNodes,
    std::vector<std::vector<ForcePlate>> forcePlateTrials,
    std::vector<int> framesPerSecond,
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        markerObservationTrials)
{
  if (markerObservationTrials.size() != kinematicInit.size())
  {
    std::cout
        << "ERROR: Passed a different number of markerObservationTrials ("
        << markerObservationTrials.size() << ") than kinematic inits ("
        << kinematicInit.size()
        << ") to DynamicsFitter::createInitialization()! Exiting with code 1."
        << std::endl;
    exit(1);
  }
  // Split the incoming poses into individual trial matrices
  std::vector<Eigen::MatrixXs> poseTrials;
  for (int trial = 0; trial < markerObservationTrials.size(); trial++)
  {
    poseTrials.push_back(kinematicInit[trial].poses);
  }

  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
      translatedMarkerMap;
  for (auto& pair : kinematicInit[0].updatedMarkerMap)
  {
    translatedMarkerMap[pair.first] = std::make_pair(
        skel->getBodyNode(pair.second.first->getName()), pair.second.second);
  }

  // Create the basic initialization
  std::shared_ptr<DynamicsInitialization> init = createInitialization(
      skel,
      translatedMarkerMap,
      trackingMarkers,
      grfNodes,
      forcePlateTrials,
      poseTrials,
      framesPerSecond,
      markerObservationTrials);

  // Copy over the initial body scaling
  init->groupScales = kinematicInit[0].groupScales;

  // Copy over the joint data
  init->joints.clear();
  for (int i = 0; i < kinematicInit[0].joints.size(); i++)
  {
    init->joints.push_back(
        skel->getJoint(kinematicInit[0].joints[i]->getName()));
  }
  init->jointsAdjacentMarkers = kinematicInit[0].jointsAdjacentMarkers;
  init->jointWeights = kinematicInit[0].jointWeights;
  init->axisWeights = kinematicInit[0].axisWeights;

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    init->jointCenters.push_back(kinematicInit[trial].jointCenters);
    init->jointAxis.push_back(kinematicInit[trial].jointAxis);
  }

  return init;
}

//==============================================================================
// This retargets a dynamics initialization to another skeleton
std::shared_ptr<DynamicsInitialization> DynamicsFitter::retargetInitialization(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::shared_ptr<dynamics::Skeleton> simplifiedSkel,
    std::shared_ptr<DynamicsInitialization> init)
{
  std::shared_ptr<DynamicsInitialization> retargeted
      = std::make_shared<DynamicsInitialization>();
  retargeted->probablyMissingGRF = init->probablyMissingGRF;

  for (int i = 0; i < init->joints.size(); i++)
  {
    retargeted->joints.push_back(
        simplifiedSkel->getJoint(init->joints[i]->getName()));
  }
  for (int i = 0; i < init->grfBodyNodes.size(); i++)
  {
    retargeted->grfBodyNodes.push_back(
        simplifiedSkel->getBodyNode(init->grfBodyNodes[i]->getName()));
  }
  for (int i = 0; i < init->contactBodies.size(); i++)
  {
    std::vector<dynamics::BodyNode*> bodySet;
    for (int j = 0; j < init->contactBodies[i].size(); j++)
    {
      bodySet.push_back(
          simplifiedSkel->getBodyNode(init->contactBodies[i][j]->getName()));
    }
    retargeted->contactBodies.push_back(bodySet);
  }

  // Inputs from files
  retargeted->forcePlateTrials = init->forcePlateTrials;
  retargeted->markerObservationTrials = init->markerObservationTrials;
  retargeted->trialTimesteps = init->trialTimesteps;

  // Assigning GRFs to specific feet
  retargeted->grfTrials = init->grfTrials;
  retargeted->grfBodyIndices = init->grfBodyIndices;

  // Pure dynamics values
  retargeted->bodyMasses = simplifiedSkel->getLinkMasses();
  retargeted->groupMasses = simplifiedSkel->getGroupMasses();
  retargeted->groupInertias = simplifiedSkel->getGroupInertias();
  retargeted->bodyCom = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(
      3, simplifiedSkel->getNumBodyNodes());
  retargeted->bodyInertia = Eigen::Matrix<s_t, 6, Eigen::Dynamic>::Zero(
      6, simplifiedSkel->getNumBodyNodes());
  for (int i = 0; i < simplifiedSkel->getNumBodyNodes(); i++)
  {
    retargeted->bodyCom.col(i) = simplifiedSkel->getBodyNode(i)->getLocalCOM();
    retargeted->bodyInertia.col(i)
        = simplifiedSkel->getBodyNode(i)->getInertia().getMomentVector();
  }

  // Values from the kinematics fitter that are relevant here as well. The key
  // difference is that the per-trial split is explicit here, because it's so
  // important for the dynamics and indexing.
  retargeted->groupScales = simplifiedSkel->getGroupScales();
  retargeted->markerOffsets = init->markerOffsets;
  retargeted->trackingMarkers = init->trackingMarkers;

  // TODO: we may need to recalculate this for simplified skeletons
  retargeted->jointsAdjacentMarkers = init->jointsAdjacentMarkers;
  retargeted->jointWeights = init->jointWeights;
  retargeted->jointCenters = init->jointCenters;
  retargeted->axisWeights = init->axisWeights;
  retargeted->jointAxis = init->jointAxis;

  // Convenience objects
  for (auto& pair : init->updatedMarkerMap)
  {
    retargeted->updatedMarkerMap[pair.first]
        = std::pair<dynamics::BodyNode*, Eigen::Vector3s>(
            simplifiedSkel->getBodyNode(pair.second.first->getName()),
            pair.second.second);
  }

  // To support regularization
  retargeted->originalGroupMasses = simplifiedSkel->getGroupMasses();
  retargeted->originalGroupCOMs = simplifiedSkel->getGroupCOMs();
  retargeted->originalGroupInertias = simplifiedSkel->getGroupInertias();
  retargeted->originalGroupScales = simplifiedSkel->getGroupScales();
  retargeted->originalMarkerOffsets = init->originalMarkerOffsets;

  Eigen::VectorXs originalSkelPos = skel->getPositions();
  Eigen::VectorXs originalSimplifiedSkelPos = simplifiedSkel->getPositions();

  skel->setPositions(Eigen::VectorXs::Zero(originalSkelPos.size()));
  simplifiedSkel->setPositions(
      Eigen::VectorXs::Zero(originalSimplifiedSkelPos.size()));

  SkeletonConverter converter(simplifiedSkel, skel);
  for (int i = 0; i < skel->getNumJoints(); i++)
  {
    auto* simplifiedJoint
        = simplifiedSkel->getJoint(skel->getJoint(i)->getName());
    if (simplifiedJoint != nullptr)
    {
      converter.linkJoints(simplifiedJoint, skel->getJoint(i));
    }
  }
  converter.createVirtualMarkers();
  for (int trial = 0; trial < init->originalPoses.size(); trial++)
  {
    retargeted->originalPoses.push_back(
        converter.convertMotion(init->originalPoses[trial]));
    retargeted->poseTrials.push_back(
        converter.convertMotion(init->poseTrials[trial]));
  }

  skel->setPositions(originalSkelPos);
  simplifiedSkel->setPositions(originalSimplifiedSkelPos);

  DynamicsFitter fitter(
      simplifiedSkel, retargeted->grfBodyNodes, retargeted->trackingMarkers);
  fitter.estimateFootGroundContacts(retargeted);

  return retargeted;
}

//==============================================================================
// This computes and returns the positions of the center of mass at each
// frame
std::vector<Eigen::Vector3s> DynamicsFitter::comPositions(
    std::shared_ptr<DynamicsInitialization> init, int trial)
{
  Eigen::VectorXs originalMasses = mSkeleton->getLinkMasses();
  Eigen::VectorXs originalPoses = mSkeleton->getPositions();

  if (trial >= init->poseTrials.size())
  {
    std::cout << "Trying to get accelerations on an out-of-bounds trial: "
              << trial << " >= " << init->poseTrials.size() << std::endl;
    exit(1);
  }
  const Eigen::MatrixXs& poses = init->poseTrials[trial];

  std::vector<Eigen::Vector3s> coms;

  mSkeleton->setLinkMasses(init->bodyMasses);
  for (int timestep = 0; timestep < poses.cols(); timestep++)
  {
    mSkeleton->setPositions(poses.col(timestep));
    Eigen::Vector3s weightedCOM = Eigen::Vector3s::Zero();
    s_t totalMass = 0.0;
    for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
    {
      totalMass += mSkeleton->getBodyNode(i)->getMass();
      weightedCOM += mSkeleton->getBodyNode(i)->getCOM()
                     * mSkeleton->getBodyNode(i)->getMass();
    }
    weightedCOM /= totalMass;
    coms.push_back(weightedCOM);
  }

  mSkeleton->setLinkMasses(originalMasses);
  mSkeleton->setPositions(originalPoses);

  return coms;
}

//==============================================================================
// This computes and returns the acceleration of the center of mass at each
// frame
std::vector<Eigen::Vector3s> DynamicsFitter::comAccelerations(
    std::shared_ptr<DynamicsInitialization> init, int trial)
{
  s_t dt = init->trialTimesteps[trial];
  std::vector<Eigen::Vector3s> coms = comPositions(init, trial);
  std::vector<Eigen::Vector3s> accs;
  for (int i = 1; i < coms.size() - 1; i++)
  {
    Eigen::Vector3s v1 = (coms[i] - coms[i - 1]) / dt;
    Eigen::Vector3s v2 = (coms[i + 1] - coms[i]) / dt;
    Eigen::Vector3s acc = (v2 - v1) / dt;
    accs.push_back(acc);
  }
  return accs;
}

//==============================================================================
// This computes and returns a list of the net forces on the center of mass,
// given the motion and link masses
std::vector<Eigen::Vector3s> DynamicsFitter::impliedCOMForces(
    std::shared_ptr<DynamicsInitialization> init,
    int trial,
    Eigen::Vector3s gravity)
{
  std::vector<Eigen::Vector3s> accs = comAccelerations(init, trial);
  s_t totalMass = init->bodyMasses.sum();

  std::vector<Eigen::Vector3s> forces;
  for (int i = 0; i < accs.size(); i++)
  {
    // f + m * g = m * a
    // f = m * (a - g)
    Eigen::Vector3s a = accs[i];
    a -= gravity;
    forces.push_back(a * totalMass);
  }
  return forces;
}

//==============================================================================
// This returns a list of the total GRF force on the body at each timestep
std::vector<Eigen::Vector3s> DynamicsFitter::measuredGRFForces(
    std::shared_ptr<DynamicsInitialization> init, int trial)
{
  std::vector<Eigen::Vector3s> forces;

  for (int timestep = 1; timestep < init->poseTrials[trial].cols() - 1;
       timestep++)
  {
    Eigen::Vector3s totalForce = Eigen::Vector3s::Zero();
    for (int i = 0; i < init->forcePlateTrials[trial].size(); i++)
    {
      if (!init->forcePlateTrials[trial][i].forces[timestep].hasNaN())
      {
        totalForce += init->forcePlateTrials[trial][i].forces[timestep];
      }
    }

    forces.push_back(totalForce);
  }

  return forces;
}

//==============================================================================
// 0. Estimate when each foot is in contact with the ground, which we can use
// to infer when we're missing GRF data on certain timesteps, so we don't let
// it mess with our optimization.
void DynamicsFitter::estimateFootGroundContacts(
    std::shared_ptr<DynamicsInitialization> init)
{
  Eigen::VectorXs originalPose = mSkeleton->getPositions();

  // 0. Expand the set of grf bodies to include any childen that are not already
  // themselves grf bodies
  //
  // Result goes in std::vector<std::vector<dynamics::BodyNode*>> contactBodies
  // in `init`
  for (int i = 0; i < init->grfBodyNodes.size(); i++)
  {
    dynamics::BodyNode* rootBody = init->grfBodyNodes[i];
    std::vector<dynamics::BodyNode*> extendedContactBodies;

    std::queue<dynamics::BodyNode*> queue;
    queue.push(rootBody);
    while (!queue.empty())
    {
      dynamics::BodyNode* cursor = queue.front();
      queue.pop();

      extendedContactBodies.push_back(cursor);
      for (int j = 0; j < cursor->getNumChildBodyNodes(); j++)
      {
        dynamics::BodyNode* child = cursor->getChildBodyNode(j);
        if (std::find(
                init->grfBodyNodes.begin(), init->grfBodyNodes.end(), child)
            == init->grfBodyNodes.end())
        {
          queue.push(child);
        }
      }
    }

    init->contactBodies.push_back(extendedContactBodies);
  }

  for (int trial = 0; trial < init->forcePlateTrials.size(); trial++)
  {
    bool noGroundCorners = true;

    // 1.1. First check for the ground level from the force plates

    s_t groundHeight = std::numeric_limits<s_t>::infinity();
    bool flatGround = true;
    for (ForcePlate& forcePlate : init->forcePlateTrials[trial])
    {
      for (Eigen::Vector3s corner : forcePlate.corners)
      {
        if (noGroundCorners)
        {
          groundHeight = corner(1);
          noGroundCorners = false;
        }
        else
        {
          if (abs(groundHeight - corner(1)) < 1e-8)
          {
            flatGround = false;
          }
        }
      }
    }

    // 1.2. Check the ground level from the GRF data, if we don't have force
    // plate data

    if (noGroundCorners)
    {
      for (int t = 0; t < init->poseTrials[trial].cols(); t++)
      {
        for (ForcePlate& forcePlate : init->forcePlateTrials[trial])
        {
          s_t height = forcePlate.centersOfPressure[t](1);
          if (noGroundCorners)
          {
            groundHeight = height;
            noGroundCorners = false;
          }
          else if (height < groundHeight)
          {
            groundHeight = height;
          }
        }
      }
    }

    assert(!isnan(groundHeight));

    // 2.0. Check for the size of the contact spheres to check for contact
    // Since each grf body actually gets a (potentially) extended set of contact
    // bodies attached to it, each grf body gets an array of contact sphere
    // radii, one for each contact sphere.

    std::vector<std::vector<s_t>> grfContactSphereSizes;
    for (int b = 0; b < init->contactBodies.size(); b++)
    {
      grfContactSphereSizes.emplace_back();
      for (int c = 0; c < init->contactBodies[b].size(); c++)
      {
        grfContactSphereSizes[grfContactSphereSizes.size() - 1].push_back(0.0);
      }
    }

    for (int t = 0; t < init->poseTrials[trial].cols(); t++)
    {
      mSkeleton->setPositions(init->poseTrials[trial].col(t));

      for (int b = 0; b < init->grfBodyNodes.size(); b++)
      {
        bool footActive
            = (init->grfTrials[trial].col(t).segment<6>(b * 6).squaredNorm()
               > 1e-3);

        // If this foot is active on this timestep, then we have to resize our
        // contact spheres to ensure that they show contact on this frame. We do
        // this by ensuring that the closest sphere is at least large enough to
        // hit contact.
        if (footActive)
        {
          // Check which of the contact bodies is closest to the ground
          s_t minDist = std::numeric_limits<s_t>::infinity();
          int closestBody = -1;
          for (int c = 0; c < init->contactBodies[b].size(); c++)
          {
            auto* body = init->contactBodies[b][c];
            Eigen::Vector3s worldPos = body->getWorldTransform().translation();
            s_t dist = worldPos(1) - groundHeight;
            if (dist < minDist)
            {
              minDist = dist;
              closestBody = c;
            }
          }

          // If our closest sphere needs to expand to hit the ground, then
          // expand it
          if (minDist > grfContactSphereSizes[b][closestBody])
          {
            grfContactSphereSizes[b][closestBody] = minDist;
          }
        }
      }
    }

    init->grfBodyContactSphereRadius.push_back(grfContactSphereSizes);
    init->groundHeight.push_back(groundHeight);
    init->flatGround.push_back(flatGround);

    // 3. Create the default force plate size, if needed.

    // 3.1. We only need a default force plate if any of the force plates in the
    // trials lack the corners array
    bool needDefaultForcePlate = false;
    for (ForcePlate& forcePlate : init->forcePlateTrials[trial])
    {
      if (forcePlate.corners.size() == 0)
      {
        needDefaultForcePlate = true;
        break;
      }
    }
    // 3.2. If we need the default force plate, now we want to go create a
    // rectangle to hold all the GRF data.
    std::vector<Eigen::Vector3s> defaultCorners;
    if (needDefaultForcePlate)
    {
      s_t minX = std::numeric_limits<s_t>::infinity();
      s_t maxX = -std::numeric_limits<s_t>::infinity();
      s_t minZ = std::numeric_limits<s_t>::infinity();
      s_t maxZ = -std::numeric_limits<s_t>::infinity();
      for (int t = 0; t < init->poseTrials[trial].cols(); t++)
      {
        for (ForcePlate& forcePlate : init->forcePlateTrials[trial])
        {
          if (forcePlate.centersOfPressure[t](0) < minX)
          {
            minX = forcePlate.centersOfPressure[t](0);
          }
          if (forcePlate.centersOfPressure[t](0) > maxX)
          {
            maxX = forcePlate.centersOfPressure[t](0);
          }
          if (forcePlate.centersOfPressure[t](2) < minZ)
          {
            minZ = forcePlate.centersOfPressure[t](2);
          }
          if (forcePlate.centersOfPressure[t](2) > maxZ)
          {
            maxZ = forcePlate.centersOfPressure[t](2);
          }
        }
      }

      // Add 10cm to each side, to be very conservative
      s_t padding = 0.10;
      minX -= padding;
      maxX += padding;
      minZ -= padding;
      maxZ += padding;

      defaultCorners.push_back(Eigen::Vector3s(minX, groundHeight, minZ));
      defaultCorners.push_back(Eigen::Vector3s(minX, groundHeight, maxZ));
      defaultCorners.push_back(Eigen::Vector3s(maxX, groundHeight, maxZ));
      defaultCorners.push_back(Eigen::Vector3s(maxX, groundHeight, minZ));
    }
    init->defaultForcePlateCorners.push_back(defaultCorners);

    // 4. Determine foot-ground contact at each trial, and figure out which
    // timesteps we think we're receiving force that isn't measured by a force
    // plate.

    std::vector<std::vector<Eigen::Vector3s>> sortedForcePlateCorners;
    for (ForcePlate& plate : init->forcePlateTrials[trial])
    {
      if (plate.corners.size() > 0)
      {
        sortedForcePlateCorners.push_back(plate.corners);
        math::prepareConvex2DShape(
            sortedForcePlateCorners[sortedForcePlateCorners.size() - 1],
            plate.corners[0],
            Eigen::Vector3s::UnitX(),
            Eigen::Vector3s::UnitZ());
      }
    }
    if (init->defaultForcePlateCorners[trial].size() > 0)
    {
      math::prepareConvex2DShape(
          init->defaultForcePlateCorners[trial],
          init->defaultForcePlateCorners[trial][0],
          Eigen::Vector3s::UnitX(),
          Eigen::Vector3s::UnitZ());
    }

    std::vector<std::vector<bool>> trialForceActive;
    std::vector<std::vector<bool>> trialSphereInContact;
    std::vector<std::vector<bool>> trialOffForcePlate;
    std::vector<bool> trialAnyOffForcePlate;
    for (int t = 0; t < init->poseTrials[trial].cols(); t++)
    {
      mSkeleton->setPositions(init->poseTrials[trial].col(t));

      std::vector<bool> forceActive;
      std::vector<bool> sphereInContact;
      std::vector<bool> offForcePlate;
      bool anyContactIsSus = false;
      for (int b = 0; b < init->grfBodyNodes.size(); b++)
      {
        // 4.1. Check the GRF to see if we are measured as being in contact
        bool footActive
            = (init->grfTrials[trial].col(t).segment<6>(b * 6).squaredNorm()
               > 1e-3);
        forceActive.push_back(footActive);

        // 4.2. Check all the contact bodies assigned to this GRF node to guess
        // if we think we might be in contact here
        bool inContact = false;
        for (int c = 0; c < init->contactBodies[b].size(); c++)
        {
          auto* body = init->contactBodies[b][c];
          Eigen::Vector3s worldPos = body->getWorldTransform().translation();
          s_t dist = worldPos(1) - groundHeight;
          if (dist < grfContactSphereSizes[b][c])
          {
            inContact = true;
          }
        }
        sphereInContact.push_back(inContact);

        // 4.3. If we think from the collider heuristic that we might be in
        // contact, but then we don't actually have any GRF, we need to be
        // suspicious that this might be a contact outside a force plate, which
        // would mean that our inverse dynamics should ignore these frames.
        bool contactIsSus = false;
        if (inContact && !footActive)
        {
          // 4.3.1. Iterate over each contact body, and check if its inside any
          // of the force plates.
          bool anyInPlate = false;
          for (int c = 0; c < init->contactBodies[b].size(); c++)
          {
            auto* body = init->contactBodies[b][c];
            Eigen::Vector3s worldPos = body->getWorldTransform().translation();
            for (ForcePlate& plate : init->forcePlateTrials[trial])
            {
              if (plate.corners.size() > 0)
              {
                if (math::convex2DShapeContains(
                        worldPos,
                        plate.corners,
                        plate.worldOrigin,
                        Eigen::Vector3s::UnitX(),
                        Eigen::Vector3s::UnitZ()))
                {
                  anyInPlate = true;
                  break;
                }
              }
            }
            if (init->defaultForcePlateCorners[trial].size() > 0)
            {
              if (math::convex2DShapeContains(
                      worldPos,
                      init->defaultForcePlateCorners[trial],
                      init->defaultForcePlateCorners[trial][0],
                      Eigen::Vector3s::UnitX(),
                      Eigen::Vector3s::UnitZ()))
              {
                anyInPlate = true;
              }
            }
            if (anyInPlate)
            {
              break;
            }
          }

          // 4.3.2. If we're NOT over a plate, then register this frame as
          // suspicious
          if (!anyInPlate)
          {
            contactIsSus = true;
            anyContactIsSus = true;
          }
        }

        offForcePlate.push_back(contactIsSus);
      }

      trialForceActive.push_back(forceActive);
      trialSphereInContact.push_back(sphereInContact);
      trialOffForcePlate.push_back(offForcePlate);
      trialAnyOffForcePlate.push_back(anyContactIsSus);
    }

    init->grfBodyForceActive.push_back(trialForceActive);
    init->grfBodySphereInContact.push_back(trialSphereInContact);
    init->grfBodyOffForcePlate.push_back(trialOffForcePlate);
    init->probablyMissingGRF.push_back(trialAnyOffForcePlate);
  }
}

//==============================================================================
// 0. Smooth the accelerations.
void DynamicsFitter::smoothAccelerations(
    std::shared_ptr<DynamicsInitialization> init)
{
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    // 0.05
    AccelerationSmoother smoother(init->poseTrials[trial].cols(), 1, 0.05);
    init->poseTrials[trial] = smoother.smooth(init->poseTrials[trial]);
  }
}

//==============================================================================
// 1. Adjust the total mass of the body, and change the initial positions and
// velocities of the body to achieve a least-squares closest COM trajectory to
// the current kinematic fit.
void DynamicsFitter::zeroLinearResidualsOnCOMTrajectory(
    std::shared_ptr<DynamicsInitialization> init)
{
  if (mSkeleton->getRootJoint()->getType()
          != dynamics::EulerFreeJoint::getStaticType()
      && mSkeleton->getRootJoint()->getType()
             != dynamics::FreeJoint::getStaticType())
  {
    std::cout
        << "ERROR: Cannot call zeroResidualsOnCOMTrajectory() on a skeleton "
           "that does "
           "not have a root joint that is either an EulerFreeJoint or "
           "FreeJoint - instead got "
        << mSkeleton->getRootJoint()->getType()
        << ". Ignoring call, and leaving DynamicsInitialization unchanged."
        << std::endl;
    return;
  }
  Eigen::Vector3s gravity = Eigen::Vector3s(0, -9.81, 0);
  s_t regularizeUnobservedTimesteps = 1.0;

  const s_t originalMass = mSkeleton->getMass();

  std::vector<Eigen::MatrixXs> trialLinearMaps;
  std::vector<Eigen::VectorXs> trialOriginalPositions;
  std::vector<Eigen::VectorXs> trialOriginalGravityOffsets;
  std::vector<std::vector<Eigen::Vector3s>> trialOriginalCOMs;
  int totalTimesteps = 0;
  int totalColsWithoutMass = 0;
  int totalRows = 0;

  for (int i = 0; i < init->poseTrials.size(); i++)
  {
    int numMissingSteps = 0;
    std::vector<bool> trialProbablyMissing;
    std::vector<int> missingStepIndices;
    if (init->probablyMissingGRF.size() > i)
    {
      trialProbablyMissing = init->probablyMissingGRF[i];
      for (int t = 0; t < trialProbablyMissing.size(); t++)
      {
        if (trialProbablyMissing[t])
        {
          numMissingSteps++;
          missingStepIndices.push_back(t);
        }
      }
    }

    std::vector<Eigen::Vector3s> grfs = measuredGRFForces(init, i);
    std::vector<Eigen::Vector3s> originalCOMs = comPositions(init, i);
    const s_t dt = init->trialTimesteps[i];
    const int numTimesteps = originalCOMs.size();

    totalTimesteps += numTimesteps;

    // This has one col each for start COM positions, start COM velocities, and
    // total mass. It outputs the physically consistent X, Y, Z coordinates of
    // COM over time, given those inputs.
    Eigen::MatrixXs trialLinearMapToPositions = Eigen::MatrixXs::Zero(
        numTimesteps * 3 + (numMissingSteps * 3), 6 + numMissingSteps * 3 + 1);
    totalColsWithoutMass += trialLinearMapToPositions.cols() - 1;
    totalRows += trialLinearMapToPositions.rows();

    // This is a vector with the original positions of our COM over time, which
    // we will use to find good values for the position, velocity, and mass of
    // our skeleton.
    Eigen::VectorXs originalCOMPositions
        = Eigen::VectorXs::Zero(numTimesteps * 3);

    Eigen::VectorXs comGravityOffset = Eigen::VectorXs::Zero(numTimesteps * 3);

    const int startPosXCol = 0;
    const int startPosYCol = 1;
    const int startPosZCol = 2;
    const int startVelXCol = 3;
    const int startVelYCol = 4;
    const int startVelZCol = 5;
    const int massCol = 6 + numMissingSteps * 3;

    Eigen::Vector3s forceVelocity = Eigen::Vector3s::Zero();
    Eigen::Vector3s forceOffset = Eigen::Vector3s::Zero();
    Eigen::Vector3s gravityVelocity = Eigen::Vector3s::Zero();
    Eigen::Vector3s gravityOffset = Eigen::Vector3s::Zero();

    for (int t = 0; t < numTimesteps; t++)
    {
      const int xRow = (t * 3);
      const int yRow = (t * 3) + 1;
      const int zRow = (t * 3) + 2;

      trialLinearMapToPositions(xRow, startPosXCol) = 1;
      trialLinearMapToPositions(yRow, startPosYCol) = 1;
      trialLinearMapToPositions(zRow, startPosZCol) = 1;

      trialLinearMapToPositions(xRow, startVelXCol) = t * dt;
      trialLinearMapToPositions(yRow, startVelYCol) = t * dt;
      trialLinearMapToPositions(zRow, startVelZCol) = t * dt;

      trialLinearMapToPositions(xRow, massCol) = forceOffset(0);
      trialLinearMapToPositions(yRow, massCol) = forceOffset(1);
      trialLinearMapToPositions(zRow, massCol) = forceOffset(2);

      // Allow the missing steps to generate arbitrary delta V on each step
      for (int j = 0; j < missingStepIndices.size(); j++)
      {
        int missingIndex = missingStepIndices[j];
        if (missingIndex >= t)
          break;
        int timestepsSinceDv = t - missingIndex;

        int missingXCol = 6 + j * 3;
        int missingYCol = 6 + j * 3 + 1;
        int missingZCol = 6 + j * 3 + 2;

        trialLinearMapToPositions(xRow, missingXCol) = dt * timestepsSinceDv;
        trialLinearMapToPositions(yRow, missingYCol) = dt * timestepsSinceDv;
        trialLinearMapToPositions(zRow, missingZCol) = dt * timestepsSinceDv;
      }

      comGravityOffset(xRow) = gravityOffset(0);
      comGravityOffset(yRow) = gravityOffset(1);
      comGravityOffset(zRow) = gravityOffset(2);

      forceVelocity += grfs[t] * dt;
      forceOffset += forceVelocity * dt;

      gravityVelocity += gravity * dt;
      gravityOffset += gravityVelocity * dt;

      originalCOMPositions(xRow) = originalCOMs[t](0);
      originalCOMPositions(yRow) = originalCOMs[t](1);
      originalCOMPositions(zRow) = originalCOMs[t](2);
    }

    // Output the delta V for missing steps to the output, so we can regularize
    // it
    for (int j = 0; j < missingStepIndices.size(); j++)
    {
      int missingXCol = 6 + j * 3;
      int missingYCol = 6 + j * 3 + 1;
      int missingZCol = 6 + j * 3 + 2;

      int missingXRow = (numTimesteps * 3) + j * 3;
      int missingYRow = (numTimesteps * 3) + j * 3 + 1;
      int missingZRow = (numTimesteps * 3) + j * 3 + 2;

      // Nothing else should've written to these rows at this point
      if (!trialLinearMapToPositions.row(missingXRow).isZero())
      {
        std::cout << "missing X Row (" << missingXRow << ") for missing index "
                  << j << " is not empty!" << std::endl
                  << trialLinearMapToPositions.row(missingXRow).transpose()
                  << std::endl;
      }
      assert(trialLinearMapToPositions.row(missingXRow).isZero());
      assert(trialLinearMapToPositions.row(missingYRow).isZero());
      assert(trialLinearMapToPositions.row(missingZRow).isZero());

      trialLinearMapToPositions(missingXRow, missingXCol)
          = regularizeUnobservedTimesteps;
      trialLinearMapToPositions(missingYRow, missingYCol)
          = regularizeUnobservedTimesteps;
      trialLinearMapToPositions(missingZRow, missingZCol)
          = regularizeUnobservedTimesteps;
    }

#ifndef NDEBUG
    Eigen::VectorXs testConfig
        = Eigen::VectorXs::Random(trialLinearMapToPositions.cols());
    testConfig.head<3>() = originalCOMs[0];
    testConfig.segment<3>(3) = (originalCOMs[1] - originalCOMs[0]);
    testConfig(testConfig.size() - 1) = 1.0 / originalMass;

    Eigen::VectorXs recoveredComPoses = trialLinearMapToPositions * testConfig;
    recoveredComPoses.segment(0, comGravityOffset.size()) += comGravityOffset;

    for (int j = 0; j < missingStepIndices.size(); j++)
    {
      int missingXCol = 6 + j * 3;
      int missingYCol = 6 + j * 3 + 1;
      int missingZCol = 6 + j * 3 + 2;

      int missingXRow = (numTimesteps * 3) + j * 3;
      int missingYRow = (numTimesteps * 3) + j * 3 + 1;
      int missingZRow = (numTimesteps * 3) + j * 3 + 2;

      if (recoveredComPoses.size() <= missingZRow)
      {
        std::cout << "Our recovered poses data doesn't seem long enough to "
                     "contain missing step delta V ["
                  << j << " / " << missingStepIndices.size() << "]. Needs "
                  << missingZRow << ", but is only len "
                  << recoveredComPoses.size() << std::endl;
      }
      assert(recoveredComPoses.size() > missingZRow);
      if (testConfig.size() <= missingZCol)
      {
        std::cout << "Our test configuration doesn't seem long enough to "
                     "contain missing step delta V ["
                  << j << " / " << missingStepIndices.size() << "]. Needs "
                  << missingZCol << ", but is only len " << testConfig.size()
                  << std::endl;
      }
      assert(testConfig.size() > missingZCol);

      if (abs(recoveredComPoses(missingXRow)
              - testConfig(missingXCol) * regularizeUnobservedTimesteps)
          > 1e-12)
      {
        std::cout << "Failed to recoverd deltaV X for missing timestep[" << j
                  << "] = " << missingStepIndices[j] << ", expected "
                  << testConfig(missingXCol) * regularizeUnobservedTimesteps
                  << " but got " << recoveredComPoses(missingXRow) << std::endl;
      }
      if (abs(recoveredComPoses(missingYRow)
              - testConfig(missingYCol) * regularizeUnobservedTimesteps)
          > 1e-12)
      {
        std::cout << "Failed to recoverd deltaV Y for missing timestep[" << j
                  << "] = " << missingStepIndices[j] << ", expected "
                  << testConfig(missingYCol) * regularizeUnobservedTimesteps
                  << " but got " << recoveredComPoses(missingYRow) << std::endl;
      }
      if (abs(recoveredComPoses(missingZRow)
              - testConfig(missingZCol) * regularizeUnobservedTimesteps)
          > 1e-12)
      {
        std::cout << "Failed to recoverd deltaV Z for missing timestep[" << j
                  << "] = " << missingStepIndices[j] << ", expected "
                  << testConfig(missingZCol) * regularizeUnobservedTimesteps
                  << " but got " << recoveredComPoses(missingZRow) << std::endl;
      }
    }

    for (int t = 1; t < numTimesteps - 1; t++)
    {
      if (init->probablyMissingGRF.size() > i && init->probablyMissingGRF[i][t])
        continue;
      Eigen::Vector3s acc = (recoveredComPoses.segment<3>((t + 1) * 3)
                             - 2 * recoveredComPoses.segment<3>(t * 3)
                             + recoveredComPoses.segment<3>((t - 1) * 3))
                            / (dt * dt);
      Eigen::Vector3s impliedForce = acc * originalMass;
      Eigen::Vector3s expectedForce = (gravity * originalMass) + grfs[t];
      Eigen::Vector3s diff = impliedForce - expectedForce;
      if (diff.norm() > 1e-7)
      {
        std::cout << "Bug detected in zeroResidualsOnCOMTrajectory() linear "
                     "formulation!"
                  << std::endl;
        Eigen::Matrix3s comp;
        comp.col(0) = impliedForce;
        comp.col(1) = expectedForce;
        comp.col(2) = diff;
        std::cout << "t=" << t << ": implied - expected - diff" << std::endl
                  << comp << std::endl;
      }
    }
#endif

    trialLinearMaps.push_back(trialLinearMapToPositions);
    trialOriginalPositions.push_back(originalCOMPositions);
    trialOriginalGravityOffsets.push_back(comGravityOffset);
    trialOriginalCOMs.push_back(originalCOMs);
  }
  int numTrials = trialLinearMaps.size();

  // Build one big unified matrix, so we can hold mass fixed across all the
  // trials. This keeps all the starting (position,velocity) pairs for each
  // trial separate, but collapses each trial's mass quantity into a single
  // column, so that we can solve them together for a single unified skeleton
  // mass that is least-squares best across all the trials at once.

  Eigen::MatrixXs unifiedLinearMap
      = Eigen::MatrixXs::Zero(totalRows, totalColsWithoutMass + 1);
  Eigen::VectorXs unifiedPositions = Eigen::VectorXs::Zero(totalRows);
  Eigen::VectorXs unifiedGravityOffset = Eigen::VectorXs::Zero(totalRows);

  int rowCursor = 0;
  int colCursor = 0;
  int massCol = unifiedLinearMap.cols() - 1;
  for (int trial = 0; trial < numTrials; trial++)
  {
    int trialTimesteps = trialOriginalPositions[trial].size();
    int trialRows = trialLinearMaps[trial].rows();
    int trialCols = trialLinearMaps[trial].cols();
    // (position,velocity) get their own unique block for each trial
    unifiedLinearMap.block(rowCursor, colCursor, trialRows, trialCols - 1)
        = trialLinearMaps[trial].block(0, 0, trialRows, trialCols - 1);
    // mass all goes into the same column for every trial
    unifiedLinearMap.block(rowCursor, massCol, trialRows, 1)
        = trialLinearMaps[trial].block(0, trialCols - 1, trialRows, 1);
    // original positions get concatenated together
    unifiedPositions.segment(rowCursor, trialTimesteps)
        = trialOriginalPositions[trial];
    unifiedGravityOffset.segment(rowCursor, trialTimesteps)
        = trialOriginalGravityOffsets[trial];
    rowCursor += trialRows;
    colCursor += trialCols;
  }

  // Now that we've formulated our problem, we can attempt to solve:
  Eigen::VectorXs tentativeResult
      = unifiedLinearMap.completeOrthogonalDecomposition().solve(
          unifiedPositions - unifiedGravityOffset);
  // The linear map is in terms of "inverse mass", so we need to invert to
  // recover original mass.
  s_t tentativeMass = 1.0 / tentativeResult(massCol);

  Eigen::VectorXs adjustedComPositions;

  // If our mass has dropped below an acceptable threshold in this solve, then
  // we need to run again but hold mass fixed.
  if (tentativeMass < 0.1 * originalMass && false)
  {
    Eigen::VectorXs massOffset
        = unifiedLinearMap.col(massCol) * (1.0 / originalMass);

    // Re-run the solve, while holding the mass fixed at `originalMass`
    Eigen::VectorXs massFixedResult
        = unifiedLinearMap
              .block(0, 0, totalTimesteps, unifiedLinearMap.cols() - 1)
              .completeOrthogonalDecomposition()
              .solve(unifiedPositions - massOffset - unifiedGravityOffset);
    tentativeResult.segment(0, massFixedResult.size()) = massFixedResult;

    // Calculate the trajectory of our COM over time
    adjustedComPositions
        = unifiedLinearMap.block(
              0, 0, totalTimesteps, unifiedLinearMap.cols() - 1)
              * massFixedResult
          + massOffset + unifiedGravityOffset;
  }
  else
  {
    // Now we want to scale every mass in the skeleton to get to `tentativeMass`
    // total
    s_t scalePercentage = tentativeMass / originalMass;
    init->bodyMasses *= scalePercentage;

    // Calculate the trajectory of our COM over time
    adjustedComPositions
        = unifiedLinearMap * tentativeResult + unifiedGravityOffset;
  }

  // Last, we need to decode the COM trajectory, and adjust our root
  // translations to hit that trajectory
  const int translationDofsStart
      = mSkeleton->getRootJoint()->getDof(3)->getIndexInSkeleton();
  int timeCursor = 0;
  for (int trial = 0; trial < numTrials; trial++)
  {
    s_t totalChange = 0.0;
    std::vector<Eigen::Vector3s> originalCOMs = trialOriginalCOMs[trial];

    for (int t = 0; t < originalCOMs.size(); t++)
    {
      Eigen::Vector3s change
          = adjustedComPositions.segment<3>(timeCursor * 3) - originalCOMs[t];
      timeCursor++;
      init->poseTrials[trial].block<3, 1>(translationDofsStart, t) += change;
      totalChange += change.norm();
    }
    std::cout << "Trial " << trial << " moved COM by an average of "
              << (totalChange / originalCOMs.size())
              << "m to achieve linear physical consistency." << std::endl;
  }
  assert(timeCursor == totalTimesteps);
}

//==============================================================================
// 1. Scale the total mass of the body (keeping the ratios of body links
// constant) to get it as close as possible to GRF gravity forces.
void DynamicsFitter::scaleLinkMassesFromGravity(
    std::shared_ptr<DynamicsInitialization> init)
{
  s_t totalGRFs = 0.0;
  s_t totalAccs = 0.0;
  s_t gravity = mSkeleton->getGravity().norm();

  for (int i = 0; i < init->poseTrials.size(); i++)
  {
    std::vector<Eigen::Vector3s> grfs = measuredGRFForces(init, i);
    std::vector<Eigen::Vector3s> accs = comAccelerations(init, i);
    for (int t = 0; t < grfs.size(); t++)
    {
      if (i >= init->probablyMissingGRF.size()
          || t >= init->probablyMissingGRF[i].size()
          || !init->probablyMissingGRF[i][t])
      {
        totalGRFs += grfs[t](1);
        totalAccs += gravity; // accs[t](1) + gravity;
      }
    }
  }

  std::cout << "Total ACCs: " << totalAccs << std::endl;
  std::cout << "Total mass: " << init->bodyMasses.sum() << std::endl;
  std::cout << "(Total ACCs) * (Total mass): "
            << totalAccs * init->bodyMasses.sum() << std::endl;
  std::cout << "Total GRFs: " << totalGRFs << std::endl;

  if (totalGRFs > 0)
  {
    s_t impliedTotalMass = totalGRFs / totalAccs;
    std::cout << "Implied total mass: " << impliedTotalMass << std::endl;
    s_t ratio = impliedTotalMass / init->bodyMasses.sum();
    init->bodyMasses *= ratio;
    std::cout << "Adjusted total mass to match GRFs: " << init->bodyMasses.sum()
              << std::endl;
  }
  else
  {
    std::cout << "NO GRF DATA in this slice!" << std::endl;
  }
}

//==============================================================================
// 2. Estimate just link masses, while holding the positions, COMs, and inertias
// constant
void DynamicsFitter::estimateLinkMassesFromAcceleration(
    std::shared_ptr<DynamicsInitialization> init, s_t regularizationWeight)
{
  Eigen::VectorXs originalPose = mSkeleton->getPositions();

  int totalTimesteps = 0;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    if (init->poseTrials.at(trial).cols() > 0)
    {
      totalTimesteps += init->poseTrials.at(trial).cols() - 2;
    }
  }

  // constants
  Eigen::Vector3s gravityVector = Eigen::Vector3s(0, -9.81, 0);
  (void)gravityVector;

  // 1. Set up the problem matrices
  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(
      (totalTimesteps * 3) + mSkeleton->getNumBodyNodes(),
      mSkeleton->getNumBodyNodes());
  Eigen::VectorXs g = Eigen::VectorXs::Zero(
      (totalTimesteps * 3) + mSkeleton->getNumBodyNodes());

#ifndef NDEBUG
  Eigen::MatrixXs A_no_gravity = Eigen::MatrixXs::Zero(
      (totalTimesteps * 3) + mSkeleton->getNumBodyNodes(),
      mSkeleton->getNumBodyNodes());
#endif

  int cursor = 0;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    Eigen::MatrixXs& poses = init->poseTrials[trial];
    if (poses.cols() <= 2)
      continue;

    // 1.1. Initialize empty position matrices for each body node

    std::map<std::string, Eigen::Matrix<s_t, 3, Eigen::Dynamic>>
        bodyPosesOverTime;
    for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
    {
      bodyPosesOverTime[mSkeleton->getBodyNode(i)->getName()]
          = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(3, poses.cols());
    }

    // 1.2. Fill position matrices for each body node

    for (int t = 0; t < poses.cols(); t++)
    {
      mSkeleton->setPositions(poses.col(t));
      for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
      {
        bodyPosesOverTime.at(mSkeleton->getBodyNode(i)->getName()).col(t)
            = mSkeleton->getBodyNode(i)->getCOM();
      }
    }

    // 1.3. Finite difference out the accelerations for each body

    s_t dt = init->trialTimesteps[trial];
    for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
    {
      auto* body = mSkeleton->getBodyNode(i);
      for (int t = 1; t < bodyPosesOverTime.at(body->getName()).cols() - 1; t++)
      {
        Eigen::Vector3s v1
            = (bodyPosesOverTime.at(body->getName()).col(t)
               - bodyPosesOverTime.at(body->getName()).col(t - 1))
              / dt;
        Eigen::Vector3s v2 = (bodyPosesOverTime.at(body->getName()).col(t + 1)
                              - bodyPosesOverTime.at(body->getName()).col(t))
                             / dt;
        Eigen::Vector3s acc = (v2 - v1) / dt;
        int timestep = cursor + t;
        A.block<3, 1>(timestep * 3, i) = acc - gravityVector;
#ifndef NDEBUG
        A_no_gravity.block<3, 1>(timestep * 3, i) = acc;
#endif
      }
    }

    // 1.4. Sum up the gravitational forces
    for (int t = 1; t < poses.cols() - 1; t++)
    {
      int timestep = cursor + t;
      for (int i = 0; i < init->forcePlateTrials[trial].size(); i++)
      {
        g.segment<3>(timestep * 3)
            += init->forcePlateTrials[trial][i].forces[t];
      }
    }

    cursor += poses.cols() - 2;
  }

  // 1.5. Add a regularization block
  int m = mSkeleton->getNumBodyNodes();
  A.block(totalTimesteps * 3, 0, m, m)
      = regularizationWeight * Eigen::MatrixXs::Identity(m, m);
  g.segment(totalTimesteps * 3, m) = regularizationWeight * init->bodyMasses;

  // 2. Now we'll go through and do some checks, if we're in debug mode
#ifndef NDEBUG
  // 2.1. Check gravity-less
  Eigen::VectorXs recoveredImpliedForces_noGravity
      = A_no_gravity * init->bodyMasses;
  std::vector<Eigen::Vector3s> comForces_noGravity;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    std::vector<Eigen::Vector3s> trialComForces_noGravity
        = impliedCOMForces(init, trial);
    comForces_noGravity.insert(
        comForces_noGravity.end(),
        trialComForces_noGravity.begin(),
        trialComForces_noGravity.end());
  }
  for (int i = 0; i < comForces_noGravity.size(); i++)
  {
    Eigen::Vector3s recovered
        = recoveredImpliedForces_noGravity.segment<3>(i * 3);
    s_t dist = (recovered - comForces_noGravity[i]).norm();
    if (dist > 1e-5)
    {
      std::cout << "Error in recovered force (no gravity) at timestep " << i
                << std::endl;
      std::cout << "Recovered from matrix form: " << recovered << std::endl;
      std::cout << "Explicit calculation: " << comForces_noGravity[i]
                << std::endl;
      std::cout << "Diff: " << dist << std::endl;
      assert(false);
    }
  }

  // 2.2. Check with gravity
  Eigen::VectorXs recoveredImpliedForces = A * init->bodyMasses;
  std::vector<Eigen::Vector3s> comForces;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    std::vector<Eigen::Vector3s> trialComForces
        = impliedCOMForces(init, trial, mSkeleton->getGravity());
    comForces.insert(
        comForces.end(), trialComForces.begin(), trialComForces.end());
  }
  for (int i = 0; i < comForces.size(); i++)
  {
    Eigen::Vector3s recovered = recoveredImpliedForces.segment<3>(i * 3);
    s_t dist = (recovered - comForces[i]).norm();
    if (dist > 1e-5)
    {
      std::cout << "Error in recovered force (with gravity) at timestep " << i
                << std::endl;
      std::cout << "Recovered from matrix form: " << recovered << std::endl;
      std::cout << "Explicit calculation: " << comForces[i] << std::endl;
      std::cout << "Diff: " << dist << std::endl;
      assert(false);
    }
  }

  // 2.3. Check GRF agreement
  std::vector<Eigen::Vector3s> grfForces;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    std::vector<Eigen::Vector3s> trialGrfForces
        = measuredGRFForces(init, trial);
    grfForces.insert(
        grfForces.end(), trialGrfForces.begin(), trialGrfForces.end());
  }
  for (int i = 0; i < grfForces.size(); i++)
  {
    Eigen::Vector3s recovered = g.segment<3>(i * 3);
    s_t dist = (recovered - grfForces[i]).norm();
    if (dist > 1e-5)
    {
      std::cout << "Error in GRF at timestep " << i << std::endl;
      std::cout << "Recovered from matrix form: " << recovered << std::endl;
      std::cout << "Explicit calculation: " << grfForces[i] << std::endl;
      std::cout << "Diff: " << dist << std::endl;
      assert(false);
    }
  }
#endif

  Eigen::MatrixXs debugMatrix
      = Eigen::MatrixXs::Zero(init->bodyMasses.size(), 3);
  debugMatrix.col(0) = init->bodyMasses;

  // Now that we've got the problem setup, we can factor and solve
  init->bodyMasses = A.completeOrthogonalDecomposition().solve(g);
  // TODO: solve this non-negatively, and closer to original values

  for (int i = 0; i < init->bodyMasses.size(); i++)
  {
    if (init->bodyMasses(i) < 0.01)
    {
      init->bodyMasses(i) = 0.01;
    }
  }

  debugMatrix.col(1) = init->bodyMasses;
  debugMatrix.col(2) = debugMatrix.col(1).cwiseQuotient(debugMatrix.col(0))
                       - Eigen::VectorXs::Ones(debugMatrix.rows());

  std::cout << "Original masses - New masses - Percent change: " << std::endl
            << debugMatrix << std::endl;

  mSkeleton->setPositions(originalPose);
}

//==============================================================================
// 3. Run larger optimization problems to minimize a weighted combination of
// residuals and marker RMSE, tweaking a controllable set of variables. This
// includes the velocity and acceleration as explicit decision variables,
// constrained by linear constraint equations. That means it needs to be
// solved with IPOPT, using the interior point method.
//
// WARNING: DOES NOT PERFORM WELL WITH WARM STARTS! Becaus it uses the
// interior point method, this doesn't warm start well. See
// runImplicitVelAccOptimization() instead.
void DynamicsFitter::runIPOPTOptimization(
    std::shared_ptr<DynamicsInitialization> init,
    DynamicsFitProblemConfig config)
{
  // Before using Eigen in a multi-threaded environment, we need to explicitly
  // call this (at least prior to Eigen 3.3)
  Eigen::initParallel();

  // Create an instance of the IpoptApplication
  //
  // We are using the factory, since this allows us to compile this
  // example with an Ipopt Windows DLL
  SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();

  // Change some options
  app->Options()->SetNumericValue("tol", static_cast<double>(mTolerance));
  app->Options()->SetStringValue(
      "linear_solver",
      "mumps"); // ma27, ma55, ma77, ma86, ma97, parsido, wsmp, mumps, custom

  app->Options()->SetStringValue(
      "hessian_approximation", "limited-memory"); // limited-memory, exacty

  /*
  app->Options()->SetStringValue(
      "scaling_method", "none"); // none, gradient-based
  */

  app->Options()->SetIntegerValue("max_iter", mIterationLimit);

  // Disable LBFGS history
  app->Options()->SetIntegerValue(
      "limited_memory_max_history", mLBFGSHistoryLength);

  // Just for debugging
  if (mCheckDerivatives)
  {
    app->Options()->SetStringValue("check_derivatives_for_naninf", "yes");
    app->Options()->SetStringValue("derivative_test", "first-order");
    app->Options()->SetNumericValue("derivative_test_perturbation", 1e-6);
  }

  if (mPrintFrequency > 0)
  {
    app->Options()->SetIntegerValue("print_frequency_iter", mPrintFrequency);
  }
  else
  {
    app->Options()->SetIntegerValue(
        "print_frequency_iter", std::numeric_limits<int>::infinity());
  }
  if (mSilenceOutput)
  {
    app->Options()->SetIntegerValue("print_level", 0);
  }
  if (mDisableLinesearch)
  {
    app->Options()->SetIntegerValue("max_soc", 0);
    app->Options()->SetStringValue("accept_every_trial_step", "yes");
  }
  app->Options()->SetIntegerValue("watchdog_shortened_iter_trigger", 0);

  std::shared_ptr<BilevelFitResult> result
      = std::make_shared<BilevelFitResult>();

  // Initialize the IpoptApplication and process the options
  Ipopt::ApplicationReturnStatus status;
  status = app->Initialize();
  if (status != Solve_Succeeded)
  {
    std::cout << std::endl
              << std::endl
              << "*** Error during initialization!" << std::endl;
    return;
  }

  // This will automatically free the problem object when finished,
  // through `problemPtr`. `problem` NEEDS TO BE ON THE HEAP or it will
  // crash. If you try to leave `problem` on the stack, you'll get invalid
  // free exceptions when IPOpt attempts to free it.
  DynamicsFitProblem* problem = new DynamicsFitProblem(
      init, mSkeleton, mTrackingMarkers, mFootNodes, config);
  if (problem->getProblemSize() == 0)
  {
    delete problem;
    std::cout << "WARNING: Optimization problem had no decision variables! "
                 "Please enable some variables, for example with "
                 "config.setIncludePoses(true)"
              << std::endl;
    return;
  }

  /*
  Eigen::VectorXs x = problem->flatten();
  Eigen::VectorXs fd = problem->finiteDifferenceGradient(x);
  Eigen::VectorXs analytical = problem->computeGradient(x);
  if (problem->debugErrors(fd, analytical, 1e-8))
  {
    std::cout << "Detected gradient errors in DynamicsFitter!! Quitting."
              << std::endl;
    exit(1);
  }
  */
  std::cout << "BEGINNING OPTIMIZATION WITH LOSS: "
            << problem->computeLoss(problem->flatten()) << std::endl;

  SmartPtr<DynamicsFitProblem> problemPtr(problem);

  // This will automatically write results back to `init` on success.
  status = app->OptimizeTNLP(problemPtr);

  if (status == Solve_Succeeded)
  {
    // Retrieve some statistics about the solve
    Index iter_count = app->Statistics()->IterationCount();
    std::cout << std::endl
              << std::endl
              << "*** The problem solved in " << iter_count << " iterations!"
              << std::endl;

    Number final_obj = app->Statistics()->FinalObjective();
    std::cout << std::endl
              << std::endl
              << "*** The final value of the objective function is "
              << final_obj << '.' << std::endl;
  }
}

//==============================================================================
// 4. This runs the same optimization problem as
// runExplicitVelAccOptimization(), but holds velocity and acc as implicit
// functions of the position values, and removes any constraints. That means
// we can optimize this using simple gradient descent with line search, and
// can warm start.
void DynamicsFitter::runSGDOptimization(
    std::shared_ptr<DynamicsInitialization> init,
    DynamicsFitProblemConfig config)
{
  config.setVelAccImplicit(true);

  // Create a problem object on the stack
  DynamicsFitProblem problem(
      init, mSkeleton, mTrackingMarkers, mFootNodes, config);
  if (problem.getProblemSize() == 0)
  {
    std::cout << "WARNING: Optimization problem had no decision variables! "
                 "Please enable some variables, for example with "
                 "config.setIncludePoses(true)"
              << std::endl;
    return;
  }

  // Guarantee that even if we aren't including the poses in our optimization,
  // the velocities and accelerations are still exactly consistent with the pose
  // data.
  problem.mConfig.setIncludePoses(true);
  problem.unflatten(problem.flatten());
  problem.mConfig.setIncludePoses(config.mIncludePoses);

  Eigen::VectorXs x = problem.flatten();
  s_t lastLoss = problem.computeLoss(x);

  Eigen::VectorXs lowerBounds = problem.flattenLowerBound();
  Eigen::VectorXs upperBounds = problem.flattenUpperBound();

  s_t stepSize = 1e-7;
  for (int i = 0; i < mIterationLimit; i++)
  {
    std::cout << "Step " << i << ": " << lastLoss << std::endl;
    Eigen::VectorXs grad = problem.computeGradient(x);

    /*
    if (i % 10 == 0)
    {
      std::cout << "Checking FD gradient." << std::endl;
      Eigen::VectorXs fd = problem.finiteDifferenceGradient(x);
      std::cout << " > Analytical gradient norm = " << grad.norm() << std::endl;
      std::cout << " > FD gradient norm = " << fd.norm() << std::endl;
      Eigen::VectorXs diff = grad - fd;
      if (diff.norm() > 1e-7)
      {
        std::cout << "Gradient diff by " << diff.norm() << "!" << std::endl;
        exit(1);
      }
    }
    */

    bool firstTry = true;
    do
    {
      Eigen::VectorXs testX = x - grad * stepSize;
      for (int i = 0; i < testX.size(); i++)
      {
        if (testX(i) > upperBounds(i))
        {
          testX(i) = upperBounds(i);
        }
        if (testX(i) < lowerBounds(i))
        {
          testX(i) = lowerBounds(i);
        }
      }
      s_t testLoss = problem.computeLoss(testX, true);
      if (testLoss < lastLoss)
      {
        x = testX;
        lastLoss = testLoss;
        problem.intermediate_callback(
            Ipopt::AlgorithmMode::RegularMode,
            i,
            lastLoss,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            nullptr,
            nullptr);
        if (firstTry)
        {
          // If we hit a success on our first try, then grow the step size
          stepSize *= 2;
        }
        break;
      }
      else
      {
        std::cout << "    Step size " << stepSize
                  << " too large! Change in loss (" << testLoss << " - "
                  << lastLoss << ") = " << (testLoss - lastLoss) << " >= 0"
                  << std::endl;
      }
      firstTry = false;
      stepSize *= 0.5;
    } while (stepSize > 1e-12);
  }

  // Save the result back to the problem init
  problem.finalize_solution(
      Ipopt::SolverReturn::SUCCESS,
      mIterationLimit,
      nullptr,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr,
      0,
      nullptr,
      nullptr);
}

//==============================================================================
// 5. This attempts to perfect the physical consistency of the data
void DynamicsFitter::computePerfectGRFs(
    std::shared_ptr<DynamicsInitialization> init)
{
  // Create a problem object on the stack
  DynamicsFitProblem problem(
      init,
      mSkeleton,
      mTrackingMarkers,
      mFootNodes,
      DynamicsFitProblemConfig(mSkeleton));
  // Compute the perfect GRF data
  problem.computePerfectGRFs();
}

//==============================================================================
// This plays the simulation forward in Nimble, using the existing GRFs and
// torques, and checks that everything matches what we expect to see
bool DynamicsFitter::checkPhysicalConsistency(
    std::shared_ptr<DynamicsInitialization> init,
    s_t maxAcceptableErrors,
    int maxTimestepsToTest)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->setGravity(mSkeleton->getGravity());
  world->setParallelVelocityAndPositionUpdates(false);
  world->addSkeleton(mSkeleton);

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    s_t dt = init->trialTimesteps[trial];
    world->setTimeStep(dt);
    mSkeleton->setTimeStep(dt);

    // Initialize state

    mSkeleton->setPositions(init->poseTrials[trial].col(1));
    Eigen::VectorXs dq
        = (init->poseTrials[trial].col(1) - init->poseTrials[trial].col(0))
          / dt;
    mSkeleton->setVelocities(dq);

    int timestepsSinceReset = 0;

    for (int t = 1; t < init->poseTrials[trial].cols() - 1; t++)
    {

      // Check that everything is what we expect, and if not, error

      Eigen::VectorXs expectedPos = init->poseTrials[trial].col(t);
      Eigen::VectorXs expectedVel = (init->poseTrials[trial].col(t)
                                     - init->poseTrials[trial].col(t - 1))
                                    / dt;
      Eigen::VectorXs expectedAcc = (init->poseTrials[trial].col(t + 1)
                                     - 2 * init->poseTrials[trial].col(t)
                                     + init->poseTrials[trial].col(t - 1))
                                    / (dt * dt);

      if (init->probablyMissingGRF[trial][t])
      {
        continue;
      }
      if (init->probablyMissingGRF[trial][t - 1]
          || timestepsSinceReset > maxTimestepsToTest)
      {
        mSkeleton->setPositions(expectedPos);
        mSkeleton->setVelocities(expectedVel);
        timestepsSinceReset = 0;
      }
      timestepsSinceReset++;

      Eigen::VectorXs posError = expectedPos - mSkeleton->getPositions();
      if (posError.cwiseAbs().maxCoeff() > maxAcceptableErrors)
      {
        std::cout << "Error in position at time: " << t << std::endl;
        Eigen::MatrixXs compare
            = Eigen::MatrixXs::Zero(mSkeleton->getNumDofs(), 3);
        compare.col(0) = expectedPos;
        compare.col(1) = mSkeleton->getPositions();
        compare.col(2) = expectedPos - mSkeleton->getPositions();
        std::cout << "Expected - Actual - Diff: " << std::endl
                  << compare << std::endl;
        return false;
      }

      Eigen::VectorXs velError = expectedVel - mSkeleton->getVelocities();
      if (velError.cwiseAbs().maxCoeff() > maxAcceptableErrors)
      {
        std::cout << "Error in velocity at time: " << t << std::endl;
        Eigen::MatrixXs compare
            = Eigen::MatrixXs::Zero(mSkeleton->getNumDofs(), 3);
        compare.col(0) = expectedVel;
        compare.col(1) = mSkeleton->getVelocities();
        compare.col(2) = expectedVel - mSkeleton->getVelocities();
        std::cout << "Expected - Actual - Diff: " << std::endl
                  << compare << std::endl;
        return false;
      }

      // Set the forces

      Eigen::VectorXs tau = init->perfectTorques[trial].col(t);
      mSkeleton->setControlForces(tau);

      for (int i = 0; i < mFootNodes.size(); i++)
      {
        // Use the world wrench
        Eigen::Vector6s worldWrench
            = init->perfectGrfTrials[trial].col(t).segment<6>(i * 6);
        Eigen::Vector6s localWrench
            = math::dAdT(mFootNodes[i]->getWorldTransform(), worldWrench);
        mFootNodes[i]->setExtWrench(localWrench);

        // Get the COP, force, torque
        Eigen::Vector3s cop
            = init->perfectGrfAsCopTorqueForces[trial].block<3, 1>(i * 9, t);
        Eigen::Vector6s copWrench
            = init->perfectGrfAsCopTorqueForces[trial].block<6, 1>(
                i * 9 + 3, t);
        Eigen::Isometry3s copT = Eigen::Isometry3s::Identity();
        copT.translation() = cop;
        Eigen::Vector6s recoveredWorldWrench = math::dAdInvT(copT, copWrench);

        if ((worldWrench - recoveredWorldWrench).norm() > 1e-10)
        {
          std::cout << "Recovered CoP wrench != world wrench" << std::endl;
          std::cout << "Recovered CoP wrench:" << std::endl
                    << recoveredWorldWrench << std::endl;
          std::cout << "world wrench:" << std::endl << worldWrench << std::endl;
          std::cout << "Diff:" << std::endl
                    << recoveredWorldWrench - worldWrench << std::endl;
          return false;
        }
      }

      // Check accelerations

      mSkeleton->computeForwardDynamics();
      Eigen::VectorXs accError = expectedAcc - mSkeleton->getAccelerations();
      if (accError.cwiseAbs().maxCoeff() > (maxAcceptableErrors / dt))
      {
        std::cout << "Error in acceleration at time: " << t << std::endl;
        Eigen::MatrixXs compare
            = Eigen::MatrixXs::Zero(mSkeleton->getNumDofs(), 3);
        compare.col(0) = expectedAcc;
        compare.col(1) = mSkeleton->getAccelerations();
        compare.col(2) = expectedAcc - mSkeleton->getAccelerations();
        std::cout << "Expected - Actual - Diff: " << std::endl
                  << compare << std::endl;
        return false;
      }

      // Take a timestep

      world->step();
    }
  }

  std::cout << "Physical consistency verified!" << std::endl;

  return true;
}

//==============================================================================
void writeVectorToCSV(std::ofstream& csvFile, Eigen::VectorXs& vec)
{
  for (int i = 0; i < vec.size(); i++)
  {
    csvFile << "," << vec(i);
  }
}

//==============================================================================
// This writes a unified CSV with a ton of different columns in it, describing
// the selected trial
void DynamicsFitter::writeCSVData(
    std::string path, std::shared_ptr<DynamicsInitialization> init, int trial)
{
  // time, pos, vel, acc, [contact] * feet, [cop, wrench] * feet
  std::ofstream csvFile;
  csvFile.open(path);

  csvFile << "time";
  for (int i = 0; i < mSkeleton->getNumDofs(); i++)
  {
    csvFile << ",pos_" << mSkeleton->getDof(i)->getName();
  }
  for (int i = 0; i < mSkeleton->getNumDofs(); i++)
  {
    csvFile << ",vel_" << mSkeleton->getDof(i)->getName();
  }
  for (int i = 0; i < mSkeleton->getNumDofs(); i++)
  {
    csvFile << ",tau_" << mSkeleton->getDof(i)->getName();
  }
  for (int i = 0; i < mSkeleton->getNumDofs(); i++)
  {
    csvFile << ",acc_" << mSkeleton->getDof(i)->getName();
  }
  csvFile << ",missing_grf_data";
  for (int i = 0; i < mFootNodes.size(); i++)
  {
    csvFile << "," << mFootNodes[i]->getName() << "_probably_contacting_ground";
  }
  for (int i = 0; i < mFootNodes.size(); i++)
  {
    csvFile << "," << mFootNodes[i]->getName() << "_contact_cop_x";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_cop_y";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_cop_z";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_moment_x";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_moment_y";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_moment_z";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_force_x";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_force_y";
    csvFile << "," << mFootNodes[i]->getName() << "_contact_force_z";
  }

  /*
  Eigen::Vector3s cop
      = init->perfectGrfAsCopTorqueForces[trial].block<3, 1>(i * 9, t);
  Eigen::Vector6s copWrench
      = init->perfectGrfAsCopTorqueForces[trial].block<6, 1>(i * 9 + 3, t);
  */

  (void)init;
  (void)trial;

  s_t time = 0.0;
  s_t dt = init->trialTimesteps[trial];

  for (int t = 1; t < init->poseTrials[trial].cols() - 1; t++)
  {
    csvFile << std::endl;

    csvFile << time;

    Eigen::VectorXs q = init->poseTrials[trial].col(t);
    Eigen::VectorXs dq
        = (init->poseTrials[trial].col(t) - init->poseTrials[trial].col(t - 1))
          / dt;
    Eigen::VectorXs ddq = (init->poseTrials[trial].col(t + 1)
                           - 2 * init->poseTrials[trial].col(t)
                           + init->poseTrials[trial].col(t - 1))
                          / (dt * dt);
    Eigen::VectorXs tau = init->perfectTorques[trial].col(t);
    Eigen::VectorXs footContactData
        = init->perfectGrfAsCopTorqueForces[trial].col(t);

    writeVectorToCSV(csvFile, q);
    writeVectorToCSV(csvFile, dq);
    writeVectorToCSV(csvFile, tau);
    writeVectorToCSV(csvFile, ddq);
    csvFile << "," << init->probablyMissingGRF[trial][t];
    for (int i = 0; i < mFootNodes.size(); i++)
    {
      csvFile << "," << init->grfBodyForceActive[trial][t][i]
          || init->grfBodySphereInContact[trial][t][i];
    }
    writeVectorToCSV(csvFile, footContactData);

    time += dt;
  }

  csvFile.close();
}

//==============================================================================
// Get the average RMSE, in meters, of the markers
s_t DynamicsFitter::computeAverageMarkerRMSE(
    std::shared_ptr<DynamicsInitialization> init)
{
  Eigen::VectorXs originalPoses = mSkeleton->getPositions();
  Eigen::VectorXs originalScales = mSkeleton->getGroupScales();
  mSkeleton->setGroupScales(init->groupScales);

  s_t result = 0;
  int count = 0;
  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    for (int i = 0; i < init->poseTrials[trial].cols(); i++)
    {
      mSkeleton->setPositions(init->poseTrials[trial].col(i));
      auto simulatedMarkers
          = mSkeleton->getMarkerMapWorldPositions(init->updatedMarkerMap);
      auto markers = init->markerObservationTrials[trial][i];
      for (auto pair : simulatedMarkers)
      {
        if (markers.count(pair.first))
        {
          result += (markers.at(pair.first) - pair.second).norm();
          count++;
        }
      }
    }
  }
  std::cout << "Marker raw RMS: " << result << std::endl;
  std::cout << "Count: " << count << std::endl;

  result /= count;

  mSkeleton->setPositions(originalPoses);
  mSkeleton->setGroupScales(originalScales);

  return result;
}

//==============================================================================
// Get the average residual force, in newtons, and torque, in
// newton-meters
std::pair<s_t, s_t> DynamicsFitter::computeAverageResidualForce(
    std::shared_ptr<DynamicsInitialization> init)
{
  Eigen::VectorXs originalPoses = mSkeleton->getPositions();
  Eigen::VectorXs originalScales = mSkeleton->getGroupScales();

  mSkeleton->setGroupScales(init->groupScales);

  std::vector<int> footIndices;
  for (auto foot : mFootNodes)
  {
    footIndices.push_back(foot->getIndexInSkeleton());
  }
  ResidualForceHelper helper(mSkeleton, footIndices);

  s_t force = 0;
  s_t torque = 0;
  int count = 0;

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    s_t dt = init->trialTimesteps[trial];
    for (int t = 1; t < init->poseTrials[trial].cols() - 1; t++)
    {
      if (init->probablyMissingGRF[trial][t])
      {
        continue;
      }
      Eigen::VectorXs q = init->poseTrials[trial].col(t);
      Eigen::VectorXs dq = (init->poseTrials[trial].col(t)
                            - init->poseTrials[trial].col(t - 1))
                           / dt;
      Eigen::VectorXs ddq = (init->poseTrials[trial].col(t + 1)
                             - 2 * init->poseTrials[trial].col(t)
                             + init->poseTrials[trial].col(t - 1))
                            / (dt * dt);
      Eigen::Vector6s residual
          = helper.calculateResidual(q, dq, ddq, init->grfTrials[trial].col(t));
      torque += residual.head<3>().norm();
      s_t frameForce = residual.tail<3>().norm();
      force += frameForce;
      count++;
    }
  }
  std::cout << "count: " << count << std::endl;
  force /= count;
  torque /= count;

  mSkeleton->setPositions(originalPoses);
  mSkeleton->setGroupScales(originalScales);

  return std::make_pair(force, torque);
}

//==============================================================================
// Get the average real measured force (in newtons) and torque (in
// newton-meters)
std::pair<s_t, s_t> DynamicsFitter::computeAverageRealForce(
    std::shared_ptr<DynamicsInitialization> init)
{
  s_t force = 0;
  s_t torque = 0;
  int count = 0;

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    for (int t = 0; t < init->poseTrials[trial].cols() - 2; t++)
    {
      for (int i = 0; i < init->forcePlateTrials[trial].size(); i++)
      {
        if (!init->forcePlateTrials[trial][i].forces[t].hasNaN()
            && !init->forcePlateTrials[trial][i].moments[t].hasNaN())
        {
          s_t forceNorm = init->forcePlateTrials[trial][i].forces[t].norm();
          if (forceNorm > 0)
          {
            force += forceNorm;
            torque += init->forcePlateTrials[trial][i].moments[t].norm();
          }
        }
      }
      count++;
    }
  }
  force /= count;
  torque /= count;

  return std::make_pair(force, torque);
}

//==============================================================================
// Get the average change in the center of pressure point (in meters) after
// "perfecting" the GRF data
s_t DynamicsFitter::computeAverageCOPChange(
    std::shared_ptr<DynamicsInitialization> init)
{
  s_t dist = 0;
  int count = 0;

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    if (init->forcePlateTrials[trial].size()
        == init->perfectForcePlateTrials[trial].size())
    {
      for (int t = 0; t < init->poseTrials[trial].cols() - 2; t++)
      {
        if (init->probablyMissingGRF[trial][t])
        {
          continue;
        }
        for (int i = 0; i < init->forcePlateTrials[trial].size(); i++)
        {
          if (init->forcePlateTrials[trial][i].forces[t].norm() > 1e-8)
          {
            s_t distNow = (init->forcePlateTrials[trial][i].centersOfPressure[t]
                           - init->perfectForcePlateTrials[trial][i]
                                 .centersOfPressure[t])
                              .norm();
            // std::cout << "CoP moved " << distNow << " at time " << t
            //           << std::endl;
            // if (distNow > 0.1)
            // {
            //   std::cout << "'Perfect' CoP [" << i << "]:" << std::endl
            //             << init->perfectForcePlateTrials[trial][i]
            //                    .centersOfPressure[t]
            //             << std::endl;
            //   std::cout << "Measured CoP [" << i << "]:" << std::endl
            //             <<
            //             init->forcePlateTrials[trial][i].centersOfPressure[t]
            //             << std::endl;
            // }
            if (distNow < 0.5)
            {
              dist += distNow;
              count++;
            }
          }
        }
      }
    }
  }
  dist /= count;
  return dist;
}

//==============================================================================
// Get the average change in the force vector (in Newtons) after "perfecting"
// the GRF data
s_t DynamicsFitter::computeAverageForceMagnitudeChange(
    std::shared_ptr<DynamicsInitialization> init)
{
  s_t dist = 0;
  int count = 0;

  for (int trial = 0; trial < init->poseTrials.size(); trial++)
  {
    for (int t = 0; t < init->poseTrials[trial].cols() - 2; t++)
    {
      if (init->probablyMissingGRF[trial][t])
      {
        continue;
      }
      if (init->forcePlateTrials[trial].size()
          == init->perfectForcePlateTrials[trial].size())
      {
        for (int i = 0; i < init->forcePlateTrials[trial].size(); i++)
        {
          if (init->forcePlateTrials[trial][i].forces[t].norm() > 1e-8)
          {
            s_t thisDist = (init->forcePlateTrials[trial][i].forces[t]
                            - init->perfectForcePlateTrials[trial][i].forces[t])
                               .norm();
            // std::cout << "t=" << t << ", plate=" << i << ": " << thisDist
            //           << "N diff" << std::endl;
            dist += thisDist;
            count++;
          }
        }
      }
    }
  }
  dist /= count;
  return dist;
}

//==============================================================================
// This debugs the current state, along with visualizations of errors
// where the dynamics do not match the force plate data
void DynamicsFitter::saveDynamicsToGUI(
    const std::string& path,
    std::shared_ptr<DynamicsInitialization> init,
    int trialIndex,
    int framesPerSecond)
{
  bool renderResidualForces = false;

  std::string skeletonLayerName = "Skeleton";
  Eigen::Vector4s skeletonLayerColor = Eigen::Vector4s(0.7, 0.7, 0.7, 1.0);
  std::string skeletonInertiaLayerName = "Skeleton Inertia";
  Eigen::Vector4s skeletonInertiaLayerColor
      = Eigen::Vector4s(0.0, 0.0, 1.0, 0.5);
  std::string originalSkeletonLayerName = "Original Skeleton";
  Eigen::Vector4s originalSkeletonLayerColor
      = Eigen::Vector4s(1.0, 0.3, 0.3, 0.3);
  std::string originalSkeletonInertiaLayerName = "Original Skeleton Inertia";
  Eigen::Vector4s originalSkeletonInertiaLayerColor
      = Eigen::Vector4s(1.0, 0.0, 0.0, 0.5);
  std::string markerErrorLayerName = "Marker Error";
  Eigen::Vector4s markerErrorLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  std::string forcePlateLayerName = "Force Plates";
  Eigen::Vector4s forcePlateLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  std::string perfectForcePlateLayerName = "'Zero Residual' Force Plates";
  Eigen::Vector4s perfectForcePlateLayerColor
      = Eigen::Vector4s(1.0, 0.0, 1.0, 1.0);
  std::string measuredForcesLayerName = "Measured Forces";
  Eigen::Vector4s measuredForcesLayerColor
      = Eigen::Vector4s(0.0, 0.0, 1.0, 1.0);
  std::string residualLayerName = "Residual Forces";
  Eigen::Vector4s residualLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  std::string impliedForcesLayerName = "Implied Forces";
  Eigen::Vector4s impliedForcesLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  std::string functionalJointCenterLayerName = "Functional Joint Centers";
  Eigen::Vector4s functionalJointCenterLayerColor
      = Eigen::Vector4s(0.0, 1.0, 0.0, 1.0);
  std::string groundLayerName = "Ground";
  Eigen::Vector4s groundLayerColor = Eigen::Vector4s(0.7, 0.7, 0.7, 1.0);
  std::string groundContactLayerName = "Ground Contact";
  Eigen::Vector4s groundContactLayerColor = Eigen::Vector4s(1.0, 1.0, 1.0, 0.5);
  Eigen::Vector4s groundContactActiveColor
      = Eigen::Vector4s(1.0, 0.5, 0.5, 0.5);
  std::string accLayerName = "Acceleration";
  Eigen::Vector4s accLayerColor = Eigen::Vector4s(0.0, 0.0, 1.0, 1.0);
  std::string markersLayerName = "Marker Traces";
  Eigen::Vector4s markersLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);

  if (trialIndex >= init->poseTrials.size())
  {
    std::cout << "Trying to visualize an out-of-bounds trialIndex: "
              << trialIndex << " >= " << init->poseTrials.size() << std::endl;
    exit(1);
  }

  Eigen::VectorXs originalMasses = mSkeleton->getLinkMasses();
  Eigen::VectorXs originalPoses = mSkeleton->getPositions();

  mSkeleton->setLinkMasses(init->bodyMasses);

  ///////////////////////////////////////////////////
  // Start actually rendering out results
  server::GUIRecording server;
  server.setFramesPerSecond(framesPerSecond);
  server.createLayer(skeletonLayerName, skeletonLayerColor);
  server.createLayer(
      skeletonInertiaLayerName, skeletonInertiaLayerColor, false);
  server.createLayer(
      originalSkeletonLayerName, originalSkeletonLayerColor, false);
  server.createLayer(
      originalSkeletonInertiaLayerName,
      originalSkeletonInertiaLayerColor,
      false);
  server.createLayer(markerErrorLayerName, markerErrorLayerColor, false);
  server.createLayer(forcePlateLayerName, forcePlateLayerColor, true);
  server.createLayer(
      perfectForcePlateLayerName, perfectForcePlateLayerColor, false);
  server.createLayer(measuredForcesLayerName, measuredForcesLayerColor, false);
  if (renderResidualForces)
  {
    server.createLayer(residualLayerName, residualLayerColor, false);
  }
  server.createLayer(impliedForcesLayerName, impliedForcesLayerColor, false);
  server.createLayer(
      functionalJointCenterLayerName, functionalJointCenterLayerColor, false);
  server.createLayer(groundLayerName, groundLayerColor);
  server.createLayer(groundContactLayerName, groundContactLayerColor, false);
  server.createLayer(accLayerName, accLayerColor, false);
  server.createLayer(markersLayerName, markersLayerColor, false);

  std::vector<ForcePlate> forcePlates = init->forcePlateTrials[trialIndex];
  std::vector<ForcePlate> perfectForcePlates;
  if (init->perfectForcePlateTrials.size() > trialIndex)
  {
    perfectForcePlates = init->perfectForcePlateTrials[trialIndex];
  }
  Eigen::MatrixXs poses = init->poseTrials[trialIndex];

  if (init->flatGround[trialIndex])
  {
    server.createBox(
        "ground",
        Eigen::Vector3s(10, 0.2, 10),
        Eigen::Vector3s(0, init->groundHeight[trialIndex] - 0.1, 0),
        Eigen::Vector3s::Zero(),
        groundLayerColor,
        groundLayerName,
        false,
        true);
  }

  for (int i = 0; i < init->contactBodies.size(); i++)
  {
    for (int j = 0; j < init->contactBodies[i].size(); j++)
    {
      server.createSphere(
          "contact_sphere_" + std::to_string(i) + "_" + std::to_string(j),
          init->grfBodyContactSphereRadius[trialIndex][i][j],
          Eigen::Vector3s::Zero(),
          init->grfBodyOffForcePlate[trialIndex][0][i]
              ? groundContactActiveColor
              : groundContactLayerColor,
          groundContactLayerName);
    }
  }

  // Render the plates as red rectangles
  for (int i = 0; i < forcePlates.size(); i++)
  {
    if (forcePlates[i].corners.size() > 0)
    {
      std::vector<Eigen::Vector3s> points;
      for (int j = 0; j < forcePlates[i].corners.size(); j++)
      {
        points.push_back(forcePlates[i].corners[j]);
      }
      points.push_back(forcePlates[i].corners[0]);

      server.createLine(
          "plate_" + std::to_string(i),
          points,
          forcePlateLayerColor,
          forcePlateLayerName);
    }
  }

  if (init->defaultForcePlateCorners[trialIndex].size() > 0)
  {
    std::vector<Eigen::Vector3s> points;
    for (int j = 0; j < init->defaultForcePlateCorners[trialIndex].size(); j++)
    {
      points.push_back(init->defaultForcePlateCorners[trialIndex][j]);
    }
    points.push_back(init->defaultForcePlateCorners[trialIndex][0]);

    server.createLine(
        "default_plate", points, forcePlateLayerColor, forcePlateLayerName);
  }

  std::vector<bool> useForces;
  s_t threshold = 0.1;
  for (int timestep = 0; timestep < poses.cols(); timestep++)
  {
    bool anyForceData = false;
    for (int i = 0; i < forcePlates.size(); i++)
    {
      if (forcePlates[i].forces[timestep].norm() > threshold)
      {
        anyForceData = true;
        break;
      }
    }
    // Force ignore frames where we think there's a contact that we're not
    // measuring
    if (init->probablyMissingGRF[trialIndex][timestep])
    {
      anyForceData = false;
    }
    (void)anyForceData;
    useForces.push_back(!init->probablyMissingGRF[trialIndex][timestep]);
  }

  ResidualForceHelper residualHelper
      = ResidualForceHelper(mSkeleton, init->grfBodyIndices);
  SpatialNewtonHelper newtonHelper = SpatialNewtonHelper(mSkeleton);

  std::vector<Eigen::Vector3s> residualForces;
  std::vector<Eigen::Vector3s> residualTorques;
  std::vector<Eigen::Vector3s> linearForceGaps;
  std::vector<Eigen::VectorXs> bodySpatialAccs;
  std::vector<Eigen::VectorXs> bodyLinearAccs;
  std::vector<Eigen::VectorXs> bodySpatialVels;
  std::vector<Eigen::VectorXs> bodyPoses;
  s_t residualNorm = 0.0;
  for (int timestep = 1; timestep < poses.cols() - 1; timestep++)
  {
    s_t dt = init->trialTimesteps[trialIndex];
    Eigen::VectorXs q = poses.col(timestep);
    Eigen::VectorXs dq = (poses.col(timestep) - poses.col(timestep - 1)) / dt;
    Eigen::VectorXs ddq = (poses.col(timestep + 1) - 2 * poses.col(timestep)
                           + poses.col(timestep - 1))
                          / (dt * dt);
    mSkeleton->setTimeStep(dt);
    mSkeleton->setPositions(q);
    mSkeleton->setVelocities(dq);
    mSkeleton->setAccelerations(ddq);

    Eigen::VectorXs bodyCOMs
        = Eigen::VectorXs::Zero(mSkeleton->getNumBodyNodes() * 3);
    Eigen::VectorXs bodyLinAccs
        = Eigen::VectorXs::Zero(mSkeleton->getNumBodyNodes() * 3);
    for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
    {
      bodyCOMs.segment<3>(i * 3) = mSkeleton->getBodyNode(i)->getCOM();
      bodyLinAccs.segment<3>(i * 3)
          = mSkeleton->getBodyNode(i)->getCOMLinearAcceleration();
    }
    bodyPoses.push_back(bodyCOMs);
    bodyLinearAccs.push_back(bodyLinAccs);
    Eigen::Vector3s gap = newtonHelper.calculateLinearForceGap(
        q, dq, ddq, init->grfTrials[trialIndex].col(timestep));
    linearForceGaps.push_back(gap);
    bodySpatialVels.push_back(mSkeleton->getCOMWorldVelocities());
    bodySpatialAccs.push_back(mSkeleton->getCOMWorldAccelerations());

    if (init->probablyMissingGRF[trialIndex][timestep])
    {
      // Keep all our indices matching
      residualForces.push_back(Eigen::Vector3s::Zero());
      residualTorques.push_back(Eigen::Vector3s::Zero());
      linearForceGaps.push_back(Eigen::Vector3s::Zero());
      continue;
    }

    Eigen::Vector6s residual = residualHelper.calculateResidual(
        q, dq, ddq, init->grfTrials[trialIndex].col(timestep));
    residualTorques.push_back(residual.head<3>());
    residualForces.push_back(residual.tail<3>());
    residualNorm += residual.squaredNorm();
  }

  // Render the marker traces
  std::map<std::string, std::vector<Eigen::Vector3s>> markerTraces;
  for (int t = 0; t < init->markerObservationTrials[trialIndex].size(); t++)
  {
    for (auto& pair : init->markerObservationTrials[trialIndex][t])
    {
      markerTraces[pair.first].push_back(pair.second);
    }
    std::vector<std::string> toDelete;
    for (auto& pair : markerTraces)
    {
      if (init->markerObservationTrials[trialIndex][t].count(pair.first) == 0)
      {
        server.createLine(
            pair.first + "_ending_at_" + std::to_string(t),
            pair.second,
            markersLayerColor,
            markersLayerName);
        toDelete.push_back(pair.first);
      }
    }
    for (std::string& key : toDelete)
    {
      markerTraces[key].clear();
    }
  }
  for (auto& pair : markerTraces)
  {
    server.createLine(
        pair.first + "_ending_at_end",
        pair.second,
        markersLayerColor,
        markersLayerName);
    server.createSphere(pair.first + "_marker", 0.01, pair.second[0]);
    server.setObjectTooltip(pair.first + "_marker", pair.first);
  }

  std::cout << "Residual norm: " << residualNorm << std::endl;

  std::vector<Eigen::Vector3s> coms = comPositions(init, trialIndex);
  std::vector<Eigen::Vector3s> comAccs = comAccelerations(init, trialIndex);
  std::vector<Eigen::Vector3s> impliedForces
      = impliedCOMForces(init, trialIndex, mSkeleton->getGravity());
  std::vector<Eigen::Vector3s> measuredForces
      = measuredGRFForces(init, trialIndex);

  std::vector<std::vector<Eigen::Vector3s>> bodyCOMTrails;
  for (int b = 0; b < mSkeleton->getNumBodyNodes(); b++)
  {
    bodyCOMTrails.emplace_back();

    server.createSphere(
        "com_acc_body_center_" + std::to_string(b),
        0.01,
        mSkeleton->getBodyNode(b)->getWorldTransform().translation(),
        accLayerColor,
        accLayerName);
  }

  for (int i = 0; i < impliedForces.size(); i++)
  {
    for (int b = 0; b < mSkeleton->getNumBodyNodes(); b++)
    {
      Eigen::Vector3s pos = bodyPoses[i].segment<3>(b * 3);
      bodyCOMTrails[b].push_back(pos);
    }

    if (i % 1 == 0 && useForces[i])
    {
      if (i > 0 && i - 1 < impliedForces.size())
      {
        std::vector<Eigen::Vector3s> impliedVector;
        impliedVector.push_back(coms[i]);
        impliedVector.push_back(coms[i] + (impliedForces[i - 1] * 0.001));
        server.createLine(
            "com_implied_" + std::to_string(i),
            impliedVector,
            impliedForcesLayerColor,
            impliedForcesLayerName);
      }

      if (measuredForces[i].norm() > 0)
      {
        std::vector<Eigen::Vector3s> measuredVector;
        measuredVector.push_back(coms[i]);
        measuredVector.push_back(coms[i] + (measuredForces[i] * 0.001));
        server.createLine(
            "com_measured_" + std::to_string(i),
            measuredVector,
            measuredForcesLayerColor,
            measuredForcesLayerName);

        if (renderResidualForces && i > 0 && i - 1 < residualForces.size()
            && i - 1 < residualTorques.size())
        {
          std::vector<Eigen::Vector3s> residualForceVector;
          residualForceVector.push_back(coms[i] + (measuredForces[i] * 0.001));
          residualForceVector.push_back(
              coms[i] + (measuredForces[i] * 0.001)
              + (residualForces[i - 1] * 0.001));
          server.createLine(
              "com_residual_force_" + std::to_string(i),
              residualForceVector,
              residualLayerColor,
              residualLayerName);

          std::vector<Eigen::Vector3s> residualTorqueVector;
          residualTorqueVector.push_back(coms[i]);
          residualTorqueVector.push_back(
              coms[i] + (residualTorques[i - 1] * 0.01));
          server.createLine(
              "com_residual_torque_" + std::to_string(i),
              residualTorqueVector,
              Eigen::Vector4s(0, 1, 0, 1),
              residualLayerName);
        }
      }
    }
  }

  server.createLine(
      "com_overall_trail", coms, Eigen::Vector4s(0, 0, 0, 1), accLayerName);
  for (int t = 0; t < comAccs.size(); t++)
  {
    std::vector<Eigen::Vector3s> accVector;
    accVector.push_back(coms[t]);
    accVector.push_back(coms[t] + comAccs[t] * 0.001);
    server.createLine(
        "com_acc_" + std::to_string(t),
        accVector,
        Eigen::Vector4s(1, 1, 0, 1),
        accLayerName);
  }

  for (int b = 0; b < mSkeleton->getNumBodyNodes(); b++)
  {
    s_t massPercentage
        = mSkeleton->getBodyNode(b)->getMass() / mSkeleton->getMass();
    server.createLine(
        "com_trail_body_" + std::to_string(b),
        bodyCOMTrails[b],
        Eigen::Vector4s(
            1 - massPercentage, 1 - massPercentage, 1 - massPercentage, 1),
        accLayerName);
  }

  std::shared_ptr<dynamics::Skeleton> originalSkeleton
      = mSkeleton->cloneSkeleton();
  originalSkeleton->setGroupScales(init->originalGroupScales);
  originalSkeleton->setGroupMasses(init->originalGroupMasses);
  originalSkeleton->setGroupCOMs(init->originalGroupCOMs);
  originalSkeleton->setGroupInertias(init->originalGroupInertias);

  // Render the joints, if we have them
  int numJoints = init->jointCenters[trialIndex].rows() / 3;
  server.createLayer(
      functionalJointCenterLayerName, functionalJointCenterLayerColor, true);
  for (int i = 0; i < numJoints; i++)
  {
    if (init->jointWeights(i) > 0)
    {
      server.setObjectTooltip(
          "joint_center_" + std::to_string(i),
          "Joint center: " + init->joints[i]->getName());
      server.createSphere(
          "joint_center_" + std::to_string(i),
          0.01 * min(3.0, (1.0 / init->jointWeights(i))),
          Eigen::Vector3s::Zero(),
          Eigen::Vector4s(
              functionalJointCenterLayerColor(0),
              functionalJointCenterLayerColor(1),
              functionalJointCenterLayerColor(2),
              init->jointWeights(i)),
          functionalJointCenterLayerName);
    }
  }
  int numAxis = init->jointAxis[trialIndex].rows() / 6;
  for (int i = 0; i < numAxis; i++)
  {
    if (init->axisWeights(i) > 0)
    {
      server.createCapsule(
          "joint_axis_" + std::to_string(i),
          0.003 * min(3.0, (1.0 / init->axisWeights(i))),
          0.1,
          Eigen::Vector3s::Zero(),
          Eigen::Vector3s::Zero(),
          Eigen::Vector4s(
              functionalJointCenterLayerColor(0),
              functionalJointCenterLayerColor(1),
              functionalJointCenterLayerColor(2),
              init->axisWeights(i)),
          functionalJointCenterLayerName);
    }
  }

  for (int timestep = 0; timestep < poses.cols(); timestep++)
  {
    mSkeleton->setPositions(poses.col(timestep));
    server.renderSkeleton(
        mSkeleton, "skel", Eigen::Vector4s::Ones() * -1, skeletonLayerName);
    server.renderSkeletonInertiaCubes(
        mSkeleton,
        "skel_inertia",
        skeletonInertiaLayerColor,
        skeletonInertiaLayerName);

    // Render foot-ground contact spheres
    for (int i = 0; i < init->contactBodies.size(); i++)
    {
      for (int j = 0; j < init->contactBodies[i].size(); j++)
      {
        server.setObjectPosition(
            "contact_sphere_" + std::to_string(i) + "_" + std::to_string(j),
            init->contactBodies[i][j]->getWorldTransform().translation());
        if (init->probablyMissingGRF[trialIndex][timestep])
        {
          if (init->grfBodyOffForcePlate[trialIndex][timestep][i])
          {
            server.setObjectColor(
                "contact_sphere_" + std::to_string(i) + "_" + std::to_string(j),
                groundContactActiveColor);
          }
          else
          {
            server.setObjectColor(
                "contact_sphere_" + std::to_string(i) + "_" + std::to_string(j),
                Eigen::Vector4s(0, 0, 1, 1));
          }
        }
        else
        {
          if (init->grfBodyOffForcePlate[trialIndex][timestep][i])
          {
            server.setObjectColor(
                "contact_sphere_" + std::to_string(i) + "_" + std::to_string(j),
                groundContactActiveColor);
          }
          else
          {
            server.setObjectColor(
                "contact_sphere_" + std::to_string(i) + "_" + std::to_string(j),
                groundContactLayerColor);
          }
        }
      }
    }

    // Render accelerations
    if (timestep > 0 && timestep < poses.cols() - 1)
    {
      Eigen::Vector3s totalForce = Eigen::Vector3s::Zero();
      for (int b = 0; b < mSkeleton->getNumBodyNodes(); b++)
      {
        Eigen::Vector3s pos = bodyPoses[timestep - 1].segment<3>(b * 3);
        bodyCOMTrails[b].push_back(pos);

        Eigen::Vector3s vel
            = bodySpatialVels[timestep - 1].segment<3>(b * 6 + 3);

        (void)vel;
        Eigen::Vector3s linAcc = bodyLinearAccs[timestep - 1].segment<3>(b * 3);
        Eigen::Vector3s linForce
            = linAcc * mSkeleton->getBodyNode(b)->getMass();
        totalForce += linForce;
        /*
        if (i > 0 && i < impliedForces.size() - 1)
        {
          s_t dt = init->trialTimesteps[trialIndex];
          Eigen::Vector3s nextPos = bodyPoses[i + 1].segment<3>(b * 3);
          Eigen::Vector3s prevPos = bodyPoses[i - 1].segment<3>(b * 3);
          Eigen::Vector3s thisVel = (pos - prevPos) / dt;
          Eigen::Vector3s nextVel = (nextPos - pos) / dt;
          Eigen::Vector3s thisAcc = (nextVel - thisVel) / dt;
          if ((thisAcc - acc).norm() > 1e-6)
          {
            std::cout << "Suspicious gap in accelerations for body " << b << "
        \""
                      << mSkeleton->getBodyNode(b)->getName()
                      << " at time t=" << i << "!" << std::endl;
            Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(3, 5);
            compare.col(0) = acc;
            compare.col(1) = thisAcc;
            compare.col(2) = acc - thisAcc;
            compare.col(3) = linAcc;
            compare.col(4) = thisAcc - linAcc;
            std::cout << "Spatial - FD - Diff - Linear - Diff" << std::endl
                      << compare << std::endl;
          }
          // Eigen::Vector3s bruteForceAcc =
        }
        */

        std::vector<Eigen::Vector3s> accVector;
        accVector.push_back(pos);
        accVector.push_back(pos + linForce * 0.001);
        server.createLine(
            "com_acc_body_" + std::to_string(b),
            accVector,
            accLayerColor,
            accLayerName);

        std::vector<Eigen::Vector3s> velVector;
        velVector.push_back(pos);
        velVector.push_back(pos + vel * 0.1);
        server.createLine(
            "com_vel_body_" + std::to_string(b),
            velVector,
            Eigen::Vector4s(1, 0.25, 0, 1),
            accLayerName);

        server.setObjectPosition(
            "com_acc_body_center_" + std::to_string(b), pos);
      }

      Eigen::Vector3s com = mSkeleton->getCOM();
      Eigen::Vector3s comAcc = mSkeleton->getCOMLinearAcceleration();
      Eigen::Vector3s comForce = comAcc * mSkeleton->getMass();
      (void)comForce; // todo: should be equal to totalForce

      std::vector<Eigen::Vector3s> accVector;
      accVector.push_back(com);
      accVector.push_back(com + totalForce * 0.001);
      server.createLine("com_acc", accVector, accLayerColor, accLayerName);
    }

    // Move the markers
    for (auto& pair : markerTraces)
    {
      if (init->markerObservationTrials[trialIndex][timestep].count(pair.first)
          > 0)
      {
        server.setObjectPosition(
            pair.first + "_marker",
            init->markerObservationTrials[trialIndex][timestep].at(pair.first));
      }
    }

    for (int i = 0; i < forcePlates.size(); i++)
    {
      server.deleteObject("force_" + std::to_string(i));
      if (forcePlates[i].forces[timestep].squaredNorm() > 0)
      {
        std::vector<Eigen::Vector3s> forcePoints;
        forcePoints.push_back(forcePlates[i].centersOfPressure[timestep]);
        forcePoints.push_back(
            forcePlates[i].centersOfPressure[timestep]
            + (forcePlates[i].forces[timestep] * 0.001));
        server.createLine(
            "force_" + std::to_string(i),
            forcePoints,
            forcePlateLayerColor,
            forcePlateLayerName);
      }
    }

    for (int i = 0; i < perfectForcePlates.size(); i++)
    {
      server.deleteObject("perfect_force_" + std::to_string(i));
      if (perfectForcePlates[i].forces[timestep].squaredNorm() > 0)
      {
        std::vector<Eigen::Vector3s> forcePoints;
        forcePoints.push_back(
            perfectForcePlates[i].centersOfPressure[timestep]);
        forcePoints.push_back(
            perfectForcePlates[i].centersOfPressure[timestep]
            + (perfectForcePlates[i].forces[timestep] * 0.001));
        server.createLine(
            "perfect_force_" + std::to_string(i),
            forcePoints,
            perfectForcePlateLayerColor,
            perfectForcePlateLayerName);
      }
    }

    // Render Marker Errors
    auto simulatedMarkers
        = mSkeleton->getMarkerMapWorldPositions(init->updatedMarkerMap);
    auto realMarkers = init->markerObservationTrials[trialIndex][timestep];
    for (auto pair : simulatedMarkers)
    {
      if (realMarkers.count(pair.first))
      {
        std::vector<Eigen::Vector3s> points;
        points.push_back(pair.second);
        points.push_back(realMarkers.at(pair.first));
        server.createLine(
            "error_" + pair.first,
            points,
            markerErrorLayerColor,
            markerErrorLayerName);
      }
    }

    // Render virtual joints
    for (int i = 0; i < numJoints; i++)
    {
      if (init->jointWeights(i) > 0)
      {
        Eigen::Vector3s inferredJointCenter
            = init->jointCenters[trialIndex].block<3, 1>(i * 3, timestep);
        server.setObjectPosition(
            "joint_center_" + std::to_string(i), inferredJointCenter);
        if (i < init->jointsAdjacentMarkers.size())
        {
          for (std::string marker : init->jointsAdjacentMarkers[i])
          {
            if (init->markerObservationTrials[trialIndex][timestep].count(
                    marker))
            {
              std::vector<Eigen::Vector3s> centerToMarker;
              centerToMarker.push_back(inferredJointCenter);
              centerToMarker.push_back(
                  init->markerObservationTrials[trialIndex][timestep][marker]);
              server.createLine(
                  "joint_center_" + std::to_string(i) + "_to_marker_" + marker,
                  centerToMarker,
                  functionalJointCenterLayerColor,
                  functionalJointCenterLayerName);
            }
          }
        }
      }
    }
    for (int i = 0; i < numAxis; i++)
    {
      if (init->axisWeights(i) > 0)
      {
        // Render an axis capsule
        server.setObjectPosition(
            "joint_axis_" + std::to_string(i),
            init->jointAxis[trialIndex].block<3, 1>(i * 6, timestep));
        Eigen::Vector3s dir
            = init->jointAxis[trialIndex].block<3, 1>(i * 6 + 3, timestep);
        Eigen::Matrix3s R = Eigen::Matrix3s::Identity();
        R.col(2) = dir;
        R.col(1) = Eigen::Vector3s::UnitZ().cross(dir);
        R.col(0) = R.col(1).cross(R.col(2));
        server.setObjectRotation(
            "joint_axis_" + std::to_string(i), math::matrixToEulerXYZ(R));
      }
    }

    // Render Original Skeleton
    originalSkeleton->setPositions(
        init->originalPoses[trialIndex].col(timestep));
    server.renderSkeleton(
        originalSkeleton,
        "original_skel",
        originalSkeletonLayerColor,
        originalSkeletonLayerName);
    server.renderSkeletonInertiaCubes(
        originalSkeleton,
        "original_skel_inertia",
        originalSkeletonInertiaLayerColor,
        originalSkeletonInertiaLayerName);
    server.saveFrame();
  }

  mSkeleton->setPositions(originalPoses);
  mSkeleton->setLinkMasses(originalMasses);

  server.writeFramesJson(path);
}

//==============================================================================
void DynamicsFitter::setTolerance(double tol)
{
  mTolerance = tol;
}

//==============================================================================
void DynamicsFitter::setIterationLimit(int limit)
{
  mIterationLimit = limit;
}

//==============================================================================
void DynamicsFitter::setLBFGSHistoryLength(int len)
{
  mLBFGSHistoryLength = len;
}

//==============================================================================
void DynamicsFitter::setCheckDerivatives(bool check)
{
  mCheckDerivatives = check;
}

//==============================================================================
void DynamicsFitter::setPrintFrequency(int freq)
{
  mPrintFrequency = freq;
}

//==============================================================================
void DynamicsFitter::setSilenceOutput(bool silent)
{
  mSilenceOutput = silent;
}

//==============================================================================
void DynamicsFitter::setDisableLinesearch(bool disable)
{
  mDisableLinesearch = disable;
}

} // namespace biomechanics
} // namespace dart