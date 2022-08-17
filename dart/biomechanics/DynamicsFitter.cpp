#include "dart/biomechanics/DynamicsFitter.hpp"

#include <memory>
#include <string>
#include <tuple>

#include "dart/biomechanics/C3DForcePlatforms.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/AssignmentMatcher.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/utils/AccelerationSmoother.hpp"

namespace dart {
namespace biomechanics {

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

  // TODO: this is certainly a more efficient way to do this, since we only care
  // about the first 6 values anyways
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
    Eigen::VectorXs forcesConcat)
{
  return calculateResidual(q, dq, ddq, forcesConcat).squaredNorm();
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
    neural::WithRespectTo* wrt)
{
  Eigen::Vector6s res = calculateResidual(q, dq, ddq, forcesConcat);
  Eigen::MatrixXs jac
      = calculateResidualJacobianWrt(q, dq, ddq, forcesConcat, wrt);
  return 2 * jac.transpose() * res;
}

//==============================================================================
// Computes the gradient of the residual norm with respect to `wrt`
Eigen::VectorXs ResidualForceHelper::finiteDifferenceResidualNormGradientWrt(
    Eigen::VectorXs q,
    Eigen::VectorXs dq,
    Eigen::VectorXs ddq,
    Eigen::VectorXs forcesConcat,
    neural::WithRespectTo* wrt)
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
        perturbed = calculateResidualNorm(
            mSkel->getPositions(),
            mSkel->getVelocities(),
            mSkel->getAccelerations(),
            forcesConcat);
        return true;
      },
      result,
      wrt == neural::WithRespectTo::POSITION ? 1e-4 : 1e-2,
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
    dynamics::MarkerMap markerMap,
    std::vector<dynamics::BodyNode*> footNodes)
  : mInit(init),
    mSkeleton(skeleton),
    mMarkerMap(markerMap),
    mFootNodes(footNodes),
    mIncludeMasses(true),
    mIncludeCOMs(true),
    mIncludeInertias(true),
    mIncludeBodyScales(true),
    mIncludePoses(true),
    mIncludeMarkerOffsets(true)
{
  // 1. Set up the markers

  for (auto& pair : markerMap)
  {
    mMarkerNames.push_back(pair.first);
    mMarkers.push_back(pair.second);
  }

  // 2. Set up the q, dq, ddq

  int dofs = skeleton->getNumDofs();
  s_t dt = skeleton->getTimeStep();
  for (int i = 0; i < init->poseTrials.size(); i++)
  {
    Eigen::MatrixXs& inputPoses = init->poseTrials[i];
    std::cout << "Trial " << i << ": " << inputPoses.cols() << std::endl;
    Eigen::MatrixXs poses = Eigen::MatrixXs::Zero(dofs, inputPoses.cols());
    Eigen::MatrixXs vels = Eigen::MatrixXs::Zero(dofs, inputPoses.cols() - 1);
    Eigen::MatrixXs accs = Eigen::MatrixXs::Zero(dofs, inputPoses.cols() - 2);
    for (int j = 0; j < inputPoses.cols(); j++)
    {
      poses.col(j) = inputPoses.col(j);
    }
    for (int j = 0; j < inputPoses.cols() - 1; j++)
    {
      vels.col(j) = (inputPoses.col(j + 1) - inputPoses.col(j)) / dt;
    }
    for (int j = 0; j < inputPoses.cols() - 2; j++)
    {
      accs.col(j) = (inputPoses.col(j + 2) - 2 * inputPoses.col(j + 1)
                     + inputPoses.col(j))
                    / (dt * dt);
    }
    mPoses.push_back(poses);
    mVels.push_back(vels);
    mAccs.push_back(accs);
  }

  // 3. Match force plates to the feet

  Eigen::VectorXs originalPose = skeleton->getPositions();

  for (int i = 0; i < mFootNodes.size(); i++)
  {
    mForceBodyIndices.push_back(mFootNodes[i]->getIndexInSkeleton());
  }

  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    std::vector<ForcePlate> forcePlates = init->forcePlateTrials[trial];

    Eigen::MatrixXs GRF
        = Eigen::MatrixXs::Zero(forcePlates.size() * 6, mPoses[trial].cols());
    for (int t = 0; t < mPoses[trial].cols(); t++)
    {
      skeleton->setPositions(mPoses[trial].col(t));

      Eigen::MatrixXs scores
          = Eigen::MatrixXs::Zero(forcePlates.size(), mFootNodes.size());
      // Now we have to assign feet to force plates
      for (int i = 0; i < mFootNodes.size(); i++)
      {
        Eigen::Vector3s footLoc
            = mFootNodes[i]->getWorldTransform().translation();
        for (int j = 0; j < forcePlates.size(); j++)
        {
          Eigen::Vector3s cop = forcePlates[j].centersOfPressure[t];
          scores(j, i) = (footLoc - cop).norm();
        }
      }
      Eigen::VectorXi assignment
          = math::AssignmentMatcher::assignRowsToColumns(scores);

      for (int i = 0; i < forcePlates.size(); i++)
      {
        int footAssignment = assignment(i);
        if (footAssignment != -1)
        {
          Eigen::Vector3s cop = forcePlates[i].centersOfPressure[t];
          Eigen::Vector3s force = forcePlates[i].forces[t];
          Eigen::Vector3s moments = forcePlates[i].moments[t];
          Eigen::Vector6s wrench = Eigen::Vector6s::Zero();
          wrench.head<3>() = moments;
          wrench.head<3>() = force;
          Eigen::Isometry3s wrenchT = Eigen::Isometry3s::Identity();
          wrenchT.translation() = cop;
          Eigen::Vector6s worldWrench = math::AdInvT(wrenchT, wrench);
          GRF.block<6, 1>(footAssignment * 6, t) = worldWrench;
        }
      }
    }
    mGRFs.push_back(GRF);
  }

  skeleton->setPositions(originalPose);

  mResidualHelper
      = std::make_shared<ResidualForceHelper>(mSkeleton, mForceBodyIndices);
}

//==============================================================================
// This returns the dimension of the decision variables (the length of the
// flatten() vector), which depends on which variables we choose to include in
// the optimization problem.
int DynamicsFitProblem::getProblemSize()
{
  int size = 0;
  if (mIncludeMasses)
  {
    size += mSkeleton->getNumScaleGroups();
  }
  if (mIncludeCOMs)
  {
    size += mSkeleton->getNumScaleGroups() * 3;
  }
  if (mIncludeInertias)
  {
    size += mSkeleton->getNumScaleGroups() * 6;
  }
  if (mIncludeBodyScales)
  {
    size += mSkeleton->getGroupScaleDim();
  }
  if (mIncludeMarkerOffsets)
  {
    size += mMarkerMap.size() * 3;
  }
  if (mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();
    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      // Add pos + vel + acc
      size += mAccs[trial].cols() * dofs * 3;
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
  if (mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    flat.segment(cursor, dim) = mSkeleton->getGroupMasses();
    cursor += dim;
  }
  if (mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    flat.segment(cursor, dim) = mSkeleton->getGroupCOMs();
    cursor += dim;
  }
  if (mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    flat.segment(cursor, dim) = mSkeleton->getGroupInertias();
    cursor += dim;
  }
  if (mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    flat.segment(cursor, dim) = mSkeleton->getGroupScales();
    cursor += dim;
  }
  if (mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      flat.segment(cursor, 3) = mMarkers[i].second;
      cursor += 3;
    }
  }
  if (mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      for (int t = 0; t < mAccs[trial].cols(); t++)
      {
        flat.segment(cursor, dofs) = mPoses[trial].col(t);
        cursor += dofs;
        flat.segment(cursor, dofs) = mVels[trial].col(t);
        cursor += dofs;
        flat.segment(cursor, dofs) = mAccs[trial].col(t);
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
  if (mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    flat.segment(cursor, dim) = mSkeleton->getGroupMassesUpperBound();
    cursor += dim;
  }
  if (mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    flat.segment(cursor, dim) = mSkeleton->getGroupCOMUpperBound();
    cursor += dim;
  }
  if (mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    flat.segment(cursor, dim) = mSkeleton->getGroupInertiasUpperBound();
    cursor += dim;
  }
  if (mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    flat.segment(cursor, dim) = mSkeleton->getGroupScalesUpperBound();
    cursor += dim;
  }
  if (mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      flat.segment(cursor, 3) = Eigen::Vector3s::Ones() * 5;
      cursor += 3;
    }
  }
  if (mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      for (int t = 0; t < mAccs[trial].cols(); t++)
      {
        flat.segment(cursor, dofs) = mSkeleton->getPositionUpperLimits();
        cursor += dofs;
        flat.segment(cursor, dofs) = mSkeleton->getVelocityUpperLimits();
        cursor += dofs;
        flat.segment(cursor, dofs) = mSkeleton->getAccelerationUpperLimits();
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
  if (mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    flat.segment(cursor, dim) = mSkeleton->getGroupMassesLowerBound();
    cursor += dim;
  }
  if (mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    flat.segment(cursor, dim) = mSkeleton->getGroupCOMLowerBound();
    cursor += dim;
  }
  if (mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    flat.segment(cursor, dim) = mSkeleton->getGroupInertiasLowerBound();
    cursor += dim;
  }
  if (mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    flat.segment(cursor, dim) = mSkeleton->getGroupScalesLowerBound();
    cursor += dim;
  }
  if (mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      flat.segment(cursor, 3) = Eigen::Vector3s::Ones() * -5;
      cursor += 3;
    }
  }
  if (mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();
    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      for (int t = 0; t < mAccs[trial].cols(); t++)
      {
        flat.segment(cursor, dofs) = mSkeleton->getPositionLowerLimits();
        cursor += dofs;
        flat.segment(cursor, dofs) = mSkeleton->getVelocityLowerLimits();
        cursor += dofs;
        flat.segment(cursor, dofs) = mSkeleton->getAccelerationLowerLimits();
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
    int dofs = mSkeleton->getNumDofs();
    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      for (int t = 0; t < mAccs[trial].cols(); t++)
      {
        mPoses[trial].col(t) = x.segment(cursor, dofs);
        cursor += dofs;
        mVels[trial].col(t) = x.segment(cursor, dofs);
        cursor += dofs;
        mAccs[trial].col(t) = x.segment(cursor, dofs);
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
s_t DynamicsFitProblem::computeLoss(Eigen::VectorXs x)
{
  unflatten(x);

  s_t sum = 0.0;

  Eigen::VectorXs originalPos = mSkeleton->getPositions();
  mSkeleton->clearExternalForces();

  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    for (int t = 0; t < mAccs[trial].cols(); t++)
    {
      mSkeleton->setPositions(mPoses[trial].col(t));

      // Add force residual RMS errors
      sum += mResidualHelper->calculateResidualNorm(
          mPoses[trial].col(t),
          mVels[trial].col(t),
          mAccs[trial].col(t),
          mGRFs[trial].col(t));

      /*
      // Add marker RMS errors
      auto markerPoses = mSkeleton->getMarkerMapWorldPositions(mMarkerMap);
      auto observedMarkerPoses = mInit->markerObservationTrials[trial][t];
      for (auto& pair : observedMarkerPoses)
      {
        if (markerPoses.count(pair.first))
        {
          Eigen::Vector3s diff = markerPoses.at(pair.first) - pair.second;
          sum += diff.squaredNorm();
        }
      }
      */
    }
  }

  return sum;
}

//==============================================================================
// This gets the gradient of the loss function
Eigen::VectorXs DynamicsFitProblem::computeGradient(Eigen::VectorXs x)
{
  unflatten(x);

  Eigen::VectorXs grad = Eigen::VectorXs::Zero(getProblemSize());

  int posesCursor = 0;
  if (mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    posesCursor += dim;
  }
  if (mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    posesCursor += dim;
  }
  if (mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    posesCursor += dim;
  }
  if (mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    posesCursor += dim;
  }
  if (mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      // Currently this has zero effect on the gradient
      posesCursor += 3;
    }
  }

  for (int trial = 0; trial < mPoses.size(); trial++)
  {
    for (int t = 0; t < mAccs[trial].cols(); t++)
    {
      int cursor = 0;
      if (mIncludeMasses)
      {
        int dim = mSkeleton->getNumScaleGroups();
        grad.segment(cursor, dim)
            += mResidualHelper->calculateResidualNormGradientWrt(
                mPoses[trial].col(t),
                mVels[trial].col(t),
                mAccs[trial].col(t),
                mGRFs[trial].col(t),
                neural::WithRespectTo::GROUP_MASSES);
        cursor += dim;
      }
      if (mIncludeCOMs)
      {
        int dim = mSkeleton->getNumScaleGroups() * 3;
        grad.segment(cursor, dim)
            += mResidualHelper->calculateResidualNormGradientWrt(
                mPoses[trial].col(t),
                mVels[trial].col(t),
                mAccs[trial].col(t),
                mGRFs[trial].col(t),
                neural::WithRespectTo::GROUP_COMS);
        cursor += dim;
      }
      if (mIncludeInertias)
      {
        int dim = mSkeleton->getNumScaleGroups() * 6;
        grad.segment(cursor, dim)
            += mResidualHelper->calculateResidualNormGradientWrt(
                mPoses[trial].col(t),
                mVels[trial].col(t),
                mAccs[trial].col(t),
                mGRFs[trial].col(t),
                neural::WithRespectTo::GROUP_INERTIAS);
        cursor += dim;
      }
      if (mIncludeBodyScales)
      {
        int dim = mSkeleton->getGroupScaleDim();
        grad.segment(cursor, dim)
            += mResidualHelper->calculateResidualNormGradientWrt(
                mPoses[trial].col(t),
                mVels[trial].col(t),
                mAccs[trial].col(t),
                mGRFs[trial].col(t),
                neural::WithRespectTo::GROUP_SCALES);
        cursor += dim;
      }
      if (mIncludeMarkerOffsets)
      {
        for (int i = 0; i < mMarkers.size(); i++)
        {
          // Currently this has zero effect on the gradient
          cursor += 3;
        }
      }

      if (mIncludePoses)
      {
        int dofs = mSkeleton->getNumDofs();
        grad.segment(posesCursor, dofs)
            = mResidualHelper->calculateResidualNormGradientWrt(
                mPoses[trial].col(t),
                mVels[trial].col(t),
                mAccs[trial].col(t),
                mGRFs[trial].col(t),
                neural::WithRespectTo::POSITION);

        posesCursor += dofs;

        grad.segment(posesCursor, dofs)
            = mResidualHelper->calculateResidualNormGradientWrt(
                mPoses[trial].col(t),
                mVels[trial].col(t),
                mAccs[trial].col(t),
                mGRFs[trial].col(t),
                neural::WithRespectTo::VELOCITY);
        posesCursor += dofs;

        grad.segment(posesCursor, dofs)
            = mResidualHelper->calculateResidualNormGradientWrt(
                mPoses[trial].col(t),
                mVels[trial].col(t),
                mAccs[trial].col(t),
                mGRFs[trial].col(t),
                neural::WithRespectTo::ACCELERATION);
        posesCursor += dofs;
      }
    }
  }

  assert(posesCursor == grad.size());

  // TODO
  return grad;
}

//==============================================================================
// This gets the gradient of the loss function
Eigen::VectorXs DynamicsFitProblem::finiteDifferenceGradient(Eigen::VectorXs x)
{
  Eigen::VectorXs result = Eigen::VectorXs::Zero(getProblemSize());

  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::VectorXs perturbedX = x;
        perturbedX(dof) += eps;
        perturbed = computeLoss(perturbedX);
        return true;
      },
      result,
      1e-3,
      true);

  return result;
}

//==============================================================================
void debugVector(
    Eigen::VectorXs fd, Eigen::VectorXs analytical, std::string name, s_t tol)
{
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
    }
  }
}

//==============================================================================
// Print out the errors in a gradient vector in human readable form
void DynamicsFitProblem::debugErrors(
    Eigen::VectorXs fd, Eigen::VectorXs analytical, s_t tol)
{
  int cursor = 0;
  if (mIncludeMasses)
  {
    int dim = mSkeleton->getNumScaleGroups();
    debugVector(
        fd.segment(cursor, dim), analytical.segment(cursor, dim), "mass", tol);
    cursor += dim;
  }
  if (mIncludeCOMs)
  {
    int dim = mSkeleton->getNumScaleGroups() * 3;
    debugVector(
        fd.segment(cursor, dim), analytical.segment(cursor, dim), "COM", tol);
    cursor += dim;
  }
  if (mIncludeInertias)
  {
    int dim = mSkeleton->getNumScaleGroups() * 6;
    debugVector(
        fd.segment(cursor, dim),
        analytical.segment(cursor, dim),
        "inertia",
        tol);
    cursor += dim;
  }
  if (mIncludeBodyScales)
  {
    int dim = mSkeleton->getGroupScaleDim();
    debugVector(
        fd.segment(cursor, dim),
        analytical.segment(cursor, dim),
        "bodyScales",
        tol);
    cursor += dim;
  }
  if (mIncludeMarkerOffsets)
  {
    for (int i = 0; i < mMarkers.size(); i++)
    {
      debugVector(
          fd.segment(cursor, 3),
          analytical.segment(cursor, 3),
          "marker_" + std::to_string(i),
          tol);
      cursor += 3;
    }
  }
  if (mIncludePoses)
  {
    int dofs = mSkeleton->getNumDofs();

    for (int trial = 0; trial < mPoses.size(); trial++)
    {
      for (int t = 0; t < mAccs[trial].cols(); t++)
      {
        debugVector(
            fd.segment(cursor, dofs),
            analytical.segment(cursor, dofs),
            "poses@t=" + std::to_string(t),
            tol);
        cursor += dofs;
        debugVector(
            fd.segment(cursor, dofs),
            analytical.segment(cursor, dofs),
            "vels@t=" + std::to_string(t),
            tol);
        cursor += dofs;
        debugVector(
            fd.segment(cursor, dofs),
            analytical.segment(cursor, dofs),
            "accs@t=" + std::to_string(t),
            tol);
        cursor += dofs;
      }
    }
  }
}

//==============================================================================
DynamicsFitProblem& DynamicsFitProblem::setIncludeMasses(bool value)
{
  mIncludeMasses = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblem& DynamicsFitProblem::setIncludeCOMs(bool value)
{
  mIncludeCOMs = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblem& DynamicsFitProblem::setIncludeInertias(bool value)
{
  mIncludeInertias = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblem& DynamicsFitProblem::setIncludePoses(bool value)
{
  mIncludePoses = value;
  return *(this);
}

//==============================================================================
DynamicsFitProblem& DynamicsFitProblem::setIncludeMarkerOffsets(bool value)
{
  mIncludeMarkerOffsets = value;
  return *(this);
}

//==============================================================================
DynamicsFitter::DynamicsFitter(
    std::shared_ptr<dynamics::Skeleton> skeleton, dynamics::MarkerMap markerMap)
  : mSkeleton(skeleton),
    mMarkerMap(markerMap){

    };

//==============================================================================
// This bundles together the objects we need in order to track a dynamics
// problem around through multiple steps of optimization
std::shared_ptr<DynamicsInitialization> DynamicsFitter::createInitialization(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::vector<std::vector<ForcePlate>> forcePlateTrials,
    std::vector<Eigen::MatrixXs> poseTrials,
    std::vector<int> framesPerSecond,
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        markerObservationTrials)
{
  std::shared_ptr<DynamicsInitialization> init
      = std::make_shared<DynamicsInitialization>();
  init->forcePlateTrials = forcePlateTrials;
  init->originalPoseTrials = poseTrials;
  init->markerObservationTrials = markerObservationTrials;
  init->bodyMasses = skel->getLinkMasses();

  for (int i = 0; i < init->originalPoseTrials.size(); i++)
  {
    utils::AccelerationSmoother smoother(
        init->originalPoseTrials[i].cols(), 0.1);
    init->poseTrials.push_back(smoother.smooth(init->originalPoseTrials[i]));
    init->trialTimesteps.push_back(1.0 / framesPerSecond[i]);
  }
  return init;
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
  for (int i = 0; i < coms.size() - 2; i++)
  {
    Eigen::Vector3s v1 = (coms[i + 1] - coms[i]) / dt;
    Eigen::Vector3s v2 = (coms[i + 2] - coms[i + 1]) / dt;
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
    bool includeGravity)
{
  std::vector<Eigen::Vector3s> accs = comAccelerations(init, trial);
  s_t totalMass = init->bodyMasses.sum();

  Eigen::Vector3s gravity = Eigen::Vector3s(0, -9.81, 0);

  std::vector<Eigen::Vector3s> forces;
  for (int i = 0; i < accs.size(); i++)
  {
    // f + m * g = m * a
    // f = m * (a - g)
    Eigen::Vector3s a = accs[i];
    if (includeGravity)
    {
      a -= gravity;
    }
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

  for (int timestep = 0; timestep < init->poseTrials[trial].cols() - 2;
       timestep++)
  {
    Eigen::Vector3s totalForce = Eigen::Vector3s::Zero();
    for (int i = 0; i < init->forcePlateTrials[trial].size(); i++)
    {
      totalForce += init->forcePlateTrials[trial][i].forces[timestep];
    }

    forces.push_back(totalForce);
  }

  return forces;
}

//==============================================================================
// 1. Scale the total mass of the body (keeping the ratios of body links
// constant) to get it as close as possible to GRF gravity forces.
void DynamicsFitter::scaleLinkMassesFromGravity(
    std::shared_ptr<DynamicsInitialization> init)
{
  s_t totalGRFs = 0.0;
  s_t totalAccs = 0.0;
  s_t gravity = 9.81;
  for (int i = 0; i < init->poseTrials.size(); i++)
  {
    std::vector<Eigen::Vector3s> grfs = measuredGRFForces(init, i);
    for (Eigen::Vector3s& grf : grfs)
    {
      totalGRFs += grf(1);
    }
    std::vector<Eigen::Vector3s> accs = comAccelerations(init, i);
    for (Eigen::Vector3s& acc : accs)
    {
      totalAccs += acc(1) + gravity;
    }
  }

  std::cout << "Total ACCs: " << totalAccs << std::endl;
  std::cout << "Total mass: " << init->bodyMasses.sum() << std::endl;
  std::cout << "(Total ACCs) * (Total mass): "
            << totalAccs * init->bodyMasses.sum() << std::endl;
  std::cout << "Total GRFs: " << totalGRFs << std::endl;

  s_t impliedTotalMass = totalGRFs / totalAccs;
  std::cout << "Implied total mass: " << impliedTotalMass << std::endl;
  s_t ratio = impliedTotalMass / init->bodyMasses.sum();
  init->bodyMasses *= ratio;
  std::cout << "Adjusted total mass to match GRFs: " << init->bodyMasses.sum()
            << std::endl;
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
  Eigen::MatrixXs A_no_gravity
      = Eigen::MatrixXs::Zero(totalTimesteps * 3, mSkeleton->getNumBodyNodes());
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
      for (int t = 0; t < bodyPosesOverTime.at(body->getName()).cols() - 2; t++)
      {
        Eigen::Vector3s v1 = (bodyPosesOverTime.at(body->getName()).col(t + 1)
                              - bodyPosesOverTime.at(body->getName()).col(t))
                             / dt;
        Eigen::Vector3s v2
            = (bodyPosesOverTime.at(body->getName()).col(t + 2)
               - bodyPosesOverTime.at(body->getName()).col(t + 1))
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
    for (int t = 0; t < poses.cols() - 2; t++)
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
        = impliedCOMForces(init, trial, false);
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
        = impliedCOMForces(init, trial, true);
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
    if (init->bodyMasses(i) < 0.001)
    {
      init->bodyMasses(i) = 0.001;
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
// This debugs the current state, along with visualizations of errors where
// the dynamics do not match the force plate data
void DynamicsFitter::saveDynamicsToGUI(
    const std::string& path,
    std::shared_ptr<DynamicsInitialization> init,
    int trialIndex,
    int framesPerSecond)
{
  std::string forcePlateLayerName = "Force Plates";
  Eigen::Vector4s forcePlateLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);
  std::string measuredForcesLayerName = "Measured Forces";
  Eigen::Vector4s measuredForcesLayerColor
      = Eigen::Vector4s(0.0, 0.0, 1.0, 1.0);
  std::string impliedForcesLayerName = "Implied Forces";
  Eigen::Vector4s impliedForcesLayerColor = Eigen::Vector4s(1.0, 0.0, 0.0, 1.0);

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
  server.renderSkeleton(mSkeleton);

  server.createLayer(forcePlateLayerName, forcePlateLayerColor);
  server.createLayer(measuredForcesLayerName, measuredForcesLayerColor);
  server.createLayer(impliedForcesLayerName, impliedForcesLayerColor);

  std::vector<ForcePlate> forcePlates = init->forcePlateTrials[trialIndex];
  Eigen::MatrixXs poses = init->poseTrials[trialIndex];

  server.createSphere(
      "skel_com", 0.02, Eigen::Vector3s::Zero(), Eigen::Vector4s(0, 0, 1, 0.5));
  server.setObjectTooltip("skel_com", "Center of Mass");

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
    useForces.push_back(anyForceData);
  }

  std::vector<Eigen::Vector3s> coms = comPositions(init, trialIndex);
  std::vector<Eigen::Vector3s> impliedForces
      = impliedCOMForces(init, trialIndex, true);
  std::vector<Eigen::Vector3s> measuredForces
      = measuredGRFForces(init, trialIndex);

  for (int i = 0; i < impliedForces.size(); i++)
  {
    if (i % 1 == 0 && useForces[i])
    {
      std::vector<Eigen::Vector3s> impliedVector;
      impliedVector.push_back(coms[i]);
      impliedVector.push_back(coms[i] + (impliedForces[i] * 0.001));
      server.createLine(
          "com_implied_" + std::to_string(i),
          impliedVector,
          impliedForcesLayerColor,
          impliedForcesLayerName);

      std::vector<Eigen::Vector3s> measuredVector;
      measuredVector.push_back(coms[i]);
      measuredVector.push_back(coms[i] + (measuredForces[i] * 0.001));
      server.createLine(
          "com_measured_" + std::to_string(i),
          measuredVector,
          measuredForcesLayerColor,
          measuredForcesLayerName);
    }
  }

  for (int timestep = 0; timestep < poses.cols(); timestep++)
  {
    mSkeleton->setPositions(poses.col(timestep));
    server.renderSkeleton(mSkeleton);

    server.setObjectPosition("skel_com", coms[timestep]);

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
    server.saveFrame();
  }

  mSkeleton->setPositions(originalPoses);
  mSkeleton->setLinkMasses(originalMasses);

  server.writeFramesJson(path);
}

} // namespace biomechanics
} // namespace dart