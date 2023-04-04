#include "dart/sensors/FilterState.hpp"

#include "dart/dynamics/Joint.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace sensors {

FilterState::FilterState()
{
}

/// This is a list of names of the joints we will be tracking in the state.
/// Any joints not in this list will not be tracked.
void FilterState::setIncludedJoints(std::vector<std::string> joints)
{
  mIncludedJoints = joints;
}

/// If true, we include the acceleration of the joints in the state.
void FilterState::setIncludeAcceleration(bool useAcc)
{
  mUseAcceleration = useAcc;
}

/// This returns the joints states that are in our `mIncludedJoints` list.
Eigen::VectorXs FilterState::getState(std::shared_ptr<dynamics::Skeleton> skel)
{
  std::vector<dynamics::Joint*> joints;
  int dofs = 0;
  for (std::string jointName : mIncludedJoints)
  {
    auto* joint = skel->getJoint(jointName);
    if (joint != nullptr)
    {
      joints.push_back(joint);
      dofs += joint->getNumDofs();
    }
  }

  int stateSize = dofs * 2;
  if (mUseAcceleration)
    stateSize += dofs;

  Eigen::VectorXs state = Eigen::VectorXs::Zero(stateSize);
  int cursor = 0;
  for (auto* joint : joints)
  {
    Eigen::VectorXs pos = joint->getPositions();
    Eigen::VectorXs vel = joint->getVelocities();
    Eigen::VectorXs acc = joint->getAccelerations();

    state.segment(cursor, pos.size()) = pos;
    cursor += pos.size();
    state.segment(cursor, vel.size()) = vel;
    cursor += vel.size();
    if (mUseAcceleration)
    {
      state.segment(cursor, acc.size()) = acc;
      cursor += acc.size();
    }
  }
  return state;
}

/// This sets the joints states that are in our `mIncludedJoints` list.
void FilterState::setState(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x)
{
  std::vector<dynamics::Joint*> joints;
  for (std::string jointName : mIncludedJoints)
  {
    auto* joint = skel->getJoint(jointName);
    if (joint != nullptr)
    {
      joints.push_back(joint);
    }
  }

  int cursor = 0;
  for (auto* joint : joints)
  {
    int dofs = joint->getNumDofs();
    joint->setPositions(x.segment(cursor, dofs));
    cursor += dofs;
    joint->setVelocities(x.segment(cursor, dofs));
    cursor += dofs;

    if (mUseAcceleration)
    {
      joint->setAccelerations(x.segment(cursor, dofs));
      cursor += dofs;
    }
  }
}

/// This method will return the predicted outputs of all the sensors in the
/// set, concatenated together.
Eigen::VectorXs FilterState::observationFunction(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x)
{
  setState(skel, x);

  int totalOutputDim = 0;
  for (auto& sensor : mSensors)
  {
    totalOutputDim += sensor->outputDim();
  }

  Eigen::VectorXs y = Eigen::VectorXs::Zero(totalOutputDim);
  int cursor = 0;
  for (auto& sensor : mSensors)
  {
    y.segment(cursor, sensor->outputDim())
        = sensor->observationFunction(skel, Eigen::VectorXs::Zero());
    cursor += sensor->outputDim();
  }
  return y;
}

/// This method will compute the total Jacobian of the observation function
Eigen::MatrixXs FilterState::observationJacobian(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x)
{
  std::vector<dynamics::Joint*> joints;
  for (std::string jointName : mIncludedJoints)
  {
    auto* joint = skel->getJoint(jointName);
    if (joint != nullptr)
    {
      joints.push_back(joint);
    }
  }

  int totalOutputDim = 0;
  for (auto& sensor : mSensors)
  {
    totalOutputDim += sensor->outputDim();
  }

  Eigen::MatrixXs J = Eigen::MatrixXs::Zero(totalOutputDim, x.size());

  int rowCursor = 0;
  for (auto& sensor : mSensors)
  {
    int sensorDim = sensor->outputDim();

    Eigen::MatrixXs sensorWrtPos = sensor->observationJacobianWrt(
        skel, Eigen::VectorXs::Zero(), neural::WithRespectTo::POSITION);
    Eigen::MatrixXs sensorWrtVel = sensor->observationJacobianWrt(
        skel, Eigen::VectorXs::Zero(), neural::WithRespectTo::VELOCITY);
    Eigen::MatrixXs sensorWrtAcc = Eigen::MatrixXs::Zero(0, 0);
    if (mUseAcceleration)
    {
      sensorWrtAcc = sensor->observationJacobianWrt(
          skel, Eigen::VectorXs::Zero(), neural::WithRespectTo::ACCELERATION);
    }

    int colCursor = 0;
    for (auto* joint : joints)
    {
      int dofs = joint->getNumDofs();
      int start = joint->getIndexInSkeleton(0);
      J.block(rowCursor, colCursor, sensorDim, dofs)
          = sensorWrtPos.block(0, start, sensorDim, dofs);
      colCursor += dofs;
      J.block(rowCursor, colCursor, sensorDim, dofs)
          = sensorWrtVel.block(0, start, sensorDim, dofs);
      colCursor += dofs;
      if (mUseAcceleration)
      {
        J.block(rowCursor, colCursor, sensorDim, dofs)
            = sensorWrtAcc.block(0, start, sensorDim, dofs);
        colCursor += dofs;
      }
    }

    rowCursor += sensorDim;
  }

  return J;
}

/// This method will compute the transition function on the given skeleton,
/// and return the new state.
Eigen::VectorXs FilterState::transitionFunction(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x, s_t dt)
{
  setState(skel, x);
  skel->integrateVelocities(dt);
  skel->integratePositions(dt);
  return getState(skel);
}

/// This method will compute the Jacobian of the transition function
Eigen::VectorXs FilterState::transitionJacobian(
    std::shared_ptr<dynamics::Skeleton> skel, Eigen::VectorXs x, s_t dt)
{
  setState(skel, x);
  Eigen::VectorXs pos = skel->getPositions();
  Eigen::VectorXs vel = skel->getVelocities();
  Eigen::VectorXs acc = skel->getAccelerations();

  Eigen::MatrixXs posPos = skel->getPosPosJac(pos, vel, dt);
  Eigen::MatrixXs velPos = skel->getVelPosJac(pos, vel, dt);

  std::vector<dynamics::Joint*> joints;
  int dofs = 0;
  for (std::string jointName : mIncludedJoints)
  {
    auto* joint = skel->getJoint(jointName);
    if (joint != nullptr)
    {
      joints.push_back(joint);
      dofs += joint->getNumDofs();
    }
  }

  int stateSize = dofs * 2;
  if (mUseAcceleration)
    stateSize += dofs;

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(stateSize, stateSize);

  // TODO: Pick up here, WIP
  assert(false && "Unimplented");

  return result;
}

/// This adds a sensor to the list
void FilterState::addSensor(std::shared_ptr<Sensor> sensor)
{
  mSensors.push_back(sensor);
}

} // namespace sensors
} // namespace dart