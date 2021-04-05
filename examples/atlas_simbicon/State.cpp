/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include "State.hpp"

#include "TerminalCondition.hpp"

// Macro for functions not implemented yet
#define NOT_YET(FUNCTION)                                                      \
  std::cout << #FUNCTION << "Not implemented yet." << std::endl;

using namespace std;

using namespace Eigen;

using namespace dart::dynamics;
using namespace dart::constraint;

//==============================================================================
State::State(SkeletonPtr _skeleton, const std::string& _name)
  : mName(_name),
    mSkeleton(_skeleton),
    mNextState(this),
    mBeginTime(0.0),
    mEndTime(0.0),
    mFrame(0),
    mElapsedTime(0.0),
    mDesiredGlobalSwingLegAngleOnSagital(0.0),
    mDesiredGlobalSwingLegAngleOnCoronal(0.0),
    mDesiredGlobalPelvisAngleOnSagital(0.0),
    mDesiredGlobalPelvisAngleOnCoronal(0.0)
{
  int dof = mSkeleton->getNumDofs();

  mDesiredJointPositions = Eigen::VectorXs::Zero(dof);
  mDesiredJointPositionsBalance = Eigen::VectorXs::Zero(dof);
  mKp = Eigen::VectorXs::Zero(dof);
  mKd = Eigen::VectorXs::Zero(dof);
  mSagitalCd = Eigen::VectorXs::Zero(dof);
  mSagitalCv = Eigen::VectorXs::Zero(dof);
  mCoronalCd = Eigen::VectorXs::Zero(dof);
  mCoronalCv = Eigen::VectorXs::Zero(dof);
  mTorque = Eigen::VectorXs::Zero(dof);

  for (int i = 0; i < dof; ++i)
  {
    mKp[i] = ATLAS_DEFAULT_KP;
    mKd[i] = ATLAS_DEFAULT_KD;
  }

  mPelvis = mSkeleton->getBodyNode("pelvis");
  mLeftFoot = mSkeleton->getBodyNode("l_foot");
  mRightFoot = mSkeleton->getBodyNode("r_foot");
  mLeftThigh = mSkeleton->getBodyNode("l_uleg");
  mRightThigh = mSkeleton->getBodyNode("r_uleg");
  mStanceFoot = nullptr;

  assert(mPelvis != nullptr);
  assert(mLeftFoot != nullptr);
  assert(mRightFoot != nullptr);
  assert(mLeftThigh != nullptr);
  assert(mRightThigh != nullptr);
  //  assert(mStanceFoot != nullptr);

  mCoronalLeftHip = mSkeleton->getDof("l_leg_hpx")->getIndexInSkeleton();  // 10
  mCoronalRightHip = mSkeleton->getDof("r_leg_hpx")->getIndexInSkeleton(); // 11
  mSagitalLeftHip = mSkeleton->getDof("l_leg_hpy")->getIndexInSkeleton();  // 13
  mSagitalRightHip = mSkeleton->getDof("r_leg_hpy")->getIndexInSkeleton(); // 14
}

//==============================================================================
State::~State()
{
}

//==============================================================================
void State::setName(string& _name)
{
  mName = _name;
}

//==============================================================================
const string& State::getName() const
{
  return mName;
}

//==============================================================================
void State::setNextState(State* _nextState)
{
  mNextState = _nextState;
}

//==============================================================================
void State::setTerminalCondition(TerminalCondition* _condition)
{
  assert(_condition != nullptr);

  mTerminalCondition = _condition;
}

//==============================================================================
void State::begin(s_t _currentTime)
{
  mBeginTime = _currentTime;
  mFrame = 0;
  mElapsedTime = 0.0;
}

//==============================================================================
void State::computeControlForce(s_t _timestep)
{
  assert(mNextState != nullptr && "Next state should be set.");

  int dof = mSkeleton->getNumDofs();
  VectorXs q = mSkeleton->getPositions();
  VectorXs dq = mSkeleton->getVelocities();

  // Compute relative joint angles from desired global angles of the pelvis and
  // the swing leg

  // Update desired joint angles with balance feedback. Equation (1) in the
  // paper
  mDesiredJointPositionsBalance = mDesiredJointPositions
                                  + getSagitalCOMDistance() * mSagitalCd
                                  + getSagitalCOMVelocity() * mSagitalCv
                                  + getCoronalCOMDistance() * mCoronalCd
                                  + getCoronalCOMVelocity() * mCoronalCv;

  //  cout << "Sagital D: " << getSagitalCOMDistance() << endl;
  //  cout << "Sagital V: " << getSagitalCOMVelocity() << endl;
  //  cout << endl;
  //  cout << "Coronal D: " << getCoronalCOMDistance() << endl;
  //  cout << "Coronal V: " << getCoronalCOMVelocity() << endl;
  //  cout << endl;

  //  cout << "Sagital left thigh : " << DART_DEGREE * getSagitalLeftLegAngle()
  //  << endl; cout << "Sagital right thigh: " << DART_DEGREE *
  //  getSagitalRightLegAngle() << endl; cout << endl; cout << "Coronal left
  //  thigh : " << DART_DEGREE * getCoronalLeftLegAngle() << endl; cout <<
  //  "Coronal right thigh: " << DART_DEGREE * getCoronalRightLegAngle() <<
  //  endl; cout << endl;

  //  cout << "Sagital pelvis: " << DART_DEGREE * getSagitalPelvisAngle() <<
  //  endl; cout << "Coronal pelvis: " << DART_DEGREE * getCoronalPelvisAngle()
  //  << endl; cout << endl;

  // Compute torques for all the joints except for hip (standing and swing)
  // joints. The first 6 dof is for base body force so it is set to zero.
  mTorque.head<6>() = Vector6s::Zero();
  for (int i = 6; i < dof; ++i)
  {
    mTorque[i]
        = -mKp[i] * (q[i] - mDesiredJointPositionsBalance[i]) - mKd[i] * dq[i];
  }
  //  cout << "q: " << q.transpose() << endl;
  //  cout << "dq: " << dq.transpose() << endl;
  //  cout << "mKp: " << mKp.transpose() << endl;
  //  cout << "mKd: " << mKd.transpose() << endl;
  //  cout << "mTorque: " << mTorque.transpose() << endl;
  //  cout << "Theta_d: " << mDesiredJointPositionsBalance.transpose() << endl;

  // Torso and swing-hip control
  _updateTorqueForStanceLeg();

  // Apply control torque to the skeleton
  mSkeleton->setForces(mTorque);

  mElapsedTime += _timestep;
  mFrame++;
}

//==============================================================================
bool State::isTerminalConditionSatisfied() const
{
  assert(mTerminalCondition != nullptr && "Invalid terminal condition.");

  return mTerminalCondition->isSatisfied();
}

//==============================================================================
void State::end(s_t _currentTime)
{
  mEndTime = _currentTime;
}

//==============================================================================
Eigen::Vector3s State::getCOM() const
{
  return mSkeleton->getCOM();
}

//==============================================================================
Eigen::Vector3s State::getCOMVelocity() const
{
  return mSkeleton->getCOMLinearVelocity();
}

//==============================================================================
Eigen::Isometry3s State::getCOMFrame() const
{
  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();

  // Y-axis
  const Eigen::Vector3s yAxis = Eigen::Vector3s::UnitY();

  // X-axis
  Eigen::Vector3s pelvisXAxis = mPelvis->getTransform().linear().col(0);
  const s_t mag = yAxis.dot(pelvisXAxis);
  pelvisXAxis -= mag * yAxis;
  const Eigen::Vector3s xAxis = pelvisXAxis.normalized();

  // Z-axis
  const Eigen::Vector3s zAxis = xAxis.cross(yAxis);

  T.translation() = getCOM();

  T.linear().col(0) = xAxis;
  T.linear().col(1) = yAxis;
  T.linear().col(2) = zAxis;

  return T;
}

//==============================================================================
s_t State::getSagitalCOMDistance()
{
  Eigen::Vector3s xAxis = getCOMFrame().linear().col(0); // x-axis
  Eigen::Vector3s d = getCOM() - getStanceAnklePosition();

  return d.dot(xAxis);
}

//==============================================================================
s_t State::getSagitalCOMVelocity()
{
  Eigen::Vector3s xAxis = getCOMFrame().linear().col(0); // x-axis
  Eigen::Vector3s v = getCOMVelocity();

  return v.dot(xAxis);
}

//==============================================================================
s_t State::getCoronalCOMDistance()
{
  Eigen::Vector3s yAxis = getCOMFrame().linear().col(2); // z-axis
  Eigen::Vector3s d = getCOM() - getStanceAnklePosition();

  return d.dot(yAxis);
}

//==============================================================================
s_t State::getCoronalCOMVelocity()
{
  Eigen::Vector3s yAxis = getCOMFrame().linear().col(2); // z-axis
  Eigen::Vector3s v = getCOMVelocity();

  return v.dot(yAxis);
}

//==============================================================================
Eigen::Vector3s State::getStanceAnklePosition() const
{
  if (mStanceFoot == nullptr)
    return getCOM();
  else
    return _getJointPosition(mStanceFoot);
}

//==============================================================================
Eigen::Vector3s State::getLeftAnklePosition() const
{
  return _getJointPosition(mLeftFoot);
}

//==============================================================================
Eigen::Vector3s State::getRightAnklePosition() const
{
  return _getJointPosition(mRightFoot);
}

//==============================================================================
s_t State::getSagitalPelvisAngle() const
{
  Matrix3s comR = getCOMFrame().linear();
  Vector3s comY = comR.col(1);

  Vector3s pelvisZ = mPelvis->getTransform().linear().col(2);
  Vector3s projPelvisZ = (comR.transpose() * pelvisZ);
  projPelvisZ[2] = 0.0;
  projPelvisZ.normalize();
  s_t angle = _getAngleBetweenTwoVectors(projPelvisZ, comY);

  Vector3s cross = comY.cross(projPelvisZ);

  if (cross[2] > 0.0)
    return angle;
  else
    return -angle;
}

//==============================================================================
s_t State::getCoronalPelvisAngle() const
{
  Matrix3s comR = getCOMFrame().linear();
  Vector3s comY = comR.col(1);
  Vector3s pelvisZ = mPelvis->getTransform().linear().col(2);
  Vector3s projPelvisZ = (comR.transpose() * pelvisZ);
  projPelvisZ[0] = 0.0;
  projPelvisZ.normalize();
  s_t angle = _getAngleBetweenTwoVectors(projPelvisZ, comY);

  Vector3s cross = comY.cross(projPelvisZ);

  if (cross[0] > 0.0)
    return angle;
  else
    return -angle;
}

//==============================================================================
s_t State::getSagitalLeftLegAngle() const
{
  Matrix3s comR = getCOMFrame().linear();
  Vector3s comY = comR.col(1);
  Vector3s thighAxisZ = mLeftThigh->getTransform().linear().col(2);
  Vector3s projThighAZ = (comR.transpose() * thighAxisZ);
  projThighAZ[2] = 0.0;
  projThighAZ.normalize();
  s_t angle = _getAngleBetweenTwoVectors(projThighAZ, comY);

  Vector3s cross = comY.cross(projThighAZ);

  if (cross[2] > 0.0)
    return angle;
  else
    return -angle;
}

//==============================================================================
s_t State::getSagitalRightLegAngle() const
{
  Matrix3s comR = getCOMFrame().linear();
  Vector3s comY = comR.col(1);
  Vector3s thighAxisZ = mRightThigh->getTransform().linear().col(2);
  Vector3s projThighAZ = (comR.transpose() * thighAxisZ);
  projThighAZ[2] = 0.0;
  projThighAZ.normalize();
  s_t angle = _getAngleBetweenTwoVectors(projThighAZ, comY);

  Vector3s cross = comY.cross(projThighAZ);

  if (cross[2] > 0.0)
    return angle;
  else
    return -angle;
}

//==============================================================================
s_t State::getCoronalLeftLegAngle() const
{
  Matrix3s comR = getCOMFrame().linear();
  Vector3s comY = comR.col(1);
  Vector3s thighAxisZ = mLeftThigh->getTransform().linear().col(2);
  Vector3s projThighAZ = (comR.transpose() * thighAxisZ);
  projThighAZ[0] = 0.0;
  projThighAZ.normalize();
  s_t angle = _getAngleBetweenTwoVectors(projThighAZ, comY);

  Vector3s cross = comY.cross(projThighAZ);

  if (cross[0] > 0.0)
    return angle;
  else
    return -angle;
}

//==============================================================================
s_t State::getCoronalRightLegAngle() const
{
  Matrix3s comR = getCOMFrame().linear();
  Vector3s comY = comR.col(1);
  Vector3s thighAxisZ = mRightThigh->getTransform().linear().col(2);
  Vector3s projThighAZ = (comR.transpose() * thighAxisZ);
  projThighAZ[0] = 0.0;
  projThighAZ.normalize();
  s_t angle = _getAngleBetweenTwoVectors(projThighAZ, comY);

  Vector3s cross = comY.cross(projThighAZ);

  if (cross[0] > 0.0)
    return angle;
  else
    return -angle;
}

//==============================================================================
Eigen::Vector3s State::_getJointPosition(BodyNode* _bodyNode) const
{
  Joint* parentJoint = _bodyNode->getParentJoint();
  Eigen::Vector3s localJointPosition
      = parentJoint->getTransformFromChildBodyNode().translation();
  return _bodyNode->getTransform() * localJointPosition;
}

//==============================================================================
s_t State::_getAngleBetweenTwoVectors(
    const Eigen::Vector3s& _v1, const Eigen::Vector3s& _v2) const
{
  return std::acos(_v1.dot(_v2) / (_v1.norm() * _v2.norm()));
}

//==============================================================================
void State::_updateTorqueForStanceLeg()
{
  // Stance leg is left leg
  if (mStanceFoot == mLeftFoot)
  {
    //    std::cout << "Sagital Pelvis Angle: " << DART_DEGREE *
    //    getSagitalPelvisAngle() << std::endl;

    // Torso control on sagital plane
    s_t pelvisSagitalAngle = getSagitalPelvisAngle();
    s_t tauTorsoSagital
        = -5000.0 * (pelvisSagitalAngle + mDesiredGlobalPelvisAngleOnSagital)
          - 1.0 * (0);
    mTorque[mSagitalLeftHip] = tauTorsoSagital - mTorque[mSagitalRightHip];

    //    cout << "Torque[mSagitalLeftHip]     : " << mTorque[mSagitalLeftHip]
    //    << endl; cout << "Torque[mSagitalRightHip]     : " <<
    //    mTorque[mSagitalRightHip] << endl; cout << "tauTorsoSagital: " <<
    //    tauTorsoSagital << endl; cout << endl;

    // Torso control on coronal plane
    s_t pelvisCoronalAngle = getCoronalPelvisAngle();
    s_t tauTorsoCoronal
        = -5000.0 * (pelvisCoronalAngle - mDesiredGlobalPelvisAngleOnCoronal)
          - 1.0 * (0);
    mTorque[mCoronalLeftHip] = -tauTorsoCoronal - mTorque[mCoronalRightHip];

    //    cout << "Torque[mCoronalLeftHip]     : " << mTorque[mCoronalLeftHip]
    //    << endl; cout << "Torque[mCoronalRightHip]     : " <<
    //    mTorque[mCoronalRightHip] << endl; cout << "tauTorsoCoronal: " <<
    //    tauTorsoCoronal << endl; cout << endl;

    //    cout << "Stance foot: Left foot" << endl;
  }
  // Stance leg is right leg
  else if (mStanceFoot == mRightFoot)
  {
    //    cout << "Stance foot: Right foot" << endl;

    // Torso control on sagital plane
    s_t pelvisSagitalAngle = getSagitalPelvisAngle();
    s_t tauTorsoSagital
        = -5000.0 * (pelvisSagitalAngle + mDesiredGlobalPelvisAngleOnSagital)
          - 1.0 * (0);
    mTorque[mSagitalRightHip] = tauTorsoSagital - mTorque[mSagitalLeftHip];

    //    cout << "Torque[mSagitalLeftHip]     : " << mTorque[mSagitalLeftHip]
    //    << endl; cout << "Torque[mSagitalRightHip]    : " <<
    //    mTorque[mSagitalRightHip] << endl; cout << "tauTorsoSagital: " <<
    //    tauTorsoSagital << endl; cout << endl;

    // Torso control on coronal plane
    s_t pelvisCoronalAngle = getCoronalPelvisAngle();
    s_t tauTorsoCoronal
        = -5000.0 * (pelvisCoronalAngle - mDesiredGlobalPelvisAngleOnCoronal)
          - 1.0 * (0);
    mTorque[mCoronalRightHip] = -tauTorsoCoronal - mTorque[mCoronalLeftHip];

    //    cout << "Torque[mCoronalLeftHip]     : " << mTorque[mCoronalLeftHip]
    //    << endl; cout << "Torque[mCoronalRightHip]     : " <<
    //    mTorque[mCoronalRightHip] << endl; cout << "tauTorsoCoronal: " <<
    //    tauTorsoCoronal << endl; cout << endl;
  }
  else
  {
    // No foot is toching the ground
  }
}

//==============================================================================
State* State::getNextState() const
{
  return mNextState;
}

//==============================================================================
s_t State::getElapsedTime() const
{
  return mElapsedTime;
}

//==============================================================================
void State::setDesiredJointPosition(const string& _jointName, s_t _val)
{
  std::size_t index = mSkeleton->getDof(_jointName)->getIndexInSkeleton();
  mDesiredJointPositions[index] = _val;
}

//==============================================================================
s_t State::getDesiredJointPosition(int _idx) const
{
  assert(
      0 <= _idx && _idx <= mDesiredJointPositions.size()
      && "Invalid joint index.");

  return mDesiredJointPositions[_idx];
}

//==============================================================================
s_t State::getDesiredJointPosition(const string& _jointName) const
{
  // TODO(JS)
  NOT_YET(State::getDesiredJointPosition());

  assert(mSkeleton->getJoint(_jointName) != nullptr);

  return mDesiredJointPositions[mJointMap.at(_jointName)];
}

//==============================================================================
void State::setDesiredSwingLegGlobalAngleOnSagital(s_t _val)
{
  mDesiredGlobalSwingLegAngleOnSagital = _val;
}

//==============================================================================
void State::setDesiredSwingLegGlobalAngleOnCoronal(s_t _val)
{
  mDesiredGlobalSwingLegAngleOnCoronal = _val;
}

//==============================================================================
void State::setDesiredPelvisGlobalAngleOnSagital(s_t _val)
{
  mDesiredGlobalPelvisAngleOnSagital = _val;
}

//==============================================================================
void State::setDesiredPelvisGlobalAngleOnCoronal(s_t _val)
{
  mDesiredGlobalPelvisAngleOnCoronal = _val;
}

//==============================================================================
void State::setProportionalGain(int _idx, s_t _val)
{
  assert(0 <= _idx && _idx <= mKp.size() && "Invalid joint index.");

  mKd[_idx] = _val;
}

//==============================================================================
void State::setProportionalGain(const string& /*_jointName*/, s_t /*_val*/)
{
  // TODO(JS)
  NOT_YET(State::setProportionalGain());
}

//==============================================================================
s_t State::getProportionalGain(int _idx) const
{
  assert(0 <= _idx && _idx <= mKp.size() && "Invalid joint index.");

  return mKp[_idx];
}

//==============================================================================
s_t State::getProportionalGain(const string& _jointName) const
{
  // TODO(JS)
  NOT_YET(State::getProportionalGain());

  assert(mSkeleton->getJoint(_jointName) != nullptr);

  return mKp[mJointMap.at(_jointName)];
}

//==============================================================================
void State::setDerivativeGain(int _idx, s_t _val)
{
  assert(0 <= _idx && _idx <= mKd.size() && "Invalid joint index.");

  mKd[_idx] = _val;
}

//==============================================================================
void State::setDerivativeGain(const string& /*_jointName*/, s_t /*_val*/)
{
  // TODO(JS)
  NOT_YET(State::setDerivativeGain());
}

//==============================================================================
s_t State::getDerivativeGain(int _idx) const
{
  assert(0 <= _idx && _idx <= mKd.size() && "Invalid joint index.");

  return mKd[_idx];
}

//==============================================================================
// s_t State::getDerivativeGain(const string& _jointName) const
//{
//  // TODO(JS)
//  NOT_YET(State::getDerivativeGain());
//}

//==============================================================================
void State::setFeedbackSagitalCOMDistance(std::size_t _index, s_t _val)
{
  assert(static_cast<int>(_index) <= mSagitalCd.size() && "Invalid index.");

  mSagitalCd[_index] = _val;
}

//==============================================================================
void State::setFeedbackSagitalCOMVelocity(std::size_t _index, s_t _val)
{
  assert(static_cast<int>(_index) <= mSagitalCv.size() && "Invalid index.");

  mSagitalCv[_index] = _val;
}

//==============================================================================
void State::setFeedbackCoronalCOMDistance(std::size_t _index, s_t _val)
{
  assert(static_cast<int>(_index) <= mCoronalCd.size() && "Invalid index.");

  mCoronalCd[_index] = _val;
}

//==============================================================================
void State::setFeedbackCoronalCOMVelocity(std::size_t _index, s_t _val)
{
  assert(static_cast<int>(_index) <= mCoronalCv.size() && "Invalid index.");

  mCoronalCv[_index] = _val;
}

//==============================================================================
void State::setStanceFootToLeftFoot()
{
  mStanceFoot = mLeftFoot;
}

//==============================================================================
void State::setStanceFootToRightFoot()
{
  mStanceFoot = mRightFoot;
}
