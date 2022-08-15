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

#include "dart/dynamics/Inertia.hpp"

#include "dart/common/Console.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
Inertia::Inertia(
    s_t _mass,
    const Eigen::Vector3s& _com,
    const Eigen::Matrix3s& _momentOfInertia)
  : mMass(_mass), mCenterOfMass(_com)
{
  setMoment(_momentOfInertia);

  // Default bounds
  mCenterOfMassLowerBound << -5, -5, -5;
  mCenterOfMassUpperBound << 5, 5, 5;
  mMassLowerBound = 0.01;
  mMassUpperBound = 20;
  mMomentLowerBound << 0.1, 0.1, 0.1, -1, -1, -1;
  mMomentUpperBound << 1, 1, 1, 1, 1, 1;
}

//==============================================================================
Inertia::Inertia(const Eigen::Matrix6s& _spatialInertiaTensor)
{
  setSpatialTensor(_spatialInertiaTensor);

  // Default bounds
  mCenterOfMassLowerBound << -5, -5, -5;
  mCenterOfMassUpperBound << 5, 5, 5;
  mMassLowerBound = 0.01;
  mMassUpperBound = 20;
  mMomentLowerBound << 0.1, 0.1, 0.1, -1, -1, -1;
  mMomentUpperBound << 1, 1, 1, 1, 1, 1;
}

//==============================================================================
Inertia::Inertia(
    s_t _mass,
    s_t _comX,
    s_t _comY,
    s_t _comZ,
    s_t _Ixx,
    s_t _Iyy,
    s_t _Izz,
    s_t _Ixy,
    s_t _Ixz,
    s_t _Iyz)
  : mMass(_mass),
    mCenterOfMass(_comX, _comY, _comZ),
    mMoment({_Ixx, _Iyy, _Izz, _Ixy, _Ixz, _Iyz})
{
  computeSpatialTensor();

  // Default bounds
  mCenterOfMassLowerBound << -5, -5, -5;
  mCenterOfMassUpperBound << 5, 5, 5;
  mMassLowerBound = 0.01;
  mMassUpperBound = 20;
  mMomentLowerBound << 0.1, 0.1, 0.1, -1, -1, -1;
  mMomentUpperBound << 1, 1, 1, 1, 1, 1;
}

//==============================================================================
void Inertia::setParameter(Param _param, s_t _value)
{
  if (_param == MASS)
  {
    mMass = _value;
  }
  else if (_param <= COM_Z)
  {
    mCenterOfMass[_param - 1] = _value;
  }
  else if (_param <= I_YZ)
  {
    mCenterOfMass[_param - 4] = _value;
  }
  else
  {
    dtwarn << "[Inertia::setParameter] Attempting to set Param #" << _param
           << ", but inertial parameters only go up to " << I_YZ
           << ". Nothing will be set.\n";
    return;
  }

  computeSpatialTensor();
}

//==============================================================================
s_t Inertia::getParameter(Param _param) const
{
  if (_param == MASS)
    return mMass;
  else if (_param <= COM_Z)
    return mCenterOfMass[_param - 1];
  else if (_param <= I_YZ)
    return mMoment[_param - 4];

  dtwarn << "[Inertia::getParameter] Requested Param #" << _param
         << ", but inertial parameters only go up to " << I_YZ
         << ". Returning 0\n";

  return 0;
}

//==============================================================================
void Inertia::setMass(s_t _mass)
{
  mMass = _mass;
  computeSpatialTensor();
}

//==============================================================================
s_t Inertia::getMass() const
{
  return mMass;
}

//==============================================================================
void Inertia::setMassLowerBound(s_t _mass)
{
  mMassLowerBound = _mass;
}

//==============================================================================
s_t Inertia::getMassLowerBound() const
{
  return mMassLowerBound;
}

//==============================================================================
void Inertia::setMassUpperBound(s_t _mass)
{
  mMassUpperBound = _mass;
}

//==============================================================================
s_t Inertia::getMassUpperBound() const
{
  return mMassUpperBound;
}

//==============================================================================
void Inertia::setLocalCOM(const Eigen::Vector3s& _com)
{
  mCenterOfMass = _com;
  computeSpatialTensor();
}

//==============================================================================
const Eigen::Vector3s& Inertia::getLocalCOM() const
{
  return mCenterOfMass;
}

//==============================================================================
void Inertia::setLocalCOMLowerBound(Eigen::Vector3s bounds)
{
  mCenterOfMassLowerBound = bounds;
}

//==============================================================================
const Eigen::Vector3s& Inertia::getLocalCOMLowerBound() const
{
  return mCenterOfMassLowerBound;
}

//==============================================================================
void Inertia::setLocalCOMUpperBound(Eigen::Vector3s bounds)
{
  mCenterOfMassUpperBound = bounds;
}

//==============================================================================
const Eigen::Vector3s& Inertia::getLocalCOMUpperBound() const
{
  return mCenterOfMassUpperBound;
}

//==============================================================================
void Inertia::setMoment(const Eigen::Matrix3s& _moment)
{
  if (!verifyMoment(_moment, true))
    dtwarn << "[Inertia::setMoment] Passing in an invalid moment of inertia "
           << "matrix. Results might not by physically accurate or "
           << "meaningful.\n";

  for (std::size_t i = 0; i < 3; ++i)
    mMoment[i] = _moment(i, i);

  mMoment[I_XY - 4] = _moment(0, 1);
  mMoment[I_XZ - 4] = _moment(0, 2);
  mMoment[I_YZ - 4] = _moment(1, 2);

  computeSpatialTensor();
}

//==============================================================================
void Inertia::setMoment(
    s_t _Ixx, s_t _Iyy, s_t _Izz, s_t _Ixy, s_t _Ixz, s_t _Iyz)
{
  mMoment[I_XX - 4] = _Ixx;
  mMoment[I_YY - 4] = _Iyy;
  mMoment[I_ZZ - 4] = _Izz;
  mMoment[I_XY - 4] = _Ixy;
  mMoment[I_XZ - 4] = _Ixz;
  mMoment[I_YZ - 4] = _Iyz;

  computeSpatialTensor();
}

//==============================================================================
Eigen::Matrix3s Inertia::getMoment() const
{
  Eigen::Matrix3s I;
  for (int i = 0; i < 3; ++i)
    I(i, i) = mMoment[i];

  I(0, 1) = I(1, 0) = mMoment[I_XY - 4];
  I(0, 2) = I(2, 0) = mMoment[I_XZ - 4];
  I(1, 2) = I(2, 1) = mMoment[I_YZ - 4];

  return I;
}

//==============================================================================
void Inertia::setMomentVector(Eigen::Vector6s moment)
{
  mMoment[I_XX - 4] = moment(0);
  mMoment[I_YY - 4] = moment(1);
  mMoment[I_ZZ - 4] = moment(2);
  mMoment[I_XY - 4] = moment(3);
  mMoment[I_XZ - 4] = moment(4);
  mMoment[I_YZ - 4] = moment(5);

  computeSpatialTensor();
}

//==============================================================================
const Eigen::Vector6s Inertia::getMomentVector() const
{
  Eigen::Vector6s vec;
  vec << mMoment[I_XX - 4], mMoment[I_YY - 4], mMoment[I_ZZ - 4],
      mMoment[I_XY - 4], mMoment[I_XZ - 4], mMoment[I_YZ - 4];
  return vec;
}

//==============================================================================
void Inertia::setMomentLowerBound(Eigen::Vector6s bound)
{
  mMomentLowerBound = bound;
}

//==============================================================================
const Eigen::Vector6s& Inertia::getMomentLowerBound() const
{
  return mMomentLowerBound;
}

//==============================================================================
void Inertia::setMomentUpperBound(Eigen::Vector6s bound)
{
  mMomentUpperBound = bound;
}

//==============================================================================
const Eigen::Vector6s& Inertia::getMomentUpperBound() const
{
  return mMomentUpperBound;
}

//==============================================================================
void Inertia::setSpatialTensor(const Eigen::Matrix6s& _spatial)
{
  if (!verifySpatialTensor(_spatial, true))
    dtwarn << "[Inertia::setSpatialTensor] Passing in an invalid spatial "
           << "inertia tensor. Results might not be physically accurate or "
           << "meaningful.\n";

  mSpatialTensor = _spatial;
  computeParameters();
}

//==============================================================================
const Eigen::Matrix6s& Inertia::getSpatialTensor() const
{
  return mSpatialTensor;
}

//==============================================================================
bool Inertia::verifyMoment(
    const Eigen::Matrix3s& _moment, bool _printWarnings, s_t _tolerance)
{
  bool valid = true;
  for (int i = 0; i < 3; ++i)
  {
    if (_moment(i, i) <= 0)
    {
      valid = false;
      if (_printWarnings)
      {
        dtwarn << "[Inertia::verifyMoment] Invalid entry for (" << i << "," << i
               << "): " << _moment(i, i) << ". Value should be positive "
               << "and greater than zero.\n";
      }
    }
  }

  for (int i = 0; i < 3; ++i)
  {
    for (int j = i + 1; j < 3; ++j)
    {
      if (abs(_moment(i, j) - _moment(j, i)) > _tolerance)
      {
        valid = false;
        if (_printWarnings)
        {
          dtwarn << "[Inertia::verifyMoment] Values for entries (" << i << ","
                 << j << ") and (" << j << "," << i << ") differ by "
                 << _moment(i, j) - _moment(j, i) << " which is more than the "
                 << "permitted tolerance (" << _tolerance << ")\n";
        }
      }
    }
  }

  return valid;
}

//==============================================================================
bool Inertia::verifySpatialTensor(
    const Eigen::Matrix6s& _spatial, bool _printWarnings, s_t _tolerance)
{

  bool valid = true;

  for (std::size_t i = 0; i < 6; ++i)
  {
    if (_spatial(i, i) <= 0)
    {
      valid = false;
      if (_printWarnings)
      {
        std::string component = i < 3 ? "moment of inertia diagonal" : "mass";
        dtwarn << "[Inertia::verifySpatialTensor] Invalid entry for (" << i
               << "," << i << "): " << _spatial(i, i) << ". Value should be "
               << "positive and greater than zero because it corresponds to "
               << component << ".\n";
      }
    }
  }

  // Off-diagonals of top left block
  for (std::size_t i = 0; i < 3; ++i)
  {
    for (std::size_t j = i + 1; j < 3; ++j)
    {
      if (abs(_spatial(i, j) - _spatial(j, i)) > _tolerance)
      {
        valid = false;
        dtwarn << "[Inertia::verifySpatialTensor] Values for entries (" << i
               << "," << j << ") and (" << j << "," << i << ") differ by "
               << _spatial(i, j) - _spatial(j, i) << " which is more than the "
               << "permitted tolerance (" << _tolerance << ")\n";
      }
    }
  }

  // Off-diagonals of bottom right block
  for (std::size_t i = 3; i < 6; ++i)
  {
    for (std::size_t j = i + 1; j < 6; ++j)
    {
      if (_spatial(i, j) != 0)
      {
        valid = false;
        if (_printWarnings)
          dtwarn << "[Inertia::verifySpatialTensor] Invalid entry for (" << i
                 << "," << i << "): " << _spatial(i, j) << ". Value should be "
                 << "exactly zero.\n";
      }

      if (_spatial(j, i) != 0)
      {
        valid = false;
        if (_printWarnings)
          dtwarn << "[Inertia::verifySpatialTensor] Invalid entry for (" << j
                 << "," << i << "): " << _spatial(j, i) << ". Value should be "
                 << "exactly zero.\n";
      }
    }
  }

  // Diagonals of the bottom left and top right blocks
  for (std::size_t k = 0; k < 2; ++k)
  {
    for (std::size_t i = 0; i < 3; ++i)
    {
      std::size_t i1 = k == 0 ? i + 3 : i;
      std::size_t i2 = k == 0 ? i : i + 3;
      if (_spatial(i1, i2) != 0)
      {
        valid = false;
        if (_printWarnings)
          dtwarn << "[Inertia::verifySpatialTensor] Invalid entry for (" << i1
                 << "," << i2 << "): " << _spatial(i1, i2) << ". Value should "
                 << "be exactly zero.\n";
      }
    }
  }

  // Check skew-symmetry in bottom left and top right
  for (std::size_t k = 0; k < 2; ++k)
  {
    for (std::size_t i = 0; i < 3; ++i)
    {
      for (std::size_t j = i + 1; j < 3; ++j)
      {
        std::size_t i1 = k == 0 ? i + 3 : i;
        std::size_t j1 = k == 0 ? j : j + 3;

        std::size_t i2 = k == 0 ? j + 3 : j;
        std::size_t j2 = k == 0 ? i : i + 3;

        if (abs(_spatial(i1, j1) + _spatial(i2, j2)) > _tolerance)
        {
          valid = false;
          if (_printWarnings)
            dtwarn << "[Inertia::verifySpatialTensor] Mismatch between entries "
                   << "(" << i1 << "," << j1 << ") and (" << i2 << "," << j2
                   << "). They should sum to zero, but instead they sum to "
                   << _spatial(i1, j1) + _spatial(i2, j2)
                   << " which is outside "
                   << "of the permitted tolerance (" << _tolerance << ").\n";
        }
      }
    }
  }

  // Check that the bottom left block is the transpose of the top right block
  // Note that we only need to check three of the components from each block,
  // because the last test ensures that both blocks are skew-symmetric
  // themselves
  for (std::size_t i = 0; i < 3; ++i)
  {
    for (std::size_t j = i + 1; j < 3; ++j)
    {
      std::size_t i1 = i;
      std::size_t j1 = j + 3;

      std::size_t i2 = j1;
      std::size_t j2 = i1;

      if (abs(_spatial(i1, j1) - _spatial(i2, j2)) > _tolerance)
      {
        valid = false;
        if (_printWarnings)
          dtwarn << "[Inertia::verifySpatialTensor] Values for  entries "
                 << "(" << i1 << "," << j1 << ") and (" << i2 << "," << j2
                 << ") "
                 << "differ by " << _spatial(i1, j1) - _spatial(i1, j2)
                 << " which is more than the permitted tolerance ("
                 << _tolerance << "). "
                 << "The bottom-left block should be the transpose of the "
                 << "top-right block.\n";
      }
    }
  }

  return valid;
}

//==============================================================================
bool Inertia::verify(bool _printWarnings, s_t _tolerance) const
{
  return verifySpatialTensor(getSpatialTensor(), _printWarnings, _tolerance);
}

//==============================================================================
bool Inertia::operator==(const Inertia& other) const
{
  return (other.mSpatialTensor == mSpatialTensor);
}

//==============================================================================
/// This rescales the object by "ratio" in each of the specified axis
void Inertia::rescale(Eigen::Vector3s ratio)
{
  // TODO: fix this!!!
  setLocalCOM(getLocalCOM().cwiseProduct(ratio));
  Eigen::Matrix3s scaledMoment = getMoment().cwiseProduct(
      ratio * ratio.transpose()); // The MOI is an integral of m*r*r, so needs
                                  // to be scaled twice
  setMoment(scaledMoment);
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to the mass
Eigen::Matrix6s Inertia::getSpatialTensorGradientWrtMass()
{
  Eigen::Matrix3s C = math::makeSkewSymmetric(mCenterOfMass);

  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();
  // Top left
  result.block<3, 3>(0, 0) = C * C.transpose();

  // Bottom left
  result.block<3, 3>(3, 0) = C.transpose();

  // Top right
  result.block<3, 3>(0, 3) = C;

  // Bottom right
  result.block<3, 3>(3, 3) = Eigen::Matrix3s::Identity();

  return result;
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to the mass
Eigen::Matrix6s Inertia::finiteDifferenceSpatialTensorGradientWrtMass()
{
  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  s_t originalMass = getMass();

  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& out) {
        setMass(originalMass + eps);
        out = getSpatialTensor();
        return true;
      },
      result,
      1e-3,
      true);

  setMass(originalMass);

  return result;
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to a specific
/// index in the COM vector
Eigen::Matrix6s Inertia::getSpatialTensorGradientWrtCOM(int index)
{
  Eigen::Matrix3s C = math::makeSkewSymmetric(mCenterOfMass);
  Eigen::Matrix3s dC = math::makeSkewSymmetric(Eigen::Vector3s::Unit(index));
  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  // Top left
  result.block<3, 3>(0, 0) = mMass * (dC * C.transpose() + C * dC.transpose());

  // Bottom left
  result.block<3, 3>(3, 0) = mMass * dC.transpose();

  // Top right
  result.block<3, 3>(0, 3) = mMass * dC;

  // Bottom right
  result.block<3, 3>(3, 3).setZero();

  return result;
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to a specific
/// index in the COM vector
Eigen::Matrix6s Inertia::finiteDifferenceSpatialTensorGradientWrtCOM(int index)
{
  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  Eigen::Vector3s originalCOM = getLocalCOM();

  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& out) {
        Eigen::Vector3s perturbed = originalCOM;
        perturbed(index) += eps;
        setLocalCOM(perturbed);
        out = getSpatialTensor();
        return true;
      },
      result,
      1e-3,
      true);

  setLocalCOM(originalCOM);

  return result;
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to a specific
/// index in the moment vector
Eigen::Matrix6s Inertia::getSpatialTensorGradientWrtMomentVector(int index)
{
  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  if (index < 3)
  {
    result(index, index) = 1;
  }
  else if (index == 3)
  {
    result(0, 1) = result(1, 0) = 1;
  }
  else if (index == 4)
  {
    result(0, 2) = result(2, 0) = 1;
  }
  else if (index == 5)
  {
    result(1, 2) = result(2, 1) = 1;
  }

  return result;
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to a specific
/// index in the moment vector
Eigen::Matrix6s Inertia::finiteDifferenceSpatialTensorGradientWrtMomentVector(
    int index)
{
  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  Eigen::Vector6s originalMoment = getMomentVector();

  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& out) {
        Eigen::Vector6s perturbed = originalMoment;
        perturbed(index) += eps;
        setMomentVector(perturbed);
        out = getSpatialTensor();
        return true;
      },
      result,
      1e-3,
      true);

  setMomentVector(originalMoment);

  return result;
}

//==============================================================================
// Note: Taken from Springer Handbook, chapter 2.2.11
void Inertia::computeSpatialTensor()
{
  Eigen::Matrix3s C = math::makeSkewSymmetric(mCenterOfMass);

  // Top left
  mSpatialTensor.block<3, 3>(0, 0) = getMoment() + mMass * C * C.transpose();

  // Bottom left
  mSpatialTensor.block<3, 3>(3, 0) = mMass * C.transpose();

  // Top right
  mSpatialTensor.block<3, 3>(0, 3) = mMass * C;

  // Bottom right
  mSpatialTensor.block<3, 3>(3, 3) = mMass * Eigen::Matrix3s::Identity();
}

//==============================================================================
void Inertia::computeParameters()
{
  mMass = mSpatialTensor(3, 3);
  Eigen::Matrix3s C = mSpatialTensor.block<3, 3>(0, 3) / mMass;
  mCenterOfMass[0] = -C(1, 2);
  mCenterOfMass[1] = C(0, 2);
  mCenterOfMass[2] = -C(0, 1);

  Eigen::Matrix3s I = mSpatialTensor.block<3, 3>(0, 0) + mMass * C * C;
  for (std::size_t i = 0; i < 3; ++i)
    mMoment[i] = I(i, i);

  mMoment[I_XY - 4] = I(0, 1);
  mMoment[I_XZ - 4] = I(0, 2);
  mMoment[I_YZ - 4] = I(1, 2);
}

} // namespace dynamics
} // namespace dart
