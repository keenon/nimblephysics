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

#include <iostream>

#include "dart/common/Console.hpp"
#include "dart/math/AssignmentMatcher.hpp"
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
  : mMass(_mass), mCenterOfMass(_com), mCachedDimsAndEulerDirty(true)
{
  setMoment(_momentOfInertia);

  // Default bounds
  mCenterOfMassLowerBound << -5, -5, -5;
  mCenterOfMassUpperBound << 5, 5, 5;
  mMassLowerBound = 0.01;
  mMassUpperBound = 100;
  mMomentLowerBound << 1e-7, 1e-7, 1e-7, -1e3, -1e3, -1e3;
  mMomentUpperBound << 1e3, 1e3, 1e3, 1e3, 1e3, 1e3;
  mDimsAndEulerLowerBound << 1e-7, 1e-7, 1e-7, -M_PI / 4, -M_PI / 4, -M_PI / 4;
  mDimsAndEulerUpperBound << 1e2, 1e2, 1e2, M_PI / 4, M_PI / 4, M_PI / 4;
}

//==============================================================================
Inertia::Inertia(const Eigen::Matrix6s& _spatialInertiaTensor)
  : mCachedDimsAndEulerDirty(true)
{
  setSpatialTensor(_spatialInertiaTensor);

  // Default bounds
  mCenterOfMassLowerBound << -5, -5, -5;
  mCenterOfMassUpperBound << 5, 5, 5;
  mMassLowerBound = 0.01;
  mMassUpperBound = 100;
  mMomentLowerBound << 1e-7, 1e-7, 1e-7, -1e3, -1e3, -1e3;
  mMomentUpperBound << 1e3, 1e3, 1e3, 1e3, 1e3, 1e3;
  mDimsAndEulerLowerBound << 1e-7, 1e-7, 1e-7, -M_PI / 4, -M_PI / 4, -M_PI / 4;
  mDimsAndEulerUpperBound << 1e2, 1e2, 1e2, M_PI / 4, M_PI / 4, M_PI / 4;
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
    mMoment({_Ixx, _Iyy, _Izz, _Ixy, _Ixz, _Iyz}),
    mCachedDimsAndEulerDirty(true)
{
  computeSpatialTensor();

  // Default bounds
  mCenterOfMassLowerBound << -5, -5, -5;
  mCenterOfMassUpperBound << 5, 5, 5;
  mMassLowerBound = 0.01;
  mMassUpperBound = 100;
  mMomentLowerBound << 1e-7, 1e-7, 1e-7, -1e3, -1e3, -1e3;
  mMomentUpperBound << 1e3, 1e3, 1e3, 1e3, 1e3, 1e3;
  mDimsAndEulerLowerBound << 1e-7, 1e-7, 1e-7, -M_PI / 4, -M_PI / 4, -M_PI / 4;
  mDimsAndEulerUpperBound << 1e2, 1e2, 1e2, M_PI / 4, M_PI / 4, M_PI / 4;
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
void Inertia::setMass(s_t _mass, bool preserveDimsAndEuler)
{
  if (mMass == _mass)
  {
    return;
  }

  Eigen::Vector6s dimsAndEuler = Eigen::Vector6s::Ones() * -1;
  if (preserveDimsAndEuler && mMass > 0
      && getMomentVector() != Eigen::Vector6s::Zero())
  {
    dimsAndEuler = getDimsAndEulerVector();
  }
  mMass = _mass;
  if (preserveDimsAndEuler && dimsAndEuler != Eigen::Vector6s::Ones() * -1)
  {
    bool oldCachedDirty = mCachedDimsAndEulerDirty;
    setMomentVector(computeMomentVector(mMass, dimsAndEuler));
    mCachedDimsAndEulerDirty = oldCachedDirty;
  }

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
  mCachedDimsAndEulerDirty = true;
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
/// Set the dims and eulers (about the center of mass)
void Inertia::setDimsAndEulerVector(Eigen::Vector6s dimsAndEuler)
{
  setMomentVector(Inertia::computeMomentVector(mMass, dimsAndEuler));
  mCachedDimsAndEuler = dimsAndEuler;
  mCachedDimsAndEulerDirty = false;
}

//==============================================================================
const Eigen::Vector6s Inertia::getDimsAndEulerVector() const
{
  if (mCachedDimsAndEulerDirty)
  {
    return Inertia::computeDimsAndEuler(mMass, getMomentVector());
  }
  return mCachedDimsAndEuler;
}

//==============================================================================
/// Set the dims and eulers bounds (about the center of mass)
void Inertia::setDimsAndEulerLowerBound(Eigen::Vector6s bound)
{
  mDimsAndEulerLowerBound = bound;
}

//==============================================================================
const Eigen::Vector6s& Inertia::getDimsAndEulerLowerBound() const
{
  return mDimsAndEulerLowerBound;
}

//==============================================================================
void Inertia::setDimsAndEulerUpperBound(Eigen::Vector6s bound)
{
  mDimsAndEulerUpperBound = bound;
}

//==============================================================================
const Eigen::Vector6s& Inertia::getDimsAndEulerUpperBound() const
{
  return mDimsAndEulerUpperBound;
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
Eigen::Matrix6s Inertia::getSpatialTensorGradientWrtMass(
    bool preserveDimsAndEuler)
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

  if (preserveDimsAndEuler)
  {
    Eigen::Vector6s dimsAndEuler = getDimsAndEulerVector();
    Eigen::Vector6s grad = computeMomentVectorGradWrtMass(mMass, dimsAndEuler);
    s_t I_XX = grad(0);
    s_t I_YY = grad(1);
    s_t I_ZZ = grad(2);
    s_t I_XY = grad(3);
    s_t I_XZ = grad(4);
    s_t I_YZ = grad(5);
    result(0, 0) += I_XX;
    result(1, 1) += I_YY;
    result(2, 2) += I_ZZ;
    result(0, 1) += I_XY;
    result(1, 0) += I_XY;
    result(0, 2) += I_XZ;
    result(2, 0) += I_XZ;
    result(1, 2) += I_YZ;
    result(2, 1) += I_YZ;
  }

  return result;
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to the mass
Eigen::Matrix6s Inertia::finiteDifferenceSpatialTensorGradientWrtMass(
    bool preserveDimsAndEuler)
{
  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  s_t originalMass = getMass();

  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& out) {
        setMass(originalMass + eps, preserveDimsAndEuler);
        out = getSpatialTensor();
        return true;
      },
      result,
      1e-3,
      true);

  setMass(originalMass, preserveDimsAndEuler);

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
/// This gets the gradient of the spatial tensor with respect to a specific
/// index in the moment vector
Eigen::Matrix6s Inertia::getSpatialTensorGradientWrtDimsAndEulerVector(
    int index)
{
  s_t mass = getMass();
  Eigen::Vector6s dimsAndEuler = getDimsAndEulerVector();
  Eigen::Vector3s dims = dimsAndEuler.head<3>();
  Eigen::Matrix3s principalAxis = (mass / 12.0)
                                  * Eigen::Vector3s(
                                        dims(1) * dims(1) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(1) * dims(1))
                                        .asDiagonal();
  Eigen::Matrix3s R = math::eulerXYZToMatrix(dimsAndEuler.tail<3>());

  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  if (index < 3)
  {
    Eigen::Matrix3s dPrincipalAxis;
    if (index == 0)
    {
      dPrincipalAxis
          = (mass / 12.0)
            * Eigen::Vector3s(0, 2 * dims(0), 2 * dims(0)).asDiagonal();
    }
    else if (index == 1)
    {
      dPrincipalAxis
          = (mass / 12.0)
            * Eigen::Vector3s(2 * dims(1), 0, 2 * dims(1)).asDiagonal();
    }
    else if (index == 2)
    {
      dPrincipalAxis
          = (mass / 12.0)
            * Eigen::Vector3s(2 * dims(2), 2 * dims(2), 0).asDiagonal();
    }
    result.block<3, 3>(0, 0) = R * dPrincipalAxis * R.transpose();
  }
  else
  {
    Eigen::Matrix3s dR
        = math::eulerXYZToMatrixGrad(dimsAndEuler.tail<3>(), index - 3);
    Eigen::Matrix3s tmp = dR * principalAxis * R.transpose();
    result.block<3, 3>(0, 0) = tmp + tmp.transpose();
  }

  return result;
}

//==============================================================================
/// This gets the gradient of the spatial tensor with respect to a specific
/// index in the moment vector
Eigen::Matrix6s
Inertia::finiteDifferenceSpatialTensorGradientWrtDimsAndEulerVector(int index)
{
  Eigen::Matrix6s result = Eigen::Matrix6s::Zero();

  Eigen::Vector6s originalDimsAndEulers = getDimsAndEulerVector();

  math::finiteDifference<Eigen::Matrix6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Matrix6s& out) {
        Eigen::Vector6s perturbed = originalDimsAndEulers;
        perturbed(index) += eps;
        setDimsAndEulerVector(perturbed);
        out = getSpatialTensor();
        return true;
      },
      result,
      1e-3,
      true);

  setDimsAndEulerVector(originalDimsAndEulers);

  return result;
}

//==============================================================================
/// This creates a copy of this inertia object
Inertia Inertia::clone() const
{
  return Inertia(mMass, mCenterOfMass, getMoment());
}

//==============================================================================
/// This computes the moment vector from a the mass, and a concatenated vector
/// for the dimensions of a cube, and the euler angles by which to rotate the
/// cube.
///
/// Notes: We choose euler angles, instead of SO3, because the gradients are
/// smoother in the very small rotation values, which is where we expect
/// optimizers to spend most of their time.
Eigen::Vector6s Inertia::computeMomentVector(
    s_t mass, Eigen::Vector6s dimsAndEuler)
{
  Eigen::Vector3s dims = dimsAndEuler.head<3>();
  Eigen::Matrix3s principalAxis = (mass / 12.0)
                                  * Eigen::Vector3s(
                                        dims(1) * dims(1) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(1) * dims(1))
                                        .asDiagonal();
  Eigen::Matrix3s R = math::eulerXYZToMatrix(dimsAndEuler.tail<3>());
  Eigen::Matrix3s rotatedInertia = R * principalAxis * R.transpose();

  s_t I_XX = rotatedInertia(0, 0);
  s_t I_YY = rotatedInertia(1, 1);
  s_t I_ZZ = rotatedInertia(2, 2);
  s_t I_XY = rotatedInertia(0, 1);
  s_t I_XZ = rotatedInertia(0, 2);
  s_t I_YZ = rotatedInertia(1, 2);

  Eigen::Vector6s result;
  result << I_XX, I_YY, I_ZZ, I_XY, I_XZ, I_YZ;
  return result;
}

//==============================================================================
Eigen::Matrix6s Inertia::computeMomentVectorJacWrtDimsAndEuler(
    s_t mass, Eigen::Vector6s dimsAndEuler)
{
  Eigen::Matrix6s J = Eigen::Matrix6s::Zero();
  (void)mass;
  (void)dimsAndEuler;

  Eigen::Vector3s dims = dimsAndEuler.head<3>();
  Eigen::Matrix3s principalAxis = (mass / 12.0)
                                  * Eigen::Vector3s(
                                        dims(1) * dims(1) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(1) * dims(1))
                                        .asDiagonal();
  Eigen::Matrix3s R = math::eulerXYZToMatrix(dimsAndEuler.tail<3>());
  for (int i = 0; i < 6; i++)
  {
    Eigen::Matrix3s dRotatedInertia;
    // Grad wrt dims
    if (i < 3)
    {
      Eigen::Matrix3s dPrincipalAxis;
      if (i == 0)
      {
        dPrincipalAxis
            = (mass / 12.0)
              * Eigen::Vector3s(0, 2 * dims(0), 2 * dims(0)).asDiagonal();
      }
      else if (i == 1)
      {
        dPrincipalAxis
            = (mass / 12.0)
              * Eigen::Vector3s(2 * dims(1), 0, 2 * dims(1)).asDiagonal();
      }
      else if (i == 2)
      {
        dPrincipalAxis
            = (mass / 12.0)
              * Eigen::Vector3s(2 * dims(2), 2 * dims(2), 0).asDiagonal();
      }
      dRotatedInertia = R * dPrincipalAxis * R.transpose();
    }
    else
    {
      Eigen::Matrix3s dR
          = math::eulerXYZToMatrixGrad(dimsAndEuler.tail<3>(), i - 3);
      Eigen::Matrix3s tmp = dR * principalAxis * R.transpose();
      dRotatedInertia = tmp + tmp.transpose();
    }
    s_t I_XX = dRotatedInertia(0, 0);
    s_t I_YY = dRotatedInertia(1, 1);
    s_t I_ZZ = dRotatedInertia(2, 2);
    s_t I_XY = dRotatedInertia(0, 1);
    s_t I_XZ = dRotatedInertia(0, 2);
    s_t I_YZ = dRotatedInertia(1, 2);

    Eigen::Vector6s result;
    result << I_XX, I_YY, I_ZZ, I_XY, I_XZ, I_YZ;
    J.col(i) = result;
  }

  return J;
}

//==============================================================================
Eigen::Matrix6s Inertia::finiteDifferenceMomentVectorJacWrtDimsAndEuler(
    s_t mass, Eigen::Vector6s dimsAndEuler)
{
  (void)mass;
  (void)dimsAndEuler;
  Eigen::Vector6s original = dimsAndEuler;
  Eigen::MatrixXs results = Eigen::MatrixXs::Zero(6, 6);
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::Vector6s plusEps = original;
        plusEps(dof) += eps;
        perturbed = computeMomentVector(mass, plusEps);
        return true;
      },
      results,
      1e-3,
      true);
  Eigen::Matrix6s J = Eigen::Matrix6s::Zero();
  J = results;
  return J;
}

//==============================================================================
Eigen::Vector6s Inertia::computeMomentVectorGradWrtMass(
    s_t mass, Eigen::Vector6s dimsAndEuler)
{
  (void)mass;

  Eigen::Vector3s dims = dimsAndEuler.head<3>();
  Eigen::Matrix3s principalAxis = (1.0 / 12.0)
                                  * Eigen::Vector3s(
                                        dims(1) * dims(1) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(2) * dims(2),
                                        dims(0) * dims(0) + dims(1) * dims(1))
                                        .asDiagonal();
  Eigen::Matrix3s R = math::eulerXYZToMatrix(dimsAndEuler.tail<3>());
  Eigen::Matrix3s rotatedInertia = R * principalAxis * R.transpose();

  s_t I_XX = rotatedInertia(0, 0);
  s_t I_YY = rotatedInertia(1, 1);
  s_t I_ZZ = rotatedInertia(2, 2);
  s_t I_XY = rotatedInertia(0, 1);
  s_t I_XZ = rotatedInertia(0, 2);
  s_t I_YZ = rotatedInertia(1, 2);

  Eigen::Vector6s result;
  result << I_XX, I_YY, I_ZZ, I_XY, I_XZ, I_YZ;
  return result;
}

//==============================================================================
Eigen::Vector6s Inertia::finiteDifferenceMomentVectorGradWrtMass(
    s_t mass, Eigen::Vector6s dimsAndEuler)
{
  Eigen::Vector6s result = Eigen::Vector6s::Zero();
  math::finiteDifference<Eigen::Vector6s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Vector6s& perturbed) {
        perturbed = computeMomentVector(mass + eps, dimsAndEuler);
        return true;
      },
      result,
      1e-3,
      true);
  return result;
}

//==============================================================================
/// This reverses computeMomentVector(), to get into a more interpretable
/// space of cube dimensions and rotations. This can then be used to visualize
/// inertia in a GUI.
Eigen::Vector6s Inertia::computeDimsAndEuler(
    s_t mass, Eigen::Vector6s momentVector)
{
  Eigen::Matrix3s inertia;
  s_t I_XX = momentVector(0);
  s_t I_YY = momentVector(1);
  s_t I_ZZ = momentVector(2);
  s_t I_XY = momentVector(3);
  s_t I_XZ = momentVector(4);
  s_t I_YZ = momentVector(5);

  inertia << I_XX, I_XY, I_XZ, I_XY, I_YY, I_YZ, I_XZ, I_YZ, I_ZZ;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3s> eigensolver(inertia);
  if (eigensolver.info() != Eigen::Success)
  {
    std::cout << "Error!! Could not recover eigenvectors from inertia matrix, "
                 "so unable to find rotation"
              << std::endl;
  }

  Eigen::Matrix3s unsortedR = eigensolver.eigenvectors();
  Eigen::Vector3s unsortedPrincipalInertia = eigensolver.eigenvalues();

  int bestX = 0;
  s_t bestXScore
      = (unsortedR.col(0).cwiseAbs() - Eigen::Vector3s::UnitX()).squaredNorm();
  for (int i = 1; i < 3; i++)
  {
    s_t score = (unsortedR.col(i).cwiseAbs() - Eigen::Vector3s::UnitX())
                    .squaredNorm();
    if (score < bestXScore)
    {
      bestXScore = score;
      bestX = i;
    }
  }
  int bestY = bestX + 1;
  if (bestY > 2)
    bestY = 0;
  s_t bestYScore = (unsortedR.col(bestY).cwiseAbs() - Eigen::Vector3s::UnitY())
                       .squaredNorm();
  for (int i = 0; i < 3; i++)
  {
    s_t score = (unsortedR.col(i).cwiseAbs() - Eigen::Vector3s::UnitY())
                    .squaredNorm();
    if (score < bestYScore && i != bestX)
    {
      bestYScore = score;
      bestY = i;
    }
  }
  int bestZ = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i != bestX && i != bestY)
    {
      bestZ = i;
      break;
    }
  }

  Eigen::Matrix3s R = Eigen::Matrix3s::Zero();
  Eigen::Vector3s principalInertia = Eigen::Vector3s::Zero();
  R.col(0) = unsortedR.col(bestX);
  R.col(1) = unsortedR.col(bestY);
  R.col(2) = unsortedR.col(bestZ);
  principalInertia(0) = unsortedPrincipalInertia(bestX);
  principalInertia(1) = unsortedPrincipalInertia(bestY);
  principalInertia(2) = unsortedPrincipalInertia(bestZ);

  bool flipX
      = (unsortedR.col(bestX) - Eigen::Vector3s::UnitX()).squaredNorm()
        > (unsortedR.col(bestX) + Eigen::Vector3s::UnitX()).squaredNorm();
  if (flipX)
  {
    R.col(0) *= -1;
  }
  bool flipY
      = (unsortedR.col(bestY) - Eigen::Vector3s::UnitY()).squaredNorm()
        > (unsortedR.col(bestY) + Eigen::Vector3s::UnitY()).squaredNorm();
  if (flipY)
  {
    R.col(1) *= -1;
  }
  bool flipZ
      = (unsortedR.col(bestZ) - Eigen::Vector3s::UnitZ()).squaredNorm()
        > (unsortedR.col(bestZ) + Eigen::Vector3s::UnitZ()).squaredNorm();
  if (flipZ)
  {
    R.col(2) *= -1;
  }

  Eigen::Vector3s euler = math::matrixToEulerXYZ(R);

  const s_t xx = principalInertia(0);
  const s_t yy = principalInertia(1);
  const s_t zz = principalInertia(2);
  assert(mass != 0);
  assert(6 * (xx + zz - yy) / mass > 0);
  Eigen::Vector3s dim = Eigen::Vector3s(
      sqrt(6 * (yy + zz - xx) / mass),
      sqrt(6 * (xx + zz - yy) / mass),
      sqrt(6 * (xx + yy - zz) / mass));
  assert(!dim.hasNaN());

  Eigen::Vector6s result;
  result.head<3>() = dim;
  result.tail<3>() = euler;
  return result;
}

//==============================================================================
/// This creates the inertia for a rectangular prism, from the original
/// formula.
Inertia Inertia::createCubeInertia(s_t mass, Eigen::Vector3s dims)
{
  s_t xx = (mass / 12.0) * (dims(1) * dims(1) + dims(2) * dims(2));
  s_t yy = (mass / 12.0) * (dims(0) * dims(0) + dims(2) * dims(2));
  s_t zz = (mass / 12.0) * (dims(0) * dims(0) + dims(1) * dims(1));
  Eigen::Matrix3s MOI = Eigen::Matrix3s::Identity();
  MOI(0, 0) = xx;
  MOI(1, 1) = yy;
  MOI(2, 2) = zz;
  return Inertia(mass, Eigen::Vector3s::Zero(), MOI);
}

//==============================================================================
/// This computes the size of a cube, ignoring the off-diagonal inertia
/// properties.
Eigen::Vector3s Inertia::getImpliedCubeDimensions() const
{
  const s_t xx = mMoment[I_XX - 4];
  const s_t yy = mMoment[I_YY - 4];
  const s_t zz = mMoment[I_ZZ - 4];
  const s_t mass = getMass();
  assert(mass != 0);
  assert(6 * (xx + zz - yy) / mass > 0);
  Eigen::Vector3s dim = Eigen::Vector3s(
      sqrt(6 * (yy + zz - xx) / mass),
      sqrt(6 * (xx + zz - yy) / mass),
      sqrt(6 * (xx + yy - zz) / mass));
  assert(!dim.hasNaN());
  return dim;
}

//==============================================================================
/// This gives a 3x6 Jacobian that relates changes in the moment vector to
/// changes in the implied dimensions.
Eigen::Matrix<s_t, 3, 6>
Inertia::getImpliedCubeDimensionsJacobianWrtMomentVector() const
{
  Eigen::Matrix<s_t, 3, 6> J = Eigen::Matrix<s_t, 3, 6>::Zero();

  const s_t xx = mMoment[I_XX - 4];
  const s_t yy = mMoment[I_YY - 4];
  const s_t zz = mMoment[I_ZZ - 4];
  const s_t mass = getMass();

  const s_t k = sqrt(1.5) / sqrt(mass);

  J(0, 0) = -k * 1.0
            / sqrt(yy + zz - xx); // d/dxx of sqrt(6 * (yy + zz - xx) / mass
  J(0, 1) = -J(0, 0);
  J(0, 2) = -J(0, 0);
  J(1, 0) = k * 1.0
            / sqrt(xx + zz - yy); // d/dxx of sqrt(6 * (xx + zz - yy) / mass)
  J(1, 1) = -J(1, 0);
  J(1, 2) = J(1, 0);
  J(2, 0) = k * 1.0
            / sqrt(xx + yy - zz); // d/dxx of sqrt(6 * (xx + yy - zz) / mass)
  J(2, 1) = J(2, 0);
  J(2, 2) = -J(2, 0);

  return J;
}

//==============================================================================
/// This gives a 3x6 Jacobian that relates changes in the moment vector to
/// changes in the implied dimensions.
Eigen::Matrix<s_t, 3, 6>
Inertia::finiteDifferenceImpliedCubeDimensionsJacobianWrtMomentVector()
{
  Eigen::Matrix<s_t, 3, 6> J = Eigen::Matrix<s_t, 3, 6>::Zero();

  Eigen::Vector6s original = getMomentVector();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(3, 6);
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ Eigen::VectorXs& perturbed) {
        Eigen::Vector6s plusEps = original;
        plusEps(dof) += eps;
        setMomentVector(plusEps);
        perturbed = getImpliedCubeDimensions();
        return true;
      },
      result,
      1e-3,
      true);
  setMomentVector(original);
  J = result;
  return J;
}

//==============================================================================
/// This gets the gradient of implied dimensions wrt the mass
Eigen::Vector3s Inertia::getImpliedCubeDimensionsGradientWrtMass() const
{
  s_t mass = getMass();
  Eigen::Vector3s dims = getImpliedCubeDimensions();
  return -dims / (2 * mass);
}

//==============================================================================
/// This gets the gradient of implied dimensions wrt the mass
Eigen::Vector3s Inertia::finiteDifferenceImpliedCubeDimensionsGradientWrtMass()
{
  s_t original = getMass();
  Eigen::Vector3s result = Eigen::Vector3s::Zero();
  math::finiteDifference<Eigen::Vector3s>(
      [&](/* in*/ s_t eps,
          /*out*/ Eigen::Vector3s& perturbed) {
        setMass(original + eps, false);
        perturbed = getImpliedCubeDimensions();
        return true;
      },
      result,
      1e-3,
      true);
  setMass(original, false);
  return result;
}

//==============================================================================
/// This gets the implied cube density. This is a useful constraint
s_t Inertia::getImpliedCubeDensity() const
{
  Eigen::Vector3s dims = getImpliedCubeDimensions();
  s_t volume = dims(0) * dims(1) * dims(2);
  return getMass() / volume;
}

//==============================================================================
/// This gets the gradient of implied density wrt the moment vector
Eigen::Vector6s Inertia::getImpliedCubeDensityGradientWrtMomentVector() const
{
  s_t mass = getMass();
  Eigen::Vector3s dims = getImpliedCubeDimensions();

  Eigen::Matrix<s_t, 3, 6> dimsWrtMomentVec
      = getImpliedCubeDimensionsJacobianWrtMomentVector();
  Eigen::Vector3s densityWrtDims = Eigen::Vector3s(
      -mass / (dims(0) * dims(0) * dims(1) * dims(2)),
      -mass / (dims(0) * dims(1) * dims(1) * dims(2)),
      -mass / (dims(0) * dims(1) * dims(2) * dims(2)));

  return dimsWrtMomentVec.transpose() * densityWrtDims;
}

//==============================================================================
/// This gets the gradient of implied density wrt the moment vector
Eigen::Vector6s
Inertia::finiteDifferenceImpliedCubeDensityGradientWrtMomentVector()
{
  Eigen::Vector6s original = getMomentVector();
  Eigen::Vector6s result = Eigen::Vector6s::Zero();
  math::finiteDifference<Eigen::Vector6s>(
      [&](/* in*/ s_t eps,
          /* in*/ int dof,
          /*out*/ s_t& perturbed) {
        Eigen::Vector6s plusEps = original;
        plusEps(dof) += eps;
        setMomentVector(plusEps);
        perturbed = getImpliedCubeDensity();
        return true;
      },
      result,
      1e-6,
      true);
  setMomentVector(original);
  return result;
}

//==============================================================================
/// This gets the gradient of implied density wrt the mass
s_t Inertia::getImpliedCubeDensityGradientWrtMass() const
{
  Eigen::Vector3s dims = getImpliedCubeDimensions();
  Eigen::Vector3s dimsWrtMass = getImpliedCubeDimensionsGradientWrtMass();

  s_t mass = getMass();
  s_t volume = dims(0) * dims(1) * dims(2);
  s_t volumeWrtMass = dimsWrtMass(0) * dims(1) * dims(2)
                      + dims(0) * dimsWrtMass(1) * dims(2)
                      + dims(0) * dims(1) * dimsWrtMass(2);
  (void)volumeWrtMass;

  return (volume - mass * volumeWrtMass) / (volume * volume);
}

//==============================================================================
/// This gets the gradient of implied density wrt the mass
s_t Inertia::finiteDifferenceImpliedCubeDensityGradientWrtMass()
{
  s_t original = getMass();
  s_t result = 0.0;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /*out*/ s_t& perturbed) {
        setMass(original + eps, false);
        perturbed = getImpliedCubeDensity();
        return true;
      },
      result,
      1e-3,
      true);
  setMass(original, false);
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
