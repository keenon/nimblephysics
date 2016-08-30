/*
 * Copyright (c) 2016, Graphics Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Humanoid Lab, Georgia Tech Research Corporation
 * Copyright (c) 2016, Personal Robotics Lab, Carnegie Mellon University
 * All rights reserved.
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

#ifndef DART_MATH_SO3_HPP_
#define DART_MATH_SO3_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename S_>
class SO3<S_, SO3Rep::RotationMatrix>
    : public SO3Base<SO3<S_, SO3Rep::RotationMatrix>>
{
public:

  using This = SO3<S_, SO3Rep::RotationMatrix>;
  using Base = SO3Base<This>;
  using S = typename Base::S;
  static constexpr SO3Rep Rep = Base::Rep;
  using MatrixType = typename Base::MatrixType;
  using DataType = Eigen::Matrix<S, 3, 3>;
  using RotationMatrix = typename Base::RotationMatrix;
  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  using Base::operator =;
  using Base::operator *=;

  SO3() : mData(DataType())
  {
    // Do nothing
  }

  SO3(const SO3& other) : mData(other.mData)
  {
    // Do nothing
  }

  explicit SO3(const so3& tangent) : mData(expMapRot(tangent))
  {
    // Do nothing
  }

  template <typename Derived>
  explicit SO3(const Eigen::MatrixBase<Derived>& rotMat) : mData(rotMat)
  {
    using namespace Eigen;
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, DataType)
  }

  explicit SO3(const RotationMatrix& rotMat) : mData(rotMat)
  {
    // Do nothing
  }

  template <typename OtherDerived>
  bool operator ==(const OtherDerived& point)
  {
    return matrix() == point.matrix();
  }

  SO3& operator=(const MatrixType& matrix)
  {
    mData = matrix;
    return *this;
  }

  const SO3 operator*(const SO3& other) const
  {
    SO3 result(*this);
    result *= other;

    return result;
  }

  /// In-place group multiplication
  void operator*=(const SO3& other)
  {
    mData *= other.mData;
  }

  void setIdentity()
  {
    mData.setIdentity();
  }

  void setRandom()
  {
    *this = exp(Tangent::Random());
    // TODO(JS): improve
  }

  const SO3 inversed() const
  {
    return SO3(mData.transpose());
  }

  static SO3 exp(const so3& tangent)
  {
    return SO3(expMapRot(tangent));
    // TODO(JS): improve
  }

  static so3 log(const This& point)
  {
    return ::dart::math::logMap(point.mData);
    // TODO(JS): improve
  }

  const MatrixType matrix() const
  {
    return mData;
  }

  /// \returns A pointer to the data array of internal data type
  S* data()
  {
    return mData.data();
  }

protected:

  template <typename>
  friend class SO3Base;

  DataType mData;
};

//==============================================================================
template <typename S_>
class SO3<S_, SO3Rep::AngleAxis>
    : public SO3Base<SO3<S_, SO3Rep::AngleAxis>>
{
public:

  using This = SO3<S_, SO3Rep::AngleAxis>;
  using Base = SO3Base<This>;
  using S = typename Base::S;
  static constexpr SO3Rep Rep = Base::Rep;
  using MatrixType = typename Base::MatrixType;
  using DataType = Eigen::Matrix<S, 3, 1>;
  using RotationMatrix = typename Base::RotationMatrix;
  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  using Base::operator =;
  using Base::operator *=;

  SO3() : mData(DataType())
  {
    // Do nothing
  }

  SO3(const SO3& other) : mData(other.mData)
  {
    // Do nothing
  }

  template <typename Derived>
  explicit SO3(const Eigen::MatrixBase<Derived>& tangent) : mData(tangent)
  {
    using namespace Eigen;
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, DataType)
  }

  explicit SO3(const so3& tangent) : mData(tangent)
  {
    // Do nothing
  }

  explicit SO3(const RotationMatrix& rotMat)
    : mData(::dart::math::logMap(rotMat))
  {
    // Do nothing
  }

  template <typename OtherDerived>
  bool operator ==(const OtherDerived& point)
  {
    return matrix() == point.matrix();
  }

  SO3& operator=(const MatrixType& matrix)
  {
    mData = ::dart::math::logMap(matrix);
    return *this;
  }

  const SO3 operator*(const SO3& other) const
  {
    SO3 result(*this);
    result *= other;

    return result;
  }

  /// In-place group multiplication
  void operator*=(const SO3& other)
  {
    *this = matrix() * other.matrix();
  }

  void setIdentity()
  {
    mData.setZero();
  }

  void setRandom()
  {
    mData.setRandom();
  }

  const SO3 inversed() const
  {
    return SO3(-mData);
  }

  static This exp(const so3& tangent)
  {
    return This(tangent);
  }

  static so3 log(const This& point)
  {
    return point.mData;
  }

  const MatrixType matrix() const
  {
    return ::dart::math::expMapRot(mData);
  }

  /// \returns A pointer to the data array of internal data type
  S* data()
  {
    return mData.data();
  }

protected:
  template <typename>
  friend class SO3Base;

  DataType mData;
};

template <typename S_, SO3Rep Mode_>
using so3 = typename SO3<S_, Mode_>::Tangent;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3_HPP_
