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
class SO3<S_, SO3RotationMatrix>
    : public SO3Base<SO3<S_, SO3RotationMatrix>>
{
public:

  using This = SO3<S_, SO3RotationMatrix>;
  using Base = SO3Base<This>;
  using S = typename Base::S;
  using Rep = typename Base::Rep;
  using MatrixType = typename Base::MatrixType;
  using RepDataType = typename Base::DataType;
  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  using Base::operator =;
  using Base::operator *;
  using Base::operator *=;

  using Base::matrix;

  SO3() : mRepData(RepDataType())
  {
    // Do nothing
  }

  SO3(const SO3& other) : Base(), mRepData(other.mRepData)
  {
    // Do nothing
  }

  SO3(SO3&& other) : mRepData(std::move(other.mRepData))
  {
    // Do nothing
  }

  template <typename Derived>
  SO3(const SO3Base<Derived>& other)
    : mRepData(detail::SO3_convert_impl<S, typename Derived::Rep, Rep>::run(
              other.derived().matrix()))
  {
    // Do nothing
  }

  template <typename Derived>
  explicit SO3(const Eigen::MatrixBase<Derived>& matrix) : mRepData(matrix)
  {
    using namespace Eigen;
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, RepDataType)
  }

  template <typename Derived>
  explicit SO3(Eigen::MatrixBase<Derived>&& matrix) : mRepData(std::move(matrix))
  {
    using namespace Eigen;
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, RepDataType)
  }

  explicit SO3(const so3& tangent) : mRepData(expMapRot(tangent))
  {
    // Do nothing
  }

  explicit SO3(so3&& tangent) : mRepData(expMapRot(std::move(tangent)))
  {
    // Do nothing
  }

  SO3& operator=(const SO3& other)
  {
    Base::operator =(other);
    return *this;
  }

  SO3& operator=(SO3&& other)
  {
    Base::operator=(std::move(other));
    return *this;
  }

  template <typename OtherDerived>
  bool operator ==(const OtherDerived& point)
  {
    return matrix() == point.matrix();
  }

  void setIdentity()
  {
    mRepData.setIdentity();
  }

  void setRandom()
  {
    *this = exp(Tangent::Random());
    // TODO(JS): improve
  }

  const SO3 inversed() const
  {
    return SO3(mRepData.transpose());
  }

  static SO3 exp(const so3& tangent)
  {
    return SO3(expMapRot(tangent));
    // TODO(JS): improve
  }

  static so3 log(const This& point)
  {
    return ::dart::math::logMap(point.mRepData);
    // TODO(JS): improve
  }

  /// \returns A pointer to the data array of internal data type
  S* data()
  {
    return mRepData.data();
  }

protected:

  template <typename>
  friend class SO3Base;

  RepDataType mRepData;
};

//==============================================================================
template <typename S_>
class SO3<S_, SO3AxisAngle>
    : public SO3Base<SO3<S_, SO3AxisAngle>>
{
public:

  using This = SO3<S_, SO3AxisAngle>;
  using Base = SO3Base<This>;
  using S = typename Base::S;
  using Rep = typename Base::Rep;
  using MatrixType = typename Base::MatrixType;
  using RepDataType = typename Base::DataType;
  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  using Base::operator =;
  using Base::operator *;
  using Base::operator *=;

  using Base::matrix;

  SO3() : mRepData(RepDataType())
  {
    // Do nothing
  }

  SO3(const SO3& other) : Base(), mRepData(other.mRepData)
  {
    // Do nothing
  }

  SO3(SO3&& other) : mRepData(std::move(other.mRepData))
  {
    // Do nothing
  }

  template <typename Derived>
  SO3(const SO3Base<Derived>& other)
    : mRepData(detail::SO3_convert_impl<S, typename Derived::Rep, Rep>::run(
              other.derived().matrix()))
  {
    // Do nothing
  }

  template <typename Derived>
  explicit SO3(const Eigen::MatrixBase<Derived>& matrix) : mRepData(matrix)
  {
    using namespace Eigen;
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, RepDataType)
  }

  template <typename Derived>
  explicit SO3(Eigen::MatrixBase<Derived>&& matrix) : mRepData(std::move(matrix))
  {
    using namespace Eigen;
    EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, RepDataType)
  }

  explicit SO3(const so3& tangent) : mRepData(tangent)
  {
    // Do nothing
  }

  explicit SO3(so3&& tangent) : mRepData(std::move(tangent))
  {
    // Do nothing
  }

  SO3& operator=(const SO3& other)
  {
    Base::operator =(other);
    return *this;
  }

  SO3& operator=(SO3&& other)
  {
    Base::operator=(std::move(other));
    return *this;
  }

  template <typename OtherDerived>
  bool operator ==(const OtherDerived& point)
  {
    return matrix() == point.matrix();
  }

  void setIdentity()
  {
    mRepData.setZero();
  }

  void setRandom()
  {
    mRepData.setRandom();
  }

  const SO3 inversed() const
  {
    return SO3(-mRepData);
  }

  static This exp(const so3& tangent)
  {
    return This(tangent);
  }

  static so3 log(const This& point)
  {
    return point.mRepData;
  }

  /// \returns A pointer to the data array of internal data type
  S* data()
  {
    return mRepData.data();
  }

protected:
  template <typename>
  friend class SO3Base;

  RepDataType mRepData;
};

template <typename S, typename Rep>
using so3 = typename SO3<S, Rep>::Tangent;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3_HPP_
