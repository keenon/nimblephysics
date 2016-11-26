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

#ifndef DART_MATH_SO3TROTATIONVECTOR_HPP_
#define DART_MATH_SO3TROTATIONVECTOR_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class SO3Vector : public SO3Base<SO3Vector<S_>>
{
public:

  using This = SO3Vector<S_>;
  using Base = SO3Base<This>;
  using S = typename Base::S;

  using RotationMatrix = typename Base::RotationMatrix;
  using RotationVector = typename Base::RotationVector;

  using RepData = typename Base::RepData;
  using Tangent = typename Base::Tangent;
  using so3 = typename Base::so3;

  using Base::operator =;
  using Base::operator *;
  using Base::operator *=;

//  using Base::getCoordinates;
  using Base::setRepData;
  using Base::getRepData;

  using Base::Exp;
  using Base::setExp;
  using Base::Log;
  using Base::getLog;

  /// \{ \name Constructors

  /// Default constructor. By default, the constructed SO(3) is not identity.
  SO3Vector() : Base()
  {
    // Do nothing
  }

  /// Copy constructor.
  SO3Vector(const SO3Vector& other) : Base(), mRepData(other.mRepData)
  {
    // Do nothing
  }

  /// Move constructor.
  SO3Vector(SO3Vector&& other) : mRepData(std::move(other.mRepData))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Vector(const SO3Base<Derived>& other)
    : Base(),
      mRepData(detail::so3_operations::so3_convert_impl<S, Derived, This>::run(
              other.getRepData()))
  {
    // Do nothing
  }

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Vector(SO3Base<Derived>&& other)
    : Base(),
      mRepData(detail::so3_operations::so3_convert_impl<S, Derived, This>::run(
              std::move(other.getRepData())))
  {
    // Do nothing
  }

  /// Construct from a raw rotation vector where the dimension is 3x1.
  template <typename Derived>
  SO3Vector(const Eigen::MatrixBase<Derived>& matrix) : Base(), mRepData(matrix)
  {
    assert(matrix.rows() == 3);
    assert(matrix.cols() == 1);
  }

  /// Construct from a raw rotation matrix where the dimension is 3x3.
  template <typename Derived>
  SO3Vector(Eigen::MatrixBase<Derived>&& matrix) : Base(), mRepData(std::move(matrix))
  {
    assert(matrix.rows() == 3);
    assert(matrix.cols() == 1);
  }

  /// \} // Constructors

  /// \{ \name Operators

  /// Assign a SO3 with the same representation.
  SO3Vector& operator=(const SO3Vector& other)
  {
    mRepData = other.mRepData;
    return *this;
  }

  /// Move in a SO3 with the same representation.
  SO3Vector& operator=(SO3Vector&& other)
  {
    mRepData = std::move(other.mRepData);
    return *this;
  }

  SO3Vector& operator=(const Eigen::AngleAxis<S>& quat)
  {
    mRepData = detail::so3_operations::so3_convert_impl<S, AngleAxis<S>, SO3Vector>::run(quat);
    // TODO(JS): improve; need a way to deduce representation type from Eigen
    // data type
    return *this;
  }

  SO3Vector& operator=(Eigen::AngleAxis<S>&& quat)
  {
    mRepData = detail::so3_operations::so3_convert_impl<S, AngleAxis<S>, SO3Vector>::run(
          std::move(quat));
    // TODO(JS): improve; need a way to deduce representation type from Eigen
    // data type
    return *this;
  }

  template <typename QuatDerived>
  SO3Vector& operator=(const Eigen::QuaternionBase<QuatDerived>& quat)
  {
    mRepData = detail::so3_operations::so3_convert_impl<S, Quaternion<S>, SO3Vector>::run(quat);
    // TODO(JS): improve; need a way to deduce representation type from Eigen
    // data type
    return *this;
  }

  template <typename QuatDerived>
  SO3Vector& operator=(Eigen::QuaternionBase<QuatDerived>&& quat)
  {
    mRepData = detail::so3_operations::so3_convert_impl<
        S, Quaternion<S>, SO3Vector>::run(
          std::move(quat));
    // TODO(JS): improve; need a way to deduce representation type from Eigen
    // data type
    return *this;
  }

  template <typename Derived>
  SO3Vector& operator=(const Eigen::MatrixBase<Derived>& matrix)
  {
    mRepData = detail::so3_operations::so3_convert_impl<
        S, SO3Matrix<S>, SO3Vector>::run(
          matrix);
    return *this;
  }
  // TODO(JS): take rotation vector as well using a template struct

  template <typename Derived>
  SO3Vector& operator=(Eigen::MatrixBase<Derived>&& matrix)
  {
    mRepData = detail::so3_operations::so3_convert_impl<
        S, SO3Matrix<S>, SO3Vector>::run(
          std::move(matrix));
    return *this;
  }

  /// Whether \b exactly equal to a SO3.
  bool operator ==(const SO3Vector& other)
  {
    return mRepData == other.mRepData;
  }

  /// \} // Operators

  /// \{ \name Representation properties

  template <typename MatrixDerived>
  void fromRotationVector(const Eigen::MatrixBase<MatrixDerived>& vector)
  {
    assert(vector.rows() == 3);
    assert(vector.cols() == 1);

    mRepData = vector;
  }

  template <typename MatrixDerived>
  void fromRotationVector(Eigen::MatrixBase<MatrixDerived>&& vector)
  {
    assert(vector.rows() == 3);
    assert(vector.cols() == 1);

    mRepData = std::move(vector);
  }

  Eigen::Matrix<S, 3, 1> toRotationVector() const
  {
    return mRepData;
  }

  void setRotationVector(const RotationVector& axisAngle)
  {
    mRepData = axisAngle;
  }

  const RotationVector& getRotationVector() const
  {
    return mRepData;
  }

  void setRandom()
  {
    mRepData.setRandom();
  }

  /// \} // Representation properties

  /// \{ \name SO3 group operations

  void setIdentity()
  {
    mRepData.setZero();
  }

  bool isIdentity()
  {
    return mRepData == RepData::Zero();
  }

  void invert()
  {
    mRepData *= static_cast<S>(-1);
  }

  const SO3Vector getInverse() const
  {
    return SO3Vector(-mRepData);
  }

  /// \} // SO3 group operations

protected:
  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

using SO3Vectorf = SO3Vector<float>;
using SO3Vectord = SO3Vector<double>;

extern template
class SO3Vector<double>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3TROTATIONVECTOR_HPP_
