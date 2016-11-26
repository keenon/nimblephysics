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

#ifndef DART_MATH_SO3ROTATIONMATRIX_HPP_
#define DART_MATH_SO3ROTATIONMATRIX_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SO3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class SO3Matrix : public SO3Base<SO3Matrix<S_>>
{
public:

  using This = SO3Matrix;
  using Base = SO3Base<SO3Matrix>;
  using S = S_;

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

  //----------------------------------------------------------------------------
  /// \{ \name Constructors
  //----------------------------------------------------------------------------

  /// Default constructor. By default, the constructed SO(3) is not identity.
  SO3Matrix();

  /// Copy constructor.
  SO3Matrix(const SO3Matrix& other);

  /// Move constructor.
  SO3Matrix(SO3Matrix&& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Matrix(const SO3Base<Derived>& other);

  /// Construct from other SO3 with different representation.
  template <typename Derived>
  SO3Matrix(SO3Base<Derived>&& other);

  /// Construct from a raw rotation matrix where the dimension is 3x3.
  template <typename Derived>
  SO3Matrix(const Eigen::MatrixBase<Derived>& matrix);

  /// Construct from a raw rotation matrix where the dimension is 3x3.
  template <typename Derived>
  SO3Matrix(Eigen::MatrixBase<Derived>&& matrix);

  /// \} // Constructors

  //----------------------------------------------------------------------------
  /// \{ \name Operators
  //----------------------------------------------------------------------------

  /// Assign a SO3 with the same representation.
  SO3Matrix& operator=(const SO3Matrix& other);

  /// Move in a SO3 with the same representation.
  SO3Matrix& operator=(SO3Matrix&& other);

  template <typename Derived>
  SO3Matrix& operator=(const Eigen::MatrixBase<Derived>& matrix);

  template <typename Derived>
  SO3Matrix& operator=(Eigen::MatrixBase<Derived>&& matrix);

  template <typename RotationDerived>
  SO3Matrix& operator=(const Eigen::RotationBase<RotationDerived, Base::Dim>& rot);

  template <typename RotationDerived>
  SO3Matrix& operator=(Eigen::RotationBase<RotationDerived, Base::Dim>&& rot);

  const RotationVector operator*(const RotationVector& vector);

  /// Whether \b exactly equal to a SO3.
  bool operator ==(const SO3Matrix& other);

  /// \} // Operators

  //----------------------------------------------------------------------------
  /// \{ \name Representation properties
  //----------------------------------------------------------------------------

  template <typename Derived>
  void setRotationMatrix(const Eigen::MatrixBase<Derived>& matrix);

  template <typename Derived>
  void setRotationMatrix(Eigen::MatrixBase<Derived>&& mat);

  const RotationMatrix& getRotationMatrix() const;

  void setRandom();

  /// \} // Representation properties

  //----------------------------------------------------------------------------
  /// \{ \name SO3 group operations
  //----------------------------------------------------------------------------

  void setIdentity();

  bool isIdentity();

  void invert();

  const SO3Matrix getInverse() const;

  /// \} // SO3 group operations

protected:

  template <typename>
  friend class SO3Base;

  RepData mRepData{RepData()};
};

using SO3Matrixf = SO3Matrix<float>;
using SO3Matrixd = SO3Matrix<double>;

extern template
class SO3Matrix<double>;

} // namespace math
} // namespace dart

#include "dart/math/detail/SO3Matrix-impl.hpp"

#endif // DART_MATH_SO3ROTATIONMATRIX_HPP_
