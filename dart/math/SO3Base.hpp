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

#ifndef DART_MATH_SO3BASE_HPP_
#define DART_MATH_SO3BASE_HPP_

#include <type_traits>

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/detail/SO3Operations.hpp"

namespace dart {
namespace math {

struct SO3Representation {};

template <typename Derived>
class SO3Base
{
public:

  static constexpr int Dim = 3;

  /// The scalar type of the coefficients
  using S = typename detail::Traits<Derived>::S;

  using RotationMatrix = Eigen::Matrix<S, Dim, Dim>;
  using RotationVector = Eigen::Matrix<S, Dim, 1>;

  /// The data type of this SO(3) representation type
  using RepData = typename detail::Traits<Derived>::RepData;

  /// The data type for the Lie algebra of SO(3) namely so(3)
  using Tangent = Eigen::Matrix<S, Dim, 1>;
  using so3 = Tangent;

  //----------------------------------------------------------------------------
  /// \{ \name Constructors
  //----------------------------------------------------------------------------

  /// Default constructor
  SO3Base() = default;

  /// Copy constructor
  SO3Base(const SO3Base&) = default;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Casters to derived class
  //----------------------------------------------------------------------------

  /// Return a reference to the derived object
  const Derived& derived() const;

  /// Return a const reference to the derived object
  Derived& derived();

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Operators
  //----------------------------------------------------------------------------

  /// Set this SO(3) from any kinds of SO(3) types
  template <typename OtherDerived>
  Derived& operator=(const SO3Base<OtherDerived>& other);

  /// Set this SO(3) from any kinds of SO(3) types
  template <typename OtherDerived>
  Derived& operator=(SO3Base<OtherDerived>&& other);

  /// Set this SO(3) from Eigen::RotationBase
  template <typename RotationDerived>
  Derived& operator=(const Eigen::RotationBase<RotationDerived, Dim>& rot);

  /// Set this SO(3) from Eigen::RotationBase
  template <typename RotationDerived>
  Derived& operator=(Eigen::RotationBase<RotationDerived, Dim>&& rot);

  /// Set this SO(3) from Eigen::MatrixBase
  template <typename MatrixDerived>
  Derived& operator=(const Eigen::MatrixBase<MatrixDerived>& matrix);

  /// Set this SO(3) from Eigen::MatrixBase
  template <typename MatrixDerived>
  Derived& operator=(Eigen::MatrixBase<MatrixDerived>&& matrix);

  /// Group multiplication
  template <typename OtherDerived>
  auto
  operator*(const SO3Base<OtherDerived>& other) const
  -> decltype(detail::so3_operations::so3_multiplication_impl<
      Derived, OtherDerived>::run(
                std::declval<RepData>(),
                std::declval<typename OtherDerived::RepData>()));

  const RotationVector operator*(const RotationVector& vector);

  /// In-place group multiplication
  template <typename OtherDerived>
  void operator*=(const SO3Base<OtherDerived>& other);

  /// Equality operator
  bool operator==(const SO3Base& other);

  /// Equality operator
  template <typename OtherDerived>
  bool operator==(const SO3Base<OtherDerived>& other);

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Representation properties
  //----------------------------------------------------------------------------

  void setRandom();

  static Derived Random();

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name SO3 group operations
  //----------------------------------------------------------------------------

  void setIdentity();

  bool isIdentity();

  static Derived Identity();

  /// Invert this SO(3) in place.
  void invert();

  /// Return the inverse of this SO(3). This SO(3) doesn't change itself.
  Derived getInverse() const;

  /// \} // SO3 group operations

  template <typename OtherDerived>
  bool isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6) const;

  bool isApprox(const Eigen::AngleAxis<S>& aa, S tol = 1e-6) const;

  template <typename QuatDerived>
  bool isApprox(const Eigen::QuaternionBase<QuatDerived>& quat, S tol = 1e-6) const;

  template <typename MatrixDerived>
  bool isApprox(const Eigen::MatrixBase<MatrixDerived>& matrix, S tol = 1e-6) const;

  //----------------------------------------------------------------------------
  /// \{ \name Exponential and logarithm maps
  //----------------------------------------------------------------------------

  static Derived Exp(const so3& tangent);

  static Derived Exp(so3&& tangent);

  void setExp(const so3& tangent);

  void setExp(so3&& tangent);

  static so3 Log(const Derived& point);

  static so3 Log(Derived&& point);

  so3 getLog() const;

  /// \}

  static RotationMatrix Hat(const Tangent& angleAxis);

  static Tangent Vee(const RotationMatrix& mat);

  //----------------------------------------------------------------------------
  /// \{ \name Representation conversions
  //----------------------------------------------------------------------------

  template <typename RepTo>
  auto to() const
  -> decltype(detail::SO3ToImpl<Derived, RepTo>::run(std::declval<RepData>()));

  // TODO(JS): implement as<OtherDerived>()

  auto toRotationMatrix() const
  -> typename detail::Traits<SO3Matrix<S>>::RepData;

  void fromRotationMatrix(const RotationMatrix& rotMat);

//  template <typename RepTo>
//  auto getCoordinates() const
//  -> decltype(detail::so3_operations::SO3ConvertImpl<Derived, RepTo>::run(
//      std::declval<RepData>()));

  /// \} // Representation conversions

  //----------------------------------------------------------------------------
  /// \{ \name Accessors to representation data
  //----------------------------------------------------------------------------

  /// Set representation data
  void setRepData(const RepData& data);

  /// Set representation data
  void setRepData(RepData&& data);

  /// Return a const reference of the raw data of the representation type
  const RepData& getRepData() const;

  /// Return a reference of the raw data of the representation type
  RepData& getRepData();

  /// \}

//  SO3Canonical canonical()
//  {
//    return canonical(detail::SO3::group_is_canonical<Derived>());
//  }

//  const SO3Canonical canonical() const
//  {
//    return canonical(detail::SO3::group_is_canonical<Derived>());
//  }

//  static constexpr bool isCanonical()
//  {
//    return detail::SO3::group_is_canonical<Derived>::value;
//  }

//private:

//  SO3Canonical canonical(std::true_type)
//  {
//    return derived();
//  }

//  const SO3Canonical canonical(std::true_type) const
//  {
//    return derived();
//  }

//  SO3Canonical canonical(std::false_type)
//  {
//    return typename detail::traits<Derived>::Canonical(derived());
//  }

//  const SO3Canonical canonical(std::false_type) const
//  {
//    return typename detail::traits<Derived>::Canonical(derived());
//  }
};

template <typename S, typename Rep>
using so3 = typename Rep::Tangent;

template <typename S, typename Rep = DefaultSO3Canonical<S>>
class SO3 : public SO3Base<Rep> {};

} // namespace math
} // namespace dart

#include "dart/math/detail/SO3Base-impl.hpp"

#endif // DART_MATH_SO3BASE_HPP_
