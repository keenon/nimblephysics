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
#include "dart/math/detail/SO3Base.hpp"

namespace dart {
namespace math {

struct SO3Representation {};

//template <typename S, typename Rep>
//auto Exp(const Eigen::Matrix<S, 3, 1>& tangent)
//-> decltype(detail::SO3::rep_convert_impl<S, RotationVectorRep, Rep>::run(tangent))
//{
//  return detail::SO3::rep_convert_impl<S, RotationVectorRep, Rep>::run(
//          tangent);
//}

template <typename Derived>
class SO3Base
{
public:

  static constexpr int Dim = 3;

  /// The scalar type of the coefficients
  using S = typename detail::traits<Derived>::S;

  using RotationMatrixType = Eigen::Matrix<S, Dim, Dim>;
  using VectorType = Eigen::Matrix<S, Dim, 1>;

  /// The representation type of this SO(3)
  using Rep = typename detail::traits<Derived>::Rep;

  /// The data type for this SO(3) representation type
  using RepDataType = typename detail::SO3::rep_traits<S, Rep>::RepDataType;

  using SO3Canonical = typename detail::traits<Derived>::SO3Canonical;

  /// The data type for the lie algebra of SO(3) called so(3)
  using Tangent = Eigen::Matrix<S, 3, 1>;
  using so3 = Tangent;

  /// \{ \name Constructors

  /// Default constructor
  SO3Base() = default;

  /// Copy constructor
  SO3Base(const SO3Base&) = default;

  /// \}

  /// \{ \name Casters to derived class

  /// A reference to the derived object
  const Derived& derived() const
  {
    return *static_cast<const Derived*>(this);
  }

  /// A const reference to the derived object
  Derived& derived()
  {
    return *static_cast<Derived*>(this);
  }

  /// \}

//  template <typename OtherRep>
//  ProxySO3<S, OtherRep> as()
//  {
//    SO3<S, OtherRep> casted(derived);

//    derived() = casted;
//  }

  /// \{ \name Operators

  /// Set this SO(3) from any kinds of SO(3) types
  template <typename OtherDerived>
  Derived& operator=(const SO3Base<OtherDerived>& other)
  {
    detail::SO3::group_assign_impl<S, Derived, OtherDerived>::run(
          derived(), other.derived());

    return derived();
  }

  /// Set this SO(3) from any kinds of SO(3) types
  template <typename OtherDerived>
  Derived& operator=(SO3Base<OtherDerived>&& other)
  {
    detail::SO3::group_assign_impl<S, Derived, OtherDerived>::run(
          derived(), std::move(other.derived()));

    return derived();
  }

  template <typename RotationDerived>
  Derived& operator=(const Eigen::RotationBase<RotationDerived, Dim>& rot)
  {
    derived() = rot;
    return derived();
  }

  template <typename RotationDerived>
  Derived& operator=(Eigen::RotationBase<RotationDerived, Dim>&& rot)
  {
    derived() = std::move(rot);
    return derived();
  }

  template <typename MatrixDerived>
  Derived& operator=(const Eigen::MatrixBase<MatrixDerived>& matrix)
  {
    derived() = matrix;
    return derived();
  }

  template <typename MatrixDerived>
  Derived& operator=(Eigen::MatrixBase<MatrixDerived>&& matrix)
  {
    derived() = std::move(matrix);
    return derived();
  }

  /// Group multiplication
  template <typename OtherDerived>
  auto
  operator*(const SO3Base<OtherDerived>& other) const
  -> decltype(detail::SO3::rep_multiplication_impl<
      S, Rep, typename OtherDerived::Rep>::run(
                std::declval<RepDataType>(),
                std::declval<typename OtherDerived::RepDataType>()))
  {
    return detail::SO3::rep_multiplication_impl<
        S, Rep, typename OtherDerived::Rep>::run(
          getRepData(), other.getRepData());
  }

  const VectorType operator*(const VectorType& vector)
  {
    return derived().operator*(vector);
  }

  /// In-place group multiplication
  template <typename OtherDerived>
  void operator*=(const SO3Base<OtherDerived>& other)
  {
    derived() = detail::SO3::group_multiplication_impl<Derived, OtherDerived>::run(
              derived(), other.derived());

//    detail::SO3::group_inplace_multiplication_impl<Derived, OtherDerived>::run(
//          derived(), other.derived());
  }

  bool operator ==(const SO3Base& other)
  {
    return derived() == other.derived();
  }

  template <typename OtherDerived>
  bool operator ==(const SO3Base<OtherDerived>& other)
  {
    return toRotationMatrix() == other.toRotationMatrix();
  }

  /// \}

  /// \{ \name Representation properties

  void setRandom()
  {
    derived().setRandom();
  }

  static Derived Random()
  {
    Derived R;
    R.setRandom();

    return R;
  }

  /// \}

  /// \{ \name SO3 group operations

  void setIdentity()
  {
    derived().setIdentity();
  }

  bool isIdentity()
  {
    return derived().isIdentity();
  }

  static Derived Identity()
  {
    Derived I;
    I.setIdentity();

    return I;
  }

  /// Invert this SO(3) in place.
  void invert()
  {
    return derived().invert();
  }

  /// Return the inverse of this SO(3). This SO(3) doesn't change itself.
  Derived getInverse() const
  {
    return derived().inverse();
  }

  /// \} // SO3 group operations

  template <typename OtherDerived>
  bool isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6) const
  {
    return detail::SO3::group_is_approx_impl<Derived, OtherDerived>::run(
          derived(), other.derived(), tol);
  }

  bool isApprox(const Eigen::AngleAxis<S>& aa, S tol = 1e-6) const
  {
    return detail::SO3::rep_is_approx_impl<S, Rep, AxisAngleRep>::run(
          getRepData(), aa, tol);
    // TODO(JS): improve; Eigen::AngleAxis and AxisAngleRep are in weak
    // connection..
  }

  template <typename QuatDerived>
  bool isApprox(const Eigen::QuaternionBase<QuatDerived>& quat, S tol = 1e-6) const
  {
    return detail::SO3::rep_is_approx_impl<S, Rep, QuaternionRep>::run(
          getRepData(), quat, tol);
    // TODO(JS): improve; Eigen::QuaternionBase and QuaternionRep are in weak
    // connection..
  }

  template <typename MatrixDerived>
  bool isApprox(const Eigen::MatrixBase<MatrixDerived>& matrix, S tol = 1e-6) const
  {
    // We assume matrix is 3x3 rotation matrix
    return detail::SO3::rep_is_approx_impl<S, Rep, RotationMatrixRep>::run(
          getRepData(), matrix, tol);
    // TODO(JS): improve; Eigen::QuaternionBase and QuaternionRep are in weak
    // connection..
  }

  static Derived Exp(const so3& tangent)
  {
    return Derived(
          detail::SO3::rep_convert_impl<S, RotationVectorRep, Rep>::run(
            tangent));
  }

  static Derived Exp(so3&& tangent)
  {
    return Derived(
          detail::SO3::rep_convert_impl<S, RotationVectorRep, Rep>::run(
            std::move(tangent)));
  }

  void setExp(const so3& tangent)
  {
    derived() = Exp(tangent);
  }

  void setExp(so3&& tangent)
  {
    derived() = Exp(std::move(tangent));
  }

  static so3 Log(const Derived& point)
  {
    return detail::SO3::rep_convert_impl<S, Rep, RotationVectorRep>::run(
          point.getRepData());
  }

  static so3 Log(Derived&& point)
  {
    return detail::SO3::rep_convert_impl<S, Rep, RotationVectorRep>::run(
          std::move(point.getRepData()));
  }

  so3 getLog() const
  {
    return Log(derived());
  }

  static RotationMatrixType Hat(const Tangent& angleAxis)
  {
    RotationMatrixType res;
    res <<  static_cast<S>(0),     -angleAxis(2),      angleAxis(1),
                 angleAxis(2), static_cast<S>(0),     -angleAxis(0),
                -angleAxis(1),      angleAxis(0), static_cast<S>(0);

    return res;
  }

  static Tangent Vee(const RotationMatrixType& mat)
  {
    // TODO(JS): Add validity check if mat is skew-symmetric for debug mode
    return Tangent(mat(2, 1), mat(0, 2), mat(1, 0));
  }

  /// \{ \name Representation conversions

  template <typename RepTo>
  auto to() const
  -> decltype(detail::SO3::rep_convert_impl<S, Rep, RepTo>::run(
      std::declval<RepDataType>()))
  {
    return detail::SO3::rep_convert_impl<S, Rep, RepTo>::run(
          getRepData());
  }

  auto toRotationMatrix() const -> decltype(to<RotationMatrixRep>())
  {
    // The return type could be either of const and const reference depending on
    // the Derived type. So the trailing return type deduction is used here.

    return to<RotationMatrixRep>();
  }

  void fromRotationMatrix(const RotationMatrixType& rotMat)
  {
    // We assume the canonical representation is the rotation matrix
    setRepData(
          detail::SO3::rep_convert_from_canonical_impl<S, Rep>::run(rotMat));
  }

  template <typename RepTo>
  auto getCoordinates() const -> decltype(to<RepTo>())
  {
    // TODO(JS): Change return type to Eigen::Matrix<S, Dim, 1> or
    // check if the raw data of RepTo is a vector type.
    static_assert(detail::SO3::rep_traits<S, RepTo>::CanBeCoordinates,
                  "Attempting to get invalid coordinate type.");

    return to<RepTo>();
  }

  /// \} // Representation conversions

  void setRepData(const RepDataType& data)
  {
    derived().mRepData = data;
  }

  void setRepData(RepDataType&& data)
  {
    derived().mRepData = std::move(data);
  }

  /// Return a const reference of the raw data of the representation type
  const RepDataType& getRepData() const
  {
    return derived().mRepData;
  }

  SO3Canonical canonical()
  {
    return canonical(detail::SO3::group_is_canonical<Derived>());
  }

  const SO3Canonical canonical() const
  {
    return canonical(detail::SO3::group_is_canonical<Derived>());
  }

  static constexpr bool isCanonical()
  {
    return detail::SO3::group_is_canonical<Derived>::value;
  }

private:

  SO3Canonical canonical(std::true_type)
  {
    return derived();
  }

  const SO3Canonical canonical(std::true_type) const
  {
    return derived();
  }

  SO3Canonical canonical(std::false_type)
  {
    return typename detail::traits<Derived>::Canonical(derived());
  }

  const SO3Canonical canonical(std::false_type) const
  {
    return typename detail::traits<Derived>::Canonical(derived());
  }
};

template <typename S, typename Rep>
using so3 = typename SO3<S, Rep>::Tangent;

template <typename S, typename Rep = DefaultSO3CanonicalRep>
class SO3 : public SO3Base<SO3<S, Rep>> {};

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3BASE_HPP_
