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
#include "dart/math/detail/SO3Utils.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename Derived>
class SO3Base
{
public:

  using S = typename detail::traits<Derived>::S;

  using MatrixType = typename detail::traits<Derived>::MatrixType; // TODO(JS): remove
  using RotationMatrixType = typename detail::traits<Derived>::RotationMatrixType;

  using Rep = typename detail::traits<Derived>::Rep;
  using DataType = typename detail::SO3_rep_traits<S, Rep>::DataType;
  // TODO(JS): rename to MatrixType

  using Tangent = Eigen::Matrix<S, 3, 1>;
  using so3 = Tangent;

  SO3Base() = default;

  SO3Base(const SO3Base&) = default;

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

  template <typename OtherDerived>
  Derived& operator=(const SO3Base<OtherDerived>& other)
  {
    detail::SO3_assign_impl<S, Derived, OtherDerived>::run(
          derived(), other.derived());

    return derived();
  }

  template <typename OtherDerived>
  Derived& operator=(SO3Base<OtherDerived>&& other)
  {
    detail::SO3_assign_impl<S, Derived, OtherDerived>::run(
          derived(), std::move(other.derived()));

    return derived();
  }

  template <typename OtherDerived>
  Derived& operator=(const Eigen::MatrixBase<OtherDerived>& matrix)
  {
    {
      using namespace Eigen;
      EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(DataType, OtherDerived)
    }

    derived().mRepData = matrix;

    return derived();
  }

  template <typename OtherDerived>
  Derived& operator=(Eigen::MatrixBase<OtherDerived>&& matrix)
  {
    {
      using namespace Eigen;
      EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(DataType, OtherDerived)
    }

    derived().mRepData = std::move(matrix);

    return derived();
  }

  template <typename OtherDerived>
  const Derived operator*(const SO3Base<OtherDerived>& other) const
  {
    Derived result(derived());
    result *= other;

    return result;
  }

  template <typename OtherDerived>
  void operator*=(const SO3Base<OtherDerived>& other)
  {
    detail::SO3_inplace_group_multiplication_impl<Derived, OtherDerived>::run(
          derived(), other.derived());
  }

  typename detail::traits<Derived>::Canonical
  canonical()
  {
    return canonical(
          std::is_same<typename detail::traits<Derived>::Canonical, Derived>());
  }

  const typename detail::traits<Derived>::Canonical
  canonical() const
  {
    return canonical(
          std::is_same<typename detail::traits<Derived>::Canonical, Derived>());
  }

  static constexpr bool isCanonical()
  {
    return std::is_same<typename detail::traits<Derived>::Canonical,
        Derived>::value;
  }

  void setIdentity()
  {
    derived().setIdentity();
  }

  void setRandom()
  {
    derived().mRepData.setRandom();
    // TODO(JS): improve
  }

  template <typename OtherDerived>
  bool isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6) const
  {
    return detail::SO3_is_approx_impl<Derived, OtherDerived>::run(
          derived(), other.derived(), tol);
    // TODO(JS): consider using geometric distance metric for measuring the
    // proximity between two point on the manifolds rather than one provided by
    // Eigen that might be the Euclidean distance metric (not sure).
  }

  static Derived Identity()
  {
    Derived I;
    I.setIdentity();

    return I;
  }

  static Derived Random()
  {
    Derived R;
    R.setRandom();

    return R;
  }

  /// \{ \name Group operators

  Derived inversed() const
  {
    return derived().inversed();
  }

  // TODO(JS): add in-place inversion void inverse()

  static MatrixType hat(const Tangent& angleAxis)
  {
    MatrixType res;
    res <<  static_cast<S>(0),     -angleAxis(2),      angleAxis(1),
                 angleAxis(2), static_cast<S>(0),     -angleAxis(0),
                -angleAxis(1),      angleAxis(0), static_cast<S>(0);

    return res;
  }

  static Tangent vee(const MatrixType& mat)
  {
    // TODO(JS): Add validity check if mat is skew-symmetric for debug mode
    return Tangent(mat(2, 1), mat(0, 2), mat(1, 0));
  }

  static Derived exp(const so3& tangent)
  {
    return derived().exp(tangent);
  }

  static so3 log(const Derived& point)
  {
    return derived().log(point);
  }

  /// \}

  RotationMatrixType toRotationMatrix() const
  {
    return detail::SO3_convert_to_canonical_impl<S, Rep>::run(
          derived().matrix());
  }

  void fromRotationMatrix(const RotationMatrixType& rotMat)
  {
    derived().matrix()
        = detail::SO3_convert_to_noncanonical_impl<S, Rep>::run(rotMat);
  }

  ///
  template <typename RepTo>
  typename detail::SO3_rep_traits<S, RepTo>::RepType
  genCoords() const
  {
//    static_assert()

    return detail::SO3_convert_impl<S, Rep, RepTo>::run(derived().matrix());
  }

  DataType& matrix()
  {
    return derived().mRepData;
  }

  const DataType& matrix() const
  {
    return derived().mRepData;
  }

  /// \returns A pointer to the data array of internal data type
  S* data()
  {
    return derived().data();
  }

  /// \returns the number of rows. \sa cols()
  std::size_t rows() const
  {
    return matrix().rows();
  }

  /// \returns the number of columns. \sa rows()
  std::size_t cols() const
  {
    return matrix().cols();
  }

  /// \returns the number of coefficients, which is rows()*cols().
  /// \sa rows(), cols()
  std::size_t size() const
  {
    return rows() * cols();
  }

private:

  typename detail::traits<Derived>::Canonical
  canonical(std::true_type)
  {
    return derived();
  }

  const typename detail::traits<Derived>::Canonical
  canonical(std::true_type) const
  {
    return derived();
  }

  typename detail::traits<Derived>::Canonical
  canonical(std::false_type)
  {
    return typename detail::traits<Derived>::Canonical(derived());
  }

  const typename detail::traits<Derived>::Canonical
  canonical(std::false_type) const
  {
    return typename detail::traits<Derived>::Canonical(derived());
  }
};

//==============================================================================
template <typename S, typename Rep = SO3RotationMatrix>
class SO3 : public SO3Base<SO3<S, Rep>> {};

//==============================================================================
template <typename Rep = SO3RotationMatrix> using SO3f = SO3<float, Rep>;
template <typename Rep = SO3RotationMatrix> using SO3d = SO3<double, Rep>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3BASE_HPP_
