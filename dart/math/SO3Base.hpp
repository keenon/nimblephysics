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

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"

namespace dart {
namespace math {

enum class SO3Rep // or SO3Rep
{
  RotationMatrix,
  AngleAxis, // or RotationVector
  // AngleAndAxis,
  EulerAngleXYZ,
  Quaternion
};

enum class SO3Param // or ParameterizationType
{
  AngleAxis, // or RotationVector
  EulerAngleXYZ,
  Quaternion
};

// Forward declarations
template <typename, SO3Rep> class SO3;

namespace detail {

//==============================================================================
template <typename S_, SO3Rep Rep_>
struct traits<SO3<S_, Rep_>>
{
  using S = S_;
  using MatrixType = Eigen::Matrix<S, 3, 3>;

  static constexpr SO3Rep Rep = Rep_;
};

//==============================================================================
template <typename S_, SO3Rep Rep_>
struct so3_rep_traits;

//==============================================================================
template <typename S_>
struct so3_rep_traits<S_, SO3Rep::RotationMatrix>
{
  static constexpr SO3Rep Rep = SO3Rep::RotationMatrix;

  using RepType = Eigen::Matrix<S_, 3, 3>;
};

//==============================================================================
template <typename S_, SO3Param Param_>
struct so3_param_traits;

//==============================================================================
template <typename S_>
struct so3_param_traits<S_, SO3Param::AngleAxis>
{
  static constexpr SO3Param Param = SO3Param::AngleAxis;

  using ParamType = Eigen::Matrix<S_, 3, 1>;
};

//==============================================================================
template <typename S, SO3Rep Rep, SO3Param Param>
struct so3_convert_rep_to_param
{
  static typename so3_param_traits<S, Param>::ParamType
  run(const typename so3_rep_traits<S, Rep>::RepType& data)
  {
    // Generic version:
    // 1. convert data to the canonical data type
    // 2. convert the canonical data type to the canonical parameter type
  }
};

} // namespace detail

//==============================================================================
template <typename Derived>
class SO3Base
{
public:

  using S = typename detail::traits<Derived>::S;
  using MatrixType = typename detail::traits<Derived>::MatrixType;

  static constexpr SO3Rep Rep = detail::traits<Derived>::Rep;

  using RotationMatrix = MatrixType;

//  using Point;
  using Tangent = Eigen::Matrix<S, 3, 1>;
  using so3 = Eigen::Matrix<S, 3, 1>;

  SO3Base<Derived>& operator=(const SO3Base<Derived>& other)
  {
    derived().mData = other.derived().mData;
    return *this;
  }

  template <typename OtherDerived>
  Derived& operator=(const SO3Base<OtherDerived>& other)
  {
    derived() = other.matrix();
    return this->derived();
  }

  const Derived operator*(const SO3Base<Derived>& other) const
  {
    return derived().operator*(other.derived());
  }

  /// In-place group multiplication
  void operator*=(const SO3Base<Derived>& other)
  {
    derived().operator*=(other.derived());
  }

  void setIdentity()
  {
    derived().setIdentity();
  }

  void setRandom()
  {
    derived().mData.setRandom();
    // TODO(JS): improve
  }

  bool isApprox(const SO3Base& other, S tol = 1e-6)
  {
    return derived().mData.isApprox(other.derived().mData, tol);
    // TODO(JS): consider using geometric distance metric in measuring the
    // proximity rather than one provided by Eigen that might be the Euclidean
    // distance metric (not sure).
  }

  template <typename OtherDerived>
  bool isApprox(const SO3Base<OtherDerived>& other, S tol = 1e-6)
  {
    return matrix().isApprox(other.matrix(), tol);
    // TODO(JS): use identical distance metric with homogeneous version of
    // isApprox()
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

  template <SO3Param Param>
  const typename detail::so3_param_traits<S, Param>::ParamType
  parameters() const
  {
    typename detail::so3_param_traits<S, Param>::ParamType params;

    return params;
  }

  const MatrixType matrix() const
  {
    return derived().matrix();
  }

  /// \returns A pointer to the data array of internal data type
  S* data()
  {
    return derived().data();
  }

protected:

  /// a reference to the derived object
  const Derived& derived() const
  {
    return *static_cast<const Derived*>(this);
  }

  /// a const reference to the derived object
  Derived& derived()
  {
    return *static_cast<Derived*>(this);
  }

};

//==============================================================================
template <typename S_, SO3Rep Mode_ = SO3Rep::RotationMatrix>
class SO3 : public SO3Base<SO3<S_, Mode_>> {};

//==============================================================================
template <SO3Rep Rep = SO3Rep::RotationMatrix> using SO3f = SO3<float, Rep>;
template <SO3Rep Rep = SO3Rep::RotationMatrix> using SO3d = SO3<double, Rep>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SO3BASE_HPP_
