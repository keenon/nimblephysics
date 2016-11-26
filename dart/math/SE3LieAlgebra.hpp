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

#ifndef DART_MATH_SE3LIEALGEBRA_HPP_
#define DART_MATH_SE3LIEALGEBRA_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
//#include "dart/math/SE3Base.hpp"

namespace dart {
namespace math {

template <typename S_>
class se3
{
public:

  using S = S_;

  using LinearMotion = Eigen::Matrix<S, 3, 1>;
  using AngularMotion = Eigen::Matrix<S, 3, 1>;

  se3() = default;

  se3(const AngularMotion& w, const LinearMotion& v)
    : mAngular(w), mLinear(v)
  {
    // Do nothing
  }

  se3(AngularMotion&& w, LinearMotion&& v)
    : mAngular(std::move(w)), mLinear(std::move(v))
  {
    // Do nothing
  }

  const AngularMotion& getAngular() const
  {
    return mAngular;
  }

  const LinearMotion& getLinear() const
  {
    return mLinear;
  }

  Eigen::Matrix<S, 6, 1> toVector() const
  {
    Eigen::Matrix<S, 6, 1> ret;

    ret << mAngular, mLinear;

    return ret;
  }

  Eigen::Matrix<S, 4, 4> toMatrix() const
  {
    Eigen::Matrix<S, 4, 4> ret;

    ret.template topLeftCorner<3, 3>() = math::makeSkewSymmetric(mAngular);
    ret.template topRightCorner<3, 1>() = mLinear;

    ret.template bottomRows<1>().setZero();

    return ret;
  }

  static Eigen::Matrix<S, 6, 6> ad(const se3& V)
  {
    Eigen::Matrix<S, 6, 6> ret;

    ret.template topLeftCorner<3, 3>() = math::makeSkewSymmetric(V.mAngular);
    ret.template topRightCorner<3, 3>().setZero();

    ret.template bottomLeftCorner<3, 3>() = math::makeSkewSymmetric(V.mAngular);
    ret.template bottomRightCorner<3, 3>() = ret.template topLeftCorner<3, 3>();

    return ret;
  }

  Eigen::Matrix<S, 6, 6> ad() const
  {
    return se3::ad(*this);
  }

  static se3 ad(const se3& V1, const se3& V2)
  {
    AngularMotion w = V1.mAngular.cross(V2.mAngular);
    AngularMotion v = V1.mLinear.cross(V2.mAngular);
    v += V1.mAngular.cross(V2.mLinear);

    return se3(std::move(w), std::move(v));
  }

protected:

  template <typename>
  friend class SE3Base;

  Eigen::Matrix<S, 3, 1> mAngular{Eigen::Matrix<S, 3, 1>::Zero()};
  Eigen::Matrix<S, 3, 1> mLinear{Eigen::Matrix<S, 3, 1>::Zero()};
};

template <typename S>
using SpatialMotion = se3<S>;

using se3f = se3<float>;
using se3d = se3<double>;

template <typename S>
se3<S> ad(const se3<S>& V1, const se3<S>& V2)
{
  typename se3<S>::AngularMotion w = V1.getAngular().cross(V2.getAngular());
  typename se3<S>::LinearMotion v = V1.getLinear().cross(V2.getAngular());
  v += V1.getAngular().cross(V2.getLinear());

  return se3<S>(std::move(w), std::move(v));
}

template <typename S>
se3<S> Ad(const Eigen::Isometry3d& T, const se3<S>& V)
{
  Eigen::Matrix<S, 3, 1> w = T.linear()*V.getAngular();
  Eigen::Matrix<S, 3, 1> v = T.translation().cross(w);
  v += T.linear()*V.getLinear();

  return se3<S>(std::move(w), std::move(v));
}

template <typename S_>
class dse3
{
public:

  using S = S_;

  using LinearForce = Eigen::Matrix<S, 3, 1>;
  using AngularForce = Eigen::Matrix<S, 3, 1>;

  dse3() = default;

  dse3(const AngularForce& m, const LinearForce& f)
    : mAngular(m), mLinear(f)
  {
    // Do nothing
  }

  dse3(AngularForce&& m, LinearForce&& f)
    : mAngular(std::move(m)), mLinear(std::move(f))
  {
    // Do nothing
  }

  const AngularForce& getAngular() const
  {
    return mAngular;
  }

  const LinearForce& getLinear() const
  {
    return mLinear;
  }

protected:

  template <typename>
  friend class SE3Base;

  Eigen::Matrix<S, 3, 1> mAngular;
  Eigen::Matrix<S, 3, 1> mLinear;
};

template <typename S>
using SpatialForce = dse3<S>;

using dse3f = dse3<float>;
using dse3d = dse3<double>;

template <typename S>
dse3<S> dad(const se3<S>& V, const dse3<S>& F)
{
  typename dse3<S>::AngularForce m = F.getAngular().cross(V.getAngular());
  m += F.getLinear().cross(V.getLinear());
  typename dse3<S>::LinearForce f = F.getLinear().cross(V.getAngular());

  return dse3<S>(std::move(m), std::move(f));
}

} // namespace math
} // namespace dart

#endif // DART_MATH_SE3LIEALGEBRA_HPP_
