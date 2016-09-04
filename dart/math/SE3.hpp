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

#ifndef DART_MATH_SE3_HPP_
#define DART_MATH_SE3_HPP_

#include <Eigen/Eigen>

#include "dart/math/MathTypes.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/SE3Base.hpp"

namespace dart {
namespace math {

//struct RotationMatrixRep : SE3Representation {};

template <typename S_, typename Rep> // TODO(JS): Rename Rep to RotRep
class SE3 : public SE3Base<SE3<S_, Rep>>
{
public:

  enum FromTranslationTag { FromTranslation };

  using S = S_;

  using This = SE3<S, Rep>;
  using Base = SE3Base<This>;

  using SO3Type = typename Base::SO3Type;
  using RotationType = typename Base::RotationType;
  using TranslationType = typename Base::TranslationType;

  using Tangent = typename Base::Tangent;
  using se3 = typename Base::se3;

  using Base::operator =;
//  using Base::operator *;
//  using Base::operator *=;

  using Base::rotation;
  using Base::translation;
//  using Base::coordinates;
//  using Base::getRepData;

  /// \{ \name Constructors

  /// Default constructor. By default, the constructed SE(3) is not identity.
  SE3() : Base()
  {
    // Do nothing
  }

  /// Copy constructor.
  explicit SE3(const SE3& other)
    : Base(),
      mRotation(other.mRotation),
      mTranslation(other.mTranslation)
  {
    // Do nothing
  }

  /// Move constructor.
  explicit SE3(SE3&& other)
    : Base(),
      mRotation(std::move(other.mRotation)),
      mTranslation(std::move(other.mTranslation))
  {
    // Do nothing
  }

  /// Construct from other SE3 with different SO3 representation.
  template <typename Derived>
  SE3(const SE3Base<Derived>& other)
    : Base(),
      mRotation(other.derived().rotation()),
      mTranslation(other.derived().translation())
  {
    // Do nothing
  }

  /// Construct from other SE3 with different SO3 representation.
  template <typename Derived>
  SE3(SE3Base<Derived>&& other)
    : Base(),
      mRotation(std::move(other.derived().rotation())),
      mTranslation(std::move(other.derived().translation()))
  {
    // Do nothing
  }

  /// Construct from SO3 and translation.
  SE3(const SO3Type& rotation,
      const TranslationType& translation = TranslationType::Zero())
    : Base(),
      mRotation(rotation),
      mTranslation(translation)
  {
    // Do nothing
  }

  /// Construct from SO3 and translation.
  SE3(SO3Type&& rotation,
      TranslationType&& translation = TranslationType::Zero())
    : Base(),
      mRotation(std::move(rotation)),
      mTranslation(std::move(translation))
  {
    // Do nothing
  }

  /// Construct from translation.
  ///
  /// Example:
  /// \code{.cpp}
  /// SE3<double>(FromTranslation, Eigen::Vector3d::Random()) tf;
  /// \endcode
  SE3(FromTranslationTag, const TranslationType& translation)
    : Base(),
      mRotation(SO3Type::Identity()),
      mTranslation(translation)
  {
    // Do nothing
  }

  /// Construct from translation.
  SE3(FromTranslationTag, TranslationType&& translation)
    : Base(),
      mRotation(std::move(SO3Type::Identity())),
      mTranslation(std::move(translation))
  {
    // Do nothing
  }

  /// \} // Constructors

  /// \{ \name Operators

  SE3& operator=(const SE3& other)
  {
    mRotation = other.mRotation;
    mTranslation = other.mTranslation;

    return *this;
  }

  SE3& operator=(SE3&& other)
  {
    mRotation = std::move(other.mRotation);
    mTranslation = std::move(other.mTranslation);

    return *this;
  }

  template <typename OtherDerived>
  SE3& operator=(const SE3Base<OtherDerived>& other)
  {
    mRotation = other.derived().rotation();
    mTranslation = other.derived().translation();

    return *this;
  }

  template <typename OtherDerived>
  SE3& operator=(SE3Base<OtherDerived>&& other)
  {
    mRotation = std::move(other.derived().rotation());
    mTranslation = std::move(other.derived().translation());

    return *this;
  }

  bool operator ==(const SE3& other)
  {
    return (mRotation == other.mRotation)
        && (mTranslation == other.mTranslation);
  }

  /// \} // Operators

  /// \{ \name Representation properties

  void setRotation(const RotationType& rotation)
  {
    mRotation = rotation;
  }

  const RotationType& getRotation() const
  {
    return mRotation;
  }

  void setTranslation(const TranslationType& translation)
  {
    mTranslation = translation;
  }

  const TranslationType& getTranslation() const
  {
    return mTranslation;
  }

  void rotate(const RotationType& rotation)
  {
    mRotation *= rotation;
  }

//  void preRotate(const RotationType& rotation)
//  {
//    mRotation = rotation * mRotation;
//  }

  void translate(const TranslationType& translation)
  {
    mTranslation += translation;
  }

  void setRandom()
  {
    mRotation.setRandom();
    mTranslation.setRandom();
  }

  /// \} // Representation properties

  /// \{ \name SO3 group operations

  void setIdentity()
  {
    mRotation.setIdentity();
    mTranslation.setZero();
  }

  using Base::isIdentity;
  using Base::invert;
  using Base::getInverse;

  /// \} // SO3 group operations

//  static SE3 exp(const SE3& tangent)
//  {
//    return SE3(expMapRot(tangent));
//    // TODO(JS): improve
//  }

//  static SE3 log(const This& point)
//  {
//    return ::dart::math::logMap(point.mRepData);
//    // TODO(JS): improve
//  }

protected:

  template <typename>
  friend class SE3Base;

  SO3<S, Rep> mRotation{SO3<S, Rep>()};
  Eigen::Matrix<S, 3, 1> mTranslation{Eigen::Matrix<S, 3, 1>()};
};

extern template
class SE3<double, RotationMatrixRep>;

extern template
class SE3<double, RotationVectorRep>;

extern template
class SE3<double, AxisAngleRep>;

extern template
class SE3<double, QuaternionRep>;

using SE3f = SE3<float, RotationMatrixRep>;
using SE3d = SE3<double, RotationMatrixRep>;

} // namespace math
} // namespace dart

#endif // DART_MATH_SE3_HPP_
