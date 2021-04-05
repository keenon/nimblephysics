/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
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

#ifndef DART_MATH_HELPERS_HPP_
#define DART_MATH_HELPERS_HPP_

// Standard Libraries
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>

// External Libraries
#include <Eigen/Dense>
// Local Headers
#include "dart/math/Constants.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/math/Random.hpp"

namespace dart {
namespace math {

//==============================================================================
template <typename T>
constexpr T toRadian(const T& degree)
{
  return degree * constants<T>::pi() / 180.0;
}

//==============================================================================
template <typename T>
constexpr T toDegree(const T& radian)
{
  return radian * 180.0 / constants<T>::pi();
}

/// \brief a cross b = (CR*a) dot b
/// const Matd CR(2,2,0.0,-1.0,1.0,0.0);
const Eigen::Matrix2s CR((Eigen::Matrix2s() << 0.0, -1.0, 1.0, 0.0).finished());

inline int delta(int _i, int _j)
{
  if (_i == _j)
    return 1;
  return 0;
}

template <typename T>
inline constexpr int sign(T x, std::false_type)
{
  return static_cast<T>(0) < x;
}

template <typename T>
inline constexpr int sign(T x, std::true_type)
{
  return (static_cast<T>(0) < x) - (x < static_cast<T>(0));
}

template <typename T>
inline constexpr int sign(T x)
{
  return sign(x, std::is_signed<T>());
}

inline s_t sqr(s_t _x)
{
  return _x * _x;
}

inline s_t Tsinc(s_t _theta)
{
  return 0.5 - sqrt(_theta) / 48;
}

inline bool isZero(s_t _theta)
{
  return (abs(_theta) < 1e-6);
}

inline s_t asinh(s_t _X)
{
  return log(_X + sqrt(_X * _X + 1));
}

inline s_t acosh(s_t _X)
{
  return log(_X + sqrt(_X * _X - 1));
}

inline s_t atanh(s_t _X)
{
  return log((1 + _X) / (1 - _X)) / 2;
}

inline s_t asech(s_t _X)
{
  return log((sqrt(-_X * _X + 1) + 1) / _X);
}

inline s_t acosech(s_t _X)
{
  return log((sign(_X) * sqrt(_X * _X + 1) + 1) / _X);
}

inline s_t acotanh(s_t _X)
{
  return log((_X + 1) / (_X - 1)) / 2;
}

#ifndef DART_USE_ARBITRARY_PRECISION
inline s_t round(s_t _x)
{
  return floor(_x + 0.5);
}
#endif

inline s_t round2(s_t _x)
{
  int gintx = static_cast<int>(floor(_x));
  if (_x - gintx < 0.5)
    return static_cast<s_t>(gintx);
  else
    return static_cast<s_t>(gintx + 1.0);
}

template <typename T>
inline T clip(const T& val, const T& lower, const T& upper)
{
  return std::max(lower, std::min(val, upper));
}

template <typename DerivedA, typename DerivedB>
inline typename DerivedA::PlainObject clip(
    const Eigen::MatrixBase<DerivedA>& val,
    const Eigen::MatrixBase<DerivedB>& lower,
    const Eigen::MatrixBase<DerivedB>& upper)
{
  return lower.cwiseMax(val.cwiseMin(upper));
}

inline bool isEqual(s_t _x, s_t _y)
{
  return (abs(_x - _y) < 1e-6);
}

// check if it is an integer
inline bool isInt(s_t _x)
{
  if (isEqual(round(_x), _x))
    return true;
  return false;
}

/// \brief Returns whether _v is a NaN (Not-A-Number) value
inline bool isNan(s_t _v)
{
#ifdef _WIN32
  return _isnan(_v) != 0;
#else
#ifdef DART_USE_ARBITRARY_PRECISION
  return isnan(_v);
#else
  return std::isnan(_v);
#endif
#endif
}

/// \brief Returns whether _m is a NaN (Not-A-Number) matrix
inline bool isNan(const Eigen::MatrixXs& _m)
{
  for (int i = 0; i < _m.rows(); ++i)
    for (int j = 0; j < _m.cols(); ++j)
      if (isNan(_m(i, j)))
        return true;

  return false;
}

/// \brief Returns whether _v is an infinity value (either positive infinity or
/// negative infinity).
inline bool isInf(s_t _v)
{
#ifdef _WIN32
  return !_finite(_v);
#else
#ifdef DART_USE_ARBITRARY_PRECISION
  return isinf(_v);
#else
  return std::isinf(_v);
#endif
#endif
}

/// \brief Returns whether _m is an infinity matrix (either positive infinity or
/// negative infinity).
inline bool isInf(const Eigen::MatrixXs& _m)
{
  for (int i = 0; i < _m.rows(); ++i)
    for (int j = 0; j < _m.cols(); ++j)
      if (isInf(_m(i, j)))
        return true;

  return false;
}

/// \brief Returns whether _m is symmetric or not
inline bool isSymmetric(const Eigen::MatrixXs& _m, s_t _tol = 1e-6)
{
  std::size_t rows = _m.rows();
  std::size_t cols = _m.cols();

  if (rows != cols)
    return false;

  for (std::size_t i = 0; i < rows; ++i)
  {
    for (std::size_t j = i + 1; j < cols; ++j)
    {
      if (abs(_m(i, j) - _m(j, i)) > _tol)
      {
        std::cout << "A: " << std::endl;
        for (std::size_t k = 0; k < rows; ++k)
        {
          for (std::size_t l = 0; l < cols; ++l)
            std::cout << std::setprecision(4) << _m(k, l) << " ";
          std::cout << std::endl;
        }

        std::cout << "A(" << i << ", " << j << "): " << _m(i, j) << std::endl;
        std::cout << "A(" << j << ", " << i << "): " << _m(i, j) << std::endl;
        return false;
      }
    }
  }

  return true;
}

inline unsigned seedRand()
{
  time_t now = time(0);
  unsigned char* p = reinterpret_cast<unsigned char*>(&now);
  unsigned seed = 0;
  std::size_t i;

  for (i = 0; i < sizeof(now); i++)
    seed = seed * (UCHAR_MAX + 2U) + p[i];

  srand(seed);
  return seed;
}

/// \deprecated Please use Random::uniform() instead.
DART_DEPRECATED(6.7)
inline s_t random(s_t _min, s_t _max)
{
  return _min + ((static_cast<s_t>(rand()) / (RAND_MAX + 1.0)) * (_max - _min));
}

/// \deprecated Please use Random::uniform() instead.
template <int N>
DART_DEPRECATED(6.7)
Eigen::Matrix<s_t, N, 1> randomVector(s_t _min, s_t _max)
{
  Eigen::Matrix<s_t, N, 1> v;
  DART_SUPPRESS_DEPRECATED_BEGIN
  for (std::size_t i = 0; i < N; ++i)
    v[i] = random(_min, _max);
  DART_SUPPRESS_DEPRECATED_END

  return v;
}

/// \deprecated Please use Random::uniform() instead.
template <int N>
DART_DEPRECATED(6.7)
Eigen::Matrix<s_t, N, 1> randomVector(s_t _limit)
{
  DART_SUPPRESS_DEPRECATED_BEGIN
  return randomVector<N>(-abs(_limit), abs(_limit));
  DART_SUPPRESS_DEPRECATED_END
}

//==============================================================================
/// \deprecated Please use Random::uniform() instead.
DART_DEPRECATED(6.7)
inline Eigen::VectorXs randomVectorXs(std::size_t size, s_t min, s_t max)
{
  Eigen::VectorXs v = Eigen::VectorXs::Zero(size);

  DART_SUPPRESS_DEPRECATED_BEGIN
  for (std::size_t i = 0; i < size; ++i)
    v[i] = random(min, max);
  DART_SUPPRESS_DEPRECATED_END

  return v;
}

//==============================================================================
/// \deprecated Please use Random::uniform() instead.
DART_DEPRECATED(6.7)
inline Eigen::VectorXs randomVectorXs(std::size_t size, s_t limit)
{
  DART_SUPPRESS_DEPRECATED_BEGIN
  return randomVectorXs(size, -abs(limit), abs(limit));
  DART_SUPPRESS_DEPRECATED_END
}

namespace suffixes {

#ifndef DART_USE_ARBITRARY_PRECISION
//==============================================================================
constexpr s_t operator"" _pi(long s_t x)
{
  return x * constants<s_t>::pi();
}

//==============================================================================
constexpr s_t operator"" _pi(unsigned long long int x)
{
  return operator"" _pi(static_cast<long s_t>(x));
}

//==============================================================================
constexpr s_t operator"" _rad(long s_t angle)
{
  return angle;
}

//==============================================================================
constexpr s_t operator"" _rad(unsigned long long int angle)
{
  return operator"" _rad(static_cast<long s_t>(angle));
}

//==============================================================================
constexpr s_t operator"" _deg(long s_t angle)
{
  return toRadian(angle);
}

//==============================================================================
constexpr s_t operator"" _deg(unsigned long long int angle)
{
  return operator"" _deg(static_cast<long s_t>(angle));
}
#endif

} // namespace suffixes

} // namespace math

namespace Color {

inline Eigen::Vector4s Red(s_t alpha)
{
  return Eigen::Vector4s(0.9, 0.1, 0.1, alpha);
}

inline Eigen::Vector3s Red()
{
  return Eigen::Vector3s(0.9, 0.1, 0.1);
}

inline Eigen::Vector3s Fuchsia()
{
  return Eigen::Vector3s(1.0, 0.0, 0.5);
}

inline Eigen::Vector4s Fuchsia(s_t alpha)
{
  return Eigen::Vector4s(1.0, 0.0, 0.5, alpha);
}

inline Eigen::Vector4s Orange(s_t alpha)
{
  return Eigen::Vector4s(1.0, 0.63, 0.0, alpha);
}

inline Eigen::Vector3s Orange()
{
  return Eigen::Vector3s(1.0, 0.63, 0.0);
}

inline Eigen::Vector4s Green(s_t alpha)
{
  return Eigen::Vector4s(0.1, 0.9, 0.1, alpha);
}

inline Eigen::Vector3s Green()
{
  return Eigen::Vector3s(0.1, 0.9, 0.1);
}

inline Eigen::Vector4s Blue(s_t alpha)
{
  return Eigen::Vector4s(0.1, 0.1, 0.9, alpha);
}

inline Eigen::Vector3s Blue()
{
  return Eigen::Vector3s(0.1, 0.1, 0.9);
}

inline Eigen::Vector4s White(s_t alpha)
{
  return Eigen::Vector4s(1.0, 1.0, 1.0, alpha);
}

inline Eigen::Vector3s White()
{
  return Eigen::Vector3s(1.0, 1.0, 1.0);
}

inline Eigen::Vector4s Black(s_t alpha)
{
  return Eigen::Vector4s(0.05, 0.05, 0.05, alpha);
}

inline Eigen::Vector3s Black()
{
  return Eigen::Vector3s(0.05, 0.05, 0.05);
}

inline Eigen::Vector4s LightGray(s_t alpha)
{
  return Eigen::Vector4s(0.9, 0.9, 0.9, alpha);
}

inline Eigen::Vector3s LightGray()
{
  return Eigen::Vector3s(0.9, 0.9, 0.9);
}

inline Eigen::Vector4s Gray(s_t alpha)
{
  return Eigen::Vector4s(0.6, 0.6, 0.6, alpha);
}

inline Eigen::Vector3s Gray()
{
  return Eigen::Vector3s(0.6, 0.6, 0.6);
}

inline Eigen::Vector4s Random(s_t alpha)
{
  return Eigen::Vector4s(
      math::Random::uniform(0.0, 1.0),
      math::Random::uniform(0.0, 1.0),
      math::Random::uniform(0.0, 1.0),
      alpha);
}

inline Eigen::Vector3s Random()
{
  return Eigen::Vector3s(
      math::Random::uniform(0.0, 1.0),
      math::Random::uniform(0.0, 1.0),
      math::Random::uniform(0.0, 1.0));
}

} // namespace Color

} // namespace dart

#endif // DART_MATH_HELPERS_HPP_
