/*
 * This file is adapted from the original in OpenSim, originally under the
 * Apache license, and is included in Nimble under the MIT license.
 */

/* -------------------------------------------------------------------------- *
 *                          OpenSim:  SimmSpline.cpp                          *
 * -------------------------------------------------------------------------- *
 * The OpenSim API is a toolkit for musculoskeletal modeling and simulation.  *
 * See http://opensim.stanford.edu and the NOTICE file for more information.  *
 * OpenSim is developed at Stanford University and supported by the US        *
 * National Institutes of Health (U54 GM072970, R24 HD065690) and by DARPA    *
 * through the Warrior Web program.                                           *
 *                                                                            *
 * Copyright (c) 2005-2017 Stanford University and the Authors                *
 * Author(s): Peter Loan                                                      *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may    *
 * not use this file except in compliance with the License. You may obtain a  *
 * copy of the License at http://www.apache.org/licenses/LICENSE-2.0.         *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 * -------------------------------------------------------------------------- */

#include "dart/math/SimmSpline.hpp"

#include <array>
#include <memory>

#include "dart/math/FiniteDifference.hpp"

using namespace dart;
using namespace math;
using namespace std;

//=============================================================================
// STATICS
//=============================================================================

//=============================================================================
// DESTRUCTOR AND CONSTRUCTORS
//=============================================================================

/// Construct and fit a spline
SimmSpline::SimmSpline(std::vector<s_t> x, std::vector<s_t> y) : _x(x), _y(y)
{
  // FIT THE SPLINE
  calcCoefficients();
}

//=============================================================================
// SET AND GET
//=============================================================================
//-----------------------------------------------------------------------------
// NUMBER OF DATA POINTS (N)
//-----------------------------------------------------------------------------
//_____________________________________________________________________________
/**
 * Get size or number of independent data points (or number of coefficients)
 * used to construct the spline.
 *
 * @return Number of data points (or number of coefficients).
 */
int SimmSpline::getSize() const
{
  return _x.size();
}

//-----------------------------------------------------------------------------
// X AND COEFFICIENTS
//-----------------------------------------------------------------------------
//_____________________________________________________________________________
/**
 * Get the array of independent variables used to construct the spline.
 * For the number of independent variable data points use getN().
 *
 * @return Pointer to the independent variable data points.
 * @see getN();
 */
const std::vector<s_t>& SimmSpline::getX() const
{
  return (_x);
}
//_____________________________________________________________________________
/**
 * Get the array of Y values for the spline.
 * For the number of Y values use getNX().
 *
 * @return Pointer to the coefficients.
 * @see getCoefficients();
 */
const std::vector<s_t>& SimmSpline::getY() const
{
  return (_y);
}

int SimmSpline::getNumberOfPoints() const
{
  return _x.size();
}

//=============================================================================
// EVALUATION
//=============================================================================
void SimmSpline::calcCoefficients()
{
  int n = _x.size();
  int nm1, nm2, i, j;
  s_t t;

  if (n < 2)
    return;

  _b.resize(n, 0.0);
  _c.resize(n, 0.0);
  _d.resize(n, 0.0);

  if (n == 2)
  {
    t = MAX(TINY_NUMBER, _x[1] - _x[0]);
    _b[0] = _b[1] = (_y[1] - _y[0]) / t;
    _c[0] = _c[1] = 0.0;
    _d[0] = _d[1] = 0.0;
    return;
  }

  nm1 = n - 1;
  nm2 = n - 2;

  /* Set up tridiagonal system:
   * b = diagonal, d = offdiagonal, c = right-hand side
   */

  _d[0] = MAX(TINY_NUMBER, _x[1] - _x[0]);
  _c[1] = (_y[1] - _y[0]) / _d[0];
  for (i = 1; i < nm1; i++)
  {
    _d[i] = MAX(TINY_NUMBER, _x[i + 1] - _x[i]);
    _b[i] = 2.0 * (_d[i - 1] + _d[i]);
    _c[i + 1] = (_y[i + 1] - _y[i]) / _d[i];
    _c[i] = _c[i + 1] - _c[i];
  }

  /* End conditions. Third derivatives at x[0] and x[n-1]
   * are obtained from divided differences.
   */

  _b[0] = -_d[0];
  _b[nm1] = -_d[nm2];
  _c[0] = 0.0;
  _c[nm1] = 0.0;

  if (n > 3)
  {
    s_t d1, d2, d3, d20, d30, d31;

    d31 = MAX(TINY_NUMBER, _x[3] - _x[1]);
    d20 = MAX(TINY_NUMBER, _x[2] - _x[0]);
    d1 = MAX(TINY_NUMBER, _x[nm1] - _x[n - 3]);
    d2 = MAX(TINY_NUMBER, _x[nm2] - _x[n - 4]);
    d30 = MAX(TINY_NUMBER, _x[3] - _x[0]);
    d3 = MAX(TINY_NUMBER, _x[nm1] - _x[n - 4]);
    _c[0] = _c[2] / d31 - _c[1] / d20;
    _c[nm1] = _c[nm2] / d1 - _c[n - 3] / d2;
    _c[0] = _c[0] * _d[0] * _d[0] / d30;
    _c[nm1] = -_c[nm1] * _d[nm2] * _d[nm2] / d3;
  }

  /* Forward elimination */

  for (i = 1; i < n; i++)
  {
    t = _d[i - 1] / _b[i - 1];
    _b[i] -= t * _d[i - 1];
    _c[i] -= t * _c[i - 1];
  }

  /* Back substitution */

  _c[nm1] /= _b[nm1];
  for (j = 0; j < nm1; j++)
  {
    i = nm2 - j;
    _c[i] = (_c[i] - _d[i] * _c[i + 1]) / _b[i];
  }

  /* compute polynomial coefficients */

  _b[nm1] = (_y[nm1] - _y[nm2]) / _d[nm2] + _d[nm2] * (_c[nm2] + 2.0 * _c[nm1]);
  for (i = 0; i < nm1; i++)
  {
    _b[i] = (_y[i + 1] - _y[i]) / _d[i] - _d[i] * (_c[i + 1] + 2.0 * _c[i]);
    _d[i] = (_c[i + 1] - _c[i]) / _d[i];
    _c[i] *= 3.0;
  }
  _c[nm1] *= 3.0;
  _d[nm1] = _d[nm2];
}

s_t SimmSpline::getX(int aIndex) const
{
  return _x[aIndex];
}

s_t SimmSpline::getY(int aIndex) const
{
  return _y[aIndex];
}

void SimmSpline::setX(int aIndex, s_t aValue)
{
  assert(
      aIndex >= 0 && aIndex < _x.size()
      && "SimmSpline::setX(): index out of bounds.");

  _x[aIndex] = aValue;
  calcCoefficients();
}

void SimmSpline::setY(int aIndex, s_t aValue)
{
  assert(
      aIndex >= 0 && aIndex < _y.size()
      && "SimmSpline::setY(): index out of bounds.");
  _y[aIndex] = aValue;
  calcCoefficients();
}

bool SimmSpline::deletePoint(int aIndex)
{
  if (_x.size() > 2 && _y.size() > 2 && aIndex < _x.size()
      && aIndex < _y.size())
  {
    _x.erase(_x.begin() + aIndex);
    _y.erase(_y.begin() + aIndex);

    // Recalculate the coefficients
    calcCoefficients();
    return true;
  }

  return false;
}

s_t SimmSpline::calcValue(s_t x) const
{
  assert(_y.size() > 0);
  assert(_b.size() > 0);
  assert(_c.size() > 0);
  assert(_d.size() > 0);

  int i, j, k;
  s_t dx;

  int n = _x.size();
  s_t aX = x;

  /* Check if the abscissa is out of range of the function. If it is,
   * then use the slope of the function at the appropriate end point to
   * extrapolate. You do this rather than printing an error because the
   * assumption is that this will only occur in relatively harmless
   * situations (like a motion file that contains an out-of-range coordinate
   * value). The rest of the SIMM code has many checks to clamp a coordinate
   * value within its range of motion, so if you make it to this function
   * and the coordinate is still out of range, deal with it quietly.
   */

  /*
  if (aX < _x[0])
    return _y[0] + (aX - _x[0]) * _b[0];
  else if (aX > _x[n - 1])
    return _y[n - 1] + (aX - _x[n - 1]) * _b[n - 1];
  */

  if (n < 3)
  {
    /* If there are only 2 function points, then set k to zero
     * (you've already checked to see if the abscissa is out of
     * range or equal to one of the endpoints).
     */
    k = 0;
  }
  else
  {
    /* Check to see if the abscissa is close to one of the end points
     * (the binary search method doesn't work well if you are at one of the
     * end points.
     */
    if (EQUAL_WITHIN_ERROR(aX, _x[0]) || aX < _x[0])
      k = 0;
    else if (EQUAL_WITHIN_ERROR(aX, _x[n - 1]) || aX > _x[n - 1])
      k = n - 1;
    else
    {
      /* Do a binary search to find which two points the abscissa is between. */
      i = 0;
      j = n;
      while (1)
      {
        k = (i + j) / 2;
        if (aX < _x[k])
          j = k;
        else if (aX > _x[k + 1])
          i = k;
        else
          break;
      }
    }
  }

  dx = aX - _x[k];
  return _y[k] + dx * (_b[k] + dx * (_c[k] + dx * _d[k]));
}

s_t SimmSpline::calcDerivative(int order, s_t x) const
{
  assert(_y.size() > 0);
  assert(_b.size() > 0);
  assert(_c.size() > 0);
  assert(_d.size() > 0);

  int i, j, k;
  s_t dx;

  int n = _x.size();
  s_t aX = x;
  int aDerivOrder = order;

  // We assume all higher order derivatives are 0, because each spline segment
  // is only order 3
  if (order > 3)
    return 0;

  /* Check if the abscissa is out of range of the function. If it is,
   * then use the slope of the function at the appropriate end point to
   * extrapolate. You do this rather than printing an error because the
   * assumption is that this will only occur in relatively harmless
   * situations (like a motion file that contains an out-of-range coordinate
   * value). The rest of the SIMM code has many checks to clamp a coordinate
   * value within its range of motion, so if you make it to this function
   * and the coordinate is still out of range, deal with it quietly.
   */

  /*
  if (aX < _x[0])
  {
    if (aDerivOrder == 1)
      return _b[0];
    else
      return 0;
  }
  else if (aX > _x[n - 1])
  {
    if (aDerivOrder == 1)
      return _b[n - 1];
    else
      return 0;
  }
  */

  /* Check to see if the abscissa is close to one of the end points
   * (the binary search method doesn't work well if you are at one of the
   * end points.
   */
  /*
  if (EQUAL_WITHIN_ERROR(aX, _x[0]))
  {
    if (aDerivOrder == 1)
      return _b[0];
    else if (aDerivOrder == 2)
      return 2.0 * _c[0];
    else if (aDerivOrder == 3)
      return 6.0 * _d[0];
    else
      return 0.0;
  }
  else if (EQUAL_WITHIN_ERROR(aX, _x[n - 1]))
  {
    if (aDerivOrder == 1)
      return _b[n - 1];
    else if (aDerivOrder == 2)
      return 2.0 * _c[n - 1];
    else if (aDerivOrder == 3)
      return 6.0 * _d[0];
    else
      return 0.0;
  }
  */

  if (n < 3)
  {
    /* If there are only 2 function points, then set k to zero
     * (you've already checked to see if the abscissa is out of
     * range or equal to one of the endpoints).
     */
    k = 0;
  }
  else
  {
    if (EQUAL_WITHIN_ERROR(aX, _x[0]) || aX < _x[0])
    {
      k = 0;
    }
    else if (EQUAL_WITHIN_ERROR(aX, _x[n - 1]) || aX > _x[n - 1])
    {
      k = n - 1;
    }
    else
    {
      /* Do a binary search to find which two points the abscissa is between. */
      i = 0;
      j = n;
      while (1)
      {
        k = (i + j) / 2;
        if (aX < _x[k])
          j = k;
        else if (aX > _x[k + 1])
          i = k;
        else
          break;
      }
    }
  }

  dx = aX - _x[k];

  if (aDerivOrder == 1)
    return (_b[k] + dx * (2.0 * _c[k] + 3.0 * dx * _d[k]));

  else if (aDerivOrder == 2)
    return (2.0 * _c[k] + 6.0 * dx * _d[k]);

  else if (aDerivOrder == 3)
    return 6.0 * _d[k];

  else
    return 0.0;
}

std::shared_ptr<CustomFunction> SimmSpline::offsetBy(s_t offset) const
{
  std::vector<s_t> newY;
  for (s_t y : this->_y)
  {
    newY.push_back(y + offset);
  }
  return std::make_shared<SimmSpline>(_x, newY);
}

s_t SimmSpline::finiteDifferenceFirstDerivative(s_t x, bool useRidders)
{
  s_t result = 0.;
  s_t eps = useRidders ? 1e-3 : 1e-7;
  math::finiteDifference(
      [&](/* in*/ s_t eps,
          /*out*/ s_t& perturbed) {
        perturbed = calcValue(x + eps);
        return true;
      },
      result,
      eps,
      useRidders);
  return result;
}

int SimmSpline::getArgumentSize() const
{
  return 1;
}

int SimmSpline::getMaxDerivativeOrder() const
{
  return 2;
}