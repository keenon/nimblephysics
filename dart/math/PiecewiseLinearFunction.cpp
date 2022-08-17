#include "dart/math/PiecewiseLinearFunction.hpp"

#include <exception>

/* -------------------------------------------------------------------------- *
 *                   OpenSim:  PiecewiseLinearFunction.cpp                    *
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

namespace dart {
namespace math {

//=============================================================================
// STATICS
//=============================================================================

PiecewiseLinearFunction::PiecewiseLinearFunction(
    std::vector<s_t> x, std::vector<s_t> y)
  : _x(x), _y(y)
{
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
 * used to construct the function.
 *
 * @return Number of data points (or number of coefficients).
 */
int PiecewiseLinearFunction::getSize() const
{
  return _x.size();
}

//-----------------------------------------------------------------------------
// X AND COEFFICIENTS
//-----------------------------------------------------------------------------
//_____________________________________________________________________________
/**
 * Get the array of independent variables used to construct the function.
 * For the number of independent variable data points use getN().
 *
 * @return Pointer to the independent variable data points.
 * @see getN();
 */
const std::vector<s_t>& PiecewiseLinearFunction::getX() const
{
  return _x;
}
//_____________________________________________________________________________
/**
 * Get the array of Y values for the function.
 * For the number of Y values use getNX().
 *
 * @return Pointer to the coefficients.
 * @see getCoefficients();
 */
const std::vector<s_t>& PiecewiseLinearFunction::getY() const
{
  return _y;
}

s_t PiecewiseLinearFunction::getX(int aIndex) const
{
  if (aIndex >= 0 && aIndex < _x.size())
    return _x[aIndex];
  else
  {
    assert(false && "PiecewiseLinearFunction::getX(): index out of bounds.");
    return 0.0;
  }
}

s_t PiecewiseLinearFunction::getY(int aIndex) const
{
  if (aIndex >= 0 && aIndex < _y.size())
    return _y[aIndex];
  else
  {
    assert(false && "PiecewiseLinearFunction::getY(): index out of bounds.");
    return 0.0;
  }
}

void PiecewiseLinearFunction::setX(int aIndex, s_t aValue)
{
  if (aIndex >= 0 && aIndex < _x.size())
  {
    _x[aIndex] = aValue;
    calcCoefficients();
  }
  else
  {
    assert(false && "PiecewiseLinearFunction::setX(): index out of bounds.");
  }
}

void PiecewiseLinearFunction::setY(int aIndex, s_t aValue)
{
  if (aIndex >= 0 && aIndex < _y.size())
  {
    _y[aIndex] = aValue;
    calcCoefficients();
  }
  else
  {
    assert(false && "PiecewiseLinearFunction::setY(): index out of bounds.");
  }
}

bool PiecewiseLinearFunction::deletePoint(int aIndex)
{
  if (_x.size() > 2 && _y.size() > 2 && aIndex < _x.size()
      && aIndex < _y.size())
  {
    _x.erase(_x.begin() + aIndex);
    _y.erase(_y.begin() + aIndex);

    // Recalculate the slopes
    calcCoefficients();
    return true;
  }

  return false;
}

int PiecewiseLinearFunction::addPoint(s_t aX, s_t aY)
{
  int i = 0;
  for (i = 0; i < _x.size(); i++)
    if (_x[i] > aX)
      break;

  _x.insert(_x.begin() + i, aX);
  _y.insert(_y.begin() + i, aY);

  // Recalculate the slopes
  calcCoefficients();

  return i;
}

//=============================================================================
// EVALUATION
//=============================================================================
void PiecewiseLinearFunction::calcCoefficients()
{
  int n = _x.size();

  if (n < 2)
    return;

  _b.reserve(n);

  for (int i = 0; i < n - 1; i++)
  {
    s_t range = MAX(TINY_NUMBER, _x[i + 1] - _x[i]);
    _b[i] = (_y[i + 1] - _y[i]) / range;
  }
  _b[n - 1] = _b[n - 2];
}

s_t PiecewiseLinearFunction::calcValue(s_t x) const
{
  int n = _x.size();
  s_t aX = x;

  if (aX < _x[0])
    return _y[0] + (aX - _x[0]) * _b[0];
  else if (aX > _x[n - 1])
    return _y[n - 1] + (aX - _x[n - 1]) * _b[n - 1];

  /* Check to see if the abscissa is close to one of the end points
   * (the binary search method doesn't work well if you are at one of the
   * end points.
   */
  if (EQUAL_WITHIN_ERROR(aX, _x[0]))
    return _y[0];
  else if (EQUAL_WITHIN_ERROR(aX, _x[n - 1]))
    return _y[n - 1];

  // Do a binary search to find which two points the abscissa is between.
  int k, i = 0;
  int j = n;
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

  return _y[k] + (aX - _x[k]) * _b[k];
}

s_t PiecewiseLinearFunction::calcDerivative(int order, s_t x) const
{
  if (order == 0)
    return 0.0;
  if (order > 1)
    return 0.0;

  int n = _x.size();
  s_t aX = x;

  if (aX < _x[0])
  {
    return _b[0];
  }
  else if (aX > _x[n - 1])
  {
    return _b[n - 1];
  }

  /* Check to see if the abscissa is close to one of the end points
   * (the binary search method doesn't work well if you are at one of the
   * end points.
   */
  if (EQUAL_WITHIN_ERROR(aX, _x[0]))
  {
    return _b[0];
  }
  else if (EQUAL_WITHIN_ERROR(aX, _x[n - 1]))
  {
    return _b[n - 1];
  }

  // Do a binary search to find which two points the abscissa is between.
  int k, i = 0;
  int j = n;
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

  return _b[k];
}

std::shared_ptr<CustomFunction> PiecewiseLinearFunction::offsetBy(
    s_t offset) const
{
  std::vector<s_t> newY;
  for (s_t y : this->_y)
  {
    newY.push_back(y + offset);
  }
  return std::make_shared<PiecewiseLinearFunction>(_x, newY);
}

} // namespace math
} // namespace dart