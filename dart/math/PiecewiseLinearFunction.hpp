#ifndef MATH_PIECEWISE_H_
#define MATH_PIECEWISE_H_

/* -------------------------------------------------------------------------- *
 *                    OpenSim:  PiecewiseLinearFunction.h                     *
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

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace math {

//=============================================================================
//=============================================================================

/**
 * A class implementing a linear function.
 *
 * This class inherits from Function and so can be used as input to
 * any class requiring a Function as input.
 *
 * @author Peter Loan
 * @version 1.0
 */
class PiecewiseLinearFunction : public CustomFunction
{
  //=============================================================================
  // MEMBER VARIABLES
  //=============================================================================
public:
  // PROPERTIES
  /** Array of values for the independent variables (i.e., the knot
  sequence).  This array must be monotonically increasing. */
  std::vector<s_t> _x;

  /** Y values. */
  std::vector<s_t> _y;

private:
  std::vector<s_t> _b;

  //=============================================================================
  // METHODS
  //=============================================================================
public:
  //--------------------------------------------------------------------------
  // CONSTRUCTION
  //--------------------------------------------------------------------------
  PiecewiseLinearFunction(std::vector<s_t> x, std::vector<s_t> y);

  //--------------------------------------------------------------------------
  // SET AND GET
  //--------------------------------------------------------------------------
public:
  int getSize() const;
  const std::vector<s_t>& getX() const;
  const std::vector<s_t>& getY() const;
  s_t getX(int aIndex) const;
  s_t getY(int aIndex) const;
  void setX(int aIndex, s_t aValue);
  void setY(int aIndex, s_t aValue);
  bool deletePoint(int aIndex);
  int addPoint(s_t aX, s_t aY);

  //--------------------------------------------------------------------------
  // EVALUATION
  //--------------------------------------------------------------------------
  s_t calcValue(s_t x) const override;
  s_t calcDerivative(int order, s_t x) const override;
  std::shared_ptr<CustomFunction> offsetBy(s_t y) const override;

private:
  void calcCoefficients();

  //=============================================================================
}; // END class PiecewiseLinearFunction

}; // namespace math
}; // namespace dart

#endif
