#ifndef MATH_OPENSIM_SIMMSPLINE_H_
#define MATH_OPENSIM_SIMMSPLINE_H_
/*
 * This file is adapted from the original in OpenSim, originally under the
 * Apache license, and is included in Nimble under the MIT license.
 */

/* -------------------------------------------------------------------------- *
 *                           OpenSim:  SimmSpline.h                           *
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

// INCLUDES
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dart/math/CustomFunction.hpp"
#include "dart/math/MathTypes.hpp"

//=============================================================================
//=============================================================================
namespace dart {
namespace math {

/**
 * A class implementing a smooth function with a cubic spline as
 * implemented in SIMM. Use a SIMM Spline if you want to reproduce
 * the behavior of a joint function created in SIMM.
 *
 * This class inherits from Function and so can be used as input to
 * any class requiring a Function as input.
 *
 * @author Peter Loan
 * @version 1.0
 */
class SimmSpline : public CustomFunction
{
  //=============================================================================
  // MEMBER VARIABLES
  //=============================================================================
protected:
  // PROPERTIES
  /** Array of values for the independent variables (i.e., the spline knot
  sequence).  This array must be monotonically increasing. */
  std::vector<s_t> _x;

  /** Y values. */
  std::vector<s_t> _y;

private:
  std::vector<s_t> _b;
  std::vector<s_t> _c;
  std::vector<s_t> _d;

  //=============================================================================
  // METHODS
  //=============================================================================
public:
  //--------------------------------------------------------------------------
  // CONSTRUCTION
  //--------------------------------------------------------------------------
  SimmSpline(std::vector<s_t> x, std::vector<s_t> y);

private:
  void setNull();
  void setupProperties();
  void setEqual(const SimmSpline& aSpline);

  //--------------------------------------------------------------------------
  // OPERATORS
  //--------------------------------------------------------------------------
public:
#ifndef SWIG
  SimmSpline& operator=(const SimmSpline& aSpline);
#endif
  //--------------------------------------------------------------------------
  // SET AND GET
  //--------------------------------------------------------------------------
public:
  int getSize() const;
  const std::vector<s_t>& getX() const;
  const std::vector<s_t>& getY() const;
  int getNumberOfPoints() const;
  s_t getX(int aIndex) const;
  s_t getY(int aIndex) const;
  void setX(int aIndex, s_t aValue);
  void setY(int aIndex, s_t aValue);
  bool deletePoint(int aIndex);
  int addPoint(s_t aX, double aY);

  //--------------------------------------------------------------------------
  // EVALUATION
  //--------------------------------------------------------------------------
  s_t calcValue(s_t x) const override;
  s_t calcDerivative(int order, s_t x) const override;
  int getArgumentSize() const;
  int getMaxDerivativeOrder() const;

  s_t finiteDifferenceFirstDerivative(s_t x, bool useRidders = true);

private:
  void calcCoefficients();
  //=============================================================================
}; // END class SimmSpline

}; // namespace math
}; // namespace dart
//=============================================================================
//=============================================================================

#endif // OPENSIM_SIMMSPLINE_H_
