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

#include <iostream>
#include <gtest/gtest.h>
#include "dart/dart.hpp"
#include "TestHelpers.hpp"

using namespace dart;

//==============================================================================
TEST(SO3, Canonicals)
{
  EXPECT_TRUE(SO3d<SO3CanonicalRep>::isCanonical());
  EXPECT_TRUE(SO3d<RotationMatrixRep>::isCanonical());
  EXPECT_FALSE(SO3d<AxisAngleRep>::isCanonical());

  const SO3d<SO3CanonicalRep> RCanonical;
  EXPECT_TRUE(RCanonical.isCanonical());
  EXPECT_TRUE(RCanonical.canonical().isCanonical());

  const SO3d<AxisAngleRep> RNonCanonical;
  EXPECT_FALSE(RNonCanonical.isCanonical());
  EXPECT_TRUE(RNonCanonical.canonical().isCanonical());
}

//==============================================================================
template <typename Derived>
void genericSO3(const math::SO3Base<Derived>& so3)
{
  so3.matrix();
}

//==============================================================================
template <typename DerivedA, typename DerivedB>
void genericSO3(math::SO3Base<DerivedA> R1,
                math::SO3Base<DerivedB> R2)
{
  R1 = R2;

//  R1 *= R2;
}

//==============================================================================
TEST(SO3, FunctionsTakingGenericSO3AsParameters)
{
  genericSO3(SO3d<RotationMatrixRep>());
  genericSO3(SO3d<AxisAngleRep>());

  genericSO3(SO3d<AxisAngleRep>::Random(),
             SO3d<AxisAngleRep>::Random());

  genericSO3(SO3d<RotationMatrixRep>::Random(),
             SO3d<AxisAngleRep>::Random());
}

//==============================================================================
template <typename SO3Type>
void testSettersAndGetters()
{
  SO3Type point;

  point.setIdentity();
  EXPECT_TRUE(point == SO3Type::Identity());
}

//==============================================================================
TEST(SO3, SettersAndGetters)
{
  testSettersAndGetters<SO3d<RotationMatrixRep>>();
  testSettersAndGetters<SO3d<AxisAngleRep>>();
  testSettersAndGetters<SO3d<QuaternionRep>>();
  testSettersAndGetters<SO3d<RotationVectorRep>>();
  // EulerAngles
}

//==============================================================================
template <typename SO3Type>
void testGroupOperations()
{
  SO3Type w1 = SO3Type::Random();
  SO3Type w2 = SO3Type::Random();

  SO3Type w3 = w1;
  w3 *= w2;

  SO3Type w4 = w1 * w2;

  EXPECT_TRUE(w3.isApprox(w4));

  SO3Type inversed1 = w1.inverse();
  SO3Type inversed2 = w1;
  inversed2.inverseInPlace();

  EXPECT_TRUE(inversed1.isApprox(inversed2));
}

//==============================================================================
TEST(SO3, GroupOperations)
{
  testGroupOperations<SO3d<RotationMatrixRep>>();
  testGroupOperations<SO3d<AxisAngleRep>>();
  testGroupOperations<SO3d<QuaternionRep>>();
}

//==============================================================================
template <typename SO3Type>
void testLieAlgebraOperations()
{
  typename SO3Type::Tangent tangent = SO3Type::Tangent::Random();

  EXPECT_TRUE(SO3Type::vee(SO3Type::hat(tangent)) == tangent);
}

//==============================================================================
TEST(SO3, LieAlgebraOperations)
{
  testLieAlgebraOperations<SO3d<RotationMatrixRep>>();
  testLieAlgebraOperations<SO3d<AxisAngleRep>>();
  testLieAlgebraOperations<SO3d<QuaternionRep>>();
}

//==============================================================================
template <typename SO3Type>
void testExponentialAndLogarithm()
{
  using so3 = typename SO3Type::so3;

  EXPECT_TRUE(SO3Type::exp(so3::Zero()) == SO3Type::Identity());
  EXPECT_TRUE(SO3Type::log(SO3Type::Identity()) == so3::Zero());
}

//==============================================================================
TEST(SO3, ExponentialAndLogarithm)
{
  testExponentialAndLogarithm<SO3d<RotationMatrixRep>>();
  testExponentialAndLogarithm<SO3d<AxisAngleRep>>();
  //testExponentialAndLogarithm<SO3d<SO3Quaternion>>(); // TODO(JS): not implemented yet
}

//==============================================================================
TEST(SO3, HeterogeneousAssignment)
{
  SO3<double, RotationMatrixRep> r1;
  SO3<double, AxisAngleRep> r2;

  r1.setRandom();
  r2.setRandom();
  EXPECT_FALSE(r1.isApprox(r2));

  r1 = r2;
  EXPECT_TRUE(r1.isApprox(r2));
}

//==============================================================================
TEST(SO3, HeterogeneousGroupMultiplication)
{
  SO3<double, RotationMatrixRep> w1;
  SO3<double, AxisAngleRep> w2;

  w1.setRandom();
  w2.setRandom();
  EXPECT_FALSE(w1.isApprox(w2));

  SO3<double, RotationMatrixRep> w3 = w1;
  EXPECT_TRUE(w3.isApprox(w1));
  w3 *= w2;

  SO3<double, AxisAngleRep> w4 = w1 * w2;

  EXPECT_TRUE(w3.isApprox(w4));
}

//==============================================================================
TEST(SO3, GeneralizedCoordinates)
{
  SO3<double> R = SO3<double>::Random();

  std::cout << R.template coordinates<AxisAngleRep>().transpose() << std::endl;
  std::cout << R.template coordinates<RotationMatrixRep>().transpose() << std::endl;
}

//==============================================================================
template <typename SO3Type>
void testInteractingWithRegularMatrices()
{
  using S = typename SO3Type::S;
  using Matrix3 = Eigen::Matrix<S, 3, 3>;

  SO3Type so3a = SO3Type::Random();

  // Assign to raw matrix
  Matrix3 rawMat3a = so3a.toRotationMatrix();
  EXPECT_TRUE(rawMat3a.isApprox(so3a.toRotationMatrix()));

  // Construct from raw matrix
  SO3Type so3b;
  so3b.fromRotationMatrix(rawMat3a);
  EXPECT_TRUE(so3b.toRotationMatrix().isApprox(rawMat3a));
}

//==============================================================================
TEST(SO3, InteractingWithRegularMatrices)
{
  testInteractingWithRegularMatrices<SO3<double>>();
  testInteractingWithRegularMatrices<SO3<double, AxisAngleRep>>();
  testInteractingWithRegularMatrices<SO3<double, QuaternionRep>>();
}

//==============================================================================
TEST(SO3, EigenTest)
{
  Eigen::Quaterniond q;
  Eigen::AngleAxisd aa;

//  Eigen::Matrix3d R;

  q = aa;
  aa = q;
}

//==============================================================================
int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
