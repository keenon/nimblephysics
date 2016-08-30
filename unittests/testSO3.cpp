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
template <typename Derived>
void genericSO3(const math::SO3Base<Derived>& so3)
{
  so3.matrix();
}

//==============================================================================
TEST(SO3, FunctionsTakingGenericSO3AsParameters)
{
  genericSO3(SO3d<SO3Rep::RotationMatrix>());
  genericSO3(SO3d<SO3Rep::AngleAxis>());
}

//==============================================================================
template <typename SO3Type>
void testSettersAndGetters()
{
  SO3Type id = SO3Type::Identity();
  genericSO3(id);

  using S = typename SO3Type::S;
  using Matrix3 = Eigen::Matrix<S, 3, 3>;

  SO3Type point;

  point.setIdentity();
  EXPECT_TRUE(point.matrix() == Matrix3::Identity());
  EXPECT_TRUE(point == SO3Type::Identity());
}

//==============================================================================
TEST(SO3, SettersAndGetters)
{
  testSettersAndGetters<SO3d<SO3Rep::RotationMatrix>>();
  testSettersAndGetters<SO3d<SO3Rep::AngleAxis>>();
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

  std::cout << w3.matrix() << std::endl;
  std::cout << w4.matrix() << std::endl;

  EXPECT_TRUE(w3.isApprox(w4));
}

//==============================================================================
TEST(SO3, GroupOperations)
{
  testGroupOperations<SO3d<SO3Rep::RotationMatrix>>();
  testGroupOperations<SO3d<SO3Rep::AngleAxis>>();
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
  testLieAlgebraOperations<SO3d<SO3Rep::RotationMatrix>>();
  testLieAlgebraOperations<SO3d<SO3Rep::AngleAxis>>();
}

//==============================================================================
template <typename SO3Type>
void testExponentialAndLogarithm()
{
  using S = typename SO3Type::S;
  using Matrix3 = Eigen::Matrix<S, 3, 3>;
  using so3 = typename SO3Type::so3;

  EXPECT_TRUE(SO3Type::exp(so3::Zero()).matrix() == Matrix3::Identity());
  EXPECT_TRUE(SO3Type::log(SO3Type::Identity()) == so3::Zero());
}

//==============================================================================
TEST(SO3, ExponentialAndLogarithm)
{
  testExponentialAndLogarithm<SO3d<SO3Rep::RotationMatrix>>();
  testExponentialAndLogarithm<SO3d<SO3Rep::AngleAxis>>();
}

//==============================================================================
TEST(SO3, HeterogeneousAssignment)
{
  SO3<double, SO3Rep::RotationMatrix> r1;
  SO3<double, SO3Rep::AngleAxis> r2;

  r1.setRandom();
  r2.setRandom();
  EXPECT_FALSE(r1.isApprox(r2));

  r1 = r2;
  EXPECT_TRUE(r1.isApprox(r2));
}

//==============================================================================
int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
