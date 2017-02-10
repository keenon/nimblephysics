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
using namespace math;

//==============================================================================
template <typename SO3T, typename RepDataT, bool IsCoordinates>
void testTraits()
{
  auto resRepData = std::is_same<
      typename math::detail::Traits<SO3T>::RepData, RepDataT>::value;
  EXPECT_TRUE(resRepData);

  auto isCoordinates = math::detail::Traits<SO3T>::IsCoordinates;
  EXPECT_EQ(isCoordinates, IsCoordinates);
}

//==============================================================================
TEST(SO3, DETAIL_TRAITS)
{
  testTraits<math::SO3Matrixd, Eigen::Matrix3d, false>();
  testTraits<math::SO3Vectord, Eigen::Vector3d, true>();
  testTraits<math::AngleAxisd, Eigen::AngleAxisd, false>();
  testTraits<math::Quaterniond, Eigen::Quaterniond, false>();
  testTraits<math::EulerXYZd, Eigen::Vector3d, true>();
  testTraits<math::EulerZYXd, Eigen::Vector3d, true>();
}

//==============================================================================
TEST(SO3, DETAIL_EIGEN_TYPE_CHECKER)
{
  using math::detail::SO3RepDataIsEigenRotationBase3Impl;
  using math::detail::SO3RepDataIsEigenMatrixBaseImpl;

  EXPECT_TRUE(SO3RepDataIsEigenRotationBase3Impl<math::AngleAxisd>::value);
  EXPECT_TRUE(SO3RepDataIsEigenRotationBase3Impl<math::Quaterniond>::value);

  EXPECT_TRUE(SO3RepDataIsEigenMatrixBaseImpl<math::SO3Matrixd>::value);
  EXPECT_TRUE(SO3RepDataIsEigenMatrixBaseImpl<math::SO3Vectord>::value);
  EXPECT_TRUE(SO3RepDataIsEigenMatrixBaseImpl<math::EulerXYZd>::value);
  EXPECT_TRUE(SO3RepDataIsEigenMatrixBaseImpl<math::EulerZYXd>::value);
}

//==============================================================================
TEST(SO3, DETAIL_EXP_LOG)
{
  // TODO(JS)
}

//==============================================================================
template <typename T>
void testDetailSfinaeTestForEigenAssignment()
{
  using math::detail::EigenHasAssignmentOperatorImpl;

  bool res;

  res = EigenHasAssignmentOperatorImpl<T, Eigen::Matrix3d>::value;
  EXPECT_TRUE(res);

  res = EigenHasAssignmentOperatorImpl<T, Eigen::AngleAxisd>::value;
  EXPECT_TRUE(res);

  res = EigenHasAssignmentOperatorImpl<T, Eigen::Quaterniond>::value;
  EXPECT_TRUE(res);
}

//==============================================================================
TEST(SO3, DetailSfinaeTestForEigenAssignment)
{
  testDetailSfinaeTestForEigenAssignment<Eigen::Matrix3d>();
  testDetailSfinaeTestForEigenAssignment<Eigen::AngleAxisd>();
  testDetailSfinaeTestForEigenAssignment<Eigen::Quaterniond>();
}

//==============================================================================
TEST(SO3, DETAIL_SO3RepDataIsSupportedByEigenImpl)
{
  using math::detail::SO3RepDataIsSupportedByEigenImpl;

  bool res;

  // SO3Matrix

//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Matrixd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Vectord>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::AngleAxisd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::Quaterniond>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerXYZd>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerZYXd>::value;
//  EXPECT_FALSE(res);

//  // SO3Vector

//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Matrixd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Vectord>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::AngleAxisd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::Quaterniond>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerXYZd>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerZYXd>::value;
//  EXPECT_FALSE(res);

//  // AngleAxis

//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Matrixd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Vectord>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::AngleAxisd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::Quaterniond>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerXYZd>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerZYXd>::value;
//  EXPECT_FALSE(res);

//  // Quaternion

//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Matrixd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::SO3Vectord>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::AngleAxisd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::Quaterniond>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerXYZd>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::SO3Matrixd, math::EulerZYXd>::value;
//  EXPECT_FALSE(res);

//  // EulerXYZ

//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerXYZd, math::SO3Matrixd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerXYZd, math::SO3Vectord>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerXYZd, math::AngleAxisd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerXYZd, math::Quaterniond>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerXYZd, math::EulerXYZd>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerXYZd, math::EulerZYXd>::value;
//  EXPECT_FALSE(res);

//  // EulerZYX

//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerZYXd, math::SO3Matrixd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerZYXd, math::SO3Vectord>::value;
//  EXPECT_FALSE(res);has_less_operator
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerZYXd, math::AngleAxisd>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerZYXd, math::Quaterniond>::value;
//  EXPECT_TRUE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerZYXd, math::EulerXYZd>::value;
//  EXPECT_FALSE(res);
//  res = SO3RepDataIsSupportedByEigenImpl<
//      math::EulerZYXd, math::EulerZYXd>::value;
//  EXPECT_FALSE(res);

  bool ress;

  ress = math::detail::EigenHasMuliplicationOperatorImpl<
      Eigen::Matrix3d, Eigen::Matrix3d>::value;
  EXPECT_TRUE(ress);

  ress = math::detail::EigenHasMuliplicationOperatorImpl<
      Eigen::AngleAxisd, Eigen::Vector3d>::value;
  EXPECT_TRUE(ress);

  auto a1 = Eigen::AngleAxisd();
  auto a2 = Eigen::Vector3d();
  a1 * a2;
}

////==============================================================================
//TEST(SO3, SO3MatrixInteractWithEigen)
//{
//  Eigen::Vector3d r = Eigen::Vector3d::Random();
//  Eigen::Matrix3d eig3x3 = SO3d::Exp(r).toRotationMatrix();
//  Eigen::MatrixXd eigXxX = SO3d::Exp(r).toRotationMatrix();

//  SO3Matrixd so3Mat1 = eig3x3;
//  SO3Matrixd so3Mat2 = eigXxX;
//  EXPECT_TRUE(so3Mat1.toRotationMatrix() == eig3x3);
//  EXPECT_TRUE(so3Mat2.toRotationMatrix() == eigXxX);

//  SO3Matrixd so3Mat3 = eig3x3 * eig3x3;
//  SO3Matrixd so3Mat4 = eigXxX * eigXxX;
//  EXPECT_TRUE(so3Mat3.toRotationMatrix() == eig3x3 * eig3x3);
//  EXPECT_TRUE(so3Mat4.toRotationMatrix() == eigXxX * eigXxX);
//}

//////==============================================================================
////TEST(SO3, Canonicals)
////{
////  EXPECT_TRUE(SO3d::isCanonical());
////  EXPECT_TRUE(SO3d::isCanonical());
////  EXPECT_TRUE(SO3Matrixd::isCanonical());
////  EXPECT_TRUE(SO3Matrixd::isCanonical());
////  EXPECT_FALSE(SO3Vectord::isCanonical());
////  EXPECT_FALSE(SO3Vectord::isCanonical());
////  EXPECT_FALSE(AngleAxisd::isCanonical());
////  EXPECT_FALSE(AngleAxisd::isCanonical());
////  EXPECT_FALSE(Quaterniond::isCanonical());
////  EXPECT_FALSE(Quaterniond::isCanonical());

////  const SO3<double, SO3Canonical> RCanonical;
////  EXPECT_TRUE(RCanonical.isCanonical());
////  EXPECT_TRUE(RCanonical.canonical().isCanonical());

////  const AngleAxisd RNonCanonical;
////  EXPECT_FALSE(RNonCanonical.isCanonical());
////  EXPECT_TRUE(RNonCanonical.canonical().isCanonical());
////}

////==============================================================================
//template <typename Derived>
//void genericSO3(const math::SO3Base<Derived>& so3)
//{
//  so3.toRotationMatrix();
//}

//////==============================================================================
////template <typename DerivedA, typename DerivedB>
////void genericSO3(math::SO3Base<DerivedA> R1,
////                math::SO3Base<DerivedB> R2)
////{
////  R1 = R2;

//////  R1 *= R2;
////}

////==============================================================================
//TEST(SO3, FunctionsTakingGenericSO3AsParameters)
//{
//  genericSO3(math::SO3Matrixd());
//  genericSO3(math::AngleAxisd());

////  genericSO3(AngleAxisd::Random(),
////             AngleAxisd::Random());

////  genericSO3(SO3Matrixd::Random(),
////             AngleAxisd::Random());
//}

////==============================================================================
//template <typename SO3Type>
//void testSettersAndGetters()
//{
//  SO3Type point;

//  point.setIdentity();
//  EXPECT_TRUE(point == SO3Type::Identity());
//}

////==============================================================================
//TEST(SO3, SettersAndGetters)
//{
//  testSettersAndGetters<math::SO3Matrixd>();
//  testSettersAndGetters<math::SO3Vectord>();
//  testSettersAndGetters<math::AngleAxisd>();
//  testSettersAndGetters<math::Quaterniond>();
//  // EulerAngles
//}

//////==============================================================================
////template <typename SO3Type>
////void testGroupOperations()
////{
////  SO3Type w1 = SO3Type::Random();
////  SO3Type w2 = SO3Type::Random();

////  SO3Type w3 = w1;
////  w3 *= w2;

////  SO3Type w4 = w1 * w2;

////  EXPECT_TRUE(w3.isApprox(w4));

////  SO3Type inverse1 = w1.getInverse();
////  SO3Type inverse2 = w1;
////  inverse2.invert();

////  EXPECT_TRUE(inverse1.isApprox(inverse2));
////}

//////==============================================================================
////TEST(SO3, GroupOperations)
////{
////  testGroupOperations<SO3Matrixd>();
////  testGroupOperations<SO3Vectord>();
////  testGroupOperations<AngleAxisd>();
////  testGroupOperations<Quaterniond>();
////}

//////==============================================================================
////template <typename SO3Type>
////void testLieAlgebraOperations()
////{
////  typename SO3Type::Tangent tangent = SO3Type::Tangent::Random();

////  EXPECT_TRUE(SO3Type::Vee(SO3Type::Hat(tangent)) == tangent);
////}

//////==============================================================================
////TEST(SO3, LieAlgebraOperations)
////{
////  testLieAlgebraOperations<SO3Matrixd>();
////  testLieAlgebraOperations<SO3Vectord>();
////  testLieAlgebraOperations<AngleAxisd>();
////  testLieAlgebraOperations<Quaterniond>();
////}

//////==============================================================================
////template <typename SO3Type>
////void testExponentialAndLogarithm()
////{
////  using so3 = typename SO3Type::so3;

////  EXPECT_TRUE(SO3Type::Exp(so3::Zero()) == SO3Type::Identity());
////  EXPECT_TRUE(SO3Type::Log(SO3Type::Identity()) == so3::Zero());

////  const auto numTests = 100u;
////  for (auto i = 0u; i < numTests; ++i)
////  {
////    so3 w1 = so3::Random();
////    EXPECT_TRUE(w1.isApprox(SO3Type::Log(SO3Type::Exp(w1))));

////    SO3Type W;
////    W.setExp(w1);
////    EXPECT_TRUE(w1.isApprox(W.getLog()));
////  }
////}

//////==============================================================================
////TEST(SO3, ExponentialAndLogarithm)
////{
////  testExponentialAndLogarithm<SO3Matrixd>();
////  testExponentialAndLogarithm<SO3Vectord>();
////  testExponentialAndLogarithm<AngleAxisd>();
////  testExponentialAndLogarithm<Quaterniond>();
////}

//////==============================================================================
////TEST(SO3, HeterogeneousAssignment)
////{
////  SO3Matrixd r1;
////  AngleAxisd r2;

////  r1.setRandom();
////  r2.setRandom();
////  EXPECT_FALSE(r1.isApprox(r2));

////  r1 = r2;
////  EXPECT_TRUE(r1.isApprox(r2));
////}

//////==============================================================================
////TEST(SO3, HeterogeneousGroupMultiplication)
////{
////  SO3Matrixd w1;
////  AngleAxisd w2;

////  w1.setRandom();
////  w2.setRandom();
////  EXPECT_FALSE(w1.isApprox(w2));

////  SO3Matrixd w3 = w1;
////  EXPECT_TRUE(w3.isApprox(w1));
////  w3 *= w2;

////  AngleAxisd w4 = w1 * w2;

////  EXPECT_TRUE(w3.isApprox(w4));
////}

////==============================================================================
//TEST(SO3, GeneralizedCoordinates)
//{
////  SO3d R = SO3d::Random();

////  Eigen::VectorXd axisAngle = R.getLog();

////  Eigen::VectorXd coords = R.getCoordinates<SO3Vectord>();

////  EXPECT_TRUE(axisAngle.isApprox(coords));
//}

//////==============================================================================
////template <typename SO3Type>
////void testInteractingWithRegularMatrices()
////{
////  using S = typename SO3Type::S;
////  using Matrix3 = Eigen::Matrix<S, 3, 3>;

////  SO3Type so3a = SO3Type::Random();

////  // Assign to raw matrix
////  Matrix3 rawMat3a = so3a.toRotationMatrix();
////  EXPECT_TRUE(rawMat3a.isApprox(so3a.toRotationMatrix()));

////  // Construct from raw matrix
////  SO3Type so3b;
////  so3b.fromRotationMatrix(rawMat3a);
////  EXPECT_TRUE(so3b.toRotationMatrix().isApprox(rawMat3a));
////}

//////==============================================================================
////TEST(SO3, InteractingWithRegularMatrices)
////{
////  testInteractingWithRegularMatrices<SO3Matrixd>();
////  testInteractingWithRegularMatrices<SO3Vectord>();
////  testInteractingWithRegularMatrices<AngleAxisd>();
////  testInteractingWithRegularMatrices<Quaterniond>();
////}

//==============================================================================
template <typename T>
void testConstruction()
{
  // TODO(JS): Implement
//  T(Eigen::Matrix3d   ()).isApprox(T::Identity());
//  T(Eigen::AngleAxisd ()).isApprox(T::Identity());
//  T(Eigen::Quaterniond()).isApprox(T::Identity());
  T(math::SO3Matrixd  ()).isApprox(T::Identity());
//  T(math::SO3Vectord  ()).isApprox(T::Identity());
//  T(math::AngleAxisd  ()).isApprox(T::Identity());
//  T(math::Quaterniond ()).isApprox(T::Identity());
//  T(math::EulerXYZd   ()).isApprox(T::Identity());
//  T(math::EulerZYXd   ()).isApprox(T::Identity());
}

//==============================================================================
TEST(SO3, Constructions)
{
  testConstruction<math::SO3Matrixd>();
//  testConstruction<math::SO3Vectord>();
//  testConstruction<math::SO3Vectord>();
//  testConstruction<math::AngleAxisd>();
//  testConstruction<math::Quaterniond>();
//  testConstruction<math::EulerXYZd>();
//  testConstruction<math::EulerZYXd>();
}

//==============================================================================
TEST(SO3, Conversions)
{
//  const math::SO3d randomSO3 = math::SO3d::Random();

//  const Eigen::Matrix3d    eigMatIn  = randomSO3.to<Eigen::Matrix3d>();
//  const Eigen::AngleAxisd  eigAaIn   = randomSO3.to<Eigen::AngleAxisd>();
//  const Eigen::Quaterniond eigQuatIn = randomSO3.to<Eigen::Quaterniond>();
//  const math::SO3Matrixd   so3MatIn  = randomSO3;
//  const math::SO3Vectord   so3VecIn  = randomSO3;
//  const math::AngleAxisd   so3AaIn   = randomSO3;
//  const math::Quaterniond  so3QuatIn = randomSO3;
//  const math::EulerXYZd    so3XYZIn  = randomSO3;
//  const math::EulerZYXd    so3ZYXIn  = randomSO3;

//  Eigen::Matrix3d    eigMatOut;
//  Eigen::AngleAxisd  eigAaOut;
//  Eigen::Quaterniond eigQuatOut;
//  math::SO3Matrixd   so3MatOut;
//  math::SO3Vectord   so3VecOut;
//  math::AngleAxisd   so3AaOut;
//  math::Quaterniond  so3QuatOut;
//  math::EulerXYZd    so3XYZOut;
//  math::EulerZYXd    so3ZYXOut;

//  //------------------------------
//  // Eigen::Matrix3d <- various types
//  //------------------------------
//  eigMatOut = eigMatIn;
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = eigAaIn;
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = eigQuatIn;
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3MatIn.toRotationMatrix();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3MatIn.to<Eigen::Matrix3d>();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3VecIn.toRotationMatrix();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3VecIn.to<Eigen::Matrix3d>();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3AaIn.toRotationMatrix();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3AaIn.to<Eigen::Matrix3d>();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3QuatIn.toRotationMatrix();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3QuatIn.to<Eigen::Matrix3d>();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3XYZIn.to<Eigen::Matrix3d>();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  eigMatOut = so3ZYXIn.to<Eigen::Matrix3d>();
//  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

//  //------------------------------
//  // Eigen::AngleAxisd <- various types
//  //------------------------------
//  eigAaOut = eigMatIn;
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = eigAaIn;
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = eigQuatIn;
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3MatIn.to<math::AngleAxisd>().getRepData();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3MatIn.to<Eigen::AngleAxisd>();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3VecIn.to<math::AngleAxisd>().getRepData();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3VecIn.to<Eigen::AngleAxisd>();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3AaIn.to<math::AngleAxisd>().getRepData();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3AaIn.to<Eigen::AngleAxisd>();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3QuatIn.to<math::AngleAxisd>().getRepData();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3QuatIn.to<Eigen::AngleAxisd>();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3XYZIn.to<Eigen::AngleAxisd>();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  eigAaOut = so3ZYXIn.to<Eigen::AngleAxisd>();
//  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

//  //------------------------------
//  // Eigen::Quaterniond <- various types
//  //------------------------------
//  eigQuatOut = eigMatIn;
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = eigAaIn;
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = eigQuatIn;
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3MatIn.to<math::Quaterniond>().getRepData();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3MatIn.to<Eigen::Quaterniond>();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3VecIn.to<math::Quaterniond>().getRepData();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3VecIn.to<Eigen::Quaterniond>();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3AaIn.to<math::Quaterniond>().getRepData();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3AaIn.to<Eigen::Quaterniond>();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3QuatIn.to<math::Quaterniond>().getRepData();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3QuatIn.to<Eigen::Quaterniond>();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3XYZIn.to<Eigen::Quaterniond>();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  eigQuatOut = so3ZYXIn.to<Eigen::Quaterniond>();
//  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

//  //------------------------------
//  // SO3Matrixd <- various types
//  //------------------------------
//  so3MatOut = eigMatIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = eigAaIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = eigQuatIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = so3MatIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = so3VecIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = so3AaIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = so3QuatIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = so3XYZIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  so3MatOut = so3ZYXIn;
//  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

//  //------------------------------
//  // SO3Vectord <- various types
//  //------------------------------
//  so3VecOut = eigMatIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = eigAaIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = eigQuatIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = so3MatIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = so3VecIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = so3AaIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = so3QuatIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = so3XYZIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  so3VecOut = so3ZYXIn;
//  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

//  //------------------------------
//  // AngleAxisd <- various types
//  //------------------------------
//  so3AaOut = eigMatIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = eigAaIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = eigQuatIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = so3MatIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = so3VecIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = so3AaIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = so3QuatIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = so3XYZIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  so3AaOut = so3ZYXIn;
//  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

//  //------------------------------
//  // Quaterniond <- various types
//  //------------------------------
//  so3QuatOut = eigMatIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = eigAaIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = eigQuatIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = so3MatIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = so3VecIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = so3AaIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = so3QuatIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = so3XYZIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  so3QuatOut = so3ZYXIn;
//  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

//  //------------------------------
//  // EulerXYZ <- various types
//  //------------------------------
//  so3XYZOut = eigMatIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = eigAaIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = eigQuatIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = so3MatIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = so3VecIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = so3AaIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = so3QuatIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = so3XYZIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  so3XYZOut = so3ZYXIn;
//  EXPECT_TRUE(so3XYZOut.isApprox(so3XYZIn));

//  //------------------------------
//  // EulerZYX <- various types
//  //------------------------------
//  so3ZYXOut = eigMatIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = eigAaIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = eigQuatIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = so3MatIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = so3VecIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = so3AaIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = so3QuatIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = so3XYZIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));

//  so3ZYXOut = so3ZYXIn;
//  EXPECT_TRUE(so3ZYXOut.isApprox(so3ZYXIn));
}

////==============================================================================
//TEST(SO3, GroupMultiplication)
//{
//  // TODO(JS):
//  auto res = math::detail::SO3RepDataIsSupportedByEigenImpl<math::AngleAxisd, math::SO3Matrixd>::value;
//  EXPECT_TRUE(res);
//}

////==============================================================================
//TEST(SO3, Performance)
//{
//  Eigen::Vector3d r = Eigen::Vector3d::Random();

//  math::SO3Matrixd so3Mat = math::SO3Matrixd::Exp(r);
//  math::SO3Vectord so3Vec = math::SO3Vectord::Exp(r);
//  math::AngleAxisd so3Aa = math::AngleAxisd::Exp(r);
//  math::Quaterniond so3Quat = math::Quaterniond::Exp(r);

//  std::cout << so3Quat.toRotationMatrix() << std::endl;
//  std::cout << so3Mat.toRotationMatrix() << std::endl;

//  Eigen::Matrix3d eigMat = SO3d::Exp(r).toRotationMatrix();
//  Eigen::AngleAxisd eigAa(eigMat);
//  Eigen::Quaterniond eigQuat(eigMat);

//  math::SO3d::Random();

//  EXPECT_TRUE(so3Mat.isApprox(eigMat));
//  EXPECT_TRUE(so3Aa.isApprox(eigAa));
//  EXPECT_TRUE(so3Quat.isApprox(eigQuat));
//#ifdef NDEBUG // release mode
//  const auto numTests = 1e+4;
//#else
//  const auto numTests = 1e+2;
//#endif
//  common::Timer t;

//  //----------------------------------------------------------------------------
//  //----------------------------------------------------------------------------
//  std::cout << "[ R = R * R ]" << std::endl;
//  std::cout << std::endl;
//  //----------------------------------------------------------------------------
//  //----------------------------------------------------------------------------

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    eigMat = eigMat * eigMat;
//  }
//  t.stop();
//  std::cout << "Eigen::Matrix3d  : " << t.getLastElapsedTime() << " (sec)" << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Mat = so3Mat * so3Mat;
//  }
//  t.stop();
//  std::cout << "math::SO3Matrixd : " << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Vec = so3Vec * so3Vec;
//  }
//  t.stop();
//  std::cout << "math::SO3Vectord : " << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    eigAa = eigAa * eigAa;
//  }
//  t.stop();
//  std::cout << "Eigen::AngleAxisd: " << t.getLastElapsedTime() << " (sec)" << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Aa = so3Aa * so3Aa;
//  }
//  t.stop();
//  std::cout << "math::AngleAxisd :" << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    eigQuat = eigQuat * eigQuat;
//  }
//  t.stop();
//  std::cout << "Eigen::Quaterniond: " << t.getLastElapsedTime() << " (sec)" << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Quat = so3Quat * so3Quat;
//  }
//  t.stop();
//  std::cout << "math::Quaterniond :" << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  //----------------------------------------------------------------------------
//  //----------------------------------------------------------------------------
//  std::cout << "[ R *= R ]" << std::endl;
//  std::cout << std::endl;
//  //----------------------------------------------------------------------------
//  //----------------------------------------------------------------------------

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    eigMat *= eigMat;
//  }
//  t.stop();
//  std::cout << "Eigen::Matrix3d   : " << t.getLastElapsedTime() << " (sec)" << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Mat *= so3Mat;
//  }
//  t.stop();
//  std::cout << "math::SO3Matrixd  : " << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Vec *= so3Vec;
//  }
//  t.stop();
//  std::cout << "math::SO3Vectord:   " << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  // AngleAxis *= AngleAxis is not supported by Eigen
//  //t.start();
//  //for (auto i = 0u; i < numTests; ++i)
//  //{
//  //  eigAa *= eigAa;
//  //}
//  //t.stop();
//  //std::cout << "Eigen::AngleAxisd : " << t.getLastElapsedTime() << " (sec)" << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Aa *= so3Aa;
//  }
//  t.stop();
//  std::cout << "math::AngleAxisd: " << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    eigQuat *= eigQuat;
//  }
//  t.stop();
//  std::cout << "Eigen::Quaterniond: " << t.getLastElapsedTime() << " (sec)" << std::endl;

//  t.start();
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3Quat *= so3Quat;
//  }
//  t.stop();
//  std::cout << "math::Quaterniond : " << t.getLastElapsedTime() << " (sec)" << std::endl;
//  std::cout << std::endl;

//  std::cout << eigQuat.toRotationMatrix() << std::endl;
//  std::cout << eigMat << std::endl;

//  std::cout << so3Quat.toRotationMatrix() << std::endl;
//  std::cout << so3Mat.toRotationMatrix() << std::endl;
//}

//==============================================================================
int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
