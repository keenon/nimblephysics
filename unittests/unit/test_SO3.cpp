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
TEST(SO3, SFINAE)
{
  using namespace math::detail::SO3;

  auto resAa = rep_is_eigen_rotation_impl<double, math::AxisAngleRep>::value;
  EXPECT_TRUE(resAa);

  auto resQuat = rep_is_eigen_rotation_impl<double, math::QuaternionRep>::value;
  EXPECT_TRUE(resQuat);

  auto resMat3x3 = rep_is_eigen_matrix_impl<double, math::RotationMatrixRep>::value;
  EXPECT_TRUE(resMat3x3);

  auto resMatXxX = rep_is_eigen_matrix_impl<double, math::RotationVectorRep>::value;
  EXPECT_TRUE(resMatXxX);
}

//==============================================================================
TEST(SO3, SO3MatrixInteractWithEigen)
{
  Eigen::Vector3d r = Eigen::Vector3d::Random();
  Eigen::Matrix3d eig3x3 = SO3d::Exp(r).toRotationMatrix();
  Eigen::MatrixXd eigXxX = SO3d::Exp(r).toRotationMatrix();

  SO3Matrixd so3Mat1 = eig3x3;
  SO3Matrixd so3Mat2 = eigXxX;
  EXPECT_TRUE(so3Mat1.toRotationMatrix() == eig3x3);
  EXPECT_TRUE(so3Mat2.toRotationMatrix() == eigXxX);

  SO3Matrixd so3Mat3 = eig3x3 * eig3x3;
  SO3Matrixd so3Mat4 = eigXxX * eigXxX;
  EXPECT_TRUE(so3Mat3.toRotationMatrix() == eig3x3 * eig3x3);
  EXPECT_TRUE(so3Mat4.toRotationMatrix() == eigXxX * eigXxX);
}

////==============================================================================
//TEST(SO3, Canonicals)
//{
//  EXPECT_TRUE(SO3d::isCanonical());
//  EXPECT_TRUE(SO3d::isCanonical());
//  EXPECT_TRUE(SO3Matrixd::isCanonical());
//  EXPECT_TRUE(SO3Matrixd::isCanonical());
//  EXPECT_FALSE(SO3Vectord::isCanonical());
//  EXPECT_FALSE(SO3Vectord::isCanonical());
//  EXPECT_FALSE(SO3AxisAngled::isCanonical());
//  EXPECT_FALSE(SO3AxisAngled::isCanonical());
//  EXPECT_FALSE(SO3Quaterniond::isCanonical());
//  EXPECT_FALSE(SO3Quaterniond::isCanonical());

//  const SO3<double, SO3CanonicalRep> RCanonical;
//  EXPECT_TRUE(RCanonical.isCanonical());
//  EXPECT_TRUE(RCanonical.canonical().isCanonical());

//  const SO3<double, AxisAngleRep> RNonCanonical;
//  EXPECT_FALSE(RNonCanonical.isCanonical());
//  EXPECT_TRUE(RNonCanonical.canonical().isCanonical());
//}

////==============================================================================
//template <typename Derived>
//void genericSO3(const math::SO3Base<Derived>& so3)
//{
//  so3.toRotationMatrix();
//}

////==============================================================================
//template <typename DerivedA, typename DerivedB>
//void genericSO3(math::SO3Base<DerivedA> R1,
//                math::SO3Base<DerivedB> R2)
//{
//  R1 = R2;

////  R1 *= R2;
//}

////==============================================================================
//TEST(SO3, FunctionsTakingGenericSO3AsParameters)
//{
//  genericSO3(SO3<double, RotationMatrixRep>());
//  genericSO3(SO3<double, AxisAngleRep>());

//  genericSO3(SO3<double, AxisAngleRep>::Random(),
//             SO3<double, AxisAngleRep>::Random());

//  genericSO3(SO3<double, RotationMatrixRep>::Random(),
//             SO3<double, AxisAngleRep>::Random());
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
//  testSettersAndGetters<SO3<double, RotationMatrixRep>>();
//  testSettersAndGetters<SO3<double, RotationVectorRep>>();
//  testSettersAndGetters<SO3<double, AxisAngleRep>>();
//  testSettersAndGetters<SO3<double, QuaternionRep>>();
//  // EulerAngles
//}

////==============================================================================
//template <typename SO3Type>
//void testGroupOperations()
//{
//  SO3Type w1 = SO3Type::Random();
//  SO3Type w2 = SO3Type::Random();

//  SO3Type w3 = w1;
//  w3 *= w2;

//  SO3Type w4 = w1 * w2;

//  EXPECT_TRUE(w3.isApprox(w4));

//  SO3Type inverse1 = w1.getInverse();
//  SO3Type inverse2 = w1;
//  inverse2.invert();

//  EXPECT_TRUE(inverse1.isApprox(inverse2));
//}

////==============================================================================
//TEST(SO3, GroupOperations)
//{
//  testGroupOperations<SO3<double, RotationMatrixRep>>();
//  testGroupOperations<SO3<double, RotationVectorRep>>();
//  testGroupOperations<SO3<double, AxisAngleRep>>();
//  testGroupOperations<SO3<double, QuaternionRep>>();
//}

////==============================================================================
//template <typename SO3Type>
//void testLieAlgebraOperations()
//{
//  typename SO3Type::Tangent tangent = SO3Type::Tangent::Random();

//  EXPECT_TRUE(SO3Type::Vee(SO3Type::Hat(tangent)) == tangent);
//}

////==============================================================================
//TEST(SO3, LieAlgebraOperations)
//{
//  testLieAlgebraOperations<SO3<double, RotationMatrixRep>>();
//  testLieAlgebraOperations<SO3<double, RotationVectorRep>>();
//  testLieAlgebraOperations<SO3<double, AxisAngleRep>>();
//  testLieAlgebraOperations<SO3<double, QuaternionRep>>();
//}

////==============================================================================
//template <typename SO3Type>
//void testExponentialAndLogarithm()
//{
//  using so3 = typename SO3Type::so3;

//  EXPECT_TRUE(SO3Type::Exp(so3::Zero()) == SO3Type::Identity());
//  EXPECT_TRUE(SO3Type::Log(SO3Type::Identity()) == so3::Zero());

//  const auto numTests = 100u;
//  for (auto i = 0u; i < numTests; ++i)
//  {
//    so3 w1 = so3::Random();
//    EXPECT_TRUE(w1.isApprox(SO3Type::Log(SO3Type::Exp(w1))));

//    SO3Type W;
//    W.setExp(w1);
//    EXPECT_TRUE(w1.isApprox(W.getLog()));
//  }
//}

////==============================================================================
//TEST(SO3, ExponentialAndLogarithm)
//{
//  testExponentialAndLogarithm<SO3<double, RotationMatrixRep>>();
//  testExponentialAndLogarithm<SO3<double, RotationVectorRep>>();
//  testExponentialAndLogarithm<SO3<double, AxisAngleRep>>();
//  testExponentialAndLogarithm<SO3<double, QuaternionRep>>();
//}

////==============================================================================
//TEST(SO3, HeterogeneousAssignment)
//{
//  SO3<double, RotationMatrixRep> r1;
//  SO3<double, AxisAngleRep> r2;

//  r1.setRandom();
//  r2.setRandom();
//  EXPECT_FALSE(r1.isApprox(r2));

//  r1 = r2;
//  EXPECT_TRUE(r1.isApprox(r2));
//}

////==============================================================================
//TEST(SO3, HeterogeneousGroupMultiplication)
//{
//  SO3<double, RotationMatrixRep> w1;
//  SO3<double, AxisAngleRep> w2;

//  w1.setRandom();
//  w2.setRandom();
//  EXPECT_FALSE(w1.isApprox(w2));

//  SO3<double, RotationMatrixRep> w3 = w1;
//  EXPECT_TRUE(w3.isApprox(w1));
//  w3 *= w2;

//  SO3<double, AxisAngleRep> w4 = w1 * w2;

//  EXPECT_TRUE(w3.isApprox(w4));
//}

//==============================================================================
TEST(SO3, GeneralizedCoordinates)
{
  SO3d R = SO3d::Random();

  Eigen::VectorXd axisAngle = R.getLog();

  Eigen::VectorXd coords = R.getCoordinates<RotationVectorRep>();

  EXPECT_TRUE(axisAngle.isApprox(coords));
}

////==============================================================================
//template <typename SO3Type>
//void testInteractingWithRegularMatrices()
//{
//  using S = typename SO3Type::S;
//  using Matrix3 = Eigen::Matrix<S, 3, 3>;

//  SO3Type so3a = SO3Type::Random();

//  // Assign to raw matrix
//  Matrix3 rawMat3a = so3a.toRotationMatrix();
//  EXPECT_TRUE(rawMat3a.isApprox(so3a.toRotationMatrix()));

//  // Construct from raw matrix
//  SO3Type so3b;
//  so3b.fromRotationMatrix(rawMat3a);
//  EXPECT_TRUE(so3b.toRotationMatrix().isApprox(rawMat3a));
//}

////==============================================================================
//TEST(SO3, InteractingWithRegularMatrices)
//{
//  testInteractingWithRegularMatrices<SO3<double, RotationMatrixRep>>();
//  testInteractingWithRegularMatrices<SO3<double, RotationVectorRep>>();
//  testInteractingWithRegularMatrices<SO3<double, AxisAngleRep>>();
//  testInteractingWithRegularMatrices<SO3<double, QuaternionRep>>();
//}

//==============================================================================
TEST(SO3, CompatibilityToEigen)
{
  //Eigen::Matrix3d    eigMatIn  = Eigen::Matrix3d::Random(); // Not a rotation matrix
  //Eigen::AngleAxisd  eigAaIn   = Eigen::AngleAxisd::Random(); // Not supported by Eigen
  //Eigen::Quaterniond eigQuatIn = Eigen::Quaterniond::Random(); // Not supported by Eigen
  Eigen::Matrix3d    eigMatIn  = SO3Matrixd::Random().to<RotationMatrixRep>();
  Eigen::AngleAxisd  eigAaIn   = SO3AxisAngled::Random().to<AxisAngleRep>();
  Eigen::Quaterniond eigQuatIn = SO3Quaterniond::Random().to<QuaternionRep>();
  SO3Matrixd         so3MatIn  = SO3Matrixd::Random();
  SO3Vectord         so3VecIn  = SO3Vectord::Random();
  SO3AxisAngled      so3AaIn   = SO3AxisAngled::Random();
  SO3Quaterniond     so3QuatIn = SO3Quaterniond::Random();

  Eigen::Matrix3d    eigMatOut;
  Eigen::AngleAxisd  eigAaOut;
  Eigen::Quaterniond eigQuatOut;
  SO3Matrixd         so3MatOut;
  SO3Vectord         so3VecOut;
  SO3AxisAngled      so3AaOut;
  SO3Quaterniond     so3QuatOut;

  //------------------------------
  // Eigen::Matrix3d <- various types
  //------------------------------
  eigMatOut = eigMatIn;
  EXPECT_TRUE(eigMatOut.isApprox(eigMatIn));

  eigMatOut = eigAaIn;
  EXPECT_TRUE(eigMatOut.isApprox(eigAaIn.toRotationMatrix()));

  eigMatOut = eigQuatIn;
  EXPECT_TRUE(eigMatOut.isApprox(eigQuatIn.toRotationMatrix()));

  eigMatOut = so3MatIn.toRotationMatrix();
  EXPECT_TRUE(eigMatOut.isApprox(so3MatIn.toRotationMatrix()));

  eigMatOut = so3VecIn.toRotationMatrix();
  EXPECT_TRUE(eigMatOut.isApprox(so3VecIn.toRotationMatrix()));

  eigMatOut = so3AaIn.toRotationMatrix();
  EXPECT_TRUE(eigMatOut.isApprox(so3AaIn.toRotationMatrix()));

  eigMatOut = so3QuatIn.toRotationMatrix();
  EXPECT_TRUE(eigMatOut.isApprox(so3QuatIn.toRotationMatrix()));

  //------------------------------
  // Eigen::AngleAxisd <- various types
  //------------------------------
  eigAaOut = eigMatIn;
  EXPECT_TRUE(eigAaOut.toRotationMatrix().isApprox(eigMatIn));

  eigAaOut = eigAaIn;
  EXPECT_TRUE(eigAaOut.isApprox(eigAaIn));

  eigAaOut = eigQuatIn;
  EXPECT_TRUE(eigAaOut.toRotationMatrix().isApprox(eigQuatIn.toRotationMatrix()));

  eigAaOut = so3MatIn.to<AxisAngleRep>();
  EXPECT_TRUE(eigAaOut.isApprox(so3MatIn.to<AxisAngleRep>()));

  eigAaOut = so3VecIn.to<AxisAngleRep>();
  EXPECT_TRUE(eigAaOut.isApprox(so3VecIn.to<AxisAngleRep>()));

  eigAaOut = so3AaIn.to<AxisAngleRep>();
  EXPECT_TRUE(eigAaOut.isApprox(so3AaIn.to<AxisAngleRep>()));

  eigAaOut = so3QuatIn.to<AxisAngleRep>();
  EXPECT_TRUE(eigAaOut.isApprox(so3QuatIn.to<AxisAngleRep>()));

  //------------------------------
  // Eigen::AngleAxisd <- various types
  //------------------------------
  eigQuatOut = eigMatIn;
  EXPECT_TRUE(eigQuatOut.toRotationMatrix().isApprox(eigMatIn));

  eigQuatOut = eigAaIn;
  EXPECT_TRUE(eigQuatOut.toRotationMatrix().isApprox(eigAaIn.toRotationMatrix()));

  eigQuatOut = eigQuatIn;
  EXPECT_TRUE(eigQuatOut.isApprox(eigQuatIn));

  eigQuatOut = so3MatIn.to<QuaternionRep>();
  EXPECT_TRUE(eigQuatOut.isApprox(so3MatIn.to<QuaternionRep>()));

  eigQuatOut = so3VecIn.to<QuaternionRep>();
  EXPECT_TRUE(eigQuatOut.isApprox(so3VecIn.to<QuaternionRep>()));

  eigQuatOut = so3AaIn.to<QuaternionRep>();
  EXPECT_TRUE(eigQuatOut.isApprox(so3AaIn.to<QuaternionRep>()));

  eigQuatOut = so3QuatIn.to<QuaternionRep>();
  EXPECT_TRUE(eigQuatOut.isApprox(so3QuatIn.to<QuaternionRep>()));

  //------------------------------
  // SO3Matrixd <- various types
  //------------------------------
  so3MatOut = eigMatIn;
  EXPECT_TRUE(so3MatOut.isApprox(eigMatIn));

  so3MatOut = eigAaIn;
  EXPECT_TRUE(so3MatOut.isApprox(eigAaIn));

  so3MatOut = eigQuatIn;
  EXPECT_TRUE(so3MatOut.isApprox(eigQuatIn));

  so3MatOut = so3MatIn;
  EXPECT_TRUE(so3MatOut.isApprox(so3MatIn));

  so3MatOut = so3VecIn;
  EXPECT_TRUE(so3MatOut.isApprox(so3VecIn));

  so3MatOut = so3AaIn;
  EXPECT_TRUE(so3MatOut.isApprox(so3AaIn));

  so3MatOut = so3QuatIn;
  EXPECT_TRUE(so3MatOut.isApprox(so3QuatIn));

  //------------------------------
  // SO3Vectord <- various types
  //------------------------------
  //so3VecOut = eigMat; // Deleted to avoid ambiguity between rotation matrix
                        // and rotation vector
  so3VecOut.fromRotationMatrix(eigMatIn);
  EXPECT_TRUE(so3VecOut.isApprox(eigMatIn));

  so3VecOut = eigAaIn;
  EXPECT_TRUE(so3VecOut.isApprox(eigAaIn));

  so3VecOut = eigQuatIn;
  EXPECT_TRUE(so3VecOut.isApprox(eigQuatIn));

  so3VecOut = so3MatIn;
  EXPECT_TRUE(so3VecOut.isApprox(so3MatIn));

  so3VecOut = so3VecIn;
  EXPECT_TRUE(so3VecOut.isApprox(so3VecIn));

  so3VecOut = so3AaIn;
  EXPECT_TRUE(so3VecOut.isApprox(so3AaIn));

  so3VecOut = so3QuatIn;
  EXPECT_TRUE(so3VecOut.isApprox(so3QuatIn));

  //------------------------------
  // SO3AxisAngled <- various types
  //------------------------------
  so3AaOut = eigMatIn;
  EXPECT_TRUE(so3AaOut.isApprox(eigMatIn));

  so3AaOut = eigAaIn;
  EXPECT_TRUE(so3AaOut.isApprox(eigAaIn));

  so3AaOut = eigQuatIn;
  EXPECT_TRUE(so3AaOut.isApprox(eigQuatIn));

  so3AaOut = so3MatIn;
  EXPECT_TRUE(so3AaOut.isApprox(so3MatIn));

  so3AaOut = so3VecIn;
  EXPECT_TRUE(so3AaOut.isApprox(so3VecIn));

  so3AaOut = so3AaIn;
  EXPECT_TRUE(so3AaOut.isApprox(so3AaIn));

  so3AaOut = so3QuatIn;
  EXPECT_TRUE(so3AaOut.isApprox(so3QuatIn));

  //------------------------------
  // SO3AxisAngled <- various types
  //------------------------------
  so3QuatOut = eigMatIn;
  EXPECT_TRUE(so3QuatOut.isApprox(eigMatIn));

  so3QuatOut = eigAaIn;
  EXPECT_TRUE(so3QuatOut.isApprox(eigAaIn));

  so3QuatOut = eigQuatIn;
  EXPECT_TRUE(so3QuatOut.isApprox(eigQuatIn));

  so3QuatOut = so3MatIn;
  EXPECT_TRUE(so3QuatOut.isApprox(so3MatIn));

  so3QuatOut = so3VecIn;
  EXPECT_TRUE(so3QuatOut.isApprox(so3VecIn));

  so3QuatOut = so3AaIn;
  EXPECT_TRUE(so3QuatOut.isApprox(so3AaIn));

  so3QuatOut = so3QuatIn;
  EXPECT_TRUE(so3QuatOut.isApprox(so3QuatIn));

  //------------------------------
  // Heterogeneous multiplication
  //------------------------------

  // TODO(JS):
}

//==============================================================================
TEST(SO3, Performance)
{
  Eigen::Vector3d r = Eigen::Vector3d::Random();

  SO3Matrixd so3Mat = SO3Matrixd::Exp(r);
  SO3Vectord so3Vec = SO3Vectord::Exp(r);
  SO3AxisAngled so3Aa = SO3AxisAngled::Exp(r);
  SO3Quaterniond so3Quat = SO3Quaterniond::Exp(r);

  std::cout << so3Quat.toRotationMatrix() << std::endl;
  std::cout << so3Mat.toRotationMatrix() << std::endl;

  Eigen::Matrix3d eigMat = SO3d::Exp(r).toRotationMatrix();
  Eigen::AngleAxisd eigAa(eigMat);
  Eigen::Quaterniond eigQuat(eigMat);

  SO3d::Random();

  EXPECT_TRUE(so3Mat.isApprox(eigMat));
  EXPECT_TRUE(so3Aa.isApprox(eigAa));
  EXPECT_TRUE(so3Quat.isApprox(eigQuat));

  const auto numTests = 2e+1;
  common::Timer t;

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  std::cout << "[ R = R * R ]" << std::endl;
  std::cout << std::endl;
  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    eigMat = eigMat * eigMat;
  }
  t.stop();
  std::cout << "Eigen::Matrix3d   : " << t.getLastElapsedTime() << " (sec)" << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Mat = so3Mat * so3Mat;
  }
  t.stop();
  std::cout << "SO3Matrixd        : " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Vec = so3Vec * so3Vec;
  }
  t.stop();
  std::cout << "SO3Vectord:         " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    eigAa = eigAa * eigAa;
  }
  t.stop();
  std::cout << "Eigen::AngleAxisd : " << t.getLastElapsedTime() << " (sec)" << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Aa = so3Aa * so3Aa;
  }
  t.stop();
  std::cout << "SO3AxisAngled:      " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    eigQuat = eigQuat * eigQuat;
  }
  t.stop();
  std::cout << "Eigen::Quaterniond: " << t.getLastElapsedTime() << " (sec)" << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Quat = so3Quat * so3Quat;
  }
  t.stop();
  std::cout << "SO3Quaterniond:     " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  std::cout << "[ R *= R ]" << std::endl;
  std::cout << std::endl;
  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    eigMat *= eigMat;
  }
  t.stop();
  std::cout << "Eigen::Matrix3d   : " << t.getLastElapsedTime() << " (sec)" << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Mat *= so3Mat;
  }
  t.stop();
  std::cout << "SO3Matrixd        : " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Vec *= so3Vec;
  }
  t.stop();
  std::cout << "SO3Vectord:         " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  // AngleAxis *= AngleAxis is not supported by Eigen
  //t.start();
  //for (auto i = 0u; i < numTests; ++i)
  //{
  //  eigAa *= eigAa;
  //}
  //t.stop();
  //std::cout << "Eigen::AngleAxisd : " << t.getLastElapsedTime() << " (sec)" << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Aa *= so3Aa;
  }
  t.stop();
  std::cout << "SO3AxisAngled:      " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    eigQuat *= eigQuat;
  }
  t.stop();
  std::cout << "Eigen::Quaterniond: " << t.getLastElapsedTime() << " (sec)" << std::endl;

  t.start();
  for (auto i = 0u; i < numTests; ++i)
  {
    so3Quat *= so3Quat;
  }
  t.stop();
  std::cout << "SO3Quaterniond:     " << t.getLastElapsedTime() << " (sec)" << std::endl;
  std::cout << std::endl;

  std::cout << eigQuat.toRotationMatrix() << std::endl;
  std::cout << eigMat << std::endl;

  std::cout << so3Quat.toRotationMatrix() << std::endl;
  std::cout << so3Mat.toRotationMatrix() << std::endl;
}

//==============================================================================
int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
