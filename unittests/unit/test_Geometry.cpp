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

#include <iostream>

#include <gtest/gtest.h>

#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/TranslationalJoint.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/simulation/World.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

#define LIE_GROUP_OPT_TOL 1e-12

// #define ALL_TESTS

/******************************************************************************/
Eigen::Matrix4d toMatrixForm(const Eigen::Vector6d& v)
{
  Eigen::Matrix4d result = Eigen::Matrix4d::Zero();

  result(0, 1) = -v(2);
  result(1, 0) = v(2);
  result(0, 2) = v(1);
  result(2, 0) = -v(1);
  result(1, 2) = -v(0);
  result(2, 1) = v(0);

  result(0, 3) = v(3);
  result(1, 3) = v(4);
  result(2, 3) = v(5);

  return result;
}

/******************************************************************************/
Eigen::Vector6d fromMatrixForm(const Eigen::Matrix4d& m)
{
  Eigen::Vector6d ret;
  ret << m(2, 1), m(0, 2), m(1, 0), m(0, 3), m(1, 3), m(2, 3);
  return ret;
}

/******************************************************************************/
void testEulerAngles(const Eigen::Vector3d& angle)
{
  Eigen::Matrix3d mat1;
  Eigen::Matrix3d mat2;

  // XYX
  mat1 = math::eulerXYXToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerXYXToMatrix(matrixToEulerXYX(mat1))));

  // XYZ
  mat1 = math::eulerXYZToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerXYZToMatrix(matrixToEulerXYZ(mat1))));

  // XZX
  mat1 = math::eulerXZXToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerXZXToMatrix(matrixToEulerXZX(mat1))));

  // XZY
  mat1 = math::eulerXZYToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerXZYToMatrix(matrixToEulerXZY(mat1))));

  // YXY
  mat1 = math::eulerYXYToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerYXYToMatrix(matrixToEulerYXY(mat1))));

  // YXZ
  mat1 = math::eulerYXZToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerYXZToMatrix(matrixToEulerYXZ(mat1))));

  // YZX
  mat1 = math::eulerYZXToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerYZXToMatrix(matrixToEulerYZX(mat1))));

  // YZY
  mat1 = math::eulerYZYToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerYZYToMatrix(matrixToEulerYZY(mat1))));

  // ZXY
  mat1 = math::eulerZXYToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerZXYToMatrix(matrixToEulerZXY(mat1))));

  // ZYX
  mat1 = math::eulerZYXToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerZYXToMatrix(matrixToEulerZYX(mat1))));

  // ZXZ
  mat1 = math::eulerZXZToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitX())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerZXZToMatrix(matrixToEulerZXZ(mat1))));

  // ZYZ
  mat1 = math::eulerZYZToMatrix(angle);
  mat2 = Eigen::AngleAxisd(angle(0), Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(angle(1), Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(angle(2), Eigen::Vector3d::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerZYZToMatrix(matrixToEulerZYZ(mat1))));
}

/******************************************************************************/
#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_GRADIENT_90)
{
  Eigen::Vector6d screwX = Eigen::Vector6d::Zero();
  screwX(0) = 1.0;
  Eigen::Vector3d point = Eigen::Vector3d::UnitY();
  // if we rotate point by screwX, it should move in the positive Z direction
  double theta = 90 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3d rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3d expectedPoint = Eigen::Vector3d::UnitZ();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  double EPS = 1e-7;
  Eigen::Vector3d perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3d expectedGradient = -Eigen::Vector3d::UnitY();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3d analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_NEGATIVE_THETA)
{
  Eigen::Vector6d screwX = Eigen::Vector6d::Zero();
  screwX(0) = 1.0;
  Eigen::Vector3d point = Eigen::Vector3d::UnitY();
  // if we rotate point by screwX, it should move in the negative Z direction
  double theta = -90 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3d rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3d expectedPoint = -Eigen::Vector3d::UnitZ();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  double EPS = 1e-7;
  Eigen::Vector3d perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3d expectedGradient = Eigen::Vector3d::UnitY();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3d analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_GRADIENT_30)
{
  Eigen::Vector6d screwX = Eigen::Vector6d::Zero();
  screwX(0) = 1.0;
  Eigen::Vector3d point = Eigen::Vector3d::UnitY();
  // if we rotate point by screwX, it should move in the positive Z direction
  double theta = 30 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3d rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0, cos(theta), sin(theta));
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  double EPS = 1e-7;
  Eigen::Vector3d perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3d expectedGradient
      = Eigen::Vector3d(0, -sin(theta), cos(theta));
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3d analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_3D)
{
  Eigen::Vector6d screw = Eigen::Vector6d::Zero();
  screw(0) = 1.0;
  screw(1) = 1.0;
  screw(2) = 1.0;
  screw.normalize();
  Eigen::Vector3d point = Eigen::Vector3d::UnitX();
  // if we rotate point by screw, it should move to the unit Z direction
  double theta = 120 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3d rotatedPoint = math::expMap(screw * theta) * point;
  Eigen::Vector3d expectedPoint = Eigen::Vector3d::UnitY();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  double EPS = 1e-7;
  Eigen::Vector3d perturbedPoint = math::expMap(screw * (theta + EPS)) * point;
  Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3d expectedGradient = Eigen::Vector3d(-1, 0, 1) * sqrt(3) / 3;
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3d analyticalGradient
      = math::gradientWrtTheta(screw, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_TRANSLATION_ONLY_GRADIENT)
{
  Eigen::Vector6d screwX = Eigen::Vector6d::Zero();
  screwX(3) = 1.0;
  Eigen::Vector3d point = Eigen::Vector3d::UnitX();
  double theta = 1.0;
  // This should be at (2, 0, 0) = 2 * UnitX()
  Eigen::Vector3d rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3d expectedPoint = 2 * Eigen::Vector3d::UnitX();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  double EPS = 1e-7;
  Eigen::Vector3d perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3d expectedGradient = Eigen::Vector3d::UnitX();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3d analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, COMPLEX_SCREW_MIXED_ROTATION_AND_TRANSLATION)
{
  Eigen::Vector6d screwX = Eigen::Vector6d::Zero();
  screwX(0) = 1.0;
  screwX(5) = -1.0;
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  // if we rotate point by screwX, it should move in the positive Z direction,
  // and up
  double theta = 90 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3d rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3d expectedPoint = Eigen::Vector3d(0, 1, -1);
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  double EPS = 1e-7;
  Eigen::Vector3d perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3d expectedGradient = Eigen::Vector3d::UnitY();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3d analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, RANDOM_SCREWS)
{
  // Make the experiments repeatable
  srand(42);

  double EPS = 1e-9;

  for (int i = 0; i < 300; i++)
  {
    Eigen::Vector6d screw = Eigen::Vector6d::Random(6);
    // First hundred: rotation only
    if (i < 100)
    {
      screw.head<3>().normalize();
      screw.tail<3>().setZero();
    }
    // Second hundred: translation only
    else if (i < 200)
    {
      screw.head<3>().setZero();
    }
    // Third hundred: mixed
    else
    {
      screw.head<3>().normalize();
    }

    // random theta between [-1, 1]
    double theta = ((double)rand() / RAND_MAX) * 2 - 1;

    Eigen::Vector3d point = Eigen::Vector3d::Random(3);

    Eigen::Vector3d rotatedPoint = math::expMap(screw * theta) * point;
    Eigen::Vector3d perturbedPoint
        = math::expMap(screw * (theta + EPS)) * point;
    Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
    Eigen::Vector3d analyticalGradient
        = math::gradientWrtTheta(screw, point, theta);
    EXPECT_TRUE(equals(analyticalGradient, bruteForceGradient, 1e-6));
    if (!equals(analyticalGradient, bruteForceGradient, 1e-6))
    {
      std::cout << "Lie group derivatives failed!" << std::endl;
      std::cout << "Random example: " << i << "/300" << std::endl;
      std::cout << "Screw:" << std::endl << screw << std::endl;
      std::cout << "Point:" << std::endl << point << std::endl;
      std::cout << "Theta:" << std::endl << theta << std::endl;
      std::cout << "Finite diff:" << std::endl
                << bruteForceGradient << std::endl;
      std::cout << "Analytical:" << std::endl
                << analyticalGradient << std::endl;
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, RANDOM_ROTATIONS)
{
  // Make the experiments repeatable
  srand(42);

  double EPS = 1e-9;

  for (int i = 0; i < 300; i++)
  {
    Eigen::Vector3d axis = Eigen::Vector3d::Random(3);
    axis.normalize();

    // random theta between [-1, 1]
    double theta = ((double)rand() / RAND_MAX) * 2 - 1;

    Eigen::Vector3d point = Eigen::Vector3d::Random(3);

    Eigen::Vector3d rotatedPoint = math::expMapRot(axis * theta) * point;
    Eigen::Vector3d perturbedPoint
        = math::expMapRot(axis * (theta + EPS)) * point;
    Eigen::Vector3d bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
    Eigen::Vector3d analyticalGradient
        = math::gradientWrtThetaPureRotation(axis, point, theta);
    EXPECT_TRUE(equals(analyticalGradient, bruteForceGradient, 1e-6));
    if (!equals(analyticalGradient, bruteForceGradient, 1e-6))
    {
      std::cout << "Lie group derivatives failed!" << std::endl;
      std::cout << "Random example: " << i << "/300" << std::endl;
      std::cout << "Axis:" << std::endl << axis << std::endl;
      std::cout << "Point:" << std::endl << point << std::endl;
      std::cout << "Theta:" << std::endl << theta << std::endl;
      std::cout << "Finite diff:" << std::endl
                << bruteForceGradient << std::endl;
      std::cout << "Analytical:" << std::endl
                << analyticalGradient << std::endl;
      return;
    }
  }
}
#endif

/******************************************************************************/
bool verifyContactPoint(
    const Eigen::Vector3d& edgeAPoint,
    const Eigen::Vector3d& edgeAPointGradient,
    const Eigen::Vector3d& edgeADir,
    const Eigen::Vector3d& edgeADirGradient,
    const Eigen::Vector3d& edgeBPoint,
    const Eigen::Vector3d& edgeBPointGradient,
    const Eigen::Vector3d& edgeBDir,
    const Eigen::Vector3d& edgeBDirGradient)
{
  const double EPS = 1e-6;
  Eigen::Vector3d original
      = getContactPoint(edgeAPoint, edgeADir, edgeBPoint, edgeBDir);
  Eigen::Vector3d perturbedPos = getContactPoint(
      edgeAPoint + edgeAPointGradient * EPS,
      edgeADir + edgeADirGradient * EPS,
      edgeBPoint + edgeBPointGradient * EPS,
      edgeBDir + edgeBDirGradient * EPS);
  Eigen::Vector3d perturbedNeg = getContactPoint(
      edgeAPoint - edgeAPointGradient * EPS,
      edgeADir - edgeADirGradient * EPS,
      edgeBPoint - edgeBPointGradient * EPS,
      edgeBDir - edgeBDirGradient * EPS);
  Eigen::Vector3d finiteDiff = (perturbedPos - perturbedNeg) / (2 * EPS);
  Eigen::Vector3d analytical = getContactPointGradient(
      edgeAPoint,
      edgeAPointGradient,
      edgeADir,
      edgeADirGradient,
      edgeBPoint,
      edgeBPointGradient,
      edgeBDir,
      edgeBDirGradient);
  if (!equals(analytical, finiteDiff, 1e-8))
  {
    std::cout << "Edge-edge contact point derivatives failed!" << std::endl;
    std::cout << "Analytical Gradient:" << std::endl << analytical << std::endl;
    std::cout << "Finite Difference Gradient:" << std::endl
              << finiteDiff << std::endl;
    std::cout << "Diff:" << std::endl << (analytical - finiteDiff) << std::endl;
    std::cout << "Original Point:" << std::endl << original << std::endl;
    std::cout << "Edge A Point:" << std::endl << edgeAPoint << std::endl;
    std::cout << "Edge A Point Gradient:" << std::endl
              << edgeAPointGradient << std::endl;
    std::cout << "Edge A Dir:" << std::endl << edgeADir << std::endl;
    std::cout << "Edge A Dir Gradient:" << std::endl
              << edgeADirGradient << std::endl;
    std::cout << "Edge B Point:" << std::endl << edgeBPoint << std::endl;
    std::cout << "Edge B Point Gradient:" << std::endl
              << edgeBPointGradient << std::endl;
    std::cout << "Edge B Dir:" << std::endl << edgeBDir << std::endl;
    std::cout << "Edge B Dir Gradient:" << std::endl
              << edgeBDirGradient << std::endl;
    return false;
  }
  return true;
}

// #ifdef ALL_TESTS
TEST(COLLISION_GEOM, CONTACT_POINT_GRADIENT)
{
  Eigen::Vector3d edgeAPoint = Eigen::Vector3d(1, 0, 0);
  Eigen::Vector3d edgeAPointGradient = Eigen::Vector3d::Zero();
  Eigen::Vector3d edgeADir = Eigen::Vector3d(-1, 0, 0);
  Eigen::Vector3d edgeADirGradient = Eigen::Vector3d::Zero();
  Eigen::Vector3d edgeBPoint = Eigen::Vector3d(0, 1, 0);
  Eigen::Vector3d edgeBPointGradient = Eigen::Vector3d::Zero();
  Eigen::Vector3d edgeBDir = Eigen::Vector3d(0, -1, 0);
  Eigen::Vector3d edgeBDirGradient = Eigen::Vector3d::Zero();

  Eigen::Vector3d expectedContactPoint = Eigen::Vector3d::Zero();

  EXPECT_TRUE(equals(
      getContactPoint(edgeAPoint, edgeADir, edgeBPoint, edgeBDir),
      expectedContactPoint,
      1e-8));

  // Start with the trivial case, 0 -> 0

  Eigen::Vector3d expectedGradient = Eigen::Vector3d::Zero();
  EXPECT_TRUE(equals(
      getContactPointGradient(
          edgeAPoint,
          edgeAPointGradient,
          edgeADir,
          edgeADirGradient,
          edgeBPoint,
          edgeBPointGradient,
          edgeBDir,
          edgeBDirGradient),
      expectedGradient,
      1e-8));

  // If the edge A point is moving parallel to the A line, we should still see 0
  // gradient

  edgeAPointGradient = edgeADir;

  EXPECT_TRUE(equals(
      getContactPointGradient(
          edgeAPoint,
          edgeAPointGradient,
          edgeADir,
          edgeADirGradient,
          edgeBPoint,
          edgeBPointGradient,
          edgeBDir,
          edgeBDirGradient),
      expectedGradient,
      1e-8));

  // If edge A point is moving parallel to the B line, we should see the
  // gradient is the same as the edgeAPointGradient

  edgeAPointGradient = edgeBDir;
  expectedGradient = edgeAPointGradient;

  EXPECT_TRUE(equals(
      getContactPointGradient(
          edgeAPoint,
          edgeAPointGradient,
          edgeADir,
          edgeADirGradient,
          edgeBPoint,
          edgeBPointGradient,
          edgeBDir,
          edgeBDirGradient),
      expectedGradient,
      1e-8));

  // If edge A point is moving parallel to the B line, and edge B point is
  // moving parallel to the A line, gradient is the sum of both

  edgeAPointGradient = edgeBDir;
  edgeBPointGradient = edgeADir;
  expectedGradient = edgeAPointGradient + edgeBPointGradient;

  EXPECT_TRUE(equals(
      getContactPointGradient(
          edgeAPoint,
          edgeAPointGradient,
          edgeADir,
          edgeADirGradient,
          edgeBPoint,
          edgeBPointGradient,
          edgeBDir,
          edgeBDirGradient),
      expectedGradient,
      1e-8));

  // If the direction of A is moving, then the overall gradient should move in
  // the same direction

  edgeAPointGradient = Eigen::Vector3d::Zero();
  edgeBPointGradient = Eigen::Vector3d::Zero();
  edgeADirGradient = Eigen::Vector3d(0, 0, 1);
  // It's divided by 2 because we're only moving half the average, the B point
  // doesn't move
  expectedGradient = edgeADirGradient / 2;

  Eigen::Vector3d result = getContactPointGradient(
      edgeAPoint,
      edgeAPointGradient,
      edgeADir,
      edgeADirGradient,
      edgeBPoint,
      edgeBPointGradient,
      edgeBDir,
      edgeBDirGradient);

  EXPECT_TRUE(equals(result, expectedGradient, 1e-8));
}
// #endif

// #ifdef ALL_TESTS
TEST(COLLISION_GEOM, RANDOM_CONTACT_POINTS)
{
  for (int i = 0; i < 700; i++)
  {
    Eigen::Vector3d edgeAPoint = Eigen::Vector3d::Random();
    Eigen::Vector3d edgeAPointGradient = Eigen::Vector3d::Zero();
    Eigen::Vector3d edgeADir = Eigen::Vector3d::Random();
    Eigen::Vector3d edgeADirGradient = Eigen::Vector3d::Zero();
    Eigen::Vector3d edgeBPoint = Eigen::Vector3d::Random();
    Eigen::Vector3d edgeBPointGradient = Eigen::Vector3d::Zero();
    Eigen::Vector3d edgeBDir = Eigen::Vector3d::Random();
    Eigen::Vector3d edgeBDirGradient = Eigen::Vector3d::Zero();

    if (i < 100)
    {
      edgeAPointGradient = Eigen::Vector3d::Random();
    }
    else if (i < 200)
    {
      edgeBPointGradient = Eigen::Vector3d::Random();
    }
    else if (i < 300)
    {
      edgeADirGradient = Eigen::Vector3d::Random();
    }
    else if (i < 400)
    {
      edgeBDirGradient = Eigen::Vector3d::Random();
    }
    else if (i < 500)
    {
      edgeAPointGradient = Eigen::Vector3d::Random();
      edgeADirGradient = Eigen::Vector3d::Random();
    }
    else if (i < 600)
    {
      edgeBPointGradient = Eigen::Vector3d::Random();
      edgeBDirGradient = Eigen::Vector3d::Random();
    }
    else if (i < 700)
    {
      edgeAPointGradient = Eigen::Vector3d::Random();
      edgeADirGradient = Eigen::Vector3d::Random();
      edgeBPointGradient = Eigen::Vector3d::Random();
      edgeBDirGradient = Eigen::Vector3d::Random();
    }

    bool result = verifyContactPoint(
        edgeAPoint,
        edgeAPointGradient,
        edgeADir,
        edgeADirGradient,
        edgeBPoint,
        edgeBPointGradient,
        edgeBDir,
        edgeBDirGradient);
    EXPECT_TRUE(result);
    if (!result)
      return;
  }
}
// #endif

/******************************************************************************/
#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, EULER_ANGLES)
{
  // TODO: Special angles such as (PI, 0, 0)

  //
  int numTest = 1;
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector3d angle = Eigen::Vector3d::Random();
    testEulerAngles(angle);
  }
}
#endif

/******************************************************************************/
#define EPSILON_EXPMAP_THETA 1.0e-3
#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, EXPONENTIAL_MAPPINGS)
{
  int numTest = 100;

  // Exp
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d s = Eigen::Vector6d::Random();
    Eigen::Isometry3d Exp_s = math::expMap(s);
    Eigen::Matrix4d Exp_s_2 = Eigen::Matrix4d::Identity();

    double theta = s.head<3>().norm();
    Eigen::Matrix3d R = Matrix3d::Zero();
    Eigen::Matrix3d qss = math::makeSkewSymmetric(s.head<3>());
    Eigen::Matrix3d qss2 = qss * qss;
    Eigen::Matrix3d P = Eigen::Matrix3d::Zero();

    if (theta < EPSILON_EXPMAP_THETA)
    {
      R = Matrix3d::Identity() + qss + 0.5 * qss2;
      P = Matrix3d::Identity() + 0.5 * qss + (1 / 6) * qss2;
    }
    else
    {
      R = Matrix3d::Identity() + (sin(theta) / theta) * qss
          + ((1 - cos(theta)) / (theta * theta)) * qss2;
      P = Matrix3d::Identity() + ((1 - cos(theta)) / (theta * theta)) * qss
          + ((theta - sin(theta)) / (theta * theta * theta)) * qss2;
    }

    Exp_s_2.topLeftCorner<3, 3>() = R;
    Exp_s_2.topRightCorner<3, 1>() = P * s.tail<3>();

    //
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        EXPECT_NEAR(Exp_s(i, j), Exp_s_2(i, j), LIE_GROUP_OPT_TOL);
  }

  // ExpAngular
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d s = Eigen::Vector6d::Random();
    s.tail<3>() = Eigen::Vector3d::Zero();
    Eigen::Isometry3d Exp_s = math::expAngular(s.head<3>());
    Eigen::Matrix4d Exp_s_2 = Eigen::Matrix4d::Identity();

    double theta = s.head<3>().norm();
    Eigen::Matrix3d R = Matrix3d::Zero();
    Eigen::Matrix3d qss = math::makeSkewSymmetric(s.head<3>());
    Eigen::Matrix3d qss2 = qss * qss;
    Eigen::Matrix3d P = Eigen::Matrix3d::Zero();

    if (theta < EPSILON_EXPMAP_THETA)
    {
      R = Matrix3d::Identity() + qss + 0.5 * qss2;
      P = Matrix3d::Identity() + 0.5 * qss + (1 / 6) * qss2;
    }
    else
    {
      R = Matrix3d::Identity() + (sin(theta) / theta) * qss
          + ((1 - cos(theta)) / (theta * theta)) * qss2;
      P = Matrix3d::Identity() + ((1 - cos(theta)) / (theta * theta)) * qss
          + ((theta - sin(theta)) / (theta * theta * theta)) * qss2;
    }

    Exp_s_2.topLeftCorner<3, 3>() = R;
    Exp_s_2.topRightCorner<3, 1>() = P * s.tail<3>();

    //
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        EXPECT_NEAR(Exp_s(i, j), Exp_s_2(i, j), LIE_GROUP_OPT_TOL);
  }

  // ExpLinear
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d s = Eigen::Vector6d::Random();
    s.head<3>() = Eigen::Vector3d::Zero();
    Eigen::Isometry3d Exp_s(Eigen::Translation3d(s.tail<3>()));
    Eigen::Matrix4d Exp_s_2 = Eigen::Matrix4d::Identity();

    double theta = s.head<3>().norm();
    Eigen::Matrix3d R = Matrix3d::Zero();
    Eigen::Matrix3d qss = math::makeSkewSymmetric(s.head<3>());
    Eigen::Matrix3d qss2 = qss * qss;
    Eigen::Matrix3d P = Eigen::Matrix3d::Zero();

    if (theta < EPSILON_EXPMAP_THETA)
    {
      R = Matrix3d::Identity() + qss + 0.5 * qss2;
      P = Matrix3d::Identity() + 0.5 * qss + (1 / 6) * qss2;
    }
    else
    {
      R = Matrix3d::Identity() + (sin(theta) / theta) * qss
          + ((1 - cos(theta)) / (theta * theta)) * qss2;
      P = Matrix3d::Identity() + ((1 - cos(theta)) / (theta * theta)) * qss
          + ((theta - sin(theta)) / (theta * theta * theta)) * qss2;
    }

    Exp_s_2.topLeftCorner<3, 3>() = R;
    Exp_s_2.topRightCorner<3, 1>() = P * s.tail<3>();

    //
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        EXPECT_NEAR(Exp_s(i, j), Exp_s_2(i, j), LIE_GROUP_OPT_TOL);
  }
  // Exponential mapping test with high values
  int numExpTests = 100;
  double min = -1e+128;
  double max = +1e+128;

  for (int idxTest = 0; idxTest < numExpTests; ++idxTest)
  {
    Eigen::Vector3d randomS = Eigen::Vector3d::Zero();

    for (int i = 0; i < 3; ++i)
      randomS[i] = Random::uniform(min, max);

    Eigen::Isometry3d T = math::expAngular(randomS);
    EXPECT_TRUE(math::verifyTransform(T));
  }

  for (int idxTest = 0; idxTest < numExpTests; ++idxTest)
  {
    Eigen::Vector6d randomS = Eigen::Vector6d::Zero();

    for (int i = 0; i < 6; ++i)
      randomS[i] = Random::uniform(min, max);

    Eigen::Isometry3d T = math::expMap(randomS);
    EXPECT_TRUE(math::verifyTransform(T));
  }
}
#endif

/******************************************************************************/
#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, ADJOINT_MAPPINGS)
{
  int numTest = 100;

  // AdT(V) == T * V * InvT
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Vector6d V = Eigen::Vector6d::Random();

    Eigen::Vector6d AdTV = AdT(T, V);

    // Ad(T, V) = T * [V] * InvT
    Eigen::Matrix4d T_V_InvT
        = T.matrix() * toMatrixForm(V) * T.inverse().matrix();
    Eigen::Vector6d T_V_InvT_se3 = fromMatrixForm(T_V_InvT);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), T_V_InvT_se3(j), LIE_GROUP_OPT_TOL);

    // Ad(T, V) = [R 0; [p]R R] * V
    Eigen::Matrix6d AdTMatrix = Eigen::Matrix6d::Zero();
    AdTMatrix.topLeftCorner<3, 3>() = T.linear();
    AdTMatrix.bottomRightCorner<3, 3>() = T.linear();
    AdTMatrix.bottomLeftCorner<3, 3>()
        = math::makeSkewSymmetric(T.translation()) * T.linear();
    Eigen::Vector6d AdTMatrix_V = AdTMatrix * V;
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTMatrix_V(j), LIE_GROUP_OPT_TOL);
  }

  // AdR == AdT([R 0; 0 1], V)
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Isometry3d R = Eigen::Isometry3d::Identity();
    R.linear() = T.linear();
    Eigen::Vector6d V = Eigen::Vector6d::Random();

    Eigen::Vector6d AdTV = AdT(R, V);
    Eigen::Vector6d AdRV = AdR(T, V);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdRV(j), LIE_GROUP_OPT_TOL);
  }

  // AdTAngular == AdT(T, se3(w, 0))
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Vector3d w = Eigen::Vector3d::Random();
    Eigen::Vector6d V = Eigen::Vector6d::Zero();
    V.head<3>() = w;

    Eigen::Vector6d AdTV = AdT(T, V);
    Eigen::Vector6d AdTAng = AdTAngular(T, w);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTAng(j), LIE_GROUP_OPT_TOL);
  }

  // AdTLinear == AdT(T, se3(w, 0))
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Vector3d v = Eigen::Vector3d::Random();
    Eigen::Vector6d V = Eigen::Vector6d::Zero();
    V.tail<3>() = v;

    Eigen::Vector6d AdTV = AdT(T, V);
    Eigen::Vector6d AdTLin = AdTLinear(T, v);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTLin(j), LIE_GROUP_OPT_TOL);
  }

  // AdTJac
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Vector3d v = Eigen::Vector3d::Random();
    Eigen::Vector6d V = Eigen::Vector6d::Zero();
    V.tail<3>() = v;

    Eigen::Vector6d AdTV = AdT(T, V);
    Eigen::Vector6d AdTLin = AdTLinear(T, v);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTLin(j), LIE_GROUP_OPT_TOL);
  }

  // AdInvT
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Isometry3d InvT = T.inverse();
    Eigen::Vector6d V = Eigen::Vector6d::Random();

    Eigen::Vector6d Ad_InvT = AdT(InvT, V);
    Eigen::Vector6d AdInv_T = AdInvT(T, V);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(Ad_InvT(j), AdInv_T(j), LIE_GROUP_OPT_TOL);
  }

  // AdInvRLinear
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Vector3d v = Eigen::Vector3d::Random();
    Eigen::Vector6d V = Eigen::Vector6d::Zero();
    V.tail<3>() = v;
    Eigen::Isometry3d R = Eigen::Isometry3d::Identity();
    R.linear() = T.linear();

    Eigen::Vector6d AdT_ = AdT(R.inverse(), V);
    Eigen::Vector6d AdInvRLinear_ = AdInvRLinear(T, v);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdT_(j), AdInvRLinear_(j), LIE_GROUP_OPT_TOL);
  }

  // dAdT
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Vector6d F = Eigen::Vector6d::Random();

    Eigen::Vector6d dAdTF = dAdT(T, F);

    // dAd(T, F) = [R 0; [p]R R]^T * F
    Eigen::Matrix6d AdTMatrix = Eigen::Matrix6d::Zero();
    AdTMatrix.topLeftCorner<3, 3>() = T.linear();
    AdTMatrix.bottomRightCorner<3, 3>() = T.linear();
    AdTMatrix.bottomLeftCorner<3, 3>()
        = math::makeSkewSymmetric(T.translation()) * T.linear();
    Eigen::Vector6d AdTTransMatrix_V = AdTMatrix.transpose() * F;
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdTF(j), AdTTransMatrix_V(j), LIE_GROUP_OPT_TOL);
  }

  // dAdInvT
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Isometry3d InvT = T.inverse();
    Eigen::Vector6d F = Eigen::Vector6d::Random();

    Eigen::Vector6d dAdInvT_F = dAdInvT(T, F);

    //
    Eigen::Vector6d dAd_InvTF = dAdT(InvT, F);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdInvT_F(j), dAd_InvTF(j), LIE_GROUP_OPT_TOL);

    // dAd(T, F) = [R 0; [p]R R]^T * F
    Eigen::Matrix6d AdInvTMatrix = Eigen::Matrix6d::Zero();
    AdInvTMatrix.topLeftCorner<3, 3>() = InvT.linear();
    AdInvTMatrix.bottomRightCorner<3, 3>() = InvT.linear();
    AdInvTMatrix.bottomLeftCorner<3, 3>()
        = math::makeSkewSymmetric(InvT.translation()) * InvT.linear();
    Eigen::Vector6d AdInvTTransMatrix_V = AdInvTMatrix.transpose() * F;
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdInvT_F(j), AdInvTTransMatrix_V(j), LIE_GROUP_OPT_TOL);
  }

  // dAdInvR
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d t = Eigen::Vector6d::Random();
    Eigen::Isometry3d T = math::expMap(t);
    Eigen::Isometry3d InvT = T.inverse();
    Eigen::Isometry3d InvR = Eigen::Isometry3d::Identity();
    InvR.linear() = InvT.linear();
    Eigen::Vector6d F = Eigen::Vector6d::Random();

    Eigen::Vector6d dAdInvR_F = dAdInvR(T, F);

    //
    Eigen::Vector6d dAd_InvTF = dAdT(InvR, F);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdInvR_F(j), dAd_InvTF(j), LIE_GROUP_OPT_TOL);
  }

  // ad
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d V = Eigen::Vector6d::Random();
    Eigen::Vector6d W = Eigen::Vector6d::Random();

    Eigen::Vector6d ad_V_W = ad(V, W);

    //
    Eigen::Matrix6d adV_Matrix = Eigen::Matrix6d::Zero();
    adV_Matrix.topLeftCorner<3, 3>() = math::makeSkewSymmetric(V.head<3>());
    adV_Matrix.bottomRightCorner<3, 3>() = math::makeSkewSymmetric(V.head<3>());
    adV_Matrix.bottomLeftCorner<3, 3>() = math::makeSkewSymmetric(V.tail<3>());
    Eigen::Vector6d adV_Matrix_W = adV_Matrix * W;

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(ad_V_W(j), adV_Matrix_W(j), LIE_GROUP_OPT_TOL);
  }

  // dad
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6d V = Eigen::Vector6d::Random();
    Eigen::Vector6d F = Eigen::Vector6d::Random();

    Eigen::Vector6d dad_V_F = dad(V, F);

    //
    Eigen::Matrix6d dadV_Matrix = Eigen::Matrix6d::Zero();
    dadV_Matrix.topLeftCorner<3, 3>() = math::makeSkewSymmetric(V.head<3>());
    dadV_Matrix.bottomRightCorner<3, 3>()
        = math::makeSkewSymmetric(V.head<3>());
    dadV_Matrix.bottomLeftCorner<3, 3>() = math::makeSkewSymmetric(V.tail<3>());
    Eigen::Vector6d dadV_Matrix_F = dadV_Matrix.transpose() * F;

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dad_V_F(j), dadV_Matrix_F(j), LIE_GROUP_OPT_TOL);
  }
}
#endif
