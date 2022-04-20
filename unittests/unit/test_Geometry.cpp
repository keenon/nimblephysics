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

#define ALL_TESTS

/******************************************************************************/
Eigen::Matrix4s toMatrixForm(const Eigen::Vector6s& v)
{
  Eigen::Matrix4s result = Eigen::Matrix4s::Zero();

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
Eigen::Vector6s fromMatrixForm(const Eigen::Matrix4s& m)
{
  Eigen::Vector6s ret;
  ret << m(2, 1), m(0, 2), m(1, 0), m(0, 3), m(1, 3), m(2, 3);
  return ret;
}

/******************************************************************************/
void testEulerAngles(const Eigen::Vector3s& angle)
{
  Eigen::Matrix3s mat1;
  Eigen::Matrix3s mat2;

  // XYX
  mat1 = math::eulerXYXToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerXYXToMatrix(matrixToEulerXYX(mat1))));

  // XYZ
  mat1 = math::eulerXYZToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerXYZToMatrix(matrixToEulerXYZ(mat1))));

  // XZX
  mat1 = math::eulerXZXToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerXZXToMatrix(matrixToEulerXZX(mat1))));

  // XZY
  mat1 = math::eulerXZYToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerXZYToMatrix(matrixToEulerXZY(mat1))));

  // YXY
  mat1 = math::eulerYXYToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerYXYToMatrix(matrixToEulerYXY(mat1))));

  // YXZ
  mat1 = math::eulerYXZToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerYXZToMatrix(matrixToEulerYXZ(mat1))));

  // YZX
  mat1 = math::eulerYZXToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerYZXToMatrix(matrixToEulerYZX(mat1))));

  // YZY
  mat1 = math::eulerYZYToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerYZYToMatrix(matrixToEulerYZY(mat1))));

  // ZXY
  mat1 = math::eulerZXYToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitY());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerZXYToMatrix(matrixToEulerZXY(mat1))));

  // ZYX
  mat1 = math::eulerZYXToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitX());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  EXPECT_TRUE(equals(mat1, eulerZYXToMatrix(matrixToEulerZYX(mat1))));

  // ZXZ
  mat1 = math::eulerZXZToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitX())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerZXZToMatrix(matrixToEulerZXZ(mat1))));

  // ZYZ
  mat1 = math::eulerZYZToMatrix(angle);
  mat2 = Eigen::AngleAxis_s(angle(0), Eigen::Vector3s::UnitZ())
         * Eigen::AngleAxis_s(angle(1), Eigen::Vector3s::UnitY())
         * Eigen::AngleAxis_s(angle(2), Eigen::Vector3s::UnitZ());

  EXPECT_TRUE(math::verifyRotation(mat1));
  EXPECT_TRUE(math::verifyRotation(mat2));
  EXPECT_TRUE(equals(mat1, mat2));
  // EXPECT_TRUE(equals(mat1, eulerZYZToMatrix(matrixToEulerZYZ(mat1))));
}

/******************************************************************************/
#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_GRADIENT_90)
{
  Eigen::Vector6s screwX = Eigen::Vector6s::Zero();
  screwX(0) = 1.0;
  Eigen::Vector3s point = Eigen::Vector3s::UnitY();
  // if we rotate point by screwX, it should move in the positive Z direction
  s_t theta = 90 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3s rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3s expectedPoint = Eigen::Vector3s::UnitZ();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3s expectedGradient = -Eigen::Vector3s::UnitY();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3s analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_NEGATIVE_THETA)
{
  Eigen::Vector6s screwX = Eigen::Vector6s::Zero();
  screwX(0) = 1.0;
  Eigen::Vector3s point = Eigen::Vector3s::UnitY();
  // if we rotate point by screwX, it should move in the negative Z direction
  s_t theta = -90 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3s rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3s expectedPoint = -Eigen::Vector3s::UnitZ();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3s expectedGradient = Eigen::Vector3s::UnitY();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3s analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_GRADIENT_30)
{
  Eigen::Vector6s screwX = Eigen::Vector6s::Zero();
  screwX(0) = 1.0;
  Eigen::Vector3s point = Eigen::Vector3s::UnitY();
  // if we rotate point by screwX, it should move in the positive Z direction
  s_t theta = 30 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3s rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3s expectedPoint = Eigen::Vector3s(0, cos(theta), sin(theta));
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3s expectedGradient
      = Eigen::Vector3s(0, -sin(theta), cos(theta));
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3s analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_ROTATION_ONLY_3D)
{
  Eigen::Vector6s screw = Eigen::Vector6s::Zero();
  screw(0) = 1.0;
  screw(1) = 1.0;
  screw(2) = 1.0;
  screw.normalize();
  Eigen::Vector3s point = Eigen::Vector3s::UnitX();
  // if we rotate point by screw, it should move to the unit Z direction
  s_t theta = 120 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3s rotatedPoint = math::expMap(screw * theta) * point;
  Eigen::Vector3s expectedPoint = Eigen::Vector3s::UnitY();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPoint = math::expMap(screw * (theta + EPS)) * point;
  Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3s expectedGradient = Eigen::Vector3s(-1, 0, 1) * sqrt(3) / 3;
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3s analyticalGradient
      = math::gradientWrtTheta(screw, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, SIMPLE_SCREW_TRANSLATION_ONLY_GRADIENT)
{
  Eigen::Vector6s screwX = Eigen::Vector6s::Zero();
  screwX(3) = 1.0;
  Eigen::Vector3s point = Eigen::Vector3s::UnitX();
  s_t theta = 1.0;
  // This should be at (2, 0, 0) = 2 * UnitX()
  Eigen::Vector3s rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3s expectedPoint = 2 * Eigen::Vector3s::UnitX();
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3s expectedGradient = Eigen::Vector3s::UnitX();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3s analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, COMPLEX_SCREW_MIXED_ROTATION_AND_TRANSLATION)
{
  Eigen::Vector6s screwX = Eigen::Vector6s::Zero();
  screwX(0) = 1.0;
  screwX(5) = -1.0;
  Eigen::Vector3s point = Eigen::Vector3s::Zero();
  // if we rotate point by screwX, it should move in the positive Z direction,
  // and up
  s_t theta = 90 * 3.1415926535 / 180;
  // This should be at (0, 0, 1) = UnitZ()
  Eigen::Vector3s rotatedPoint = math::expMap(screwX * theta) * point;
  Eigen::Vector3s expectedPoint = Eigen::Vector3s(0, 1, -1);
  EXPECT_TRUE(equals(rotatedPoint, expectedPoint, 1e-6));

  s_t EPS = 1e-7;
  Eigen::Vector3s perturbedPoint = math::expMap(screwX * (theta + EPS)) * point;
  Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
  Eigen::Vector3s expectedGradient = Eigen::Vector3s::UnitY();
  EXPECT_TRUE(equals(bruteForceGradient, expectedGradient, 1e-6));

  Eigen::Vector3s analyticalGradient
      = math::gradientWrtTheta(screwX, point, theta);
  EXPECT_TRUE(equals(analyticalGradient, expectedGradient, 1e-6));
}
#endif

#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, RANDOM_SCREWS)
{
  // Make the experiments repeatable
  srand(42);

  s_t EPS = 1e-9;

  for (int i = 0; i < 300; i++)
  {
    Eigen::Vector6s screw = Eigen::Vector6s::Random(6);
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
    s_t theta = ((s_t)rand() / RAND_MAX) * 2 - 1;

    Eigen::Vector3s point = Eigen::Vector3s::Random(3);

    Eigen::Vector3s rotatedPoint = math::expMap(screw * theta) * point;
    Eigen::Vector3s perturbedPoint
        = math::expMap(screw * (theta + EPS)) * point;
    Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
    Eigen::Vector3s analyticalGradient
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

  s_t EPS = 1e-9;

  for (int i = 0; i < 300; i++)
  {
    Eigen::Vector3s axis = Eigen::Vector3s::Random(3);
    axis.normalize();

    // random theta between [-1, 1]
    s_t theta = ((s_t)rand() / RAND_MAX) * 2 - 1;

    Eigen::Vector3s point = Eigen::Vector3s::Random(3);

    Eigen::Vector3s rotatedPoint = math::expMapRot(axis * theta) * point;
    Eigen::Vector3s perturbedPoint
        = math::expMapRot(axis * (theta + EPS)) * point;
    Eigen::Vector3s bruteForceGradient = (perturbedPoint - rotatedPoint) / EPS;
    Eigen::Vector3s analyticalGradient
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
    const Eigen::Vector3s& edgeAPoint,
    const Eigen::Vector3s& edgeAPointGradient,
    const Eigen::Vector3s& edgeADir,
    const Eigen::Vector3s& edgeADirGradient,
    const Eigen::Vector3s& edgeBPoint,
    const Eigen::Vector3s& edgeBPointGradient,
    const Eigen::Vector3s& edgeBDir,
    const Eigen::Vector3s& edgeBDirGradient)
{
  const s_t EPS = 1e-6;
  Eigen::Vector3s original
      = getContactPoint(edgeAPoint, edgeADir, edgeBPoint, edgeBDir);
  Eigen::Vector3s perturbedPos = getContactPoint(
      edgeAPoint + edgeAPointGradient * EPS,
      edgeADir + edgeADirGradient * EPS,
      edgeBPoint + edgeBPointGradient * EPS,
      edgeBDir + edgeBDirGradient * EPS);
  Eigen::Vector3s perturbedNeg = getContactPoint(
      edgeAPoint - edgeAPointGradient * EPS,
      edgeADir - edgeADirGradient * EPS,
      edgeBPoint - edgeBPointGradient * EPS,
      edgeBDir - edgeBDirGradient * EPS);
  Eigen::Vector3s finiteDiff = (perturbedPos - perturbedNeg) / (2 * EPS);
  Eigen::Vector3s analytical = getContactPointGradient(
      edgeAPoint,
      edgeAPointGradient,
      edgeADir,
      edgeADirGradient,
      edgeBPoint,
      edgeBPointGradient,
      edgeBDir,
      edgeBDirGradient);
  if (!equals(analytical, finiteDiff, finiteDiff.norm() * 1e-8))
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

#ifdef ALL_TESTS
TEST(COLLISION_GEOM, EDGE_EDGE_GRADIENT)
{
  Eigen::Vector3s edgeAPoint = Eigen::Vector3s(1, 0, 0);
  Eigen::Vector3s edgeAPointGradient = Eigen::Vector3s::Zero();
  Eigen::Vector3s edgeADir = Eigen::Vector3s(-1, 0, 0);
  Eigen::Vector3s edgeADirGradient = Eigen::Vector3s::Zero();
  Eigen::Vector3s edgeBPoint = Eigen::Vector3s(0, 1, 0);
  Eigen::Vector3s edgeBPointGradient = Eigen::Vector3s::Zero();
  Eigen::Vector3s edgeBDir = Eigen::Vector3s(0, -1, 0);
  Eigen::Vector3s edgeBDirGradient = Eigen::Vector3s::Zero();

  Eigen::Vector3s expectedContactPoint = Eigen::Vector3s::Zero();

  EXPECT_TRUE(equals(
      getContactPoint(edgeAPoint, edgeADir, edgeBPoint, edgeBDir),
      expectedContactPoint,
      1e-8));

  // Start with the trivial case, 0 -> 0

  Eigen::Vector3s expectedGradient = Eigen::Vector3s::Zero();
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

  edgeAPointGradient = Eigen::Vector3s::Zero();
  edgeBPointGradient = Eigen::Vector3s::Zero();
  edgeADirGradient = Eigen::Vector3s(0, 0, 1);
  // It's divided by 2 because we're only moving half the average, the B point
  // doesn't move
  expectedGradient = edgeADirGradient / 2;

  Eigen::Vector3s result = getContactPointGradient(
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
#endif

#ifdef ALL_TESTS
TEST(COLLISION_GEOM, RANDOM_EDGE_EDGE_GRADIENTS)
{
  for (int i = 0; i < 700; i++)
  {
    Eigen::Vector3s edgeAPoint = Eigen::Vector3s::Random();
    Eigen::Vector3s edgeAPointGradient = Eigen::Vector3s::Zero();
    Eigen::Vector3s edgeADir = Eigen::Vector3s::Random();
    Eigen::Vector3s edgeADirGradient = Eigen::Vector3s::Zero();
    Eigen::Vector3s edgeBPoint = Eigen::Vector3s::Random();
    Eigen::Vector3s edgeBPointGradient = Eigen::Vector3s::Zero();
    Eigen::Vector3s edgeBDir = Eigen::Vector3s::Random();
    Eigen::Vector3s edgeBDirGradient = Eigen::Vector3s::Zero();

    if (i < 100)
    {
      edgeAPointGradient = Eigen::Vector3s::Random();
    }
    else if (i < 200)
    {
      edgeBPointGradient = Eigen::Vector3s::Random();
    }
    else if (i < 300)
    {
      edgeADirGradient = Eigen::Vector3s::Random();
    }
    else if (i < 400)
    {
      edgeBDirGradient = Eigen::Vector3s::Random();
    }
    else if (i < 500)
    {
      edgeAPointGradient = Eigen::Vector3s::Random();
      edgeADirGradient = Eigen::Vector3s::Random();
    }
    else if (i < 600)
    {
      edgeBPointGradient = Eigen::Vector3s::Random();
      edgeBDirGradient = Eigen::Vector3s::Random();
    }
    else if (i < 700)
    {
      edgeAPointGradient = Eigen::Vector3s::Random();
      edgeADirGradient = Eigen::Vector3s::Random();
      edgeBPointGradient = Eigen::Vector3s::Random();
      edgeBDirGradient = Eigen::Vector3s::Random();
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
#endif

/******************************************************************************/
bool verifyClosestPoint(
    const Eigen::Vector3s& edgePoint,
    const Eigen::Vector3s& edgePointGradient,
    const Eigen::Vector3s& edgeDir,
    const Eigen::Vector3s& edgeDirGradient,
    const Eigen::Vector3s& goalPoint,
    const Eigen::Vector3s& goalPointGradient)
{
  const s_t EPS = 1e-6;
  Eigen::Vector3s original = closestPointOnLine(edgePoint, edgeDir, goalPoint);
  Eigen::Vector3s perturbedPos = closestPointOnLine(
      edgePoint + edgePointGradient * EPS,
      edgeDir + edgeDirGradient * EPS,
      goalPoint + goalPointGradient * EPS);
  Eigen::Vector3s perturbedNeg = closestPointOnLine(
      edgePoint - edgePointGradient * EPS,
      edgeDir - edgeDirGradient * EPS,
      goalPoint - goalPointGradient * EPS);
  Eigen::Vector3s finiteDiff = (perturbedPos - perturbedNeg) / (2 * EPS);
  Eigen::Vector3s analytical = closestPointOnLineGradient(
      edgePoint,
      edgePointGradient,
      edgeDir,
      edgeDirGradient,
      goalPoint,
      goalPointGradient);
  if (!equals(analytical, finiteDiff, 1e-8))
  {
    std::cout << "Edge-edge contact point derivatives failed!" << std::endl;
    std::cout << "Analytical Gradient:" << std::endl << analytical << std::endl;
    std::cout << "Finite Difference Gradient:" << std::endl
              << finiteDiff << std::endl;
    std::cout << "Diff:" << std::endl << (analytical - finiteDiff) << std::endl;
    std::cout << "Original Point:" << std::endl << original << std::endl;
    std::cout << "Edge Point:" << std::endl << edgePoint << std::endl;
    std::cout << "Edge Point Gradient:" << std::endl
              << edgePointGradient << std::endl;
    std::cout << "Edge Dir:" << std::endl << edgeDir << std::endl;
    std::cout << "Edge Dir Gradient:" << std::endl
              << edgeDirGradient << std::endl;
    std::cout << "Goal Point:" << std::endl << goalPoint << std::endl;
    std::cout << "Goal Point Gradient:" << std::endl
              << goalPointGradient << std::endl;
    return false;
  }
  return true;
}

#ifdef ALL_TESTS
TEST(COLLISION_GEOM, RANDOM_CLOSEST_POINT_GRADIENTS)
{
  for (int i = 0; i < 700; i++)
  {
    Eigen::Vector3s edgePoint = Eigen::Vector3s::Random();
    Eigen::Vector3s edgePointGradient = Eigen::Vector3s::Zero();
    Eigen::Vector3s edgeDir = Eigen::Vector3s::Random();
    Eigen::Vector3s edgeDirGradient = Eigen::Vector3s::Zero();
    Eigen::Vector3s goalPoint = Eigen::Vector3s::Random();
    Eigen::Vector3s goalPointGradient = Eigen::Vector3s::Zero();

    if (i < 100)
    {
      edgePointGradient = Eigen::Vector3s::Random();
    }
    else if (i < 200)
    {
      edgeDirGradient = Eigen::Vector3s::Random();
    }
    else if (i < 300)
    {
      goalPointGradient = Eigen::Vector3s::Random();
    }
    else if (i < 400)
    {
      edgePointGradient = Eigen::Vector3s::Random();
      edgeDirGradient = Eigen::Vector3s::Random();
    }
    else if (i < 600)
    {
      edgePointGradient = Eigen::Vector3s::Random();
      goalPointGradient = Eigen::Vector3s::Random();
    }
    else if (i < 700)
    {
      edgePointGradient = Eigen::Vector3s::Random();
      edgeDirGradient = Eigen::Vector3s::Random();
      goalPointGradient = Eigen::Vector3s::Random();
    }

    bool result = verifyClosestPoint(
        edgePoint,
        edgePointGradient,
        edgeDir,
        edgeDirGradient,
        goalPoint,
        goalPointGradient);
    EXPECT_TRUE(result);
    if (!result)
      return;
  }
}
#endif

#ifdef ALL_TESTS
TEST(DISTANCE, CLOSEST_POINT_ON_SEGMENT_ON_SEG)
{
  Eigen::Vector3s a = Eigen::Vector3s(1, 0, 0);
  Eigen::Vector3s b = Eigen::Vector3s(-1, 0, 0);
  Eigen::Vector3s p = Eigen::Vector3s(0, 3, 0);
  s_t dist = math::distanceToSegment(a, b, p);
  EXPECT_EQ(dist, 3.0);
}
#endif

#ifdef ALL_TESTS
TEST(DISTANCE, CLOSEST_POINT_ON_SEGMENT_A)
{
  Eigen::Vector3s a = Eigen::Vector3s(1, 0, 0);
  Eigen::Vector3s b = Eigen::Vector3s(-1, 0, 0);
  Eigen::Vector3s p = Eigen::Vector3s(3, 0, 0);
  s_t dist = math::distanceToSegment(a, b, p);
  EXPECT_EQ(dist, 2.0);
}
#endif

#ifdef ALL_TESTS
TEST(DISTANCE, CLOSEST_POINT_ON_SEGMENT_B)
{
  Eigen::Vector3s a = Eigen::Vector3s(1, 0, 0);
  Eigen::Vector3s b = Eigen::Vector3s(-1, 0, 0);
  Eigen::Vector3s p = Eigen::Vector3s(-3, 0, 0);
  s_t dist = math::distanceToSegment(a, b, p);
  EXPECT_EQ(dist, 2.0);
}
#endif

/******************************************************************************/
#ifdef ALL_TESTS
TEST(LIE_GROUP_OPERATORS, EULER_ANGLES)
{
  // TODO: Special angles such as (PI, 0, 0)

  //
  int numTest = 1;
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector3s angle = Eigen::Vector3s::Random();
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
    Eigen::Vector6s s = Eigen::Vector6s::Random();
    Eigen::Isometry3s Exp_s = math::expMap(s);
    Eigen::Matrix4s Exp_s_2 = Eigen::Matrix4s::Identity();

    s_t theta = s.head<3>().norm();
    Eigen::Matrix3s R = Matrix3s::Zero();
    Eigen::Matrix3s qss = math::makeSkewSymmetric(s.head<3>());
    Eigen::Matrix3s qss2 = qss * qss;
    Eigen::Matrix3s P = Eigen::Matrix3s::Zero();

    if (theta < EPSILON_EXPMAP_THETA)
    {
      R = Matrix3s::Identity() + qss + 0.5 * qss2;
      P = Matrix3s::Identity() + 0.5 * qss + (1 / 6) * qss2;
    }
    else
    {
      R = Matrix3s::Identity() + (sin(theta) / theta) * qss
          + ((1 - cos(theta)) / (theta * theta)) * qss2;
      P = Matrix3s::Identity() + ((1 - cos(theta)) / (theta * theta)) * qss
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
    Eigen::Vector6s s = Eigen::Vector6s::Random();
    s.tail<3>() = Eigen::Vector3s::Zero();
    Eigen::Isometry3s Exp_s = math::expAngular(s.head<3>());
    Eigen::Matrix4s Exp_s_2 = Eigen::Matrix4s::Identity();

    s_t theta = s.head<3>().norm();
    Eigen::Matrix3s R = Matrix3s::Zero();
    Eigen::Matrix3s qss = math::makeSkewSymmetric(s.head<3>());
    Eigen::Matrix3s qss2 = qss * qss;
    Eigen::Matrix3s P = Eigen::Matrix3s::Zero();

    if (theta < EPSILON_EXPMAP_THETA)
    {
      R = Matrix3s::Identity() + qss + 0.5 * qss2;
      P = Matrix3s::Identity() + 0.5 * qss + (1 / 6) * qss2;
    }
    else
    {
      R = Matrix3s::Identity() + (sin(theta) / theta) * qss
          + ((1 - cos(theta)) / (theta * theta)) * qss2;
      P = Matrix3s::Identity() + ((1 - cos(theta)) / (theta * theta)) * qss
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
    Eigen::Vector6s s = Eigen::Vector6s::Random();
    s.head<3>() = Eigen::Vector3s::Zero();
    Eigen::Isometry3s Exp_s(Eigen::Translation3s(s.tail<3>()));
    Eigen::Matrix4s Exp_s_2 = Eigen::Matrix4s::Identity();

    s_t theta = s.head<3>().norm();
    Eigen::Matrix3s R = Matrix3s::Zero();
    Eigen::Matrix3s qss = math::makeSkewSymmetric(s.head<3>());
    Eigen::Matrix3s qss2 = qss * qss;
    Eigen::Matrix3s P = Eigen::Matrix3s::Zero();

    if (theta < EPSILON_EXPMAP_THETA)
    {
      R = Matrix3s::Identity() + qss + 0.5 * qss2;
      P = Matrix3s::Identity() + 0.5 * qss + (1 / 6) * qss2;
    }
    else
    {
      R = Matrix3s::Identity() + (sin(theta) / theta) * qss
          + ((1 - cos(theta)) / (theta * theta)) * qss2;
      P = Matrix3s::Identity() + ((1 - cos(theta)) / (theta * theta)) * qss
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
  s_t min = -1e+128;
  s_t max = +1e+128;

  for (int idxTest = 0; idxTest < numExpTests; ++idxTest)
  {
    Eigen::Vector3s randomS = Eigen::Vector3s::Zero();

    for (int i = 0; i < 3; ++i)
      randomS[i] = Random::uniform(min, max);

    Eigen::Isometry3s T = math::expAngular(randomS);
    EXPECT_TRUE(math::verifyTransform(T));
  }

  for (int idxTest = 0; idxTest < numExpTests; ++idxTest)
  {
    Eigen::Vector6s randomS = Eigen::Vector6s::Zero();

    for (int i = 0; i < 6; ++i)
      randomS[i] = Random::uniform(min, max);

    Eigen::Isometry3s T = math::expMap(randomS);
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
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Vector6s V = Eigen::Vector6s::Random();

    Eigen::Vector6s AdTV = AdT(T, V);

    // Ad(T, V) = T * [V] * InvT
    Eigen::Matrix4s T_V_InvT
        = T.matrix() * toMatrixForm(V) * T.inverse().matrix();
    Eigen::Vector6s T_V_InvT_se3 = fromMatrixForm(T_V_InvT);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), T_V_InvT_se3(j), LIE_GROUP_OPT_TOL);

    // Ad(T, V) = [R 0; [p]R R] * V
    Eigen::Matrix6s AdTMatrix = Eigen::Matrix6s::Zero();
    AdTMatrix.topLeftCorner<3, 3>() = T.linear();
    AdTMatrix.bottomRightCorner<3, 3>() = T.linear();
    AdTMatrix.bottomLeftCorner<3, 3>()
        = math::makeSkewSymmetric(T.translation()) * T.linear();
    Eigen::Vector6s AdTMatrix_V = AdTMatrix * V;
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTMatrix_V(j), LIE_GROUP_OPT_TOL);
  }

  // AdR == AdT([R 0; 0 1], V)
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Isometry3s R = Eigen::Isometry3s::Identity();
    R.linear() = T.linear();
    Eigen::Vector6s V = Eigen::Vector6s::Random();

    Eigen::Vector6s AdTV = AdT(R, V);
    Eigen::Vector6s AdRV = AdR(T, V);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdRV(j), LIE_GROUP_OPT_TOL);
  }

  // AdTAngular == AdT(T, se3(w, 0))
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Vector3s w = Eigen::Vector3s::Random();
    Eigen::Vector6s V = Eigen::Vector6s::Zero();
    V.head<3>() = w;

    Eigen::Vector6s AdTV = AdT(T, V);
    Eigen::Vector6s AdTAng = AdTAngular(T, w);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTAng(j), LIE_GROUP_OPT_TOL);
  }

  // AdTLinear == AdT(T, se3(w, 0))
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Vector3s v = Eigen::Vector3s::Random();
    Eigen::Vector6s V = Eigen::Vector6s::Zero();
    V.tail<3>() = v;

    Eigen::Vector6s AdTV = AdT(T, V);
    Eigen::Vector6s AdTLin = AdTLinear(T, v);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTLin(j), LIE_GROUP_OPT_TOL);
  }

  // AdTJac
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Vector3s v = Eigen::Vector3s::Random();
    Eigen::Vector6s V = Eigen::Vector6s::Zero();
    V.tail<3>() = v;

    Eigen::Vector6s AdTV = AdT(T, V);
    Eigen::Vector6s AdTLin = AdTLinear(T, v);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdTV(j), AdTLin(j), LIE_GROUP_OPT_TOL);
  }

  // AdInvT
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Isometry3s InvT = T.inverse();
    Eigen::Vector6s V = Eigen::Vector6s::Random();

    Eigen::Vector6s Ad_InvT = AdT(InvT, V);
    Eigen::Vector6s AdInv_T = AdInvT(T, V);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(Ad_InvT(j), AdInv_T(j), LIE_GROUP_OPT_TOL);
  }

  // AdInvRLinear
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Vector3s v = Eigen::Vector3s::Random();
    Eigen::Vector6s V = Eigen::Vector6s::Zero();
    V.tail<3>() = v;
    Eigen::Isometry3s R = Eigen::Isometry3s::Identity();
    R.linear() = T.linear();

    Eigen::Vector6s AdT_ = AdT(R.inverse(), V);
    Eigen::Vector6s AdInvRLinear_ = AdInvRLinear(T, v);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(AdT_(j), AdInvRLinear_(j), LIE_GROUP_OPT_TOL);
  }

  // dAdT
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Vector6s F = Eigen::Vector6s::Random();

    Eigen::Vector6s dAdTF = dAdT(T, F);

    // dAd(T, F) = [R 0; [p]R R]^T * F
    Eigen::Matrix6s AdTMatrix = Eigen::Matrix6s::Zero();
    AdTMatrix.topLeftCorner<3, 3>() = T.linear();
    AdTMatrix.bottomRightCorner<3, 3>() = T.linear();
    AdTMatrix.bottomLeftCorner<3, 3>()
        = math::makeSkewSymmetric(T.translation()) * T.linear();
    Eigen::Vector6s AdTTransMatrix_V = AdTMatrix.transpose() * F;
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdTF(j), AdTTransMatrix_V(j), LIE_GROUP_OPT_TOL);
  }

  // dAdInvT
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Isometry3s InvT = T.inverse();
    Eigen::Vector6s F = Eigen::Vector6s::Random();

    Eigen::Vector6s dAdInvT_F = dAdInvT(T, F);

    //
    Eigen::Vector6s dAd_InvTF = dAdT(InvT, F);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdInvT_F(j), dAd_InvTF(j), LIE_GROUP_OPT_TOL);

    // dAd(T, F) = [R 0; [p]R R]^T * F
    Eigen::Matrix6s AdInvTMatrix = Eigen::Matrix6s::Zero();
    AdInvTMatrix.topLeftCorner<3, 3>() = InvT.linear();
    AdInvTMatrix.bottomRightCorner<3, 3>() = InvT.linear();
    AdInvTMatrix.bottomLeftCorner<3, 3>()
        = math::makeSkewSymmetric(InvT.translation()) * InvT.linear();
    Eigen::Vector6s AdInvTTransMatrix_V = AdInvTMatrix.transpose() * F;
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdInvT_F(j), AdInvTTransMatrix_V(j), LIE_GROUP_OPT_TOL);
  }

  // dAdInvR
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s t = Eigen::Vector6s::Random();
    Eigen::Isometry3s T = math::expMap(t);
    Eigen::Isometry3s InvT = T.inverse();
    Eigen::Isometry3s InvR = Eigen::Isometry3s::Identity();
    InvR.linear() = InvT.linear();
    Eigen::Vector6s F = Eigen::Vector6s::Random();

    Eigen::Vector6s dAdInvR_F = dAdInvR(T, F);

    //
    Eigen::Vector6s dAd_InvTF = dAdT(InvR, F);

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dAdInvR_F(j), dAd_InvTF(j), LIE_GROUP_OPT_TOL);
  }

  // ad
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s V = Eigen::Vector6s::Random();
    Eigen::Vector6s W = Eigen::Vector6s::Random();

    Eigen::Vector6s ad_V_W = ad(V, W);

    //
    Eigen::Matrix6s adV_Matrix = Eigen::Matrix6s::Zero();
    adV_Matrix.topLeftCorner<3, 3>() = math::makeSkewSymmetric(V.head<3>());
    adV_Matrix.bottomRightCorner<3, 3>() = math::makeSkewSymmetric(V.head<3>());
    adV_Matrix.bottomLeftCorner<3, 3>() = math::makeSkewSymmetric(V.tail<3>());
    Eigen::Vector6s adV_Matrix_W = adV_Matrix * W;

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(ad_V_W(j), adV_Matrix_W(j), LIE_GROUP_OPT_TOL);
  }

  // dad
  for (int i = 0; i < numTest; ++i)
  {
    Eigen::Vector6s V = Eigen::Vector6s::Random();
    Eigen::Vector6s F = Eigen::Vector6s::Random();

    Eigen::Vector6s dad_V_F = dad(V, F);

    //
    Eigen::Matrix6s dadV_Matrix = Eigen::Matrix6s::Zero();
    dadV_Matrix.topLeftCorner<3, 3>() = math::makeSkewSymmetric(V.head<3>());
    dadV_Matrix.bottomRightCorner<3, 3>()
        = math::makeSkewSymmetric(V.head<3>());
    dadV_Matrix.bottomLeftCorner<3, 3>() = math::makeSkewSymmetric(V.tail<3>());
    Eigen::Vector6s dadV_Matrix_F = dadV_Matrix.transpose() * F;

    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(dad_V_F(j), dadV_Matrix_F(j), LIE_GROUP_OPT_TOL);
  }
}
#endif
