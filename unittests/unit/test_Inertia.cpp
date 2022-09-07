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

#include <cstdlib>

#include <dart/dynamics/Inertia.hpp>
#include <dart/math/Random.hpp>
#include <gtest/gtest.h>

#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"

#include "TestHelpers.hpp"

using namespace dart;

#define ALL_TESTS

dynamics::Inertia generateRandomInertia()
{
  const auto mass = math::Random::uniform<s_t>(0.1, 10.0);
  const auto com = math::Random::uniform<Eigen::Vector3s>(-5, 5);

  Eigen::Vector6s dimsAndEuler = Eigen::Vector6s::Random();
  // Dims
  dimsAndEuler.head<3>() *= 10;
  dimsAndEuler.head<3>() += Eigen::Vector3s::Ones();
  // Euler
  dimsAndEuler.tail<3>() *= M_PI / 4;

  Eigen::Vector6s moment
      = dynamics::Inertia::computeMomentVector(mass, dimsAndEuler);

  const auto i_xx = moment(0);
  const auto i_yy = moment(1);
  const auto i_zz = moment(2);
  const auto i_xy = moment(3);
  const auto i_xz = moment(4);
  const auto i_yz = moment(5);

  dynamics::Inertia inertia(
      mass, com[0], com[1], com[2], i_xx, i_yy, i_zz, i_xy, i_xz, i_yz);
  return inertia;
}

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, RotatedCubeRecovery)
{
  const int numIter = 100;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    Eigen::Vector3s cubeDims = Eigen::Vector3s::Random();
    cubeDims(0) = abs(cubeDims(0));
    cubeDims(1) = abs(cubeDims(1));
    cubeDims(2) = abs(cubeDims(2));
    Eigen::Vector3s cubeEuler = Eigen::Vector3s::Random() * 0.1;

    Eigen::Vector6s dimsAndEuler;
    dimsAndEuler.head<3>() = cubeDims;
    dimsAndEuler.tail<3>() = cubeEuler;
    s_t mass = ((s_t)rand() / RAND_MAX) + 1.0;

    Eigen::Vector6s momentVec
        = dynamics::Inertia::computeMomentVector(mass, dimsAndEuler);
    Eigen::Vector6s recovered
        = dynamics::Inertia::computeDimsAndEuler(mass, momentVec);

    Eigen::Vector3s recoveredEuler = recovered.tail<3>();

    Eigen::Matrix3s originalR = math::eulerXYZToMatrix(cubeEuler);
    Eigen::Matrix3s recoveredR = math::eulerXYZToMatrix(recoveredEuler);
    if (!equals(originalR, recoveredR, 1e-8))
    {
      std::cout << "Recovered Rotation != Original Rotation:" << std::endl;
      std::cout << "Recovered:" << std::endl << recoveredR << std::endl;
      std::cout << "Original:" << std::endl << originalR << std::endl;
      std::cout << "Diff:" << std::endl << recoveredR - originalR << std::endl;
      EXPECT_TRUE(equals(recoveredR, originalR, 1e-8));
    }

    if (!equals(recovered, dimsAndEuler, 1e-8))
    {
      std::cout << "Recovered != Original cube dims:" << std::endl;
      std::cout << "Recovered:" << std::endl << recovered << std::endl;
      std::cout << "Original:" << std::endl << dimsAndEuler << std::endl;
      std::cout << "Diff:" << std::endl
                << recovered - dimsAndEuler << std::endl;
      EXPECT_TRUE(equals(recovered, dimsAndEuler, 1e-8));
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, MomentJac)
{
  const int numIter = 100;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    Eigen::Vector3s cubeDims = Eigen::Vector3s::Random();
    cubeDims(0) = abs(cubeDims(0));
    cubeDims(1) = abs(cubeDims(1));
    cubeDims(2) = abs(cubeDims(2));
    Eigen::Vector3s cubeEuler = Eigen::Vector3s::Random() * 0.1;

    Eigen::Vector6s dimsAndEuler;
    dimsAndEuler.head<3>() = cubeDims;
    dimsAndEuler.tail<3>() = cubeEuler;
    s_t mass = ((s_t)rand() / RAND_MAX) + 1.0;

    Eigen::Matrix6s analytical
        = dynamics::Inertia::computeMomentVectorJacWrtDimsAndEuler(
            mass, dimsAndEuler);
    Eigen::Matrix6s fd
        = dynamics::Inertia::finiteDifferenceMomentVectorJacWrtDimsAndEuler(
            mass, dimsAndEuler);

    if (!equals(analytical, fd, 1e-8))
    {
      std::cout << "Jac of moment vector not equal:" << std::endl;
      std::cout << "Analytical:" << std::endl << analytical << std::endl;
      std::cout << "FD:" << std::endl << fd << std::endl;
      std::cout << "Diff:" << std::endl << analytical - fd << std::endl;
      EXPECT_TRUE(equals(analytical, fd, 1e-8));
    }

    Eigen::Vector6s analyticalWrtMass
        = dynamics::Inertia::computeMomentVectorGradWrtMass(mass, dimsAndEuler);
    Eigen::Vector6s fdWrtMass
        = dynamics::Inertia::finiteDifferenceMomentVectorGradWrtMass(
            mass, dimsAndEuler);

    if (!equals(analyticalWrtMass, fdWrtMass, 1e-8))
    {
      std::cout << "Grad of moment vector not equal:" << std::endl;
      std::cout << "Analytical:" << std::endl << analyticalWrtMass << std::endl;
      std::cout << "FD:" << std::endl << fdWrtMass << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalWrtMass - fdWrtMass << std::endl;
      EXPECT_TRUE(equals(analyticalWrtMass, fdWrtMass, 1e-8));
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, CubeDimsRecovery)
{
  const int numIter = 10;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    Eigen::Vector3s cubeDims = Eigen::Vector3s::Random();
    cubeDims(0) = abs(cubeDims(0));
    cubeDims(1) = abs(cubeDims(1));
    cubeDims(2) = abs(cubeDims(2));
    s_t mass = ((s_t)rand() / RAND_MAX) + 1.0;
    dynamics::Inertia inertia
        = dynamics::Inertia::createCubeInertia(mass, cubeDims);
    EXPECT_TRUE(inertia.verify());

    Eigen::Vector3s recoveredDims = inertia.getImpliedCubeDimensions();
    if (!equals(recoveredDims, cubeDims, 1e-8))
    {
      std::cout << "Recovered != Original cube dims:" << std::endl;
      std::cout << "Recovered:" << std::endl << recoveredDims << std::endl;
      std::cout << "Original:" << std::endl << cubeDims << std::endl;
      std::cout << "Diff:" << std::endl
                << recoveredDims - cubeDims << std::endl;
      EXPECT_TRUE(equals(recoveredDims, cubeDims, 1e-8));
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, CubeDimsGrads)
{
  const int numIter = 10;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    Eigen::Vector3s cubeDims = Eigen::Vector3s::Random();
    cubeDims(0) = abs(cubeDims(0));
    cubeDims(1) = abs(cubeDims(1));
    cubeDims(2) = abs(cubeDims(2));
    cubeDims = cubeDims * 0.9 + Eigen::Vector3s::Ones() * 0.1;
    s_t mass = ((s_t)rand() / RAND_MAX) + 1.0;
    dynamics::Inertia inertia
        = dynamics::Inertia::createCubeInertia(mass, cubeDims);
    EXPECT_TRUE(inertia.verify());

    Eigen::Vector3s analyticalDimsWrtMass
        = inertia.getImpliedCubeDimensionsGradientWrtMass();
    Eigen::Vector3s fdDimsWrtMass
        = inertia.finiteDifferenceImpliedCubeDimensionsGradientWrtMass();

    if (!equals(analyticalDimsWrtMass, fdDimsWrtMass, 1e-8))
    {
      std::cout << "Error on dims wrt mass:" << std::endl;
      std::cout << "Analytical:" << std::endl
                << analyticalDimsWrtMass << std::endl;
      std::cout << "FD:" << std::endl << fdDimsWrtMass << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalDimsWrtMass - fdDimsWrtMass << std::endl;
      EXPECT_TRUE(equals(analyticalDimsWrtMass, fdDimsWrtMass, 1e-8));
      return;
    }

    Eigen::MatrixXs analyticalDimsWrtMoment
        = inertia.getImpliedCubeDimensionsJacobianWrtMomentVector();
    Eigen::MatrixXs fdDimsWrtMoment
        = inertia
              .finiteDifferenceImpliedCubeDimensionsJacobianWrtMomentVector();

    if (!equals(analyticalDimsWrtMoment, fdDimsWrtMoment, 1e-8))
    {
      std::cout << "Error on dims wrt moment vec:" << std::endl;
      std::cout << "Analytical:" << std::endl
                << analyticalDimsWrtMoment << std::endl;
      std::cout << "FD:" << std::endl << fdDimsWrtMoment << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalDimsWrtMoment - fdDimsWrtMoment << std::endl;
      EXPECT_TRUE(equals(analyticalDimsWrtMoment, fdDimsWrtMoment, 1e-8));
      return;
    }

    s_t analyticalDensityWrtMass
        = inertia.getImpliedCubeDensityGradientWrtMass();
    s_t fdDensityWrtMass
        = inertia.finiteDifferenceImpliedCubeDensityGradientWrtMass();

    if (abs(analyticalDensityWrtMass - fdDensityWrtMass) > 1e-8)
    {
      std::cout << "Error on density wrt mass:" << std::endl;
      std::cout << "Analytical:" << std::endl
                << analyticalDensityWrtMass << std::endl;
      std::cout << "FD:" << std::endl << fdDensityWrtMass << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalDensityWrtMass - fdDensityWrtMass << std::endl;
      EXPECT_NEAR(analyticalDensityWrtMass, fdDensityWrtMass, 1e-8);
      return;
    }

    Eigen::Vector6s analyticalDensityWrtMoment
        = inertia.getImpliedCubeDensityGradientWrtMomentVector();
    Eigen::Vector6s fdDensityWrtMoment
        = inertia.finiteDifferenceImpliedCubeDensityGradientWrtMomentVector();

    if (!equals(analyticalDensityWrtMoment, fdDensityWrtMoment, 1e-8))
    {
      std::cout << "Error on density wrt moment:" << std::endl;
      std::cout << "Analytical:" << std::endl
                << analyticalDensityWrtMoment << std::endl;
      std::cout << "FD:" << std::endl << fdDensityWrtMoment << std::endl;
      std::cout << "Diff:" << std::endl
                << analyticalDensityWrtMoment - fdDensityWrtMoment << std::endl;
      EXPECT_TRUE(equals(analyticalDensityWrtMoment, fdDensityWrtMoment, 1e-8));
      return;
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, Verification)
{
  const int numIter = 10;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    dynamics::Inertia inertia = generateRandomInertia();
    EXPECT_TRUE(inertia.verify());
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, MassGradients)
{
  const int numIter = 10;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    dynamics::Inertia inertia = generateRandomInertia();

    Eigen::Matrix6s analytical = inertia.getSpatialTensorGradientWrtMass();
    Eigen::Matrix6s fd = inertia.finiteDifferenceSpatialTensorGradientWrtMass();

    if (!equals(analytical, fd, 1e-8))
    {
      std::cout << "Gradient of spatial inertia wrt mass does not match!"
                << std::endl;
      std::cout << "Analytical: " << std::endl << analytical << std::endl;
      std::cout << "Finite difference: " << std::endl << fd << std::endl;
      std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
      EXPECT_TRUE(equals(analytical, fd, 1e-8));
      return;
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, COMGradients)
{
  const int numIter = 10;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    dynamics::Inertia inertia = generateRandomInertia();

    for (int index = 0; index < 3; index++)
    {
      Eigen::Matrix6s analytical
          = inertia.getSpatialTensorGradientWrtCOM(index);
      Eigen::Matrix6s fd
          = inertia.finiteDifferenceSpatialTensorGradientWrtCOM(index);

      if (!equals(analytical, fd, 1e-8))
      {
        std::cout << "Gradient of spatial inertia wrt COM " << index
                  << " does not match!" << std::endl;
        std::cout << "Analytical: " << std::endl << analytical << std::endl;
        std::cout << "Finite difference: " << std::endl << fd << std::endl;
        std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
        EXPECT_TRUE(equals(analytical, fd, 1e-8));
        return;
      }
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, MomentGradients)
{
  const int numIter = 10;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    dynamics::Inertia inertia = generateRandomInertia();

    for (int index = 0; index < 6; index++)
    {
      Eigen::Matrix6s analytical
          = inertia.getSpatialTensorGradientWrtMomentVector(index);
      Eigen::Matrix6s fd
          = inertia.finiteDifferenceSpatialTensorGradientWrtMomentVector(index);

      if (!equals(analytical, fd, 1e-8))
      {
        std::cout << "Gradient of spatial inertia wrt Moment vector " << index
                  << " does not match!" << std::endl;
        std::cout << "Analytical: " << std::endl << analytical << std::endl;
        std::cout << "Finite difference: " << std::endl << fd << std::endl;
        std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
        EXPECT_TRUE(equals(analytical, fd, 1e-8));
        return;
      }
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(Inertia, DimsAndEulersGradients)
{
  const int numIter = 10;

  srand(42);

  for (int i = 0; i < numIter; ++i)
  {
    dynamics::Inertia inertia = generateRandomInertia();

    for (int index = 0; index < 6; index++)
    {
      Eigen::Matrix6s analytical
          = inertia.getSpatialTensorGradientWrtDimsAndEulerVector(index);
      Eigen::Matrix6s fd
          = inertia.finiteDifferenceSpatialTensorGradientWrtDimsAndEulerVector(
              index);

      if (!equals(analytical, fd, 1e-8))
      {
        std::cout << "Gradient of spatial inertia wrt dims and eulers vector "
                  << index << " does not match!" << std::endl;
        std::cout << "Analytical: " << std::endl << analytical << std::endl;
        std::cout << "Finite difference: " << std::endl << fd << std::endl;
        std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
        EXPECT_TRUE(equals(analytical, fd, 1e-8));
        return;
      }
    }
  }
}
#endif