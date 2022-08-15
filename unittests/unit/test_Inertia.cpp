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

#include <dart/dynamics/Inertia.hpp>
#include <dart/math/Random.hpp>
#include <gtest/gtest.h>

#include "TestHelpers.hpp"

using namespace dart;

dynamics::Inertia generateRandomInertia()
{
  const auto mass = math::Random::uniform<s_t>(0.1, 10.0);
  const auto com = math::Random::uniform<Eigen::Vector3s>(-5, 5);
  const auto i_xx = math::Random::uniform<s_t>(0.1, 1);
  const auto i_yy = math::Random::uniform<s_t>(0.1, 1);
  const auto i_zz = math::Random::uniform<s_t>(0.1, 1);
  const auto i_xy = math::Random::uniform<s_t>(-1, 1);
  const auto i_xz = math::Random::uniform<s_t>(-1, 1);
  const auto i_yz = math::Random::uniform<s_t>(-1, 1);

  dynamics::Inertia inertia(
      mass, com[0], com[1], com[2], i_xx, i_yy, i_zz, i_xy, i_xz, i_yz);
  return inertia;
}

//==============================================================================
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

//==============================================================================
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

//==============================================================================
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

//==============================================================================
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