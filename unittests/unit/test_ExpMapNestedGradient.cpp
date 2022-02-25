

#include <iostream>

#include <gtest/gtest.h>

#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/PrismaticJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/TranslationalJoint.hpp"
#include "dart/dynamics/WeldJoint.hpp"
#include "dart/math/FiniteDifference.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/simulation/World.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

#define LIE_GROUP_OPT_TOL 1e-12

TEST(EXPMAP_MAG, RANDOM)
{
  Eigen::Vector3s screw = Eigen::Vector3s::Random();
  Eigen::Matrix3s screwR = math::expAngular(screw).linear();

  Eigen::Matrix3s analytical = math::expMapMagGradient(screw);
  Eigen::Matrix3s bruteForce
      = math::finiteDifferenceExpMapMagGradient(screw, true);

  if (!equals(analytical, bruteForce, 1e-12))
  {
    std::cout << "screw: " << std::endl << screw << std::endl;
    std::cout << "screwR: " << std::endl << screwR << std::endl;
    std::cout << "Brute force:" << std::endl
              << bruteForce << std::endl
              << "Analytical: " << std::endl
              << analytical << std::endl;
    std::cout << "Not equal" << std::endl;
    EXPECT_TRUE(equals(analytical, bruteForce, 1e-12));
  }
}

TEST(D_LOGMAP, RANDOM)
{
  srand(42);
  for (int i = 0; i < 1000; i++)
  {
    Eigen::Matrix3s R = math::expMapRot(Eigen::Vector3s::Random());
    Eigen::Matrix3s dR = Eigen::Matrix3s::Random();

    Eigen::Vector3s analytical = math::dLogMap(R, dR);
    Eigen::Vector3s bruteForce = math::finiteDifferenceDLogMap(R, dR, true);

    if (!equals(analytical, bruteForce, 1e-10))
    {
      std::cout << "R: " << std::endl << R << std::endl;
      std::cout << "dR: " << std::endl << dR << std::endl;
      std::cout << "Brute force:" << std::endl
                << bruteForce << std::endl
                << "Analytical: " << std::endl
                << analytical << std::endl
                << "Diff: " << std::endl
                << bruteForce - analytical << std::endl;
      std::cout << "Not equal" << std::endl;
      EXPECT_TRUE(equals(analytical, bruteForce, 1e-10));
      return;
    }
  }
}

TEST(EXPMAP_NESTED, CONSTANTS)
{
  Eigen::Vector3s original = Eigen::Vector3s::UnitZ() * M_PI;
  Eigen::Vector3s screw = Eigen::Vector3s::UnitY();

  Eigen::Matrix3s originalR = math::expAngular(original).linear();
  Eigen::Matrix3s screwR = math::expAngular(screw).linear();

  // We're doing total = screwR * originalR

  Eigen::Vector3s analytical = math::expMapNestedGradient(original, screw);
  Eigen::Vector3s bruteForce
      = math::finiteDifferenceExpMapNestedGradient(original, screw, true);

  if (!equals(analytical, bruteForce, 1e-12))
  {
    std::cout << "Not equal" << std::endl;

    std::cout << "original:" << std::endl
              << original << std::endl
              << "screw: " << std::endl
              << screw << std::endl;
    std::cout << "originalR:" << std::endl
              << originalR << std::endl
              << "screwR: " << std::endl
              << screwR << std::endl;

    std::cout << "Brute force:" << std::endl
              << bruteForce << std::endl
              << "Analytical: " << std::endl
              << analytical << std::endl;
    EXPECT_TRUE(equals(analytical, bruteForce, 1e-12));
  }
}

TEST(EXPMAP_NESTED, RANDOM_MATRIX)
{
  srand(42);
  for (int i = 0; i < 1000; i++)
  {
    Eigen::Vector3s original = Eigen::Vector3s::Random();
    Eigen::Vector3s screw = Eigen::Vector3s::Random();

    Eigen::MatrixXs R_s = math::expMapRot(screw);
    Eigen::MatrixXs R_o = math::expMapRot(original);
    Eigen::MatrixXs R = R_s * R_o;
    Eigen::Matrix3s analytical = math::makeSkewSymmetric(screw) * R_o;

    Eigen::Matrix3s bruteForce;
    bool useRidders = true;
    s_t eps = useRidders ? 1e-3 : 1e-7;
    math::finiteDifference<Eigen::Matrix3s>(
        [&](/* in*/ s_t eps,
            /*out*/ Eigen::Matrix3s& perturbed) {
          perturbed = expMapRot(screw * eps) * R_o;
          return true;
        },
        bruteForce,
        eps,
        useRidders);

    if (!equals(analytical, bruteForce, 1e-12))
    {
      std::cout << "Not equal" << std::endl;
      std::cout << "Brute force:" << std::endl
                << bruteForce << std::endl
                << "Analytical: " << std::endl
                << analytical << std::endl
                << "Diff: " << std::endl
                << bruteForce - analytical << std::endl;
      EXPECT_TRUE(equals(analytical, bruteForce, 1e-12));
      return;
    }

    Eigen::Vector3s analyticalLogMap = math::dLogMap(R, analytical);
    Eigen::Vector3s bruteForceLogMap
        = math::finiteDifferenceDLogMap(R, analytical, true);
    if (!equals(analyticalLogMap, bruteForceLogMap, 1e-10))
    {
      std::cout << "Brute force:" << std::endl
                << bruteForceLogMap << std::endl
                << "Analytical: " << std::endl
                << analyticalLogMap << std::endl
                << "Diff: " << std::endl
                << bruteForceLogMap - analyticalLogMap << std::endl;
      std::cout << "Not equal" << std::endl;
      EXPECT_TRUE(equals(analyticalLogMap, bruteForceLogMap, 1e-10));
      return;
    }
  }
}

TEST(EXPMAP_NESTED, RANDOM)
{
  srand(42);
  for (int i = 0; i < 1000; i++)
  {
    Eigen::Vector3s original = Eigen::Vector3s::Random();
    Eigen::Vector3s screw = Eigen::Vector3s::Random();

    Eigen::Matrix3s originalR = math::expAngular(original).linear();
    Eigen::Matrix3s screwR = math::expAngular(screw).linear();

    // We're doing total = screwR * originalR

    Eigen::Vector3s analytical = math::expMapNestedGradient(original, screw);
    Eigen::Vector3s bruteForce
        = math::finiteDifferenceExpMapNestedGradient(original, screw, true);

    if (!equals(analytical, bruteForce, 1e-10))
    {
      std::cout << "Not equal" << std::endl;

      std::cout << "original:" << std::endl
                << original << std::endl
                << "screw: " << std::endl
                << screw << std::endl;
      std::cout << "originalR:" << std::endl
                << originalR << std::endl
                << "screwR: " << std::endl
                << screwR << std::endl;

      std::cout << "Brute force:" << std::endl
                << bruteForce << std::endl
                << "Analytical: " << std::endl
                << analytical << std::endl;
      EXPECT_TRUE(equals(analytical, bruteForce, 1e-10));

      Eigen::Vector3s analytical = math::expMapNestedGradient(original, screw);
      (void)analytical;
      return;
    }
  }
}