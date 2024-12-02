#include <iostream>
#include <memory>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/WithRespectTo.hpp"
#include "dart/utils/AccelerationSmoother.hpp"

#include "TestHelpers.hpp"

#define ALL_TESTS

using namespace dart;
using namespace utils;
using namespace biomechanics;

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, RECOVERY_NO_GROUPS)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  Eigen::VectorXs originalLinkMasses = standard.skeleton->getLinkMasses();
  standard.skeleton->setLinearizedMasses(
      standard.skeleton->getLinearizedMasses());
  Eigen::VectorXs recoveredLinkMasses = standard.skeleton->getLinkMasses();

  EXPECT_TRUE(equals(originalLinkMasses, recoveredLinkMasses));
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, RECOVERY_GROUPS)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();
  Eigen::VectorXs originalLinkMasses = standard.skeleton->getLinkMasses();
  standard.skeleton->setLinearizedMasses(
      standard.skeleton->getLinearizedMasses());
  Eigen::VectorXs recoveredLinkMasses = standard.skeleton->getLinkMasses();

  EXPECT_TRUE(equals(originalLinkMasses, recoveredLinkMasses));
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, JACOBIAN_BACK_TO_GROUP_MASSES)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  Eigen::MatrixXs analytical
      = standard.skeleton->getGroupMassesJacobianWrtLinearizedMasses();
  Eigen::MatrixXs fd
      = standard.skeleton
            ->finiteDifferenceGroupMassesJacobianWrtLinearizedMasses();

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Total mass: " << standard.skeleton->getMass() << std::endl;
    std::cout << "Group masses: " << std::endl
              << standard.skeleton->getGroupMasses() << std::endl;
    std::cout << "Linearized mass: " << std::endl
              << standard.skeleton->getLinearizedMasses() << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, JACOBIAN_OF_COM)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  Eigen::MatrixXs analytical
      = standard.skeleton->getUnnormalizedCOMJacobianWrtLinearizedMasses();
  Eigen::MatrixXs fd
      = standard.skeleton
            ->finiteDifferenceUnnormalizedCOMJacobianWrtLinearizedMasses();

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Total mass: " << standard.skeleton->getMass() << std::endl;
    std::cout << "Group masses: " << std::endl
              << standard.skeleton->getGroupMasses() << std::endl;
    std::cout << "Linearized mass: " << std::endl
              << standard.skeleton->getLinearizedMasses() << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, JACOBIAN_OF_COM_FD_ACC)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  Eigen::MatrixXs analytical
      = standard.skeleton->getUnnormalizedCOMFDAccJacobianWrtLinearizedMasses();
  Eigen::MatrixXs fd
      = standard.skeleton
            ->finiteDifferenceUnnormalizedCOMFDAccJacobianWrtLinearizedMasses();

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Total mass: " << standard.skeleton->getMass() << std::endl;
    std::cout << "Group masses: " << std::endl
              << standard.skeleton->getGroupMasses() << std::endl;
    std::cout << "Linearized mass: " << std::endl
              << standard.skeleton->getLinearizedMasses() << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, JACOBIAN_OF_COM_ANALYTICAL_ACC)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  Eigen::MatrixXs analytical
      = standard.skeleton
            ->getUnnormalizedCOMAnalyticalAccJacobianWrtLinearizedMasses();
  Eigen::MatrixXs fd
      = standard.skeleton
            ->finiteDifferenceUnnormalizedCOMAnalyticalAccJacobianWrtLinearizedMasses();

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Total mass: " << standard.skeleton->getMass() << std::endl;
    std::cout << "Group masses: " << std::endl
              << standard.skeleton->getGroupMasses() << std::endl;
    std::cout << "Linearized mass: " << std::endl
              << standard.skeleton->getLinearizedMasses() << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, JACOBIAN_OF_ACC_OFFSET)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  Eigen::MatrixXs analytical
      = standard.skeleton
            ->getUnnormalizedCOMAccelerationOffsetJacobianWrtLinearizedMasses();
  Eigen::MatrixXs fd
      = standard.skeleton
            ->finiteDifferenceUnnormalizedCOMAccelerationOffsetJacobianWrtLinearizedMasses();

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Total mass: " << standard.skeleton->getMass() << std::endl;
    std::cout << "Group masses: " << std::endl
              << standard.skeleton->getGroupMasses() << std::endl;
    std::cout << "Linearized mass: " << std::endl
              << standard.skeleton->getLinearizedMasses() << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, JACOBIAN_OF_C)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  Eigen::MatrixXs analytical = standard.skeleton->getJacobianOfC(
      neural::WithRespectTo::LINEARIZED_MASSES);
  Eigen::MatrixXs fd = standard.skeleton->finiteDifferenceJacobianOfC(
      neural::WithRespectTo::LINEARIZED_MASSES);

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Didn't agree on Jac of C wrt LINEARIZED_MASSES" << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif

#ifdef ALL_TESTS
TEST(LINEARIZED_MASS_MAPPING, JACOBIAN_OF_M)
{
  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/grf/Sprinter/Models/"
      "optimized_scale_and_markers.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  Eigen::VectorXs randomRightHandSide = standard.skeleton->getRandomPose();

  Eigen::MatrixXs analytical = standard.skeleton->getJacobianOfM(
      randomRightHandSide, neural::WithRespectTo::LINEARIZED_MASSES);
  Eigen::MatrixXs fd = standard.skeleton->finiteDifferenceJacobianOfM(
      randomRightHandSide, neural::WithRespectTo::LINEARIZED_MASSES);

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Didn't agree on Jac of M wrt LINEARIZED_MASSES" << std::endl;
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif