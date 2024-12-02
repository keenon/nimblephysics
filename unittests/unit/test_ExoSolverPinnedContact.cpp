#include <iostream>
#include <memory>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/exo/ExoSolverPinnedContact.hpp"
#include "dart/math/MathTypes.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;
using namespace exo;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, COMPARE_ANALYTICAL_TO_IMPLICIT_FD_NO_FORCES)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  (void)solver;

  // Without exo torques and without contact forces
  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs tau = Eigen::VectorXs::Random(realSkel->getNumDofs());
    Eigen::VectorXs exoTorques = Eigen::VectorXs::Zero(0);
    Eigen::VectorXs contactForces = Eigen::VectorXs::Zero(0);

    Eigen::VectorXs analyticalAcc
        = solver.analyticalForwardDynamics(dq, tau, exoTorques, contactForces);
    Eigen::VectorXs implicitAcc
        = solver.implicitForwardDynamics(dq, tau, exoTorques, contactForces);

    if (!equals(analyticalAcc, implicitAcc, 1e-6))
    {
      std::cout << "Analytical: " << analyticalAcc.transpose() << std::endl;
      std::cout << "Implicit: " << implicitAcc.transpose() << std::endl;
      std::cout << "Diff: " << (analyticalAcc - implicitAcc).transpose()
                << std::endl;
      EXPECT_TRUE(equals(analyticalAcc, implicitAcc, 1e-6));
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, COMPARE_ANALYTICAL_TO_FD_CONTACT_JACOBIAN)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::MatrixXs J = solver.getContactJacobian();
    Eigen::MatrixXs J_fd = solver.finiteDifferenceContactJacobian();

    if (!equals(J, J_fd, 1e-6))
    {
      std::cout << "Analytical: " << std::endl << J << std::endl;
      std::cout << "Finite Difference: " << std::endl << J_fd << std::endl;
      std::cout << "Diff: " << std::endl << (J - J_fd) << std::endl;
      EXPECT_TRUE(equals(J, J_fd, 1e-6));
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, COMPARE_ANALYTICAL_TO_IMPLICIT_FD_WITH_FORCES)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add a knee exo
  solver.addMotorDof(realSkel->getDof("knee_angle_r")->getIndexInSkeleton());
  solver.addMotorDof(realSkel->getDof("knee_angle_l")->getIndexInSkeleton());

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  // With exo torques and without contact forces
  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs tau = Eigen::VectorXs::Random(realSkel->getNumDofs());
    Eigen::VectorXs exoTorques = Eigen::VectorXs::Random(2);
    Eigen::VectorXs contactForces = Eigen::VectorXs::Random(3);

    Eigen::VectorXs analyticalAcc
        = solver.analyticalForwardDynamics(dq, tau, exoTorques, contactForces);
    Eigen::VectorXs implicitAcc
        = solver.implicitForwardDynamics(dq, tau, exoTorques, contactForces);

    if (!equals(analyticalAcc, implicitAcc, 1e-6))
    {
      Eigen::VectorXs contactJointTorques
          = solver.getContactJacobian().transpose() * contactForces;
      Eigen::VectorXs MinvContactJointTorques
          = realSkel->getInvMassMatrix() * contactJointTorques;
      Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(analyticalAcc.size(), 4);
      compare.col(0) = analyticalAcc;
      compare.col(1) = implicitAcc;
      compare.col(2) = analyticalAcc - implicitAcc;
      compare.col(3) = MinvContactJointTorques;
      std::cout
          << "Analytical - Implicit - Diff - Minv * Contact Joint Torques:"
          << std::endl
          << compare << std::endl;
      EXPECT_TRUE(equals(analyticalAcc, implicitAcc, 1e-6));
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, TEST_INVERSE_DYNAMICS_HUMAN_TORQUES)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  (void)solver;

  // Without exo torques and without contact forces
  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs tau = Eigen::VectorXs::Random(realSkel->getNumDofs());
    Eigen::VectorXs exoTorques = Eigen::VectorXs::Zero(0);
    Eigen::VectorXs contactForces = Eigen::VectorXs::Zero(0);

    Eigen::VectorXs ddq
        = solver.implicitForwardDynamics(dq, tau, exoTorques, contactForces);

    Eigen::VectorXs recoveredTau
        = solver.estimateHumanTorques(dq, ddq, contactForces, exoTorques);

    if (!equals(tau, recoveredTau, 1e-6))
    {
      std::cout << "Tau: " << tau.transpose() << std::endl;
      std::cout << "Recovered Tau: " << recoveredTau.transpose() << std::endl;
      std::cout << "Diff: " << (tau - recoveredTau).transpose() << std::endl;
      EXPECT_TRUE(equals(tau, recoveredTau, 1e-6));
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, TEST_PINNED_FORWARD_DYNAMICS)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs tau = Eigen::VectorXs::Random(realSkel->getNumDofs());
    std::pair<Eigen::VectorXs, Eigen::VectorXs>
        jointAccelerationsAndContactForces
        = solver.getPinnedVirtualDynamics(dq, tau);
    std::pair<Eigen::MatrixXs, Eigen::VectorXs> jointAccelerationsLinearMap
        = solver.getPinnedVirtualDynamicsLinearMap(dq);
    Eigen::VectorXs ddq = jointAccelerationsAndContactForces.first;
    Eigen::VectorXs contactForces = jointAccelerationsAndContactForces.second;

    Eigen::VectorXs linearMapDdq = jointAccelerationsLinearMap.first * tau
                                   + jointAccelerationsLinearMap.second;
    if (!equals(ddq, linearMapDdq, 1e-6))
    {
      std::cout << "ddq: " << ddq.transpose() << std::endl;
      std::cout << "linearMapDdq: " << linearMapDdq.transpose() << std::endl;
      std::cout << "Diff: " << (ddq - linearMapDdq).transpose() << std::endl;
      EXPECT_TRUE(equals(ddq, linearMapDdq, 1e-6));
      return;
    }

    Eigen::VectorXs ddqRecovered = solver.implicitForwardDynamics(
        dq, tau, Eigen::VectorXs::Zero(0), contactForces);

    Eigen::VectorXs recoveredLinearAcc
        = solver.getContactJacobian() * ddqRecovered;
    if (recoveredLinearAcc.norm() > 1e-6)
    {
      std::cout << "Recoverd Linear Acc: " << recoveredLinearAcc.transpose()
                << std::endl;
      EXPECT_TRUE(recoveredLinearAcc.norm() < 1e-6);
      return;
    }

    if (!equals(ddq, ddqRecovered, 1e-6))
    {
      std::cout << "ddq: " << ddq.transpose() << std::endl;
      std::cout << "ddqRecovered: " << ddqRecovered.transpose() << std::endl;
      std::cout << "Diff: " << (ddq - ddqRecovered).transpose() << std::endl;
      EXPECT_TRUE(equals(ddq, ddqRecovered, 1e-6));
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, TEST_PINNED_INVERSE_DYNAMICS)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs tau = Eigen::VectorXs::Random(realSkel->getNumDofs());
    tau.head<6>().setZero();
    std::pair<Eigen::VectorXs, Eigen::VectorXs>
        jointAccelerationsAndContactForces
        = solver.getPinnedVirtualDynamics(dq, tau);
    Eigen::VectorXs ddq = jointAccelerationsAndContactForces.first;
    Eigen::VectorXs contactForces = jointAccelerationsAndContactForces.second;

    std::pair<Eigen::VectorXs, Eigen::VectorXs> jointTorquesAndContactForces
        = solver.getPinnedTotalTorques(
            dq,
            ddq,
            Eigen::VectorXs::Zero(dq.size()),
            Eigen::VectorXs::Zero(pins.size() * 3));
    std::pair<Eigen::MatrixXs, Eigen::VectorXs> jointTorquesLinearMap
        = solver.getPinnedTotalTorquesLinearMap(dq);
    Eigen::VectorXs recoveredTau = jointTorquesAndContactForces.first;
    Eigen::VectorXs recoveredContactForces
        = jointTorquesAndContactForces.second;

    Eigen::VectorXs linearMapTau
        = jointTorquesLinearMap.first * ddq + jointTorquesLinearMap.second;
    if (!equals(tau, linearMapTau, 1e-6))
    {
      std::cout << "Tau: " << tau.transpose() << std::endl;
      std::cout << "linearMapTau: " << linearMapTau.transpose() << std::endl;
      std::cout << "Diff: " << (tau - linearMapTau).transpose() << std::endl;
      EXPECT_TRUE(equals(tau, linearMapTau, 1e-6));
      return;
    }

    if (!equals(tau, recoveredTau, 1e-6))
    {
      std::cout << "Tau: " << tau.transpose() << std::endl;
      std::cout << "Recovered Tau: " << recoveredTau.transpose() << std::endl;
      std::cout << "Diff: " << (tau - recoveredTau).transpose() << std::endl;
      EXPECT_TRUE(equals(tau, recoveredTau, 1e-6));
      return;
    }

    if (!equals(contactForces, recoveredContactForces, 1e-6))
    {
      std::cout << "Contact Forces: " << contactForces.transpose() << std::endl;
      std::cout << "Recovered Contact Forces: "
                << recoveredContactForces.transpose() << std::endl;
      std::cout << "Diff: "
                << (contactForces - recoveredContactForces).transpose()
                << std::endl;
      EXPECT_TRUE(equals(contactForces, recoveredContactForces, 1e-6));
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, TEST_EXO_TORQUES_LINEAR_MAP)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  // Add a knee exo
  solver.addMotorDof(realSkel->getDof("knee_angle_r")->getIndexInSkeleton());
  solver.addMotorDof(realSkel->getDof("knee_angle_l")->getIndexInSkeleton());

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs tau = Eigen::VectorXs::Random(realSkel->getNumDofs());
    tau.head<6>().setZero();

    Eigen::VectorXs exoTorques = solver.solveFromBiologicalTorques(
        dq,
        tau,
        Eigen::VectorXs::Zero(dq.size()),
        Eigen::VectorXs::Zero(pins.size() * 3));

    std::pair<Eigen::MatrixXs, Eigen::VectorXs> exoTorquesLinearMap
        = solver.getExoTorquesLinearMap(dq);

    Eigen::VectorXs exoTorquesLinearMapTau
        = exoTorquesLinearMap.first * tau + exoTorquesLinearMap.second;

    if (!equals(exoTorques, exoTorquesLinearMapTau, 1e-6))
    {
      std::cout << "Tau: " << exoTorques.transpose() << std::endl;
      std::cout << "exoTorquesLinearMapTau: "
                << exoTorquesLinearMapTau.transpose() << std::endl;
      std::cout << "Diff: " << (exoTorques - exoTorquesLinearMapTau).transpose()
                << std::endl;
      EXPECT_TRUE(equals(exoTorques, exoTorquesLinearMapTau, 1e-6));
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, TEST_EXO_FORWARD_DYNAMICS)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  // Add a knee exo
  solver.addMotorDof(realSkel->getDof("knee_angle_r")->getIndexInSkeleton());
  solver.addMotorDof(realSkel->getDof("knee_angle_l")->getIndexInSkeleton());

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs humanTau = Eigen::VectorXs::Random(realSkel->getNumDofs());
    humanTau.head<6>().setZero();

    std::tuple<Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs> result
        = solver.getPinnedForwardDynamicsForExoAndHuman(dq, humanTau);
    Eigen::VectorXs ddq = std::get<0>(result);
    Eigen::VectorXs contactForce = std::get<1>(result);
    Eigen::VectorXs exoTorques = std::get<2>(result);

    std::pair<Eigen::MatrixXs, Eigen::VectorXs>
        humanTorquesLinearForwardDynamics
        = solver.getPinnedForwardDynamicsForExoAndHumanLinearMap(dq);
    Eigen::VectorXs recoveredDdq
        = humanTorquesLinearForwardDynamics.first * humanTau
          + humanTorquesLinearForwardDynamics.second;

    Eigen::VectorXs recoveredHumanTau
        = solver.estimateHumanTorques(dq, ddq, contactForce, exoTorques);
    if (!equals(humanTau, recoveredHumanTau, 1e-6))
    {
      std::cout << "Tau: " << humanTau.transpose() << std::endl;
      std::cout << "Recovered Tau: " << recoveredHumanTau.transpose()
                << std::endl;
      std::cout << "Diff: " << (humanTau - recoveredHumanTau).transpose()
                << std::endl;
      EXPECT_TRUE(equals(humanTau, recoveredHumanTau, 1e-6));
      return;
    }

    Eigen::VectorXs forwardDdq = solver.implicitForwardDynamics(
        dq, humanTau, exoTorques, contactForce);
    if (!equals(ddq, forwardDdq, 1e-6))
    {
      std::cout << "ddq: " << ddq.transpose() << std::endl;
      std::cout << "forwardDdq: " << forwardDdq.transpose() << std::endl;
      std::cout << "Diff: " << (ddq - forwardDdq).transpose() << std::endl;
      EXPECT_TRUE(equals(ddq, forwardDdq, 1e-6));
      return;
    }

    if (!equals(ddq, recoveredDdq, 1e-6))
    {
      std::cout << "ddq: " << ddq.transpose() << std::endl;
      std::cout << "recoveredDdq: " << recoveredDdq.transpose() << std::endl;
      std::cout << "Diff: " << (ddq - recoveredDdq).transpose() << std::endl;
      EXPECT_TRUE(equals(ddq, recoveredDdq, 1e-6));
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, TEST_RECONSTRUCT_EXO_DYNAMICS)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  // Add a knee exo
  solver.addMotorDof(realSkel->getDof("knee_angle_r")->getIndexInSkeleton());
  solver.addMotorDof(realSkel->getDof("knee_angle_l")->getIndexInSkeleton());

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs ddq = realSkel->getRandomVelocity();
    Eigen::MatrixXs J = solver.getContactJacobian();
    Eigen::VectorXs compliantDdq
        = ddq - J.completeOrthogonalDecomposition().solve(J * ddq);
    Eigen::VectorXs recoveredConstraints = J * compliantDdq;
    if (recoveredConstraints.norm() > 1e-6)
    {
      std::cout << "recoveredConstraints: " << recoveredConstraints.transpose()
                << std::endl;
      EXPECT_TRUE(recoveredConstraints.norm() < 1e-6);
      return;
    }

    std::pair<Eigen::VectorXs, Eigen::VectorXs> humanAndExoTorques
        = solver.getHumanAndExoTorques(dq, compliantDdq);

    std::tuple<Eigen::VectorXs, Eigen::VectorXs, Eigen::VectorXs>
        recoveredResults = solver.getPinnedForwardDynamicsForExoAndHuman(
            dq, humanAndExoTorques.first);
    Eigen::VectorXs recoveredDdq = std::get<0>(recoveredResults);
    Eigen::VectorXs contactForce = std::get<1>(recoveredResults);
    Eigen::VectorXs exoTorques = std::get<2>(recoveredResults);
    (void)contactForce;
    (void)exoTorques;

    if (!equals(compliantDdq, recoveredDdq, 1e-6))
    {
      std::cout << "ddq: " << compliantDdq.transpose() << std::endl;
      std::cout << "recoveredDdq: " << recoveredDdq.transpose() << std::endl;
      std::cout << "Diff: " << (compliantDdq - recoveredDdq).transpose()
                << std::endl;
      EXPECT_TRUE(equals(compliantDdq, recoveredDdq, 1e-6));
      return;
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(EXO_SOLVER_PINNED_CONTACT, ADJUST_TO_RESIDUALS_AND_CLAMPS)
{
  OpenSimFile file = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim");
  std::shared_ptr<dynamics::Skeleton> realSkel = file.skeleton;
  std::shared_ptr<dynamics::Skeleton> virtualSkel = realSkel->cloneSkeleton();

  ExoSolverPinnedContact solver = ExoSolverPinnedContact(realSkel, virtualSkel);

  // Add contact points
  std::vector<std::pair<int, Eigen::Vector3s>> pins;
  pins.emplace_back(
      realSkel->getBodyNode("calcn_r")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  pins.emplace_back(
      realSkel->getBodyNode("calcn_l")->getIndexInSkeleton(),
      Eigen::Vector3s(0.0, 0, 0.0));
  solver.setContactPins(pins);

  // Add a knee exo
  solver.addMotorDof(realSkel->getDof("knee_angle_r")->getIndexInSkeleton());
  solver.addMotorDof(realSkel->getDof("knee_angle_l")->getIndexInSkeleton());

  for (int i = 0; i < 10; i++)
  {
    Eigen::VectorXs q = realSkel->getRandomPose();
    solver.setPositions(q);

    Eigen::VectorXs dq = realSkel->getRandomVelocity();
    Eigen::VectorXs ddq = realSkel->getRandomVelocity();
    Eigen::MatrixXs J = solver.getContactJacobian();
    Eigen::VectorXs contactForces = Eigen::VectorXs::Random(J.rows());
    Eigen::VectorXs compliantDdq
        = solver.getClosestRealAccelerationConsistentWithPinsAndContactForces(
            dq, ddq, contactForces);

    Eigen::VectorXs recoveredConstraints = J * compliantDdq;
    if (recoveredConstraints.norm() > 1e-6)
    {
      std::cout << "recoveredConstraints: " << recoveredConstraints.transpose()
                << std::endl;
      EXPECT_TRUE(recoveredConstraints.norm() < 1e-6);
      return;
    }

    Eigen::VectorXs tau = solver.estimateHumanTorques(
        dq, compliantDdq, contactForces, Eigen::VectorXs::Zero(2));
    Eigen::Vector6s residual = tau.head<6>();

    if (residual.norm() > 1e-6)
    {
      std::cout << "residual: " << residual.transpose() << std::endl;
      EXPECT_TRUE(residual.norm() < 1e-6);
      return;
    }

    Eigen::VectorXs recoveredDdq
        = solver.getPinnedVirtualDynamics(dq, tau).first;
    if (!equals(compliantDdq, recoveredDdq, 1e-6))
    {
      Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(compliantDdq.size(), 3);
      compare.col(0) = compliantDdq;
      compare.col(1) = recoveredDdq;
      compare.col(2) = compliantDdq - recoveredDdq;
      std::cout << "ddq - recoveredDdq - diff: " << std::endl
                << compare << std::endl;
      EXPECT_TRUE(equals(compliantDdq, recoveredDdq, 1e-6));
      return;
    }

    Eigen::VectorXs totalRealTorques
        = solver.getPinnedTotalTorques(dq, recoveredDdq, tau, contactForces)
              .first;
    Eigen::VectorXs netTorques = totalRealTorques - tau;
    if (netTorques.norm() > 1e-6)
    {
      std::cout << "netTorques: " << netTorques.transpose() << std::endl;
      EXPECT_TRUE(netTorques.norm() < 1e-6);
      return;
    }
  }
}
#endif