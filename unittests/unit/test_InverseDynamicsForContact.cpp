#include <iostream>

#include <gtest/gtest.h>

#include "dart/dart.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/utils/utils.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace utils;

#define ALL_TESTS

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_SINGLE_CONTACT)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  Eigen::VectorXs pos = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs vel = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs acc = Eigen::VectorXs::Random(skel->getNumDofs());

  skel->setPositions(pos);
  skel->setVelocities(vel);
  Skeleton::ContactInverseDynamicsResult result
      = skel->getContactInverseDynamics(acc, skel->getBodyNode("l_foot"));

  s_t error = result.sumError();
  EXPECT_LE(error, 1e-8);
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_MULTI_CONTACT_WITH_GUESS)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  Eigen::VectorXs pos = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs vel = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs acc = Eigen::VectorXs::Random(skel->getNumDofs());

  std::vector<const dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector6s> wrenchGuesses;
  nodes.push_back(skel->getBodyNode("l_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());
  nodes.push_back(skel->getBodyNode("r_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());

  skel->setPositions(pos);
  skel->setVelocities(vel);
  Skeleton::MultipleContactInverseDynamicsResult result
      = skel->getMultipleContactInverseDynamics(acc, nodes, wrenchGuesses);

  s_t error = result.sumError();
  EXPECT_LE(error, 1e-8);
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_MULTI_CONTACT_NO_GUESS)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  Eigen::VectorXs pos = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs vel = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs acc = Eigen::VectorXs::Random(skel->getNumDofs());

  std::vector<const dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector6s> wrenchGuesses;
  nodes.push_back(skel->getBodyNode("l_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());
  nodes.push_back(skel->getBodyNode("r_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());

  skel->setPositions(pos);
  skel->setVelocities(vel);
  // This is just to be able to compare the torque values against, and be sure
  // we're minimizing them
  Skeleton::MultipleContactInverseDynamicsResult resultWithGuess
      = skel->getMultipleContactInverseDynamics(acc, nodes, wrenchGuesses);

  Skeleton::MultipleContactInverseDynamicsResult resultNoGuess
      = skel->getMultipleContactInverseDynamics(
          acc, nodes, std::vector<Eigen::Vector6s>());

  s_t error = resultNoGuess.sumError();
  EXPECT_LE(error, 1e-8);

  for (int i = 0; i < nodes.size(); i++)
  {
    s_t torqueNormWithGuess
        = resultWithGuess.contactWrenches[i].head<3>().norm();
    s_t torqueNormNoGuess = resultNoGuess.contactWrenches[i].head<3>().norm();
    // The torque norm with no guess should be pretty much the minimum
    EXPECT_TRUE(torqueNormNoGuess <= torqueNormWithGuess);
    /*
    std::cout << "result centered at 0: " << resultWithGuess.contactWrenches[i]
              << std::endl;
    std::cout << "result with no guess: " << resultNoGuess.contactWrenches[i]
              << std::endl;
    */
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_MULTI_CONTACT_MULTI_TIMESTEP)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  int numTimesteps = 5;

  Eigen::MatrixXs pos
      = Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps);
  for (int i = 1; i < pos.cols(); i++)
  {
    pos.col(i)
        = pos.col(i - 1) + Eigen::VectorXs::Random(skel->getNumDofs()) * 0.001;
  }

  std::vector<const dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector6s> wrenchGuesses;
  nodes.push_back(skel->getBodyNode("l_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());
  nodes.push_back(skel->getBodyNode("r_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());

  // This is just to be able to compare the torque values against, and be sure
  // we're minimizing them
  Skeleton::MultipleContactInverseDynamicsOverTimeResult resultOverTime
      = skel->getMultipleContactInverseDynamicsOverTime(pos, nodes, 1.0, 1.0);

  s_t error = resultOverTime.sumError();
  EXPECT_LE(error, 1e-8);

  // This is just to be able to compare the torque values against, and be sure
  // we're minimizing them
  Skeleton::MultipleContactInverseDynamicsOverTimeResult
      resultOverTimeMoreSmoothing
      = skel->getMultipleContactInverseDynamicsOverTime(pos, nodes, 10.0, 1.0);

  std::cout << "Smoothness @ 1: " << resultOverTime.computeSmoothnessLoss()
            << std::endl;
  std::cout << "Smoothness @ 10: "
            << resultOverTimeMoreSmoothing.computeSmoothnessLoss() << std::endl;

  EXPECT_LE(
      resultOverTimeMoreSmoothing.computeSmoothnessLoss(),
      resultOverTime.computeSmoothnessLoss());
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_COP_PROJECTION)
{
  Eigen::Vector6s worldWrench = Eigen::Vector6s::Random();
  s_t groundHeight = 0.1;
  int axis = 1;
  Eigen::Vector9s proj = projectWrenchToCoP(worldWrench, groundHeight, axis);

  Eigen::Vector3s cop = proj.segment<3>(0);
  Eigen::Vector6s copWrench = proj.segment<6>(3);

  // We expect that the tau is some multiple of f
  Eigen::Vector3s tau = copWrench.head<3>();
  Eigen::Vector3s f = copWrench.tail<3>();

  if (f.isZero(1e-10))
  {
    if (!tau.isZero(1e-10))
    {
      std::cout << "Expected f and tau to point in the same direction, but got "
                   "a zero f and non-zero tau!"
                << std::endl;
      std::cout << "f:" << std::endl << f << std::endl;
      std::cout << "tau:" << std::endl << tau << std::endl;
      EXPECT_TRUE(tau.isZero(1e-10));
      return;
    }
  }
  else
  {
    Eigen::Vector3s div = tau.cwiseQuotient(f);
    Eigen::Vector3s expected = Eigen::Vector3s::Ones() * div.mean();
    if (!equals(div, expected, 1e-8))
    {
      std::cout
          << "Expected tau to be a multiple of f, but it doesn't seem to be."
          << std::endl;
      std::cout << "tau/f: " << std::endl << div << std::endl;
      std::cout << "diff from mean: " << std::endl
                << (div - expected) << std::endl;
      EXPECT_TRUE(equals(div, expected, 1e-8));
      return;
    }
  }

  Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
  T.translation() = cop;
  Eigen::Vector6s recoveredCopWrench = math::dAdT(T, worldWrench);
  if (!equals(recoveredCopWrench, copWrench) > 1e-8)
  {
    std::cout << "Resulting wreches didn't match expectations!" << std::endl;
    std::cout << "Expected: " << std::endl << copWrench << std::endl;
    std::cout << "Recovered from CoP and world wrench: " << std::endl
              << recoveredCopWrench << std::endl;
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_COP_PROJECTION_JACOBIANS)
{
  Eigen::Vector6s worldWrench = Eigen::Vector6s::Random();
  s_t groundHeight = 0.1;
  int axis = 1;
  Eigen::Matrix<s_t, 9, 6> analytical
      = getProjectWrenchToCoPJacobian(worldWrench, groundHeight, axis);
  Eigen::Matrix<s_t, 9, 6> fd = finiteDifferenceProjectWrenchToCoPJacobian(
      worldWrench, groundHeight, axis);

  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Analytical: " << std::endl << analytical << std::endl;
    std::cout << "FD: " << std::endl << fd << std::endl;
    std::cout << "Diff: " << std::endl << fd - analytical << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_MULTI_CONTACT_COP)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  Eigen::VectorXs pos = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs vel = Eigen::VectorXs::Random(skel->getNumDofs());
  Eigen::VectorXs acc = Eigen::VectorXs::Random(skel->getNumDofs());

  std::vector<const dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector9s> copWrenches;
  std::vector<Eigen::Vector6s> bodyWrenches;
  nodes.push_back(skel->getBodyNode("l_foot"));
  copWrenches.push_back(Eigen::Vector9s::Random());
  bodyWrenches.push_back(Eigen::Vector6s::Random());
  nodes.push_back(skel->getBodyNode("r_foot"));
  copWrenches.push_back(Eigen::Vector9s::Random());
  bodyWrenches.push_back(Eigen::Vector6s::Random());

  skel->setPositions(pos);
  skel->setVelocities(vel);

  ///////
  // Check internal computations in the problem
  Skeleton::MultipleContactCoPProblem problem
      = skel->createMultipleContactInverseDynamicsNearCoPProblem(
          acc, nodes, copWrenches, 0.1, 1);

  // Check initial guess
  Eigen::VectorXs x = problem.getInitialGuess();
  s_t initialViolation = problem.getConstraintErrors(x).norm();
  if (initialViolation > 1e-10)
  {
    std::cout << "Initial guess violated constraints!" << std::endl;
    std::cout << "Constraint value: " << std::endl
              << problem.getConstraintErrors(x) << std::endl;
    EXPECT_TRUE(initialViolation <= 1e-10);
    return;
  }

  // Check gradients
  Eigen::VectorXs analytical = problem.getUnconstrainedGradient(x);
  Eigen::VectorXs fd = problem.finiteDifferenceUnconstrainedGradient(x);
  if (!equals(analytical, fd, 1e-8))
  {
    std::cout << "Unconstrained gradient != finite differenced version"
              << std::endl;
    Eigen::MatrixXs compare = Eigen::MatrixXs::Zero(analytical.size(), 3);
    compare.col(0) = analytical;
    compare.col(1) = fd;
    compare.col(2) = analytical - fd;
    std::cout << "analytical - fd - diff:" << std::endl << compare << std::endl;
    EXPECT_TRUE(equals(analytical, fd, 1e-8));
    return;
  }

  // Check projection onto the null space
  Eigen::VectorXs projected = problem.projectToNullSpace(analytical);
  std::cout << "analytical grad: " << std::endl << analytical << std::endl;
  std::cout << "null(analytical grad): " << std::endl << projected << std::endl;
  for (int i = -10; i < 10; i++)
  {
    Eigen::VectorXs newX = x + projected * i;
    s_t projectedViolation = problem.getConstraintErrors(newX).norm();
    if (projectedViolation > 1e-10)
    {
      std::cout << "x + null(dx) violated constraints!" << std::endl;
      std::cout << "i: " << i << std::endl;
      std::cout << "null(dx): " << std::endl << projected << std::endl;
      std::cout << "Constraint value: " << std::endl
                << problem.getConstraintErrors(newX) << std::endl;
      EXPECT_TRUE(projectedViolation <= 1e-10);
      return;
    }
  }

  Skeleton::MultipleContactInverseDynamicsResult result
      = skel->getMultipleContactInverseDynamicsNearCoP(
          acc, nodes, bodyWrenches, 0.1, 1);

  s_t error = result.sumError();
  EXPECT_TRUE(error < 1e-8);
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, TEST_MULTI_CONTACT_MULTI_TIMESTEP_PREV_FORCE)
{
  // set precision to 256 bits (double has only 53 bits)
#ifdef DART_USE_ARBITRARY_PRECISION
  mpfr::mpreal::set_default_prec(256);
#endif

  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  int numTimesteps = 5;

  Eigen::MatrixXs pos
      = Eigen::MatrixXs::Random(skel->getNumDofs(), numTimesteps);
  for (int i = 1; i < pos.cols(); i++)
  {
    pos.col(i)
        = pos.col(i - 1) + Eigen::VectorXs::Random(skel->getNumDofs()) * 0.001;
  }

  Eigen::VectorXs firstVel
      = skel->getPositionDifferences(pos.col(1), pos.col(0))
        / skel->getTimeStep();
  Eigen::VectorXs secondVel
      = skel->getPositionDifferences(pos.col(2), pos.col(1))
        / skel->getTimeStep();
  Eigen::VectorXs acc = (secondVel - firstVel) / skel->getTimeStep();
  skel->setPositions(pos.col(0));
  skel->setVelocities(firstVel);
  auto singleFootResult
      = skel->getContactInverseDynamics(acc, skel->getBodyNode("r_foot"));
  EXPECT_LE(singleFootResult.sumError(), 1e-8);

  std::vector<const dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector6s> prevTimestepWrenches;
  nodes.push_back(skel->getBodyNode("l_foot"));
  prevTimestepWrenches.push_back(Eigen::Vector6s::Zero());
  nodes.push_back(skel->getBodyNode("r_foot"));
  prevTimestepWrenches.push_back(singleFootResult.contactWrench);

  // This is just to be able to compare the torque values against, and be sure
  // we're minimizing them
  Skeleton::MultipleContactInverseDynamicsOverTimeResult resultOverTime
      = skel->getMultipleContactInverseDynamicsOverTime(
          pos,
          nodes,
          0.001,
          0.0,
          [](s_t) { return 0.0; },
          prevTimestepWrenches,
          100.0);

  s_t error = resultOverTime.sumError();
  EXPECT_LE(error, 1e-8);

  // This is satisfiable, so if it's the main goal by a factor of 100000, we
  // should hit it pretty close
  EXPECT_LE(resultOverTime.computePrevForceLoss(), 1e-4);

  /*
  // This is just to be able to compare the torque values against, and be sure
  // we're minimizing them
  Skeleton::MultipleContactInverseDynamicsOverTimeResult
      resultOverTimeMoreSmoothing
      = skel->getMultipleContactInverseDynamicsOverTime(
          pos,
          nodes,
          0.0,
          0.0,
          [](s_t) { return 0.0; },
          prevTimestepWrenches,
          100.0);
  std::cout << "Prev force loss @ 1: " << resultOverTime.computePrevForceLoss()
            << std::endl;
  std::cout << "Prev force loss @ 10: "
            << resultOverTimeMoreSmoothing.computePrevForceLoss() << std::endl;
  EXPECT_LE(
      resultOverTimeMoreSmoothing.computePrevForceLoss(),
      resultOverTime.computePrevForceLoss());
  */
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(INV_DYN_FOR_CONTACT, EXPLORE_RECOVER_CENTER_OF_PRESSURE)
{
  Eigen::Vector3s f = Eigen::Vector3s(1.0, 0.0, 1.0);
  Eigen::Vector3s tau = Eigen::Vector3s(1.0, 1.0, 0.0);

  // We want the component of the torque that's parallel to the force
  Eigen::Vector3s residual = f.dot(tau) * (f / f.squaredNorm());

  // we want to find r, such that r cross f = tau
  Eigen::Matrix3s skew = math::makeSkewSymmetric(f);
  Eigen::Matrix3s skewInverse = skew.inverse();
  Eigen::Vector3s r = -skewInverse * (tau - residual);
  Eigen::Vector3s recoveredTau = r.cross(f) + residual;
  Eigen::Vector3s diff = tau - recoveredTau;
  s_t error = diff.norm();

  std::cout << "error: " << error << std::endl;

  Eigen::Vector3s r2
      = -skew.completeOrthogonalDecomposition().solve(tau - residual);
  Eigen::Vector3s recoveredTau2 = r2.cross(f) + residual;
  Eigen::Vector3s diff2 = tau - recoveredTau2;
  s_t error2 = diff2.norm();

  std::cout << "error 2: " << error2 << std::endl;

  // https://math.stackexchange.com/questions/950729/solving-for-first-term-in-vector-product
  Eigen::Vector3s r3 = f.cross(tau) / (f.norm() * f.norm());
  Eigen::Vector3s recoveredTau3 = r3.cross(f) + residual;
  Eigen::Vector3s diff3 = tau - recoveredTau3;
  s_t error3 = diff3.norm();

  std::cout << "error 3: " << error3 << std::endl;
}
#endif