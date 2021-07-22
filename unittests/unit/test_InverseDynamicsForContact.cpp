#include <iostream>

#include <gtest/gtest.h>

#include "dart/dart.hpp"
#include "dart/utils/utils.hpp"

#include "TestHelpers.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace utils;

//==============================================================================
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
  Eigen::VectorXs nextVel
      = vel
        + Eigen::VectorXs::Random(skel->getNumDofs()) * world->getTimeStep();

  skel->setPositions(pos);
  skel->setVelocities(vel);
  Skeleton::ContactInverseDynamicsResult result
      = skel->getContactInverseDynamics(nextVel, skel->getBodyNode("l_foot"));

  s_t error = result.sumError();
  EXPECT_TRUE(error < 1e-11);
}

//==============================================================================
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
  Eigen::VectorXs nextVel
      = vel
        + Eigen::VectorXs::Random(skel->getNumDofs()) * world->getTimeStep();

  std::vector<dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector6s> wrenchGuesses;
  nodes.push_back(skel->getBodyNode("l_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());
  nodes.push_back(skel->getBodyNode("r_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Zero());

  skel->setPositions(pos);
  skel->setVelocities(vel);
  Skeleton::MultipleContactInverseDynamicsResult result
      = skel->getMultipleContactInverseDynamics(nextVel, nodes, wrenchGuesses);

  s_t error = result.sumError();
  EXPECT_TRUE(error < 1e-11);
}

//==============================================================================
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
  Eigen::VectorXs nextVel
      = vel
        + Eigen::VectorXs::Random(skel->getNumDofs()) * world->getTimeStep();

  std::vector<dynamics::BodyNode*> nodes;
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
      = skel->getMultipleContactInverseDynamics(nextVel, nodes, wrenchGuesses);

  Skeleton::MultipleContactInverseDynamicsResult resultNoGuess
      = skel->getMultipleContactInverseDynamics(
          nextVel, nodes, std::vector<Eigen::Vector6s>());

  s_t error = resultNoGuess.sumError();
  EXPECT_TRUE(error < 1e-11);

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

//==============================================================================
<<<<<<< HEAD
=======
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

  std::vector<dynamics::BodyNode*> nodes;
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
  EXPECT_LE(error, 1e-11);

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

//==============================================================================
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

  std::vector<dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector6s> wrenchGuesses;
  nodes.push_back(skel->getBodyNode("l_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Random());
  nodes.push_back(skel->getBodyNode("r_foot"));
  wrenchGuesses.push_back(Eigen::Vector6s::Random());

  // This is just to be able to compare the torque values against, and be sure
  // we're minimizing them
  Skeleton::MultipleContactInverseDynamicsOverTimeResult resultOverTime
      = skel->getMultipleContactInverseDynamicsOverTime(
          pos, nodes, 1.0, 1.0, wrenchGuesses, 1.0);

  s_t error = resultOverTime.sumError();
  EXPECT_LE(error, 1e-11);

  // This is just to be able to compare the torque values against, and be sure
  // we're minimizing them
  Skeleton::MultipleContactInverseDynamicsOverTimeResult
      resultOverTimeMoreSmoothing
      = skel->getMultipleContactInverseDynamicsOverTime(
          pos, nodes, 1.0, 1.0, wrenchGuesses, 10.0);

  std::cout << "Prev force loss @ 1: " << resultOverTime.computePrevForceLoss()
            << std::endl;
  std::cout << "Prev force loss @ 10: "
            << resultOverTimeMoreSmoothing.computePrevForceLoss() << std::endl;

  EXPECT_LE(
      resultOverTimeMoreSmoothing.computePrevForceLoss(),
      resultOverTime.computePrevForceLoss());
}

//==============================================================================
>>>>>>> 1405a09db6cc8c5503e409023431e0c3dd8c4003
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