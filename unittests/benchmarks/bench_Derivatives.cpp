#include <chrono>
#include <iostream>
#include <thread>

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/performance/PerformanceLog.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/SkelParser.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace performance;

static void BM_Jacobian_Of_C_q_Numerical(benchmark::State& state)
{
  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  common::Uri uri = "dart://sample/skel/test/s_t_pendulum.skel";

  // World
  WorldPtr world = World::create();
  world = utils::SkelParser::readWorld(uri);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  while (state.KeepRunning())
  {
    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int k = 0; k < dof; ++k)
      {
        q[k] = math::Random::uniform(qLB, qUB);
        dq[k] = math::Random::uniform(dqLB, dqUB);
        ddq[k] = math::Random::uniform(ddqLB, ddqUB);
      }
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      Eigen::MatrixXs dC_numerical
          = skel->finiteDifferenceJacobianOfC(neural::WithRespectTo::POSITION);
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Jacobian_Of_C_q_Numerical);

static void BM_Jacobian_Of_C_q_Analytical(benchmark::State& state)
{
  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  common::Uri uri = "dart://sample/skel/test/s_t_pendulum.skel";

  // World
  WorldPtr world = World::create();
  world = utils::SkelParser::readWorld(uri);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  while (state.KeepRunning())
  {
    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int k = 0; k < dof; ++k)
      {
        q[k] = math::Random::uniform(qLB, qUB);
        dq[k] = math::Random::uniform(dqLB, dqUB);
        ddq[k] = math::Random::uniform(ddqLB, ddqUB);
      }
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      Eigen::MatrixXs C_dq_analytic
          = skel->getJacobianOfC(neural::WithRespectTo::POSITION);
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Jacobian_Of_C_q_Analytical);

static void BM_Jacobian_Of_C_dq_Numerical(benchmark::State& state)
{
  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  common::Uri uri = "dart://sample/skel/test/s_t_pendulum.skel";

  // World
  WorldPtr world = World::create();
  world = utils::SkelParser::readWorld(uri);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  while (state.KeepRunning())
  {
    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int k = 0; k < dof; ++k)
      {
        q[k] = math::Random::uniform(qLB, qUB);
        dq[k] = math::Random::uniform(dqLB, dqUB);
        ddq[k] = math::Random::uniform(ddqLB, ddqUB);
      }
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      Eigen::MatrixXs dC_numerical
          = skel->finiteDifferenceJacobianOfC(neural::WithRespectTo::VELOCITY);
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Jacobian_Of_C_dq_Numerical);

static void BM_Jacobian_Of_C_dq_Analytical(benchmark::State& state)
{
  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  common::Uri uri = "dart://sample/skel/test/s_t_pendulum.skel";

  // World
  WorldPtr world = World::create();
  world = utils::SkelParser::readWorld(uri);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  while (state.KeepRunning())
  {
    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int k = 0; k < dof; ++k)
      {
        q[k] = math::Random::uniform(qLB, qUB);
        dq[k] = math::Random::uniform(dqLB, dqUB);
        ddq[k] = math::Random::uniform(ddqLB, ddqUB);
      }
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      Eigen::MatrixXs C_dq_analytic
          = skel->getJacobianOfC(neural::WithRespectTo::VELOCITY);
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Jacobian_Of_C_dq_Analytical);

static void BM_Jacobian_Of_Minv_q_Numerical(benchmark::State& state)
{
  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  common::Uri uri = "dart://sample/skel/test/s_t_pendulum.skel";

  // World
  WorldPtr world = World::create();
  world = utils::SkelParser::readWorld(uri);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  while (state.KeepRunning())
  {
    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int k = 0; k < dof; ++k)
      {
        q[k] = math::Random::uniform(qLB, qUB);
        dq[k] = math::Random::uniform(dqLB, dqUB);
        ddq[k] = math::Random::uniform(ddqLB, ddqUB);
      }
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      for (auto j = 0; j < 100; ++j)
      {
        Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
        Eigen::MatrixXs dC_numerical = skel->finiteDifferenceJacobianOfMinv(
            x, neural::WithRespectTo::POSITION);
      }
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Jacobian_Of_Minv_q_Numerical);

static void BM_Jacobian_Of_Minv_q_Analytical(benchmark::State& state)
{
  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  common::Uri uri = "dart://sample/skel/test/s_t_pendulum.skel";

  // World
  WorldPtr world = World::create();
  world = utils::SkelParser::readWorld(uri);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  while (state.KeepRunning())
  {
    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int k = 0; k < dof; ++k)
      {
        q[k] = math::Random::uniform(qLB, qUB);
        dq[k] = math::Random::uniform(dqLB, dqUB);
        ddq[k] = math::Random::uniform(ddqLB, ddqUB);
      }
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      for (auto j = 0; j < 100; ++j)
      {
        Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
        Eigen::MatrixXs C_dq_analytic = skel->getJacobianOfMinv_Direct(
            x, neural::WithRespectTo::POSITION);
      }
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Jacobian_Of_Minv_q_Analytical);

static void BM_Jacobian_Of_Minv_q_Analytical_ID(benchmark::State& state)
{
  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  common::Uri uri = "dart://sample/skel/test/s_t_pendulum.skel";

  // World
  WorldPtr world = World::create();
  world = utils::SkelParser::readWorld(uri);
  world->setGravity(Eigen::Vector3s(0, -9.81, 0));

  while (state.KeepRunning())
  {
    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int k = 0; k < dof; ++k)
      {
        q[k] = math::Random::uniform(qLB, qUB);
        dq[k] = math::Random::uniform(dqLB, dqUB);
        ddq[k] = math::Random::uniform(ddqLB, ddqUB);
      }
      skel->setPositions(q);
      skel->setVelocities(dq);
      skel->setAccelerations(ddq);

      for (auto j = 0; j < 100; ++j)
      {
        Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
        Eigen::MatrixXs C_dq_analytic
            = skel->getJacobianOfMinv_ID(x, neural::WithRespectTo::POSITION);
      }
    }
  }
}
// Register the function as a benchmark
BENCHMARK(BM_Jacobian_Of_Minv_q_Analytical_ID);

BENCHMARK_MAIN();
