#include <chrono>
#include <iostream>
#include <thread>

#include <benchmark/benchmark.h>
#include <dart/gui/gui.hpp>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/gui/glut/TrajectoryReplayWindow.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/IKMapping.hpp"
#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/performance/PerformanceLog.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Problem.hpp"
#include "dart/trajectory/SingleShot.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/TrajectoryConstants.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace trajectory;
using namespace performance;

static void BM_Cartpole_DART_Featherstone(benchmark::State& state)
{
  SkeletonPtr cartpole = createCartpole();

  double dt = 0.001;
  for (auto _ : state)
  {
    cartpole->computeForwardDynamics();
    cartpole->integrateVelocities(dt);
    cartpole->integratePositions(dt);
  }
}
BENCHMARK(BM_Cartpole_DART_Featherstone);

static void BM_Cartpole_Simple_Featherstone(benchmark::State& state)
{
  SkeletonPtr cartpole = createCartpole();
  SimpleFeatherstone simple;
  simple.populateFromSkeleton(cartpole);

  double* pos = (double*)malloc(simple.len() * sizeof(double));
  double* vel = (double*)malloc(simple.len() * sizeof(double));
  double* force = (double*)malloc(simple.len() * sizeof(double));
  double* accel = (double*)malloc(simple.len() * sizeof(double));

  for (int i = 0; i < simple.len(); i++)
  {
    pos[i] = cartpole->getPosition(i);
    vel[i] = cartpole->getVelocity(i);
    force[i] = cartpole->getForce(i);
  }

  double dt = 0.001;
  for (auto _ : state)
  {
    simple.forwardDynamics(pos, vel, force, accel);
    for (int i = 0; i < simple.len(); i++)
    {
      pos[i] += vel[i] * dt;
      vel[i] += accel[i] * dt;
    }
  }

  free(pos);
  free(vel);
  free(force);
  free(accel);
}
BENCHMARK(BM_Cartpole_Simple_Featherstone);

static void BM_20_Joint_DART_Featherstone(benchmark::State& state)
{
  SkeletonPtr arm = createMultiarmRobot(20, 0.2);

  double dt = 0.001;
  for (auto _ : state)
  {
    arm->computeForwardDynamics();
    arm->integrateVelocities(dt);
    arm->integratePositions(dt);
  }
}
BENCHMARK(BM_20_Joint_DART_Featherstone);

static void BM_20_Joint_Simple_Featherstone(benchmark::State& state)
{
  SkeletonPtr arm = createMultiarmRobot(20, 0.2);
  SimpleFeatherstone simple;
  simple.populateFromSkeleton(arm);

  double* pos = (double*)malloc(simple.len() * sizeof(double));
  double* vel = (double*)malloc(simple.len() * sizeof(double));
  double* force = (double*)malloc(simple.len() * sizeof(double));
  double* accel = (double*)malloc(simple.len() * sizeof(double));

  for (int i = 0; i < simple.len(); i++)
  {
    pos[i] = arm->getPosition(i);
    vel[i] = arm->getVelocity(i);
    force[i] = arm->getForce(i);
  }

  double dt = 0.001;
  for (auto _ : state)
  {
    simple.forwardDynamics(pos, vel, force, accel);
    for (int i = 0; i < simple.len(); i++)
    {
      pos[i] += vel[i] * dt;
      vel[i] += accel[i] * dt;
    }
  }

  free(pos);
  free(vel);
  free(force);
  free(accel);
}
BENCHMARK(BM_20_Joint_Simple_Featherstone);

BENCHMARK_MAIN();