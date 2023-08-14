#include <gtest/gtest.h>

#include "dart/math/GraphFlowDiscretizer.hpp"

#define ALL_TESTS

using namespace dart;
using namespace math;

#ifdef ALL_TESTS
TEST(IKLimits, SIMPLE_CLEANING)
{
  std::vector<std::pair<int, int>> arcs;
  arcs.emplace_back(0, 1);
  std::vector<bool> attachedToSink;
  attachedToSink.push_back(true);
  attachedToSink.push_back(false);
  GraphFlowDiscretizer discretizer(2, arcs, attachedToSink);

  Eigen::MatrixXs energyLevels = Eigen::MatrixXs::Zero(2, 3);
  Eigen::MatrixXs arcRates = Eigen::MatrixXs::Zero(1, 3);

  for (int t = 0; t < 3; t++)
  {
    energyLevels(1, t) = t;
  }

  Eigen::MatrixXs cleaned = discretizer.cleanUpArcRates(energyLevels, arcRates);

  EXPECT_EQ(cleaned(0, 0), 1);
  EXPECT_EQ(cleaned(0, 1), 1);
  EXPECT_EQ(cleaned(0, 2), 0);
}
#endif

#ifdef ALL_TESTS
TEST(IKLimits, JUST_CREATION)
{
  std::vector<std::pair<int, int>> arcs;
  std::vector<bool> attachedToSink;
  attachedToSink.push_back(true);
  GraphFlowDiscretizer discretizer(1, arcs, attachedToSink);

  int numTimesteps = 3;

  Eigen::MatrixXs energyLevels = Eigen::MatrixXs::Zero(1, numTimesteps);
  Eigen::MatrixXs arcRates = Eigen::MatrixXs::Zero(0, numTimesteps);

  for (int t = 0; t < numTimesteps; t++)
  {
    energyLevels(0, t) = t + 1;
  }

  std::vector<ParticlePath> particles
      = discretizer.discretize(numTimesteps, energyLevels, arcRates);

  // We expect one particle to be created per timestep
  EXPECT_EQ(particles.size(), numTimesteps);

  for (int t = 0; t < numTimesteps; t++)
  {
    EXPECT_EQ(particles[t].startTime, t);
    EXPECT_EQ(particles[t].nodeHistory.size(), numTimesteps - t);
  }
}
#endif

#ifdef ALL_TESTS
TEST(IKLimits, SIMPLE_TRANSFER)
{
  std::vector<std::pair<int, int>> arcs;
  arcs.emplace_back(0, 1);
  std::vector<bool> attachedToSink;
  attachedToSink.push_back(false);
  attachedToSink.push_back(false);
  GraphFlowDiscretizer discretizer(2, arcs, attachedToSink);

  int numTimesteps = 5;
  Eigen::MatrixXs energyLevels = Eigen::MatrixXs::Zero(2, numTimesteps);
  Eigen::MatrixXs arcRates = Eigen::MatrixXs::Zero(1, numTimesteps);

  for (int t = 0; t < numTimesteps; t++)
  {
    energyLevels(0, t) = numTimesteps - t;
    energyLevels(1, t) = t;
    arcRates(0, t) = 1;
  }

  std::vector<ParticlePath> particles
      = discretizer.discretize(numTimesteps, energyLevels, arcRates);

  // We expect one particle to be created per timestep
  EXPECT_EQ(particles.size(), numTimesteps);

  for (int i = 0; i < particles.size(); i++)
  {
    EXPECT_EQ(particles[i].startTime, 0);
    EXPECT_EQ(particles[i].nodeHistory.size(), numTimesteps);
  }

  for (int t = 0; t < numTimesteps; t++)
  {
    Eigen::MatrixXs quantizedEnergy = Eigen::MatrixXs::Zero(2, numTimesteps);
    for (int i = 0; i < particles.size(); i++)
    {
      quantizedEnergy(particles[i].nodeHistory[t]) += 1;
    }
    EXPECT_EQ(quantizedEnergy(0), energyLevels(0, t));
    EXPECT_EQ(quantizedEnergy(1), energyLevels(1, t));
  }
}
#endif

#ifdef ALL_TESTS
TEST(IKLimits, CREATION_AND_TRANSFER)
{
  std::vector<std::pair<int, int>> arcs;
  arcs.emplace_back(0, 1);
  std::vector<bool> attachedToSink;
  attachedToSink.push_back(true);
  attachedToSink.push_back(false);
  GraphFlowDiscretizer discretizer(2, arcs, attachedToSink);

  int numTimesteps = 5;
  Eigen::MatrixXs energyLevels = Eigen::MatrixXs::Zero(2, numTimesteps);
  Eigen::MatrixXs arcRates = Eigen::MatrixXs::Zero(1, numTimesteps);

  for (int t = 0; t < numTimesteps; t++)
  {
    energyLevels(1, t) = t;
    arcRates(0, t) = 1;
  }

  // TODO: Figure out exactly what behavior is desirable here.
  // Do we want to create a particle at the beginning of the first timestep at
  // the root node, or wait for that to be "pulled" into existence by the first
  // timestep's flow, and then the particle begins existing on the next
  // timestep?

  std::vector<ParticlePath> particles
      = discretizer.discretize(numTimesteps - 1, energyLevels, arcRates);

  // We expect one particle to be created per timestep, TODO: except the first
  // timestep, because there's no energy in the 0 node at that timestep?
  EXPECT_EQ(particles.size(), numTimesteps - 1);

  /*
  for (int i = 0; i < particles.size(); i++)
  {
    // We expect each particle to be created in the 0 node, and then move to the
    // 1 node, and remain there for the duration
    EXPECT_EQ(particles[i].startTime, i);

    // We expect each particle to be created in the 0 node, and then move to the
    // 1 node, and remain there for the duration
    EXPECT_EQ(particles[i].nodeHistory.size(), numTimesteps - 1 - i);
    EXPECT_EQ(particles[i].nodeHistory[0], 0);
    for (int j = 1; j < particles[i].nodeHistory.size(); j++)
    {
      EXPECT_EQ(particles[i].nodeHistory[j], 1);
    }
  }
  */
}
#endif