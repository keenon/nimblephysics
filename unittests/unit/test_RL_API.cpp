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
TEST(RL_API, TEST_SIMPLE_STATE)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  int stateDim = world->getStateSize();
  int dofs = world->getNumDofs();
  EXPECT_EQ(stateDim, 2 * dofs);
  int actionDim = world->getActionSize();
  EXPECT_EQ(actionDim, dofs);
  std::vector<int> actionSpace = world->getActionSpace();
  EXPECT_EQ(actionSpace.size(), dofs);
  for (int i = 0; i < dofs; i++)
  {
    EXPECT_EQ(actionSpace[i], i);
  }
}

//==============================================================================
TEST(RL_API, ACTION_TOO_SMALL)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  Eigen::VectorXs action = Eigen::VectorXs(3);
  action << 1.0, 2.0, 3.0;
  world->setAction(action);
}

//==============================================================================
TEST(RL_API, TEST_CUSTOM_ACTION_SPACE)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  newActionSpace.push_back(7);
  newActionSpace.push_back(9);
  world->setActionSpace(newActionSpace);
  Eigen::VectorXs action = Eigen::VectorXs(3);
  action << 1.0, 2.0, 3.0;
  world->setAction(action);

  Eigen::VectorXs controls = world->getControlForces();
  for (int i = 0; i < controls.size(); i++)
  {
    if (i == 3)
    {
      EXPECT_EQ(controls(i), action(0));
    }
    else if (i == 7)
    {
      EXPECT_EQ(controls(i), action(1));
    }
    else if (i == 9)
    {
      EXPECT_EQ(controls(i), action(2));
    }
    else
    {
      EXPECT_EQ(controls(i), 0.0);
    }
  }

  Eigen::VectorXs recoveredAction = world->getAction();
  EXPECT_TRUE(equals(recoveredAction, action));
}

//==============================================================================
TEST(RL_API, TEST_CUSTOM_ACTION_SPACE_OOB)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  newActionSpace.push_back(7);
  newActionSpace.push_back(999);
  world->setActionSpace(newActionSpace);
  // We expect the above call to fail, and have no effect
  EXPECT_EQ(world->getNumDofs(), world->getActionSize());
}

//==============================================================================
TEST(RL_API, TEST_CUSTOM_ACTION_SPACE_OOB_2)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  newActionSpace.push_back(7);
  newActionSpace.push_back(-1);
  world->setActionSpace(newActionSpace);
  // We expect the above call to fail, and have no effect
  EXPECT_EQ(world->getNumDofs(), world->getActionSize());
}

//==============================================================================
TEST(RL_API, TEST_CUSTOM_ACTION_SPACE_OOB_3)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(999);
  newActionSpace.push_back(-1);
  world->setActionSpace(newActionSpace);
  // We expect the above call to fail, and have no effect
  EXPECT_EQ(world->getNumDofs(), world->getActionSize());
}

//==============================================================================
TEST(RL_API, TEST_CUSTOM_ACTION_SPACE_DUPLICATE)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  newActionSpace.push_back(7);
  newActionSpace.push_back(3);
  world->setActionSpace(newActionSpace);
  // We expect the above call to fail, and have no effect
  EXPECT_EQ(world->getNumDofs(), world->getActionSize());
}

//==============================================================================
TEST(RL_API, TEST_STATE_JAC)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  Eigen::MatrixXs stateJac = world->getStateJacobian();
  Eigen::MatrixXs stateJacFd = world->finiteDifferenceStateJacobian();
  // Don't use too tight a tolerance, since we didn't bother to implement
  // Ridders here
  EXPECT_TRUE(equals(stateJac, stateJacFd, 1e-7));
}

//==============================================================================
TEST(RL_API, TEST_SIMPLE_ACTION_JAC)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  Eigen::MatrixXs actionJac = world->getStateJacobian();
  Eigen::MatrixXs actionJacFd = world->finiteDifferenceStateJacobian();
  // Don't use too tight a tolerance, since we didn't bother to implement
  // Ridders here
  EXPECT_TRUE(equals(actionJac, actionJacFd, 1e-7));
}

//==============================================================================
TEST(RL_API, TEST_CUSTOM_ACTION_JAC)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  newActionSpace.push_back(7);
  newActionSpace.push_back(9);
  world->setActionSpace(newActionSpace);

  Eigen::MatrixXs actionJac = world->getStateJacobian();
  Eigen::MatrixXs actionJacFd = world->finiteDifferenceStateJacobian();
  // Don't use too tight a tolerance, since we didn't bother to implement
  // Ridders here
  EXPECT_TRUE(equals(actionJac, actionJacFd, 1e-7));
}

//==============================================================================
TEST(RL_API, TEST_ADD_CUSTOM_ACTION)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  world->setActionSpace(newActionSpace);
  world->addDofToActionSpace(5);

  std::vector<int> recoveredActionSpace = world->getActionSpace();
  EXPECT_EQ(2, recoveredActionSpace.size());
  EXPECT_EQ(3, recoveredActionSpace[0]);
  EXPECT_EQ(5, recoveredActionSpace[1]);
}

//==============================================================================
TEST(RL_API, TEST_ADD_CUSTOM_ACTION_DUPLICATE)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  world->setActionSpace(newActionSpace);
  world->addDofToActionSpace(3);

  std::vector<int> recoveredActionSpace = world->getActionSpace();
  EXPECT_EQ(1, recoveredActionSpace.size());
  EXPECT_EQ(3, recoveredActionSpace[0]);
}

//==============================================================================
TEST(RL_API, TEST_REMOVE_CUSTOM_ACTION)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(3);
  newActionSpace.push_back(5);
  world->setActionSpace(newActionSpace);
  world->removeDofFromActionSpace(3);

  std::vector<int> recoveredActionSpace = world->getActionSpace();
  EXPECT_EQ(1, recoveredActionSpace.size());
  EXPECT_EQ(5, recoveredActionSpace[0]);
}

//==============================================================================
TEST(RL_API, TEST_REMOVE_NON_EXISTENT_CUSTOM_ACTION)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::vector<int> newActionSpace;
  newActionSpace.push_back(5);
  world->setActionSpace(newActionSpace);
  world->removeDofFromActionSpace(3);

  std::vector<int> recoveredActionSpace = world->getActionSpace();
  EXPECT_EQ(1, recoveredActionSpace.size());
  EXPECT_EQ(5, recoveredActionSpace[0]);
}