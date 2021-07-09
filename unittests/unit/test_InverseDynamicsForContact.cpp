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
  std::cout << "Error: " << error << std::endl;
}

//==============================================================================
TEST(INV_DYN_FOR_CONTACT, TEST_MULTI_CONTACT)
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
  std::cout << "Error: " << error << std::endl;
}