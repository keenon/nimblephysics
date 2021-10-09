#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include <dart/utils/urdf/urdf.hpp>
#include <dart/utils/utils.hpp>
#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/constraint/BoxedLcpConstraintSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/IKMapping.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/IPOptOptimizer.hpp"
#include "dart/trajectory/LossFn.hpp"
#include "dart/trajectory/MultiShot.hpp"
#include "dart/trajectory/Solution.hpp"
#include "dart/trajectory/TrajectoryRollout.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

// #define TEST1
#define TEST2
// #define TEST3
// #define FULL_TEST

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace server;
using namespace realtime;

std::shared_ptr<simulation::World> createWorld()
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();

  std::shared_ptr<dynamics::Skeleton> cartpole
      = dart::utils::UniversalLoader::loadSkeleton(
          world.get(), "dart://sample/urdf/cartpole.urdf");
  return world;
}

TEST(CARTPOLE, MASS)
{
  std::shared_ptr<simulation::World> world = createWorld();

  Eigen::VectorXs testVel = world->getVelocities();

  EXPECT_TRUE(verifyAnalyticalJacobians(world, true));
  EXPECT_TRUE(verifyVelGradients(world, testVel));
  EXPECT_TRUE(verifyWrtMass(world));
}