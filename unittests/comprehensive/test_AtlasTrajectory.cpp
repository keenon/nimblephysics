/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

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
#include "dart/collision/fcl/FCLCollisionDetector.hpp"
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

#define ALL_TESTS
// #define TEST1
// #define TEST2

#ifdef ALL_TESTS
#define TEST1
#define TEST2
#endif

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace server;
using namespace realtime;

std::shared_ptr<simulation::World> createWorld(
    double target_x, double target_y, double target_z)
{
  // Create a world
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->setSlowDebugResultsAgainstFD(true);

  // Set gravity of the world
  // world->setConstraintForceMixingEnabled(true);
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));

  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::UniversalLoader::loadSkeleton(
          world.get(), "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  std::shared_ptr<dynamics::Skeleton> ground
      = dart::utils::UniversalLoader::loadSkeleton(
          world.get(), "dart://sample/sdf/atlas/ground.urdf");
  ground->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
      false);

  // Set initial configuration for Atlas robot
  atlas->setPosition(0, -0.5 * dart::math::constantsd::pi());
  atlas->setPosition(4, -0.01);

  Eigen::VectorXd forceLimits = Eigen::VectorXd::Ones(atlas->getNumDofs()) * 50;
  forceLimits.segment<6>(0).setZero();
  atlas->setForceUpperLimits(forceLimits);
  atlas->setForceLowerLimits(forceLimits * -1);
  Eigen::VectorXd posLimits = Eigen::VectorXd::Ones(atlas->getNumDofs()) * 10;
  atlas->setPositionUpperLimits(posLimits);
  atlas->setPositionLowerLimits(posLimits * -1);
  Eigen::VectorXd velLimits = Eigen::VectorXd::Ones(atlas->getNumDofs()) * 20;
  atlas->setVelocityUpperLimits(velLimits);
  atlas->setVelocityLowerLimits(velLimits * -1);

  // Create target

  SkeletonPtr target = Skeleton::create("target");
  std::pair<WeldJoint*, BodyNode*> targetJointPair
      = target->createJointAndBodyNodePair<WeldJoint>(nullptr);
  WeldJoint* targetJoint = targetJointPair.first;
  BodyNode* targetBody = targetJointPair.second;
  Eigen::Isometry3d targetOffset = Eigen::Isometry3d::Identity();
  targetOffset.translation() = Eigen::Vector3d(target_x, target_y, target_z);
  targetJoint->setTransformFromParentBodyNode(targetOffset);
  std::shared_ptr<BoxShape> targetShape(
      new BoxShape(Eigen::Vector3d(0.1, 0.1, 0.1)));
  ShapeNode* targetVisual
      = targetBody->createShapeNodeWith<VisualAspect>(targetShape);
  targetVisual->getVisualAspect()->setColor(Eigen::Vector3d(0.8, 0.5, 0.5));
  targetVisual->getVisualAspect()->setCastShadows(false);

  world->addSkeleton(target);

  return world;
}

#ifdef TEST1
TEST(ATLAS, BROKEN_1)
{
  double target_x = 0.5;
  double target_y = 1.0;
  double target_z = -1.0;
  std::shared_ptr<simulation::World> world
      = createWorld(target_x, target_y, target_z);

  Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(33);
  brokenPos << -1.571, -0.00288326, 0.00165361, 0.000259454, -0.0102512,
      1.84494e-05, 0.000848932, 0.00334193, 0.00028172, -0.000263459,
      0.000695572, 0.0150866, -0.00288076, -0.0202479, 0.000544379, 0.000227742,
      0.0050451, -0.00294216, 0.000669709, 0.000114237, 0.0646027, -2.75759e-05,
      -7.91484e-05, -0.0106369, 2.12794e-05, 0.0116945, 0.000591883,
      0.000231497, 0.00423308, -0.00163132, -0.000462828, 3.65345e-05,
      0.0220306;
  Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(33);
  brokenVel << -0.0355665, -0.712732, -0.0197312, 0.0644581, -0.00523274,
      -0.0570926, 0.0472622, 0.825493, 0.0338969, -0.0882721, 0.183699, 0.17503,
      -0.649341, 0.0965665, 0.0195836, 0.0421524, 1.19344, -0.665203, 0.184475,
      -0.00623832, 10.9633, 0.0267205, 0.106822, -4.22165, -0.204534, 5.39066,
      0.0195849, 0.0421863, 1.07122, -0.443072, 0.0845663, -0.00623839, 2.65921;
  Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(33);
  brokenForce << 0, 0, 0, 0, 0, 0, -7.44122, -1.70693, -5.2703, -2.6099,
      -5.7076, 0.985185, -6.3785, 0.372082, -9.87066, 7.78529, 0.081559,
      3.65908, 1.93437, -4.29761, -6.52332, -4.70401, 2.88616, -0.000431554,
      -3.0544, -4.1798, 2.00762, 7.96103, -8.62058, -5.69036, 7.01415, 4.63665,
      3.18185;
  Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero(96);
  brokenLCPCache << 0.0891057, -0.00904432, -0.0351203, 0.138667, -0.0574322,
      -0.0753322, 0.00241692, 0.00236652, -0.00056386, 0.000854223, 0.000815419,
      0.000623825, 0, 0, 0, 0.0304843, 0.0304843, 0.0304843, 0.0163738,
      0.0163738, 0.0163738, 0.0703085, 0.00465171, 0.0703085, 0.00146164,
      0.00146164, 0.00146164, 0.000207407, 0.000207407, 0.000207407,
      0.000385974, 0.000385974, 0.000385974, 0.000115742, 0.000115742,
      0.000115742, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.17541e-05, 4.17541e-05,
      4.17541e-05, 0.0793674, 0.0237075, -0.0513795, 0.00384577, 0.000954421,
      -0.00198845, 0.00242838, 3.52905e-05, 8.63474e-05, 0.000481381,
      0.000481381, -0.000204222, 0.0805577, -0.0760965, 0.0637852, 0.066307,
      -0.0269539, -0.0107056, 0.0453631, 0.0425435, -0.0198708, 0.03682,
      0.000316104, 0.0191842, 0.00535146, 0.00535146, 0.000228775, 0.00825139,
      0.00272869, -0.00210738, 0.000103567, 0.000103567, 0.000103567, 0, 0, 0,
      0.0169716, -0.0113465, -0.00288131, 0.0058814, 0.00459289, 0.00055222,
      0.0253774, -0.0160258, 0.0132533, 0.00691794, 0.00652204, 0.000479448;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setExternalForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyVelGradients(world, brokenVel));
  // EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  // EXPECT_TRUE(verifyF_c(world));

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  Eigen::VectorXd animatePos = brokenPos;
  int i = 0;
  Ticker ticker(0.01);
  ticker.registerTickListener([&](long time) {
    world->setPositions(animatePos);
    animatePos += brokenVel * 0.001;

    i++;
    if (i >= 100)
    {
      animatePos = brokenPos;
      i = 0;
    }
    // world->step();
    server.renderWorld(world);
  });

  server.registerConnectionListener([&]() { ticker.start(); });
  while (server.isServing())
  {
    // spin
  }

  std::shared_ptr<neural::BackpropSnapshot> snapshot
      = neural::forwardPass(world, true);
  EXPECT_TRUE(snapshot->areResultsStandardized());
}
#endif // TEST1

#ifdef TEST2
TEST(ATLAS, BROKEN_2)
{
  double target_x = 0.5;
  double target_y = 1.0;
  double target_z = -1.0;
  std::shared_ptr<simulation::World> world
      = createWorld(target_x, target_y, target_z);

  Eigen::VectorXd brokenPos = Eigen::VectorXd::Zero(33);
  brokenPos << -1.57102, -0.00298104, 0.00176851, 0.00027332, -0.0102446,
      2.25516e-05, 0.000838035, 0.00349415, 0.000302042, -0.000268057,
      0.000681061, 0.0150735, -0.00286418, -0.0202515, 0.000771647, 0.000250073,
      0.00527326, -0.00307521, 0.000825464, -2.66419e-05, 0.0646006,
      -3.23823e-05, -6.50984e-05, -0.0106499, 4.97962e-06, 0.011691,
      0.000771647, 0.000250029, 0.00449108, -0.0019256, 0.00045803,
      -2.65981e-05, 0.0220327;
  Eigen::VectorXd brokenVel = Eigen::VectorXd::Zero(33);
  brokenVel << -0.0355415, -0.712449, -0.0197237, 0.0644239, -0.00522888,
      -0.0571042, 0.0472592, 0.825169, 0.0338612, -0.0882515, 0.183756,
      0.175005, -0.649385, 0.0966408, 0.0195587, 0.042304, 1.19296, -0.664948,
      0.184412, -0.00625319, 10.9633, 0.0267056, 0.106785, -4.22162, -0.204477,
      5.3907, 0.0195588, 0.0422993, 1.07078, -0.442892, 0.0845375, -0.0062484,
      2.65919;
  Eigen::VectorXd brokenForce = Eigen::VectorXd::Zero(33);
  brokenForce << 0, 0, 0, 0, 0, 0, -7.44122, -1.70693, -5.2703, -2.6099,
      -5.7076, 0.985185, -6.3785, 0.372082, -9.87066, 7.78529, 0.081559,
      3.65908, 1.93437, -4.29761, -6.52332, -4.70401, 2.88616, -0.000431554,
      -3.0544, -4.1798, 2.00762, 7.96103, -8.62058, -5.69036, 7.01415, 4.63665,
      3.18185;
  Eigen::VectorXd brokenLCPCache = Eigen::VectorXd::Zero(96);
  brokenLCPCache << 0.0644031, 0.0150676, -0.0296244, 0.158559, -0.00653536,
      0.0099797, 0, 0, 0, 0, 0, 0, 0.00463209, -0.00454052, 0.000261391,
      0.0248994, -0.0248994, 0.0248994, 0.044138, 0.0375062, 0.0407045,
      0.0267827, -0.00882794, -0.0218109, 0, 0, 0, 0.000369062, -0.000369062,
      -0.000369062, 0.000910177, -0.000910177, -0.000910177, 0.000150379,
      -0.000150379, -0.000150379, 0.00481744, -0.00481744, -0.00481744,
      0.000321504, -0.000321504, 3.68888e-05, 0.016982, -0.012651, -0.0104374,
      0.00331276, 0.00188842, 0.00122089, 0.0796555, 0.0254065, -0.0488774,
      0.00235633, 0.00110651, -0.00194003, 0.000712114, 0.000712114,
      -0.000417799, 0.000243683, 0.000243683, -9.37137e-05, 0.081236, -0.072496,
      0.0627802, 0.0674356, -0.0317289, -0.0291392, 0.0489792, 0.0456844,
      -0.0217452, 0.0314362, -0.00162502, 0.0274901, 0.00530486, 0.00530486,
      0.000687711, 0.00823003, 0.00273202, -0.000945777, 7.96146e-05,
      7.96146e-05, 7.09966e-05, 0, 0, 0, 0.019291, -0.0139819, 0.00407424,
      0.00634801, 0.00588934, -0.000270012, 0.0253013, -0.0165163, 0.0164835,
      0.00720889, 0.00616984, 0.000379624;
  world->setPositions(brokenPos);
  world->setVelocities(brokenVel);
  world->setExternalForces(brokenForce);
  world->setCachedLCPSolution(brokenLCPCache);

  // EXPECT_TRUE(verifyScratch(world, WithRespectTo::POSITION));
  EXPECT_TRUE(verifyF_c(world));
  // EXPECT_TRUE(verifyPosVelJacobian(world, brokenVel));
  // EXPECT_TRUE(verifyAnalyticalJacobians(world));
  // EXPECT_TRUE(verifyVelGradients(world, brokenVel));

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  Eigen::VectorXd animatePos = brokenPos;
  int i = 0;
  Ticker ticker(0.01);
  ticker.registerTickListener([&](long /* time */) {
    world->setPositions(animatePos);
    animatePos += brokenVel * 0.001;

    i++;
    if (i >= 100)
    {
      animatePos = brokenPos;
      i = 0;
    }
    // world->step();
    server.renderWorld(world);
  });

  server.registerConnectionListener([&]() { ticker.start(); });
  while (server.isServing())
  {
    // spin
  }

  std::shared_ptr<neural::BackpropSnapshot> snapshot
      = neural::forwardPass(world, true);
  EXPECT_TRUE(snapshot->areResultsStandardized());
}
#endif // TEST2

#ifdef ALL_TESTS
TEST(ATLAS_EXAMPLE, FULL_TEST)
{
  double target_x = 0.5;
  double target_y = 1.0;
  double target_z = -1.0;
  std::shared_ptr<simulation::World> world
      = createWorld(target_x, target_y, target_z);
  std::shared_ptr<dynamics::Skeleton> atlas = world->getSkeleton(0);

  trajectory::LossFn loss([target_x, target_y, target_z](
                              const trajectory::TrajectoryRollout* rollout) {
    const Eigen::VectorXd lastPos = rollout->getPosesConst("ik").col(
        rollout->getPosesConst("ik").cols() - 1);

    double diffX = lastPos(0) - target_x;
    double diffY = lastPos(1) - target_y;
    double diffZ = lastPos(2) - target_z;

    return diffX * diffX + diffY * diffY + diffZ * diffZ;
  });

  std::shared_ptr<neural::IKMapping> ikMapping
      = std::make_shared<neural::IKMapping>(world);
  // ikMapping->addLinearBodyNode(atlas->getBodyNode(0));
  ikMapping->addLinearBodyNode(atlas->getBodyNode("l_hand"));
  // atlas->getBodyNode("l_hand")

  // world->setTimeStep(0.01);

  std::shared_ptr<trajectory::MultiShot> trajectory
      = std::make_shared<trajectory::MultiShot>(world, loss, 300, 10, false);
  trajectory->addMapping("ik", ikMapping);
  trajectory->setParallelOperationsEnabled(false);

  GUIWebsocketServer server;
  server.serve(8070);
  server.renderWorld(world);

  Eigen::VectorXd atlasPos = world->getPositions();
  /*
  for (int i = 0; i < 1000; i++)
  {
    Eigen::VectorXd pos = ikMapping->getPositions(world);
    Eigen::MatrixXd jac = ikMapping->getMappedPosToRealPosJac(world);
    double diffX = pos(0) - target_x;
    double diffY = pos(1) - target_y;

    Eigen::Vector3d grad = Eigen::Vector3d(2.0 * diffX, 2.0 * diffY, 0.0);
    double loss = diffX * diffX + diffY * diffY;

    Eigen::VectorXd posDiff = jac * grad;
    atlasPos -= posDiff * 0.001;
    world->setPositions(atlasPos);
    server.renderWorld(world);
  }
  */

  /*
  server.renderBasis(
      10.0, "basis", Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
      */

  /*
  Ticker ticker(0.1);
  ticker.registerTickListener([&](long time) {
    Eigen::VectorXd pos = ikMapping->getPositions(world);
    Eigen::MatrixXd jac = ikMapping->getMappedPosToRealPosJac(world);
    double diffX = pos(0) - target_x;
    double diffY = pos(1) - target_y;
    double diffZ = pos(2) - target_z;

    Eigen::Vector3d grad
        = Eigen::Vector3d(2.0 * diffX, 2.0 * diffY, 2.0 * diffZ);
    double loss = diffX * diffX + diffY * diffY + diffZ * diffZ;

    Eigen::VectorXd posDiff = jac * grad;
    posDiff.segment(0, 6).setZero();
    atlasPos -= posDiff * 0.01;
    world->setPositions(atlasPos);
    server.renderWorld(world);
  });
  */

  // server.registerConnectionListener([&]() { ticker.start(); });

  trajectory::IPOptOptimizer optimizer;
  optimizer.setLBFGSHistoryLength(5);
  optimizer.setTolerance(1e-4);
  optimizer.setCheckDerivatives(true);
  optimizer.setIterationLimit(500);
  optimizer.registerIntermediateCallback(
      [&](trajectory::Problem* problem, int step, double primal, double dual) {
        const Eigen::MatrixXd poses
            = problem->getRolloutCache(world)->getPosesConst();
        const Eigen::MatrixXd vels
            = problem->getRolloutCache(world)->getVelsConst();
        std::cout << "Rendering trajectory lines" << std::endl;
        server.renderTrajectoryLines(world, poses);
        world->setPositions(poses.col(0));
        server.renderWorld(world);
        return true;
      });
  std::shared_ptr<trajectory::Solution> result
      = optimizer.optimize(trajectory.get());

  int i = 0;
  const Eigen::MatrixXd poses
      = result->getStep(result->getNumSteps() - 1).rollout->getPosesConst();
  const Eigen::MatrixXd vels
      = result->getStep(result->getNumSteps() - 1).rollout->getVelsConst();

  server.renderTrajectoryLines(world, poses);

  Ticker ticker(0.1);
  ticker.registerTickListener([&](long time) {
    world->setPositions(poses.col(i));
    // world->setVelocities(vels.col(i));

    i++;
    if (i >= poses.cols())
    {
      i = 0;
    }
    // world->step();
    server.renderWorld(world);
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  while (server.isServing())
  {
    // spin
  }
}
#endif // ALL_TESTS
