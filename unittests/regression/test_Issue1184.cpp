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

#include <gtest/gtest.h>
#include <TestHelpers.hpp>

#include <dart/collision/bullet/BulletCollisionDetector.hpp>
#include <dart/dynamics/BoxShape.hpp>
#include <dart/dynamics/FreeJoint.hpp>
#include <dart/dynamics/PlaneShape.hpp>
#include <dart/dynamics/SphereShape.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <dart/dynamics/WeldJoint.hpp>
#include <dart/simulation/World.hpp>

//==============================================================================
TEST(Issue1184, Accuracy)
{

  struct ShapeInfo
  {
    dart::dynamics::ShapePtr shape;
    s_t offset;
  };

  std::function<ShapeInfo()> makePlaneGround =
      []()
  {
    return ShapeInfo{
      std::make_shared<dart::dynamics::PlaneShape>(
            Eigen::Vector3s::UnitZ(), 0.0),
      0.0};
  };

  std::function<ShapeInfo()> makeBoxGround =
      []()
  {
    const s_t thickness = 0.1;
    return ShapeInfo{
      std::make_shared<dart::dynamics::BoxShape>(
            Eigen::Vector3s(100.0, 100.0, thickness)),
      -thickness/2.0};
  };

  std::function<dart::dynamics::ShapePtr(s_t)> makeBoxObject =
      [](const s_t s) -> dart::dynamics::ShapePtr
  {
    return std::make_shared<dart::dynamics::BoxShape>(
          Eigen::Vector3s::Constant(2*s));
  };

  std::function<dart::dynamics::ShapePtr(s_t)> makeSphereObject =
      [](const s_t s) -> dart::dynamics::ShapePtr
  {
    return std::make_shared<dart::dynamics::SphereShape>(s);
  };


#ifndef NDEBUG
  const auto groundInfoFunctions = {makePlaneGround};
  const auto objectShapeFunctions = {makeSphereObject};
  const auto halfsizes = {10.0};
  const auto fallingModes = {true};
  const s_t dropHeight = 0.1;
  const s_t tolerance = 1e-3;
#else
  const auto groundInfoFunctions = {makePlaneGround, makeBoxGround};
  const auto objectShapeFunctions = {makeBoxObject, makeSphereObject};
  const auto halfsizes = {0.25, 1.0, 5.0, 10.0, 20.0};
  const auto fallingModes = {true, false};
  const s_t dropHeight = 1.0;
  const s_t tolerance = 1e-3;
#endif

  for(const auto& groundInfoFunction : groundInfoFunctions)
  {
    for(const auto& objectShapeFunction : objectShapeFunctions)
    {
      for(const s_t halfsize : halfsizes)
      {
        for(const bool falling : fallingModes)
        {
          auto world = dart::simulation::World::create("test");
          world->getConstraintSolver()->setCollisionDetector(
                dart::collision::BulletCollisionDetector::create());

          Eigen::Isometry3s tf_object = Eigen::Isometry3s::Identity();
          const s_t initialHeight = falling? dropHeight+halfsize : halfsize;
          tf_object.translate(initialHeight*Eigen::Vector3s::UnitZ());

          auto object = dart::dynamics::Skeleton::create("ball");
          object->createJointAndBodyNodePair<dart::dynamics::FreeJoint>()
              .first->setTransform(tf_object);

          const auto objectShape = objectShapeFunction(halfsize);
          object->getBodyNode(0)->createShapeNodeWith<
              dart::dynamics::VisualAspect,
              dart::dynamics::CollisionAspect>(objectShape);


          world->addSkeleton(object);

          const ShapeInfo groundInfo = groundInfoFunction();
          auto ground = dart::dynamics::Skeleton::create("ground");
          ground->createJointAndBodyNodePair<dart::dynamics::WeldJoint>()
              .second->createShapeNodeWith<
                dart::dynamics::VisualAspect,
                dart::dynamics::CollisionAspect>(groundInfo.shape);

          Eigen::Isometry3s tf_ground = Eigen::Isometry3s::Identity();
          tf_ground.translate(groundInfo.offset*Eigen::Vector3s::UnitZ());
          ground->getJoint(0)->setTransformFromParentBodyNode(tf_ground);

          world->addSkeleton(ground);

          // time until the object will strike
          const s_t t_strike = falling?
              sqrt(-2.0*dropHeight/world->getGravity()[2]) : 0.0;

          // give the object some time to settle
          const s_t min_time = 0.5;
          const s_t t_limit = 30.0*t_strike + min_time;

          s_t lowestHeight = std::numeric_limits<s_t>::infinity();
          s_t time = 0.0;
          while(time < t_limit)
          {
            world->step();
            const s_t currentHeight =
                object->getBodyNode(0)->getTransform().translation()[2];

            if(currentHeight < lowestHeight)
              lowestHeight = currentHeight;

            time = world->getTime();
          }

          // The simulation should have run for at least two seconds
          ASSERT_LE(min_time, time);

          EXPECT_GE(halfsize+tolerance, lowestHeight)
              << "object type: " << objectShape->getType()
              << "\nground type: " << groundInfo.shape->getType()
              << "\nfalling: " << falling << "\n";

          const s_t finalHeight =
              object->getBodyNode(0)->getTransform().translation()[2];

          EXPECT_NEAR(halfsize, finalHeight, tolerance)
              << "object type: " << objectShape->getType()
              << "\nground type: " << groundInfo.shape->getType()
              << "\nfalling: " << falling << "\n";
        }
      }
    }
  }

}
