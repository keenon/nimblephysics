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
TEST(UniversalLoader, LOAD_SDF_ATLAS_FROM_DART)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world, "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");
  EXPECT_TRUE(skel != nullptr);
}

//==============================================================================
TEST(UniversalLoader, LOAD_SDF_ATLAS_FROM_RELATIVE_PATH)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  // This assumes we're running from dart/build/unittests/unit
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world,
      "../../../data/sdf/atlas/"
      "atlas_v3_no_head.sdf");
  EXPECT_TRUE(skel != nullptr);
}

//==============================================================================
TEST(UniversalLoader, LOAD_SKEL_FROM_DART)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world, "dart://sample//skel/test/cube_skeleton.skel");
  EXPECT_TRUE(skel != nullptr);
}

//==============================================================================
TEST(UniversalLoader, LOAD_SKEL_FROM_RELATIVE_PATH)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  // This assumes we're running from dart/build/unittests/unit
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world, "../../../data/skel/test/cube_skeleton.skel");
  EXPECT_TRUE(skel != nullptr);
}

//==============================================================================
TEST(UniversalLoader, LOAD_URDF_FROM_DART)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world, "dart://sample/sdf/atlas/ground.urdf");
  EXPECT_TRUE(skel != nullptr);
}

//==============================================================================
TEST(UniversalLoader, LOAD_URDF_FROM_RELATIVE_PATH)
{
  std::shared_ptr<simulation::World> world = simulation::World::create();
  // This assumes we're running from dart/build/unittests/unit
  std::shared_ptr<dynamics::Skeleton> skel = UniversalLoader::loadSkeleton(
      world, "../../../data/sdf/atlas/ground.urdf");
  EXPECT_TRUE(skel != nullptr);
}