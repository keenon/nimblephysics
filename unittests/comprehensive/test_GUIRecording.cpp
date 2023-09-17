#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include <dart/utils/urdf/urdf.hpp>
#include <dart/utils/utils.hpp>
#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIRecording.hpp"

#include "TestHelpers.hpp"
#include "stdio.h"

// #define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;
using namespace server;
using namespace realtime;
using namespace biomechanics;

// #ifdef ALL_TESTS
TEST(REALTIME, GUI_SERVER_2)
{
  GUIRecording recording;

  recording.createLayer(
      "Test layer", Eigen::Vector4s(1.0, 0.5, 0.5, 1.0), true);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/welk007/unscaled_generic.osim");
  auto mot = OpenSimParser::loadMot(
      standard.skeleton,
      "dart://sample/osim/welk007/"
      "c3d_Trimmed_running_natural2_manual_scaling_ik.mot");

  int startTime = 100;
  int endTime = 500;
  recording.setSpanWarning(
      startTime, endTime, "warn_obj", "Suspicious marker jump for MKR2");
  recording.setSpanWarning(
      startTime, endTime, "warn_hand", "Suspicious marker jump for MKR2");

  recording.setSpanWarning(
      700, 702, "warn2", "Suspicious marker jump for MKR2");

  for (int i = 0; i < mot.poses.cols(); i++)
  {
    standard.skeleton->setPositions(mot.poses.col(i));
    recording.renderSkeleton(standard.skeleton);

    if (i >= startTime && i <= endTime)
    {
      recording.setObjectWarning(
          "skel_Skeleton_tibia_l_0",
          "warn_obj",
          "Suspicious marker jump for MKR2");
      recording.setObjectWarning(
          "skel_Skeleton_hand_l_21", "warn_hand", "Warning on the hands");
    }
    else
    {
      recording.deleteObjectWarning("skel_Skeleton_tibia_l_0", "warn_obj");
      recording.deleteObjectWarning("skel_Skeleton_hand_l_21", "warn_hand");
    }
    if (i >= 700 && i <= 702)
    {
      recording.setObjectWarning(
          "skel_Skeleton_tibia_l_0", "warn2", "Flickering warning");
    }
    else
    {
      recording.deleteObjectWarning("skel_Skeleton_tibia_l_0", "warn2");
    }

    recording.saveFrame();
  }

  recording.writeFramesJson("../../../javascript/src/data/movement2.bin");
}
// #endif

#ifdef ALL_TESTS
TEST(RECORDING, GUI_SERVER)
{
  // Create a world
  /*
  std::shared_ptr<simulation::World> world
      = dart::utils::SkelParser::readWorld("dart://sample/skel/cartpole.skel");
      */

  std::shared_ptr<simulation::World> world = simulation::World::create();

  // Set gravity of the world
  // world->setPenetrationCorrectionEnabled(true);
  world->setGravity(Eigen::Vector3s(0.0, -9.81, 0.0));

  // Load ground and Atlas robot and add them to the world
  dart::utils::DartLoader urdfLoader;
  std::shared_ptr<dynamics::Skeleton> ground
      = urdfLoader.parseSkeleton("dart://sample/sdf/atlas/ground.urdf");

  std::shared_ptr<dynamics::Skeleton> atlas
      = dart::utils::SdfParser::readSkeleton(
          "dart://sample/sdf/atlas/atlas_v3_no_head.sdf");

  /*
  std::shared_ptr<dynamics::Skeleton> kr5
      = urdfLoader.parseSkeleton("dart://sample/urdf/KR5/KR5 sixx R650.urdf");
  */

  world->addSkeleton(ground);
  world->addSkeleton(atlas);
  // world->addSkeleton(kr5);

  // Set initial configuration for Atlas robot
  atlas->setPosition(0, -0.5 * dart::math::constantsd::pi());
  atlas->setPosition(3, 0.75);

  // Disable the ground from casting its own shadows
  ground->getBodyNode(0)->getShapeNode(0)->getVisualAspect()->setCastShadows(
      false);

  /*
  while (true)
  {
    world->step();
  }
  */

  GUIRecording recording;
  recording.renderWorld(world);
  recording.saveFrame();

  for (int i = 0; i < 100; i++)
  {
    world->step();
    recording.renderWorld(world);
    recording.saveFrame();
  }

  std::cout << recording.getFrameJson(3) << std::endl;

  // recording.saveFramesJson("./atlas_recording.json");
}
#endif
