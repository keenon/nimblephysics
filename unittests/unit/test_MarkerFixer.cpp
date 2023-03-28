#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// #include <experimental/filesystem>
#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/MarkerFixer.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;

// #define ALL_TESTS

#ifdef ALL_TESTS
TEST(MarkerFixer, SPRINTER_ARM_RIPPLE)
{
  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/grf/Sprinter2/MarkerData/JA1Gait35.trc");

  RippleReductionProblem problem(markerTrajectories.markerTimesteps);
  MarkersErrorReport report;
  problem.smooth(&report);
  for (std::string warning : report.warnings)
  {
    std::cout << "WARN: " << warning << std::endl;
  }
  // LARM
  // RP5MT
  problem.saveToGUI("RTOE", "../../../javascript/src/data/movement2.bin");
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFixer, FIX_SPRINT_MARKERS)
{
  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/grf/Sprinter/MarkerData/JA1Gait35.trc");

  auto report = MarkerFixer::generateDataErrorsReport(
      markerTrajectories.markerTimesteps,
      (1.0 / (s_t)markerTrajectories.framesPerSecond));
  for (auto& timestep : report.markerObservationsAttemptedFixed)
  {
    for (auto pair : timestep)
    {
      bool hasNan = pair.second.hasNaN();
      if (hasNan)
      {
        EXPECT_FALSE(pair.second.hasNaN());
        return;
      }
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFixer, FIX_CMU_MARKERS)
{
  // Get the raw marker trajectory data
  auto markerTrajectories
      = C3DLoader::loadC3D("dart://sample/c3d/cmu_02_05.c3d");

  auto report = MarkerFixer::generateDataErrorsReport(
      markerTrajectories.markerTimesteps,
      (1.0 / (s_t)markerTrajectories.framesPerSecond));
  for (auto& timestep : report->markerObservationsAttemptedFixed)
  {
    for (auto pair : timestep)
    {
      bool hasNan = pair.second.hasNaN();
      if (hasNan)
      {
        EXPECT_FALSE(pair.second.hasNaN());
        return;
      }
    }
  }
}
#endif

#ifdef ALL_TESTS
TEST(MarkerFixer, COMPARISON_TEST)
{
  // Get the raw marker trajectory data
  auto markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/osim/11_01_Marilyn_Bug/prod/MarkerData/markers_smpl.trc");

  auto report = MarkerFixer::generateDataErrorsReport(
      markerTrajectories.markerTimesteps,
      (1.0 / (s_t)markerTrajectories.framesPerSecond));
  for (auto& msg : report.info)
  {
    std::cout << "INFO: " << msg << std::endl;
  }
  for (auto& msg : report.warnings)
  {
    std::cout << "WARNING: " << msg << std::endl;
  }
  for (auto& timestep : report.markerObservationsAttemptedFixed)
  {
    for (auto pair : timestep)
    {
      bool hasNan = pair.second.hasNaN();
      if (hasNan)
      {
        EXPECT_FALSE(pair.second.hasNaN());
        return;
      }
    }
  }
}
#endif

// #ifdef ALL_TESTS
TEST(MarkerFixer, PASS_THROUGH_UNCHANGED)
{
  // Get the raw marker trajectory data
  auto markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/grf/AddBiomechanicsTutorialFiles/motion_capture_walk_trimmed.trc");

  auto report = MarkerFixer::generateDataErrorsReport(
      markerTrajectories.markerTimesteps,
      (1.0 / (s_t)markerTrajectories.framesPerSecond));
  for (std::string warning : report->warnings) {
    std::cout << "WARN: " << warning << std::endl;
  }

  EXPECT_EQ(0, report->warnings.size());
  EXPECT_EQ(0, report->info.size());
}
// #endif
