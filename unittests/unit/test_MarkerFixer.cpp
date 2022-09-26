#include <cstdio>
#include <filesystem>
#include <fstream>
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

// #ifdef ALL_TESTS
TEST(MarkerFixer, SPRINTER_ARM_RIPPLE)
{
  // Get the raw marker trajectory data
  OpenSimTRC markerTrajectories = OpenSimParser::loadTRC(
      "dart://sample/grf/Sprinter/MarkerData/JA1Gait35.trc");

  RippleReductionProblem problem(markerTrajectories.markerTimesteps);
  problem.smooth();
  problem.saveToGUI("LARM", "../../../javascript/src/data/movement2.bin");
}
// #endif