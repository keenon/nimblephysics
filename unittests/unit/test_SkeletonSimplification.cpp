#include <memory>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIRecording.hpp"
#include "dart/server/GUIStateMachine.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

TEST(SkeletonSimplification, SIMPLIFY_OSIM)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/grf/Subject4/Models/"
            "optimized_scale_and_markers.osim")
            // "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;

  std::map<std::string, std::string> mergeBodiesInto;
  mergeBodiesInto["ulna_r"] = "radius_r";
  mergeBodiesInto["ulna_l"] = "radius_l";
  std::shared_ptr<dynamics::Skeleton> simplified
      = osim->simplifySkeleton("clone", mergeBodiesInto);

  const std::string& path = "../../../javascript/src/data/movement2.bin";
  server::GUIRecording recording;

  for (int i = 0; i < 100; i++)
  {
    s_t percentage = (s_t)i / 100;

    auto* dof = osim->getDof("knee_angle_r");
    dof->setPosition(
        percentage
            * (dof->getPositionUpperLimit() - dof->getPositionLowerLimit())
        + dof->getPositionLowerLimit());
    auto* simplifiedDof = simplified->getDof("walker_knee_r");
    simplifiedDof->setPosition(
        percentage
            * (simplifiedDof->getPositionUpperLimit()
               - simplifiedDof->getPositionLowerLimit())
        + simplifiedDof->getPositionLowerLimit());

    recording.renderSkeleton(simplified);
    recording.renderSkeleton(osim, "original", Eigen::Vector4s(1, 0, 0, 0.5));
    recording.saveFrame();
  }

  recording.writeFramesJson(path);
}