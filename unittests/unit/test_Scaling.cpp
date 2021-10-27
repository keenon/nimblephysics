#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

#ifdef ALL_TESTS
TEST(Scaling, SCALE_BODY_NODES)
{
  std::shared_ptr<dynamics::Skeleton> osim
      = OpenSimParser::parseOsim(
            "dart://sample/osim/Rajagopal2015/Rajagopal2015.osim")
            .skeleton;

  osim->getBodyNode("tibia_l")->setScale(Eigen::Vector3s(1.1, 1.2, 1.3));

  server::GUIWebsocketServer server;
  server.serve(8070);
  server.renderSkeleton(osim);
  server.setAutoflush(false);

  Ticker ticker(1.0 / 20);

  int ticks = 0;
  ticker.registerTickListener([&](long /* time */) {
    int offset = (ticks++) % 200;
    double percentage = ((double)offset / 200);
    double scale = 0.5 + percentage;
    std::cout << "Scale: " << scale << std::endl;
    osim->getBodyNode("tibia_l")->setScale(
        Eigen::Vector3s(scale, scale, scale));
    server.renderSkeleton(osim);
    server.flush();
  });

  server.registerConnectionListener([&]() { ticker.start(); });

  server.blockWhileServing();
}
#endif