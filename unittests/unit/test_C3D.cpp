#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/C3D.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;

// #define ALL_TESTS

TEST(C3D, COMPARE_TO_TRC)
{
  biomechanics::C3D c3d
      = biomechanics::C3DLoader::loadC3D("dart://sample/c3d/JA1Gait35.c3d");
  biomechanics::OpenSimTRC trc = biomechanics::OpenSimParser::loadTRC(
      "dart://sample/osim/Sprinter/run0900cms.trc");

  EXPECT_EQ(c3d.markerTimesteps.size(), trc.markerTimesteps.size());
  for (int i = 0; i < trc.markerTimesteps.size(); i++)
  {
    EXPECT_TRUE(c3d.markerTimesteps[i].size() <= trc.markerTimesteps[i].size());
    for (auto& pair : c3d.markerTimesteps[i])
    {
      if (trc.markerTimesteps[i].count(pair.first) == 0)
      {
        EXPECT_TRUE(trc.markerTimesteps[i].count(pair.first) > 0);
        return;
      }

      Eigen::Vector3s c3dVec = c3d.markerTimesteps[i][pair.first];
      Eigen::Vector3s trcVec = trc.markerTimesteps[i][pair.first];

      if (!equals(c3dVec, trcVec, 1e-9))
      {
        std::cout << "Mismatch on frame " << i << ":" << pair.first
                  << std::endl;
        std::cout << "TRC: " << std::endl << trcVec << std::endl;
        std::cout << "C3D: " << std::endl << c3dVec << std::endl;
        std::cout << "Diff: " << std::endl << trcVec - c3dVec << std::endl;
        EXPECT_TRUE(equals(c3dVec, trcVec, 1e-9));
        return;
      }
    }
  }
}

#ifdef ALL_TESTS
TEST(C3D, LOAD)
{
  biomechanics::C3D c3d
      = biomechanics::C3DLoader::loadC3D("dart://sample/c3d/JA1Gait35.c3d");
  // = biomechanics::C3DLoader::loadC3D("dart://sample/c3d/S01DS402.c3d");
  // = biomechanics::C3DLoader::loadC3D("dart://sample/c3d/S01DB201.c3d");

  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderBasis(1.0);
  biomechanics::C3DLoader::debugToGUI(c3d, server);
}
#endif