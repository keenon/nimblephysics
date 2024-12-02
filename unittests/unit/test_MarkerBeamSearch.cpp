#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/MarkerBeamSearch.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/C3D.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

using namespace dart;
using namespace biomechanics;

#define ALL_TESTS

#ifdef ALL_TESTS
TEST(BEAM_SEARCH, DOES_NOT_CRASH)
{
  biomechanics::C3D c3d
      = biomechanics::C3DLoader::loadC3D("dart://sample/c3d/JA1Gait35.c3d");

  for (auto& pair : c3d.markerTimesteps[0])
  {
    std::string markerName = pair.first;
    auto result = MarkerBeamSearch::search(
        markerName, c3d.markerTimesteps, c3d.timestamps);
  }
}
#endif