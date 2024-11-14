#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/MarkerBeamSearch.hpp"
#include "dart/biomechanics/MarkerMultiBeamSearch.hpp"
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
TEST(MULTI_BEAM_SEARCH, DOES_NOT_CRASH)
{
  biomechanics::C3D c3d
      = biomechanics::C3DLoader::loadC3D("dart://sample/c3d/cmu_02_05.c3d");

  std::vector<std::string> markerNames;
  for (auto& pair : c3d.markerTimesteps[0])
  {
    markerNames.push_back(pair.first);
  }

  std::vector<std::map<std::string, Eigen::Vector3d>> markerTimesteps;
  for (int i = 0; i < std::min<int>(c3d.markerTimesteps.size(), 3000); i++)
  {
    markerTimesteps.push_back(c3d.markerTimesteps[i]);
  }

  auto result = MarkerMultiBeamSearch::search(
      markerNames, markerTimesteps, c3d.timestamps, 20, 7.0, 1000.0, 1, 100);
}
#endif