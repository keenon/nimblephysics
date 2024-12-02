#include <algorithm> // std::sort
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/ForcePlate.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/C3D.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;

// #define ALL_TESTS

// #ifdef ALL_TESTS
TEST(FORCE_PLATES, RESAMPLE_IDENTITY)
{
  ForcePlate plate;
  for (int i = 0; i < 3; i++)
  {
    plate.centersOfPressure.push_back(Eigen::Vector3s::UnitX() * i);
    plate.forces.push_back(Eigen::Vector3s::UnitY());
    plate.moments.push_back(Eigen::Vector3s::UnitY());
  }

  // Check that we the resampling matrix recovers itself, when the moments are
  // parallel to the forces
  auto pair = plate.getResamplingMatrixAndGroundHeights();
  ForcePlate recoveredPlate;
  recoveredPlate.setResamplingMatrixAndGroundHeights(pair.first, pair.second);

  for (int i = 0; i < 3; i++)
  {
    EXPECT_TRUE(plate.centersOfPressure[i].isApprox(
        recoveredPlate.centersOfPressure[i]));
    EXPECT_TRUE(plate.forces[i].isApprox(recoveredPlate.forces[i]));
    EXPECT_TRUE(plate.moments[i].isApprox(recoveredPlate.moments[i]));
  }
}
// #endif