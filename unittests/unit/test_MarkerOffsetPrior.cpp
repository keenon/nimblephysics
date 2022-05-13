#include <algorithm> // std::sort
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <ccd/ccd.h>
#include <gtest/gtest.h>
#include <math.h>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/MarkerLabeller.hpp"
#include "dart/biomechanics/MarkerOffsetPrior.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

using namespace dart;
using namespace biomechanics;
using namespace server;

// #define ALL_TESTS

#ifdef ALL_TESTS
TEST(MARKER_PRIOR, BASIC_GUI)
{
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);

  OpenSimFile standard = OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal2015_v3_scaled/"
      "Rajagopal2015_passiveCal_hipAbdMoved.osim");
  standard.skeleton->autogroupSymmetricSuffixes();

  MarkerOffsetPrior prior(standard.skeleton, standard.markersMap);

  prior.debugToGUI(server);
  server->blockWhileServing();
}
#endif