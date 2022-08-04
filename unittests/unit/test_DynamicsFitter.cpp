#include <memory>

#include <gtest/gtest.h>

#include "dart/biomechanics/DynamicsFitter.hpp"
#include "dart/biomechanics/IKErrorReport.hpp"
#include "dart/biomechanics/MarkerFitter.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/UniversalLoader.hpp"
#include "dart/utils/sdf/sdf.hpp"
#include "dart/utils/urdf/urdf.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

#define JACOBIAN_TESTS
// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

std::shared_ptr<DynamicsInitialization> runEngine(
    OpenSimFile standard,
    std::vector<std::vector<ForcePlate>> forcePlateTrials,
    std::vector<Eigen::MatrixXs> poseTrials,
    std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
        markerObservationTrials,
    std::vector<int> framesPerSecond,
    bool saveGUI = false)
{
  standard.skeleton->zeroTranslationInCustomFunctions();
  standard.skeleton->autogroupSymmetricSuffixes();
  if (standard.skeleton->getBodyNode("hand_r") != nullptr)
  {
    standard.skeleton->setScaleGroupUniformScaling(
        standard.skeleton->getBodyNode("hand_r"));
  }
  standard.skeleton->autogroupSymmetricPrefixes("ulna", "radius");
  standard.skeleton->setPositionLowerLimit(0, -M_PI);
  standard.skeleton->setPositionUpperLimit(0, M_PI);
  standard.skeleton->setPositionLowerLimit(1, -M_PI);
  standard.skeleton->setPositionUpperLimit(1, M_PI);
  standard.skeleton->setPositionLowerLimit(2, -M_PI);
  standard.skeleton->setPositionUpperLimit(2, M_PI);

  DynamicsFitter fitter(standard.skeleton, standard.markersMap);

  std::shared_ptr<DynamicsInitialization> init = fitter.createInitialization(
      forcePlateTrials, poseTrials, framesPerSecond, markerObservationTrials);
  fitter.scaleLinkMassesFromGravity(init);
  fitter.estimateLinkMassesFromAcceleration(init);

  if (saveGUI)
  {
    std::cout << "Saving trajectory..." << std::endl;
    std::cout << "FPS: " << framesPerSecond[0] << std::endl;
    fitter.saveDynamicsToGUI(
        "../../../javascript/src/data/movement2.bin",
        init,
        0,
        framesPerSecond[0]);
  }

  return init;
}

std::shared_ptr<DynamicsInitialization> runEngine(
    std::string modelPath,
    std::vector<std::string> motFiles,
    std::vector<std::string> c3dFiles,
    std::vector<std::string> trcFiles,
    std::vector<std::string> grfFiles,
    bool saveGUI = false)
{
  std::vector<Eigen::MatrixXs> poseTrials;
  std::vector<C3D> c3ds;
  std::vector<std::vector<std::map<std::string, Eigen::Vector3s>>>
      markerObservationTrials;
  std::vector<int> framesPerSecond;
  std::vector<std::vector<ForcePlate>> forcePlates;

  OpenSimFile standard = OpenSimParser::parseOsim(modelPath);
  for (int i = 0; i < motFiles.size(); i++)
  {
    OpenSimMot mot = OpenSimParser::loadMot(standard.skeleton, motFiles[i]);
    poseTrials.push_back(mot.poses);
  }

  for (std::string& path : c3dFiles)
  {
    C3D c3d = C3DLoader::loadC3D(path);
    c3ds.push_back(c3d);
    markerObservationTrials.push_back(c3d.markerTimesteps);
    forcePlates.push_back(c3d.forcePlates);
    framesPerSecond.push_back(c3d.framesPerSecond);
  }

  for (int i = 0; i < trcFiles.size(); i++)
  {
    OpenSimTRC trc = OpenSimParser::loadTRC(trcFiles[i]);
    framesPerSecond.push_back(trc.framesPerSecond);

    markerObservationTrials.push_back(trc.markerTimesteps);
    if (i < grfFiles.size())
    {
      std::vector<ForcePlate> grf
          = OpenSimParser::loadGRF(grfFiles[i], trc.framesPerSecond);
      forcePlates.push_back(grf);
    }
    else
    {
      forcePlates.emplace_back();
    }
  }

  return runEngine(
      standard,
      forcePlates,
      poseTrials,
      markerObservationTrials,
      framesPerSecond,
      saveGUI);
}

#ifdef ALL_TESTS
TEST(DynamicsFitter, MASS_INITIALIZATION)
{
  std::vector<std::string> motFiles;
  std::vector<std::string> c3dFiles;
  std::vector<std::string> trcFiles;
  std::vector<std::string> grfFiles;

  motFiles.push_back("dart://sample/grf/Subject4/IK/walking1_ik.mot");
  trcFiles.push_back("dart://sample/grf/Subject4/MarkerData/walking1.trc");
  grfFiles.push_back("dart://sample/grf/Subject4/ID/walking1_grf.mot");

  runEngine(
      "dart://sample/grf/Subject4/Models/"
      "optimized_scale_and_markers.osim",
      motFiles,
      c3dFiles,
      trcFiles,
      grfFiles,
      true);
}
#endif