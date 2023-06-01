#include "dart/biomechanics/C3DLoader.hpp"

#include <algorithm> // std::sort
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <ezc3d/ezc3d.h>
#include <ezc3d/ezc3d_all.h>
#include <math.h>

#include "dart/biomechanics/C3DForcePlatforms.hpp"
#include "dart/common/LocalResourceRetriever.hpp"
#include "dart/common/ResourceRetriever.hpp"
#include "dart/common/Uri.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/utils/CompositeResourceRetriever.hpp"
#include "dart/utils/DartResourceRetriever.hpp"
#include "dart/utils/PackageResourceRetriever.hpp"

namespace dart {

namespace biomechanics {

//==============================================================================
std::string getAbsolutePath(std::string uri)
{
  const utils::CompositeResourceRetrieverPtr resourceRetriever
      = std::make_shared<utils::CompositeResourceRetriever>();
  common::LocalResourceRetrieverPtr localResourceRetriever
      = std::make_shared<common::LocalResourceRetriever>();
  resourceRetriever->addSchemaRetriever("file", localResourceRetriever);
  utils::PackageResourceRetrieverPtr packageRetriever
      = std::make_shared<utils::PackageResourceRetriever>(
          localResourceRetriever);
  resourceRetriever->addSchemaRetriever("package", packageRetriever);
  resourceRetriever->addSchemaRetriever(
      "dart", utils::DartResourceRetriever::create());
  return resourceRetriever->getFilePath(uri);
}

/*
  // Choose the convention we will use for reading for the force data based on
  // which force data convention yields the lowest average RMS distance from
  // each CoP to its nearest mocap point. The intuition behind this method is
  // that conventions that are blatantly wrong will tend to be pretty far away
  // from the nearest foot marker.
  */

//==============================================================================
/// This computes the weighted distance from each CoP to its nearest marker on
/// that timestep. This is used as part of the heuristic to guess which
/// convention a C3D file is using for storing its GRF data.
s_t C3D::getWeightedDistFromCoPToNearestMarker()
{
  s_t dist = 0;
  s_t totalWeight = 0;

  for (ForcePlate& plate : forcePlates)
  {
    for (int i = 0; i < markerTimesteps.size(); i++)
    {
      Eigen::Vector3s cop = plate.centersOfPressure[i];

      s_t minDist = std::numeric_limits<double>::infinity();
      for (auto& pair : markerTimesteps[i])
      {
        s_t dist = (pair.second - cop).norm();
        if (dist < minDist)
        {
          minDist = dist;
        }
      }
      if (isfinite(minDist))
      {
        s_t weight = plate.forces[i].norm();

        assert(!isnan(minDist));
        dist += minDist * weight;
        totalWeight += weight;
      }
    }
  }

  if (totalWeight == 0)
    return 0.0;

  assert(!isnan(dist));
  assert(!isnan(totalWeight));
  assert(totalWeight > 0);

  return dist / totalWeight;
}

//==============================================================================
C3D C3DLoader::loadC3D(const std::string& uri)
{
  C3D bestResult = loadC3DWithGRFConvention(uri, 0);
  s_t bestResultRMS = bestResult.getWeightedDistFromCoPToNearestMarker();

  // If there are no GRF's, no need to try multiple conventions
  if (bestResultRMS == 0)
  {
    return bestResult;
  }

  for (int i = 1; i < biomechanics::FORCE_PLATFORM_NUM_CONVENTIONS; i++)
  {
    C3D competingConvention = loadC3DWithGRFConvention(uri, i);
    s_t competingRMS
        = competingConvention.getWeightedDistFromCoPToNearestMarker();
    std::cout << "Tried force plate convention " << i << ". Best RMS "
              << bestResultRMS << " vs this RMS " << competingRMS << std::endl;
    if (competingRMS < bestResultRMS)
    {
      bestResult = competingConvention;
      bestResultRMS = competingRMS;
    }
  }
  return bestResult;
}

//==============================================================================
C3D C3DLoader::loadC3DWithGRFConvention(const std::string& uri, int convention)
{
  C3D result;
  const std::string path = getAbsolutePath(uri);

  std::string fullPath = getAbsolutePath(uri);

  ezc3d::c3d data(fullPath);

  double frameRate = data.header().frameRate();
  std::cout << "Framerate: " << frameRate << std::endl;
  result.framesPerSecond = frameRate;
  int numFrames = data.header().nbFrames();
  int analogFramesPerFrame = data.header().nbAnalogByFrame();

  // Read the units that the mocap points are declared in
  double mocapDataScaleFactor = 1.0;
  const ezc3d::ParametersNS::Parameters& params = data.parameters();
  if (params.isGroup("POINT"))
  {
    const ezc3d::ParametersNS::GroupNS::Group& pointGroup
        = params.group(params.groupIdx("POINT"));
    if (pointGroup.isParameter("UNITS"))
    {
      const ezc3d::ParametersNS::GroupNS::Parameter& pointUnits
          = pointGroup.parameter(pointGroup.parameterIdx("UNITS"));
      const std::vector<std::string>& pointValues = pointUnits.valuesAsString();
      if (pointValues.size() > 0)
      {
        const std::string pointUnit = pointValues[0];
        std::cout << "Point units: " << pointUnit << std::endl;
        if (pointUnit == "mm")
        {
          mocapDataScaleFactor = 0.001;
        }
        else if (pointUnit == "cm")
        {
          mocapDataScaleFactor = 0.01;
        }
        else if (pointUnit == "ft")
        {
          mocapDataScaleFactor = 0.3048;
        }
        else if (pointUnit == "in")
        {
          mocapDataScaleFactor = 0.0254;
        }
        else if (pointUnit == "m")
        {
          mocapDataScaleFactor = 1.0;
        }
      }
    }
  }

  // Copy down the names of the points
  for (const std::string& name : data.pointNames())
  {
    std::string fixed = name;
    for (int i = 0; i < fixed.size(); i++)
    {
      if (fixed[i] == '*')
      {
        fixed[i] = 'x';
      }
    }
    if (fixed.find_first_of(":", 0) != std::string::npos)
    {
      fixed = fixed.substr(fixed.find_first_of(":") + 1);
    }
    result.markers.push_back(fixed);
  }

  // Load in the force platforms
  ForcePlatforms pf(data, convention);
  const std::vector<ForcePlatform>& forcePlatforms = pf.forcePlatforms();

  // Force plate data
  const ezc3d::ParametersNS::GroupNS::Group& groupFP(
      data.parameters().group("FORCE_PLATFORM"));
  const std::vector<double>& all_origins(
      groupFP.parameter("ORIGIN").valuesAsDouble());
  const std::vector<int>& types = groupFP.parameter("TYPE").valuesAsInt();
  std::cout << "All origins: ";
  for (double d : all_origins)
  {
    std::cout << d << ", ";
  }
  std::cout << std::endl;
  std::cout << "All types: ";
  for (int i : types)
  {
    std::cout << i << ", ";
  }
  std::cout << std::endl;

  // Process all the force platforms
  std::vector<double> forcePlateForceScaleFactors;
  std::vector<double> forcePlateMomentScaleFactors;
  std::vector<double> forcePlatePositionScaleFactors;
  for (int j = 0; j < forcePlatforms.size(); j++)
  {
    double forceScaleFactor = 1.0;
    if (forcePlatforms[j].forceUnit() == "N")
    {
      forceScaleFactor = 1.0;
    }
    else if (forcePlatforms[j].forceUnit() == "mN")
    {
      forceScaleFactor = 0.001;
    }
    else if (forcePlatforms[j].forceUnit() == "cN")
    {
      forceScaleFactor = 0.01;
    }
    forcePlateForceScaleFactors.push_back(forceScaleFactor);

    double momentScaleFactor = 1.0;
    if (forcePlatforms[j].momentUnit() == "Nm")
    {
      momentScaleFactor = 1.0;
    }
    if (forcePlatforms[j].momentUnit() == "Nmm")
    {
      momentScaleFactor = 0.001;
    }
    if (forcePlatforms[j].momentUnit() == "Ncm")
    {
      momentScaleFactor = 0.01;
    }
    forcePlateMomentScaleFactors.push_back(momentScaleFactor);

    std::cout << "forcePlatform forceUnit: " << forcePlatforms[j].forceUnit()
              << std::endl;
    std::cout << "forcePlatform momentUnit: " << forcePlatforms[j].momentUnit()
              << std::endl;
    std::cout << "forcePlatform positionUnit: "
              << forcePlatforms[j].positionUnit() << std::endl;
    std::cout << "forcePlatform origin: " << forcePlatforms[j].origin().x()
              << ", " << forcePlatforms[j].origin().y() << ", "
              << forcePlatforms[j].origin().z() << std::endl;
    std::cout << "forcePlatform mean corners: "
              << forcePlatforms[j].meanCorners().x() << ", "
              << forcePlatforms[j].meanCorners().y() << ", "
              << forcePlatforms[j].meanCorners().z() << std::endl;

    double positionScaleFactor = 1.0;
    if (forcePlatforms[j].positionUnit() == "mm")
    {
      positionScaleFactor = 0.001;
    }
    else if (forcePlatforms[j].positionUnit() == "cm")
    {
      positionScaleFactor = 0.01;
    }
    else if (forcePlatforms[j].positionUnit() == "ft")
    {
      positionScaleFactor = 0.3048;
    }
    else if (forcePlatforms[j].positionUnit() == "in")
    {
      positionScaleFactor = 0.0254;
    }
    else if (forcePlatforms[j].positionUnit() == "m")
    {
      positionScaleFactor = 1.0;
    }
    forcePlatePositionScaleFactors.push_back(positionScaleFactor);

    result.forcePlates.emplace_back();
    ForcePlate& forcePlate = result.forcePlates[j];
    forcePlate.worldOrigin
        = (forcePlatforms[j].meanCorners() + forcePlatforms[j].origin())
          * positionScaleFactor;
    for (const auto& corner : forcePlatforms[j].corners())
    {
      forcePlate.corners.push_back(Eigen::Vector3s(
          corner.x() * positionScaleFactor,
          corner.y() * positionScaleFactor,
          corner.z() * positionScaleFactor));
      // Corner 0 = +x +y
      // Corner 1 = -x +y
      // Corner 2 = -x -y
      // Corner 3 = +x -y
    }
  }

  int startFrame = 2;
  for (int t = 0; t < numFrames - startFrame; t++)
  {
    result.timestamps.push_back(t / frameRate);

    result.markerTimesteps.emplace_back();
    std::map<std::string, Eigen::Vector3s>& map = result.markerTimesteps.at(t);
    for (int i = 0; i < result.markers.size(); i++)
    {
      const std::string& name = result.markers[i];
      Eigen::Vector3s pt = Eigen::Vector3s(
          data.data().frame(t + startFrame).points().point(i).x()
              * mocapDataScaleFactor,
          data.data().frame(t + startFrame).points().point(i).y()
              * mocapDataScaleFactor,
          data.data().frame(t + startFrame).points().point(i).z()
              * mocapDataScaleFactor);
      if (pt == Eigen::Vector3s::Zero() || pt.hasNaN())
      {
        // Don't store points with all zeros, since those are "unobserved"
      }
      else
      {
        map[name] = pt;
      }
    }

    for (int j = 0; j < forcePlatforms.size(); j++)
    {
      int frame = analogFramesPerFrame * (t + startFrame);
      result.forcePlates[j].timestamps.push_back(t / frameRate);
      result.forcePlates[j].forces.push_back(
          forcePlatforms[j].forces()[frame] * forcePlateForceScaleFactors[j]);
      result.forcePlates[j].moments.push_back(
          forcePlatforms[j].Tz()[frame] * forcePlateMomentScaleFactors[j]);
      result.forcePlates[j].centersOfPressure.push_back(
          forcePlatforms[j].CoP()[frame] * forcePlatePositionScaleFactors[j]);
    }
  }

  result.dataRotation = Eigen::Matrix3s::Identity();
  // Automatically rotate the result so that the force plates are on the ground
  if (result.forcePlates.size() > 0
      && result.forcePlates[0].corners.size() == 4)
  {
    Eigen::Vector3s up
        = (result.forcePlates[0].corners[1] - result.forcePlates[0].corners[0])
              .cross(
                  (result.forcePlates[0].corners[2]
                   - result.forcePlates[0].corners[1]))
              .normalized();
    s_t groundLevel = result.forcePlates[0].corners[0].dot(up);
    // Flip the direction of "up" if the markers are showing up as below the
    // ground
    if (result.markerTimesteps.size() > 0)
    {
      s_t sumDist = 0.0;
      for (auto& pair : result.markerTimesteps[(int)std::round(
               result.markerTimesteps.size() / 2)])
      {
        sumDist += pair.second.dot(up) - groundLevel;
      }
      if (sumDist < 0)
      {
        up *= -1;
      }
    }
    // Flip the direction of "up" if the force plates are showing up as
    // producing downward sum forces
    s_t sumForcesUp = 0.0;
    for (int i = 0; i < result.forcePlates.size(); i++)
    {
      for (int t = 0; t < result.forcePlates[i].forces.size(); t++)
      {
        sumForcesUp += up.dot(result.forcePlates[i].forces[t]);
      }
    }
    if (sumForcesUp < 0)
    {
      up *= -1;
    }

    // Try to get "up" to point exactly along a unit axis, to make subsequent
    // math more accurate
    up = up.normalized();
    if (abs(up(0)) < 0.01)
    {
      up(0) = 0;
    }
    if (abs(up(1)) < 0.01)
    {
      up(1) = 0;
    }
    if (abs(up(2)) < 0.01)
    {
      up(2) = 0;
    }
    up = up.normalized();

    // Complete the "up" vector into a full basis
    Eigen::Matrix3s R = Eigen::Matrix3s::Identity();
    if (up == Eigen::Vector3s::UnitY())
    {
      // Do nothing
    }
    else
    {
      // We want "up" to point at UnitY, so we need a rotation that'll get us
      // there
      Eigen::Vector3s rotVector
          = up.cross(Eigen::Vector3s::UnitY()).normalized() * M_PI / 2;
      // R = math::expMapRot(rotVector);
      // Rotate by 90deg along the Y axis
      R = math::expMapRot(-Eigen::Vector3s::UnitY() * M_PI / 2)
          * math::expMapRot(rotVector);

      result.dataRotation = R;
#ifndef NDEBUG
      Eigen::Vector3s recovered = R * up;
      s_t diff = (recovered - Eigen::Vector3s::UnitY()).squaredNorm();
      if (diff > 1e-16)
      {
        std::cout << "Bad R!" << std::endl;
      }
      // assert(diff < 1e-16);
#endif

      /*
      Eigen::Vector3s x = up.cross(Eigen::Vector3s::UnitY()).normalized();
      Eigen::Vector3s z = x.cross(up).normalized();
      R.row(0) = x;
      R.row(1) = up;
      R.row(2) = z;
      */
    }

    // Now go through and rotate everything by R
    for (int i = 0; i < result.forcePlates.size(); i++)
    {
      result.forcePlates[i].worldOrigin = R * result.forcePlates[i].worldOrigin;
      for (int j = 0; j < result.forcePlates[i].corners.size(); j++)
      {
        result.forcePlates[i].corners[j] = R * result.forcePlates[i].corners[j];
      }
      for (int t = 0; t < result.forcePlates[i].forces.size(); t++)
      {
        result.forcePlates[i].forces[t] = R * result.forcePlates[i].forces[t];
        result.forcePlates[i].centersOfPressure[t]
            = R * result.forcePlates[i].centersOfPressure[t];
        result.forcePlates[i].moments[t] = R * result.forcePlates[i].moments[t];
      }
    }
    for (int t = 0; t < result.markerTimesteps.size(); t++)
    {
      for (auto& pair : result.markerTimesteps[t])
      {
        pair.second = R * pair.second;
      }
    }
  }

  // These are useful for faster access to the pre-random-order marker data in
  // certain situations, for example in neural models
  result.shuffledMarkersMatrix = Eigen::MatrixXs::Zero(
      result.markers.size() * 3, result.markerTimesteps.size());
  result.shuffledMarkersMatrixMask = Eigen::MatrixXs::Zero(
      result.markers.size() * 3, result.markerTimesteps.size());

  std::vector<std::string> markerNames = result.markers;
  auto rng = std::default_random_engine();

  for (int t = 0; t < result.markerTimesteps.size(); t++)
  {
    int counter = 0;

    std::shuffle(std::begin(markerNames), std::end(markerNames), rng);

    for (std::string name : markerNames)
    {
      if (result.markerTimesteps[t].count(name) > 0)
      {
        result.shuffledMarkersMatrix.block(counter * 3, t, 3, 1)
            = result.markerTimesteps[t][name];
        result.shuffledMarkersMatrixMask.block<3, 1>(counter * 3, t)
            .setConstant(1.0);
        counter++;
      }
    }
  }

  return result;
}

//==============================================================================
/// This will check if markers
/// obviously "flip" during the trajectory, and unflip them.
void C3DLoader::fixupMarkerFlips(C3D* c3d)
{
  for (int i = 1; i < c3d->markerTimesteps.size(); i++)
  {
    std::unordered_map<std::string, std::string> closestMarkerFromLastTimestep;

    for (std::string& marker : c3d->markers)
    {
      closestMarkerFromLastTimestep[marker] = marker;

      // If we see the marker on both timesteps, then evaluate which markers
      // were the closest on last timestep to this marker
      if (c3d->markerTimesteps[i].count(marker) > 0
          && c3d->markerTimesteps[i - 1].count(marker) > 0)
      {
        Eigen::Vector3s thisTimestep = c3d->markerTimesteps[i].at(marker);
        for (auto& pair : c3d->markerTimesteps[i - 1])
        {
          s_t dist = (pair.second - thisTimestep).norm();
          if (dist < (c3d->markerTimesteps[i - 1].at(
                          closestMarkerFromLastTimestep.at(marker))
                      - thisTimestep)
                         .norm())
          {
            closestMarkerFromLastTimestep[marker] = pair.first;
          }
        }
      }
    }

    for (std::string& marker : c3d->markers)
    {
      // If we weren't closest to ourselves, and instead we were closest to
      // another marker AND IT WAS CLOSEST TO US, then we've detected a trivial
      // flip, and we can flip back.
      if (closestMarkerFromLastTimestep[marker] != marker
          && closestMarkerFromLastTimestep
                     [closestMarkerFromLastTimestep[marker]]
                 == marker)
      {
        Eigen::Vector3s tmp = c3d->markerTimesteps[i][marker];
        std::string otherMarker = closestMarkerFromLastTimestep[marker];

        c3d->markerTimesteps[i][marker] = c3d->markerTimesteps[i][otherMarker];
        c3d->markerTimesteps[i][otherMarker] = tmp;
        closestMarkerFromLastTimestep[marker] = marker;
        closestMarkerFromLastTimestep[otherMarker] = otherMarker;
      }
    }
  }
}

//==============================================================================
void C3DLoader::debugToGUI(
    C3D& file, std::shared_ptr<server::GUIWebsocketServer> server)
{
  // Render the plates as red rectangles
  for (int i = 0; i < file.forcePlates.size(); i++)
  {
    std::vector<Eigen::Vector3s> points;
    for (int j = 0; j < file.forcePlates[i].corners.size(); j++)
    {
      points.push_back(file.forcePlates[i].corners[j]);
    }
    points.push_back(file.forcePlates[i].corners[0]);

    server->createLine(
        "plate_" + std::to_string(i),
        points,
        Eigen::Vector4s(1.0, 0., 0., 1.0));

    server->createSphere(
        "plate_" + std::to_string(i) + "_0",
        0.015,
        file.forcePlates[i].corners[0],
        Eigen::Vector4s(1.0, 0.0, 0.0, 1.0));
    server->createSphere(
        "plate_" + std::to_string(i) + "_1",
        0.015,
        file.forcePlates[i].corners[1],
        Eigen::Vector4s(0.0, 1.0, 0.0, 1.0));
    server->createSphere(
        "plate_" + std::to_string(i) + "_2",
        0.015,
        file.forcePlates[i].corners[2],
        Eigen::Vector4s(0.0, 0.0, 1.0, 1.0));
    server->createSphere(
        "plate_" + std::to_string(i) + "_origin",
        0.05,
        file.forcePlates[i].worldOrigin,
        Eigen::Vector4s(0.5, 0.5, 0.5, 1.0));
  }

  // Create spheres for the markers
  for (int i = 0; i < file.markers.size(); i++)
  {
    server->createSphere(
        "marker_" + std::to_string(i),
        0.015,
        Eigen::Vector3s::Zero(),
        Eigen::Vector4s(1.0, 0.7, 0.0, 1.0));
  }

  // Render the markers over time
  int timestep = 0;
  std::shared_ptr<realtime::Ticker> ticker
      = std::make_shared<realtime::Ticker>(1.0 / 50);
  ticker->registerTickListener([&](long) {
    for (int i = 0; i < file.markers.size(); i++)
    {
      server->setObjectPosition(
          "marker_" + std::to_string(i),
          file.markerTimesteps[timestep][file.markers[i]]);
    }

    for (int i = 0; i < file.forcePlates.size(); i++)
    {
      server->deleteObject("force_" + std::to_string(i));
      if (file.forcePlates[i].forces[timestep].squaredNorm() > 0)
      {
        std::vector<Eigen::Vector3s> forcePoints;
        forcePoints.push_back(file.forcePlates[i].centersOfPressure[timestep]);
        forcePoints.push_back(
            file.forcePlates[i].centersOfPressure[timestep]
            + (file.forcePlates[i].forces[timestep] * 0.001));
        server->createLine(
            "force_" + std::to_string(i),
            forcePoints,
            Eigen::Vector4s(1.0, 0, 0, 1.));
      }
    }

    timestep++;
    if (timestep >= file.markerTimesteps.size())
    {
      timestep = 0;
    }
  });

  server->registerConnectionListener([ticker]() { ticker->start(); });
  // TODO: it'd be nice if this method didn't block forever, but we need to hold
  // onto a bunch of resources otherwise
  server->blockWhileServing();
}

} // namespace biomechanics
} // namespace dart