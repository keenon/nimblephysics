#ifndef DART_BIOMECH_MARKEROFFSETPRIOR_HPP_
#define DART_BIOMECH_MARKEROFFSETPRIOR_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <mutex>
#include <vector>

#include <Eigen/Dense>

#include "dart/biomechanics/Anthropometrics.hpp"
#include "dart/biomechanics/C3DLoader.hpp"
#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

enum MarkerType
{
  // Tracking markers / triads: move anywhere at roughly similar distance from
  // the bone vertices
  TRACKING = 0,
  // Joint landmarks (if you're near a joint center): stay near the joint
  // center, and stay at a similar angle
  JOINT_LANDMARK = 1,
  // Body landmarks are pointy bone segments that are probably easy to locate on
  // a person, so we should assume lower error for these
  BODY_LANDMARK = 2,
  // Inferred joint centers: ignore these, they're artificial
  VIRTUAL = 3
};

class MarkerMovementSpace
{
public:
  MarkerMovementSpace(
      const std::string& markerName,
      dynamics::BodyNode* bodyNode,
      Eigen::Vector3s markerOffset);

  /// This renders out the marker to a GUI
  void debugToGUI(
      std::shared_ptr<server::GUIWebsocketServer> server,
      const Eigen::Isometry3s& bodyTransform,
      const Eigen::Vector3s& bodyScale,
      Eigen::Vector4s color);

  std::string mMarkerName;
  MarkerType mMarkerType;
  Eigen::Vector3s mAnchorPoint;
  Eigen::Vector3s mMarkerOffset;
};

class MarkerOffsetPrior
{
public:
  MarkerOffsetPrior(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      dynamics::MarkerMap markersMap);

  /// This renders out the skeleton to the GUI, along with shapes representing
  /// the various inferred skin surfaces that the marker offset prior uses
  void debugToGUI(std::shared_ptr<server::GUIWebsocketServer> server);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  dynamics::MarkerMap mMarkersMap;
  std::map<std::string, MarkerMovementSpace> mMarkerMovementSpaces;
};

}; // namespace biomechanics
}; // namespace dart
#endif