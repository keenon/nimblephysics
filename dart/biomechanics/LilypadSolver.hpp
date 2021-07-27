#ifndef DART_NEURAL_LILYPAD_HPP_
#define DART_NEURAL_LILYPAD_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

struct LilypadCell
{
public:
  LilypadCell();

public:
  int x;
  int y;

  s_t groundLowerBound;
  s_t groundUpperBound;

  std::vector<dynamics::BodyNode::MovingVertex> mSlowVerts;
  std::vector<dynamics::BodyNode::MovingVertex> mFastVerts;
};

class LilypadSolver
{
public:
  LilypadSolver(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      std::vector<const dynamics::BodyNode*> groundContactBodies,
      Eigen::Vector3s groundNormal,
      s_t tileSize);

  /// Get the body nodes that are in contact with the ground, as we currently
  /// understand the ground level
  std::vector<const dynamics::BodyNode*> getContactBodies();

  /// Here we can set the velocity threshold that distinguishes "slow" vertices
  /// from "fast" vertices. Only slow vertices can form the basis of lilypads.
  void setVerticalVelThreshold(s_t threshold);

  void setLateralVelThreshold(s_t threshold);

  void setVerticalAccelerationThreshold(s_t threshold);

  /// This threshold is expressed as a percentage. If we take the percentage
  /// distance, 0.0 being the lowest vertex on a body, and 1.0 being the highest
  /// vertex, this threshold throws out vertices that are higher than the bottom
  /// section.
  void setBottomThresholdPercentage(s_t threshold);

  /// This will attempt to find the lilypads in the supplied pose data
  void process(Eigen::MatrixXs poses, int startTime = 0);

  /// This returns the appropriate cell for a given position.
  LilypadCell& getCell(Eigen::Vector3s pos);

  /// This will debug all the processed data over to our GUI, so we can see the
  /// vertices and patterns that the solver is using.
  void debugToGUI(std::shared_ptr<server::GUIWebsocketServer> server);

  /// This clears the Lilypad storage, reducing the amount of memory we can
  /// leak.
  void clear();

protected:
  s_t lilypadRadius;
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  std::vector<const dynamics::BodyNode*> mBodies;

  /// These complete a basis of the space
  Eigen::Vector3s mGroundNormal;
  Eigen::Vector3s mXNormal;
  Eigen::Vector3s mYNormal;

  /// This gives the width and length of each tile of our space.
  s_t mTileSize;

  /// Anything below this velocity is considered "slow" and can be the basis for
  /// a lilypad
  s_t mVerticalVelThreshold;
  /// Anything below this velocity is considered "slow" and can be the basis for
  /// a lilypad
  s_t mLateralVelThreshold;
  /// Anything above this acceleration can be the basis for a lilypad
  s_t mVerticalAccelerationThreshold;

  /// This threshold is expressed as a percentage. If we take the percentage
  /// distance, 0.0 being the lowest vertex on a body, and 1.0 being the highest
  /// vertex, this threshold throws out vertices that are higher than the bottom
  /// section.
  s_t mBottomThresholdPercentage;

  std::map<std::pair<int, int>, LilypadCell> mPads;
};

} // namespace biomechanics

} // namespace dart

#endif