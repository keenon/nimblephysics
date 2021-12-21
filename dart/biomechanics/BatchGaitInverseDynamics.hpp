#ifndef DART_BIOMECH_BATCH_ID_HPP_
#define DART_BIOMECH_BATCH_ID_HPP_

#include <memory>
// #include <unordered_map>
#include <map>
#include <vector>

#include "dart/include_eigen.hpp"

#include "dart/biomechanics/LilypadSolver.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

namespace dart {

namespace biomechanics {

struct ContactRegimeSection
{
  std::vector<const dynamics::BodyNode*> groundContactBodies;
  int startTime;
  int endTime;
  std::vector<std::vector<Eigen::Vector6s>> wrenches;

  ContactRegimeSection(
      std::vector<const dynamics::BodyNode*> groundContactBodies,
      int startTime,
      int endTime);
};

class BatchGaitInverseDynamics
{
public:
  /// This will attempt to create the best possible trajectory. The result of
  /// the computation will be stored internally in the object.
  BatchGaitInverseDynamics(
      std::shared_ptr<dynamics::Skeleton> skeleton,
      Eigen::MatrixXs poses,
      std::vector<const dynamics::BodyNode*> groundContactBodies,
      Eigen::Vector3s groundNormal,
      s_t tileSize,
      int maxSectionLength = 100,
      s_t smoothingWeight = 1.0,
      s_t minTorqueWeight = 1.0,
      s_t prevContactWeight = 0.1,
      s_t blendWeight = 1.0,
      s_t blendSteepness = 10.0);

  int numTimesteps();

  ContactRegimeSection& getSectionForTimestep(int timestep);

  std::vector<const dynamics::BodyNode*> getContactBodiesAtTimestep(
      int timestep);

  std::vector<Eigen::Vector6s> getContactWrenchesAtTimestep(int timestep);

  /// This will debug all the processed data over to our GUI, so we can see the
  /// contact forces and positions animated
  void debugLilypadToGUI(std::shared_ptr<server::GUIWebsocketServer> server);

  /// This will debug all the processed data over to our GUI, so we can see the
  /// contact forces and positions animated
  void debugTimestepToGUI(
      std::shared_ptr<server::GUIWebsocketServer> server, int timestep);

protected:
  std::shared_ptr<dynamics::Skeleton> mSkeleton;
  Eigen::MatrixXs mPoses;
  std::vector<const dynamics::BodyNode*> mBodies;
  std::vector<ContactRegimeSection> mContactRegimeSections;
  Eigen::Vector3s mGroundNormal;
  s_t mTileSize;
  LilypadSolver mLilypad;
};

} // namespace biomechanics
} // namespace dart

#endif