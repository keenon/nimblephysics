#include "dart/biomechanics/LilypadSolver.hpp"

#include <assimp/scene.h>
#include <math.h>

#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/Shape.hpp"
#include "dart/dynamics/Skeleton.hpp"

namespace dart {

using namespace dynamics;

namespace biomechanics {

LilypadCell::LilypadCell()
  : groundLowerBound(std::numeric_limits<s_t>::infinity()),
    groundUpperBound(-std::numeric_limits<s_t>::infinity())
{
}

LilypadSolver::LilypadSolver(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    std::vector<const dynamics::BodyNode*> groundContactBodies,
    Eigen::Vector3s groundNormal,
    s_t tileSize)
  : mSkeleton(skeleton),
    mBodies(groundContactBodies),
    mGroundNormal(groundNormal),
    mVerticalVelThreshold(0.2),
    mLateralVelThreshold(0.2),
    mVerticalAccelerationThreshold(0.0),
    mBottomThresholdPercentage(0.1),
    mTileSize(tileSize)
{
  Eigen::Vector3s up = Eigen::Vector3s::UnitZ();
  mXNormal = groundNormal.cross(up);
  if (mXNormal.norm() < 0.01)
  {
    up = Eigen::Vector3s::UnitX();
    mXNormal = groundNormal.cross(up);
  }
  mXNormal.normalize();
  mYNormal = mXNormal.cross(groundNormal).normalized();
};

/// Get the body nodes that are in contact with the ground, as we currently
/// understand the ground level
std::vector<const dynamics::BodyNode*> LilypadSolver::getContactBodies()
{
  std::vector<const dynamics::BodyNode*> bodies;
  for (const dynamics::BodyNode* body : mBodies)
  {
    std::vector<BodyNode::MovingVertex> vertices
        = body->getMovingVerticesInWorldSpace();
    for (BodyNode::MovingVertex& vert : vertices)
    {
      LilypadCell& cell = getCell(vert.pos);
      s_t vertHeight = mGroundNormal.dot(vert.pos);

      s_t verticalVel = vert.vel.dot(mGroundNormal);
      Eigen::Vector3s lateralVelVector = vert.vel - mGroundNormal * verticalVel;
      s_t lateralVel = lateralVelVector.norm();

      if (cell.groundLowerBound < vertHeight
          && cell.groundUpperBound > vertHeight
          && std::abs(verticalVel) < mVerticalVelThreshold
          && lateralVel < mLateralVelThreshold)
      {
        bodies.push_back(body);
        break;
      }
    }
  }
  return bodies;
};

/// This will attempt to find the lilypads in the pose data
void LilypadSolver::process(Eigen::MatrixXs poses, int startTime)
{
  Eigen::VectorXs originalPos = mSkeleton->getPositions();
  Eigen::VectorXs originalVel = mSkeleton->getVelocities();
  Eigen::VectorXs originalControlForces = mSkeleton->getControlForces();

  // s_t squaredVelThreshold = mVelThreshold * mVelThreshold;

  // Collect fast and slow vertices
  for (int i = 0; i < poses.cols() - 2; i++)
  {
    Eigen::VectorXs vel
        = mSkeleton->getPositionDifferences(poses.col(i + 1), poses.col(i))
          / mSkeleton->getTimeStep();
    Eigen::VectorXs vel2
        = mSkeleton->getPositionDifferences(poses.col(i + 2), poses.col(i + 1))
          / mSkeleton->getTimeStep();
    Eigen::VectorXs accel = mSkeleton->getVelocityDifferences(vel2, vel)
                            / mSkeleton->getTimeStep();
    mSkeleton->setPositions(poses.col(i));
    mSkeleton->setVelocities(vel);
    mSkeleton->setAccelerations(accel);
    for (const dynamics::BodyNode* body : mBodies)
    {
      std::vector<dynamics::BodyNode::MovingVertex> movingVerts
          = body->getMovingVerticesInWorldSpace(startTime + i);

      s_t top = -std::numeric_limits<s_t>::infinity();
      s_t bottom = std::numeric_limits<s_t>::infinity();
      for (dynamics::BodyNode::MovingVertex& vert : movingVerts)
      {
        s_t height = vert.pos.dot(mGroundNormal);
        if (height > top)
          top = height;
        if (height < bottom)
          bottom = height;
      }
      s_t bodyHeight = top - bottom;

      for (dynamics::BodyNode::MovingVertex& vert : movingVerts)
      {
        s_t height = vert.pos.dot(mGroundNormal);
        s_t heightPercentage = (height - bottom) / bodyHeight;
        if (heightPercentage > mBottomThresholdPercentage)
          continue;

        LilypadCell& cell = getCell(vert.pos);

        s_t verticalVel = vert.vel.dot(mGroundNormal);
        Eigen::Vector3s lateralVelVector
            = vert.vel - mGroundNormal * verticalVel;
        s_t lateralVel = lateralVelVector.norm();
        s_t verticalAccel = vert.accel.dot(mGroundNormal);
        if (std::abs(verticalVel) > mVerticalVelThreshold
            || lateralVel > mLateralVelThreshold
            || verticalAccel < mVerticalAccelerationThreshold)
        {
          cell.mFastVerts.push_back(vert);
          /*
          // If we're moving fast, and we're below what we thought the ground
          // level was, then we've obviously made a mistake and the ground isn't
          // where we thought it was
          if (height < cell.groundLowerBound)
          {
            cell.groundLowerBound = std::numeric_limits<s_t>::infinity();
            cell.groundUpperBound = -std::numeric_limits<s_t>::infinity();
          }
          */
        }
        else
        {
          cell.mSlowVerts.push_back(vert);
          if (cell.groundLowerBound > height)
          {
            cell.groundLowerBound = height;
          }
          // (height + bodyHeight) is an upper bound on groundUpperBound
          if (cell.groundUpperBound > height + bodyHeight)
          {
            cell.groundUpperBound = height + bodyHeight;
          }

          if (cell.groundUpperBound < height)
          {
            cell.groundUpperBound = height;
          }
        }
      }
    }
  }

  // For each slow vertex, check if it's a lower bound on its neighbors

  mSkeleton->setPositions(originalPos);
  mSkeleton->setVelocities(originalVel);
  mSkeleton->setControlForces(originalControlForces);
};

/// Here we can set the velocity threshold that distinguishes "slow" vertices
/// from "fast" vertices. Only slow vertices can form the basis of lilypads.
void LilypadSolver::setVerticalVelThreshold(s_t threshold)
{
  mVerticalVelThreshold = threshold;
}

void LilypadSolver::setLateralVelThreshold(s_t threshold)
{
  mLateralVelThreshold = threshold;
}

void LilypadSolver::setVerticalAccelerationThreshold(s_t threshold)
{
  mVerticalAccelerationThreshold = threshold;
}

/// This threshold is expressed as a percentage. If we take the percentage
/// distance, 0.0 being the lowest vertex on a body, and 1.0 being the highest
/// vertex, this threshold throws out vertices that are higher than the bottom
/// section.
void LilypadSolver::setBottomThresholdPercentage(s_t threshold)
{
  mBottomThresholdPercentage = threshold;
}

/// This returns the appropriate cell for a given position.
LilypadCell& LilypadSolver::getCell(Eigen::Vector3s pos)
{
  s_t xPos = pos.dot(mXNormal);
  s_t yPos = pos.dot(mYNormal);

  int x = (int)ceil(xPos / mTileSize);
  int y = (int)ceil(yPos / mTileSize);

  LilypadCell& cell = mPads[std::make_pair(x, y)];
  cell.x = x;
  cell.y = y;

  return cell;
}

/// This will debug all the processed data over to our GUI, so we can see the
/// vertices and patterns that the solver is using.
void LilypadSolver::debugToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server)
{
  bool oldAutoflush = server->getAutoflush();
  server->setAutoflush(false);

  server->deleteObjectsByPrefix("lilypad_tile_");

  for (auto& pair : mPads)
  {
    LilypadCell& cell = pair.second;

    /*
    for (int i = 0; i < cell.mSlowVerts.size(); i++)
    {
      std::vector<Eigen::Vector3s> line;
      line.push_back(cell.mSlowVerts[i].pos);
      line.push_back(cell.mSlowVerts[i].pos + 0.01 * cell.mSlowVerts[i].vel);
      server->createLine(
          "slow_verts_(" + std::to_string(pair.first.first) + ","
              + std::to_string(pair.first.second) + ")_" + std::to_string(i),
          line,
          Eigen::Vector3s(0.5, 0.5, 1.0));
    }
    for (int i = 0; i < cell.mFastVerts.size(); i++)
    {
      std::vector<Eigen::Vector3s> line;
      line.push_back(cell.mFastVerts[i].pos);
      line.push_back(cell.mFastVerts[i].pos + 0.01 * cell.mFastVerts[i].vel);
      server->createLine(
          "fast_verts_(" + std::to_string(pair.first.first) + ","
              + std::to_string(pair.first.second) + ")_" + std::to_string(i),
          line,
          Eigen::Vector3s(1.0, 0.5, 0.5));
    }
    */

    if (isinf(cell.groundLowerBound) || isinf(cell.groundUpperBound))
    {
      continue;
    }

    Eigen::Vector3s size = Eigen::Vector3s(
        mTileSize, cell.groundUpperBound - cell.groundLowerBound, mTileSize);
    Eigen::Vector3s pos = Eigen::Vector3s(
        mTileSize * cell.x - mTileSize / 2,
        cell.groundLowerBound
            + (cell.groundUpperBound - cell.groundLowerBound) / 2,
        mTileSize * cell.y - mTileSize / 2);
    Eigen::Vector3s euler = Eigen::Vector3s(0, 0, 0);
    Eigen::Vector3s color = Eigen::Vector3s(0.5, 0.5, 0.5);

    server->createBox(
        "lilypad_tile_" + std::to_string(cell.x) + "," + std::to_string(cell.y),
        size,
        pos,
        euler,
        color,
        false,
        true);
  }

  server->flush();
  server->setAutoflush(oldAutoflush);
};

void LilypadSolver::clear()
{
  mPads.clear();
}

}; // namespace biomechanics
}; // namespace dart