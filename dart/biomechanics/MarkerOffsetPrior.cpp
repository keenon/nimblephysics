#include "dart/biomechanics/MarkerOffsetPrior.hpp"

#include <vector>

#include "dart/math/Geometry.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

//==============================================================================
MarkerMovementSpace::MarkerMovementSpace(
    const std::string& markerName,
    dynamics::BodyNode* bodyNode,
    Eigen::Vector3s markerOffset)
  : mMarkerOffset(markerOffset), mMarkerName(markerName)
{
  // 1. Collect the joint anchors, in (scale invariant) body space
  std::vector<Eigen::Vector3s> potentialJointAnchors;
  potentialJointAnchors.push_back(bodyNode->getParentJoint()
                                      ->getTransformFromChildBodyNode()
                                      .translation());
  for (int i = 0; i < bodyNode->getNumChildJoints(); i++)
  {
    potentialJointAnchors.push_back(bodyNode->getChildJoint(i)
                                        ->getTransformFromParentBodyNode()
                                        .translation());
  }

  // 2. Find the nearest and furthest away joint anchor
  Eigen::Vector3s closestJointAnchor = potentialJointAnchors[0];
  s_t distanceToClosestJointAnchor = (markerOffset - closestJointAnchor).norm();
  Eigen::Vector3s furthestJointAnchor = potentialJointAnchors[0];
  s_t distanceToFurthestJointAnchor
      = (markerOffset - furthestJointAnchor).norm();

  for (int i = 0; i < potentialJointAnchors.size(); i++)
  {
    s_t dist = (markerOffset - potentialJointAnchors[i]).norm();
    if (dist < distanceToClosestJointAnchor)
    {
      distanceToClosestJointAnchor = dist;
      closestJointAnchor = potentialJointAnchors[i];
    }
    if (dist > distanceToFurthestJointAnchor)
    {
      distanceToFurthestJointAnchor = dist;
      furthestJointAnchor = potentialJointAnchors[i];
    }
  }

  s_t distanceRatio
      = distanceToClosestJointAnchor / distanceToFurthestJointAnchor;

  // 3. Check if we're basically inside a joint. If so, odds are this is a
  // virtual "joint center" marker (which seems to show up a lot in processed
  // Vicon data, for example). These markers can then be ignored.
  if (distanceToClosestJointAnchor < 1e-9)
  {
    mMarkerType = VIRTUAL;
  }

  // 4. Check the ratio of joint anchor distances to decide whether this is a
  // joint anchor or not
  else if (distanceRatio < 0.22)
  {
    mMarkerType = JOINT_LANDMARK;
    mAnchorPoint = closestJointAnchor;
  }

  // 5. This means we're not a joint anchor, so we need to instead find the
  // approximate surface plane to stick to, if one exists
  else
  {
    mMarkerType = TRACKING;
  }
}

//==============================================================================
/// This renders out the marker to a GUI
void MarkerMovementSpace::debugToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server,
    const Eigen::Isometry3s& bodyTransform,
    const Eigen::Vector3s& bodyScale,
    Eigen::Vector4s color)
{
  (void)bodyScale;

  if (mMarkerType == JOINT_LANDMARK || mMarkerType == BODY_LANDMARK)
  {
    std::vector<Eigen::Vector3s> points;
    points.push_back(bodyTransform * mAnchorPoint);
    points.push_back(bodyTransform * mMarkerOffset);
    server->createLine(mMarkerName + "_parentBody", points, color);
  }
  if (mMarkerType == VIRTUAL)
  {
    color(3) = 0.1;
  }
  if (mMarkerType == TRACKING)
  {
    color(3) = 0.5;
  }

  server->createSphere(mMarkerName, 0.01, bodyTransform * mMarkerOffset, color);
}

//==============================================================================
MarkerOffsetPrior::MarkerOffsetPrior(
    std::shared_ptr<dynamics::Skeleton> skeleton,
    dynamics::MarkerMap markersMap)
  : mSkeleton(skeleton), mMarkersMap(markersMap)
{
  for (auto& pair : markersMap)
  {
    mMarkerMovementSpaces.emplace(
        pair.first,
        MarkerMovementSpace(pair.first, pair.second.first, pair.second.second));
  }
};

//==============================================================================
/// This renders out the skeleton to the GUI, along with shapes representing
/// the various inferred skin surfaces that the marker offset prior uses
void MarkerOffsetPrior::debugToGUI(
    std::shared_ptr<server::GUIWebsocketServer> server)
{
  // 0. Create an array of colors we can use to distinguish between body nodes
  std::vector<Eigen::Vector4s> colors;
  colors.push_back(Eigen::Vector4s(1, 0, 0, 1));
  colors.push_back(Eigen::Vector4s(0, 1, 0, 1));
  colors.push_back(Eigen::Vector4s(0, 0, 1, 1));
  colors.push_back(Eigen::Vector4s(1, 1, 0, 1));
  colors.push_back(Eigen::Vector4s(1, 0, 1, 1));
  colors.push_back(Eigen::Vector4s(0, 1, 1, 1));

  // 1. Render skeleton bodies with appropriate colors
  for (int i = 0; i < mSkeleton->getNumBodyNodes(); i++)
  {
    Eigen::Vector4s color = colors.at(i % colors.size());
    auto* body = mSkeleton->getBodyNode(i);

    // Render the meshes
    for (int k = 0; k < body->getNumShapeNodes(); k++)
    {
      dynamics::ShapeNode* shapeNode = body->getShapeNode(k);
      dynamics::Shape* shape = shapeNode->getShape().get();

      std::stringstream shapeNameStream;
      shapeNameStream << mSkeleton->getName();
      shapeNameStream << "_";
      shapeNameStream << body->getName();
      shapeNameStream << "_";
      shapeNameStream << k;
      std::string shapeName = shapeNameStream.str();

      if (!shapeNode->hasVisualAspect())
        continue;

      if (shape == nullptr)
      {
        dtwarn << "Found a ShapeNode with no attached Shape object. This can "
                  "sometimes happen if you're trying to load models with "
                  "corrupted mesh files. Ignoring: "
               << shapeName << std::endl;
        continue;
      }

      dynamics::VisualAspect* visual = shapeNode->getVisualAspect(true);
      if (visual == nullptr)
        continue;

      if (!visual->isHidden())
      {
        if (shape->getType() == "MeshShape")
        {
          dynamics::MeshShape* meshShape
              = dynamic_cast<dynamics::MeshShape*>(shape);
          server->createMeshASSIMP(
              shapeName,
              meshShape->getMesh(),
              meshShape->getMeshPath(),
              shapeNode->getWorldTransform().translation(),
              math::matrixToEulerXYZ(shapeNode->getWorldTransform().linear()),
              meshShape->getScale(),
              color,
              "",
              visual->getCastShadows(),
              visual->getReceiveShadows());
        }
      }
    }
  }

  // 2. Render each marker's movement space
  for (auto pair : mMarkerMovementSpaces)
  {
    auto* bodyNode = mMarkersMap[pair.first].first;
    int index = bodyNode->getIndexInSkeleton();
    Eigen::Vector4s color = colors.at(index % colors.size());

    pair.second.debugToGUI(
        server, bodyNode->getWorldTransform(), bodyNode->getScale(), color);
  }
};

} // namespace biomechanics
} // namespace dart