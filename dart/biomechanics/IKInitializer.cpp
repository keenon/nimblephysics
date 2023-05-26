#include "dart/biomechanics/IKInitializer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <unsupported/Eigen/Polynomials>

#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/EulerFreeJoint.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/dynamics/UniversalJoint.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"

namespace dart {
namespace biomechanics {

//==============================================================================
/// This is a helper struct that is used to simplify the code in
/// estimatePosesClosedForm()
typedef struct AnnotatedStackedBody
{
  std::shared_ptr<struct StackedBody> stackedBody;
  std::vector<Eigen::Isometry3s> relativeTransforms;
  std::vector<std::shared_ptr<struct StackedJoint>> adjacentJoints;
  std::vector<Eigen::Vector3s> adjacentJointCenters;
  std::vector<std::string> adjacentMarkers;
  std::vector<Eigen::Vector3s> adjacentMarkerCenters;
} AnnotatedStackedBody;

//==============================================================================
Eigen::Vector3s getStackedJointCenterFromJointCentersVector(
    std::shared_ptr<struct StackedJoint> joint,
    Eigen::VectorXs jointWorldCenters)
{
  Eigen::Vector3s jointWorldCenter = Eigen::Vector3s::Zero();
  for (dynamics::Joint* joint : joint->joints)
  {
    jointWorldCenter
        += jointWorldCenters.segment<3>(joint->getJointIndexInSkeleton() * 3);
  }
  jointWorldCenter /= joint->joints.size();
  return jointWorldCenter;
}

//==============================================================================
/// This returns true if the given body is the parent of the joint OR if
/// there's a hierarchy of fixed joints that connect it to the parent
bool isDynamicParentOfJoint(std::string bodyName, dynamics::Joint* joint)
{
  while (true)
  {
    if (joint->getParentBodyNode() == nullptr)
      return false;
    if (bodyName == joint->getParentBodyNode()->getName())
    {
      return true;
    }
    // Recurse up the chain, as long as we're traversing only fixed joints
    if (joint->getParentBodyNode()->getParentJoint() != nullptr
        && joint->getParentBodyNode()->getParentJoint()->isFixed())
    {
      joint = joint->getParentBodyNode()->getParentJoint();
    }
    else
    {
      return false;
    }
  }
}

//==============================================================================
/// This returns true if the given body is the child of the joint OR if
/// there's a hierarchy of fixed joints that connect it to the child
bool isDynamicChildOfJoint(std::string bodyName, dynamics::Joint* joint)
{
  while (true)
  {
    if (joint->getChildBodyNode() == nullptr)
      return false;
    if (bodyName == joint->getChildBodyNode()->getName())
    {
      return true;
    }
    // Recurse down the chain, as long as we're traversing only fixed joints
    if (joint->getChildBodyNode()->getNumChildJoints() == 1
        && joint->getChildBodyNode()->getChildJoint(0)->isFixed())
    {
      joint = joint->getChildBodyNode()->getChildJoint(0);
    }
    else
    {
      return false;
    }
  }
}

//==============================================================================
bool isCoplanar(const std::vector<Eigen::Vector3s>& points)
{
  // Ensure there are at least 4 points
  if (points.size() < 4)
  {
    return true;
  }

  // Choose the first three points and form two vectors
  const Eigen::Vector3s& p1 = points[0];
  const Eigen::Vector3s& p2 = points[1];
  const Eigen::Vector3s& p3 = points[2];
  const Eigen::Vector3s v1 = p2 - p1;
  const Eigen::Vector3s v2 = p3 - p1;

  // Calculate the cross-product of the two vectors to get a normal vector
  const Eigen::Vector3s normal = v1.cross(v2).normalized();

  // Check if all remaining points are coplanar with the first three points
  for (size_t i = 3; i < points.size(); ++i)
  {
    // Form a vector from one of the original three points to the current point
    const Eigen::Vector3s& point = points[i];
    const Eigen::Vector3s v = point - p1;

    // Calculate the dot product of the vector with the normal vector
    const double dot_product = v.dot(normal);

    // Check if the dot product is greater zero, and if so we have found a
    // non-coplanar point
    const s_t threshold = 1e-3;
    if (std::abs(dot_product) > threshold)
    {
      return false;
    }
  }

  return true;
}

//==============================================================================
bool isPositiveDefinite(const Eigen::MatrixXd& A)
{
  if (A.rows() != A.cols())
    return false;

  // Check symmetry
  if (!A.isApprox(A.transpose()))
    return false;

  // Compute eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);

  // Check for success
  if (solver.info() != Eigen::Success)
    return false;

  // Check if all eigenvalues are positive
  if ((solver.eigenvalues().array() > 0).all())
    return true;
  else
  {
    std::cout << "Got non-positive eigenvalues: " << solver.eigenvalues()
              << std::endl;
    return false;
  }
}

// solve cubic equation x^3 + a*x^2 + b*x + c = 0
#define TwoPi 6.28318530717958648

static double _root3(double x)
{
  double s = 1.;
  while (x < 1.)
  {
    x *= 8.;
    s *= 0.5;
  }
  while (x > 8.)
  {
    x *= 0.125;
    s *= 2.;
  }
  double r = 1.5;
  r -= 1. / 3. * (r - x / (r * r));
  r -= 1. / 3. * (r - x / (r * r));
  r -= 1. / 3. * (r - x / (r * r));
  r -= 1. / 3. * (r - x / (r * r));
  r -= 1. / 3. * (r - x / (r * r));
  r -= 1. / 3. * (r - x / (r * r));
  return r * s;
}

double root3(double x)
{
  if (x > 0)
    return _root3(x);
  else if (x < 0)
    return -_root3(-x);
  else
    return 0.;
}

int SolveP3(double* x, double a, double b, double c)
{
  const double eps = 1e-14;
  double a2 = a * a;
  double q = (a2 - 3 * b) / 9;
  double r = (a * (2 * a2 - 9 * b) + 27 * c) / 54;
  // equation y^3 - 3q*y + r/2 = 0 where x = y-a/3
  if (fabs(q) < eps)
  { // y^3 =-r/2	!!! Thanks to John Fairman <jfairman1066@gmail.com>
    if (fabs(r) < eps)
    { // three identical roots
      x[0] = x[1] = x[2] = -a / 3;
      return (1);
    }
    // y^3 =-r/2
    x[0] = root3(-r / 2);
    x[1] = x[0] * 0.5;
    x[2] = x[0] * sqrt(3.) / 2;
    return (1);
  }
  // now favs(q)>eps
  double r2 = r * r;
  double q3 = q * q * q;
  double A, B;
  if (r2 <= (q3 + eps))
  { //<<-- FIXED!
    double t = r / sqrt(q3);
    if (t < -1)
      t = -1;
    if (t > 1)
      t = 1;
    t = acos(t);
    a /= 3;
    q = -2 * sqrt(q);
    x[0] = q * cos(t / 3) - a;
    x[1] = q * cos((t + TwoPi) / 3) - a;
    x[2] = q * cos((t - TwoPi) / 3) - a;
    return (3);
  }
  else
  {
    // A =-pow(fabs(r)+sqrt(r2-q3),1./3);
    A = -root3(fabs(r) + sqrt(r2 - q3));
    if (r < 0)
      A = -A;
    B = (A == 0 ? 0 : q / A);

    a /= 3;
    x[0] = (A + B) - a;
    x[1] = -0.5 * (A + B) - a;
    x[2] = 0.5 * sqrt(3.) * (A - B);
    if (fabs(x[2]) < eps)
    {
      x[2] = x[1];
      return (1);
    }
    return (1);
  }
}

//==============================================================================
Eigen::Vector3s ensureOnSameSideOfPlane(
    const std::vector<Eigen::Vector3s>& neutralPoints,
    Eigen::Vector3s neutralGoal,
    std::vector<Eigen::Vector3s> actualPoints,
    Eigen::Vector3s ambiguousReconstructionFromActualPoints)
{
  if (neutralPoints.size() < 3 || actualPoints.size() < 3)
  {
    return ambiguousReconstructionFromActualPoints;
  }

  // Choose the first three points and form two vectors
  const Eigen::Vector3s neutralNormal
      = ((neutralPoints[1] - neutralPoints[0])
             .cross(neutralPoints[2] - neutralPoints[0]))
            .normalized();
  s_t neutralGoalDistanceFromNormal
      = (neutralGoal - neutralPoints[0]).dot(neutralNormal);

  const Eigen::Vector3s actualNormal
      = ((actualPoints[1] - actualPoints[0])
             .cross(actualPoints[2] - actualPoints[0]))
            .normalized();
  s_t ambiguousReconstructionDistanceFromNormal
      = (ambiguousReconstructionFromActualPoints - actualPoints[0])
            .dot(actualNormal);

  if (neutralGoalDistanceFromNormal * ambiguousReconstructionDistanceFromNormal
      < 0)
  {
    // If they're on opposite sides of the plane, flip the `actualGoal` to be on
    // the same side of the plane as the `neutralGoal`.
    return ambiguousReconstructionFromActualPoints
           - 2 * ambiguousReconstructionDistanceFromNormal * actualNormal;
  }
  else
  {
    // If they're already on the same side of the plane, don't worry
    return ambiguousReconstructionFromActualPoints;
  }
}

//==============================================================================
IKInitializer::IKInitializer(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
        markers,
    std::map<std::string, bool> markerIsAnatomical,
    std::vector<std::map<std::string, Eigen::Vector3s>> markerObservations,
    std::vector<bool> newClip,
    s_t modelHeightM)
  : mSkel(skel),
    mMarkerObservations(markerObservations),
    mModelHeightM(modelHeightM),
    mNewClip(newClip)
{
  // 1. Convert the marker map to an ordered list
  for (auto& pair : markers)
  {
    mMarkerNameToIndex[pair.first] = mMarkers.size();
    mMarkerNames.push_back(pair.first);
    mMarkers.push_back(pair.second);
    if (markerIsAnatomical.count(pair.first))
    {
      mMarkerIsAnatomical.push_back(markerIsAnatomical[pair.first]);
    }
    else
    {
      mMarkerIsAnatomical.push_back(false);
    }
  }

  Eigen::VectorXs oldPositions = mSkel->getPositions();
  mSkel->setPositions(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
  Eigen::VectorXs oldScales = mSkel->getBodyScales();
  if (modelHeightM > 0)
  {
    mSkel->setBodyScales(Eigen::VectorXs::Ones(mSkel->getNumBodyNodes() * 3));
    s_t defaultHeight
        = mSkel->getHeight(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
    if (defaultHeight > 0)
    {
      s_t ratio = modelHeightM / defaultHeight;
      mSkel->setBodyScales(mSkel->getBodyScales() * ratio);
      s_t newHeight
          = mSkel->getHeight(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
      (void)newHeight;
      assert(abs(newHeight - modelHeightM) < 1e-6);
    }
  }

  // 2. Create the simplified skeleton
  Eigen::VectorXs jointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());

  // 2.1. Set up the initial stacked bodies and joints, with appropriate
  // pointers to each other
  mStackedJoints.clear();
  mStackedBodies.clear();
  for (int i = 0; i < mSkel->getNumBodyNodes(); i++)
  {
    mStackedBodies.push_back(std::make_shared<struct StackedBody>());
    mStackedBodies[i]->bodies.push_back(mSkel->getBodyNode(i));
    mStackedBodies[i]->name = mSkel->getBodyNode(i)->getName();
    // Count the number of markers on this body.
    mStackedBodies[i]->numMarkers = 0;
    for (auto& marker : mMarkers)
    {
      if (marker.first->getName() == mStackedBodies[i]->name)
      {
        mStackedBodies[i]->numMarkers += 1;
      }
    }
  }
  for (int i = 0; i < mSkel->getNumJoints(); i++)
  {
    mStackedJoints.push_back(std::make_shared<struct StackedJoint>());
    mStackedJoints[i]->joints.push_back(mSkel->getJoint(i));
    mStackedJoints[i]->name = mSkel->getJoint(i)->getName();
  }
  for (int i = 0; i < mSkel->getNumBodyNodes(); i++)
  {
    dynamics::BodyNode* body = mSkel->getBodyNode(i);
    assert(body->getParentJoint() != nullptr);
    mStackedBodies[i]->parentJoint
        = mStackedJoints[body->getParentJoint()->getJointIndexInSkeleton()];
    for (int j = 0; j < body->getNumChildJoints(); j++)
    {
      mStackedBodies[i]->childJoints.push_back(
          mStackedJoints[body->getChildJoint(j)->getJointIndexInSkeleton()]);
    }
  }
  for (int i = 0; i < mSkel->getNumJoints(); i++)
  {
    dynamics::Joint* joint = mSkel->getJoint(i);
    assert(joint->getChildBodyNode() != nullptr);
    mStackedJoints[i]->childBody
        = mStackedBodies[joint->getChildBodyNode()->getIndexInSkeleton()];
    if (joint->getParentBodyNode() == nullptr)
    {
      mStackedJoints[i]->parentBody = nullptr;
    }
    else
    {
      mStackedJoints[i]->parentBody
          = mStackedBodies[joint->getParentBodyNode()->getIndexInSkeleton()];
    }
  }
  // At this stage, we should have exactly the same number of "stacked" nodes as
  // original nodes, since no stacking has yet taken place.
  assert(mStackedBodies.size() == mSkel->getNumBodyNodes());
  assert(mStackedJoints.size() == mSkel->getNumJoints());

  // 2.2. Collapse "leaf" stacked bodies into their parent bodies if they have
  // fewer than 3 markers.
  while (true)
  {
    std::shared_ptr<struct StackedJoint> toCollapse = nullptr;

    // 2.2.1. Find a "leaf" body with fewer than 3 markers and mark the parent
    // joint to be collapsed.
    for (auto& mStackedBody : mStackedBodies)
    {
      if (mStackedBody->childJoints.empty())
      {
        if (mStackedBody->numMarkers < 3)
        {
          toCollapse = mStackedBody->parentJoint;
        }
      }
    }

    if (toCollapse != nullptr)
    {
      // 2.2.2. Merge the connected bodies
      std::shared_ptr<struct StackedBody> parentBody = toCollapse->parentBody;
      std::shared_ptr<struct StackedBody> childBody = toCollapse->childBody;
      std::cout << "Discovered leaf body \"" << childBody->name << "\" with "
                << "fewer than 3 markers." << std::endl;
      std::cout << " -> As a result merging bodies \"" << childBody->name
                << "\" into \"" << parentBody->name << "\"" << std::endl;
      parentBody->numMarkers += childBody->numMarkers;
      parentBody->bodies.insert(
          parentBody->bodies.end(),
          childBody->bodies.begin(),
          childBody->bodies.end());
      // Re-attach all the child joints to the parent body
      for (auto& childJoint : childBody->childJoints)
      {
        childJoint->parentBody = parentBody;
        parentBody->childJoints.push_back(childJoint);
      }

      // 2.2.3. Remove the joint and body from the lists
      auto jointToDeleteIterator
          = std::find(mStackedJoints.begin(), mStackedJoints.end(), toCollapse);
      assert(jointToDeleteIterator != mStackedJoints.end());
      mStackedJoints.erase(jointToDeleteIterator);
      auto bodyToDeleteIterator
          = std::find(mStackedBodies.begin(), mStackedBodies.end(), childBody);
      assert(bodyToDeleteIterator != mStackedBodies.end());
      mStackedBodies.erase(bodyToDeleteIterator);
      auto childJointToDeletIterator = std::find(
          parentBody->childJoints.begin(),
          parentBody->childJoints.end(),
          toCollapse);
      assert(jointToDeleteIterator != parentBody->childJoints.end());
      parentBody->childJoints.erase(childJointToDeletIterator);
    }
    else
    {
      break;
    }
  }

  // 2.3. Collapse the stacked bodies that are connected by a fixed joint, and
  // simply remove the fixed joint.

  while (true)
  {
    std::shared_ptr<struct StackedJoint> toCollapse = nullptr;

    for (int i = 0; i < mStackedJoints.size(); i++)
    {
      bool allFixed = true;
      for (auto* joint : mStackedJoints[i]->joints)
      {
        if (!joint->isFixed())
        {
          allFixed = false;
          break;
        }
      }
      if (allFixed)
      {
        toCollapse = mStackedJoints[i];
      }
    }

    if (toCollapse != nullptr)
    {
      // 2.3.1. Merge the connected bodies
      std::shared_ptr<struct StackedBody> parentBody = toCollapse->parentBody;
      std::shared_ptr<struct StackedBody> childBody = toCollapse->childBody;
      std::cout << "Discovered locked joint \"" << toCollapse->name << "\""
                << std::endl;
      std::cout << " -> As a result merging bodies \"" << childBody->name
                << "\" into \"" << parentBody->name << "\"" << std::endl;
      parentBody->bodies.insert(
          parentBody->bodies.end(),
          childBody->bodies.begin(),
          childBody->bodies.end());
      // Re-attach all the child joints to the parent body
      for (auto& childJoint : childBody->childJoints)
      {
        childJoint->parentBody = parentBody;
        parentBody->childJoints.push_back(childJoint);
      }

      // 2.3.2. Remove the joint and body from the lists
      auto jointToDeleteIterator
          = std::find(mStackedJoints.begin(), mStackedJoints.end(), toCollapse);
      assert(jointToDeleteIterator != mStackedJoints.end());
      mStackedJoints.erase(jointToDeleteIterator);
      auto bodyToDeleteIterator
          = std::find(mStackedBodies.begin(), mStackedBodies.end(), childBody);
      assert(bodyToDeleteIterator != mStackedBodies.end());
      mStackedBodies.erase(bodyToDeleteIterator);
      auto childJointToDeletIterator = std::find(
          parentBody->childJoints.begin(),
          parentBody->childJoints.end(),
          toCollapse);
      assert(jointToDeleteIterator != parentBody->childJoints.end());
      parentBody->childJoints.erase(childJointToDeletIterator);
    }
    else
    {
      break;
    }
  }

  // 3. Filter to just the stacked joints that are connected to at least three
  // markers, since none of our closed-form algorithms will work with less than
  // that, and the iterative algorithms don't use the stacked joints. While
  // we're at it, measure the distances between each joint center and the
  // adjacent markers.
  Eigen::VectorXs markerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);
  std::vector<std::shared_ptr<struct StackedJoint>> stackedJointsToRemove;
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    Eigen::Vector3s jointWorldCenter
        = getStackedJointCenterFromJointCentersVector(
            mStackedJoints[i], jointWorldPositions);

    std::map<std::string, s_t> jointToMarkerSquaredDistances;

    for (int j = 0; j < mMarkers.size(); j++)
    {
      assert(mStackedJoints[i]->childBody != nullptr);
      if (mStackedJoints[i]->parentBody != nullptr
          && std::find(
                 mStackedJoints[i]->parentBody->bodies.begin(),
                 mStackedJoints[i]->parentBody->bodies.end(),
                 mMarkers[j].first)
                 != mStackedJoints[i]->parentBody->bodies.end())
      {
        jointToMarkerSquaredDistances[mMarkerNames[j]]
            = (jointWorldCenter - markerWorldPositions.segment<3>(j * 3))
                  .squaredNorm();
      }
      else if (
          std::find(
              mStackedJoints[i]->childBody->bodies.begin(),
              mStackedJoints[i]->childBody->bodies.end(),
              mMarkers[j].first)
          != mStackedJoints[i]->childBody->bodies.end())
      {
        jointToMarkerSquaredDistances[mMarkerNames[j]]
            = (jointWorldCenter - markerWorldPositions.segment<3>(j * 3))
                  .squaredNorm();
      }
    }

    if (jointToMarkerSquaredDistances.size() >= 3)
    {
      mJointToMarkerSquaredDistances[mStackedJoints[i]->name]
          = jointToMarkerSquaredDistances;
    }
    else
    {
      stackedJointsToRemove.push_back(mStackedJoints[i]);
    }
  }
  for (int i = 0; i < stackedJointsToRemove.size(); i++)
  {
    mStackedJoints.erase(
        std::find(
            mStackedJoints.begin(),
            mStackedJoints.end(),
            stackedJointsToRemove[i]),
        mStackedJoints.end());
  }

  // 4. See if there are any connected pairs of joints, and if so measure the
  // distance between them on the skeleton.
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    std::shared_ptr<struct StackedJoint> joint1 = mStackedJoints[i];
    Eigen::Vector3s joint1WorldCenter
        = getStackedJointCenterFromJointCentersVector(
            joint1, jointWorldPositions);
    for (int j = i + 1; j < mStackedJoints.size(); j++)
    {
      std::shared_ptr<struct StackedJoint> joint2 = mStackedJoints[j];
      Eigen::Vector3s joint2WorldCenter
          = getStackedJointCenterFromJointCentersVector(
              joint2, jointWorldPositions);

      // 3.1. If the joints are connected to the same body, then record them
      if (joint1->parentBody == joint2->childBody
          || joint2->parentBody == joint1->childBody
          || joint1->parentBody == joint2->parentBody)
      {
        Eigen::Vector3s joint1ToJoint2 = joint2WorldCenter - joint1WorldCenter;
        s_t joint1ToJoint2SquaredDistance = joint1ToJoint2.squaredNorm();
        assert(joint1ToJoint2SquaredDistance > 0);

        mJointToJointSquaredDistances[joint1->name][joint2->name]
            = joint1ToJoint2SquaredDistance;
        mJointToJointSquaredDistances[joint2->name][joint1->name]
            = joint1ToJoint2SquaredDistance;
      }
    }
  }

  mSkel->setPositions(oldPositions);
  mSkel->setBodyScales(oldScales);
}

//==============================================================================
/// This runs the full IK initialization algorithm, and leaves the answers in
/// the public fields of this class
void IKInitializer::runFullPipeline(bool logOutput)
{
  prescaleBasedOnAnatomicalMarkers();

  // Use MDS, despite its many flaws, to arrive at decent initial guesses for
  // joint centers that we can use to center the subsequent least-squares fits
  // if a joint doesn't move very much during a trial (like if your arms are at
  // your side during a whole trial).
  closedFormMDSJointCenterSolver();
  // Use the pivot finding, where there is the huge wealth of marker information
  // (3+ markers on adjacent body segments) to make it possible
  closedFormPivotFindingJointCenterSolver();
  recenterAxisJointsBasedOnBoneAngles();

  // Fill in the parts of the body scales and poses that we can in closed form
  estimateGroupScalesClosedForm();
  // estimatePosesClosedForm(logOutput);
  estimatePosesWithIK(logOutput);
}

//==============================================================================
/// This takes advantage of the fact that we assume anatomical markers have
/// pretty much known locations on their body segment. That means that two or
/// more anatomical markers on the same body segment can tell you about the
/// body segment scaling, which can then help inform the distances between
/// those markers and the joints, which helps all the subsequent steps in the
/// pipeline be more accurate.
s_t IKInitializer::prescaleBasedOnAnatomicalMarkers(bool logOutput)
{
  (void)logOutput;

  Eigen::VectorXs jointWorldCenters
      = mSkel->getJointWorldPositions(mSkel->getJoints());
  Eigen::VectorXs markerWorldCenters = mSkel->getMarkerWorldPositions(mMarkers);

  // 0. Record the defaultScale value, based on height, if we have it
  Eigen::VectorXs originalBodyScales = mSkel->getBodyScales();
  mSkel->setBodyScales(Eigen::VectorXs::Ones(mSkel->getNumBodyNodes() * 3));
  s_t defaultScale = 1.0;
  if (mModelHeightM > 0)
  {
    s_t defaultHeight
        = mSkel->getHeight(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
    if (defaultHeight > 0)
    {
      defaultScale = mModelHeightM / defaultHeight;
    }
  }
  mSkel->setBodyScales(originalBodyScales);

  for (auto& stackedBody : mStackedBodies)
  {
    // 1. First, we're going to collect all the anatomical markers attached to
    // this `stackedBody`

    std::vector<std::string> anatomicalMarkerNames;
    std::vector<Eigen::Vector3s> anatomicalMarkerWorldPositions;
    std::vector<Eigen::Vector3s> anatomicalMarkerLocalPositions;
    for (int i = 0; i < mMarkers.size(); i++)
    {
      if (std::find(
              stackedBody->bodies.begin(),
              stackedBody->bodies.end(),
              mMarkers[i].first)
          != stackedBody->bodies.end())
      {
        if (mMarkerIsAnatomical[i])
        {
          Eigen::Vector3s markerWorldPosition
              = markerWorldCenters.segment<3>(i * 3);
          anatomicalMarkerNames.push_back(mMarkerNames[i]);
          anatomicalMarkerWorldPositions.push_back(markerWorldPosition);
          anatomicalMarkerLocalPositions.push_back(
              stackedBody->bodies[0]->getWorldTransform().inverse()
              * markerWorldPosition);
        }
      }
    }

    // If there aren't at least two anatomical markers, then we won't have
    // anything to go on for prescaling this body, so don't worry about it
    if (anatomicalMarkerNames.size() < 2)
    {
      continue;
    }

    // 2. Now, we need to collect the average distances between the anatomical
    // markers in the marker data, the variance of those measurements.

    std::map<std::string, std::map<std::string, std::vector<s_t>>>
        measuredMarkerPairDistances;
    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      for (int i = 0; i < anatomicalMarkerNames.size(); i++)
      {
        for (int j = i + 1; j < anatomicalMarkerNames.size(); j++)
        {
          if (mMarkerObservations[t].count(anatomicalMarkerNames[i])
              && mMarkerObservations[t].count(anatomicalMarkerNames[j]))
          {
            measuredMarkerPairDistances
                [anatomicalMarkerNames[i]][anatomicalMarkerNames[j]]
                    .push_back(
                        (mMarkerObservations[t][anatomicalMarkerNames[i]]
                         - mMarkerObservations[t][anatomicalMarkerNames[j]])
                            .norm());
          }
        }
      }
    }

    std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights;
    for (int i = 0; i < anatomicalMarkerNames.size(); i++)
    {
      std::string firstMarker = anatomicalMarkerNames[i];
      if (measuredMarkerPairDistances.count(firstMarker) == 0)
        continue;
      for (int j = i + 1; j < anatomicalMarkerNames.size(); j++)
      {
        std::string secondMarker = anatomicalMarkerNames[j];
        if (measuredMarkerPairDistances[firstMarker].count(secondMarker) == 0)
          continue;

        std::vector<s_t>& observations
            = measuredMarkerPairDistances[firstMarker][secondMarker];

        s_t averageMeasuredDistance = 0.0;
        for (s_t distance : observations)
        {
          averageMeasuredDistance += distance;
        }
        averageMeasuredDistance /= std::max<int>(observations.size(), 1);
        s_t variance = 0.0;
        for (s_t distance : observations)
        {
          variance += (distance - averageMeasuredDistance)
                      * (distance - averageMeasuredDistance);
        }
        s_t percentVariance = variance / averageMeasuredDistance;
        // TODO: come up with a nice formula to use percentVariance to determine
        // weight
        (void)percentVariance;

        pairDistancesWithWeights.push_back(
            std::make_tuple(i, j, averageMeasuredDistance, 1.0));
      }
    }

    // 3. Now we're going to try to rescale the body based on the measured
    // marker pair distances.
    Eigen::Vector3s scale = getLocalScale(
        anatomicalMarkerLocalPositions,
        pairDistancesWithWeights,
        defaultScale,
        logOutput);

    // Now we scale the body
    for (auto& body : stackedBody->bodies)
    {
      if (logOutput)
      {
        std::cout << "Using anatomical markers to pre-scale body \""
                  << body->getName() << "\" by " << scale.transpose()
                  << std::endl;
      }
      body->setScale(scale);
    }
  }

  // 4. Now, armed with our updated body scales, we're going to recompute the
  // marker<->joint distances, and joint<->joint distances.

  Eigen::VectorXs updatedJointWorldCenters
      = mSkel->getJointWorldPositions(mSkel->getJoints());
  Eigen::VectorXs updatedMarkerWorldCenters
      = mSkel->getMarkerWorldPositions(mMarkers);

  // 4.1. Update joint<->marker squared distances
  for (std::shared_ptr<struct StackedJoint> joint : mStackedJoints)
  {
    if (mJointToMarkerSquaredDistances.count(joint->name))
    {
      auto& map = mJointToMarkerSquaredDistances[joint->name];
      for (auto& pair2 : map)
      {
        pair2.second = (getStackedJointCenterFromJointCentersVector(
                            joint, updatedJointWorldCenters)
                        - updatedMarkerWorldCenters.segment<3>(
                            mMarkerNameToIndex[pair2.first] * 3))
                           .squaredNorm();
      }
    }
  }

  // 4.2. Update joint<->joint squared distances
  for (std::shared_ptr<struct StackedJoint> joint1 : mStackedJoints)
  {
    for (std::shared_ptr<struct StackedJoint> joint2 : mStackedJoints)
    {
      if (mJointToJointSquaredDistances.count(joint1->name)
          && mJointToJointSquaredDistances[joint1->name].count(joint2->name))
      {
        mJointToJointSquaredDistances[joint1->name][joint2->name]
            = (getStackedJointCenterFromJointCentersVector(
                   joint1, updatedJointWorldCenters)
               - getStackedJointCenterFromJointCentersVector(
                   joint2, updatedJointWorldCenters))
                  .squaredNorm();
      }
    }
  }

  return 0.0;
}

//==============================================================================
/// For each timestep, and then for each joint, this sets up and runs an MDS
/// algorithm to triangulate the joint center location from the known marker
/// locations of the markers attached to the joint's two body segments, and
/// then known distance from the joint center to each marker.
s_t IKInitializer::closedFormMDSJointCenterSolver(bool logOutput)
{
  // 0. Save the default pose and scale's version of the joint centers and
  // marker locations, so that we can use it to detect and resolve coplanar
  // ambiguity in subsequent steps.
  Eigen::VectorXs oldPositions = mSkel->getPositions();
  Eigen::VectorXs oldScales = mSkel->getBodyScales();
  mSkel->setPositions(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
  mSkel->setBodyScales(Eigen::VectorXs::Ones(mSkel->getNumBodyNodes() * 3));
  Eigen::VectorXs neutralSkelJointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());
  Eigen::VectorXs neutralSkelMarkerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);
  mSkel->setPositions(oldPositions);
  mSkel->setBodyScales(oldScales);
  std::map<std::string, Eigen::Vector3s>
      neutralSkelJointCenterWorldPositionsMap;
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    neutralSkelJointCenterWorldPositionsMap[mStackedJoints[i]->name]
        = getStackedJointCenterFromJointCentersVector(
            mStackedJoints[i], neutralSkelJointWorldPositions);
  }
  std::map<std::string, Eigen::Vector3s> neutralSkelMarkerWorldPositionsMap;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    neutralSkelMarkerWorldPositionsMap[mMarkerNames[i]]
        = neutralSkelMarkerWorldPositions.segment<3>(i * 3);
  }

  s_t totalMarkerError = 0.0;
  int count = 0;
  mJointCenters.clear();
  mJointCentersEstimateSource.clear();
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    std::vector<std::shared_ptr<struct StackedJoint>> joints
        = getJointsAttachedToObservedMarkers(t);
    std::map<std::string, Eigen::Vector3s> lastSolvedJointCenters;
    std::map<std::string, Eigen::Vector3s> solvedJointCenters;
    while (true)
    {
      for (std::shared_ptr<struct StackedJoint> joint : joints)
      {
        // If we already solved this joint, no need to go again
        // if (lastSolvedJointCenters.count(joint->name))
        //   continue;

        std::vector<Eigen::Vector3s> adjacentPointLocations;
        std::vector<s_t> adjacentPointSquaredDistances;
        // These are the locations of each of the adjacent points, but on the
        // unscaled skeleton in the neutral pose, which we use to detect and
        // resolve coplanar ambiguity during reconstruction.
        std::vector<Eigen::Vector3s> adjacentPointLocationsInNeutralSkel;

        // 1. Find the markers that are adjacent to this joint and visible on
        // this frame
        for (auto& pair : mJointToMarkerSquaredDistances[joint->name])
        {
          if (mMarkerObservations[t].count(pair.first))
          {
            adjacentPointLocations.push_back(
                mMarkerObservations[t][pair.first]);
            adjacentPointSquaredDistances.push_back(pair.second);
            adjacentPointLocationsInNeutralSkel.push_back(
                neutralSkelMarkerWorldPositionsMap[pair.first]);
          }
        }

        for (auto& pair : mJointToJointSquaredDistances[joint->name])
        {
          if (lastSolvedJointCenters.count(pair.first))
          {
            adjacentPointLocations.push_back(
                lastSolvedJointCenters[pair.first]);
            adjacentPointSquaredDistances.push_back(pair.second);
            adjacentPointLocationsInNeutralSkel.push_back(
                neutralSkelJointCenterWorldPositionsMap[pair.first]);
          }
        }

        if (adjacentPointLocations.size() < 3)
          continue;

        // 2. Setup the squared-distance matrix
        int dim = adjacentPointLocations.size() + 1;
        Eigen::MatrixXs D = Eigen::MatrixXs::Zero(dim, dim);
        for (int i = 0; i < adjacentPointLocations.size(); i++)
        {
          for (int j = i + 1; j < adjacentPointLocations.size(); j++)
          {
            D(i, j) = (adjacentPointLocations[i] - adjacentPointLocations[j])
                          .squaredNorm();
            // The distance matrix is symmetric
            D(j, i) = D(i, j);
          }
        }
        for (int i = 0; i < adjacentPointLocations.size(); i++)
        {
          D(i, dim - 1) = adjacentPointSquaredDistances[i];
          D(dim - 1, i) = D(i, dim - 1);
        }

        // 3. Solve the distance matrix
        Eigen::MatrixXs pointCloud = getPointCloudFromDistanceMatrix(D);
        assert(!pointCloud.hasNaN());
        Eigen::MatrixXs transformed
            = mapPointCloudToData(pointCloud, adjacentPointLocations);

        // 4. Collect error statistics
        s_t pointCloudError = 0.0;
        for (int j = 0; j < adjacentPointLocations.size(); j++)
        {
          pointCloudError
              += (adjacentPointLocations[j] - transformed.col(j)).norm();
        }
        pointCloudError /= adjacentPointLocations.size();
        if (logOutput)
        {
          std::cout << "Joint center " << joint->name
                    << " point cloud reconstruction error: " << pointCloudError
                    << "m" << std::endl;
        }
        totalMarkerError += pointCloudError;
        count++;

        // 5. If the point cloud is co-planar (or very close to it), then
        // there's ambiguity about which side of the plane to place the joint
        // center. We'll check co-planarity on the neutral skeleton also, to
        // avoid having noise in the real marker data fool us into thinking
        // there isn't a coplanar ambiguity. Then we can also use the neutral
        // skeleton to resolve which side of the plane to place the joint
        // center.
        Eigen::Vector3s jointCenter = transformed.col(dim - 1);
        assert(!jointCenter.hasNaN());
        if (isCoplanar(adjacentPointLocationsInNeutralSkel)
            || isCoplanar(adjacentPointLocations))
        {
          if (logOutput)
          {
            std::cout << "Joint " << joint->name
                      << " has coplanar support, and is therefore ambiguous! "
                         "Resolving ambiguity."
                      << std::endl;
          }
          // If the point cloud is coplanar, then we need to resolve the
          // ambiguity about which side of the plane to place the joint center.
          // We do this by checking which side of the plane the joint center is
          // on in the neutral pose, and ensuring that the joint center is on
          // the same side of the plane in the reconstructed pose.
          jointCenter = ensureOnSameSideOfPlane(
              adjacentPointLocationsInNeutralSkel,
              neutralSkelJointCenterWorldPositionsMap[joint->name],
              adjacentPointLocations,
              jointCenter);
          assert(!jointCenter.hasNaN());
        }

        solvedJointCenters[joint->name] = jointCenter;
      }
      if (solvedJointCenters.size() == lastSolvedJointCenters.size())
        break;
      lastSolvedJointCenters = solvedJointCenters;
    }
    mJointCenters.push_back(lastSolvedJointCenters);

    std::map<std::string, JointCenterEstimateSource> estimateSources;
    for (auto& pair : lastSolvedJointCenters)
    {
      estimateSources[pair.first] = JointCenterEstimateSource::MDS;
    }
    mJointCentersEstimateSource.push_back(estimateSources);
  }
  return totalMarkerError / count;
}

//==============================================================================
/// This first finds an approximate rigid body trajectory for each body
/// that has at least 3 markers on it, and then sets up and solves a linear
/// system of equations for each joint to find the pair of offsets in the
/// adjacent body nodes that results in the least offset between the joint
/// centers.
s_t IKInitializer::closedFormPivotFindingJointCenterSolver(bool logOutput)
{
  // 0. Ensure that we've got enough space in our joint centers vector
  while (mJointCenters.size() < mMarkerObservations.size())
  {
    mJointCenters.push_back(std::map<std::string, Eigen::Vector3s>());
  }
  while (mJointCentersEstimateSource.size() < mMarkerObservations.size())
  {
    mJointCentersEstimateSource.push_back(
        std::map<std::string, JointCenterEstimateSource>());
  }
  while (mJointAxisDirs.size() < mMarkerObservations.size())
  {
    mJointAxisDirs.push_back(std::map<std::string, Eigen::Vector3s>());
  }

  // 0. Precompute all the marker variances
  std::map<std::string, Eigen::Vector3s> markerMeans;
  std::map<std::string, int> markerObservationCounts;
  std::map<std::string, s_t> markerVariances;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    markerMeans[mMarkerNames[i]] = Eigen::Vector3s::Zero();
    markerVariances[mMarkerNames[i]] = 0.0;
    markerObservationCounts[mMarkerNames[i]] = 0;
  }
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      if (mMarkerObservations[t].count(mMarkerNames[i]))
      {
        markerMeans[mMarkerNames[i]] += mMarkerObservations[t][mMarkerNames[i]];
        markerObservationCounts[mMarkerNames[i]]++;
      }
    }
  }
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (markerObservationCounts[mMarkerNames[i]] > 0)
    {
      markerMeans[mMarkerNames[i]] /= markerObservationCounts[mMarkerNames[i]];
    }
  }
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    for (int i = 0; i < mMarkerNames.size(); i++)
    {
      if (mMarkerObservations[t].count(mMarkerNames[i]))
      {
        Eigen::Vector3s diff = mMarkerObservations[t][mMarkerNames[i]]
                               - markerMeans[mMarkerNames[i]];
        markerVariances[mMarkerNames[i]] += diff.squaredNorm();
      }
    }
  }
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (markerObservationCounts[mMarkerNames[i]] > 0)
    {
      markerVariances[mMarkerNames[i]]
          /= markerObservationCounts[mMarkerNames[i]];
    }
    if (logOutput)
    {
      std::cout << "Marker \"" << mMarkerNames[i]
                << "\" variance: " << markerVariances[mMarkerNames[i]]
                << std::endl;
    }
  }

  // 1. Find all the bodies with at least 3 markers on them
  std::map<std::string, int> bodyMarkerCounts;
  for (int i = 0; i < mMarkers.size(); i++)
  {
    auto& marker = mMarkers[i];
    // Only accept markers whose variance is above a certain threshold
    if (markerVariances[mMarkerNames[i]] > 0.01)
    {
      for (auto& stackedBody : mStackedBodies)
      {
        if (std::find(
                stackedBody->bodies.begin(),
                stackedBody->bodies.end(),
                marker.first)
            != stackedBody->bodies.end())
        {
          if (bodyMarkerCounts.count(stackedBody->name) == 0)
            bodyMarkerCounts[stackedBody->name] = 0;
          bodyMarkerCounts[stackedBody->name]++;
          break;
        }
      }
    }
  }

  // 2. Find all the joints with bodies on both sides that have 3 markers on
  // them, and keep track of all adjacent bodies.
  std::vector<std::shared_ptr<struct StackedJoint>> jointsToSolveWithPivots;
  std::vector<std::shared_ptr<struct StackedJoint>>
      jointsToSolveWithChangPollard;
  std::vector<std::shared_ptr<struct StackedBody>> bodiesToFindTransformsFor;
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    std::shared_ptr<struct StackedJoint> joint = mStackedJoints[i];
    if (joint->parentBody == nullptr)
      continue;

    if (logOutput)
    {
      std::cout << "Joint " << joint->name << " has "
                << bodyMarkerCounts[joint->parentBody->name]
                << " markers on parent body, and "
                << bodyMarkerCounts[joint->childBody->name]
                << " markers on child body" << std::endl;
    }

    if (bodyMarkerCounts[joint->parentBody->name] >= 3
        && bodyMarkerCounts[joint->childBody->name] >= 3)
    {
      jointsToSolveWithPivots.push_back(joint);
    }
    else if (
        (bodyMarkerCounts[joint->parentBody->name] >= 3
         || bodyMarkerCounts[joint->childBody->name] >= 3)
        && (bodyMarkerCounts[joint->parentBody->name] > 0
            && bodyMarkerCounts[joint->childBody->name] > 0))
    {
      jointsToSolveWithChangPollard.push_back(joint);
    }
    // Keep track of bodies with enough markers, so we can solve for their
    // transforms over time
    if (bodyMarkerCounts[joint->parentBody->name] >= 3)
    {
      if (std::find(
              bodiesToFindTransformsFor.begin(),
              bodiesToFindTransformsFor.end(),
              joint->parentBody)
          == bodiesToFindTransformsFor.end())
      {
        bodiesToFindTransformsFor.push_back(joint->parentBody);
      }
    }
    if (bodyMarkerCounts[joint->childBody->name] >= 3)
    {
      if (std::find(
              bodiesToFindTransformsFor.begin(),
              bodiesToFindTransformsFor.end(),
              joint->childBody)
          == bodiesToFindTransformsFor.end())
      {
        bodiesToFindTransformsFor.push_back(joint->childBody);
      }
    }
  }

  (void)logOutput;
  // 3. Find the approximate rigid body trajectory for each body. The original
  // transform doesn't matter, since we're just looking for the relative
  // transform between the two bodies. So, arbitrarily, we choose the first
  // timestep where 3 markers are observed as the identity transform.
  std::map<std::string, std::map<int, Eigen::Isometry3s>> bodyTrajectories;
  for (std::shared_ptr<struct StackedBody> body : bodiesToFindTransformsFor)
  {
    // 3.1. Collect the names of the markers we're attached to
    std::vector<std::string> attachedMarkers;
    for (int i = 0; i < mMarkers.size(); i++)
    {
      auto marker = mMarkers[i];
      if (std::find(body->bodies.begin(), body->bodies.end(), marker.first)
          != body->bodies.end())
      {
        attachedMarkers.push_back(mMarkerNames[i]);
      }
    }
    // Go through all the timesteps, solving for relative transforms on the
    // timesteps that we can.
    std::map<int, Eigen::Isometry3s> bodyTrajectory;
    std::vector<std::string> visibleMarkersCloud;
    std::vector<Eigen::Vector3s> visibleMarkerCloudIdentityTransform;
    bool foundFirstFrame = false;
    s_t averagePointReconstructionError = 0.0;
    int countedTimesteps = 0;
    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      // 3.2. We first search through all the frames to find the first frame
      // where there are at least 3 markers visible on the body. Call this the
      // identity transform.
      if (!foundFirstFrame)
      {
        visibleMarkersCloud.clear();
        visibleMarkerCloudIdentityTransform.clear();
        for (std::string marker : attachedMarkers)
        {
          if (mMarkerObservations[t].count(marker))
          {
            visibleMarkersCloud.push_back(marker);
            visibleMarkerCloudIdentityTransform.push_back(
                mMarkerObservations[t][marker]);
          }
        }
        if (visibleMarkersCloud.size() >= 3)
        {
          foundFirstFrame = true;
          bodyTrajectory[t] = Eigen::Isometry3s::Identity();
        }
      }
      // 3.3. Once we have the identity transform, we can solve for the relative
      // transform of subsequent frames, if enough markers are visible.
      else
      {
        std::vector<Eigen::Vector3s> identityMarkerCloud;
        std::vector<Eigen::Vector3s> currentMarkerCloud;
        std::vector<s_t> weights;
        for (int i = 0; i < visibleMarkersCloud.size(); i++)
        {
          std::string marker = visibleMarkersCloud[i];
          if (mMarkerObservations[t].count(marker))
          {
            identityMarkerCloud.push_back(
                visibleMarkerCloudIdentityTransform[i]);
            currentMarkerCloud.push_back(mMarkerObservations[t][marker]);
            weights.push_back(1.0);
          }
        }

        // 3.4. If we have enough markers, we can solve for the relative
        // transform at this frame
        if (identityMarkerCloud.size() >= 3)
        {
          Eigen::Isometry3s worldTransform = getPointCloudToPointCloudTransform(
              identityMarkerCloud, currentMarkerCloud, weights);
          if (logOutput)
          {
            s_t error = 0.0;
            for (int i = 0; i < identityMarkerCloud.size(); i++)
            {
              error += (currentMarkerCloud[i]
                        - worldTransform * identityMarkerCloud[i])
                           .norm();
            }
            error /= identityMarkerCloud.size();
            averagePointReconstructionError += error;
            countedTimesteps++;
          }
          bodyTrajectory[t] = worldTransform;
        }
      }
    }
    bodyTrajectories[body->name] = bodyTrajectory;

    if (logOutput)
    {
      std::cout << "Body transforms for \"" << body->name
                << "\" reconstructed over " << countedTimesteps
                << " timesteps with average error "
                << averagePointReconstructionError / countedTimesteps << "m"
                << std::endl;
    }
  }

  // 4. Now we can solve for the relative transforms of each joint with 3+
  // markers on BOTH SIDES using pivots, by setting up a linear system of
  // equations where the unknowns are the relative transforms from each body to
  // the joint center, and the constraints are that those map to the same
  // location in world space on each timestep.
  s_t avgJointCenterError = 0.0;
  int solvedJoints = 0;
  for (std::shared_ptr<struct StackedJoint> joint : jointsToSolveWithPivots)
  {
    // At this point, the parent body node shouldn't be null
    assert(joint->parentBody != nullptr);
    std::string parentBodyName = joint->parentBody->name;
    std::string childBodyName = joint->childBody->name;

    // 4.1. Collect all the usable timesteps where transforms of parent and
    // child are both known
    std::vector<int> usableTimesteps;
    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      if (bodyTrajectories[parentBodyName].count(t)
          && bodyTrajectories[childBodyName].count(t))
      {
        usableTimesteps.push_back(t);
      }
    }
    if (usableTimesteps.size() < 5)
    {
      // Insufficient sample to solve for joint center
      continue;
    }

    // 4.2. Compute the average offset of the joint in the parent and child
    // frames, if we've already done some sort of solve to estimate the joint
    // center. We'll center our solution around that previous estimate.
    Eigen::Vector3s parentAvgOffset = Eigen::Vector3s::Zero();
    Eigen::Vector3s childAvgOffset = Eigen::Vector3s::Zero();
    int observationCount = 0;
    for (int t : usableTimesteps)
    {
      if (mJointCenters[t].count(joint->name))
      {
        parentAvgOffset += bodyTrajectories[parentBodyName][t].inverse()
                           * mJointCenters[t][joint->name];
        childAvgOffset += bodyTrajectories[childBodyName][t].inverse()
                          * mJointCenters[t][joint->name];
        observationCount++;
      }
    }
    if (observationCount > 0)
    {
      parentAvgOffset /= observationCount;
      childAvgOffset /= observationCount;
    }

    // 4.3. Subsample timesteps if necessary to keep problem size reasonable
    const int maxSamples = 500;
    if (usableTimesteps.size() > maxSamples)
    {
      std::vector<int> sampledIndices
          = math::evenlySpacedTimesteps(usableTimesteps.size(), maxSamples);
      std::vector<int> sampledTimesteps;
      for (int index : sampledIndices)
      {
        sampledTimesteps.push_back(usableTimesteps[index]);
      }
      usableTimesteps = sampledTimesteps;
    }

    // 4.4. Setup the linear system of equations
    int cols = 6;
    int rows = usableTimesteps.size() * 3;

    Eigen::MatrixXs A = Eigen::MatrixXs::Zero(rows, cols);
    Eigen::VectorXs b = Eigen::VectorXs::Zero(rows);
    for (int i = 0; i < usableTimesteps.size(); i++)
    {
      Eigen::Isometry3s parentTransform
          = bodyTrajectories[parentBodyName][usableTimesteps[i]];
      Eigen::Isometry3s childTransform
          = bodyTrajectories[childBodyName][usableTimesteps[i]];
      A.block<3, 3>(i * 3, 0) = parentTransform.linear();
      A.block<3, 3>(i * 3, 3) = -1 * childTransform.linear();
      b.segment<3>(i * 3)
          = parentTransform.translation() - childTransform.translation();
    }

    Eigen::VectorXs centerAnswerOn = Eigen::VectorXs::Zero(6);
    centerAnswerOn.head<3>() = parentAvgOffset;
    centerAnswerOn.tail<3>() = childAvgOffset;

    Eigen::VectorXs target = Eigen::VectorXs::Zero(rows);
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXs offsets
        = svd.solve(target - b - A * centerAnswerOn) + centerAnswerOn;
    assert(offsets.size() == 6);

    // 4.4. Decode the results of our solution into average values
    Eigen::Vector3s parentOffset = offsets.segment<3>(0);
    Eigen::Vector3s childOffset = offsets.segment<3>(3);

    s_t error = 0.0;
    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      Eigen::Isometry3s parentTransform = bodyTrajectories[parentBodyName][t];
      Eigen::Isometry3s childTransform = bodyTrajectories[childBodyName][t];
      Eigen::Vector3s parentWorldCenter = parentTransform * parentOffset;
      Eigen::Vector3s childWorldCenter = childTransform * childOffset;
      Eigen::Vector3s jointCenter
          = (parentWorldCenter + childWorldCenter) / 2.0;
      error += (parentWorldCenter - jointCenter).norm();
      error += (childWorldCenter - jointCenter).norm();

      // 4.4.1. Record the result
      mJointCenters[t][joint->name] = jointCenter;
      mJointCentersEstimateSource[t][joint->name]
          = JointCenterEstimateSource::LEAST_SQUARES_EXACT;
    }

    // 4.5. Check if we're on an axis, which will show up if there's a 0
    // singular value.
    Eigen::VectorXs singularValues = svd.singularValues();
    Eigen::MatrixXs V = svd.matrixV();
#ifndef NDEBUG
    Eigen::MatrixXs A_reconstructed
        = svd.matrixU() * singularValues.asDiagonal() * V.transpose();
    assert((A - A_reconstructed).norm() < 1e-8);
#endif
    // We normalize the singular values to be agnostic to the size of the data
    // we're dealing with in the original matrix.
    s_t conditionNumber = singularValues(0) / singularValues(5);
    if (logOutput)
    {
      std::cout << "Joint \"" << joint->name << "\" condition number is "
                << conditionNumber << std::endl;
    }
    if (std::abs(conditionNumber) > 50)
    {
      if (logOutput)
      {
        std::cout << "Joint \"" << joint->name << "\" is an axis joint!"
                  << std::endl;
      }
      // 4.5.1. We found an axis! Record it.
      Eigen::Vector6s nullSpace = V.col(5);
      Eigen::Vector3s parentLocalAxis = nullSpace.segment<3>(0).normalized();
      Eigen::Vector3s childLocalAxis = nullSpace.segment<3>(3).normalized();

      // 4.5.2. Transform the axis direction back to world space
      for (int t = 0; t < mMarkerObservations.size(); t++)
      {
        Eigen::Isometry3s parentTransform = bodyTrajectories[parentBodyName][t];
        Eigen::Isometry3s childTransform = bodyTrajectories[childBodyName][t];
        Eigen::Vector3s parentWorldAxis
            = parentTransform.linear() * parentLocalAxis;
        Eigen::Vector3s childWorldAxis
            = childTransform.linear() * childLocalAxis;
        Eigen::Vector3s jointAxis = (parentWorldAxis + childWorldAxis) / 2.0;
        // 4.5.3. Record the result
        mJointAxisDirs[t][joint->name] = jointAxis.normalized();
        mJointCentersEstimateSource[t][joint->name]
            = JointCenterEstimateSource::LEAST_SQUARES_AXIS;
      }
    }

    if (logOutput)
    {
      std::cout << "Solved for joint \"" << joint->name
                << "\" with a linear system on body transforms with "
                << usableTimesteps.size() << " samples, error: " << error << "m"
                << std::endl;
    }

    error /= usableTimesteps.size() * 2;
    avgJointCenterError += error;
    solvedJoints++;
  }

  // 5. Now we can solve for the relative transforms of each joint with 3+
  // markers on just ONE SIDE, and 1-2 markers on the other using
  // ChangPollard2006 (which does a better job handling small ranges of motion
  // than vanilla least-squares)
  for (std::shared_ptr<struct StackedJoint> joint :
       jointsToSolveWithChangPollard)
  {
    // 5.1. Figure out which body is going to be the reference frame, and which
    // body has only a few markers to solve with
    std::shared_ptr<struct StackedBody> anchorBody = nullptr;
    std::shared_ptr<struct StackedBody> otherBody = nullptr;
    if (bodyMarkerCounts[joint->parentBody->name] >= 3)
    {
      assert(bodyMarkerCounts[joint->childBody->name] < 3);
      assert(bodyMarkerCounts[joint->childBody->name] > 0);
      anchorBody = joint->parentBody;
      otherBody = joint->childBody;
    }
    else
    {
      assert(bodyMarkerCounts[joint->parentBody->name] < 3);
      assert(bodyMarkerCounts[joint->parentBody->name] > 0);
      assert(bodyMarkerCounts[joint->childBody->name] >= 3);
      anchorBody = joint->childBody;
      otherBody = joint->parentBody;
    }
    if (logOutput)
    {
      std::cout << "Joint \"" << joint->name << "\" has anchor body \""
                << anchorBody->name << "\" and other body \"" << otherBody->name
                << "\"" << std::endl;
    }

    // 5.2. Work out which markers will be moving around on the other body
    std::vector<std::string> movingMarkers;
    for (int i = 0; i < mMarkers.size(); i++)
    {
      if (std::find(
              otherBody->bodies.begin(),
              otherBody->bodies.end(),
              mMarkers[i].first)
          != otherBody->bodies.end())
      {
        movingMarkers.push_back(mMarkerNames[i]);
      }
    }
    if (logOutput)
    {
      std::cout << "Joint \"" << joint->name << "\" has "
                << movingMarkers.size() << " markers attached to \""
                << otherBody->name
                << "\", and we will find their relative joint center in the "
                   "frame of \""
                << anchorBody->name << "\":" << std::endl;
      for (std::string marker : movingMarkers)
      {
        std::cout << "  " << marker << std::endl;
      }
    }

    // 5.3. Go through and transform all the markers into the anchor body's
    // frame.
    std::map<std::string, std::vector<Eigen::Vector3s>> anchorBodyMarkerClouds;
    int uniqueMarkerObservationsFound = 0;
    for (auto& pair : bodyTrajectories[anchorBody->name])
    {
      int t = pair.first;
      Eigen::Isometry3s anchorTransform = pair.second;
      for (std::string movingMarker : movingMarkers)
      {
        if (mMarkerObservations[t].count(movingMarker))
        {
          if (anchorBodyMarkerClouds.count(movingMarker) == 0)
          {
            anchorBodyMarkerClouds[movingMarker]
                = std::vector<Eigen::Vector3s>();
          }
          Eigen::Vector3s observation = anchorTransform.inverse()
                                        * mMarkerObservations[t][movingMarker];
          bool foundDuplicate = false;
          for (Eigen::Vector3s existingObservation :
               anchorBodyMarkerClouds[movingMarker])
          {
            if ((existingObservation - observation).norm() < 1e-10)
            {
              foundDuplicate = true;
              break;
            }
          }
          if (!foundDuplicate)
          {
            anchorBodyMarkerClouds[movingMarker].push_back(observation);
            uniqueMarkerObservationsFound++;
          }
        }
      }
    }

    if (uniqueMarkerObservationsFound < 3)
    {
      if (logOutput)
      {
        std::cout << "Got insufficient unique marker observations ("
                  << uniqueMarkerObservationsFound << ") to solve joint \""
                  << joint->name << "\" with Chang Pollard 2006" << std::endl;
      }
      continue;
    }

    bool anyMarkersTooCloseToJoint = false;
    for (std::string marker : movingMarkers)
    {
      if (mJointToMarkerSquaredDistances[joint->name].count(marker) > 0)
      {
        if (mJointToMarkerSquaredDistances[joint->name][marker] < 0.01)
        {
          if (logOutput)
          {
            std::cout << "Joint \"" << joint->name << "\" detected marker \""
                      << marker
                      << "\" is too close to the joint (perhaps this is a "
                         "synthetic joint center marker?), which would cause a "
                         "singularity on Chang Pollard 2006, so we're going to "
                         "go straight to least-squares."
                      << std::endl;
          }
          anyMarkersTooCloseToJoint = true;
          break;
        }
      }
    }

    std::vector<std::vector<Eigen::Vector3s>> markerTraces;
    for (auto& pair : anchorBodyMarkerClouds)
    {
      Eigen::Vector3s meanObservation = Eigen::Vector3s::Zero();
      for (Eigen::Vector3s observation : pair.second)
      {
        meanObservation += observation;
      }
      meanObservation /= pair.second.size();

      s_t markerVariance = 0;
      for (Eigen::Vector3s observation : pair.second)
      {
        markerVariance += (observation - meanObservation).squaredNorm();
      }
      markerVariance /= pair.second.size();

      if (markerVariance > 0.01)
      {
        markerTraces.push_back(pair.second);
        if (logOutput)
        {
          std::cout << "Using marker \"" << pair.first << "\" with variance "
                    << markerVariance << std::endl;
        }
      }
      else
      {
        if (logOutput)
        {
          std::cout << "Skipping marker \"" << pair.first
                    << "\" because its variance is " << markerVariance
                    << std::endl;
        }
      }
    }

    if (markerTraces.size() == 0)
    {
      if (logOutput)
      {
        std::cout << "Have no markers left after filtering for sufficient "
                     "variance in local space, so not solving joint \""
                  << joint->name << "\"" << std::endl;
      }
      continue;
    }

    // 5.4. Get the local joint center in the anchor body
    Eigen::Vector3s jointCenter;
    if (anyMarkersTooCloseToJoint)
    {
      if (logOutput)
      {
        std::cout << "Solving for joint \"" << joint->name
                  << "\" with Least Squares" << std::endl;
      }
      jointCenter = leastSquaresConcentricSphereFit(markerTraces, logOutput);
    }
    else
    {
      if (logOutput)
      {
        std::cout << "Solving for joint \"" << joint->name
                  << "\" with Chang Pollard 2006" << std::endl;
      }
      jointCenter
          = getChangPollard2006JointCenterMultiMarker(markerTraces, logOutput);
    }

    // 5.5. Transform the result back into world space, and record them
    for (auto& pair : bodyTrajectories[anchorBody->name])
    {
      int t = pair.first;
      Eigen::Isometry3s anchorTransform = pair.second;
      Eigen::Vector3s jointWorldCenter = anchorTransform * jointCenter;
      mJointCenters[t][joint->name] = jointWorldCenter;
      mJointCentersEstimateSource[t][joint->name]
          = JointCenterEstimateSource::LEAST_SQUARES_EXACT;
    }

    // 5.6. Get the axis direction in the anchor body, along with the
    // corresponding singular value
    std::pair<Eigen::Vector3s, s_t> axisDirAndConditionNumber
        = gamageLasenby2002AxisFit(markerTraces);
    Eigen::Vector3s jointAxisDir = axisDirAndConditionNumber.first;
    s_t conditionNumber = axisDirAndConditionNumber.second;
    if (std::abs(conditionNumber) > 50)
    {
      // 5.6.1. We found an axis! Record it.
      if (logOutput)
      {
        std::cout << "Joint \"" << joint->name
                  << "\" found a large condition value (" << conditionNumber
                  << ") in its linear system, so it's an axis joint."
                  << std::endl;
      }
      // 5.6.2. Transform the axis direction back to world space
      for (auto& pair : bodyTrajectories[anchorBody->name])
      {
        int t = pair.first;
        Eigen::Isometry3s anchorTransform = pair.second;
        Eigen::Vector3s jointWorldAxis
            = anchorTransform.linear() * jointAxisDir;
        mJointAxisDirs[t][joint->name] = jointWorldAxis.normalized();
        mJointCentersEstimateSource[t][joint->name]
            = JointCenterEstimateSource::LEAST_SQUARES_AXIS;
      }
    }

    if (logOutput)
    {
      std::cout << "Solved for joint \"" << joint->name
                << "\" with Chang Pollard 2006" << std::endl;
    }
  }

  avgJointCenterError /= solvedJoints;
  return avgJointCenterError;
}

//==============================================================================
/// For joints where we believe they're floating along an axis, and we have at
/// least one of either the parent or child joint has a known joint center
/// WITHOUT the axis amiguity, we can slide the joint center along the axis
/// until the angle formed by the axis and the known adjacent joint matches
/// what is in the skeleton.
s_t IKInitializer::recenterAxisJointsBasedOnBoneAngles(bool logOutput)
{
  // 0. Save the default pose and scale's version of the joint centers and
  // marker locations, so that we can use it to detect and resolve coplanar
  // ambiguity in subsequent steps.
  Eigen::VectorXs oldPositions = mSkel->getPositions();
  Eigen::VectorXs oldScales = mSkel->getBodyScales();
  mSkel->setPositions(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
  mSkel->setBodyScales(Eigen::VectorXs::Ones(mSkel->getNumBodyNodes() * 3));
  Eigen::VectorXs neutralSkelJointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());
  Eigen::VectorXs neutralSkelMarkerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);
  std::map<std::string, Eigen::Vector3s> neutralSkelMarkerWorldPositionsMap;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    neutralSkelMarkerWorldPositionsMap[mMarkerNames[i]]
        = neutralSkelMarkerWorldPositions.segment<3>(i * 3);
  }
  std::map<std::string, Eigen::Vector3s>
      neutralSkelJointCenterWorldPositionsMap;
  std::map<std::string, Eigen::Vector3s> neutralSkelJointAxisWorldDirectionsMap;
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    neutralSkelJointCenterWorldPositionsMap[mStackedJoints[i]->name]
        = getStackedJointCenterFromJointCentersVector(
            mStackedJoints[i], neutralSkelJointWorldPositions);
    neutralSkelJointAxisWorldDirectionsMap[mStackedJoints[i]->name]
        = mStackedJoints[i]
              ->joints[0]
              ->getWorldAxisScrewForPosition(0)
              .head<3>()
              .normalized();
    std::string jointType = mStackedJoints[i]->joints[0]->getType();

    bool hasNoInternalTranslation
        = jointType == dynamics::RevoluteJoint::getStaticType()
          || jointType == dynamics::EulerJoint::getStaticType()
          || jointType == dynamics::BallJoint::getStaticType()
          || jointType == dynamics::UniversalJoint::getStaticType();

    // If we're not a pure revolute joint (for example, a CustomJoint driven by
    // evil splines) we need to do some extra work to estimate a useful joint
    // axis in the neutral pose.
    if (!hasNoInternalTranslation)
    {
      if (logOutput)
      {
        std::cout << "Joint \"" << mStackedJoints[i]->name
                  << "\" is not a pure revolute joint, so we're going to "
                     "estimate a joint axis based on a sweep of virtual "
                     "markers, and then a Gamage Lasenby axis fit."
                  << std::endl;
      }
      dynamics::Joint* revoluteJoint = mStackedJoints[i]->joints[0];
      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
      for (int i = 0; i < 3; i++)
      {
        markers.push_back(std::make_pair(
            revoluteJoint->getChildBodyNode(), Eigen::Vector3s::Unit(i)));
      }

      s_t min = revoluteJoint->getDof(0)->getPositionLowerLimit();
      s_t max = revoluteJoint->getDof(0)->getPositionUpperLimit();
      int steps = 2000;
      s_t stepSize = (max - min) / (s_t)steps;

      std::vector<std::vector<Eigen::Vector3s>> markerObservationsOverSweep;
      for (int i = 0; i < markers.size(); i++)
      {
        markerObservationsOverSweep.push_back(std::vector<Eigen::Vector3s>());
      }
      for (int i = 0; i < steps; i++)
      {
        s_t pos = min + stepSize * i;
        revoluteJoint->setPosition(0, pos);
        Eigen::VectorXs markerWorldPos
            = mSkel->getMarkerWorldPositions(markers);
        for (int j = 0; j < markers.size(); j++)
        {
          markerObservationsOverSweep[j].push_back(
              markerWorldPos.segment<3>(j * 3));
        }
      }
      // Reset to the neutral pose when we're done
      revoluteJoint->setPosition(0, 0.0);

      Eigen::Vector3s axis
          = gamageLasenby2002AxisFit(markerObservationsOverSweep).first;
      if (logOutput)
      {
        std::cout << "Joint \"" << mStackedJoints[i]->name
                  << "\" found Gamage Lasenby axis in neutral pos: "
                  << axis.transpose() << std::endl;
      }
      neutralSkelJointAxisWorldDirectionsMap[mStackedJoints[i]->name] = axis;
    }
  }
  mSkel->setPositions(oldPositions);
  mSkel->setBodyScales(oldScales);

  // 1. Sort the joints into two categories: "axis joints", where the solver
  // found an axis along which the joint center is ambiguous, and "center
  // joints", where the solver believes it knows the joint center without
  // ambiguity.
  std::vector<std::shared_ptr<struct StackedJoint>> axisJoints;
  std::vector<std::shared_ptr<struct StackedJoint>> centerJoints;
  std::map<std::string, bool> isJointAxis;

  for (std::shared_ptr<struct StackedJoint> joint : mStackedJoints)
  {
    int numCentersObserved = 0;
    int numAxisObserved = 0;
    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      if (mJointCenters[t].count(joint->name))
      {
        numCentersObserved++;
      }
      if (mJointAxisDirs[t].count(joint->name))
      {
        numAxisObserved++;
      }
    }
    if (numCentersObserved > 0 && numAxisObserved > 0)
    {
      isJointAxis[joint->name] = true;
      if (logOutput)
      {
        std::cout
            << "Joint \"" << joint->name
            << "\" has found axis, we're going to attempt to re-center it."
            << std::endl;
      }
      axisJoints.push_back(joint);
    }
    else if (numCentersObserved > 0)
    {
      isJointAxis[joint->name] = false;
      centerJoints.push_back(joint);
    }
  }

  // 2. We're going to repeatedly look for an axis joint that meets our criteria
  // for "solvability", in order to work out how far along the axis the joint
  // center lies relative to adjacent "center joints". Once we've solved an
  // "axis joint", and adjusted its center location, we'll move it to be a
  // "center joint" and repeat the process until we can't find any more joints
  // to improve.
  (void)logOutput;
  while (true)
  {
    bool anyChanged = false;

    for (int i = 0; i < axisJoints.size(); i++)
    {
      std::shared_ptr<struct StackedJoint> joint = axisJoints[i];
      bool isBallJoint = joint->joints[0]->getNumDofs() > 1;

      // 2.1. Find the joints that are not axis joints, that are adjacent to
      // this joint. This includes both parent and as many children as fit the
      // criteria.
      std::vector<std::shared_ptr<struct StackedJoint>> adjacentPointJoints;
      if (joint->parentBody != nullptr
          && !isJointAxis[joint->parentBody->parentJoint->name])
      {
        adjacentPointJoints.push_back(joint->parentBody->parentJoint);
      }
      for (auto& childJoint : joint->childBody->childJoints)
      {
        if (!isJointAxis[childJoint->name])
        {
          adjacentPointJoints.push_back(childJoint);
        }
      }

      // We'll hold off solving joints until at least one adjacent joint has
      // been solved, so that we can work our way out/in from unambiguous joint
      // centers, since those provide so much useful info on joint center
      // location.
      if (adjacentPointJoints.size() > 0 || isBallJoint)
      {
        if (logOutput && isBallJoint)
        {
          std::cout << "Classifying " << joint->name
                    << " as a ball joint for the purposes of recentering along "
                       "the axis, so we'll ignore offset information to the "
                       "parent along the axis."
                    << std::endl;
        }

        // 2.2. For non-ball joints: collect the joint centers and axis
        // direction on the neutral skeleton, so we can determine our ratio of
        // parallel to perpendicular distances in the neutral pose
        const Eigen::Vector3s neutralAxis
            = neutralSkelJointAxisWorldDirectionsMap[joint->name];
        const Eigen::Vector3s neutralJointCenter
            = neutralSkelJointCenterWorldPositionsMap[joint->name];

        s_t averageOffset = 0.0;
        int numTimestepsCounted = 0;
        std::vector<s_t> recordedOffsets;

        // 2.3. Now we're going to run through every timestep and solve each
        // one, and then collect an average offset to apply
        for (int t = 0; t < mMarkerObservations.size(); t++)
        {
          if (mJointCenters[t].count(joint->name) == 0
              || mJointAxisDirs[t].count(joint->name) == 0)
          {
            continue;
          }

          std::vector<std::pair<Eigen::Vector3s, s_t>> pointsAndRadii;
          std::vector<s_t> weights;

          // 2.3.1. Collect the points and radii for the adjacent joint(s) that
          // have unambiguous joint centers, if we're not a ball joint and so
          // can rely on consistent geometry between the axis and the adjacent
          // joint centers with known locations.
          if (!isBallJoint)
          {
            for (std::shared_ptr<struct StackedJoint> pointJoint :
                 adjacentPointJoints)
            {
              if (mJointCenters[t].count(pointJoint->name) == 0)
                continue;

              Eigen::Vector3s adjacentNeutralJointCenter
                  = neutralSkelJointCenterWorldPositionsMap[pointJoint->name];

              // 2.3.1.1. Find the ratio on the neutral skeleton for these two
              // joints between (distance) / (distance perpendicular to axis)
              Eigen::Vector3s neutralOffset
                  = adjacentNeutralJointCenter - neutralJointCenter;
              s_t neutralDistAlongAxis = neutralOffset.dot(neutralAxis);
              Eigen::Vector3s neutralOffsetParallel
                  = neutralDistAlongAxis * neutralAxis;
              Eigen::Vector3s neutralOffsetPerp
                  = (neutralOffset - neutralOffsetParallel);
              assert(
                  (neutralOffsetParallel + neutralOffsetPerp - neutralOffset)
                      .norm()
                  < 1e-15);
              assert(
                  (neutralJointCenter + neutralOffsetParallel
                   + neutralOffsetPerp - adjacentNeutralJointCenter)
                      .norm()
                  < 1e-15);
              s_t neutralRatio
                  = neutralOffset.norm() / neutralOffsetPerp.norm();

              // 2.3.1.2. Find the ratio on the current skeleton for these two
              // joints between (distance) / (distance perpendicular to axis)
              Eigen::Vector3s jointCenter = mJointCenters[t][joint->name];
              Eigen::Vector3s jointAxis = mJointAxisDirs[t][joint->name];
              Eigen::Vector3s adjacentJointCenter
                  = mJointCenters[t][pointJoint->name];
              Eigen::Vector3s offset = adjacentJointCenter - jointCenter;
              s_t distAlongAxis = offset.dot(jointAxis);
              Eigen::Vector3s offsetParallel = distAlongAxis * jointAxis;
              Eigen::Vector3s offsetPerp = (offset - offsetParallel);

              s_t impliedDistance = offsetPerp.norm() * neutralRatio;

              // 2.3.1.3. Record the implied distance to our objective set
              pointsAndRadii.emplace_back(
                  mJointCenters[t][pointJoint->name], impliedDistance);
              weights.push_back(1.0);
            }
          }

          // 2.4. Now we're going to record all the available marker distances
          // as well, since those provide useful disambiguating signal, and can
          // help compensate for otherwise large errors caused by small changes
          // in joint axis.

          for (auto& pair : mJointToMarkerSquaredDistances[joint->name])
          {
            std::string markerName = pair.first;
            s_t squaredDistance = pair.second;
            if (mMarkerObservations[t].count(markerName) == 0)
              continue;
            Eigen::Vector3s markerWorldPos = mMarkerObservations[t][markerName];
            pointsAndRadii.emplace_back(markerWorldPos, sqrt(squaredDistance));
            bool markerIsAnatomical
                = mMarkerIsAnatomical[mMarkerNameToIndex[markerName]];
            weights.push_back(markerIsAnatomical ? 1.0 : 0.01);
          }

          Eigen::Vector3s newCenter = centerPointOnAxis(
              mJointCenters[t][joint->name],
              mJointAxisDirs[t][joint->name],
              pointsAndRadii,
              weights);
          s_t distAlongAxis = (newCenter - mJointCenters[t][joint->name])
                                  .dot(mJointAxisDirs[t][joint->name]);
          averageOffset += distAlongAxis;
          numTimestepsCounted++;
          recordedOffsets.push_back(distAlongAxis);
        }

        averageOffset /= numTimestepsCounted;

        s_t offsetVariance = 0.0;
        for (int i = 0; i < recordedOffsets.size(); i++)
        {
          offsetVariance += (recordedOffsets[i] - averageOffset)
                            * (recordedOffsets[i] - averageOffset);
        }
        offsetVariance /= recordedOffsets.size();

        if (logOutput)
        {
          std::cout << "Joint \"" << joint->name
                    << "\" average offset along axis: " << averageOffset
                    << "m with variance " << offsetVariance << "m^2"
                    << std::endl;
        }

        // 2.5. Now we're going to go through and apply the average offset we
        // just found to the joint centers
        for (int t = 0; t < mMarkerObservations.size(); t++)
        {
          if (mJointCenters[t].count(joint->name) == 0
              || mJointAxisDirs[t].count(joint->name) == 0)
            continue;
          Eigen::Vector3s jointCenter = mJointCenters[t][joint->name];
          Eigen::Vector3s jointAxis = mJointAxisDirs[t][joint->name];
          Eigen::Vector3s newCenter = jointCenter + averageOffset * jointAxis;
          mJointCenters[t][joint->name] = newCenter;
          mJointCentersEstimateSource[t][joint->name]
              = JointCenterEstimateSource::LEAST_SQUARES_AXIS;
        }
      }

      // 2.6. Now we can mark this joint as having a reliable joint center,
      // so we can use it as support in the next iteration of the algorithm.
      centerJoints.push_back(joint);
      axisJoints.erase(axisJoints.begin() + i);
      isJointAxis[joint->name] = false;
      anyChanged = true;
      break;
    }
    if (!anyChanged)
    {
      break;
    }
  }

  return 0.0;
}

//==============================================================================
/// This uses the current guesses for the joint centers to re-estimate the
/// bone sizes (based on distance between joint centers) and then use that to
/// get the group scale vector. This also uses the joint centers to estimate
/// the body positions.
void IKInitializer::estimateGroupScalesClosedForm(bool log)
{
  Eigen::VectorXs originalPose = mSkel->getPositions();
  Eigen::VectorXs originalBodyScales = mSkel->getBodyScales();
  mSkel->setBodyScales(Eigen::VectorXs::Ones(mSkel->getNumBodyNodes() * 3));
  mSkel->setPositions(Eigen::VectorXs::Zero(mSkel->getNumDofs()));

  Eigen::VectorXs jointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());
  Eigen::VectorXs markerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);

  s_t defaultScale = 1.0;
  if (mModelHeightM > 0)
  {
    s_t defaultHeight
        = mSkel->getHeight(Eigen::VectorXs::Zero(mSkel->getNumDofs()));
    if (defaultHeight > 0)
    {
      defaultScale = mModelHeightM / defaultHeight;
    }
  }

  // 1. Find a scale for all the bodies that we can
  std::map<std::string, Eigen::Vector3s> bodyScales;
  std::map<std::string, Eigen::Vector3s> bodyScaleWeights;
  for (std::shared_ptr<struct StackedBody> bodyNode : mStackedBodies)
  {
    std::vector<Eigen::Vector3s> localPoints;

    // 1.1. Collect joints adjacent to this body that we can potentially use
    // to help scale
    std::vector<std::shared_ptr<struct StackedJoint>> adjacentJoints;
    // Only use the parent joint to scale if it's not the root joint. The root
    // joint doesn't really have a meaningful "joint center", so we don't want
    // to use it to scale.
    if (bodyNode->parentJoint->parentBody != nullptr)
    {
      adjacentJoints.push_back(bodyNode->parentJoint);
    }
    for (int i = 0; i < bodyNode->childJoints.size(); i++)
    {
      adjacentJoints.push_back(bodyNode->childJoints[i]);
    }
    for (int i = 0; i < adjacentJoints.size(); i++)
    {
      Eigen::Vector3s jointWorldPos
          = getStackedJointCenterFromJointCentersVector(
              adjacentJoints[i], jointWorldPositions);
      Eigen::Vector3s jointLocalPos
          = bodyNode->bodies[0]->getWorldTransform().inverse() * jointWorldPos;
      localPoints.push_back(jointLocalPos);
    }

    // 1.2. Collect anatomical markers attached to this body that we can use to
    // help scale
    std::vector<std::string> anatomicalMarkers;
    for (int i = 0; i < mMarkers.size(); i++)
    {
      if (mMarkerIsAnatomical[i]
          && std::find(
                 bodyNode->bodies.begin(),
                 bodyNode->bodies.end(),
                 mMarkers[i].first)
                 != bodyNode->bodies.end())
      {
        anatomicalMarkers.push_back(mMarkerNames[i]);
        Eigen::Vector3s markerLocalPos
            = bodyNode->bodies[0]->getWorldTransform().inverse()
              * markerWorldPositions.segment<3>(i * 3);
        localPoints.push_back(markerLocalPos);
      }
    }

    // 1.3. Go through all the timesteps, and collect average distances between
    // the various points of interest (joint center estimates, anatomical
    // markers) in world space.
    Eigen::MatrixXd avgDistances
        = Eigen::MatrixXd::Zero(localPoints.size(), localPoints.size());
    Eigen::MatrixXi counts
        = Eigen::MatrixXi::Zero(localPoints.size(), localPoints.size());

    for (int t = 0; t < mMarkerObservations.size(); t++)
    {
      for (int i = 0; i < localPoints.size(); i++)
      {
        // 1.3.1. For the i'th localPoint, retrieve its world position at time t
        Eigen::Vector3s point1Pos = Eigen::Vector3s::Zero();
        if (i < adjacentJoints.size())
        {
          // If this is a joint, and we don't have an estimate, skip this point
          // on this timestep
          if (mJointCenters[t].count(adjacentJoints[i]->name) == 0)
            continue;
          point1Pos = mJointCenters[t][adjacentJoints[i]->name];
        }
        else
        {
          std::string markerName = anatomicalMarkers[i - adjacentJoints.size()];
          // If this is a marker, and we don't have an observation, skip this
          // point on this timestep
          if (mMarkerObservations[t].count(markerName) == 0)
            continue;
          point1Pos = mMarkerObservations[t][markerName];
        }

        for (int j = i + 1; j < localPoints.size(); j++)
        {
          // 1.3.2. For the j'th localPoint, retrieve its world position at time
          // t
          Eigen::Vector3s point2Pos = Eigen::Vector3s::Zero();
          if (j < adjacentJoints.size())
          {
            // If this is a joint, and we don't have an estimate, skip this
            // point on this timestep
            if (mJointCenters[t].count(adjacentJoints[j]->name) == 0)
              continue;
            point2Pos = mJointCenters[t][adjacentJoints[j]->name];
          }
          else
          {
            std::string markerName
                = anatomicalMarkers[j - adjacentJoints.size()];
            // If this is a marker, and we don't have an observation, skip this
            // point on this timestep
            if (mMarkerObservations[t].count(markerName) == 0)
              continue;
            point2Pos = mMarkerObservations[t][markerName];
          }

          // 1.3.3. If we make it here, we've got two points we can use to
          // estimate.
          s_t distance = (point1Pos - point2Pos).norm();
          avgDistances(i, j) += distance;
          avgDistances(j, i) += distance;
          counts(i, j) += 1;
          counts(j, i) += 1;
        }
      }
    }

    // 1.4. Now we've got a matrix of average distances between all the points,
    // we can construct a weighted set of observations and then use our local
    // scale method to reconstruct the scale of the body.
    std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights;
    for (int i = 0; i < localPoints.size(); i++)
    {
      for (int j = i + 1; j < localPoints.size(); j++)
      {
        if (counts(i, j) > 0)
        {
          s_t avgDistance = avgDistances(i, j) / counts(i, j);
          // TODO: do something clever with the weights
          s_t weight = 1.0;
          pairDistancesWithWeights.push_back(
              std::make_tuple(i, j, avgDistance, weight));
        }
      }
    }
    Eigen::Vector3s scale = getLocalScale(
        localPoints, pairDistancesWithWeights, defaultScale, log);

    // 1.5. Apply that scale to all the bodies in this stacked body
    for (dynamics::BodyNode* body : bodyNode->bodies)
    {
      body->setScale(scale);
    }
  }

  // 2. Ensure that the scaling is symmetric across groups, by condensing into
  // the group scales vector and then re-setting the body scales from that
  // vector.
  mGroupScales = mSkel->getGroupScales();
  mSkel->setGroupScales(mGroupScales);
}

//==============================================================================
/// This takes the current guesses for the joint centers and the group scales,
/// and just runs straightforward IK to get the body positions.
s_t IKInitializer::estimatePosesWithIK(bool logOutput)
{
  std::shared_ptr<dynamics::Skeleton> skelBallJoints
      = mSkel->convertSkeletonToBallJoints();
  Eigen::VectorXs lastPose = Eigen::VectorXs::Zero(mSkel->getNumDofs());
  s_t avgLoss = 0.0;
  mPoses.clear();
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    if (t % 100 == 0 || t == mMarkerObservations.size() - 1)
    {
      std::cout << "IKInitializer solved IK " << t << "/"
                << mMarkerObservations.size() << std::endl;
    }
    bool newClip = mNewClip[t] || t == 0;
    // 1. Solve IK for new clips using ball joints
    if (newClip)
    {
      // 1.1. Find a linearized list of markers to target
      std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
      std::vector<Eigen::Vector3s> markerPoses;
      std::vector<bool> anatomicalMarkers;
      for (int i = 0; i < mMarkers.size(); i++)
      {
        if (mMarkerObservations[t].count(mMarkerNames[i]))
        {
          markers.emplace_back(
              skelBallJoints->getBodyNode(mMarkers[i].first->getName()),
              mMarkers[i].second);
          markerPoses.push_back(mMarkerObservations[t][mMarkerNames[i]]);
          anatomicalMarkers.push_back(mMarkerIsAnatomical[i]);
        }
      }
      Eigen::VectorXs markerTarget = Eigen::VectorXs::Zero(markers.size() * 3);
      for (int i = 0; i < markers.size(); i++)
      {
        markerTarget.segment<3>(i * 3) = markerPoses[i];
      }

      // 1.2. Convert the visible joints over into pointers to the ball joints
      // skeleton
      std::vector<dynamics::Joint*> joints;
      std::vector<std::vector<int>> jointClusters;
      std::vector<Eigen::Vector3s> jointPoses;
      for (int i = 0; i < mStackedJoints.size(); i++)
      {
        if (mJointCenters[t].count(mStackedJoints[i]->name))
        {
          jointPoses.push_back(mJointCenters[t][mStackedJoints[i]->name]);
          std::vector<int> clusterIndices;
          for (dynamics::Joint* joint : mStackedJoints[i]->joints)
          {
            clusterIndices.push_back(joints.size());
            joints.push_back(skelBallJoints->getJoint(joint->getName()));
          }
          jointClusters.push_back(clusterIndices);
        }
      }
      Eigen::VectorXs jointClusterTarget
          = Eigen::VectorXs::Zero(joints.size() * 3);
      for (int i = 0; i < joints.size(); i++)
      {
        jointClusterTarget.segment<3>(i * 3) = jointPoses[i];
      }

      // 1.3. Solve the actual IK
      math::solveIK(
          mSkel->convertPositionsToBallSpace(lastPose),
          mSkel->getPositionUpperLimits(),
          mSkel->getPositionLowerLimits(),
          markerTarget.size() + jointClusterTarget.size(),
          [&](const Eigen::VectorXs& pos, bool clamp) {
            // 1.3.1. Set poses on the ball joint skeleton, by default not
            // clamping to limits
            skelBallJoints->setPositions(pos);
            if (clamp)
            {
              // If we're clamping to limits, do it in the original skeleton
              // joint space, not in ball space
              mSkel->setPositions(mSkel->convertPositionsFromBallSpace(pos));
              mSkel->clampPositionsToLimits();
              skelBallJoints->setPositions(
                  mSkel->convertPositionsToBallSpace(mSkel->getPositions()));
            }
            return skelBallJoints->getPositions();
          },
          [&](Eigen::Ref<Eigen::VectorXs> diff,
              Eigen::Ref<Eigen::MatrixXs> jac) {
            // 1.3.2. Evaluate the error and the Jacobian relating dError / dPos

            // 1.3.2.1. First we need to compute the marker error, and marker
            // Jacobian
            Eigen::VectorXs markerPositions
                = skelBallJoints->getMarkerWorldPositions(markers);
            diff.segment(0, markerPositions.size())
                = markerPositions - markerTarget;
            jac.block(0, 0, markerPositions.size(), jac.cols())
                = skelBallJoints
                      ->getMarkerWorldPositionsJacobianWrtJointPositions(
                          markers);
            for (int i = 0; i < anatomicalMarkers.size(); i++)
            {
              if (!anatomicalMarkers[i])
              {
                diff.segment(i * 3, 3) *= 0.1;
                jac.block(i * 3, 0, 3, jac.cols()) *= 0.1;
              }
            }

            // 1.3.2.2. Next we need to compute the joint cluster error, and
            // joint cluster Jacobian
            Eigen::VectorXs jointPositions
                = skelBallJoints->getJointWorldPositions(joints);
            Eigen::MatrixXs jointJacobian
                = skelBallJoints
                      ->getJointWorldPositionsJacobianWrtJointPositions(joints);
            jac.block(
                   markerPositions.size(),
                   0,
                   jointClusters.size() * 3,
                   jac.cols())
                .setZero();
            for (int i = 0; i < jointClusters.size(); i++)
            {
              int clusterRow = markerPositions.size() + i * 3;
              Eigen::Vector3s clusterCenter = Eigen::Vector3s::Zero();
              for (int j : jointClusters[i])
              {
                clusterCenter += jointPositions.segment<3>(j * 3)
                                 / jointClusters[i].size();
                jac.block(clusterRow, 0, 3, jac.cols())
                    += jointJacobian.block(j * 3, 0, 3, jac.cols())
                       / jointClusters[i].size();
              }
              Eigen::Vector3s clusterTarget
                  = jointClusterTarget.segment<3>(i * 3);
              diff.segment<3>(clusterRow) = clusterCenter - clusterTarget;
            }
          },
          [&](Eigen::Ref<Eigen::VectorXs> pos) {
            pos = skelBallJoints->getRandomPose();
          },
          math::IKConfig()
              .setLogOutput(logOutput)
              .setMaxRestarts(100)
              .setConvergenceThreshold(1e-10));
    }
    // 2. Solve all clips using ordinary joints (no ball joints)
    // 2.1. Find a linearized list of markers to target
    std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
    std::vector<Eigen::Vector3s> markerPoses;
    std::vector<bool> anatomicalMarkers;
    for (int i = 0; i < mMarkers.size(); i++)
    {
      if (mMarkerObservations[t].count(mMarkerNames[i]))
      {
        markers.emplace_back(mMarkers[i]);
        markerPoses.push_back(mMarkerObservations[t][mMarkerNames[i]]);
        anatomicalMarkers.push_back(mMarkerIsAnatomical[i]);
      }
    }
    Eigen::VectorXs markerTarget = Eigen::VectorXs::Zero(markers.size() * 3);
    for (int i = 0; i < markers.size(); i++)
    {
      markerTarget.segment<3>(i * 3) = markerPoses[i];
    }

    // 2.2. Convert the visible joints over into pointers to the ball joints
    // skeleton
    std::vector<dynamics::Joint*> joints;
    std::vector<std::vector<int>> jointClusters;
    std::vector<Eigen::Vector3s> jointPoses;
    for (int i = 0; i < mStackedJoints.size(); i++)
    {
      if (mJointCenters[t].count(mStackedJoints[i]->name))
      {
        jointPoses.push_back(mJointCenters[t][mStackedJoints[i]->name]);
        std::vector<int> clusterIndices;
        for (dynamics::Joint* joint : mStackedJoints[i]->joints)
        {
          clusterIndices.push_back(joints.size());
          joints.push_back(mSkel->getJoint(joint->getName()));
        }
        jointClusters.push_back(clusterIndices);
      }
    }
    Eigen::VectorXs jointClusterTarget
        = Eigen::VectorXs::Zero(joints.size() * 3);
    for (int i = 0; i < joints.size(); i++)
    {
      jointClusterTarget.segment<3>(i * 3) = jointPoses[i];
    }

    // 2.3. Solve the actual IK
    s_t ikLoss = math::solveIK(
        mSkel->getPositions(),
        mSkel->getPositionUpperLimits(),
        mSkel->getPositionLowerLimits(),
        markerTarget.size() + jointClusterTarget.size(),
        [&](const Eigen::VectorXs& pos, bool clamp) {
          // 2.3.1. Set poses on the ball joint skeleton, by default not
          // clamping to limits
          mSkel->setPositions(pos);
          if (clamp)
          {
            // If we're clamping to limits, do it in the original skeleton
            // joint space, not in ball space
            mSkel->clampPositionsToLimits();
          }
          return mSkel->getPositions();
        },
        [&](Eigen::Ref<Eigen::VectorXs> diff, Eigen::Ref<Eigen::MatrixXs> jac) {
          // 2.3.2. Evaluate the error and the Jacobian relating dError / dPos

          // 2.3.2.1. First we need to compute the marker error, and marker
          // Jacobian
          Eigen::VectorXs markerPositions
              = mSkel->getMarkerWorldPositions(markers);
          diff.segment(0, markerPositions.size())
              = markerPositions - markerTarget;
          jac.block(0, 0, markerPositions.size(), jac.cols())
              = mSkel->getMarkerWorldPositionsJacobianWrtJointPositions(
                  markers);
          for (int i = 0; i < anatomicalMarkers.size(); i++)
          {
            if (!anatomicalMarkers[i])
            {
              diff.segment(i * 3, 3) *= 0.1;
              jac.block(i * 3, 0, 3, jac.cols()) *= 0.1;
            }
          }

          // 2.3.2.2. Next we need to compute the joint cluster error, and joint
          // cluster Jacobian
          Eigen::VectorXs jointPositions
              = mSkel->getJointWorldPositions(joints);
          Eigen::MatrixXs jointJacobian
              = mSkel->getJointWorldPositionsJacobianWrtJointPositions(joints);
          jac.block(
                 markerPositions.size(),
                 0,
                 jointClusters.size() * 3,
                 jac.cols())
              .setZero();
          for (int i = 0; i < jointClusters.size(); i++)
          {
            int clusterRow = markerPositions.size() + i * 3;
            Eigen::Vector3s clusterCenter = Eigen::Vector3s::Zero();
            for (int j : jointClusters[i])
            {
              clusterCenter
                  += jointPositions.segment<3>(j * 3) / jointClusters[i].size();
              jac.block(clusterRow, 0, 3, jac.cols())
                  += jointJacobian.block(j * 3, 0, 3, jac.cols())
                     / jointClusters[i].size();
            }
            Eigen::Vector3s clusterTarget
                = jointClusterTarget.segment<3>(i * 3);
            diff.segment<3>(clusterRow) = clusterCenter - clusterTarget;
          }
        },
        [&](Eigen::Ref<Eigen::VectorXs> pos) { pos = mSkel->getRandomPose(); },
        math::IKConfig()
            .setLogOutput(logOutput)
            .setMaxRestarts(1)
            .setStartClamped(true)
            .setConvergenceThreshold(1e-10));

    // 3. Save the result
    mPoses.push_back(mSkel->getPositions());
    mPosesClosedFormEstimateAvailable.push_back(
        Eigen::VectorXi::Zero(mSkel->getNumDofs()));
    avgLoss += ikLoss;
    lastPose = mSkel->getPositions();
  }
  avgLoss /= mMarkerObservations.size();
  return avgLoss;
}

//==============================================================================
/// WARNING: You must have already called estimateGroupScalesClosedForm()!
/// This uses the joint centers to estimate the body positions.
s_t IKInitializer::estimatePosesClosedForm(bool logOutput)
{
  Eigen::VectorXs originalPose = mSkel->getPositions();
  mSkel->setPositions(Eigen::VectorXs::Zero(mSkel->getNumDofs()));

  // Compute the joint world positions
  Eigen::VectorXs jointWorldPositions
      = mSkel->getJointWorldPositions(mSkel->getJoints());

  // 1. We'll want to use annotations on top of our stacked bodies, so construct
  // structs to make that easier.
  std::vector<AnnotatedStackedBody> annotatedGroups;
  for (int i = 0; i < mStackedBodies.size(); i++)
  {
    AnnotatedStackedBody group;
    group.stackedBody = mStackedBodies[i];
    annotatedGroups.push_back(group);
  }
  // 1.1.1. Compute body relative transforms
  for (int i = 0; i < annotatedGroups.size(); i++)
  {
    for (int j = 0; j < annotatedGroups[i].stackedBody->bodies.size(); j++)
    {
      annotatedGroups[i].relativeTransforms.push_back(
          annotatedGroups[i]
              .stackedBody->bodies[0]
              ->getWorldTransform()
              .inverse()
          * annotatedGroups[i].stackedBody->bodies[j]->getWorldTransform());
    }
  }
  // 1.1.2. Compute adjacent (stacked) joints and their relative transforms
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    std::shared_ptr<struct StackedJoint> joint = mStackedJoints[i];
    Eigen::Vector3s jointCenter = getStackedJointCenterFromJointCentersVector(
        joint, jointWorldPositions);

    for (auto& annotatedGroup : annotatedGroups)
    {
      if (joint->childBody == annotatedGroup.stackedBody
          || joint->parentBody == annotatedGroup.stackedBody)
      {
        annotatedGroup.adjacentJoints.push_back(joint);
        annotatedGroup.adjacentJointCenters.push_back(
            annotatedGroup.stackedBody->bodies[0]->getWorldTransform().inverse()
            * jointCenter);
      }
    }
  }
  // 1.1.3. Compute adjacent markers and their relative transforms
  Eigen::VectorXs markerWorldPositions
      = mSkel->getMarkerWorldPositions(mMarkers);
  for (int i = 0; i < mMarkers.size(); i++)
  {
    auto& marker = mMarkers[i];
    for (auto& weldedGroup : annotatedGroups)
    {
      if (std::find(
              weldedGroup.stackedBody->bodies.begin(),
              weldedGroup.stackedBody->bodies.end(),
              marker.first)
          != weldedGroup.stackedBody->bodies.end())
      {
        weldedGroup.adjacentMarkers.push_back(mMarkerNames[i]);
        Eigen::Vector3s markerCenter = markerWorldPositions.segment<3>(i * 3);
        weldedGroup.adjacentMarkerCenters.push_back(
            weldedGroup.stackedBody->bodies[0]->getWorldTransform().inverse()
            * markerCenter);
      }
    }
  }
  const Eigen::MatrixXi& isJointParentOf = mSkel->getJointParentMap();
  Eigen::MatrixXi isStackedJointParentOfBody
      = Eigen::MatrixXi::Zero(mStackedJoints.size(), mStackedBodies.size());
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    for (int j = 0; j < mStackedBodies.size(); j++)
    {
      for (auto* potentialParent : mStackedJoints[i]->joints)
      {
        for (auto* potentialChildBody : mStackedBodies[j]->bodies)
        {
          if ((potentialParent == potentialChildBody->getParentJoint())
              || (isJointParentOf(
                      potentialParent->getJointIndexInSkeleton(),
                      potentialChildBody->getParentJoint()
                          ->getJointIndexInSkeleton())
                  > 0))
          {
            isStackedJointParentOfBody(i, j) = 1;
            break;
          }
        }
        if (isStackedJointParentOfBody(i, j))
          break;
      }
    }
  }

  // 2. Now we can go through each timestep and do "closed form IK", by first
  // estimating the body positions and then estimating the joint angles to
  // achieve those body positions.
  mPoses.clear();
  mBodyTransforms.clear();
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    // 2.1. Estimate the body positions in world space of each welded group from
    // the joint estimates and marker estimates, when they're available.
    std::map<std::string, Eigen::Isometry3s> estimatedBodyWorldTransforms;
    for (auto& annotatedGroup : annotatedGroups)
    {
      // 2.1.1. Collect all the visible points in world space that we can use to
      // estimate the body transform
      std::vector<Eigen::Vector3s> visibleAdjacentPointsInWorldSpace;
      std::vector<std::string> visibleAdjacentPointNames;
      std::vector<Eigen::Vector3s> visibleAdjacentPointsInLocalSpace;
      std::vector<s_t> visibleAdjacentPointWeights;
      for (int j = 0; j < annotatedGroup.adjacentJoints.size(); j++)
      {
        if (mJointCenters[t].count(annotatedGroup.adjacentJoints[j]->name) > 0)
        {
          visibleAdjacentPointsInWorldSpace.push_back(
              mJointCenters[t][annotatedGroup.adjacentJoints[j]->name]);
          visibleAdjacentPointsInLocalSpace.push_back(
              annotatedGroup.adjacentJointCenters[j]);
          visibleAdjacentPointNames.push_back(
              "Joint " + annotatedGroup.adjacentJoints[j]->name);
          JointCenterEstimateSource estimate
              = mJointCentersEstimateSource[t].at(
                  annotatedGroup.adjacentJoints[j]->name);
          if (estimate == JointCenterEstimateSource::MDS)
          {
            // Pretty much ignore MDS estimates when scaling, since those
            // estimates are based on our scaling guesses in the first place, so
            // that introduces some circularity into the process
            visibleAdjacentPointWeights.push_back(1e-3);
          }
          else if (estimate == JointCenterEstimateSource::LEAST_SQUARES_EXACT)
          {
            visibleAdjacentPointWeights.push_back(1.0);
          }
          else if (estimate == JointCenterEstimateSource::LEAST_SQUARES_AXIS)
          {
            visibleAdjacentPointWeights.push_back(0.5);
          }
        }
      }
      for (int j = 0; j < annotatedGroup.adjacentMarkers.size(); j++)
      {
        if (mMarkerObservations[t].count(annotatedGroup.adjacentMarkers[j]) > 0)
        {
          visibleAdjacentPointsInWorldSpace.push_back(
              mMarkerObservations[t][annotatedGroup.adjacentMarkers[j]]);
          visibleAdjacentPointsInLocalSpace.push_back(
              annotatedGroup.adjacentMarkerCenters[j]);
          visibleAdjacentPointNames.push_back(
              "Marker " + annotatedGroup.adjacentMarkers[j]);
          bool isAnatomical = mMarkerIsAnatomical
              [mMarkerNameToIndex[annotatedGroup.adjacentMarkers[j]]];
          visibleAdjacentPointWeights.push_back(isAnatomical ? 1.0 : 0.01);
        }
      }
      assert(
          visibleAdjacentPointsInLocalSpace.size()
          == visibleAdjacentPointsInWorldSpace.size());

      s_t visibleAdjacentPointsWeightSum = 0.0;
      for (int j = 0; j < visibleAdjacentPointsInLocalSpace.size(); j++)
      {
        visibleAdjacentPointsWeightSum += visibleAdjacentPointWeights[j];
      }

      // We need at least 3 points, and they need to be semi-reliable point
      // estimates
      if (visibleAdjacentPointsInLocalSpace.size() >= 3
          && visibleAdjacentPointsWeightSum >= 2.0)
      {
        // 3.1.2. Compute the root body transform from the visible points, and
        // then use that to compute the other body transforms
        Eigen::Isometry3s rootBodyWorldTransform
            = getPointCloudToPointCloudTransform(
                visibleAdjacentPointsInLocalSpace,
                visibleAdjacentPointsInWorldSpace,
                visibleAdjacentPointWeights);
        for (int j = 0; j < annotatedGroup.stackedBody->bodies.size(); j++)
        {
          estimatedBodyWorldTransforms[annotatedGroup.stackedBody->bodies[j]
                                           ->getName()]
              = rootBodyWorldTransform * annotatedGroup.relativeTransforms[j];
        }

        s_t error = 0.0;
        for (int j = 0; j < visibleAdjacentPointsInLocalSpace.size(); j++)
        {
          Eigen::Vector3s reconstructedPoint
              = rootBodyWorldTransform * visibleAdjacentPointsInLocalSpace[j];
          error += (visibleAdjacentPointsInWorldSpace[j] - reconstructedPoint)
                       .norm();
        }
        error /= visibleAdjacentPointsInLocalSpace.size();

        if (logOutput)
        {
          std::cout << "Estimated bodies ";
          for (int j = 0; j < annotatedGroup.stackedBody->bodies.size(); j++)
          {
            std::cout << "\""
                      << annotatedGroup.stackedBody->bodies[j]->getName()
                      << "\" ";
          }
          std::cout << " at time " << t << " with error " << error << "m from "
                    << visibleAdjacentPointsInLocalSpace.size()
                    << " points: " << std::endl;
          for (int j = 0; j < visibleAdjacentPointsInLocalSpace.size(); j++)
          {
            std::cout << "  " << visibleAdjacentPointNames[j] << ": "
                      << visibleAdjacentPointsInWorldSpace[j].transpose()
                      << " reconstructed as "
                      << (rootBodyWorldTransform
                          * visibleAdjacentPointsInLocalSpace[j])
                             .transpose()
                      << std::endl;
          }
        }
      }
    }
    mBodyTransforms.push_back(estimatedBodyWorldTransforms);

    // 2.2. Now that we have the estimated body positions, we can estimate the
    // joint angles.
    Eigen::VectorXs jointAngles = Eigen::VectorXs::Zero(mSkel->getNumDofs());
    Eigen::VectorXi jointAnglesClosedFormEstimate
        = Eigen::VectorXi::Zero(mSkel->getNumDofs());

    int NUM_PASSES = 1;
    for (int pass = 0; pass < NUM_PASSES; pass++)
    {
      std::map<std::string, bool> solvedJoints;
      std::map<std::string, bool> jointsImpossibleToSolve;
      std::map<std::string, Eigen::Isometry3s> jointChildTransform;
      if (logOutput)
      {
        std::cout << "Running IK Pass " << pass << "/" << NUM_PASSES
                  << " upward" << std::endl;
      }
      while (true)
      {
        bool foundJointToSolve = false;
        // We want to accumulate snowballs of joints as we roll up the kinematic
        // tree, from the leaves inwards. We want to solve the leaves using just
        // the world transform we already have computed, but then for higher
        // level joints we want to also include the solved children in the point
        // cloud fit.
        for (int i = 0; i < mStackedJoints.size(); i++)
        {
          auto& joint = mStackedJoints[i];
          if (solvedJoints.count(joint->name))
            continue;
          // Only estimate joint angles for joints connecting two bodies that we
          // have estimated locations for
          if (!(joint->parentBody == nullptr
                || estimatedBodyWorldTransforms.count(joint->parentBody->name))
              || estimatedBodyWorldTransforms.count(joint->childBody->name)
                     == 0)
          {
            jointsImpossibleToSolve[joint->name] = true;
            continue;
          }

          bool solvedAllChildren = true;
          for (int i = 0; i < joint->childBody->childJoints.size(); i++)
          {
            // If we haven't solved this child joint yet, and the child joint
            // hasn't been marked as being "impossible to solve" because it's
            // attached to bodies that don't have defined transforms, then we
            // aren't ready to solve this joint yet
            if ((solvedJoints.count(joint->childBody->childJoints[i]->name)
                 == 0)
                && (jointsImpossibleToSolve.count(
                        joint->childBody->childJoints[i]->name)
                    == 0))
            {
              solvedAllChildren = false;
              break;
            }
          }

          if (!solvedAllChildren)
            continue;

          // 2.2.1. Get the relative transformation we estimate is taking place
          // over the joint
          Eigen::Isometry3s parentTransform = Eigen::Isometry3s::Identity();
          if (joint->parentBody != nullptr)
          {
            parentTransform
                = estimatedBodyWorldTransforms[joint->parentBody->name];
          }
          Eigen::Isometry3s childTransform
              = estimatedBodyWorldTransforms[joint->childBody->name];

          if (joint->childBody->childJoints.size() == 0)
          {
            // 2.2.1. If this is a leaf node, read the transform off of the body
            // transforms already computed
          }
          else
          {
            // 2.2.2. If we've got children, we want to include their joint
            // angles in the point cloud fit. This is somewhat more involved.
            // First step is to set the skeleton into the position we've
            // estimated so far.
            mSkel->setPositions(jointAngles);
            Eigen::VectorXs jointWorldPositions
                = mSkel->getJointWorldPositions(mSkel->getJoints());
            Eigen::VectorXs markerWorldPositions
                = mSkel->getMarkerWorldPositions(mMarkers);
            Eigen::Isometry3s rootBodyWorldTransform
                = joint->childBody->bodies[0]->getWorldTransform();

            // 2.2.3. Now we need to go through the children and get a point
            // cloud in the coordinates of the root child body
            std::vector<Eigen::Vector3s> visibleAdjacentPointsInWorldSpace;
            std::vector<Eigen::Vector3s> visibleAdjacentPointsInLocalSpace;
            std::vector<s_t> visibleAdjacentPointWeights;

            for (int j = 0; j < annotatedGroups.size(); j++)
            {
              if (isStackedJointParentOfBody(i, j))
              {
                s_t selfMultiplier = (i == j) ? 1.0 : 0.1;

                auto& annotatedGroup = annotatedGroups[j];
                for (int j = 0; j < annotatedGroup.adjacentJoints.size(); j++)
                {
                  if (mJointCenters[t].count(
                          annotatedGroup.adjacentJoints[j]->name)
                      > 0)
                  {
                    visibleAdjacentPointsInWorldSpace.push_back(
                        mJointCenters[t]
                                     [annotatedGroup.adjacentJoints[j]->name]);
                    visibleAdjacentPointsInLocalSpace.push_back(
                        rootBodyWorldTransform.inverse()
                        * getStackedJointCenterFromJointCentersVector(
                            annotatedGroup.adjacentJoints[j],
                            jointWorldPositions));
                    visibleAdjacentPointWeights.push_back(selfMultiplier * 1.0);
                  }
                }
                for (int j = 0; j < annotatedGroup.adjacentMarkers.size(); j++)
                {
                  if (mMarkerObservations[t].count(
                          annotatedGroup.adjacentMarkers[j])
                      > 0)
                  {
                    visibleAdjacentPointsInWorldSpace.push_back(
                        mMarkerObservations[t]
                                           [annotatedGroup.adjacentMarkers[j]]);
                    int markerIndex = std::find(
                                          mMarkerNames.begin(),
                                          mMarkerNames.end(),
                                          annotatedGroup.adjacentMarkers[j])
                                      - mMarkerNames.begin();
                    visibleAdjacentPointsInLocalSpace.push_back(
                        rootBodyWorldTransform.inverse()
                        * markerWorldPositions.segment<3>(markerIndex * 3));
                    visibleAdjacentPointWeights.push_back(
                        selfMultiplier * 0.01);
                  }
                }
              }
            }
            assert(
                visibleAdjacentPointsInLocalSpace.size()
                == visibleAdjacentPointsInWorldSpace.size());

            // 3.1.2. Compute the root body transform from the visible points,
            // and then use that to compute the other body transforms
            childTransform = getPointCloudToPointCloudTransform(
                visibleAdjacentPointsInLocalSpace,
                visibleAdjacentPointsInWorldSpace,
                visibleAdjacentPointWeights);

            if (NUM_PASSES > 1)
            {
              // Update the estimated body transform for next iteration
              estimatedBodyWorldTransforms[joint->childBody->name]
                  = childTransform;
            }
          }
          jointChildTransform[joint->name] = childTransform;

          Eigen::Isometry3s totalJointTransform
              = parentTransform.inverse() * childTransform;

          Eigen::Isometry3s remainingJointTransform = totalJointTransform;
          for (int j = 0; j < joint->joints.size(); j++)
          {
            dynamics::Joint* subJoint = joint->joints[j];

            // 2.2.2. Convert the remaining estimated tranformation into joint
            // coordinates on this joint
            Eigen::VectorXs pos = subJoint->getNearestPositionToDesiredRotation(
                remainingJointTransform.linear());
            subJoint->setPositions(pos);
            Eigen::Isometry3s recoveredJointTransform
                = subJoint->getRelativeTransform();
            if (subJoint->getType() == dynamics::FreeJoint::getStaticType()
                || subJoint->getType()
                       == dynamics::EulerFreeJoint::getStaticType())
            {
              Eigen::Vector3s translationOffset
                  = (recoveredJointTransform.translation()
                     - totalJointTransform.translation());
              pos.tail<3>() -= translationOffset;
              subJoint->setPositions(pos);
              recoveredJointTransform = subJoint->getRelativeTransform();
            }
            Eigen::Isometry3s newRemainingJointTransform
                = recoveredJointTransform.inverse() * remainingJointTransform;

            if (logOutput)
            {
              s_t translationError = (totalJointTransform.translation()
                                      - recoveredJointTransform.translation())
                                         .norm();
              s_t rotationError = (totalJointTransform.linear()
                                   - recoveredJointTransform.linear())
                                      .norm();
              std::cout << "Estimating joint " << subJoint->getName()
                        << " from adjacent body transforms with error "
                        << translationError << "m and " << rotationError
                        << " on rotation" << std::endl;
            }
            remainingJointTransform = newRemainingJointTransform;

            // 2.2.3. Save our estimate back to our joint angles
            jointAngles.segment(
                subJoint->getIndexInSkeleton(0), subJoint->getNumDofs())
                = pos;
            jointAnglesClosedFormEstimate
                .segment(
                    subJoint->getIndexInSkeleton(0), subJoint->getNumDofs())
                .setConstant(1.0);
          }

          solvedJoints[joint->name] = true;
          foundJointToSolve = true;
        }

        if (!foundJointToSolve)
        {
          // All the joints should either be marked as "solved" or "impossible
          // to solve"
          assert(
              jointsImpossibleToSolve.size() + solvedJoints.size()
              == mStackedJoints.size());
          break;
        }
      }

      // Now we want to do one more pass, this time going from the root back out
      // to the leaves, where we can re-estimate the joint angles now that the
      // parent body may have changed its location.

      if (logOutput)
      {
        std::cout << "Running IK Pass " << pass << "/" << NUM_PASSES
                  << " downward" << std::endl;
      }
      std::map<std::string, bool> solvedParents;
      while (true)
      {
        bool foundJointToSolve = false;
        // We want to accumulate snowballs of joints as we roll up the kinematic
        // tree, from the leaves inwards. We want to solve the leaves using just
        // the world transform we already have computed, but then for higher
        // level joints we want to also include the solved children in the point
        // cloud fit.
        for (int i = 0; i < mStackedJoints.size(); i++)
        {
          // No need to solve twice
          if (solvedParents.count(mStackedJoints[i]->name))
            continue;

          // If we haven't solved the parent joint yet, continue
          if (mStackedJoints[i]->parentBody != nullptr
              && solvedParents.count(
                     mStackedJoints[i]->parentBody->parentJoint->name)
                     == 0)
            continue;

          // If this joint isn't connected on both sides to bodies we've got
          // estimated transforms for, then we can't solve it
          if (jointsImpossibleToSolve.count(mStackedJoints[i]->name))
            continue;

          std::shared_ptr<struct StackedJoint> joint = mStackedJoints[i];

          Eigen::Isometry3s parentTransform = Eigen::Isometry3s::Identity();
          if (joint->parentBody != nullptr)
          {
            mSkel->setPositions(jointAngles);
            parentTransform = joint->parentBody->bodies[0]->getWorldTransform();
            if (NUM_PASSES > 1)
            {
              // Update the estimated body transform for next iteration
              estimatedBodyWorldTransforms[joint->parentBody->name]
                  = parentTransform;
            }
          }
          assert(jointChildTransform.count(joint->name) > 0);
          Eigen::Isometry3s childTransform = jointChildTransform[joint->name];

          Eigen::Isometry3s totalJointTransform
              = parentTransform.inverse() * childTransform;

          Eigen::Isometry3s remainingJointTransform = totalJointTransform;
          assert(remainingJointTransform.translation().allFinite());
          assert(remainingJointTransform.linear().allFinite());
          for (int j = 0; j < joint->joints.size(); j++)
          {
            dynamics::Joint* subJoint = joint->joints[j];

            // 2.2.2. Convert the remaining estimated tranformation into joint
            // coordinates on this joint
            Eigen::VectorXs pos = subJoint->getNearestPositionToDesiredRotation(
                remainingJointTransform.linear());
            assert(pos.allFinite());
            subJoint->setPositions(pos);
            Eigen::Isometry3s recoveredJointTransform
                = subJoint->getRelativeTransform();
            assert(recoveredJointTransform.linear().allFinite());
            assert(recoveredJointTransform.translation().allFinite());
            if (subJoint->getType() == dynamics::FreeJoint::getStaticType()
                || subJoint->getType()
                       == dynamics::EulerFreeJoint::getStaticType())
            {
              Eigen::Vector3s translationOffset
                  = (recoveredJointTransform.translation()
                     - totalJointTransform.translation());
              assert(translationOffset.allFinite());
              pos.tail<3>() -= translationOffset;
              assert(pos.tail<3>().allFinite());
              subJoint->setPositions(pos);
              recoveredJointTransform = subJoint->getRelativeTransform();
            }
            assert(recoveredJointTransform.translation().allFinite());
            assert(recoveredJointTransform.linear().allFinite());

            if (logOutput)
            {
              s_t translationError = (remainingJointTransform.translation()
                                      - recoveredJointTransform.translation())
                                         .norm();
              s_t rotationError = (remainingJointTransform.linear()
                                   - recoveredJointTransform.linear())
                                      .norm();
              assert(
                  rotationError < 1e3); // This should be numerically impossible
              std::cout
                  << "Re-estimating (this time with a known parent) joint "
                  << subJoint->getName()
                  << " from adjacent body transforms with error "
                  << translationError << "m and " << rotationError
                  << " on rotation" << std::endl;
            }

            assert(remainingJointTransform.translation().allFinite());
            assert(remainingJointTransform.linear().allFinite());
            remainingJointTransform
                = recoveredJointTransform.inverse() * remainingJointTransform;

            // 2.2.3. Save our estimate back to our joint angles
            jointAngles.segment(
                subJoint->getIndexInSkeleton(0), subJoint->getNumDofs())
                = pos;
            jointAnglesClosedFormEstimate
                .segment(
                    subJoint->getIndexInSkeleton(0), subJoint->getNumDofs())
                .setConstant(1.0);
          }

          solvedParents[mStackedJoints[i]->name] = true;
          foundJointToSolve = true;
          break;
        }
        if (!foundJointToSolve)
        {
          break;
        }
      }
    }

    mPoses.push_back(jointAngles);
    mPosesClosedFormEstimateAvailable.push_back(jointAnglesClosedFormEstimate);
  }

  mSkel->setPositions(originalPose);

  // TODO: compute marker reconstruction error and return that
  return 0.0;
}

//==============================================================================
/// This gets the average distance between adjacent joint centers in our
/// current joint center estimates.
std::map<std::string, std::map<std::string, s_t>>
IKInitializer::estimateJointToJointDistances()
{
  // 1. Recompute the joint to joint squared distances

  std::map<std::string, std::map<std::string, s_t>> jointToJointAvgDistances;
  std::map<std::string, std::map<std::string, int>>
      jointToJointAvgDistancesCount;
  for (int t = 0; t < mMarkerObservations.size(); t++)
  {
    for (auto& pair : mJointToJointSquaredDistances)
    {
      std::string joint1Name = pair.first;
      for (auto& pair2 : pair.second)
      {
        std::string joint2Name = pair2.first;
        if (joint1Name == joint2Name)
        {
          continue;
        }
        if (mJointCenters[t].count(joint1Name) == 0)
        {
          continue;
        }
        if (mJointCenters[t].count(joint2Name) == 0)
        {
          continue;
        }
        Eigen::Vector3s joint1Center = mJointCenters[t][joint1Name];
        Eigen::Vector3s joint2Center = mJointCenters[t][joint2Name];
        s_t distance = (joint1Center - joint2Center).norm();
        if (jointToJointAvgDistances[joint1Name].count(joint2Name) == 0)
        {
          jointToJointAvgDistances[joint1Name][joint2Name] = 0;
          jointToJointAvgDistancesCount[joint1Name][joint2Name] = 0;
        }
        jointToJointAvgDistances[joint1Name][joint2Name] += distance;
        jointToJointAvgDistancesCount[joint1Name][joint2Name]++;
      }
    }
  }
  for (auto& pair : jointToJointAvgDistances)
  {
    std::string joint1Name = pair.first;
    for (auto& pair2 : pair.second)
    {
      std::string joint2Name = pair2.first;
      jointToJointAvgDistances[joint1Name][joint2Name]
          /= jointToJointAvgDistancesCount[joint1Name][joint2Name];
    }
  }
  return jointToJointAvgDistances;
}

//==============================================================================
/// This gets the subset of markers that are visible at a given timestep
std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
IKInitializer::getObservedMarkers(int t)
{
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>> markers;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (mMarkerObservations[t].count(mMarkerNames[i]))
    {
      markers.push_back(mMarkers[i]);
    }
  }
  return markers;
}

//==============================================================================
/// This gets the subset of markers that are visible at a given timestep
std::vector<std::string> IKInitializer::getObservedMarkerNames(int t)
{
  std::vector<std::string> markerNames;
  for (int i = 0; i < mMarkerNames.size(); i++)
  {
    if (mMarkerObservations[t].count(mMarkerNames[i]))
    {
      markerNames.push_back(mMarkerNames[i]);
    }
  }
  return markerNames;
}

//==============================================================================
/// This gets the subset of joints that are attached to markers that are
/// visible at a given timestep
std::vector<std::shared_ptr<struct StackedJoint>>
IKInitializer::getJointsAttachedToObservedMarkers(int t)
{
  std::vector<std::shared_ptr<struct StackedJoint>> joints;
  for (int i = 0; i < mStackedJoints.size(); i++)
  {
    std::shared_ptr<struct StackedJoint> joint = mStackedJoints[i];
    if (mJointToMarkerSquaredDistances.count(joint->name) == 0)
    {
      continue;
    }
    for (auto& pair : mJointToMarkerSquaredDistances[joint->name])
    {
      if (mMarkerObservations[t].count(pair.first))
      {
        joints.push_back(joint);
        break;
      }
    }
  }
  return joints;
}

//==============================================================================
/// This gets the world center estimates for joints that are attached to
/// markers that are visible at this timestep.
std::map<std::string, Eigen::Vector3s>
IKInitializer::getJointsAttachedToObservedMarkersCenters(int t)
{
  return mJointCenters[t];
}

//==============================================================================
/// This gets the squared distance between a joint and a marker on an adjacent
/// body segment.
std::map<std::string, s_t> IKInitializer::getJointToMarkerSquaredDistances(
    std::string jointName)
{
  return mJointToMarkerSquaredDistances[jointName];
}

//==============================================================================
/// This will reconstruct a centered Euclidean point cloud from a distance
/// matrix.
Eigen::MatrixXs IKInitializer::getPointCloudFromDistanceMatrix(
    const Eigen::MatrixXs& distances)
{
  int n = distances.rows();
  Eigen::MatrixXs centering_matrix = Eigen::MatrixXs::Identity(n, n)
                                     - Eigen::MatrixXs::Constant(n, n, 1.0 / n);
  Eigen::MatrixXs double_centering_matrix
      = -0.5 * centering_matrix * distances * centering_matrix;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXs> eigensolver(
      double_centering_matrix);
  Eigen::VectorXs eigenvalues = eigensolver.eigenvalues();
  Eigen::MatrixXs eigenvectors = eigensolver.eigenvectors();
  assert(eigensolver.info() == Eigen::Success);
  assert(!eigenvalues.hasNaN());
  assert(!eigenvectors.hasNaN());

  int k = 3;
  Eigen::MatrixXs k_eigenvectors = eigenvectors.rightCols(k);
  Eigen::VectorXs k_eigenvalues
      = eigenvalues.tail(k).cwiseMax(1e-16).cwiseSqrt();
  assert(!k_eigenvalues.hasNaN());
  assert(!k_eigenvectors.hasNaN());

  return k_eigenvalues.asDiagonal() * k_eigenvectors.transpose();
}

//==============================================================================
/// This will rotate and translate a point cloud to match the first N points
/// as closely as possible to the passed in matrix
Eigen::MatrixXs IKInitializer::mapPointCloudToData(
    const Eigen::MatrixXs& pointCloud,
    std::vector<Eigen::Vector3s> firstNPoints)
{
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> targetPointCloud
      = Eigen::Matrix<s_t, 3, Eigen::Dynamic>::Zero(3, firstNPoints.size());
  for (int i = 0; i < firstNPoints.size(); i++)
  {
    targetPointCloud.col(i) = firstNPoints[i];
  }
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> sourcePointCloud
      = pointCloud.block(0, 0, 3, firstNPoints.size());

  assert(sourcePointCloud.cols() == targetPointCloud.cols());

  // Compute the centroids of the source and target points
  Eigen::Vector3s sourceCentroid = sourcePointCloud.rowwise().mean();
  Eigen::Vector3s targetCentroid = targetPointCloud.rowwise().mean();

#ifndef NDEBUG
  Eigen::Vector3s sourceAvg = Eigen::Vector3s::Zero();
  Eigen::Vector3s targetAvg = Eigen::Vector3s::Zero();
  for (int i = 0; i < sourcePointCloud.cols(); i++)
  {
    sourceAvg += sourcePointCloud.col(i);
    targetAvg += targetPointCloud.col(i);
  }
  sourceAvg /= sourcePointCloud.cols();
  targetAvg /= targetPointCloud.cols();
  assert((sourceAvg - sourceCentroid).norm() < 1e-12);
  assert((targetAvg - targetCentroid).norm() < 1e-12);
#endif

  // Compute the centered source and target points
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> centeredSourcePoints
      = sourcePointCloud.colwise() - sourceCentroid;
  Eigen::Matrix<s_t, 3, Eigen::Dynamic> centeredTargetPoints
      = targetPointCloud.colwise() - targetCentroid;

#ifndef NDEBUG
  assert(std::abs(centeredSourcePoints.rowwise().mean().norm()) < 1e-8);
  assert(std::abs(centeredTargetPoints.rowwise().mean().norm()) < 1e-8);
  for (int i = 0; i < sourcePointCloud.cols(); i++)
  {
    Eigen::Vector3s expectedCenteredSourcePoints
        = sourcePointCloud.col(i) - sourceAvg;
    assert(
        (centeredSourcePoints.col(i) - expectedCenteredSourcePoints).norm()
        < 1e-12);
    Eigen::Vector3s expectedCenteredTargetPoints
        = targetPointCloud.col(i) - targetAvg;
    assert(
        (centeredTargetPoints.col(i) - expectedCenteredTargetPoints).norm()
        < 1e-12);
  }
#endif

  // Compute the covariance matrix
  Eigen::Matrix3s covarianceMatrix
      = centeredTargetPoints * centeredSourcePoints.transpose();

  // Compute the singular value decomposition of the covariance matrix
  Eigen::JacobiSVD<Eigen::Matrix3s> svd(
      covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3s U = svd.matrixU();
  Eigen::Matrix3s V = svd.matrixV();

  // Compute the rotation matrix and translation vector
  Eigen::Matrix3s R = U * V.transpose();
  // Normally, we would want to check the determinant of R here, to ensure that
  // we're only doing right-handed rotations. HOWEVER, because we're using a
  // point cloud, we may actually have to flip the data along an axis to get it
  // to match up, so we skip the determinant check.

  // Transform the source point cloud to the target point cloud
  Eigen::MatrixXs transformed = Eigen::MatrixXs::Zero(3, pointCloud.cols());
  for (int i = 0; i < pointCloud.cols(); i++)
  {
    transformed.col(i)
        = R * (pointCloud.col(i).head<3>() - sourceCentroid) + targetCentroid;
  }
  return transformed;
}

//==============================================================================
/// This will give the world transform necessary to apply to the local points
/// (worldT * p[i] for all localPoints) to get the local points to match the
/// world points as closely as possible.
Eigen::Isometry3s IKInitializer::getPointCloudToPointCloudTransform(
    std::vector<Eigen::Vector3s> localPoints,
    std::vector<Eigen::Vector3s> worldPoints,
    std::vector<s_t> weights)
{
  assert(localPoints.size() > 0);
  assert(worldPoints.size() > 0);
  assert(localPoints.size() == worldPoints.size());

  // Compute the centroids of the local and world points
  Eigen::Vector3s localCentroid = Eigen::Vector3s::Zero();
  s_t sumWeights = 0.0;
  for (int i = 0; i < localPoints.size(); i++)
  {
    Eigen::Vector3s& point = localPoints[i];
    sumWeights += weights[i];
    localCentroid += point * weights[i];
  }
  localCentroid /= sumWeights;
  Eigen::Vector3s worldCentroid = Eigen::Vector3s::Zero();
  for (int i = 0; i < worldPoints.size(); i++)
  {
    Eigen::Vector3s& point = worldPoints[i];
    worldCentroid += point * weights[i];
  }
  worldCentroid /= sumWeights;

  // Compute the centered local and world points
  std::vector<Eigen::Vector3s> centeredLocalPoints;
  std::vector<Eigen::Vector3s> centeredWorldPoints;
  for (int i = 0; i < localPoints.size(); i++)
  {
    centeredLocalPoints.push_back(localPoints[i] - localCentroid);
    centeredWorldPoints.push_back(worldPoints[i] - worldCentroid);
  }

  // Compute the covariance matrix
  Eigen::Matrix3s covarianceMatrix = Eigen::Matrix3s::Zero();
  for (int i = 0; i < localPoints.size(); i++)
  {
    covarianceMatrix += weights[i] * centeredWorldPoints[i]
                        * centeredLocalPoints[i].transpose();
  }

  // Compute the singular value decomposition of the covariance matrix
  Eigen::JacobiSVD<Eigen::Matrix3s> svd(
      covarianceMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3s U = svd.matrixU();
  Eigen::Matrix3s V = svd.matrixV();

  // Compute the rotation matrix and translation vector
  Eigen::Matrix3s R = U * V.transpose();
  if (R.determinant() < 0)
  {
    Eigen::Matrix3s scales = Eigen::Matrix3s::Identity();
    scales(2, 2) = -1;
    R = U * scales * V.transpose();
  }
  Eigen::Vector3s translation = worldCentroid - R * localCentroid;

  Eigen::Isometry3s transform = Eigen::Isometry3s::Identity();
  transform.linear() = R;
  transform.translation() = translation;
  return transform;
}

//==============================================================================
/// This tries to solve the least-squares problem to get the local scales for
/// a body, such that the distances between the local points match the input
/// distances as closely as possible, with a weighted preference.
Eigen::Vector3s IKInitializer::getLocalScale(
    std::vector<Eigen::Vector3s> localPoints,
    std::vector<std::tuple<int, int, s_t, s_t>> pairDistancesWithWeights,
    s_t defaultAxisScale,
    bool logOutput)
{
  // Special case, if we have no observations, return the default
  if (pairDistancesWithWeights.size() == 0)
  {
    return Eigen::Vector3s::Constant(defaultAxisScale);
  }
  // Special case, if we have only one observation, then scale uniformly
  // according to that observation
  if (pairDistancesWithWeights.size() == 1)
  {
    Eigen::Vector3s& a = localPoints[std::get<0>(pairDistancesWithWeights[0])];
    Eigen::Vector3s& b = localPoints[std::get<1>(pairDistancesWithWeights[0])];
    s_t d = std::get<2>(pairDistancesWithWeights[0]);
    s_t w = std::get<3>(pairDistancesWithWeights[0]);
    (void)w;
    s_t ratio = d / (a - b).norm();
    if (ratio < 0.75 * defaultAxisScale || ratio > 1.25 * defaultAxisScale)
    {
      ratio = defaultAxisScale;
    }
    return Eigen::Vector3s::Constant(ratio);
  }
  // General case, we have more than one observation, and would like to scale
  // non-uniformly
  else
  {
    // Our basic formula is:
    // s[0]^2*(a[0]-b[0])^2 + s[1]^2*(a[1]-b[1])^2 + s[2]^2*(a[2]-b[2])^2 = d^2

    Eigen::MatrixXs A
        = Eigen::MatrixXs::Zero(pairDistancesWithWeights.size(), 3);
    Eigen::VectorXs distances
        = Eigen::VectorXs::Zero(pairDistancesWithWeights.size());

    for (int i = 0; i < pairDistancesWithWeights.size(); i++)
    {
      Eigen::Vector3s& a
          = localPoints[std::get<0>(pairDistancesWithWeights[i])];
      Eigen::Vector3s& b
          = localPoints[std::get<1>(pairDistancesWithWeights[i])];
      s_t d = std::get<2>(pairDistancesWithWeights[i]);
      s_t w = std::get<3>(pairDistancesWithWeights[i]);

      A(i, 0) = w * (a[0] - b[0]) * (a[0] - b[0]);
      A(i, 1) = w * (a[1] - b[1]) * (a[1] - b[1]);
      A(i, 2) = w * (a[2] - b[2]) * (a[2] - b[2]);
      distances(i) = w * d * d;
    }

    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::Vector3s scaleSquared = svd.solve(distances);
    Eigen::Vector3s scale = scaleSquared.cwiseAbs().cwiseSqrt();

    Eigen::Vector3s outputSensitivity = svd.matrixV() * svd.singularValues();
    if (logOutput)
    {
      std::cout << "Output sensitivity: " << outputSensitivity.transpose()
                << std::endl;
    }

    for (int i = 0; i < 3; i++)
    {
      // If the singular value is either zero (no info), or indicates that the
      // scale is way too sensitive (localPoints are too close together along
      // this axis), then we need to default this scale.
      if (abs(outputSensitivity(i)) < 0.002 || abs(outputSensitivity(i)) > 100)
      {
        scale(i) = defaultAxisScale;
      }
      // Also, if we're outside of reasonable scale bounds, just default that
      // axis, because it indicates that something went wrong in estimating.
      else if (
          scale(i) < 0.75 * defaultAxisScale
          || scale(i) > 1.25 * defaultAxisScale)
      {
        scale(i) = defaultAxisScale;
      }
    }

    return scale;
  }
}

//==============================================================================
/// This implements the method in "Constrained least-squares optimization for
/// robust estimation of center of rotation" by Chang and Pollard, 2006. This
/// is the simpler version, that only supports a single marker.
///
/// This assumes we're operating in a frame where the joint center is fixed in
/// place, and `markerTrace` is a list of marker positions over time in that
/// frame. So `markerTrace[t]` is the location of the marker on its t'th
/// observation. This method will return the joint center in that frame.
///
/// The benefit of using this instead of the
/// `closedFormPivotFindingJointCenterSolver` is that we can run it on pairs
/// of bodies where only one side of the pair has 3 markers on it (commonly,
/// the ankle).
///
/// IMPORTANT NOTE: This method assumes there are no markers that are
/// literally coincident with the joint center. It has a singularity in that
/// case, and will produce garbage. However, the only reason we would have a
/// marker on top of the joint center is if someone actually gave us marker
/// data with virtual joint centers already computed, in which case maybe we
/// just respect their wishes?
Eigen::Vector3s IKInitializer::getChangPollard2006JointCenterMultiMarker(
    std::vector<std::vector<Eigen::Vector3s>> markerTraces, bool log)
{
  // This algorithm performs really poorly if the joint center is less than
  // 0.1 units away from the markers, which often happens when your distance
  // metric is in meters. To get around this, we pre-scale all the data up by
  // SCALE_FACTOR, and scale down the result at the end by the same amount.
  const s_t SCALE_FACTOR = 50.0;

  int numMarkers = markerTraces.size();
  int uDim = 4 + numMarkers;

  std::vector<Eigen::MatrixXs> Ds;
  int sumRows = 0;
  for (auto& markerTrace : markerTraces)
  {
    // Construct the matrix D, described in Equation 12 in the paper
    Eigen::MatrixXs D = Eigen::MatrixXs::Zero(markerTrace.size(), 4);
    for (int i = 0; i < markerTrace.size(); i++)
    {
      Eigen::Vector3s scaledPoint = SCALE_FACTOR * markerTrace[i];
      D(i, 0) = scaledPoint.squaredNorm();
      D(i, 1) = scaledPoint(0);
      D(i, 2) = scaledPoint(1);
      D(i, 3) = scaledPoint(2);
    }
    if (log)
    {
      std::cout << "The data matrix D[" << Ds.size() << "] (scaled by "
                << SCALE_FACTOR << "): " << std::endl
                << D << std::endl;
    }
    Ds.push_back(D);
    sumRows += D.rows();
  }
  Eigen::MatrixXs D = Eigen::MatrixXs::Zero(sumRows, uDim);
  int rowCursor = 0;
  for (int i = 0; i < Ds.size(); i++)
  {
    D.block(rowCursor, 0, Ds[i].rows(), Ds[i].cols()) = Ds[i];
    D.block(rowCursor, 4 + i, Ds[i].rows(), 1).setConstant(1.0);
    rowCursor += Ds[i].rows();
  }
  if (log)
  {
    std::cout << "The assembled data matrix D (scaled by " << SCALE_FACTOR
              << "): " << std::endl
              << D << std::endl;
  }

  // Construct the matrix S, described in Equation 13 in the paper
  Eigen::MatrixXs S = D.transpose() * D;
  if (log)
  {
    std::cout << "The objective matrix S (scaled by " << SCALE_FACTOR
              << "^2): " << std::endl
              << S << std::endl;
  }

  // Construct the constraint matrix C, described in Equation 14 in the paper
  Eigen::MatrixXs C = Eigen::MatrixXs::Zero(uDim, uDim);
  // C(0, 0) = 1;
  C(1, 1) = numMarkers;
  C(2, 2) = numMarkers;
  C(3, 3) = numMarkers;
  for (int i = 4; i < uDim; i++)
  {
    C(i, 0) = -2;
    C(0, i) = -2;
  }
  if (log)
  {
    std::cout << "Constraint matrix C: " << std::endl << C << std::endl;
  }

  // Create a GeneralizedEigenSolver object
  Eigen::GeneralizedEigenSolver<Eigen::MatrixXs> solver;

  // Compute the generalized eigenvalues and eigenvectors
  solver.compute(S, C);

  auto result = solver.info();
  if (result == Eigen::NumericalIssue)
  {
    std::cout << "GeneralizedEigenSolver failed with Eigen::NumericalIssue"
              << std::endl;
  }
  else if (result == Eigen::NoConvergence)
  {
    std::cout << "GeneralizedEigenSolver failed with Eigen::NoConvergence"
              << std::endl;
  }
  else if (result == Eigen::InvalidInput)
  {
    std::cout << "GeneralizedEigenSolver failed with Eigen::InvalidInput"
              << std::endl;
  }
  assert(result == Eigen::Success);

  // Print the generalized eigenvalues - we use the opposite sign convention
  // for the eigenvalues as the solver
  Eigen::VectorXcd complexEigenvalues
      = solver.alphas().cwiseQuotient(solver.betas()).transpose();
  if (log)
  {
    std::cout << "The generalized eigenvalues of S and C are:\n"
              << complexEigenvalues << std::endl;
  }

  // Print the generalized eigenvectors
  Eigen::MatrixXcd complexEigenvectors = solver.eigenvectors();
  if (log)
  {
    std::cout << "The generalized eigenvectors of S and C are:\n"
              << complexEigenvectors << std::endl;
  }

  // Reconstruct the solution from the eigenvectors, described after equation
  // 17 in the paper. To quote: "The best-fit solution is the generalized
  // eigenvector u with non-negative eigenvalue l which has the least cost
  // according to Eq. (13) and subject to Eq. (14)"
  s_t bestCost = std::numeric_limits<s_t>::infinity();
  Eigen::Vector3s reconstructedCenter = Eigen::Vector3s::Zero();
  for (int i = 0; i < complexEigenvalues.size(); i++)
  {
    if (complexEigenvalues(i).imag() != 0
        || complexEigenvalues(i).real() <= 1e-12)
      continue;
    Eigen::VectorXs eigenvector = complexEigenvectors.col(i).real();

    // Normalized by constraint according to Equation 14 in the paper
    s_t constraint = eigenvector.dot(C * eigenvector);
    s_t scaleBy = sqrt(1.0 / constraint);
    if (!isfinite(scaleBy))
      continue;
    Eigen::VectorXs u = eigenvector * scaleBy;
    if (u.hasNaN())
      continue;
    s_t a = u(0);
    // This means we're in a numerically unstable fit, due to extremely low
    // marker noise (in practice this pretty much only happens on synthetic
    // data)
    if (std::abs(u(0)) < 1e-8)
    {
      continue;
    }

    if (log)
    {
      std::cout << "Un-Normalized result [" << i << "]: " << std::endl
                << eigenvector << std::endl;
      std::cout << "Constraint [" << i
                << "] (will be normalized to 1.0): " << constraint << std::endl;
      std::cout << "Normalized result [" << i << "]: " << std::endl
                << u << std::endl;
    }

#ifndef NDEBUG
    s_t eigenvalue = complexEigenvalues(i).real();
    // Double check that the rescaled eigenvector still satisfies the equation
    // (it must, cause everything is linear)
    Eigen::VectorXs S_u = S * u;
    Eigen::VectorXs C_u = C * u;
    Eigen::VectorXs diff = S_u - eigenvalue * C_u;
    s_t diffNorm = diff.norm() / max(1.0, S_u.norm());
    assert(diffNorm < 1e-6);
#endif

#ifndef NDEBUG
    if (log)
      std::cout << "Normalized Constraint [" << i << "]: " << u.dot(C * u)
                << std::endl;
    assert(abs(u.dot(C * u) - 1.0) < 1e-6);
#endif

    // Cost according to Equation 13 in the paper
    s_t cost = u.dot(S * u);
    cost /= a * a;
    if (log)
      std::cout << "Cost[" << i << "]: " << cost << std::endl;
    Eigen::Vector3s center = u.segment<3>(1) / (-2.0 * a);
    if (log)
      std::cout << "Center[" << i << "]: " << std::endl
                << (center / SCALE_FACTOR) << std::endl;

#ifndef NDEBUG
    if (markerTraces.size() == 1)
    {
      // Check that the polynomial loss actually maps back to the original
      // values
      s_t radiusSquared = (center.squaredNorm() - u(4) / u(0));
      s_t radius = sqrt(radiusSquared);
      for (int i = 0; i < markerTraces.size(); i++)
      {
        for (int j = 0; j < markerTraces[i].size(); j++)
        {
          Eigen::Vector5s polynomial;
          polynomial.head<4>() = Ds[i].row(j);
          polynomial(4) = 1.0;

          s_t algebraicDistance = polynomial.dot(u) / u(0);
          Eigen::Vector3s scaledPoint = SCALE_FACTOR * markerTraces[i][j];
          s_t originalDistance
              = (scaledPoint - center).squaredNorm() - radiusSquared;
          if (abs(algebraicDistance - originalDistance)
                  / max(1.0, abs(originalDistance))
              >= 1e-8)
          {
            std::cout << "Data polynomial failed to reconstruct data point ["
                      << scaledPoint.transpose()
                      << "] with distance from proposed center "
                      << (scaledPoint - center).norm()
                      << " and proposed radius " << radius << std::endl;
            std::cout << "Polynomial: " << polynomial.transpose() << std::endl;
            std::cout << "U: " << u.transpose() << std::endl;
            std::cout << "Algebraic distance: " << algebraicDistance
                      << std::endl;
            std::cout << "Reconstructed center: " << center.transpose()
                      << std::endl;
            std::cout << "Reconstructed radius: " << radius << std::endl;
            std::cout << "Original distance: " << originalDistance << std::endl;
            std::cout << "Normalized Error: "
                      << abs(algebraicDistance - originalDistance)
                             / max(1.0, abs(originalDistance))
                      << std::endl;
          }
          assert(
              abs(algebraicDistance - originalDistance)
                  / max(1.0, abs(originalDistance))
              < 1e-8);
        }
      }
    }
#endif

    if (cost < bestCost)
    {
      bestCost = cost;
      reconstructedCenter = center;
    }
  }

  if (!std::isfinite(bestCost))
  {
    // We fall back to a least-squares fit when there's no valid solution,
    // which generally means that there was no noise (and therefore no radius
    // variability) in the marker data.
    if (log)
    {
      std::cout << "Didn't get any valid solutions (perhaps because there was "
                   "no noise in the data and this joint only had a single "
                   "marker attached, which often happens with synthetic "
                   "data, or because there was a marker for the joint center). "
                   "Falling back to least-squares fit"
                << std::endl;
    }
    reconstructedCenter = leastSquaresConcentricSphereFit(markerTraces, log);
  }
  else
  {
    reconstructedCenter /= SCALE_FACTOR;
  }

  return reconstructedCenter;
}

//==============================================================================
/// This implements a simple least-squares problem to find the center of a
/// sphere of unknown radius, with samples along the hull given by `points`.
/// This tolerates noise less well than the ChangPollard2006 method, because
/// it biases the radius of the sphere towards zero in the presence of
/// ambiguity, because it is part of the least-squares terms. However, this
/// method will work even on data with zero noise, whereas ChangPollard2006
/// will fail when there is zero noise (for example, on synthetic datasets).
Eigen::Vector3s IKInitializer::leastSquaresConcentricSphereFit(
    std::vector<std::vector<Eigen::Vector3s>> traces, bool logOutput)
{
  int dim = 0;
  for (auto& trace : traces)
    dim += trace.size();

  std::vector<int> usableTimesteps;
  if (traces.size() > 0)
  {
    for (int i = 0; i < traces[0].size(); i++)
    {
      usableTimesteps.push_back(i);
    }
  }

  const int maxSamples = 500;
  if (usableTimesteps.size() > maxSamples)
  {
    std::vector<int> sampledIndices
        = math::evenlySpacedTimesteps(usableTimesteps.size(), maxSamples);
    std::vector<int> sampledTimesteps;
    for (int index : sampledIndices)
    {
      sampledTimesteps.push_back(usableTimesteps[index]);
    }
    usableTimesteps = sampledTimesteps;
  }

  Eigen::VectorXs f = Eigen::VectorXs::Zero(dim);
  Eigen::MatrixXs A = Eigen::MatrixXs::Zero(dim, 3 + traces.size());
  int rowCursor = 0;
  int colCursor = 3;
  for (auto& trace : traces)
  {
    for (int i : usableTimesteps)
    {
      Eigen::Vector3s& p = trace[i];
      f(rowCursor) = p.squaredNorm();

      A(rowCursor, 0) = 2 * p(0);
      A(rowCursor, 1) = 2 * p(1);
      A(rowCursor, 2) = 2 * p(2);
      A(rowCursor, colCursor) = 1;

      rowCursor++;
    }
    colCursor++;
  }
  Eigen::VectorXs c = A.completeOrthogonalDecomposition().solve(f);
  Eigen::Vector3s center = c.head<3>();
  if (logOutput)
  {
    std::cout << "Least-squares joint center fit:" << std::endl;
    std::cout << "Matrix A:" << std::endl << A << std::endl;
    std::cout << "Vector f:" << std::endl << f << std::endl;
    std::cout << "Vector c = A.solve(f):" << std::endl << c << std::endl;
    std::cout << "Resulting center:" << std::endl << center << std::endl;
  }
  return center;
}

//==============================================================================
/// This implements the least-squares method in Gamage and Lasenby 2002, "New
/// least squares solutions for estimating the average centre of rotation and
/// the axis of rotation"
std::pair<Eigen::Vector3s, s_t> IKInitializer::gamageLasenby2002AxisFit(
    std::vector<std::vector<Eigen::Vector3s>> traces)
{
  Eigen::MatrixXs A = Eigen::Matrix3s::Zero();
  for (std::vector<Eigen::Vector3s> trace : traces)
  {
    Eigen::Vector3s mean = Eigen::Vector3s::Zero();
    Eigen::Matrix3s meanOuterProduct = Eigen::Matrix3s::Zero();
    for (Eigen::Vector3s& point : trace)
    {
      mean += point;
      meanOuterProduct += point * point.transpose();
    }
    mean /= trace.size();
    meanOuterProduct /= trace.size();

    A += meanOuterProduct - (mean * mean.transpose());
  }

  Eigen::JacobiSVD<Eigen::MatrixXs> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3s axis = svd.matrixV().col(2);
  s_t conditionNumber = svd.singularValues()(0) / svd.singularValues()(2);
  return std::make_pair(axis, conditionNumber);
}

std::vector<double> IKInitializer::findCubicRealRoots(
    double a, double b, double c, double d)
{
  double results[3];
  int numRoots = SolveP3(results, b / a, c / a, d / a);
  std::vector<double> roots;
  for (int i = 0; i < numRoots; i++)
  {
    roots.push_back(results[i]);
  }
  return roots;

  /*
  Eigen::Vector4d coeffs;
  coeffs << d, c, b, a; // The coefficients of the cubic polynomial.

  Eigen::PolynomialSolver<double, Eigen::Dynamic> solver(coeffs);

  Eigen::PolynomialSolver<double, Eigen::Dynamic>::RootsType eigenRoots;
  eigenRoots = solver.roots();

  std::vector<double> roots;
  for (int i = 0; i < eigenRoots.size(); i++)
  {
    if (std::abs(eigenRoots(i).imag()) < 1e-10)
    {
      roots.push_back(eigenRoots(i).real());
    }
  }

  return roots;
  */
}

//==============================================================================
/// This method will find the best point on the line given by f(x) = (c + a*x)
/// that minimizes: Sum_i weights[i] * (f(x) -
/// pointsAndRadii[i].first).squaredNorm() - pointsAndRadii[i].second^2)^2
Eigen::Vector3s IKInitializer::centerPointOnAxis(
    Eigen::Vector3s center,
    Eigen::Vector3s axis,
    std::vector<std::pair<Eigen::Vector3s, s_t>> pointsAndRadii,
    std::vector<s_t> weights)
{
  s_t a = 0.0;
  s_t b = 0.0;
  s_t c = 0.0;
  s_t d = 0.0;

  const s_t e = axis.squaredNorm();
  for (int i = 0; i < pointsAndRadii.size(); i++)
  {
    const s_t weight = weights.size() > i ? weights[i] : 1.0;
    const s_t f = (center - pointsAndRadii[i].first).dot(axis);
    const s_t g = (center - pointsAndRadii[i].first).squaredNorm()
                  - pointsAndRadii[i].second * pointsAndRadii[i].second;

    a += weight * 4.0 * e * e;
    b += weight * 12.0 * e * f;
    c += weight * (4.0 * e * g + 8.0 * f * f);
    d += weight * 4.0 * f * g;
  }

  std::vector<s_t> roots = findCubicRealRoots(a, b, c, d);
  if (roots.size() == 0)
  {
    std::cout << "Failed to solve cubic in centerPointOnAxis() for polynomial "
              << a << " * x^3 + " << b << " * x^2 + " << c << " * x + " << d
              << ", returning original center point" << std::endl;
    return center;
  }

  s_t bestRoot = roots[0];
  s_t bestLoss = std::numeric_limits<double>::infinity();
  for (s_t root : roots)
  {
    s_t loss = 0.0;
    const Eigen::Vector3s resultingCenter = center + root * axis;

#ifndef NDEBUG
    s_t polynomialLoss = 0.0;
#endif

    for (int i = 0; i < pointsAndRadii.size(); i++)
    {
      const s_t weight = weights.size() > i ? weights[i] : 1.0;

      // Compute the loss according to the original definition, which is the
      // square of the error between the squared desired radius and the
      // squared distance from the center.
      const s_t linearErrorOnSquaredDistances
          = ((pointsAndRadii[i].first - resultingCenter).squaredNorm()
             - pointsAndRadii[i].second * pointsAndRadii[i].second);
      loss += weight * linearErrorOnSquaredDistances
              * linearErrorOnSquaredDistances;

#ifndef NDEBUG
      // If we're in debug mode, we also compute the loss using the
      // polynomial, just to check that everything is equivalent. This is to
      // verify that we didn't make an algebra mistake in our polynomial
      // expansion, and so we're likely to trust the optimal value.
      const s_t f = (center - pointsAndRadii[i].first).dot(axis);
      const s_t g = (center - pointsAndRadii[i].first).squaredNorm()
                    - pointsAndRadii[i].second * pointsAndRadii[i].second;
      polynomialLoss
          += weight
             * (e * e * root * root * root * root
                + 4 * e * f * root * root * root + 2 * e * g * root * root
                + 4 * f * f * root * root + 4 * f * g * root + g * g);
#endif
    }

#ifndef NDEBUG
    s_t error = std::abs(polynomialLoss - loss);
    if (std::abs(polynomialLoss) > 1.0)
    {
      error /= std::abs(polynomialLoss);
    }
    if (error > 1e-8)
    {
      std::cout << "Polynomial loss (" << polynomialLoss
                << ") differs from standard loss (" << loss << ") by "
                << std::abs(polynomialLoss - loss) << std::endl;
    }
    assert(error < 1e-8);
#endif
    if (loss < bestLoss)
    {
      bestLoss = loss;
      bestRoot = root;
    }
  }

  return center + bestRoot * axis;
}

//==============================================================================
std::vector<std::map<std::string, Eigen::Vector3s>>&
IKInitializer::getJointCenters()
{
  return mJointCenters;
}

//==============================================================================
std::string IKInitializer::debugJointEstimateSource(
    int timestep, std::string jointName)
{
  if (timestep < 0 || timestep > mJointCentersEstimateSource.size())
  {
    return "Invalid timestep";
  }
  if (mJointCentersEstimateSource[timestep].find(jointName)
      == mJointCentersEstimateSource[timestep].end())
  {
    return "No estimate";
  }
  JointCenterEstimateSource source
      = mJointCentersEstimateSource[timestep][jointName];
  if (source == JointCenterEstimateSource::MDS)
  {
    return "MDS";
  }
  else if (source == JointCenterEstimateSource::LEAST_SQUARES_EXACT)
  {
    return "LEAST_SQUARES_EXACT";
  }
  else if (source == JointCenterEstimateSource::LEAST_SQUARES_AXIS)
  {
    return "LEAST_SQUARES_AXIS";
  }

  return "Unrecognized source";
}

//==============================================================================
std::vector<std::map<std::string, Eigen::Vector3s>>&
IKInitializer::getJointAxisDirs()
{
  return mJointAxisDirs;
}

//==============================================================================
std::map<std::string, std::map<std::string, s_t>>&
IKInitializer::getJointToMarkerSquaredDistances()
{
  return mJointToMarkerSquaredDistances;
}

//==============================================================================
std::map<std::string, std::map<std::string, s_t>>&
IKInitializer::getJointToJointSquaredDistances()
{
  return mJointToJointSquaredDistances;
}

//==============================================================================
std::vector<std::shared_ptr<struct StackedBody>>&
IKInitializer::getStackedBodies()
{
  return mStackedBodies;
}

//==============================================================================
std::vector<std::shared_ptr<struct StackedJoint>>&
IKInitializer::getStackedJoints()
{
  return mStackedJoints;
}

//==============================================================================
std::vector<std::map<std::string, Eigen::Isometry3s>>&
IKInitializer::getBodyTransforms()
{
  return mBodyTransforms;
}

//==============================================================================
Eigen::VectorXs IKInitializer::getGroupScales()
{
  return mGroupScales;
}

//==============================================================================
std::vector<Eigen::VectorXs> IKInitializer::getPoses()
{
  return mPoses;
}

//==============================================================================
std::vector<Eigen::VectorXi>
IKInitializer::getPosesClosedFormEstimateAvailable()
{
  return mPosesClosedFormEstimateAvailable;
}

//==============================================================================
std::vector<std::map<std::string, Eigen::Vector3s>>&
IKInitializer::getDebugKnownSyntheticJointCenters()
{
  return mDebugKnownSyntheticJointCenters;
}

} // namespace biomechanics
} // namespace dart