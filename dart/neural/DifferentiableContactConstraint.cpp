#include "dart/neural/DifferentiableContactConstraint.hpp"

#include "dart/collision/Contact.hpp"
#include "dart/constraint/ConstraintBase.hpp"
#include "dart/constraint/ContactConstraint.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

namespace dart {

using namespace constraint;
namespace neural {

//==============================================================================
DifferentiableContactConstraint::DifferentiableContactConstraint(
    std::shared_ptr<constraint::ConstraintBase> constraint,
    int index,
    s_t constraintForce)
  : mConstraint(constraint),
    mIndex(index),
    mConstraintForce(constraintForce),
    mWorldConstraintJacCacheDirty(true)
{
  if (mConstraint->isContactConstraint())
  {
    mContactConstraint
        = std::static_pointer_cast<constraint::ContactConstraint>(mConstraint);
    // This needs to be explicitly copied, otherwise the memory is overwritten
    mContact = std::make_shared<collision::Contact>(
        mContactConstraint->getContact());
  }
  for (auto skel : constraint->getSkeletons())
  {
    mSkeletons.push_back(skel->getName());
    mSkeletonOriginalPositions.push_back(skel->getPositions());
  }
}

//==============================================================================
Eigen::Vector3s DifferentiableContactConstraint::getContactWorldPosition()
{
  if (!mConstraint->isContactConstraint())
  {
    return Eigen::Vector3s::Zero();
  }
  return mContact->point;
}

//==============================================================================
Eigen::Vector3s DifferentiableContactConstraint::getContactWorldNormal()
{
  if (!mConstraint->isContactConstraint())
  {
    return Eigen::Vector3s::Zero();
  }
  return mContact->normal;
}

//==============================================================================
Eigen::Vector3s DifferentiableContactConstraint::getContactWorldForceDirection()
{
  if (!mConstraint->isContactConstraint())
  {
    return Eigen::Vector3s::Zero();
  }
  if (mIndex == 0)
  {
    return mContact->normal;
  }
  else
  {
    return mContactConstraint->getTangentBasisMatrixODE(mContact->normal)
        .col(mIndex - 1);
  }
}

//==============================================================================
Eigen::Vector6s DifferentiableContactConstraint::getWorldForce()
{
  Eigen::Vector6s worldForce = Eigen::Vector6s();
  worldForce.head<3>()
      = getContactWorldPosition().cross(getContactWorldForceDirection());
  worldForce.tail<3>() = getContactWorldForceDirection();
  return worldForce;
}

//==============================================================================
collision::ContactType DifferentiableContactConstraint::getContactType()
{
  if (!mConstraint->isContactConstraint())
  {
    // UNSUPPORTED is the default, and means we won't attempt to get gradients
    // for how the contact point moves as we move the skeletons.
    return collision::ContactType::UNSUPPORTED;
  }
  return mContact->type;
}

//==============================================================================
collision::Contact& DifferentiableContactConstraint::getContact()
{
  return *(mContact.get());
}

//==============================================================================
/// This figures out what type of contact this skeleton is involved in.
DofContactType DifferentiableContactConstraint::getDofContactType(
    dynamics::DegreeOfFreedom* dof)
{
  bool isParentA = dof->isParentOfFast(mContactConstraint->getBodyNodeA());
  bool isParentB = dof->isParentOfFast(mContactConstraint->getBodyNodeB());
  // If we're a parent of both contact points, it's a self-contact down the tree
  if (isParentA && isParentB)
  {
    return DofContactType::SELF_COLLISION;
  }
  // If we're not a parent of either point, it's not an issue
  else if (!isParentA && !isParentB)
  {
    return DofContactType::NONE;
  }
  // If we're just a parent of A
  else if (isParentA)
  {
    switch (getContactType())
    {
      case collision::ContactType::FACE_VERTEX:
        return DofContactType::FACE;
      case collision::ContactType::VERTEX_FACE:
        return DofContactType::VERTEX;
      case collision::ContactType::EDGE_EDGE:
        return DofContactType::EDGE_A;
      case collision::ContactType::SPHERE_BOX:
        return DofContactType::SPHERE_TO_BOX;
      case collision::ContactType::BOX_SPHERE:
        return DofContactType::BOX_TO_SPHERE;
      case collision::ContactType::SPHERE_SPHERE:
        return DofContactType::SPHERE_A;
      case collision::ContactType::SPHERE_FACE:
        return DofContactType::SPHERE_TO_FACE;
      case collision::ContactType::FACE_SPHERE:
        return DofContactType::FACE_TO_SPHERE;
      case collision::ContactType::SPHERE_EDGE:
        return DofContactType::SPHERE_TO_EDGE;
      case collision::ContactType::EDGE_SPHERE:
        return DofContactType::EDGE_TO_SPHERE;
      case collision::ContactType::SPHERE_VERTEX:
        return DofContactType::SPHERE_TO_VERTEX;
      case collision::ContactType::VERTEX_SPHERE:
        return DofContactType::VERTEX_TO_SPHERE;
      case collision::ContactType::SPHERE_PIPE:
        return DofContactType::SPHERE_TO_PIPE;
      case collision::ContactType::PIPE_SPHERE:
        return DofContactType::PIPE_TO_SPHERE;
      case collision::ContactType::PIPE_PIPE:
        return DofContactType::PIPE_A;
      case collision::ContactType::PIPE_VERTEX:
        return DofContactType::PIPE_TO_VERTEX;
      case collision::ContactType::VERTEX_PIPE:
        return DofContactType::VERTEX_TO_PIPE;
      case collision::ContactType::PIPE_EDGE:
        return DofContactType::PIPE_TO_EDGE;
      case collision::ContactType::EDGE_PIPE:
        return DofContactType::EDGE_TO_PIPE;
      default:
        return DofContactType::UNSUPPORTED;
    }
  }
  // If we're just a parent of B
  else if (isParentB)
  {
    switch (getContactType())
    {
      case collision::ContactType::FACE_VERTEX:
        return DofContactType::VERTEX;
      case collision::ContactType::VERTEX_FACE:
        return DofContactType::FACE;
      case collision::ContactType::EDGE_EDGE:
        return DofContactType::EDGE_B;
      case collision::ContactType::SPHERE_BOX:
        return DofContactType::BOX_TO_SPHERE;
      case collision::ContactType::BOX_SPHERE:
        return DofContactType::SPHERE_TO_BOX;
      case collision::ContactType::SPHERE_SPHERE:
        return DofContactType::SPHERE_B;
      case collision::ContactType::SPHERE_FACE:
        return DofContactType::FACE_TO_SPHERE;
      case collision::ContactType::FACE_SPHERE:
        return DofContactType::SPHERE_TO_FACE;
      case collision::ContactType::SPHERE_EDGE:
        return DofContactType::EDGE_TO_SPHERE;
      case collision::ContactType::EDGE_SPHERE:
        return DofContactType::SPHERE_TO_EDGE;
      case collision::ContactType::SPHERE_VERTEX:
        return DofContactType::VERTEX_TO_SPHERE;
      case collision::ContactType::VERTEX_SPHERE:
        return DofContactType::SPHERE_TO_VERTEX;
      case collision::ContactType::SPHERE_PIPE:
        return DofContactType::PIPE_TO_SPHERE;
      case collision::ContactType::PIPE_SPHERE:
        return DofContactType::SPHERE_TO_PIPE;
      case collision::ContactType::PIPE_PIPE:
        return DofContactType::PIPE_B;
      case collision::ContactType::PIPE_VERTEX:
        return DofContactType::VERTEX_TO_PIPE;
      case collision::ContactType::VERTEX_PIPE:
        return DofContactType::PIPE_TO_VERTEX;
      case collision::ContactType::PIPE_EDGE:
        return DofContactType::EDGE_TO_PIPE;
      case collision::ContactType::EDGE_PIPE:
        return DofContactType::PIPE_TO_EDGE;
      default:
        return DofContactType::UNSUPPORTED;
    }
  }

  // Control should never reach this point
  return DofContactType::NONE;
}

//==============================================================================
Eigen::VectorXs DifferentiableContactConstraint::getConstraintForces(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  // If this constraint doesn't touch this skeleton, then return all 0s
  auto skelNameCursor
      = std::find(mSkeletons.begin(), mSkeletons.end(), skel->getName());
  if (skelNameCursor == mSkeletons.end())
  {
    return Eigen::VectorXs::Zero(skel->getNumDofs());
  }
  // Check that the skeletons are where we left them, otherwise these
  // computations will be wrong
  int index = std::distance(mSkeletons.begin(), skelNameCursor);
  Eigen::VectorXs oldPositions = skel->getPositions();
  // assert((oldPositions - mSkeletonOriginalPositions[index]).squaredNorm() ==
  // 0.0);
  skel->setPositions(mSkeletonOriginalPositions[index]);

  Eigen::Vector6s worldForce = getWorldForce();

  Eigen::VectorXs taus = Eigen::VectorXs::Zero(skel->getNumDofs());
  for (int i = 0; i < skel->getNumDofs(); i++)
  {
    auto dof = skel->getDof(i);
    s_t multiple = getControlForceMultiple(dof);
    if (multiple == 0)
    {
      taus(i) = 0.0;
    }
    else
    {
      Eigen::Vector6s worldTwist = getWorldScrewAxisForForce(dof);
      taus(i) = worldTwist.dot(worldForce) * multiple;
    }
  }

  skel->setPositions(oldPositions);

  return taus;
}

//==============================================================================
Eigen::VectorXs DifferentiableContactConstraint::getConstraintForces(
    simulation::World* world, std::vector<std::string> skelNames)
{
  int totalDofs = 0;
  for (auto name : skelNames)
    totalDofs += world->getSkeleton(name)->getNumDofs();
  Eigen::VectorXs taus = Eigen::VectorXs::Zero(totalDofs);
  int cursor = 0;
  for (auto name : skelNames)
  {
    auto skel = world->getSkeleton(name);
    int dofs = skel->getNumDofs();
    taus.segment(cursor, dofs) = getConstraintForces(skel);
    cursor += dofs;
  }
  return taus;
}

//==============================================================================
Eigen::VectorXs DifferentiableContactConstraint::getConstraintForces(
    simulation::World* world)
{
  Eigen::VectorXs taus = Eigen::VectorXs::Zero(world->getNumDofs());
  int cursor = 0;
  for (int i = 0; i < world->getNumSkeletons(); i++)
  {
    auto skel = world->getSkeleton(i);
    int dofs = skel->getNumDofs();
    taus.segment(cursor, dofs) = getConstraintForces(skel);
    cursor += dofs;
  }
  return taus;
}

//==============================================================================
/// Returns the gradient of the contact position with respect to the
/// specified dof of this skeleton
Eigen::Vector3s DifferentiableContactConstraint::getContactPositionGradient(
    dynamics::DegreeOfFreedom* dof)
{
  Eigen::Vector3s contactPos = getContactWorldPosition();
  DofContactType type = getDofContactType(dof);

  if (type == FACE || type == SPHERE_TO_VERTEX || type == PIPE_TO_VERTEX)
  {
    return Eigen::Vector3s::Zero();
  }

  int jointIndex = dof->getIndexInJoint();
  Eigen::Vector6s worldTwist
      = dof->getJoint()->getWorldAxisScrewForPosition(jointIndex);

  if (type == SPHERE_A)
  {
    s_t weight = mContact->radiusB / (mContact->radiusA + mContact->radiusB);
    return weight * math::gradientWrtTheta(worldTwist, mContact->centerA, 0.0);
  }
  else if (type == SPHERE_B)
  {
    s_t weight = mContact->radiusA / (mContact->radiusA + mContact->radiusB);
    return weight * math::gradientWrtTheta(worldTwist, mContact->centerB, 0.0);
  }
  else if (type == SPHERE_TO_BOX)
  {
    Eigen::Vector3s sphereGrad
        = math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);
    if (mContact->face1Locked)
    {
      sphereGrad
          = sphereGrad
            - mContact->face1Normal * mContact->face1Normal.dot(sphereGrad);
    }
    if (mContact->face2Locked)
    {
      sphereGrad
          = sphereGrad
            - mContact->face2Normal * mContact->face2Normal.dot(sphereGrad);
    }
    if (mContact->face3Locked)
    {
      sphereGrad
          = sphereGrad
            - mContact->face3Normal * mContact->face3Normal.dot(sphereGrad);
    }
    return sphereGrad;
  }
  else if (type == BOX_TO_SPHERE)
  {
    Eigen::Vector3s negSphereGrad
        = -math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);
    if (mContact->face1Locked)
    {
      negSphereGrad
          = negSphereGrad
            - mContact->face1Normal * mContact->face1Normal.dot(negSphereGrad);
    }
    if (mContact->face2Locked)
    {
      negSphereGrad
          = negSphereGrad
            - mContact->face2Normal * mContact->face2Normal.dot(negSphereGrad);
    }
    if (mContact->face3Locked)
    {
      negSphereGrad
          = negSphereGrad
            - mContact->face3Normal * mContact->face3Normal.dot(negSphereGrad);
    }
    // Step 2. Now reverse the result by moving the point by the box
    // transform, which undoes the sphere "movement" but leaves helpful
    // clipping effects in place.
    Eigen::Vector3s contactPosGrad
        = math::gradientWrtTheta(worldTwist, contactPos, 0.0);
    return contactPosGrad + negSphereGrad;
  }
  else if (
      type == VERTEX || type == SELF_COLLISION || type == VERTEX_TO_SPHERE
      || type == VERTEX_TO_PIPE)
  {
    return math::gradientWrtTheta(worldTwist, contactPos, 0.0);
  }
  else if (type == EDGE_A)
  {
    Eigen::Vector3s edgeAPosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0);
    Eigen::Vector3s edgeADirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);

    return math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        edgeAPosGradient,
        mContact->edgeADir,
        edgeADirGradient,
        mContact->edgeBFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeBDir,
        Eigen::Vector3s::Zero());
  }
  else if (type == EDGE_B)
  {
    Eigen::Vector3s edgeBPosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeBFixedPoint, 0.0);
    Eigen::Vector3s edgeBDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeBDir, 0.0);

    return math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->edgeBFixedPoint,
        edgeBPosGradient,
        mContact->edgeBDir,
        edgeBDirGradient);
  }
  else if (type == SPHERE_TO_FACE)
  {
    return math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);
  }
  else if (type == FACE_TO_SPHERE)
  {
    if (getContactType() == collision::ContactType::SPHERE_FACE)
    {
      return -math::gradientWrtThetaPureRotation(
          worldTwist.head<3>(), mContact->normal * mContact->sphereRadius, 0.0);
    }
    else
    {
      return math::gradientWrtThetaPureRotation(
          worldTwist.head<3>(), mContact->normal * mContact->sphereRadius, 0.0);
    }
  }
  else if (type == SPHERE_TO_EDGE)
  {
    return math::closestPointOnLineGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->sphereCenter,
        math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0));
  }
  else if (type == EDGE_TO_SPHERE)
  {
    return math::closestPointOnLineGradient(
        mContact->edgeAFixedPoint,
        math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0),
        mContact->edgeADir,
        math::gradientWrtThetaPureRotation(
            worldTwist.head<3>(), mContact->edgeADir, 0.0),
        mContact->sphereCenter,
        Eigen::Vector3s::Zero());
  }
  else if (type == SPHERE_TO_PIPE)
  {
    s_t weight = mContact->pipeRadius
                    / (mContact->sphereRadius + mContact->pipeRadius);

    Eigen::Vector3s rawGrad
        = math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);
    Eigen::Vector3s parallelComponent
        = mContact->pipeDir.dot(rawGrad) * mContact->pipeDir;
    Eigen::Vector3s perpendicularComponent = rawGrad - parallelComponent;
    return parallelComponent + (weight * perpendicularComponent);
  }
  else if (type == PIPE_TO_SPHERE)
  {
    Eigen::Vector3s rawGrad = math::closestPointOnLineGradient(
        mContact->pipeFixedPoint,
        math::gradientWrtTheta(worldTwist, mContact->pipeFixedPoint, 0.0),
        mContact->pipeDir,
        math::gradientWrtThetaPureRotation(
            worldTwist.head<3>(), mContact->pipeDir, 0.0),
        mContact->sphereCenter,
        Eigen::Vector3s::Zero());
    s_t weight = mContact->sphereRadius
                    / (mContact->sphereRadius + mContact->pipeRadius);
    return weight * rawGrad;
  }
  else if (type == PIPE_A)
  {
    Eigen::Vector3s edgeAPosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0);
    Eigen::Vector3s edgeADirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);

    return math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        edgeAPosGradient,
        mContact->edgeADir,
        edgeADirGradient,
        mContact->edgeBFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeBDir,
        Eigen::Vector3s::Zero(),
        mContact->radiusA,
        mContact->radiusB);
  }
  else if (type == PIPE_B)
  {
    Eigen::Vector3s edgeBPosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeBFixedPoint, 0.0);
    Eigen::Vector3s edgeBDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeBDir, 0.0);

    return math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->edgeBFixedPoint,
        edgeBPosGradient,
        mContact->edgeBDir,
        edgeBDirGradient,
        mContact->radiusA,
        mContact->radiusB);
  }
  else if (type == DofContactType::PIPE_TO_EDGE)
  {
    Eigen::Vector3s pipePosGradient
        = math::gradientWrtTheta(worldTwist, mContact->pipeFixedPoint, 0.0);
    Eigen::Vector3s pipeDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->pipeDir, 0.0);

    return math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->pipeFixedPoint,
        pipePosGradient,
        mContact->pipeDir,
        pipeDirGradient,
        0.0,
        1.0);
  }
  else if (type == DofContactType::EDGE_TO_PIPE)
  {
    Eigen::Vector3s edgePosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0);
    Eigen::Vector3s edgeDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);

    return math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        edgePosGradient,
        mContact->edgeADir,
        edgeDirGradient,
        mContact->pipeFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->pipeDir,
        Eigen::Vector3s::Zero(),
        0.0,
        1.0);
  }

  // Default case
  return Eigen::Vector3s::Zero();
}

//==============================================================================
/// Returns the gradient of the contact normal with respect to the
/// specified dof of this skeleton
Eigen::Vector3s DifferentiableContactConstraint::getContactNormalGradient(
    dynamics::DegreeOfFreedom* dof)
{
  Eigen::Vector3s normal = getContactWorldNormal();
  DofContactType type = getDofContactType(dof);
  if (type == VERTEX || type == SPHERE_TO_FACE)
  {
    return Eigen::Vector3s::Zero();
  }
  int jointIndex = dof->getIndexInJoint();
  Eigen::Vector6s worldTwist
      = dof->getJoint()->getWorldAxisScrewForPosition(jointIndex);

  // TODO(keenon): figure out a way to patch the Unit Z singularity in the
  // friction cone computations
  if (mContact->normal.cross(Eigen::Vector3s::UnitZ()).squaredNorm() < 1e-7
      && mIndex > 0)
  {
    dterr << "Attempting to get a contact normal gradient through a "
             "frictional contact where the underlying contact normal is very "
             "near Unit Z! This can lead to gradient issues because of how "
             "the friction cone basis are computed. The Unit Z normal is a "
             "non-differentiable singularity for the friction basis, and the "
             "friction directions can change very rapidly near the Unit Z "
             "contact normal. If you can reformulate "
             "your problem to change the angle of this contact, please do "
             "so. Otherwise, you'll have to wait for a patch to address this "
             "issue."
          << std::endl;
  }

  if (type == SPHERE_A)
  {
    s_t norm = (mContact->centerA - mContact->centerB).norm();
    Eigen::Vector3s posGrad
        = math::gradientWrtTheta(worldTwist, mContact->centerA, 0.0);
    posGrad /= norm;
    posGrad -= mContact->normal.dot(posGrad) * mContact->normal;
    return posGrad;
  }
  else if (type == SPHERE_B)
  {
    s_t norm = (mContact->centerA - mContact->centerB).norm();
    Eigen::Vector3s posGrad
        = math::gradientWrtTheta(worldTwist, mContact->centerB, 0.0);
    posGrad /= norm;
    posGrad -= mContact->normal.dot(posGrad) * mContact->normal;
    return -posGrad;
  }
  else if (type == SPHERE_TO_BOX)
  {
    s_t norm = (mContact->sphereCenter - mContact->point).norm();
    Eigen::Vector3s contactPosGrad = getContactPositionGradient(dof);
    Eigen::Vector3s spherePosGrad
        = math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);
    if (norm > 1e-5)
    {
      contactPosGrad /= norm;
      spherePosGrad /= norm;
    }
    Eigen::Vector3s totalGrad = Eigen::Vector3s::Zero();
    // The normal flips direction depending on which order the contact detection
    // was called in :(
    if (mContact->type == collision::BOX_SPHERE)
    {
      // normal = contact_pt - sphereCenter
      totalGrad = contactPosGrad - spherePosGrad;
    }
    else if (mContact->type == collision::SPHERE_BOX)
    {
      // normal = sphereCenter - contact_pt
      totalGrad = spherePosGrad - contactPosGrad;
    }
    else
    {
      assert(false && "Illegal contact type detected");
    }
    totalGrad -= totalGrad.dot(normal) * normal;
    return totalGrad;
  }
  else if (type == BOX_TO_SPHERE)
  {
    s_t norm = (mContact->sphereCenter - mContact->point).norm();
    Eigen::Vector3s contactPosGrad = getContactPositionGradient(dof);
    // spherePosGrad = 0 here, because we don't move the sphere when we move the
    // box
    if (norm > 1e-5)
    {
      contactPosGrad /= norm;
    }
    Eigen::Vector3s totalGrad = Eigen::Vector3s::Zero();
    // The normal flips direction depending on which order the contact detection
    // was called in :(
    if (mContact->type == collision::BOX_SPHERE)
    {
      // normal = contact_pt - sphereCenter
      totalGrad = contactPosGrad;
    }
    else if (mContact->type == collision::SPHERE_BOX)
    {
      // normal = sphereCenter - contact_pt
      totalGrad = -contactPosGrad;
    }
    else
    {
      assert(false && "Illegal contact type detected");
    }
    totalGrad -= totalGrad.dot(normal) * normal;
    return totalGrad;
  }
  else if (type == FACE || type == SELF_COLLISION || type == FACE_TO_SPHERE)
  {
    return math::gradientWrtThetaPureRotation(worldTwist.head<3>(), normal, 0);
  }
  else if (type == EDGE_A)
  {
    Eigen::Vector3s edgeADirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);

    s_t sign = 1.0;
    Eigen::Vector3s normal = mContact->edgeBDir.cross(mContact->edgeADir);
    if (normal.dot(mContact->normal) < 0)
    {
      sign = -1.0;
    }

    return sign * mContact->edgeBDir.cross(edgeADirGradient);
  }
  else if (type == EDGE_B)
  {
    Eigen::Vector3s edgeBDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeBDir, 0.0);

    s_t sign = 1.0;
    Eigen::Vector3s normal = mContact->edgeBDir.cross(mContact->edgeADir);
    if (normal.dot(mContact->normal) < 0)
    {
      sign = -1.0;
    }

    return sign * edgeBDirGradient.cross(mContact->edgeADir);
  }
  if (type == SPHERE_TO_VERTEX)
  {
    s_t norm = (mContact->sphereCenter - mContact->vertexPoint).norm();
    Eigen::Vector3s posGrad
        = math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);
    posGrad /= norm;
    posGrad -= mContact->normal.dot(posGrad) * mContact->normal;
    if (getContactType() == collision::ContactType::SPHERE_VERTEX)
    {
      return posGrad;
    }
    else
    {
      return -posGrad;
    }
  }
  else if (type == VERTEX_TO_SPHERE)
  {
    s_t norm = (mContact->vertexPoint - mContact->sphereCenter).norm();
    Eigen::Vector3s posGrad
        = math::gradientWrtTheta(worldTwist, mContact->vertexPoint, 0.0);
    posGrad /= norm;
    posGrad -= mContact->normal.dot(posGrad) * mContact->normal;
    if (getContactType() == collision::ContactType::VERTEX_SPHERE)
    {
      return posGrad;
    }
    else
    {
      return -posGrad;
    }
  }
  else if (type == SPHERE_TO_EDGE)
  {
    Eigen::Vector3s closestPointGrad = math::closestPointOnLineGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->sphereCenter,
        math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0));
    Eigen::Vector3s sphereCenterGrad
        = math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);

    s_t norm = (mContact->edgeAClosestPoint - mContact->sphereCenter).norm();
    Eigen::Vector3s normGrad = closestPointGrad - sphereCenterGrad;
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::SPHERE_EDGE)
    {
      return -normGrad;
    }
    else
    {
      return normGrad;
    }
  }
  else if (type == EDGE_TO_SPHERE)
  {
    Eigen::Vector3s closestPointGrad = math::closestPointOnLineGradient(
        mContact->edgeAFixedPoint,
        math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0),
        mContact->edgeADir,
        math::gradientWrtThetaPureRotation(
            worldTwist.head<3>(), mContact->edgeADir, 0.0),
        mContact->sphereCenter,
        Eigen::Vector3s::Zero());

    s_t norm = (mContact->edgeAClosestPoint - mContact->sphereCenter).norm();
    Eigen::Vector3s normGrad = closestPointGrad;
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::EDGE_SPHERE)
    {
      return normGrad;
    }
    else
    {
      return -normGrad;
    }
  }
  else if (type == SPHERE_TO_PIPE)
  {
    s_t norm = (mContact->pipeClosestPoint - mContact->sphereCenter).norm();
    Eigen::Vector3s normGrad
        = math::gradientWrtTheta(worldTwist, mContact->sphereCenter, 0.0);
    normGrad -= normGrad.dot(mContact->pipeDir) * mContact->pipeDir;
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::SPHERE_PIPE)
    {
      return normGrad;
    }
    else
    {
      return -normGrad;
    }
  }
  else if (type == PIPE_TO_SPHERE)
  {
    Eigen::Vector3s closestPointGrad = math::closestPointOnLineGradient(
        mContact->pipeFixedPoint,
        math::gradientWrtTheta(worldTwist, mContact->pipeFixedPoint, 0.0),
        mContact->pipeDir,
        math::gradientWrtThetaPureRotation(
            worldTwist.head<3>(), mContact->pipeDir, 0.0),
        mContact->sphereCenter,
        Eigen::Vector3s::Zero());

    s_t norm = (mContact->pipeClosestPoint - mContact->sphereCenter).norm();
    Eigen::Vector3s normGrad = closestPointGrad;
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::PIPE_SPHERE)
    {
      return normGrad;
    }
    else
    {
      return -normGrad;
    }
  }
  else if (type == PIPE_A)
  {
    Eigen::Vector3s edgeAPosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0);
    Eigen::Vector3s edgeADirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);

    Eigen::Vector3s closestPointAGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        edgeAPosGradient,
        mContact->edgeADir,
        edgeADirGradient,
        mContact->edgeBFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeBDir,
        Eigen::Vector3s::Zero(),
        0.0,
        1.0);

    Eigen::Vector3s closestPointBGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        edgeAPosGradient,
        mContact->edgeADir,
        edgeADirGradient,
        mContact->edgeBFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeBDir,
        Eigen::Vector3s::Zero(),
        1.0,
        0.0);

    Eigen::Vector3s normGrad = closestPointAGrad - closestPointBGrad;
    s_t norm
        = (mContact->edgeAClosestPoint - mContact->edgeBClosestPoint).norm();
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    return normGrad;
  }
  else if (type == PIPE_B)
  {
    Eigen::Vector3s edgeBPosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeBFixedPoint, 0.0);
    Eigen::Vector3s edgeBDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeBDir, 0.0);

    Eigen::Vector3s closestPointAGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->edgeBFixedPoint,
        edgeBPosGradient,
        mContact->edgeBDir,
        edgeBDirGradient,
        0.0,
        1.0);

    Eigen::Vector3s closestPointBGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->edgeBFixedPoint,
        edgeBPosGradient,
        mContact->edgeBDir,
        edgeBDirGradient,
        1.0,
        0.0);

    Eigen::Vector3s normGrad = closestPointAGrad - closestPointBGrad;
    s_t norm
        = (mContact->edgeAClosestPoint - mContact->edgeBClosestPoint).norm();
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    return normGrad;
  }
  else if (type == VERTEX_TO_PIPE)
  {
    Eigen::Vector3s pointGrad
        = math::gradientWrtTheta(worldTwist, mContact->point, 0.0);

    Eigen::Vector3s closestPointGrad = math::closestPointOnLineGradient(
        mContact->pipeFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->pipeDir,
        Eigen::Vector3s::Zero(),
        mContact->point,
        math::gradientWrtTheta(worldTwist, mContact->point, 0.0));

    s_t norm = (mContact->pipeClosestPoint - mContact->point).norm();
    Eigen::Vector3s normGrad = closestPointGrad - pointGrad;
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::PIPE_VERTEX)
    {
      return normGrad;
    }
    else
    {
      return -normGrad;
    }
  }
  else if (type == PIPE_TO_VERTEX)
  {
    Eigen::Vector3s closestPointGrad = math::closestPointOnLineGradient(
        mContact->pipeFixedPoint,
        math::gradientWrtTheta(worldTwist, mContact->pipeFixedPoint, 0.0),
        mContact->pipeDir,
        math::gradientWrtThetaPureRotation(
            worldTwist.head<3>(), mContact->pipeDir, 0.0),
        mContact->point,
        Eigen::Vector3s::Zero());

    s_t norm = (mContact->pipeClosestPoint - mContact->point).norm();
    Eigen::Vector3s normGrad = closestPointGrad;
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::PIPE_VERTEX)
    {
      return normGrad;
    }
    else
    {
      return -normGrad;
    }
  }
  else if (type == DofContactType::PIPE_TO_EDGE)
  {
    Eigen::Vector3s pipePosGradient
        = math::gradientWrtTheta(worldTwist, mContact->pipeFixedPoint, 0.0);
    Eigen::Vector3s pipeDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->pipeDir, 0.0);

    Eigen::Vector3s pipeClosestPointGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->pipeFixedPoint,
        pipePosGradient,
        mContact->pipeDir,
        pipeDirGradient,
        0.0,
        1.0);

    Eigen::Vector3s edgeClosestPointGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->edgeADir,
        Eigen::Vector3s::Zero(),
        mContact->pipeFixedPoint,
        pipePosGradient,
        mContact->pipeDir,
        pipeDirGradient,
        1.0,
        0.0);

    Eigen::Vector3s normGrad = edgeClosestPointGrad - pipeClosestPointGrad;
    s_t norm
        = (mContact->edgeAClosestPoint - mContact->pipeClosestPoint).norm();
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::PIPE_EDGE)
    {
      return normGrad;
    }
    else
    {
      return -normGrad;
    }
  }
  else if (type == DofContactType::EDGE_TO_PIPE)
  {
    Eigen::Vector3s edgePosGradient
        = math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0);
    Eigen::Vector3s edgeDirGradient = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);

    Eigen::Vector3s pipeClosestPointGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        edgePosGradient,
        mContact->edgeADir,
        edgeDirGradient,
        mContact->pipeFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->pipeDir,
        Eigen::Vector3s::Zero(),
        0.0,
        1.0);

    Eigen::Vector3s edgeClosestPointGrad = math::getContactPointGradient(
        mContact->edgeAFixedPoint,
        edgePosGradient,
        mContact->edgeADir,
        edgeDirGradient,
        mContact->pipeFixedPoint,
        Eigen::Vector3s::Zero(),
        mContact->pipeDir,
        Eigen::Vector3s::Zero(),
        1.0,
        0.0);

    Eigen::Vector3s normGrad = edgeClosestPointGrad - pipeClosestPointGrad;
    s_t norm
        = (mContact->edgeAClosestPoint - mContact->pipeClosestPoint).norm();
    normGrad /= norm;
    normGrad -= mContact->normal.dot(normGrad) * mContact->normal;

    if (getContactType() == collision::ContactType::PIPE_EDGE)
    {
      return normGrad;
    }
    else
    {
      return -normGrad;
    }
  }

  // Default case
  return Eigen::Vector3s::Zero();
}

//==============================================================================
/// Returns the gradient of the contact force with respect to the
/// specified dof of this skeleton
Eigen::Vector3s DifferentiableContactConstraint::getContactForceGradient(
    dynamics::DegreeOfFreedom* dof)
{
  DofContactType type = getDofContactType(dof);
  if (type == VERTEX || type == NONE)
  {
    return Eigen::Vector3s::Zero();
  }

  Eigen::Vector3s contactNormal = getContactWorldNormal();
  Eigen::Vector3s normalGradient = getContactNormalGradient(dof);
  if (mIndex == 0 || normalGradient.squaredNorm() <= 1e-12)
    return normalGradient;
  else
  {
    return mContactConstraint
        ->getTangentBasisMatrixODEGradient(contactNormal, normalGradient)
        .col(mIndex - 1);
  }
}

//==============================================================================
/// Returns the gradient of the full 6d twist force
Eigen::Vector6s DifferentiableContactConstraint::getContactWorldForceGradient(
    dynamics::DegreeOfFreedom* dof)
{
  Eigen::Vector3s position = getContactWorldPosition();
  Eigen::Vector3s force = getContactWorldForceDirection();
  Eigen::Vector3s forceGradient = getContactForceGradient(dof);
  Eigen::Vector3s positionGradient = getContactPositionGradient(dof);

  Eigen::Vector6s result = Eigen::Vector6s::Zero();
  result.head<3>()
      = position.cross(forceGradient) + positionGradient.cross(force);
  result.tail<3>() = forceGradient;
  return result;
}

//==============================================================================
EdgeData DifferentiableContactConstraint::getEdgeGradient(
    dynamics::DegreeOfFreedom* dof)
{
  EdgeData data;
  data.edgeAPos = Eigen::Vector3s::Zero();
  data.edgeADir = Eigen::Vector3s::Zero();
  data.edgeBPos = Eigen::Vector3s::Zero();
  data.edgeBDir = Eigen::Vector3s::Zero();

  int jointIndex = dof->getIndexInJoint();
  Eigen::Vector6s worldTwist
      = dof->getJoint()->getWorldAxisScrewForPosition(jointIndex);

  DofContactType type = getDofContactType(dof);
  if (type == EDGE_A)
  {
    data.edgeAPos
        = math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0);
    data.edgeADir = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);
  }
  else if (type == EDGE_B)
  {
    data.edgeBPos
        = math::gradientWrtTheta(worldTwist, mContact->edgeBFixedPoint, 0.0);
    data.edgeBDir = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeBDir, 0.0);
  }
  else if (type == SELF_COLLISION)
  {
    data.edgeAPos
        = math::gradientWrtTheta(worldTwist, mContact->edgeAFixedPoint, 0.0);
    data.edgeADir = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeADir, 0.0);
    data.edgeBPos
        = math::gradientWrtTheta(worldTwist, mContact->edgeBFixedPoint, 0.0);
    data.edgeBDir = math::gradientWrtThetaPureRotation(
        worldTwist.head<3>(), mContact->edgeBDir, 0.0);
  }

  return data;
}

//==============================================================================
/// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6s
DifferentiableContactConstraint::getScrewAxisForPositionGradient(
    dynamics::DegreeOfFreedom* screwDof, dynamics::DegreeOfFreedom* rotateDof)
{
  // Special case: all angular DOFs within FreeJoints effect each other in
  // special ways
  if (screwDof->getJoint() == rotateDof->getJoint()
      && screwDof->getJoint()->getType()
             == dynamics::FreeJoint::getStaticType())
  {
    dynamics::FreeJoint* freeJoint
        = static_cast<dynamics::FreeJoint*>(screwDof->getJoint());
    int axisIndex = screwDof->getIndexInJoint();
    int rotateIndex = rotateDof->getIndexInJoint();
    if (axisIndex < 3)
    {
      return freeJoint->getScrewAxisGradientForPosition(axisIndex, rotateIndex);
    }
    else
    {
      // The translation axes aren't effected by anything
      return Eigen::Vector6s::Zero();
    }
  }
  // Special case: all DOFs within BallJoints effect each other in special ways
  else if (
      screwDof->getJoint() == rotateDof->getJoint()
      && screwDof->getJoint()->getType()
             == dynamics::BallJoint::getStaticType())
  {
    dynamics::BallJoint* ballJoint
        = static_cast<dynamics::BallJoint*>(screwDof->getJoint());
    int axisIndex = screwDof->getIndexInJoint();
    int rotateIndex = rotateDof->getIndexInJoint();
    if (axisIndex < 3 && rotateIndex < 3)
    {
      return ballJoint->getScrewAxisGradientForPosition(axisIndex, rotateIndex);
    }
  }
  // General case:
  if (!rotateDof->isParentOfFast(screwDof))
    return Eigen::Vector6s::Zero();

  Eigen::Vector6s axisWorldTwist = getWorldScrewAxisForPosition(screwDof);
  Eigen::Vector6s rotateWorldTwist = getWorldScrewAxisForPosition(rotateDof);
  return math::ad(rotateWorldTwist, axisWorldTwist);
}

//==============================================================================
/// Returns the gradient of the screw axis with respect to the rotate dof
Eigen::Vector6s DifferentiableContactConstraint::getScrewAxisForForceGradient(
    dynamics::DegreeOfFreedom* screwDof, dynamics::DegreeOfFreedom* rotateDof)
{
  // Special case: all angular DOFs within FreeJoints effect each other in
  // special ways
  if (screwDof->getJoint() == rotateDof->getJoint()
      && screwDof->getJoint()->getType()
             == dynamics::FreeJoint::getStaticType())
  {
    dynamics::FreeJoint* freeJoint
        = static_cast<dynamics::FreeJoint*>(screwDof->getJoint());
    int axisIndex = screwDof->getIndexInJoint();
    int rotateIndex = rotateDof->getIndexInJoint();

    return freeJoint->getScrewAxisGradientForForce(axisIndex, rotateIndex);
  }
  // Special case: all DOFs within BallJoints effect each other in special ways
  else if (
      screwDof->getJoint() == rotateDof->getJoint()
      && screwDof->getJoint()->getType()
             == dynamics::BallJoint::getStaticType())
  {
    dynamics::BallJoint* ballJoint
        = static_cast<dynamics::BallJoint*>(screwDof->getJoint());
    int axisIndex = screwDof->getIndexInJoint();
    int rotateIndex = rotateDof->getIndexInJoint();
    if (axisIndex < 3 && rotateIndex < 3)
    {
      return ballJoint->getScrewAxisGradientForForce(axisIndex, rotateIndex);
    }
  }
  // General case:
  if (!rotateDof->isParentOfFast(screwDof))
    return Eigen::Vector6s::Zero();

  Eigen::Vector6s axisWorldTwist = getWorldScrewAxisForForce(screwDof);
  Eigen::Vector6s rotateWorldTwist = getWorldScrewAxisForPosition(rotateDof);
  return math::ad(rotateWorldTwist, axisWorldTwist);
}

//==============================================================================
/// Returns the gradient of the screw axis with respect to the rotate dof
///
/// Unlike its sibling, getScrewAxisForForceGradient(), this allows passing
/// in values that are otherwise repeatedly computed.
Eigen::Vector6s
DifferentiableContactConstraint::getScrewAxisForForceGradient_Optimized(
    dynamics::DegreeOfFreedom* screwDof,
    dynamics::DegreeOfFreedom* rotateDof,
    const Eigen::Vector6s& axisWorldTwist)
{
  // Special case: all angular DOFs within FreeJoints effect each other in
  // special ways
  if (screwDof->getJoint() == rotateDof->getJoint()
      && screwDof->getJoint()->getType()
             == dynamics::FreeJoint::getStaticType())
  {
    dynamics::FreeJoint* freeJoint
        = static_cast<dynamics::FreeJoint*>(screwDof->getJoint());
    int axisIndex = screwDof->getIndexInJoint();
    int rotateIndex = rotateDof->getIndexInJoint();

    return freeJoint->getScrewAxisGradientForForce(axisIndex, rotateIndex);
  }
  // Special case: all DOFs within BallJoints effect each other in special ways
  else if (
      screwDof->getJoint() == rotateDof->getJoint()
      && screwDof->getJoint()->getType()
             == dynamics::BallJoint::getStaticType())
  {
    dynamics::BallJoint* ballJoint
        = static_cast<dynamics::BallJoint*>(screwDof->getJoint());
    int axisIndex = screwDof->getIndexInJoint();
    int rotateIndex = rotateDof->getIndexInJoint();
    if (axisIndex < 3 && rotateIndex < 3)
    {
      return ballJoint->getScrewAxisGradientForForce(axisIndex, rotateIndex);
    }
  }
  // In the optimized version of this code, we ensure that we only call this
  // method if we know that the rotateDof is a parent of the screwDof

#ifndef NDEBUG
  assert(rotateDof->isParentOf(screwDof));
#endif

  Eigen::Vector6s rotateWorldTwist = getWorldScrewAxisForPosition(rotateDof);
  return math::ad(rotateWorldTwist, axisWorldTwist);
}

//==============================================================================
/// This is the analytical Jacobian for the contact position
math::LinearJacobian
DifferentiableContactConstraint::getContactPositionJacobian(
    std::shared_ptr<simulation::World> world)
{
  math::LinearJacobian jac = math::LinearJacobian::Zero(3, world->getNumDofs());
  int i = 0;
  for (auto dof : world->getDofs())
  {
    jac.col(i) = getContactPositionGradient(dof);
    i++;
  }
  return jac;
}

//==============================================================================
/// This is the analytical Jacobian for the contact position
math::LinearJacobian
DifferentiableContactConstraint::getContactPositionJacobian(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  math::LinearJacobian jac = math::LinearJacobian::Zero(3, skel->getNumDofs());
  int i = 0;
  for (auto dof : skel->getDofs())
  {
    jac.col(i) = getContactPositionGradient(dof);
    i++;
  }
  return jac;
}

//==============================================================================
/// This is the analytical Jacobian for the contact normal
math::LinearJacobian
DifferentiableContactConstraint::getContactForceDirectionJacobian(
    std::shared_ptr<simulation::World> world)
{
  math::LinearJacobian jac = math::LinearJacobian::Zero(3, world->getNumDofs());
  int i = 0;
  for (auto dof : world->getDofs())
  {
    jac.col(i) = getContactForceGradient(dof);
    i++;
  }
  assert(i == jac.cols());
  return jac;
}

//==============================================================================
/// This is the analytical Jacobian for the contact normal
math::LinearJacobian
DifferentiableContactConstraint::getContactForceDirectionJacobian(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  math::LinearJacobian jac = math::LinearJacobian::Zero(3, skel->getNumDofs());
  int i = 0;
  for (auto dof : skel->getDofs())
  {
    jac.col(i) = getContactForceGradient(dof);
    i++;
  }
  return jac;
}

//==============================================================================
math::Jacobian DifferentiableContactConstraint::getContactForceJacobian(
    std::shared_ptr<simulation::World> world)
{
  Eigen::Vector3s pos = getContactWorldPosition();
  Eigen::Vector3s dir = getContactWorldForceDirection();
  math::LinearJacobian posJac = getContactPositionJacobian(world);
  math::LinearJacobian dirJac = getContactForceDirectionJacobian(world);
  math::Jacobian jac = math::Jacobian::Zero(6, world->getNumDofs());

  // tau = pos cross dir
  for (int i = 0; i < world->getNumDofs(); i++)
  {
    jac.block<3, 1>(0, i) = pos.cross(dirJac.col(i)) + posJac.col(i).cross(dir);
  }
  // f = dir
  jac.block(3, 0, 3, world->getNumDofs()) = dirJac;

  return jac;
}

//==============================================================================
math::Jacobian DifferentiableContactConstraint::getContactForceJacobian(
    std::shared_ptr<dynamics::Skeleton> skel)
{
  Eigen::Vector3s pos = getContactWorldPosition();
  Eigen::Vector3s dir = getContactWorldForceDirection();
  math::LinearJacobian posJac = getContactPositionJacobian(skel);
  math::LinearJacobian dirJac = getContactForceDirectionJacobian(skel);
  math::Jacobian jac = math::Jacobian::Zero(6, skel->getNumDofs());

  // tau = pos cross dir
  for (int i = 0; i < skel->getNumDofs(); i++)
  {
    jac.block<3, 1>(0, i) = pos.cross(dirJac.col(i)) + posJac.col(i).cross(dir);
  }
  // f = dir
  jac.block(3, 0, 3, skel->getNumDofs()) = dirJac;

  return jac;
}

//==============================================================================
/// This gets the constraint force for a given DOF
s_t DifferentiableContactConstraint::getConstraintForce(
    dynamics::DegreeOfFreedom* dof)
{
  s_t multiple = getControlForceMultiple(dof);
  Eigen::Vector6s worldForce = getWorldForce();
  Eigen::Vector6s worldTwist = getWorldScrewAxisForForce(dof);
  return worldTwist.dot(worldForce) * multiple;
}

//==============================================================================
/// This gets the gradient of constraint force at this joint with respect to
/// another joint
s_t DifferentiableContactConstraint::getConstraintForceDerivative(
    dynamics::DegreeOfFreedom* dof, dynamics::DegreeOfFreedom* wrt)
{
  s_t multiple = getControlForceMultiple(dof);
  Eigen::Vector6s worldForce = getWorldForce();
  Eigen::Vector6s gradientOfWorldForce = getContactWorldForceGradient(wrt);
  Eigen::Vector6s gradientOfWorldTwist = getScrewAxisForForceGradient(dof, wrt);
  Eigen::Vector6s worldTwist = getWorldScrewAxisForForce(dof);
  return (worldTwist.dot(gradientOfWorldForce)
          + gradientOfWorldTwist.dot(worldForce))
         * multiple;
}

//==============================================================================
/// This returns an analytical Jacobian relating the skeletons that this
/// contact touches.
const Eigen::MatrixXs&
DifferentiableContactConstraint::getConstraintForcesJacobian(
    std::shared_ptr<simulation::World> world)
{
  if (mWorldConstraintJacCacheDirty)
  {
    int dim = world->getNumDofs();
    math::Jacobian forceJac = getContactForceJacobian(world);
    Eigen::Vector6s force = getWorldForce();
    std::vector<dynamics::DegreeOfFreedom*> dofs = world->getDofs();

    ////////////////////////////////////////////////////////////////////
    // Compute a slow, but known to be correct version
    ////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
    Eigen::MatrixXs slowJacCache = Eigen::MatrixXs::Zero(dim, dim);
    for (int row = 0; row < dim; row++)
    {
      s_t multiple = getControlForceMultiple(dofs[row]);
      if (multiple == 0.0)
        continue;

      Eigen::Vector6s axis = getWorldScrewAxisForForce(dofs[row]);

      for (int wrt = 0; wrt < dim; wrt++)
      {
        // Anything that goes in this inner loop is called O(n^2 * C), where n
        // is the number of DOFs, and c is the number of contacts. For something
        // like Atlas, with a lot of contacts and a lot of joints, this gets
        // called A LOT. So it's very important that this be fast.

        Eigen::Vector6s screwAxisGradient
            = getScrewAxisForForceGradient(dofs[row], dofs[wrt]);
        Eigen::Vector6s forceGradient = forceJac.col(wrt);
        slowJacCache(row, wrt)
            = multiple
              * (screwAxisGradient.dot(force) + axis.dot(forceGradient));
      }
    }
#endif

    ////////////////////////////////////////////////////////////////////
    // </slow version>
    ////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////
    // Compute the same thing, but hopefully faster
    ////////////////////////////////////////////////////////////////////

    mWorldConstraintJacCache = Eigen::MatrixXs::Zero(dim, dim);
    for (int row = 0; row < dim; row++)
    {
      s_t multiple = getControlForceMultiple(dofs[row]);
      if (multiple == 0.0)
        continue;

      Eigen::Vector6s axis = getWorldScrewAxisForForce(dofs[row]);
      // Eigen::Vector6s axisWorldTwist = getWorldScrewAxisForForce(dofs[row]);

      // Each element [i] of this vector is the forceJac col(i) dotted with
      // axis.
      mWorldConstraintJacCache.row(row)
          = multiple * forceJac.transpose() * axis;

      dynamics::Joint* jointCursor = dofs[row]->getJoint();

      // Include all the DOFs in this joint, if it's a FreeJoint or BallJoint
      if (jointCursor->getType() != dynamics::FreeJoint::getStaticType()
          && jointCursor->getType() != dynamics::BallJoint::getStaticType())
      {
        dynamics::BodyNode* cursorParentBody = jointCursor->getParentBodyNode();
        if (cursorParentBody != nullptr)
        {
          jointCursor = cursorParentBody->getParentJoint();
        }
        else
        {
          jointCursor = nullptr;
        }
      }

      while (jointCursor != nullptr)
      {
        for (int i = 0; i < jointCursor->getNumDofs(); i++)
        {
          int wrt = jointCursor->getIndexInSkeleton(i)
                    + world->getSkeletonDofOffset(jointCursor->getSkeleton());
          Eigen::Vector6s screwAxisGradient
              = getScrewAxisForForceGradient_Optimized(
                  dofs[row], dofs[wrt], axis); // axisWorldTwist
          mWorldConstraintJacCache(row, wrt)
              += multiple * screwAxisGradient.dot(force);
        }
        dynamics::BodyNode* cursorParentBody = jointCursor->getParentBodyNode();
        if (cursorParentBody != nullptr)
        {
          jointCursor = cursorParentBody->getParentJoint();
        }
        else
        {
          jointCursor = nullptr;
        }
      }

      /*
      for (int wrt = 0; wrt < dim; wrt++)
      {
        // Anything that goes in this inner loop is called O(n^2 * C), where n
        // is the number of DOFs, and c is the number of contacts. For something
        // like Atlas, with a lot of contacts and a lot of joints, this gets
        // called A LOT. So it's very important that this be fast.

        Eigen::Vector6s screwAxisGradient
            = getScrewAxisForForceGradient_Optimized(
                dofs[row], dofs[wrt], axisWorldTwist);
        Eigen::Vector6s forceGradient = forceJac.col(wrt);
        mWorldConstraintJacCache(row, wrt)
            = multiple
              * (screwAxisGradient.dot(force) + axis.dot(forceGradient));
      }
      */
    }

    ////////////////////////////////////////////////////////////////////
    // </fast version>
    ////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
    if (slowJacCache != mWorldConstraintJacCache)
    {
      std::cout << "Slow version" << std::endl << slowJacCache << std::endl;
      std::cout << "Faster version" << std::endl
                << mWorldConstraintJacCache << std::endl;
      std::cout << "Diff" << std::endl
                << slowJacCache - mWorldConstraintJacCache << std::endl;
    }
    assert(slowJacCache == mWorldConstraintJacCache);
#endif

    mWorldConstraintJacCacheDirty = false;
  }

  return mWorldConstraintJacCache;
}

//==============================================================================
/// This computes and returns the analytical Jacobian relating how changes in
/// the positions of wrt's DOFs changes the constraint forces on skel.
Eigen::MatrixXs DifferentiableContactConstraint::getConstraintForcesJacobian(
    std::shared_ptr<dynamics::Skeleton> skel,
    std::shared_ptr<dynamics::Skeleton> wrt)
{
  math::Jacobian forceJac = getContactForceJacobian(wrt);
  Eigen::Vector6s force = getWorldForce();

  Eigen::MatrixXs result
      = Eigen::MatrixXs::Zero(skel->getNumDofs(), wrt->getNumDofs());
  for (int row = 0; row < skel->getNumDofs(); row++)
  {
    Eigen::Vector6s axis = getWorldScrewAxisForForce(skel->getDof(row));
    s_t multiple = getControlForceMultiple(skel->getDof(row));
    if (multiple != 0)
    {
      for (int col = 0; col < wrt->getNumDofs(); col++)
      {
        Eigen::Vector6s screwAxisGradient
            = getScrewAxisForForceGradient(skel->getDof(row), wrt->getDof(col));
        Eigen::Vector6s forceGradient = forceJac.col(col);
        result(row, col)
            = multiple
              * (screwAxisGradient.dot(force) + axis.dot(forceGradient));
      }
    }
    else
    {
      result.row(row).setZero();
    }
  }

  return result;
}

//==============================================================================
/// This computes and returns the analytical Jacobian relating how changes in
/// the positions of wrt's DOFs changes the constraint forces on all the
/// skels.
Eigen::MatrixXs DifferentiableContactConstraint::getConstraintForcesJacobian(
    std::vector<std::shared_ptr<dynamics::Skeleton>> skels,
    std::shared_ptr<dynamics::Skeleton> wrt)
{
  math::Jacobian forceJac = getContactForceJacobian(wrt);
  Eigen::Vector6s force = getWorldForce();

  int numRows = 0;
  for (auto skel : skels)
    numRows += skel->getNumDofs();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(numRows, wrt->getNumDofs());

  int row = 0;
  for (auto skel : skels)
  {
    for (int i = 0; i < skel->getNumDofs(); i++)
    {
      s_t multiple = getControlForceMultiple(skel->getDof(i));
      if (multiple != 0)
      {
        Eigen::Vector6s axis = getWorldScrewAxisForForce(skel->getDof(i));
        for (int col = 0; col < wrt->getNumDofs(); col++)
        {
          Eigen::Vector6s screwAxisGradient
              = getScrewAxisForForceGradient(skel->getDof(i), wrt->getDof(col));
          Eigen::Vector6s forceGradient = forceJac.col(col);
          result(row, col)
              = multiple
                * (screwAxisGradient.dot(force) + axis.dot(forceGradient));
        }
      }
      else
      {
        result.row(row).setZero();
      }
      row++;
    }
  }

  return result;
}

//==============================================================================
/// This computes and returns the analytical Jacobian relating how changes in
/// the positions of any of the DOFs changes the constraint forces on all the
/// skels.
Eigen::MatrixXs DifferentiableContactConstraint::getConstraintForcesJacobian(
    std::shared_ptr<simulation::World> world,
    std::vector<std::shared_ptr<dynamics::Skeleton>> skels)
{
  int dofs = 0;
  for (auto skel : skels)
    dofs += skel->getNumDofs();

  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(dofs, dofs);

  /*
  int cursor = 0;
  for (auto skel : skels)
  {
    result.block(0, cursor, dofs, skel->getNumDofs())
        = getConstraintForcesJacobian(skels, skel);
    cursor += skel->getNumDofs();
  }
  */

  const Eigen::MatrixXs& cachedWorld = getConstraintForcesJacobian(world);
  int rowCursor = 0;
  for (auto rowSkel : skels)
  {
    int rowDof = rowSkel->getNumDofs();
    if (rowDof == 0)
      continue;
    int rowWorldOffset = world->getSkeletonDofOffset(rowSkel);
    int colCursor = 0;
    for (auto colSkel : skels)
    {
      int colDof = colSkel->getNumDofs();
      if (colDof == 0)
        continue;
      int colWorldOffset = world->getSkeletonDofOffset(colSkel);

      result.block(rowCursor, colCursor, rowDof, colDof)
          = cachedWorld.block(rowWorldOffset, colWorldOffset, rowDof, colDof);
      colCursor += colDof;
    }
    rowCursor += rowDof;
  }

  return result;
}

//==============================================================================
/// The linear Jacobian for the contact position
math::LinearJacobian
DifferentiableContactConstraint::bruteForceContactPositionJacobian(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  int dofs = world->getNumDofs();
  math::LinearJacobian jac = math::LinearJacobian(3, dofs);

  const s_t EPS = 1e-7;

  Eigen::VectorXs positions = world->getPositions();

  for (int i = 0; i < dofs; i++)
  {
    snapshot.restore();
    Eigen::VectorXs perturbedPositions = positions;
    perturbedPositions(i) += EPS;
    world->setPositions(perturbedPositions);
    std::shared_ptr<BackpropSnapshot> backpropSnapshot
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraintPos
        = getPeerConstraint(backpropSnapshot);

    snapshot.restore();
    perturbedPositions = positions;
    perturbedPositions(i) -= EPS;
    world->setPositions(perturbedPositions);
    std::shared_ptr<BackpropSnapshot> backpropSnapshotNeg
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraintNeg
        = getPeerConstraint(backpropSnapshotNeg);

    jac.col(i) = (peerConstraintPos->getContactWorldPosition()
                  - peerConstraintNeg->getContactWorldPosition())
                 / (2 * EPS);
  }

  snapshot.restore();

  return jac;
}

//==============================================================================
/// The linear Jacobian for the contact normal
math::LinearJacobian
DifferentiableContactConstraint::bruteForceContactForceDirectionJacobian(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  int dofs = world->getNumDofs();
  math::LinearJacobian jac = math::LinearJacobian(3, dofs);

  const s_t EPS = 1e-6;

  Eigen::VectorXs positions = world->getPositions();

  for (int i = 0; i < dofs; i++)
  {
    snapshot.restore();
    Eigen::VectorXs perturbedPositions = positions;
    perturbedPositions(i) += EPS;
    world->setPositions(perturbedPositions);
    std::shared_ptr<BackpropSnapshot> backpropSnapshotPos
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraintPos
        = getPeerConstraint(backpropSnapshotPos);

    snapshot.restore();
    perturbedPositions = positions;
    perturbedPositions(i) -= EPS;
    world->setPositions(perturbedPositions);
    std::shared_ptr<BackpropSnapshot> backpropSnapshotNeg
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraintNeg
        = getPeerConstraint(backpropSnapshotNeg);

    jac.col(i) = (peerConstraintPos->getContactWorldForceDirection()
                  - peerConstraintNeg->getContactWorldForceDirection())
                 / (2 * EPS);
  }

  snapshot.restore();

  return jac;
}

//==============================================================================
/// This is the brute force version of getWorldForceJacobian()
math::Jacobian DifferentiableContactConstraint::bruteForceContactForceJacobian(
    std::shared_ptr<simulation::World> world)
{
  RestorableSnapshot snapshot(world);

  int dofs = world->getNumDofs();
  math::Jacobian jac = math::Jacobian(6, dofs);

  const s_t EPS = 1e-6;

  Eigen::VectorXs positions = world->getPositions();

  for (int i = 0; i < dofs; i++)
  {
    snapshot.restore();
    Eigen::VectorXs perturbedPositions = positions;
    perturbedPositions(i) += EPS;
    world->setPositions(perturbedPositions);
    std::shared_ptr<BackpropSnapshot> backpropSnapshot
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraint
        = getPeerConstraint(backpropSnapshot);

    snapshot.restore();
    perturbedPositions = positions;
    perturbedPositions(i) -= EPS;
    world->setPositions(perturbedPositions);
    std::shared_ptr<BackpropSnapshot> backpropSnapshotNeg
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraintNeg
        = getPeerConstraint(backpropSnapshotNeg);

    jac.col(i)
        = (peerConstraint->getWorldForce() - peerConstraintNeg->getWorldForce())
          / (2 * EPS);
  }

  snapshot.restore();

  return jac;
}

//==============================================================================
/// This is the brute force version of getConstraintForcesJacobian()
Eigen::MatrixXs
DifferentiableContactConstraint::bruteForceConstraintForcesJacobian(
    std::shared_ptr<simulation::World> world)
{
  int dims = world->getNumDofs();
  Eigen::MatrixXs result = Eigen::MatrixXs::Zero(dims, dims);

  RestorableSnapshot snapshot(world);

  Eigen::VectorXs originalPosition = world->getPositions();
  const s_t EPS = 1e-7;

  std::shared_ptr<BackpropSnapshot> originalBackpropSnapshot
      = neural::forwardPass(world, true);
  std::shared_ptr<DifferentiableContactConstraint> originalPeerConstraint
      = getPeerConstraint(originalBackpropSnapshot);
  Eigen::VectorXs originalOut
      = originalPeerConstraint->getConstraintForces(world.get());

  for (int i = 0; i < dims; i++)
  {
    Eigen::VectorXs tweakedPosition = originalPosition;
    tweakedPosition(i) += EPS;
    world->setPositions(tweakedPosition);
    std::shared_ptr<BackpropSnapshot> backpropSnapshot
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraint
        = getPeerConstraint(backpropSnapshot);
    Eigen::VectorXs newOut = peerConstraint->getConstraintForces(world.get());

    tweakedPosition = originalPosition;
    tweakedPosition(i) -= EPS;
    world->setPositions(tweakedPosition);
    std::shared_ptr<BackpropSnapshot> backpropSnapshotNeg
        = neural::forwardPass(world, true);
    std::shared_ptr<DifferentiableContactConstraint> peerConstraintNeg
        = getPeerConstraint(backpropSnapshotNeg);
    Eigen::VectorXs newOutNeg
        = peerConstraintNeg->getConstraintForces(world.get());

    result.col(i) = (newOut - newOutNeg) / (2 * EPS);
  }

  snapshot.restore();

  return result;
}

//==============================================================================
Eigen::Vector3s
DifferentiableContactConstraint::estimatePerturbedContactPosition(
    std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, s_t eps)
{
  Eigen::Vector3s contactPos = getContactWorldPosition();
  DofContactType type = getDofContactType(skel->getDof(dofIndex));

  if (type == SPHERE_A)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    s_t weight = mContact->radiusB / (mContact->radiusA + mContact->radiusB);
    Eigen::Vector3s posDiff
        = (rotation * mContact->centerA) - mContact->centerA;
    posDiff *= weight;
    return contactPos + posDiff;
  }
  else if (type == SPHERE_B)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    s_t weight = mContact->radiusA / (mContact->radiusA + mContact->radiusB);
    Eigen::Vector3s posDiff
        = (rotation * mContact->centerB) - mContact->centerB;
    posDiff *= weight;
    return contactPos + posDiff;
  }
  else if (type == SPHERE_TO_BOX)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    Eigen::Vector3s perturbedSphereCenter = rotation * mContact->sphereCenter;
    Eigen::Vector3s diff = perturbedSphereCenter - mContact->sphereCenter;
    if (mContact->face1Locked)
    {
      diff = diff - mContact->face1Normal * mContact->face1Normal.dot(diff);
    }
    if (mContact->face2Locked)
    {
      diff = diff - mContact->face2Normal * mContact->face2Normal.dot(diff);
    }
    if (mContact->face3Locked)
    {
      diff = diff - mContact->face3Normal * mContact->face3Normal.dot(diff);
    }
    return contactPos + diff;
  }
  else if (type == BOX_TO_SPHERE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    // Step 1. First pretend the sphere is moving relative to the box, by the
    // inverse of the box transform
    Eigen::Vector3s inversePerturbedSphereCenter
        = rotation.inverse() * mContact->sphereCenter;
    Eigen::Vector3s diff
        = inversePerturbedSphereCenter - mContact->sphereCenter;
    if (mContact->face1Locked)
    {
      diff = diff - mContact->face1Normal * mContact->face1Normal.dot(diff);
    }
    if (mContact->face2Locked)
    {
      diff = diff - mContact->face2Normal * mContact->face2Normal.dot(diff);
    }
    if (mContact->face3Locked)
    {
      diff = diff - mContact->face3Normal * mContact->face3Normal.dot(diff);
    }
    // Step 2. Now reverse the result by moving the point by the box transform,
    // which undoes the sphere "movement" but leaves helpful clipping effects in
    // place.
    Eigen::Vector3s inverseSphereMovement = contactPos + diff;
    return rotation * inverseSphereMovement;
  }
  else if (type == VERTEX || type == SELF_COLLISION || type == VERTEX_TO_SPHERE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    Eigen::Vector3s perturbedContactPos = rotation * contactPos;
    return perturbedContactPos;
  }
  else if (type == FACE || type == SPHERE_TO_VERTEX)
  {
    return contactPos;
  }
  else if (type == EDGE_A)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();
    Eigen::Vector3s contactPoint = math::getContactPoint(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->edgeBFixedPoint,
        mContact->edgeBDir);
    return contactPoint;
  }
  else if (type == EDGE_B)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();
    Eigen::Vector3s contactPoint = math::getContactPoint(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->edgeBFixedPoint,
        rotation * mContact->edgeBDir);
    return contactPoint;
  }
  else if (type == SPHERE_TO_FACE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Vector3s diff
        = (translation * mContact->sphereCenter) - mContact->sphereCenter;
    return mContact->point + diff;
  }
  else if (type == FACE_TO_SPHERE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    rotation.translation().setZero();
    Eigen::Vector3s diff = ((rotation * mContact->normal) - mContact->normal)
                           * mContact->sphereRadius;
    if (getContactType() == collision::ContactType::SPHERE_FACE)
    {
      return mContact->point - diff;
    }
    else
    {
      return mContact->point + diff;
    }
  }
  else if (type == SPHERE_TO_EDGE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    return math::closestPointOnLine(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->sphereCenter);
  }
  else if (type == EDGE_TO_SPHERE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    rotation.translation().setZero();
    return math::closestPointOnLine(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->sphereCenter);
  }
  else if (type == SPHERE_TO_PIPE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    s_t weight = mContact->pipeRadius
                    / (mContact->sphereRadius + mContact->pipeRadius);

    Eigen::Vector3s posDiff
        = (rotation * mContact->sphereCenter) - mContact->sphereCenter;
    Eigen::Vector3s parallelComponent
        = mContact->pipeDir.dot(posDiff) * mContact->pipeDir;
    Eigen::Vector3s perpendicularComponent = posDiff - parallelComponent;

    Eigen::Vector3s weightedDiff
        = parallelComponent + (weight * perpendicularComponent);

    return contactPos + weightedDiff;
  }
  else if (type == PIPE_TO_SPHERE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    rotation.translation().setZero();
    Eigen::Vector3s posDiff = math::closestPointOnLine(
                                  translation * mContact->pipeFixedPoint,
                                  rotation * mContact->pipeDir,
                                  mContact->sphereCenter)
                              - mContact->pipeClosestPoint;

    s_t weight = mContact->sphereRadius
                    / (mContact->sphereRadius + mContact->pipeRadius);

    return contactPos + (weight * posDiff);
  }
  else if (type == PIPE_A)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();
    Eigen::Vector3s contactPoint = math::getContactPoint(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->edgeBFixedPoint,
        mContact->edgeBDir,
        mContact->radiusA,
        mContact->radiusB);
    return contactPoint;
  }
  else if (type == PIPE_B)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();
    Eigen::Vector3s contactPoint = math::getContactPoint(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->edgeBFixedPoint,
        rotation * mContact->edgeBDir,
        mContact->radiusA,
        mContact->radiusB);
    return contactPoint;
  }
  else if (type == VERTEX_TO_PIPE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    return translation * mContact->point;
  }
  else if (type == PIPE_TO_VERTEX)
  {
    return mContact->point;
  }
  else if (type == DofContactType::PIPE_TO_EDGE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();
    Eigen::Vector3s contactPoint = math::getContactPoint(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->pipeFixedPoint,
        rotation * mContact->pipeDir,
        0.0,
        1.0);
    return contactPoint;
  }
  else if (type == DofContactType::EDGE_TO_PIPE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();
    Eigen::Vector3s contactPoint = math::getContactPoint(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->pipeFixedPoint,
        mContact->pipeDir,
        0.0,
        1.0);
    return contactPoint;
  }

  // Default case
  return contactPos;
}

//==============================================================================
Eigen::Vector3s DifferentiableContactConstraint::estimatePerturbedContactNormal(
    std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, s_t eps)
{
  Eigen::Vector3s normal = getContactWorldNormal();
  DofContactType type = getDofContactType(skel->getDof(dofIndex));
  if (type == VERTEX || type == SPHERE_TO_FACE)
  {
    return normal;
  }

  Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
  Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
  if (type == SPHERE_A)
  {
    s_t norm = (mContact->centerA - mContact->centerB).norm();
    Eigen::Vector3s posDiff
        = (rotation * mContact->centerA) - mContact->centerA;
    posDiff /= norm;
    return (normal + posDiff).normalized();
  }
  else if (type == SPHERE_B)
  {
    s_t norm = (mContact->centerA - mContact->centerB).norm();
    Eigen::Vector3s posDiff
        = (rotation * mContact->centerB) - mContact->centerB;
    posDiff /= norm;
    return (normal - posDiff).normalized();
  }
  else if (type == SPHERE_TO_BOX)
  {
    s_t norm = (mContact->sphereCenter - mContact->point).norm();
    Eigen::Vector3s newContactPos
        = estimatePerturbedContactPosition(skel, dofIndex, eps);
    Eigen::Vector3s contactDiff = newContactPos - mContact->point;
    Eigen::Vector3s sphereDiff
        = (rotation * mContact->sphereCenter) - mContact->sphereCenter;
    if (norm > 1e-5)
    {
      contactDiff /= norm;
      sphereDiff /= norm;
    }
    Eigen::Vector3s perturbedNormal = normal;
    // The normal flips direction depending on which order the contact detection
    // was called in :(
    if (mContact->type == collision::BOX_SPHERE)
    {
      // normal = contact_pt - sphereCenter
      perturbedNormal += contactDiff - sphereDiff;
    }
    else if (mContact->type == collision::SPHERE_BOX)
    {
      // normal = sphereCenter - contact_pt
      perturbedNormal += sphereDiff - contactDiff;
    }
    return perturbedNormal.normalized();
  }
  else if (type == BOX_TO_SPHERE)
  {
    s_t norm = (mContact->sphereCenter - mContact->point).norm();
    Eigen::Vector3s newContactPos
        = estimatePerturbedContactPosition(skel, dofIndex, eps);
    Eigen::Vector3s contactDiff = newContactPos - mContact->point;
    if (norm > 1e-5)
    {
      contactDiff /= norm;
    }
    // sphereDiff = 0 here, because the sphere doesn't move when we perturb the
    // box, because they're on different skeleton branches
    Eigen::Vector3s perturbedNormal = normal;
    // The normal flips direction depending on which order the contact detection
    // was called in :(
    if (mContact->type == collision::BOX_SPHERE)
    {
      // normal = contact_pt - sphereCenter
      perturbedNormal += contactDiff;
    }
    else if (mContact->type == collision::SPHERE_BOX)
    {
      // normal = sphereCenter - contact_pt
      perturbedNormal -= contactDiff;
    }
    return perturbedNormal.normalized();
  }
  else if (type == FACE || type == SELF_COLLISION || type == FACE_TO_SPHERE)
  {
    rotation.translation().setZero();
    Eigen::Vector3s perturbedNormal = rotation * normal;
    return perturbedNormal;
  }
  else if (type == EDGE_A)
  {
    rotation.translation().setZero();
    Eigen::Vector3s normal
        = (mContact->edgeBDir).cross(rotation * mContact->edgeADir);
    if (normal.dot(mContact->normal) < 0)
    {
      normal *= -1;
    }
    return normal;
  }
  else if (type == EDGE_B)
  {
    rotation.translation().setZero();
    Eigen::Vector3s normal
        = (rotation * mContact->edgeBDir).cross(mContact->edgeADir);
    if (normal.dot(mContact->normal) < 0)
    {
      normal *= -1;
    }
    return normal;
  }
  else if (type == SPHERE_TO_VERTEX)
  {
    s_t norm = (mContact->sphereCenter - mContact->vertexPoint).norm();
    Eigen::Vector3s posDiff
        = (rotation * mContact->sphereCenter) - mContact->sphereCenter;
    posDiff /= norm;
    if (getContactType() == collision::ContactType::SPHERE_VERTEX)
    {
      return (normal + posDiff).normalized();
    }
    else
    {
      return (normal - posDiff).normalized();
    }
  }
  else if (type == VERTEX_TO_SPHERE)
  {
    s_t norm = (mContact->vertexPoint - mContact->sphereCenter).norm();
    Eigen::Vector3s posDiff
        = (rotation * mContact->vertexPoint) - mContact->vertexPoint;
    posDiff /= norm;
    if (getContactType() == collision::ContactType::VERTEX_SPHERE)
    {
      return (normal + posDiff).normalized();
    }
    else
    {
      return (normal - posDiff).normalized();
    }
  }
  else if (type == SPHERE_TO_EDGE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);

    Eigen::Vector3s closestPoint = math::closestPointOnLine(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->sphereCenter);
    Eigen::Vector3s sphereCenter = translation * mContact->sphereCenter;
    Eigen::Vector3s normal = (closestPoint - sphereCenter).normalized();

    if (getContactType() == collision::ContactType::SPHERE_EDGE)
    {
      return -normal;
    }
    else
    {
      return normal;
    }
  }
  else if (type == EDGE_TO_SPHERE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    rotation.translation().setZero();
    Eigen::Vector3s closestPoint = math::closestPointOnLine(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->sphereCenter);
    Eigen::Vector3s normal
        = (closestPoint - mContact->sphereCenter).normalized();

    if (getContactType() == collision::ContactType::EDGE_SPHERE)
    {
      return normal;
    }
    else
    {
      return -normal;
    }
  }
  else if (type == SPHERE_TO_PIPE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    s_t norm = (mContact->sphereCenter - mContact->pipeClosestPoint).norm();
    Eigen::Vector3s posDiff
        = (rotation * mContact->sphereCenter) - mContact->sphereCenter;
    posDiff -= posDiff.dot(mContact->pipeDir) * mContact->pipeDir;
    posDiff /= norm;
    if (getContactType() == collision::ContactType::SPHERE_PIPE)
    {
      return (normal + posDiff).normalized();
    }
    else
    {
      return (normal - posDiff).normalized();
    }
  }
  else if (type == PIPE_TO_SPHERE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = math::expMap(worldTwist * eps);
    rotation.translation().setZero();
    s_t norm = (mContact->sphereCenter - mContact->pipeClosestPoint).norm();
    Eigen::Vector3s posDiff = math::closestPointOnLine(
                                  translation * mContact->pipeFixedPoint,
                                  rotation * mContact->pipeDir,
                                  mContact->sphereCenter)
                              - mContact->pipeClosestPoint;
    posDiff /= norm;
    if (getContactType() == collision::ContactType::PIPE_SPHERE)
    {
      return (normal + posDiff).normalized();
    }
    else
    {
      return (normal - posDiff).normalized();
    }
  }
  else if (type == PIPE_A)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();

    Eigen::Vector3s pipeANearestPoint = math::getContactPoint(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->edgeBFixedPoint,
        mContact->edgeBDir,
        0.0,
        1.0);

    Eigen::Vector3s pipeBNearestPoint = math::getContactPoint(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->edgeBFixedPoint,
        mContact->edgeBDir,
        1.0,
        0.0);

    return (pipeANearestPoint - pipeBNearestPoint).normalized();
  }
  else if (type == PIPE_B)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();

    Eigen::Vector3s pipeANearestPoint = math::getContactPoint(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->edgeBFixedPoint,
        rotation * mContact->edgeBDir,
        0.0,
        1.0);

    Eigen::Vector3s pipeBNearestPoint = math::getContactPoint(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->edgeBFixedPoint,
        rotation * mContact->edgeBDir,
        1.0,
        0.0);

    return (pipeANearestPoint - pipeBNearestPoint).normalized();
  }
  else if (type == VERTEX_TO_PIPE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Vector3s vertex = translation * mContact->point;
    Eigen::Vector3s closestPoint = math::closestPointOnLine(
        mContact->pipeFixedPoint, mContact->pipeDir, vertex);
    if (getContactType() == collision::ContactType::PIPE_VERTEX)
    {
      return (closestPoint - vertex).normalized();
    }
    else
    {
      return (vertex - closestPoint).normalized();
    }
  }
  else if (type == PIPE_TO_VERTEX)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();
    Eigen::Vector3s closestPoint = math::closestPointOnLine(
        translation * mContact->pipeFixedPoint,
        rotation * mContact->pipeDir,
        mContact->point);
    if (getContactType() == collision::ContactType::PIPE_VERTEX)
    {
      return (closestPoint - mContact->point).normalized();
    }
    else
    {
      return (mContact->point - closestPoint).normalized();
    }
  }
  else if (type == DofContactType::PIPE_TO_EDGE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();

    Eigen::Vector3s pipeClosestPoint = math::getContactPoint(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->pipeFixedPoint,
        rotation * mContact->pipeDir,
        0.0,
        1.0);

    Eigen::Vector3s edgeClosestPoint = math::getContactPoint(
        mContact->edgeAFixedPoint,
        mContact->edgeADir,
        translation * mContact->pipeFixedPoint,
        rotation * mContact->pipeDir,
        1.0,
        0.0);

    if (getContactType() == collision::ContactType::PIPE_EDGE)
    {
      return (edgeClosestPoint - pipeClosestPoint).normalized();
    }
    else
    {
      return (pipeClosestPoint - edgeClosestPoint).normalized();
    }
  }
  else if (type == DofContactType::EDGE_TO_PIPE)
  {
    Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
    Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
    Eigen::Isometry3s rotation = translation;
    rotation.translation().setZero();

    Eigen::Vector3s pipeClosestPoint = math::getContactPoint(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->pipeFixedPoint,
        mContact->pipeDir,
        0.0,
        1.0);

    Eigen::Vector3s edgeClosestPoint = math::getContactPoint(
        translation * mContact->edgeAFixedPoint,
        rotation * mContact->edgeADir,
        mContact->pipeFixedPoint,
        mContact->pipeDir,
        1.0,
        0.0);

    if (getContactType() == collision::ContactType::PIPE_EDGE)
    {
      return (edgeClosestPoint - pipeClosestPoint).normalized();
    }
    else
    {
      return (pipeClosestPoint - edgeClosestPoint).normalized();
    }
  }

  // Default case
  return normal;
}

//==============================================================================
Eigen::Vector3s
DifferentiableContactConstraint::estimatePerturbedContactForceDirection(
    std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, s_t eps)
{
  Eigen::Vector3s forceDir = getContactWorldForceDirection();
  DofContactType type = getDofContactType(skel->getDof(dofIndex));
  if (type == VERTEX || type == NONE)
  {
    return forceDir;
  }
  Eigen::Vector3s contactNormal
      = estimatePerturbedContactNormal(skel, dofIndex, eps);
  if (mIndex == 0)
    return contactNormal;
  else
  {
    return mContactConstraint->getTangentBasisMatrixODE(contactNormal)
        .col(mIndex - 1);
  }
}

//==============================================================================
/// Just for testing: This analytically estimates how edges will move under a
/// perturbation
EdgeData DifferentiableContactConstraint::estimatePerturbedEdges(
    std::shared_ptr<dynamics::Skeleton> skel, int dofIndex, s_t eps)
{
  EdgeData data;
  data.edgeAPos = Eigen::Vector3s::Zero();
  data.edgeADir = Eigen::Vector3s::Zero();
  data.edgeBPos = Eigen::Vector3s::Zero();
  data.edgeBDir = Eigen::Vector3s::Zero();

  dynamics::DegreeOfFreedom* dof = skel->getDof(dofIndex);
  DofContactType type = getDofContactType(dof);

  Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(skel, dofIndex);
  Eigen::Isometry3s translation = math::expMap(worldTwist * eps);
  Eigen::Isometry3s rotation = translation;
  rotation.translation().setZero();

  if (type == EDGE_A)
  {
    data.edgeAPos = translation * mContact->edgeAFixedPoint;
    data.edgeADir = rotation * mContact->edgeADir;
    data.edgeBPos = mContact->edgeBFixedPoint;
    data.edgeBDir = mContact->edgeBDir;
  }
  else if (type == EDGE_B)
  {
    data.edgeAPos = mContact->edgeAFixedPoint;
    data.edgeADir = mContact->edgeADir;
    data.edgeBPos = translation * mContact->edgeBFixedPoint;
    data.edgeBDir = rotation * mContact->edgeBDir;
  }
  else if (type == SELF_COLLISION)
  {
    data.edgeAPos = translation * mContact->edgeAFixedPoint;
    data.edgeADir = rotation * mContact->edgeADir;
    data.edgeBPos = translation * mContact->edgeBFixedPoint;
    data.edgeBDir = rotation * mContact->edgeBDir;
  }

  return data;
}

//==============================================================================
EdgeData DifferentiableContactConstraint::getEdges()
{
  EdgeData data;
  data.edgeAPos = Eigen::Vector3s::Zero();
  data.edgeADir = Eigen::Vector3s::Zero();
  data.edgeBPos = Eigen::Vector3s::Zero();
  data.edgeBDir = Eigen::Vector3s::Zero();

  if (getContactType() == collision::ContactType::EDGE_EDGE)
  {
    data.edgeAPos = mContact->edgeAFixedPoint;
    data.edgeADir = mContact->edgeADir;
    data.edgeBPos = mContact->edgeBFixedPoint;
    data.edgeBDir = mContact->edgeBDir;
  }

  return data;
}

//==============================================================================
/// Just for testing: This analytically estimates how a screw axis will move
/// when rotated by another screw.
Eigen::Vector6s
DifferentiableContactConstraint::estimatePerturbedScrewAxisForPosition(
    dynamics::DegreeOfFreedom* axis,
    dynamics::DegreeOfFreedom* rotate,
    s_t eps)
{
  Eigen::Vector6s originalAxisWorldTwist = getWorldScrewAxisForPosition(axis);

  if (axis->getJoint() == rotate->getJoint()
      && axis->getJoint()->getType() == dynamics::FreeJoint::getStaticType())
  {
    dynamics::FreeJoint* freeJoint
        = static_cast<dynamics::FreeJoint*>(axis->getJoint());
    int axisIndex = axis->getIndexInJoint();
    int rotateIndex = rotate->getIndexInJoint();
    if (axisIndex < 3)
    {
      return freeJoint->estimatePerturbedScrewAxisForPosition(
          axisIndex, rotateIndex, eps);
    }
    else
    {
      // The translation joints aren't effected by anything
      return originalAxisWorldTwist;
    }
  }
  else if (
      axis->getJoint() == rotate->getJoint()
      && axis->getJoint()->getType() == dynamics::BallJoint::getStaticType())
  {
    dynamics::BallJoint* ballJoint
        = static_cast<dynamics::BallJoint*>(axis->getJoint());
    int axisIndex = axis->getIndexInJoint();
    int rotateIndex = rotate->getIndexInJoint();
    assert(axisIndex < 3 && rotateIndex < 3);
    return ballJoint->estimatePerturbedScrewAxisForPosition(
        axisIndex, rotateIndex, eps);
  }

  if (!rotate->isParentOfFast(axis))
    return originalAxisWorldTwist;

  Eigen::Vector6s rotateWorldTwist = getWorldScrewAxisForPosition(rotate);
  Eigen::Isometry3s transform = math::expMap(rotateWorldTwist * eps);
  Eigen::Vector6s transformedAxis
      = math::AdT(transform, originalAxisWorldTwist);
  return transformedAxis;
}

//==============================================================================
/// Just for testing: This analytically estimates how a screw axis will move
/// when rotated by another screw.
Eigen::Vector6s
DifferentiableContactConstraint::estimatePerturbedScrewAxisForForce(
    dynamics::DegreeOfFreedom* axis,
    dynamics::DegreeOfFreedom* rotate,
    s_t eps)
{
  Eigen::Vector6s originalAxisWorldTwist = getWorldScrewAxisForForce(axis);

  if (axis->getJoint() == rotate->getJoint()
      && axis->getJoint()->getType() == dynamics::FreeJoint::getStaticType())
  {
    dynamics::FreeJoint* freeJoint
        = static_cast<dynamics::FreeJoint*>(axis->getJoint());
    int axisIndex = axis->getIndexInJoint();
    int rotateIndex = rotate->getIndexInJoint();

    return freeJoint->estimatePerturbedScrewAxisForForce(
        axisIndex, rotateIndex, eps);
  }
  else if (
      axis->getJoint() == rotate->getJoint()
      && axis->getJoint()->getType() == dynamics::BallJoint::getStaticType())
  {
    dynamics::BallJoint* ballJoint
        = static_cast<dynamics::BallJoint*>(axis->getJoint());
    int axisIndex = axis->getIndexInJoint();
    int rotateIndex = rotate->getIndexInJoint();
    assert(axisIndex < 3 && rotateIndex < 3);
    return ballJoint->estimatePerturbedScrewAxisForForce(
        axisIndex, rotateIndex, eps);
  }

  if (!rotate->isParentOfFast(axis))
    return originalAxisWorldTwist;

  Eigen::Vector6s rotateWorldTwist = getWorldScrewAxisForPosition(rotate);
  Eigen::Isometry3s transform = math::expMap(rotateWorldTwist * eps);
  Eigen::Vector6s transformedAxis
      = math::AdT(transform, originalAxisWorldTwist);
  return transformedAxis;
}

//==============================================================================
void DifferentiableContactConstraint::setOffsetIntoWorld(
    int offset, bool isUpperBoundConstraint)
{
  mOffsetIntoWorld = offset;
  mIsUpperBoundConstraint = isUpperBoundConstraint;
}

//==============================================================================
Eigen::Vector3s
DifferentiableContactConstraint::bruteForcePerturbedContactPosition(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<dynamics::Skeleton> skel,
    int dofIndex,
    s_t eps)
{
  RestorableSnapshot snapshot(world);

  auto dof = skel->getDof(dofIndex);
  dof->setPosition(dof->getPosition() + eps);

  std::shared_ptr<BackpropSnapshot> backpropSnapshot
      = neural::forwardPass(world, true);
  std::shared_ptr<DifferentiableContactConstraint> peerConstraint
      = getPeerConstraint(backpropSnapshot);

  if (peerConstraint == nullptr)
  {
    std::cout << "bruteForcePerturbedContactPosition() failed to find a peer "
                 "constraint!"
              << std::endl;
    std::cout << "Perturbed snapshot num clamping: "
              << backpropSnapshot->getNumClamping() << std::endl;
    // Dirty velocities
    std::shared_ptr<DifferentiableContactConstraint> peerConstraint
        = getPeerConstraint(backpropSnapshot);
  }
  assert(peerConstraint != nullptr && "bruteForcePerturbedContactPosition() was unable to find a peer constraint to compare against.");

  snapshot.restore();

  return peerConstraint->getContactWorldPosition();
}

//==============================================================================
Eigen::Vector3s
DifferentiableContactConstraint::bruteForcePerturbedContactNormal(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<dynamics::Skeleton> skel,
    int dofIndex,
    s_t eps)
{
  RestorableSnapshot snapshot(world);

  auto dof = skel->getDof(dofIndex);
  dof->setPosition(dof->getPosition() + eps);

  std::shared_ptr<BackpropSnapshot> backpropSnapshot
      = neural::forwardPass(world, true);
  std::shared_ptr<DifferentiableContactConstraint> peerConstraint
      = getPeerConstraint(backpropSnapshot);

  snapshot.restore();

  return peerConstraint->getContactWorldNormal();
}

//==============================================================================
Eigen::Vector3s
DifferentiableContactConstraint::bruteForcePerturbedContactForceDirection(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<dynamics::Skeleton> skel,
    int dofIndex,
    s_t eps)
{
  RestorableSnapshot snapshot(world);

  auto dof = skel->getDof(dofIndex);
  dof->setPosition(dof->getPosition() + eps);

  std::shared_ptr<BackpropSnapshot> backpropSnapshot
      = neural::forwardPass(world, true);
  std::shared_ptr<DifferentiableContactConstraint> peerConstraint
      = getPeerConstraint(backpropSnapshot);

  snapshot.restore();

  return peerConstraint->getContactWorldForceDirection();
}

//==============================================================================
/// Just for testing: This perturbs the world position of a skeleton to read a
/// screw axis will move when rotated by another screw.
Eigen::Vector6s DifferentiableContactConstraint::bruteForceScrewAxisForPosition(
    dynamics::DegreeOfFreedom* axis,
    dynamics::DegreeOfFreedom* rotate,
    s_t eps)
{
  s_t originalPos = rotate->getPosition();
  rotate->setPosition(originalPos + eps);

  Eigen::Vector6s worldTwist = getWorldScrewAxisForPosition(axis);

  rotate->setPosition(originalPos);

  return worldTwist;
}

//==============================================================================
/// Just for testing: This perturbs the world position of a skeleton to read a
/// screw axis will move when rotated by another screw.
Eigen::Vector6s DifferentiableContactConstraint::bruteForceScrewAxisForForce(
    dynamics::DegreeOfFreedom* axis,
    dynamics::DegreeOfFreedom* rotate,
    s_t eps)
{
  s_t originalPos = rotate->getPosition();
  rotate->setPosition(originalPos + eps);

  Eigen::Vector6s worldTwist = getWorldScrewAxisForForce(axis);

  rotate->setPosition(originalPos);

  return worldTwist;
}

//==============================================================================
/// Just for testing: This perturbs the world position of a skeleton  to read
/// how edges will move.
EdgeData DifferentiableContactConstraint::bruteForceEdges(
    std::shared_ptr<simulation::World> world,
    std::shared_ptr<dynamics::Skeleton> skel,
    int dofIndex,
    s_t eps)
{
  EdgeData data;
  data.edgeAPos = Eigen::Vector3s::Zero();
  data.edgeADir = Eigen::Vector3s::Zero();
  data.edgeBPos = Eigen::Vector3s::Zero();
  data.edgeBDir = Eigen::Vector3s::Zero();

  if (getContactType() != collision::ContactType::EDGE_EDGE)
  {
    return data;
  }

  RestorableSnapshot snapshot(world);

  auto dof = skel->getDof(dofIndex);
  dof->setPosition(dof->getPosition() + eps);

  std::shared_ptr<BackpropSnapshot> backpropSnapshot
      = neural::forwardPass(world, true);
  std::shared_ptr<DifferentiableContactConstraint> peerConstraint
      = getPeerConstraint(backpropSnapshot);

  snapshot.restore();

  return peerConstraint->getEdges();
}

//==============================================================================
int DifferentiableContactConstraint::getIndexInConstraint()
{
  return mIndex;
}

//==============================================================================
// This returns 1.0 by default, 0.0 if this constraint doesn't effect the
// specified DOF, and -1.0 if the constraint effects this dof negatively.
s_t DifferentiableContactConstraint::getControlForceMultiple(
    dynamics::DegreeOfFreedom* dof)
{
  if (!mConstraint->isContactConstraint())
    return 1.0;

  bool isParentA = dof->isParentOfFast(mContactConstraint->getBodyNodeA());
  bool isParentB = dof->isParentOfFast(mContactConstraint->getBodyNodeB());

  // This means it's a self-collision, and we're up stream, so the net effect is
  // 0
  if (isParentA && isParentB)
  {
    return 0.0;
  }
  // If we're in skel A
  if (isParentA)
  {
    return 1.0;
  }
  // If we're in skel B
  if (isParentB)
  {
    return -1.0;
  }

  return 0.0;
}

//==============================================================================
Eigen::Vector6s DifferentiableContactConstraint::getWorldScrewAxisForPosition(
    std::shared_ptr<dynamics::Skeleton> skel, int dofIndex)
{
  return getWorldScrewAxisForPosition(skel->getDof(dofIndex));
}

//==============================================================================
Eigen::Vector6s DifferentiableContactConstraint::getWorldScrewAxisForPosition(
    dynamics::DegreeOfFreedom* dof)
{
  int jointIndex = dof->getIndexInJoint();
  return dof->getJoint()->getWorldAxisScrewForPosition(jointIndex);
}

//==============================================================================
Eigen::Vector6s DifferentiableContactConstraint::getWorldScrewAxisForForce(
    std::shared_ptr<dynamics::Skeleton> skel, int dofIndex)
{
  return getWorldScrewAxisForForce(skel->getDof(dofIndex));
}

//==============================================================================
Eigen::Vector6s DifferentiableContactConstraint::getWorldScrewAxisForForce(
    dynamics::DegreeOfFreedom* dof)
{
  int jointIndex = dof->getIndexInJoint();
  return dof->getJoint()->getWorldAxisScrewForVelocity(jointIndex);
}

//==============================================================================
std::shared_ptr<DifferentiableContactConstraint>
DifferentiableContactConstraint::getPeerConstraint(
    std::shared_ptr<neural::BackpropSnapshot> snapshot)
{
  std::vector<std::shared_ptr<DifferentiableContactConstraint>> otherConstraints
      = snapshot->getDifferentiableConstraints();

  s_t minDistance = std::numeric_limits<s_t>::infinity();
  std::shared_ptr<DifferentiableContactConstraint> closestConstraint = nullptr;
  for (std::shared_ptr<DifferentiableContactConstraint>& constraint :
       otherConstraints)
  {
    s_t distance
        = (constraint->getContactWorldPosition() - getContactWorldPosition())
              .squaredNorm()
          + (constraint->getContactWorldNormal() - getContactWorldNormal())
                .squaredNorm();
    if (mIndex == constraint->mIndex && distance < minDistance)
    {
      closestConstraint = constraint;
      minDistance = distance;
    }
  }

  assert(closestConstraint != nullptr && "This probably means eps is too large, and we're moving to a state where contact has changed.");

  return closestConstraint;
}

} // namespace neural
} // namespace dart
