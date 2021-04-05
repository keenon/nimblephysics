#include "dart/dynamics/SimpleFeatherstone.hpp"

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"

namespace dart {
namespace dynamics {

// This creates a new JointAndBody object in our vector, and returns it by
// reference
JointAndBody& SimpleFeatherstone::emplaceBack()
{
  mJointsAndBodies.emplace_back();
  mScratchSpace.emplace_back();
  return mJointsAndBodies.at(mJointsAndBodies.size() - 1);
}

int SimpleFeatherstone::len()
{
  return mJointsAndBodies.size();
}

// This computes accelerations
void SimpleFeatherstone::forwardDynamics(
    s_t* pos,
    s_t* vel,
    s_t* force,
    /* OUT */ s_t* accelerations)
{
  // Forward pass
  for (int i = 0; i < len(); i++)
  {
    mScratchSpace[i].transformFromParent
        = mJointsAndBodies[i].transformFromParent
          * math::expMap(mJointsAndBodies[i].axis * pos[i])
          * mJointsAndBodies[i].transformFromChildren;
    if (mJointsAndBodies[i].parentIndex != -1)
    {
      mScratchSpace[i].spatialVelocity
          = math::AdInvT(
                mScratchSpace[i].transformFromParent,
                mScratchSpace[mJointsAndBodies[i].parentIndex].spatialVelocity)
            + mJointsAndBodies[i].axis * vel[i];
    }
    else
    {
      mScratchSpace[i].spatialVelocity = mJointsAndBodies[i].axis * vel[i];
    }
    mScratchSpace[i].partialAcceleration = math::ad(
        mScratchSpace[i].spatialVelocity, mJointsAndBodies[i].axis * vel[i]);
    // Zero out scratch space to prepare for sums in backwards pass
    mScratchSpace[i].articulatedInertia.setZero();
    mScratchSpace[i].articulatedBiasForce.setZero();
  }
  // Backward pass
  for (int i = len() - 1; i >= 0; i--)
  {
    mScratchSpace[i].articulatedInertia += mJointsAndBodies[i].inertia;
    mScratchSpace[i].articulatedBiasForce -= math::dad(
        mScratchSpace[i].spatialVelocity,
        mJointsAndBodies[i].inertia * mScratchSpace[i].spatialVelocity);

    mScratchSpace[i].psi
        = 1.0
          / (mJointsAndBodies[i].axis.transpose()
             * mScratchSpace[i].articulatedInertia * mJointsAndBodies[i].axis)
                .value();

    mScratchSpace[i].phi
        = mScratchSpace[i].psi * mScratchSpace[i].articulatedInertia
          * mJointsAndBodies[i].axis * mJointsAndBodies[i].axis.transpose()
          * mScratchSpace[i].articulatedInertia;

    // Total force on the joint, see GenericJoint.hpp:2028 for DART equivalent
    // Inside GenericJoint::addChildBiasForceToDynamic()
    mScratchSpace[i].totalForce
        = (force[i]
           - (mJointsAndBodies[i].axis.transpose()
              * (mScratchSpace[i].articulatedInertia
                     * mScratchSpace[i].partialAcceleration
                 + mScratchSpace[i].articulatedBiasForce))
                 .value());

    if (mJointsAndBodies[i].parentIndex == -1)
      continue;

    // Sum into our parents
    // See GenericJoint.hpp:1801 for DART equivalent,
    // GenericJoint::addChildArtInertiaToDynamic()
    // AIS = Articulated_Inertia_times_axiS
    Eigen::Vector6s AIS
        = mScratchSpace[i].articulatedInertia * mJointsAndBodies[i].axis;
    Eigen::MatrixXs PI = mScratchSpace[i].articulatedInertia;
    PI.noalias() -= AIS * mScratchSpace[i].psi * AIS.transpose();
    mScratchSpace[mJointsAndBodies[i].parentIndex].articulatedInertia
        += math::transformInertia(
            mScratchSpace[i].transformFromParent.inverse(), PI);

    Eigen::Vector6s beta
        = mScratchSpace[i].articulatedBiasForce
          + mScratchSpace[i].articulatedInertia
                * (mScratchSpace[i].partialAcceleration
                   + mJointsAndBodies[i].axis * mScratchSpace[i].psi
                         * mScratchSpace[i].totalForce);

    mScratchSpace[mJointsAndBodies[i].parentIndex].articulatedBiasForce
        += math::dAdInvT(mScratchSpace[i].transformFromParent, beta);
  }
  // Last forward pass
  for (int i = 0; i < len(); i++)
  {
    accelerations[i]
        = mScratchSpace[i].psi
          * (force[i]
             - (mJointsAndBodies[i].axis.transpose()
                * mScratchSpace[i].articulatedInertia
                * (mJointsAndBodies[i].parentIndex != -1
                       ? (math::AdInvT(
                              mScratchSpace[i].transformFromParent,
                              mScratchSpace[mJointsAndBodies[i].parentIndex]
                                  .spatialAcceleration)
                          + mScratchSpace[i].partialAcceleration)
                       : (mScratchSpace[i].partialAcceleration)))
                   .value()
             - (mJointsAndBodies[i].axis.transpose()
                * mScratchSpace[i].articulatedBiasForce)
                   .value());
    mScratchSpace[i].spatialAcceleration
        = accelerations[i] * mJointsAndBodies[i].axis
          + mScratchSpace[i].partialAcceleration;
    if (mJointsAndBodies[i].parentIndex != -1)
      mScratchSpace[i].spatialAcceleration += math::AdInvT(
          mScratchSpace[i].transformFromParent,
          mScratchSpace[mJointsAndBodies[i].parentIndex].spatialAcceleration);
  }
}

// This gets the values from a DART skeleton to populate our Featherstone
// implementation
void SimpleFeatherstone::populateFromSkeleton(
    const std::shared_ptr<dynamics::Skeleton>& skeleton)
{
  for (int i = 0; i < skeleton->getNumDofs(); i++)
  {
    JointAndBody& jointAndBody = emplaceBack();

    DegreeOfFreedom* dof = skeleton->getDof(i);
    assert(
        dof->getIndexInJoint() == 0
        && "SimpleFeatherstone only supports a single DOF per joint");
    jointAndBody.axis = dof->getJoint()->getRelativeJacobian().col(0);
    jointAndBody.transformFromChildren
        = dof->getJoint()->getTransformFromChildBodyNode();
    jointAndBody.transformFromParent
        = dof->getJoint()->getTransformFromParentBodyNode();
    jointAndBody.inertia
        = dof->getChildBodyNode()->getInertia().getSpatialTensor();
    jointAndBody.parentIndex = -1;
    if (dof->getJoint()->getParentBodyNode() != nullptr)
    {
      jointAndBody.parentIndex = dof->getJoint()
                                     ->getParentBodyNode()
                                     ->getParentJoint()
                                     ->getDof(0)
                                     ->getIndexInSkeleton();
    }
  }
}

} // namespace dynamics
} // namespace dart