#include <iostream>

#include <gtest/gtest.h>

#include "dart/collision/CollisionObject.hpp"
#include "dart/collision/Contact.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/SimpleFeatherstone.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/DifferentiableContactConstraint.hpp"
#include "dart/neural/NeuralConstants.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/neural/WithRespectToMass.hpp"
#include "dart/simulation/World.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"
#include "stdio.h"

#define ALL_TESTS

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;
using namespace neural;

void verifySkeleton(SkeletonPtr skel)
{
  skel->setGravity(Eigen::Vector3s::Zero());

  dynamics::SimpleFeatherstone simple;
  simple.populateFromSkeleton(skel);
  s_t* pos = (s_t*)malloc(sizeof(s_t) * simple.len());
  s_t* vel = (s_t*)malloc(sizeof(s_t) * simple.len());
  s_t* force = (s_t*)malloc(sizeof(s_t) * simple.len());
  s_t* accel = (s_t*)malloc(sizeof(s_t) * simple.len());

  for (int j = 0; j < 10; j++)
  {
    skel->setPositions(Eigen::VectorXs::Random(skel->getNumDofs()));
    skel->setVelocities(Eigen::VectorXs::Random(skel->getNumDofs()));
    skel->setControlForces(Eigen::VectorXs::Random(skel->getNumDofs()));

    for (int i = 0; i < simple.len(); i++)
    {
      pos[i] = skel->getPosition(i);
      vel[i] = skel->getVelocity(i);
      force[i] = skel->getControlForce(i);
    }

    // make "accel" hold acceleration according to SimpleFeatherstone
    simple.forwardDynamics(pos, vel, force, accel);
    Eigen::VectorXs simpleAccel
        = Eigen::Map<Eigen::VectorXs>(accel, simple.len());
    // get real acceleration
    skel->computeForwardDynamics();
    Eigen::VectorXs realAccel = skel->getAccelerations();

    if (!equals(simpleAccel, realAccel))
    {
      std::cout << "Expected acceleration: " << std::endl
                << realAccel << std::endl;
      std::cout << "Got acceleration: " << std::endl
                << simpleAccel << std::endl;

      for (int i = 0; i < skel->getNumDofs(); i++)
      {
        auto dof = skel->getDof(i);
        std::cout << "Checking DOF " << i << ": " << std::endl;
        if (!equals(
                dof->getChildBodyNode()->getArticulatedInertia(),
                simple.mScratchSpace[i].articulatedInertia))
        {
          std::cout << "Expected articulated inertia " << i << ": " << std::endl
                    << dof->getChildBodyNode()->getArticulatedInertia()
                    << std::endl;
          std::cout << "Got articulated inertia " << i << ": " << std::endl
                    << simple.mScratchSpace[i].articulatedInertia << std::endl;
        }
        if (!equals(
                dof->getChildBodyNode()->getPartialAcceleration(),
                simple.mScratchSpace[i].partialAcceleration))
        {
          std::cout << "Expected partial acceleration " << i << ": "
                    << std::endl
                    << dof->getChildBodyNode()->getPartialAcceleration()
                    << std::endl;
          std::cout << "Got partial acceleration " << i << ": " << std::endl
                    << simple.mScratchSpace[i].partialAcceleration << std::endl;
        }
        if (!equals(
                dof->getChildBodyNode()->getSpatialVelocity(),
                simple.mScratchSpace[i].spatialVelocity))
        {
          std::cout << "Expected spatial velocity " << i << ": " << std::endl
                    << dof->getChildBodyNode()->getSpatialVelocity()
                    << std::endl;
          std::cout << "Got spatial velocity " << i << ": " << std::endl
                    << simple.mScratchSpace[i].spatialVelocity << std::endl;
        }
        if (!equals(
                dof->getChildBodyNode()->getBiasForce(),
                simple.mScratchSpace[i].articulatedBiasForce))
        {
          std::cout << "Expected bias force " << i << ": " << std::endl
                    << dof->getChildBodyNode()->getBiasForce() << std::endl;
          std::cout << "Got bias force " << i << ": " << std::endl
                    << simple.mScratchSpace[i].articulatedBiasForce
                    << std::endl;
        }
        /*
        childJoint->addChildBiasForceTo(
            mBiasForce,
            childBodyNode->getArticulatedInertiaImplicit(),
            childBodyNode->mBiasForce,
            childBodyNode->getPartialAcceleration());

        */
      }

      EXPECT_TRUE(equals(simpleAccel, realAccel));
      return;
    }
  }

  free(pos);
  free(vel);
  free(force);
  free(accel);
}

#ifdef ALL_TESTS
TEST(FEATHERSTONE, LINK_5)
{
  verifySkeleton(createMultiarmRobot(5, 0.2));
}
#endif

/*
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaImplicitToDynamic(
    Eigen::Matrix6s& parentArtInertia,
    const Eigen::Matrix6s& childArtInertia)
{
  // Child body's articulated inertia
  JacobianMatrix AIS = childArtInertia * getRelativeJacobianStatic();
  Eigen::Matrix6s PI = childArtInertia;
  PI.noalias() -= AIS * mInvProjArtInertiaImplicit * AIS.transpose();
  assert(!math::isNan(PI));

  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  parentArtInertia += math::transformInertia(
        this->getRelativeTransform().inverse(), PI);
}
*/

/*
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::addChildArtInertiaToDynamic(
    Eigen::Matrix6s& parentArtInertia,
    const Eigen::Matrix6s& childArtInertia)
{
  // Child body's articulated inertia
  JacobianMatrix AIS = childArtInertia * getRelativeJacobianStatic();
  Eigen::Matrix6s PI = childArtInertia;
  PI.noalias() -= AIS * mInvProjArtInertia * AIS.transpose();
  assert(!math::isNan(PI));

  // Add child body's articulated inertia to parent body's articulated inertia.
  // Note that mT should be updated.
  parentArtInertia += math::transformInertia(
        this->getRelativeTransform().inverse(), PI);
}
*/

/*
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaDynamic(
    const Eigen::Matrix6s& artInertia)
{
  // Projected articulated inertia
  const JacobianMatrix& Jacobian = getRelativeJacobianStatic();
  const Matrix projAI = Jacobian.transpose() * artInertia * Jacobian;

  // Inversion of projected articulated inertia
  mInvProjArtInertia = math::inverse<ConfigSpaceT>(projAI);

  // Verification
  assert(!math::isNan(mInvProjArtInertia));
}
*/

/*
template <class ConfigSpaceT>
void GenericJoint<ConfigSpaceT>::updateInvProjArtInertiaImplicitDynamic(
    const Eigen::Matrix6s& artInertia,
    s_t timeStep)
{
  // Projected articulated inertia
  const JacobianMatrix& Jacobian = getRelativeJacobianStatic();
  Matrix projAI = Jacobian.transpose() * artInertia * Jacobian;

  // Add additional inertia for implicit damping and spring force
  projAI +=
      (timeStep * Base::mAspectProperties.mDampingCoefficients
       + timeStep * timeStep *
Base::mAspectProperties.mSpringStiffnesses).asDiagonal();

  // Inversion of projected articulated inertia
  mInvProjArtInertiaImplicit = math::inverse<ConfigSpaceT>(projAI);

  // Verification
  assert(!math::isNan(mInvProjArtInertiaImplicit));
}
*/