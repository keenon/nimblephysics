#include "dart/neural/BackpropSnapshot.hpp"

#include <iostream>

#include "dart/constraint/ConstraintSolver.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/ConstrainedGroupGradientMatrices.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace math;
using namespace dynamics;
using namespace simulation;

namespace dart {
namespace neural {

//==============================================================================
BackpropSnapshot::BackpropSnapshot(
    WorldPtr world,
    Eigen::VectorXd forwardPassPosition,
    Eigen::VectorXd forwardPassVelocity,
    Eigen::VectorXd forwardPassTorques)
{
  mWorld = world;
  mTimeStep = world->getTimeStep();
  mForwardPassPosition = forwardPassPosition;
  mForwardPassVelocity = forwardPassVelocity;
  mForwardPassTorques = forwardPassTorques;
  mNumDOFs = 0;
  mNumConstraintDim = 0;
  mSkeletons = std::vector<SkeletonPtr>();
  mSkeletons.reserve(world->getNumSkeletons());

  // Collect all the constraint groups attached to each skeleton

  for (std::size_t i = 0; i < world->getNumSkeletons(); i++)
  {
    SkeletonPtr skel = world->getSkeleton(i);
    mSkeletons.push_back(skel);
    mSkeletonOffset.insert(std::make_pair(skel->getName(), mNumDOFs));
    mNumDOFs += skel->getNumDofs();

    std::shared_ptr<ConstrainedGroupGradientMatrices> gradientMatrix
        = skel->getGradientConstraintMatrices();
    if (gradientMatrix
        && std::find(
               mGradientMatrices.begin(),
               mGradientMatrices.end(),
               gradientMatrix)
               == mGradientMatrices.end())
    {
      mGradientMatrices.push_back(gradientMatrix);
      mNumConstraintDim += gradientMatrix->getNumConstraintDim();
    }
  }
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getForwardPassPosition()
{
  return mForwardPassPosition;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getForwardPassVelocity()
{
  return mForwardPassVelocity;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getForwardPassTorques()
{
  return mForwardPassTorques;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getClampingConstraintMatrix()
{
  std::size_t numClamping = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    numClamping += mGradientMatrices[i]->getClampingConstraintMatrix().cols();
  }
  Eigen::MatrixXd clampingConstraintMatrix
      = Eigen::MatrixXd(mNumDOFs, numClamping);
  clampingConstraintMatrix.setZero();
  std::size_t constraintCursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXd groupClampsMatrix
        = mGradientMatrices[i]->getClampingConstraintMatrix();

    // shuffle the clamps into the main matrix
    std::size_t dofCursorGroup = 0;
    for (std::size_t k = 0; k < mGradientMatrices[i]->getSkeletons().size();
         k++)
    {
      SkeletonPtr skel = mGradientMatrices[i]->getSkeletons()[k];
      // This maps to the row in the world matrix
      std::size_t dofCursorWorld
          = mSkeletonOffset.find(skel->getName())->second;

      // The source block in the groupClamps matrix is a row section at
      // (dofCursorGroup, 0) of full width (skel->getNumDOFs(),
      // groupClamps.cols()), which we want to copy into our unified
      // clampingConstraintMatrix.

      // The destination block in clampingConstraintMatrix is the column
      // corresponding to this constraint group's constraint set, and the row
      // corresponding to this skeleton's offset into the world at
      // (dofCursorWorld, constraintCursor).

      clampingConstraintMatrix.block(
          dofCursorWorld,
          constraintCursor,
          skel->getNumDofs(),
          groupClampsMatrix.cols())
          = groupClampsMatrix.block(
              dofCursorGroup, 0, skel->getNumDofs(), groupClampsMatrix.cols());

      dofCursorGroup += skel->getNumDofs();
    }

    constraintCursor += groupClampsMatrix.cols();
  }
  return clampingConstraintMatrix;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getUpperBoundConstraintMatrix()
{
  std::size_t numUpperBound = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    numUpperBound
        += mGradientMatrices[i]->getUpperBoundConstraintMatrix().cols();
  }
  Eigen::MatrixXd clampingConstraintMatrix
      = Eigen::MatrixXd(mNumDOFs, numUpperBound);
  clampingConstraintMatrix.setZero();
  std::size_t constraintCursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXd groupUpperBoundsMatrix
        = mGradientMatrices[i]->getUpperBoundConstraintMatrix();
    if (groupUpperBoundsMatrix.cols() == 0)
      continue;

    // shuffle the clamps into the main matrix
    std::size_t dofCursorGroup = 0;
    for (std::size_t k = 0; k < mGradientMatrices[i]->getSkeletons().size();
         k++)
    {
      SkeletonPtr skel = mGradientMatrices[i]->getSkeletons()[k];
      std::size_t dofCursorWorld
          = mSkeletonOffset.find(skel->getName())->second;

      // The source block in the groupClamps matrix is a row section at
      // (dofCursorGroup, 0) of full width (skel->getNumDOFs(),
      // groupClamps.cols()), which we want to copy into our unified
      // clampingConstraintMatrix.

      // The destination block in clampingConstraintMatrix is the column
      // corresponding to this constraint group's constraint set, and the row
      // corresponding to this skeleton's offset into the world at
      // (dofCursorWorld, constraintCursor).

      clampingConstraintMatrix.block(
          dofCursorWorld,
          constraintCursor,
          skel->getNumDofs(),
          groupUpperBoundsMatrix.cols())
          = groupUpperBoundsMatrix.block(
              dofCursorGroup,
              0,
              skel->getNumDofs(),
              groupUpperBoundsMatrix.cols());

      dofCursorGroup += skel->getNumDofs();
    }

    constraintCursor += groupUpperBoundsMatrix.cols();
  }
  return clampingConstraintMatrix;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getUpperBoundMappingMatrix()
{
  std::size_t numUpperBound = 0;
  std::size_t numClamping = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    numUpperBound
        += mGradientMatrices[i]->getUpperBoundConstraintMatrix().cols();
    numClamping += mGradientMatrices[i]->getClampingConstraintMatrix().cols();
  }

  Eigen::MatrixXd mappingMatrix = Eigen::MatrixXd(numUpperBound, numClamping);
  mappingMatrix.setZero();

  std::size_t cursorUpperBound = 0;
  std::size_t cursorClamping = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXd groupMappingMatrix
        = mGradientMatrices[i]->getUpperBoundMappingMatrix();
    mappingMatrix.block(
        cursorUpperBound,
        cursorClamping,
        groupMappingMatrix.rows(),
        groupMappingMatrix.cols())
        = groupMappingMatrix;

    cursorUpperBound += groupMappingMatrix.rows();
    cursorClamping += groupMappingMatrix.cols();
  }

  return mappingMatrix;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getMassMatrix()
{
  Eigen::MatrixXd massMatrix = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  massMatrix.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t skelDOF = mSkeletons[i]->getNumDofs();
    massMatrix.block(cursor, cursor, skelDOF, skelDOF)
        = mSkeletons[i]->getMassMatrix();
    cursor += skelDOF;
  }
  return massMatrix;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getInvMassMatrix()
{
  Eigen::MatrixXd invMassMatrix = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  invMassMatrix.setZero();
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mSkeletons.size(); i++)
  {
    std::size_t skelDOF = mSkeletons[i]->getNumDofs();
    invMassMatrix.block(cursor, cursor, skelDOF, skelDOF)
        = mSkeletons[i]->getInvMassMatrix();
    cursor += skelDOF;
  }
  return invMassMatrix;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getContactConstraintImpluses()
{
  if (mGradientMatrices.size() == 1)
    return mGradientMatrices[0]->getContactConstraintImpluses();
  Eigen::VectorXd mX = Eigen::VectorXd(mNumConstraintDim);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    mX.segment(cursor, mGradientMatrices[i]->getNumConstraintDim())
        = mGradientMatrices[i]->getContactConstraintImpluses();
    cursor += mGradientMatrices[i]->getNumConstraintDim();
  }
  return mX;
}

//==============================================================================
Eigen::VectorXi BackpropSnapshot::getContactConstraintMappings()
{
  if (mGradientMatrices.size() == 1)
    return mGradientMatrices[0]->getContactConstraintMappings();
  Eigen::VectorXi fIndex = Eigen::VectorXi(mNumConstraintDim);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    fIndex.segment(cursor, mGradientMatrices[i]->getNumConstraintDim())
        = mGradientMatrices[i]->getContactConstraintMappings();
    cursor += mGradientMatrices[i]->getNumConstraintDim();
  }
  return fIndex;
}

/*
MatrixXd finiteDifferenceVelVelJacobian(
    SkeletonPtr skel, WorldPtr world, VectorXd velocities)
{
  FullSnapshot snapshot(world);

  MatrixXd J(skel->getNumDofs(), skel->getNumDofs());

  skel->setVelocities(velocities);
  world->step(false);

  VectorXd originalVel = skel->getVelocities();

  double EPSILON = 1e-7;
  for (auto i = 0; i < skel->getNumDofs(); i++)
  {
    snapshot.restore();

    VectorXd tweakedVel = VectorXd(velocities);
    tweakedVel(i) += EPSILON;
    skel->setVelocities(tweakedVel);
    world->step(false);

    VectorXd velChange = (skel->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();

  return J;
}
*/

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceVelVelJacobian()
{
  RestorableSnapshot snapshot(mWorld);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  mWorld->setVelocities(mForwardPassVelocity);
  mWorld->step(false);

  Eigen::VectorXd originalVel = mWorld->getVelocities();

  double EPSILON = 1e-7;
  for (std::size_t i = 0; i < mWorld->getNumDofs(); i++)
  {
    snapshot.restore();

    Eigen::VectorXd tweakedVel = Eigen::VectorXd(mForwardPassVelocity);
    tweakedVel(i) += EPSILON;
    mWorld->setVelocities(tweakedVel);
    mWorld->step(false);

    Eigen::VectorXd velChange
        = (mWorld->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceForceVelJacobian()
{
  RestorableSnapshot snapshot(mWorld);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  mWorld->setVelocities(mForwardPassVelocity);
  mWorld->step(false);

  Eigen::VectorXd originalForces = mWorld->getForces();
  Eigen::VectorXd originalVel = mWorld->getVelocities();

  double EPSILON = 1e-7;
  for (auto i = 0; i < mWorld->getNumDofs(); i++)
  {
    snapshot.restore();

    mWorld->setVelocities(mForwardPassVelocity);
    Eigen::VectorXd tweakedForces = Eigen::VectorXd(originalForces);
    tweakedForces(i) += EPSILON;
    mWorld->setForces(tweakedForces);

    mWorld->step(false);

    Eigen::VectorXd velChange
        = (mWorld->getVelocities() - originalVel) / EPSILON;
    J.col(i).noalias() = velChange;
  }

  snapshot.restore();

  return J;
}

//==============================================================================
BackpropSnapshot::~BackpropSnapshot()
{
}

} // namespace neural
} // namespace dart