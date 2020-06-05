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
  mNumClamping = 0;
  mNumUpperBound = 0;
  mNumBouncing = 0;
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
      mNumClamping += gradientMatrix->getClampingConstraintMatrix().cols();
      mNumUpperBound += gradientMatrix->getUpperBoundConstraintMatrix().cols();
      mNumBouncing += gradientMatrix->getBouncingConstraintMatrix().cols();
    }
  }
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getForceVelJacobian()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix();

  if (A_ub.size() > 0 && E.size() > 0)
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * (A_c + A_ub * E) * P_c * Minv);
  }
  else
  {
    return mTimeStep * Minv
           * (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs)
              - mTimeStep * A_c * P_c * Minv);
  }
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getVelVelJacobian()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd A_ub = getUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();
  Eigen::MatrixXd P_c = getProjectionIntoClampsMatrix();
  Eigen::MatrixXd Minv = getInvMassMatrix();
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(
      mNumDOFs, mNumDOFs); // TODO(keenon): B needs to be set properly.
  Eigen::MatrixXd parts1 = A_c + A_ub * E;
  Eigen::MatrixXd parts2 = mTimeStep * Minv * parts1 * P_c;
  /*
  std::cout << "A_c: " << std::endl << A_c << std::endl;
  std::cout << "A_ub: " << std::endl << A_ub << std::endl;
  std::cout << "E: " << std::endl << E << std::endl;
  std::cout << "P_c: " << std::endl << P_c << std::endl;
  std::cout << "Minv: " << std::endl << Minv << std::endl;
  std::cout << "mTimestep: " << mTimeStep << std::endl;
  std::cout << "A_c + A_ub * E: " << std::endl << parts1 << std::endl;
  std::cout << "mTimestep * Minv * (A_c + A_ub * E) * P_c: " << std::endl
            << parts2 << std::endl;
            */
  return (Eigen::MatrixXd::Identity(mNumDOFs, mNumDOFs) - parts2) * B;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getPosPosJacobian()
{
  Eigen::MatrixXd A_b = getBouncingConstraintMatrix();

  // Construct the W matrix we'll need to use to solve for our closest approx
  Eigen::MatrixXd W = Eigen::MatrixXd(A_b.cols(), A_b.rows() * A_b.rows());
  for (std::size_t i = 0; i < A_b.cols(); i++)
  {
    Eigen::VectorXd a_i = A_b.col(i);
    for (std::size_t j = 0; j < A_b.rows(); j++)
    {
      W.block(j * A_b.rows(), i, A_b.rows(), 1) = a_i(j) * a_i;
    }
  }

  // We want to center the solution around the identity matrix, and find the
  // least-squares deviation along the diagonals that gets us there.
  Eigen::VectorXd center = Eigen::VectorXd::Zero(mNumDOFs * mNumDOFs);
  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    center((i * mNumDOFs) + i) = 1;
  }

  // Solve the linear system
  Eigen::VectorXd q
      = center
        - W.transpose().completeOrthogonalDecomposition().solve(
            getRestitutionDiagonals() + (W.eval().transpose() * center));

  // Recover X from the q vector
  Eigen::MatrixXd X = Eigen::MatrixXd(mNumDOFs, mNumDOFs);
  for (std::size_t i = 0; i < mNumDOFs; i++)
  {
    X.col(i) = q.segment(i * mNumDOFs, mNumDOFs);
  }

  return X;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getVelPosJacobian()
{
  return mTimeStep * getPosPosJacobian();
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
  return assembleMatrix(MatrixToAssemble::CLAMPING);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getMassedClampingConstraintMatrix()
{
  return assembleMatrix(MatrixToAssemble::MASSED_CLAMPING);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getUpperBoundConstraintMatrix()
{
  return assembleMatrix(MatrixToAssemble::UPPER_BOUND);
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getMassedUpperBoundConstraintMatrix()
{
  return assembleMatrix(MatrixToAssemble::MASSED_UPPER_BOUND);
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
Eigen::MatrixXd BackpropSnapshot::getBouncingConstraintMatrix()
{
  return assembleMatrix(MatrixToAssemble::BOUNCING);
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

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getBounceDiagonals()
{
  if (mGradientMatrices.size() == 1)
    return mGradientMatrices[0]->getBounceDiagonals();
  Eigen::VectorXd restitutionCoeffs = Eigen::VectorXd(mNumClamping);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    restitutionCoeffs.segment(
        cursor, mGradientMatrices[i]->getNumConstraintDim())
        = mGradientMatrices[i]->getBounceDiagonals();
    cursor += mGradientMatrices[i]->getNumConstraintDim();
  }
  return restitutionCoeffs;
}

//==============================================================================
Eigen::VectorXd BackpropSnapshot::getRestitutionDiagonals()
{
  if (mGradientMatrices.size() == 1)
    return mGradientMatrices[0]->getRestitutionDiagonals();
  Eigen::VectorXd contactDistances = Eigen::VectorXd(mNumBouncing);
  std::size_t cursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    contactDistances.segment(
        cursor, mGradientMatrices[i]->getNumConstraintDim())
        = mGradientMatrices[i]->getRestitutionDiagonals();
    cursor += mGradientMatrices[i]->getNumConstraintDim();
  }
  return contactDistances;
}

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
Eigen::MatrixXd BackpropSnapshot::finiteDifferencePosPosJacobian(
    std::size_t subdivisions)
{
  RestorableSnapshot snapshot(mWorld);

  double oldTimestep = mWorld->getTimeStep();
  mWorld->setTimeStep(oldTimestep / subdivisions);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  mWorld->setPositions(mForwardPassPosition);
  mWorld->setVelocities(mForwardPassVelocity);
  mWorld->setForces(mForwardPassTorques);

  for (std::size_t j = 0; j < subdivisions; j++)
    mWorld->step(false);

  Eigen::VectorXd originalPosition = mWorld->getPositions();

  // IMPORTANT: EPSILON must be larger than the distance traveled in a single
  // subdivided timestep. Ideally much larger.
  double EPSILON = 1e-1 / subdivisions;
  for (auto i = 0; i < mWorld->getNumDofs(); i++)
  {
    snapshot.restore();

    mWorld->setVelocities(mForwardPassVelocity);
    mWorld->setForces(mForwardPassTorques);

    Eigen::VectorXd tweakedPositions = Eigen::VectorXd(mForwardPassPosition);
    tweakedPositions(i) += EPSILON;
    mWorld->setPositions(tweakedPositions);

    for (std::size_t j = 0; j < subdivisions; j++)
      mWorld->step(false);

    Eigen::VectorXd posChange
        = (mWorld->getPositions() - originalPosition) / EPSILON;
    J.col(i).noalias() = posChange;
  }

  mWorld->setTimeStep(oldTimestep);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::finiteDifferenceVelPosJacobian(
    std::size_t subdivisions)
{
  RestorableSnapshot snapshot(mWorld);

  double oldTimestep = mWorld->getTimeStep();
  mWorld->setTimeStep(oldTimestep / subdivisions);

  Eigen::MatrixXd J(mNumDOFs, mNumDOFs);

  mWorld->setPositions(mForwardPassPosition);
  mWorld->setVelocities(mForwardPassVelocity);
  mWorld->setForces(mForwardPassTorques);

  for (std::size_t j = 0; j < subdivisions; j++)
    mWorld->step(false);

  Eigen::VectorXd originalPosition = mWorld->getPositions();

  double EPSILON = 1e-3 / subdivisions;
  for (auto i = 0; i < mWorld->getNumDofs(); i++)
  {
    snapshot.restore();

    mWorld->setPositions(mForwardPassPosition);
    mWorld->setForces(mForwardPassTorques);

    Eigen::VectorXd tweakedVelocity = Eigen::VectorXd(mForwardPassVelocity);
    tweakedVelocity(i) += EPSILON;
    mWorld->setVelocities(tweakedVelocity);

    for (std::size_t j = 0; j < subdivisions; j++)
      mWorld->step(false);

    Eigen::VectorXd posChange
        = (mWorld->getPositions() - originalPosition) / EPSILON;
    J.col(i).noalias() = posChange;
  }

  mWorld->setTimeStep(oldTimestep);
  snapshot.restore();

  return J;
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::getProjectionIntoClampsMatrix()
{
  Eigen::MatrixXd A_c = getClampingConstraintMatrix();
  Eigen::MatrixXd V_c = getMassedClampingConstraintMatrix();
  Eigen::MatrixXd V_ub = getMassedUpperBoundConstraintMatrix();
  Eigen::MatrixXd E = getUpperBoundMappingMatrix();

  /*
  std::cout << "A_c: " << std::endl << A_c << std::endl;
  std::cout << "A_ub: " << std::endl << A_ub << std::endl;
  std::cout << "E: " << std::endl << E << std::endl;
  */

  Eigen::MatrixXd constraintForceToImpliedTorques = V_c + (V_ub * E);
  Eigen::MatrixXd forceToVel
      = A_c.eval().transpose() * constraintForceToImpliedTorques;
  Eigen::MatrixXd velToForce
      = forceToVel.completeOrthogonalDecomposition().pseudoInverse();
  Eigen::MatrixXd bounce = getBounceDiagonals().asDiagonal();
  /*
  std::cout << "forceToVel: " << std::endl << forceToVel << std::endl;
  std::cout << "forceToVel^-1: " << std::endl << velToForce << std::endl;
  std::cout << "mTimeStep: " << mTimeStep << std::endl;
  */
  return (1.0 / mTimeStep) * velToForce * bounce * A_c.transpose();
}

//==============================================================================
BackpropSnapshot::~BackpropSnapshot()
{
}

//==============================================================================
Eigen::MatrixXd BackpropSnapshot::assembleMatrix(MatrixToAssemble whichMatrix)
{
  std::size_t numCols = 0;
  if (whichMatrix == MatrixToAssemble::CLAMPING
      || whichMatrix == MatrixToAssemble::MASSED_CLAMPING)
    numCols = mNumClamping;
  else if (
      whichMatrix == MatrixToAssemble::UPPER_BOUND
      || whichMatrix == MatrixToAssemble::MASSED_UPPER_BOUND)
    numCols = mNumUpperBound;
  else if (whichMatrix == MatrixToAssemble::BOUNCING)
    numCols = mNumBouncing;

  Eigen::MatrixXd matrix = Eigen::MatrixXd(mNumDOFs, numCols);
  matrix.setZero();
  std::size_t constraintCursor = 0;
  for (std::size_t i = 0; i < mGradientMatrices.size(); i++)
  {
    Eigen::MatrixXd groupMatrix;

    if (whichMatrix == MatrixToAssemble::CLAMPING)
      groupMatrix = mGradientMatrices[i]->getClampingConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::MASSED_CLAMPING)
      groupMatrix = mGradientMatrices[i]->getMassedClampingConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::UPPER_BOUND)
      groupMatrix = mGradientMatrices[i]->getUpperBoundConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::MASSED_UPPER_BOUND)
      groupMatrix = mGradientMatrices[i]->getMassedUpperBoundConstraintMatrix();
    else if (whichMatrix == MatrixToAssemble::BOUNCING)
      groupMatrix = mGradientMatrices[i]->getBouncingConstraintMatrix();

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

      matrix.block(
          dofCursorWorld,
          constraintCursor,
          skel->getNumDofs(),
          groupMatrix.cols())
          = groupMatrix.block(
              dofCursorGroup, 0, skel->getNumDofs(), groupMatrix.cols());

      dofCursorGroup += skel->getNumDofs();
    }

    constraintCursor += groupMatrix.cols();
  }
  return matrix;
}

} // namespace neural
} // namespace dart