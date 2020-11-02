#include "dart/trajectory/MultiShot.hpp"

#include <future>
#include <vector>

#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/BackpropSnapshot.hpp"
#include "dart/neural/NeuralUtils.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;
using namespace dynamics;
using namespace simulation;
using namespace neural;

#define LOG_PERFORMANCE_MULTI_SHOT

namespace dart {
namespace trajectory {

//==============================================================================
MultiShot::MultiShot(
    std::shared_ptr<simulation::World> world,
    LossFn loss,
    int steps,
    int shotLength,
    bool tuneStartingState)
  : AbstractShot(world, loss, steps), mParallelOperationsEnabled(false)
{
  mShotLength = shotLength;
  mTuneStartingState = tuneStartingState;

  int stepsRemaining = steps;
  bool isFirst = true;
  LossFn zeroLoss = LossFn();
  while (stepsRemaining > 0)
  {
    int shot = std::min(shotLength, stepsRemaining);
    mShots.push_back(std::make_shared<SingleShot>(
        world, zeroLoss, shot, !isFirst || tuneStartingState));
    stepsRemaining -= shot;
    isFirst = false;
  }
}

//==============================================================================
MultiShot::~MultiShot()
{
  // std::cout << "Freeing MultiShot: " << this << std::endl;
}

//==============================================================================
void MultiShot::setParallelOperationsEnabled(bool enabled)
{
  mParallelOperationsEnabled = enabled;
  if (enabled)
  {
    // Before using Eigen in a multi-threaded environment, we need to explicitly
    // call this (at least prior to Eigen 3.3)
    Eigen::initParallel();

    mParallelWorlds.clear();
    for (int i = 0; i < mShots.size(); i++)
    {
      mParallelWorlds.push_back(mWorld->clone());
    }
  }
}

//==============================================================================
/// This sets the mapping we're using to store the representation of the Shot.
/// WARNING: THIS IS A POTENTIALLY DESTRUCTIVE OPERATION! This will rewrite
/// the internal representation of the Shot to use the new mapping, and if the
/// new mapping is underspecified compared to the old mapping, you may lose
/// information. It's not guaranteed that you'll get back the same trajectory
/// if you switch to a different mapping, and then switch back.
///
/// This will affect the values you get back from getStates() - they'll now be
/// returned in the view given by `mapping`. That's also the represenation
/// that'll be passed to IPOPT, and updated on each gradient step. Therein
/// lies the power of changing the representation mapping: There will almost
/// certainly be mapped spaces that are easier to optimize in than native
/// joint space, at least initially.
void MultiShot::switchRepresentationMapping(
    std::shared_ptr<simulation::World> world,
    const std::string& mapping,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.switchRepresentationMapping");
  }
#endif

  for (auto shot : mShots)
  {
    shot->switchRepresentationMapping(world, mapping, thisLog);
  }
  AbstractShot::switchRepresentationMapping(world, mapping, thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This adds a mapping through which the loss function can interpret the
/// output. We can have multiple loss mappings at the same time, and loss can
/// use arbitrary combinations of multiple views, as long as it can provide
/// gradients.
void MultiShot::addMapping(
    const std::string& key, std::shared_ptr<neural::Mapping> mapping)
{
  AbstractShot::addMapping(key, mapping);
  for (const std::shared_ptr<SingleShot> shot : mShots)
  {
    shot->addMapping(key, mapping);
  }
}

//==============================================================================
/// This removes the loss mapping at a particular key
void MultiShot::removeMapping(const std::string& key)
{
  AbstractShot::removeMapping(key);
  for (const std::shared_ptr<SingleShot> shot : mShots)
  {
    shot->removeMapping(key);
  }
}

//==============================================================================
/// Returns the length of the flattened problem state
int MultiShot::getFlatDynamicProblemDim(
    std::shared_ptr<simulation::World> world) const
{
  int sum = 0;
  for (const std::shared_ptr<SingleShot> shot : mShots)
  {
    sum += shot->getFlatDynamicProblemDim(world);
  }
  return sum;
}

//==============================================================================
/// Returns the length of the knot-point constraint vector
int MultiShot::getConstraintDim() const
{
  return AbstractShot::getConstraintDim()
         + getRepresentationStateSize() * (mShots.size() - 1);
}

//==============================================================================
/// This computes the values of the constraints
void MultiShot::computeConstraints(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.computeConstraints");
  }
#endif

  int cursor = 0;
  int numParentConstraints = AbstractShot::getConstraintDim();
  AbstractShot::computeConstraints(
      world, constraints.segment(0, numParentConstraints), thisLog);
  cursor += numParentConstraints;

  if (mParallelOperationsEnabled)
  {
    std::vector<std::future<void>> futures;
    for (int i = 1; i < mShots.size(); i++)
    {
      futures.push_back(std::async(
          &MultiShot::asyncPartComputeConstraints,
          this,
          i,
          mParallelWorlds[i],
          constraints,
          cursor,
          thisLog));
      cursor += getRepresentationStateSize();
    }
    for (int i = 0; i < futures.size(); i++)
    {
      futures[i].wait();
    }
  }
  else
  {
    for (int i = 1; i < mShots.size(); i++)
    {
      constraints.segment(cursor, getRepresentationStateSize())
          = mShots[i - 1]->getFinalState(world, thisLog)
            - mShots[i]->getStartState();
      cursor += getRepresentationStateSize();
    }
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This is a single async call for computing constraints in parallel, if
/// we're using multithreading
void MultiShot::asyncPartComputeConstraints(
    int index,
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> constraints,
    int cursor,
    PerformanceLog* log)
{
  constraints.segment(cursor, getRepresentationStateSize())
      = mShots[index - 1]->getFinalState(world, log)
        - mShots[index]->getStartState();
}

//==============================================================================
/// This copies a shot down into a single flat vector
void MultiShot::flatten(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.flatten");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatDynamicProblemDim(world);
    shot->flatten(world, flatStatic, flatDynamic.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the parameters out of a flat vector
void MultiShot::unflatten(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::VectorXd>& flatStatic,
    const Eigen::Ref<const Eigen::VectorXd>& flatDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.unflatten");
  }
#endif

  // Set any static values on the main world that's been passed in
  int abstractNumDynamic = AbstractShot::getFlatDynamicProblemDim(world);
  int abstractNumStatic = AbstractShot::getFlatStaticProblemDim(world);
  AbstractShot::unflatten(
      world,
      flatStatic.segment(0, abstractNumStatic),
      flatDynamic.segment(0, abstractNumDynamic),
      thisLog);

  // Now set the values for all the parallel world objects
  mRolloutCacheDirty = true;
  int cursor = 0;
  for (int i = 0; i < mShots.size(); i++)
  {
    std::shared_ptr<SingleShot>& shot = mShots[i];
    int dim = shot->getFlatDynamicProblemDim(world);
    shot->unflatten(
        mParallelOperationsEnabled ? mParallelWorlds[i] : world,
        flatStatic,
        flatDynamic.segment(cursor, dim),
        thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void MultiShot::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getUpperBounds");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatDynamicProblemDim(world);
    shot->getUpperBounds(
        world, flatStatic, flatDynamic.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void MultiShot::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getLowerBounds");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatDynamicProblemDim(world);
    shot->getLowerBounds(
        world, flatStatic, flatDynamic.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void MultiShot::getConstraintUpperBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getConstraintUpperBounds");
  }
#endif

  flat.setZero();
  AbstractShot::getConstraintUpperBounds(
      flat.segment(0, AbstractShot::getConstraintDim()), thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void MultiShot::getConstraintLowerBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getConstraintLowerBounds");
  }
#endif

  flat.setZero();
  AbstractShot::getConstraintLowerBounds(
      flat.segment(0, AbstractShot::getConstraintDim()), thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void MultiShot::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> flatDynamic,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getInitialGuess");
  }
#endif

  int cursor = 0;
  for (const std::shared_ptr<SingleShot>& shot : mShots)
  {
    int dim = shot->getFlatDynamicProblemDim(world);
    shot->getInitialGuess(
        world, flatStatic, flatDynamic.segment(cursor, dim), thisLog);
    cursor += dim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the
/// constraints. This returns a matrix that's (getConstraintDim(),
/// getFlatProblemDim()).
void MultiShot::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacStatic,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.backpropJacobian");
  }
#endif

  assert(jacStatic.cols() == getFlatStaticProblemDim(world));
  assert(jacStatic.rows() == getConstraintDim());
  assert(jacDynamic.cols() == getFlatDynamicProblemDim(world));
  assert(jacDynamic.rows() == getConstraintDim());

  int rowCursor = 0;
  int colCursor = 0;

  jacStatic.setZero();
  jacDynamic.setZero();

  // Handle custom constraints
  int numParentConstraints = AbstractShot::getConstraintDim();
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  AbstractShot::backpropJacobian(
      world,
      jacStatic.block(0, 0, numParentConstraints, staticDim),
      jacDynamic.block(0, 0, numParentConstraints, dynamicDim),
      thisLog);
  rowCursor += numParentConstraints;

  // Add in knot point constraints
  int stateDim = getRepresentationStateSize();
  if (mParallelOperationsEnabled)
  {
    std::vector<std::future<void>> futures;
    for (int i = 1; i < mShots.size(); i++)
    {
      int dynamicDim = mShots[i - 1]->getFlatDynamicProblemDim(world);
      futures.push_back(std::async(
          &MultiShot::asyncPartBackpropJacobian,
          this,
          i,
          mParallelWorlds[i],
          jacStatic,
          jacDynamic,
          rowCursor,
          colCursor,
          thisLog));
      colCursor += dynamicDim;
      rowCursor += stateDim;
    }
  }
  else
  {
    // We need a copy of the jacStatic matrix, or else every SingleShot will
    // clear it and redo the sum over time. Instead, we give every SingleShot a
    // scratch matrix to use, which it can safely zero, and then sum the
    // results.
    for (int i = 1; i < mShots.size(); i++)
    {
      int dim = mShots[i - 1]->getFlatDynamicProblemDim(world);
      mShots[i - 1]->backpropJacobianOfFinalState(
          world,
          jacStatic.block(rowCursor, 0, stateDim, staticDim),
          jacDynamic.block(rowCursor, colCursor, stateDim, dim),
          thisLog);
      colCursor += dim;
      jacDynamic.block(rowCursor, colCursor, stateDim, stateDim)
          = -1 * Eigen::MatrixXd::Identity(stateDim, stateDim);
      rowCursor += stateDim;
    }
  }

  // We don't include the last shot in the constraints, cause it doesn't end in
  // a knot point
  assert(
      colCursor
      == jacDynamic.cols()
             - mShots[mShots.size() - 1]->getFlatDynamicProblemDim(world));
  assert(rowCursor == jacDynamic.rows());

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This is a single async call for computing constraints in parallel, if
/// we're using multithreading
void MultiShot::asyncPartBackpropJacobian(
    int index,
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacStatic,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> jacDynamic,
    int rowCursor,
    int colCursor,
    PerformanceLog* log)
{
  int stateDim = getRepresentationStateSize();
  int dynamicDim = mShots[index - 1]->getFlatDynamicProblemDim(world);
  mShots[index - 1]->backpropJacobianOfFinalState(
      world,
      jacStatic.block(rowCursor, 0, stateDim, jacStatic.cols()),
      jacDynamic.block(rowCursor, colCursor, stateDim, dynamicDim),
      log);
  colCursor += dynamicDim;
  jacDynamic.block(rowCursor, colCursor, stateDim, stateDim)
      = -1 * Eigen::MatrixXd::Identity(stateDim, stateDim);
  rowCursor += stateDim;
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int MultiShot::getNumberNonZeroJacobianStatic(
    std::shared_ptr<simulation::World> world)
{
  int nnzj = AbstractShot::getNumberNonZeroJacobianStatic(world);

  int stateDim = getRepresentationStateSize();
  int staticDim = getFlatStaticProblemDim(world);
  nnzj += staticDim * (stateDim * (mShots.size() - 1));

  return nnzj;
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int MultiShot::getNumberNonZeroJacobianDynamic(
    std::shared_ptr<simulation::World> world)
{
  int nnzj = AbstractShot::getNumberNonZeroJacobianDynamic(world);

  int stateDim = getRepresentationStateSize();
  int staticDim = getFlatStaticProblemDim(world);

  for (int i = 0; i < mShots.size() - 1; i++)
  {
    int shotDim = mShots[i]->getFlatDynamicProblemDim(world);
    // The main Jacobian block
    nnzj += shotDim * stateDim;
    // The -I at the end
    nnzj += stateDim;
  }

  return nnzj;
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void MultiShot::getJacobianSparsityStructureStatic(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getJacobianSparsityStructure");
  }
#endif

  int sparseCursor = 0;
  int rowCursor = 0;
  int colCursor = 0;
  int stateDim = getRepresentationStateSize();

  // Handle custom constraints
  int abstractNnzj = AbstractShot::getNumberNonZeroJacobianStatic(world);
  AbstractShot::getJacobianSparsityStructureStatic(
      world,
      rows.segment(0, abstractNnzj),
      cols.segment(0, abstractNnzj),
      thisLog);
  rowCursor += AbstractShot::getConstraintDim();
  sparseCursor += abstractNnzj;

  // Handle the static data, in row-major order
  int staticDim = getFlatStaticProblemDim(world);
  for (int row = rowCursor; row < rowCursor + (stateDim * (mShots.size() - 1));
       row++)
  {
    for (int col = colCursor; col < colCursor + staticDim; col++)
    {
      rows(sparseCursor) = row;
      cols(sparseCursor) = col;
      sparseCursor++;
    }
  }
  colCursor += staticDim;
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void MultiShot::getJacobianSparsityStructureDynamic(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getJacobianSparsityStructure");
  }
#endif

  int sparseCursor = 0;
  int rowCursor = 0;
  int colCursor = 0;

  int stateDim = getRepresentationStateSize();

  // Handle custom constraints
  int abstractNnzj = AbstractShot::getNumberNonZeroJacobianDynamic(world);
  AbstractShot::getJacobianSparsityStructureDynamic(
      world,
      rows.segment(0, abstractNnzj),
      cols.segment(0, abstractNnzj),
      thisLog);
  rowCursor += AbstractShot::getConstraintDim();
  sparseCursor += abstractNnzj;

  // Handle knot point constraints
  for (int i = 1; i < mShots.size(); i++)
  {
    int dim = mShots[i - 1]->getFlatDynamicProblemDim(world);
    // This is the main Jacobian
    for (int col = colCursor; col < colCursor + dim; col++)
    {
      for (int row = rowCursor; row < rowCursor + stateDim; row++)
      {
        rows(sparseCursor) = row;
        cols(sparseCursor) = col;
        sparseCursor++;
      }
    }
    colCursor += dim;
    // This is the negative identity at the end
    for (int q = 0; q < stateDim; q++)
    {
      rows(sparseCursor) = rowCursor + q;
      cols(sparseCursor) = colCursor + q;
      sparseCursor++;
    }
    rowCursor += stateDim;
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This writes the Jacobian to a sparse vector
void MultiShot::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> sparseStatic,
    Eigen::Ref<Eigen::VectorXd> sparseDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getSparseJacobian");
  }
#endif

  int cursorStatic = AbstractShot::getNumberNonZeroJacobianStatic(world);
  int cursorDynamic = AbstractShot::getNumberNonZeroJacobianDynamic(world);

  AbstractShot::getSparseJacobian(
      world,
      sparseStatic.segment(0, cursorStatic),
      sparseDynamic.segment(0, cursorDynamic),
      thisLog);

  int stateDim = getRepresentationStateSize();

  if (mParallelOperationsEnabled)
  {
    std::vector<std::future<void>> futures;
    for (int i = 1; i < mShots.size(); i++)
    {
      int dimStatic = mShots[i - 1]->getFlatStaticProblemDim(world);
      int dimDynamic = mShots[i - 1]->getFlatDynamicProblemDim(world);

      futures.push_back(std::async(
          &MultiShot::asyncPartGetSparseJacobian,
          this,
          i,
          mParallelWorlds[i],
          sparseStatic,
          sparseDynamic,
          cursorStatic,
          cursorDynamic,
          thisLog));

      cursorDynamic += (dimDynamic + 1) * stateDim;
      cursorStatic += dimStatic * stateDim;
    }
    for (int i = 0; i < futures.size(); i++)
    {
      futures[i].wait();
    }
  }
  else
  {
    Eigen::VectorXd neg = Eigen::VectorXd::Ones(stateDim) * -1;
    for (int i = 1; i < mShots.size(); i++)
    {
      int dimStatic = mShots[i - 1]->getFlatStaticProblemDim(world);
      int dimDynamic = mShots[i - 1]->getFlatDynamicProblemDim(world);

      // Get the dense Jacobians for static and dynamic regions
      Eigen::MatrixXd jacStatic = Eigen::MatrixXd::Zero(stateDim, dimStatic);
      Eigen::MatrixXd jacDynamic = Eigen::MatrixXd::Zero(stateDim, dimDynamic);
      mShots[i - 1]->backpropJacobianOfFinalState(
          world, jacStatic, jacDynamic, thisLog);

      // Copy over the static Jacobian to the global static region (this will
      // overwrite the same region a bunch of times)

      for (int row = 0; row < stateDim; row++)
      {
        sparseStatic.segment(cursorStatic, dimStatic) = jacStatic.row(row);
        cursorStatic += dimStatic;
      }

      // Copy over the dynamic Jacobian to a unique spot in the dynamic region

      for (int col = 0; col < dimDynamic; col++)
      {
        sparseDynamic.segment(cursorDynamic, stateDim) = jacDynamic.col(col);
        cursorDynamic += stateDim;
      }
      // This is the negative identity at the end of our segment in the dynamic
      // region
      sparseDynamic.segment(cursorDynamic, stateDim) = neg;

      cursorDynamic += stateDim;
    }
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This writes the Jacobian to a sparse vector
void MultiShot::asyncPartGetSparseJacobian(
    int index,
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> sparseStatic,
    Eigen::Ref<Eigen::VectorXd> sparseDynamic,
    int cursorStatic,
    int cursorDynamic,
    PerformanceLog* log)
{
  int stateDim = getRepresentationStateSize();

  int dimStatic = mShots[index - 1]->getFlatStaticProblemDim(world);
  int dimDynamic = mShots[index - 1]->getFlatDynamicProblemDim(world);

  // Get the dense Jacobians for static and dynamic regions
  Eigen::MatrixXd jacStatic = Eigen::MatrixXd::Zero(stateDim, dimStatic);
  Eigen::MatrixXd jacDynamic = Eigen::MatrixXd::Zero(stateDim, dimDynamic);
  mShots[index - 1]->backpropJacobianOfFinalState(
      world, jacStatic, jacDynamic, log);

  // Copy over the static Jacobian

  for (int row = 0; row < stateDim; row++)
  {
    sparseStatic.segment(cursorStatic, dimStatic) = jacStatic.row(row);
    cursorStatic += dimStatic;
  }

  // Copy over the dynamic Jacobian

  for (int col = 0; col < dimDynamic; col++)
  {
    sparseDynamic.segment(cursorDynamic, stateDim) = jacDynamic.col(col);
    cursorDynamic += stateDim;
  }
  // This is the negative identity at the end
  Eigen::VectorXd neg = Eigen::VectorXd::Ones(stateDim) * -1;
  sparseDynamic.segment(cursorDynamic, stateDim) = neg;
  cursorDynamic += stateDim;
}

//==============================================================================
/// This returns the snapshots from a fresh unroll
std::vector<neural::MappedBackpropSnapshotPtr> MultiShot::getSnapshots(
    std::shared_ptr<simulation::World> world, PerformanceLog* log)
{
  std::vector<neural::MappedBackpropSnapshotPtr> vec;
  for (std::shared_ptr<SingleShot> shot : mShots)
  {
    for (neural::MappedBackpropSnapshotPtr ptr : shot->getSnapshots(world, log))
    {
      vec.push_back(ptr);
    }
  }
  return vec;
}

//==============================================================================
/// This populates the passed in matrices with the values from this trajectory
void MultiShot::getStates(
    std::shared_ptr<simulation::World> world,
    /* OUT */ TrajectoryRollout* rollout,
    PerformanceLog* log,
    bool useKnots)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getStates");
  }
#endif

  int posDim = getRepresentation()->getPosDim();
  int velDim = getRepresentation()->getVelDim();
  int forceDim = getRepresentation()->getForceDim();
  assert(rollout->getPoses(mRepresentationMapping).cols() == mSteps);
  assert(rollout->getPoses(mRepresentationMapping).rows() == posDim);
  assert(rollout->getVels(mRepresentationMapping).cols() == mSteps);
  assert(rollout->getVels(mRepresentationMapping).rows() == velDim);
  assert(rollout->getForces(mRepresentationMapping).cols() == mSteps);
  assert(rollout->getForces(mRepresentationMapping).rows() == forceDim);
  int cursor = 0;
  if (useKnots)
  {
    if (mParallelOperationsEnabled)
    {
      std::vector<std::future<void>> futures;
      for (int i = 0; i < mShots.size(); i++)
      {
        int steps = mShots[i]->getNumSteps();
        futures.push_back(std::async(
            &MultiShot::asyncPartGetStates,
            this,
            i,
            mParallelWorlds[i],
            rollout,
            cursor,
            steps,
            thisLog));
        cursor += steps;
      }
      for (int i = 0; i < futures.size(); i++)
      {
        futures[i].wait();
      }
    }
    else
    {
      for (int i = 0; i < mShots.size(); i++)
      {
        int steps = mShots[i]->getNumSteps();
        TrajectoryRolloutRef slice = rollout->slice(cursor, steps);
        mShots[i]->getStates(world, &slice, thisLog, true);
        cursor += steps;
      }
    }
  }
  else
  {
    RestorableSnapshot snapshot(world);
    getRepresentation()->setPositions(world, mShots[0]->mStartPos);
    getRepresentation()->setVelocities(world, mShots[0]->mStartVel);
    for (int i = 0; i < mShots.size(); i++)
    {
      for (int j = 0; j < mShots[i]->mSteps; j++)
      {
        Eigen::VectorXd forces = mShots[i]->mForces.col(j);
        getRepresentation()->setForces(world, forces);
        world->step();
        for (auto pair : mMappings)
        {
          rollout->getPoses(pair.first).col(cursor)
              = pair.second->getPositions(world);
          rollout->getVels(pair.first).col(cursor)
              = pair.second->getVelocities(world);
          rollout->getForces(pair.first).col(cursor)
              = pair.second->getForces(world);
        }
        cursor++;
      }
    }
  }
  assert(cursor == mSteps);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
void MultiShot::asyncPartGetStates(
    int index,
    std::shared_ptr<simulation::World> world,
    /* OUT */ TrajectoryRollout* rollout,
    int cursor,
    int steps,
    PerformanceLog* log)
{
  TrajectoryRolloutRef slice = rollout->slice(cursor, steps);
  mShots[index]->getStates(world, &slice, log, true);
}

//==============================================================================
/// This returns the concatenation of (start pos, start vel) for convenience
Eigen::VectorXd MultiShot::getStartState()
{
  return mShots[0]->getStartState();
}

//==============================================================================
/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
Eigen::VectorXd MultiShot::getFinalState(
    std::shared_ptr<simulation::World> world, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.getFinalState");
  }
#endif

  Eigen::VectorXd ret
      = mShots[mShots.size() - 1]->getFinalState(world, thisLog);

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return ret;
}

//==============================================================================
/// This returns the debugging name of a given DOF
std::string MultiShot::getFlatDimName(
    std::shared_ptr<simulation::World> world, int dim)
{
  int staticDim = getFlatStaticProblemDim(world);
  if (dim < staticDim)
  {
    return "Static Dim " + std::to_string(dim);
  }
  dim -= staticDim;
  for (int i = 0; i < mShots.size(); i++)
  {
    int shotDim = mShots[i]->getFlatDynamicProblemDim(world);
    if (dim < shotDim)
    {
      // Using (dim + staticDim) as the value we pass to getFlatDimName() is
      // crucial here, because our SingleShot children each assume they own the
      // static region themselves, which isn't true in multiple shooting.
      return "Shot " + std::to_string(i) + " "
             + mShots[i]->getFlatDimName(world, dim + staticDim);
    }
    dim -= shotDim;
  }
  return "Error OOB";
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void MultiShot::backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* gradWrtRollout,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> gradStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> gradDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (log != nullptr)
  {
    thisLog = log->startRun("MultiShot.backpropGradientWrt");
  }
#endif

  int cursorDynamicDims = 0;
  int cursorSteps = 0;
  if (mParallelOperationsEnabled)
  {
    std::vector<std::future<void>> futures;
    Eigen::VectorXd gradStaticScratch
        = Eigen::VectorXd::Zero(gradStatic.size() * mShots.size());
    for (int i = 0; i < mShots.size(); i++)
    {
      int steps = mShots[i]->getNumSteps();
      int dynamicDim = mShots[i]->getFlatDynamicProblemDim(world);
      futures.push_back(std::async(
          &MultiShot::asyncPartBackpropGradientWrt,
          this,
          i,
          mParallelWorlds[i],
          gradWrtRollout,
          gradStaticScratch.segment(i * gradStatic.size(), gradStatic.size()),
          gradDynamic,
          cursorDynamicDims,
          cursorSteps,
          thisLog));
      cursorSteps += steps;
      cursorDynamicDims += dynamicDim;
    }
    gradStatic.setZero();
    for (int i = 0; i < futures.size(); i++)
    {
      futures[i].wait();
      gradStatic += gradStaticScratch.segment(
          i * gradStatic.size(), gradStatic.size());
    }
  }
  else
  {
    gradStatic.setZero();
    Eigen::VectorXd gradStaticScratch
        = Eigen::VectorXd::Zero(gradStatic.size());
    for (int i = 0; i < mShots.size(); i++)
    {
      int steps = mShots[i]->getNumSteps();
      int dynamicDim = mShots[i]->getFlatDynamicProblemDim(world);
      const TrajectoryRolloutConstRef slice
          = gradWrtRollout->sliceConst(cursorSteps, steps);
      mShots[i]->backpropGradientWrt(
          world,
          &slice,
          gradStaticScratch,
          gradDynamic.segment(cursorDynamicDims, dynamicDim),
          thisLog);
      gradStatic += gradStaticScratch;
      cursorSteps += steps;
      cursorDynamicDims += dynamicDim;
    }
  }

#ifdef LOG_PERFORMANCE_MULTI_SHOT
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void MultiShot::asyncPartBackpropGradientWrt(
    int index,
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* gradWrtRollout,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> gradStatic,
    /* OUT */ Eigen::Ref<Eigen::VectorXd> gradDynamic,
    int cursorDims,
    int cursorSteps,
    PerformanceLog* log)
{
  int steps = mShots[index]->getNumSteps();
  int dynamicDim = mShots[index]->getFlatDynamicProblemDim(world);
  const TrajectoryRolloutConstRef slice
      = gradWrtRollout->sliceConst(cursorSteps, steps);
  mShots[index]->backpropGradientWrt(
      world,
      &slice,
      gradStatic,
      gradDynamic.segment(cursorDims, dynamicDim),
      log);
}

} // namespace trajectory
} // namespace dart