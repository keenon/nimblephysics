#include "dart/trajectory/Problem.hpp"

#include <iostream>

#include <coin/IpIpoptApplication.hpp>
#include <coin/IpReturnCodes.hpp>
#include <coin/IpSolveStatistics.hpp>

#include "dart/neural/IdentityMapping.hpp"
#include "dart/neural/Mapping.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/simulation/World.hpp"

#define LOG_PERFORMANCE_PROBLEM

namespace dart {
namespace trajectory {

//==============================================================================
/// Default constructor
Problem::Problem(
    std::shared_ptr<simulation::World> world, LossFn loss, int steps)
  : mWorld(world),
    mLoss(loss),
    mSteps(steps),
    mRolloutCacheDirty(true),
    mExploreAlternateStrategies(false)
{
  std::shared_ptr<neural::Mapping> identityMapping
      = std::make_shared<neural::IdentityMapping>(world);
  mRepresentationMapping = "identity";
  mMappings[mRepresentationMapping] = identityMapping;
}

//==============================================================================
Problem::~Problem()
{
  // std::cout << "Freeing Problem: " << this << std::endl;
}

//==============================================================================
/// This updates the loss function for this trajectory
void Problem::setLoss(LossFn loss)
{
  mLoss = loss;
}

//==============================================================================
/// Add a custom constraint function to the trajectory
void Problem::addConstraint(LossFn loss)
{
  mConstraints.push_back(loss);
}

//==============================================================================
/// Register constant metadata, which will be passed along to the loss
/// function, but will not be backpropagated into.
void Problem::setMetadata(std::string key, Eigen::MatrixXs value)
{
  mMetadata[key] = value;
}

//==============================================================================
/// This returns the whole map for metadata
std::unordered_map<std::string, Eigen::MatrixXs>& Problem::getMetadataMap()
{
  return mMetadata;
}

//==============================================================================
/// If set to true, our backprop for finding the gradient may use alternative
/// contact strategies, other than the ones that are technically "correct".
void Problem::setExploreAlternateStrategies(bool flag)
{
  mExploreAlternateStrategies = flag;
}

//==============================================================================
/// If set to true, our backprop for finding the gradient may use alternative
/// contact strategies, other than the ones that are technically "correct".
bool Problem::getExploreAlternateStrategies()
{
  return mExploreAlternateStrategies;
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
void Problem::switchRepresentationMapping(
    std::shared_ptr<simulation::World> /* world */,
    const std::string& mapping,
    PerformanceLog* /* log */)
{
  // Reset the main representation mapping
  mRepresentationMapping = mapping;
  // Clear our cached trajectory
  mRolloutCacheDirty = true;
}

//==============================================================================
/// This adds a mapping through which the loss function can interpret the
/// output. We can have multiple loss mappings at the same time, and loss can
/// use arbitrary combinations of multiple views, as long as it can provide
/// gradients.
void Problem::addMapping(
    const std::string& key, std::shared_ptr<neural::Mapping> mapping)
{
  mMappings[key] = mapping;
  // Clear our cached trajectory
  mRolloutCacheDirty = true;
}

//==============================================================================
/// This returns true if there is a loss mapping at the specified key
bool Problem::hasMapping(const std::string& key)
{
  return mMappings.find(key) != mMappings.end();
}

//==============================================================================
/// This returns the loss mapping at the specified key
std::shared_ptr<neural::Mapping> Problem::getMapping(const std::string& key)
{
  return mMappings[key];
}

//==============================================================================
/// This returns a reference to all the mappings in this shot
std::unordered_map<std::string, std::shared_ptr<neural::Mapping>>&
Problem::getMappings()
{
  return mMappings;
}

//==============================================================================
/// This removes the loss mapping at a particular key
void Problem::removeMapping(const std::string& key)
{
  mMappings.erase(key);
  // Clear our cached trajectory
  mRolloutCacheDirty = true;
}

//==============================================================================
/// Returns the sum of posDim() + velDim() for the current representation
/// mapping
int Problem::getRepresentationStateSize() const
{
  return getRepresentation()->getPosDim() + getRepresentation()->getVelDim();
}

//==============================================================================
const std::string& Problem::getRepresentationName() const
{
  return mRepresentationMapping;
}

//==============================================================================
/// Returns the representation currently being used
const std::shared_ptr<neural::Mapping> Problem::getRepresentation() const
{
  return mMappings.at(mRepresentationMapping);
}

//==============================================================================
/// Returns the length of the flattened problem state
int Problem::getFlatProblemDim(std::shared_ptr<simulation::World> world) const
{
  return getFlatStaticProblemDim(world) + getFlatDynamicProblemDim(world);
}

//==============================================================================
int Problem::getFlatStaticProblemDim(
    std::shared_ptr<simulation::World> world) const
{
  return world->getMassDims();
}

//==============================================================================
int Problem::getFlatDynamicProblemDim(
    std::shared_ptr<simulation::World> /* world */) const
{
  return 0;
}

//==============================================================================
/// This copies a shot down into a single flat vector
void Problem::flatten(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
    // Subclasses use flatDynamic, but we don't because at this level of
    // abstraction we don't know about the temporal aspects of the problem.
    /* OUT */ Eigen::Ref<Eigen::VectorXs> /* flatDynamic */,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.flatten");
  }
#endif

  flatStatic.segment(0, world->getMassDims()) = world->getMasses();

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This copies a shot down into a single flat vector
void Problem::flatten(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  flatten(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the parameters out of a flat vector
void Problem::unflatten(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::VectorXs>& flatStatic,
    // Subclasses use flatDynamic, but we don't because at this level of
    // abstraction we don't know about the temporal aspects of the problem.
    const Eigen::Ref<const Eigen::VectorXs>& /* flatDynamic */,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.unflatten");
  }
#endif

  world->setMasses(flatStatic.segment(0, world->getMassDims()));

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the parameters out of a flat vector
void Problem::unflatten(
    std::shared_ptr<simulation::World> world,
    const Eigen::Ref<const Eigen::VectorXs>& flat,
    PerformanceLog* log)
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  unflatten(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void Problem::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  getUpperBounds(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void Problem::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  getLowerBounds(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
void Problem::getUpperBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
    // Subclasses use flatDynamic, but we don't because at this level of
    // abstraction we don't know about the temporal aspects of the problem.
    /* OUT */ Eigen::Ref<Eigen::VectorXs> /* flatDynamic */,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getUpperBounds");
  }
#endif

  flatStatic.segment(0, world->getMassDims()) = world->getMassUpperLimits();

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
void Problem::getLowerBounds(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
    // Subclasses use flatDynamic, but we don't because at this level of
    // abstraction we don't know about the temporal aspects of the problem.
    /* OUT */ Eigen::Ref<Eigen::VectorXs> /* flatDynamic */,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getLowerBounds");
  }
#endif

  flatStatic.segment(0, world->getMassDims()) = world->getMassLowerLimits();

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void Problem::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flatStatic,
    // Subclasses use flatDynamic, but we don't because at this level of
    // abstraction we don't know about the temporal aspects of the problem.
    /* OUT */ Eigen::Ref<Eigen::VectorXs> /* flatDynamic */,
    PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getInitialGuess");
  }
#endif

  flatStatic.segment(0, world->getMassDims()) = world->getMasses();

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void Problem::getConstraintUpperBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getConstraintUpperBounds");
  }
#endif

  assert(flat.size() == mConstraints.size());
  for (int i = 0; i < mConstraints.size(); i++)
  {
    flat(i) = mConstraints[i].getUpperBound();
  }

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
void Problem::getConstraintLowerBounds(
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flat, PerformanceLog* log) const
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getConstraintLowerBounds");
  }
#endif

  assert(flat.size() == mConstraints.size());
  for (int i = 0; i < mConstraints.size(); i++)
  {
    flat(i) = mConstraints[i].getLowerBound();
  }

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This returns the initial guess for the values of X when running an
/// optimization
void Problem::getInitialGuess(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> flat,
    PerformanceLog* log) const
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  getInitialGuess(
      world,
      flat.segment(0, staticDim),
      flat.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
int Problem::getConstraintDim() const
{
  return mConstraints.size();
}

//==============================================================================
/// This computes the values of the constraints, assuming that the constraint
/// vector being passed in is only the size of mConstraints
void Problem::computeConstraints(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> constraints,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.computeConstraints");
  }
#endif

  assert(constraints.size() == mConstraints.size());

  for (int i = 0; i < mConstraints.size(); i++)
  {
    constraints(i)
        = mConstraints[i].getLoss(getRolloutCache(world, thisLog), thisLog);
  }

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
void Problem::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXs> jac,
    PerformanceLog* log)
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  int numConstraints = getConstraintDim();
  assert(jac.rows() == numConstraints);
  assert(jac.cols() == staticDim + dynamicDim);
  backpropJacobian(
      world,
      jac.block(0, 0, numConstraints, staticDim),
      jac.block(0, staticDim, numConstraints, dynamicDim),
      log);
}

//==============================================================================
/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
void Problem::backpropJacobian(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacStatic,
    /* OUT */ Eigen::Ref<Eigen::MatrixXs> jacDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.backpropJacobian");
  }
#endif

  assert(jacStatic.rows() == mConstraints.size());
  assert(jacStatic.cols() == getFlatStaticProblemDim(world));
  assert(jacDynamic.rows() == mConstraints.size());
  assert(jacDynamic.cols() == getFlatDynamicProblemDim(world));

  Eigen::VectorXs gradStatic
      = Eigen::VectorXs::Zero(getFlatStaticProblemDim(world));
  Eigen::VectorXs gradDynamic
      = Eigen::VectorXs::Zero(getFlatDynamicProblemDim(world));
  for (int i = 0; i < mConstraints.size(); i++)
  {
    mConstraints[i].getLossAndGradient(
        getRolloutCache(world, thisLog),
        /* OUT */ getGradientWrtRolloutCache(world, thisLog),
        thisLog);
    gradStatic.setZero();
    gradDynamic.setZero();
    backpropGradientWrt(
        world,
        getGradientWrtRolloutCache(world, thisLog),
        /* OUT */ gradStatic,
        /* OUT */ gradDynamic,
        thisLog);
    jacDynamic.row(i) = gradDynamic;
    jacStatic.row(i) = gradStatic;
  }

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int Problem::getNumberNonZeroJacobianStatic(
    std::shared_ptr<simulation::World> world)
{
  return mConstraints.size() * getFlatStaticProblemDim(world);
}

//==============================================================================
/// This gets the number of non-zero entries in the Jacobian
int Problem::getNumberNonZeroJacobianDynamic(
    std::shared_ptr<simulation::World> world)
{
  return mConstraints.size() * getFlatDynamicProblemDim(world);
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void Problem::getJacobianSparsityStructure(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  int nnzjStatic = getNumberNonZeroJacobianStatic(world);
  int nnzjDynamic = getNumberNonZeroJacobianDynamic(world);
  assert(
      nnzjStatic + nnzjDynamic == rows.size()
      && nnzjStatic + nnzjDynamic == cols.size());
  getJacobianSparsityStructureStatic(
      world, rows.segment(0, nnzjStatic), cols.segment(0, nnzjStatic), log);
  getJacobianSparsityStructureDynamic(
      world,
      rows.segment(nnzjStatic, nnzjDynamic),
      cols.segment(nnzjStatic, nnzjDynamic),
      log);
  // Bump all the dynamic elements over by `staticCols`
  int staticCols = getFlatStaticProblemDim(world);
  cols.segment(nnzjStatic, nnzjDynamic)
      += Eigen::VectorXi::Ones(nnzjDynamic) * staticCols;
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void Problem::getJacobianSparsityStructureDynamic(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getJacobianSparsityStructure");
  }
#endif

  assert(rows.size() == Problem::getNumberNonZeroJacobianDynamic(world));
  assert(cols.size() == Problem::getNumberNonZeroJacobianDynamic(world));
  int cursor = 0;
  // Do row-major ordering
  for (int j = 0; j < mConstraints.size(); j++)
  {
    for (int i = 0; i < getFlatDynamicProblemDim(world); i++)
    {
      rows(cursor) = j;
      cols(cursor) = i;
      cursor++;
    }
  }
  assert(cursor == Problem::getNumberNonZeroJacobianDynamic(world));

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets the structure of the non-zero entries in the Jacobian
void Problem::getJacobianSparsityStructureStatic(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXi> rows,
    Eigen::Ref<Eigen::VectorXi> cols,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getJacobianSparsityStructure");
  }
#endif

  assert(rows.size() == Problem::getNumberNonZeroJacobianStatic(world));
  assert(cols.size() == Problem::getNumberNonZeroJacobianStatic(world));
  int cursor = 0;
  // Do row-major ordering
  for (int j = 0; j < mConstraints.size(); j++)
  {
    for (int i = 0; i < getFlatStaticProblemDim(world); i++)
    {
      rows(cursor) = j;
      cols(cursor) = i;
      cursor++;
    }
  }
  assert(cursor == Problem::getNumberNonZeroJacobianStatic(world));

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This writes the Jacobian to a pair of sparse vectors, separating out the
/// static and dynamic regions.
void Problem::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> sparseStatic,
    Eigen::Ref<Eigen::VectorXs> sparseDynamic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getSparseJacobian");
  }
#endif

  assert(sparseStatic.size() == Problem::getNumberNonZeroJacobianStatic(world));
  assert(
      sparseDynamic.size() == Problem::getNumberNonZeroJacobianDynamic(world));

  sparseStatic.setZero();
  sparseDynamic.setZero();

  int cursorDynamic = 0;
  int cursorStatic = 0;
  int nStatic = getFlatStaticProblemDim(world);
  int nDynamic = getFlatDynamicProblemDim(world);
  for (int i = 0; i < mConstraints.size(); i++)
  {
    mConstraints[i].getLossAndGradient(
        getRolloutCache(world, thisLog),
        /* OUT */ getGradientWrtRolloutCache(world, thisLog),
        thisLog);
    backpropGradientWrt(
        world,
        getGradientWrtRolloutCache(world, thisLog),
        /* OUT */ sparseStatic.segment(cursorStatic, nStatic),
        /* OUT */ sparseDynamic.segment(cursorDynamic, nDynamic),
        thisLog);
    cursorStatic += nStatic;
    cursorDynamic += nDynamic;
  }

  assert(cursorStatic == sparseStatic.size());
  assert(cursorDynamic == sparseDynamic.size());

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This writes the Jacobian to a sparse vector
void Problem::getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> sparse,
    PerformanceLog* log)
{
  int nnzjStatic = getNumberNonZeroJacobianStatic(world);
  int nnzjDynamic = getNumberNonZeroJacobianDynamic(world);
  // Simply concatenate the two results together
  getSparseJacobian(
      world,
      sparse.segment(0, nnzjStatic),
      sparse.segment(nnzjStatic, nnzjDynamic),
      log);
}

//==============================================================================
/// This computes the gradient in the flat problem space, automatically
/// computing the gradients of the loss function as part of the call
void Problem::backpropGradient(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> grad,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.backpropGradient");
  }
#endif

  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);

  mLoss.getLossAndGradient(
      getRolloutCache(world, thisLog),
      /* OUT */ getGradientWrtRolloutCache(world, thisLog),
      thisLog);
  backpropGradientWrt(
      world,
      getGradientWrtRolloutCache(world, thisLog),
      /* OUT */ grad.segment(0, staticDim),
      /* OUT */ grad.segment(staticDim, dynamicDim),
      thisLog);

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
void Problem::backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout* gradWrtRollout,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> grad,
    PerformanceLog* log)
{
  int staticDim = getFlatStaticProblemDim(world);
  int dynamicDim = getFlatDynamicProblemDim(world);
  backpropGradientWrt(
      world,
      gradWrtRollout,
      grad.segment(0, staticDim),
      grad.segment(staticDim, dynamicDim),
      log);
}

//==============================================================================
/// Get the loss for the rollout
s_t Problem::getLoss(
    std::shared_ptr<simulation::World> world, PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getLoss");
  }
#endif

  s_t val = mLoss.getLoss(getRolloutCache(world, thisLog), thisLog);

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return val;
}

//==============================================================================
/// This gets called at the beginning of backpropGradientWrt(), as an
/// opportunity to zero out any static gradient values being managed by
/// Problem.
void Problem::initializeStaticGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> gradStatic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.initializeStaticGradient");
  }
#endif

  gradStatic.segment(0, world->getMassDims()).setZero();

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This adds anything to the static gradient that we need to. It needs to be
/// called for every timestep during backpropGradientWrt().
void Problem::accumulateStaticGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> gradStatic,
    neural::LossGradient& thisTimestep,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.accumulateStaticGradient");
  }
#endif

  gradStatic.segment(0, world->getMassDims()) += thisTimestep.lossWrtMass;

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This gets called at the beginning of backpropJacobianOfFinalState() in
/// SingleShot, as an opportunity to zero out any static jacobian values being
/// managed by Problem.
void Problem::initializeStaticJacobianOfFinalState(
    std::shared_ptr<simulation::World> /* world */,
    Eigen::Ref<Eigen::MatrixXs> jacStatic,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.initializeStaticJacobianOfFinalState");
  }
#endif

  jacStatic.setZero();

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This adds anything to the static gradient that we need to. It needs to be
/// called for every timestep during backpropJacobianOfFinalState() in
/// SingleShot.
void Problem::accumulateStaticJacobianOfFinalState(
    std::shared_ptr<simulation::World> /* world */,
    Eigen::Ref<Eigen::MatrixXs> jacStatic,
    TimestepJacobians& thisTimestep,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.accumulateStaticJacobianOfFinalState");
  }
#endif

  jacStatic.block(
      0, 0, thisTimestep.massPos.rows(), thisTimestep.massPos.cols())
      += thisTimestep.massPos;
  jacStatic.block(
      thisTimestep.massPos.rows(),
      0,
      thisTimestep.massVel.rows(),
      thisTimestep.massVel.cols())
      += thisTimestep.massVel;

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif
}

//==============================================================================
/// This sets the forces in this trajectory from the passed in matrix. This
/// updates the knot points as part of the update, and returns a mapping of
/// where indices moved around to, so that we can update lagrange multipliers
/// etc.
Eigen::VectorXi Problem::updateWithForces(
    std::shared_ptr<simulation::World> world,
    Eigen::MatrixXs forces,
    PerformanceLog* log)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.updateWithForces");
  }
#endif

  Eigen::VectorXi mapping = Eigen::VectorXi::Zero(getFlatProblemDim(world));

  neural::RestorableSnapshot snapshot(world);
  TrajectoryRollout* rollout = getRolloutCache(world, thisLog)->copy();

  for (int i = 0; i < mSteps; i++)
  {
    mMappings[mRepresentationMapping]->setForces(world, forces.col(i));
    for (std::string mapping : rollout->getMappings())
    {
      mMappings[mapping]->getPositionsInPlace(
          world, rollout->getPoses(mapping).col(i));
      mMappings[mapping]->getVelocitiesInPlace(
          world, rollout->getVels(mapping).col(i));
      mMappings[mapping]->getForcesInPlace(
          world, rollout->getForces(mapping).col(i));
    }
    world->step();
  }

  setStates(world, rollout, thisLog);

  snapshot.restore();
  delete rollout;

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return mapping;
}

//==============================================================================
const TrajectoryRollout* Problem::getRolloutCache(
    std::shared_ptr<simulation::World> world,
    PerformanceLog* log,
    bool /* useKnots */)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getRolloutCache");
  }
#endif

  if (mRolloutCacheDirty)
  {
    mRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    getStates(
        world,
        /* OUT */ mRolloutCache.get(),
        thisLog);
    mGradWrtRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    mRolloutCacheDirty = false;
  }

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return mRolloutCache.get();
}

//==============================================================================
TrajectoryRollout* Problem::getGradientWrtRolloutCache(
    std::shared_ptr<simulation::World> world,
    PerformanceLog* log,
    bool /* useKnots */)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_PROBLEM
  if (log != nullptr)
  {
    thisLog = log->startRun("Problem.getGradientWrtRolloutCache");
  }
#endif

  if (mRolloutCacheDirty)
  {
    mRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    getStates(
        world,
        /* OUT */ mRolloutCache.get(),
        thisLog);
    mGradWrtRolloutCache = std::make_shared<TrajectoryRolloutReal>(this);
    mRolloutCacheDirty = false;
  }

#ifdef LOG_PERFORMANCE_PROBLEM
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return mGradWrtRolloutCache.get();
}

//==============================================================================
/// This computes finite difference Jacobians analagous to
/// backpropGradient()
void Problem::finiteDifferenceGradient(
    std::shared_ptr<simulation::World> world,
    /* OUT */ Eigen::Ref<Eigen::VectorXs> grad)
{
  int dims = getFlatProblemDim(world);
  Eigen::VectorXs flat = Eigen::VectorXs::Zero(dims);
  flatten(world, flat, nullptr);

  assert(grad.size() == dims);

  const s_t EPS = 1e-6;

  for (int i = 0; i < dims; i++)
  {
    flat(i) += EPS;
    unflatten(world, flat, nullptr);
    s_t posLoss = mLoss.getLoss(getRolloutCache(world, nullptr), nullptr);
    flat(i) -= EPS;

    flat(i) -= EPS;
    unflatten(world, flat, nullptr);
    s_t negLoss = mLoss.getLoss(getRolloutCache(world, nullptr), nullptr);
    flat(i) += EPS;

    grad(i) = (posLoss - negLoss) / (2 * EPS);
  }
}

//==============================================================================
int Problem::getNumSteps()
{
  return mSteps;
}

//==============================================================================
/// Returns the dimension of the mass vector
int Problem::getMassDims()
{
  return mWorld->getMassDims();
}

//==============================================================================
/// This gets the total number of non-zero entries in the Jacobian
int Problem::getNumberNonZeroJacobian(std::shared_ptr<simulation::World> world)
{
  return getNumberNonZeroJacobianStatic(world)
         + getNumberNonZeroJacobianDynamic(world);
}

//==============================================================================
/// This computes finite difference Jacobians analagous to backpropJacobians()
void Problem::finiteDifferenceJacobian(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXs> jac)
{
  int dim = getFlatProblemDim(world);
  int numConstraints = getConstraintDim();
  assert(jac.cols() == dim);
  assert(jac.rows() == numConstraints);

  Eigen::VectorXs originalConstraints = Eigen::VectorXs::Zero(numConstraints);
  computeConstraints(world, originalConstraints, nullptr);
  Eigen::VectorXs flat = Eigen::VectorXs::Zero(dim);
  flatten(world, flat, nullptr);

  const s_t EPS = 1e-7;

  Eigen::VectorXs positiveConstraints = Eigen::VectorXs::Zero(numConstraints);
  Eigen::VectorXs negativeConstraints = Eigen::VectorXs::Zero(numConstraints);
  for (int i = 0; i < dim; i++)
  {
    flat(i) += EPS;
    unflatten(world, flat, nullptr);
    computeConstraints(world, positiveConstraints, nullptr);
    flat(i) -= EPS;

    flat(i) -= EPS;
    unflatten(world, flat, nullptr);
    computeConstraints(world, negativeConstraints, nullptr);
    flat(i) += EPS;

    jac.col(i) = (positiveConstraints - negativeConstraints) / (2 * EPS);
  }

  // Reset to original state
  unflatten(world, flat, nullptr);
}

} // namespace trajectory
} // namespace dart