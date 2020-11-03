#include "dart/trajectory/LossFn.hpp"

#include "dart/utils/tl_optional.hpp"

#define LOG_PERFORMANCE_LOSS_FN

using namespace dart;

namespace dart {
namespace trajectory {

//==============================================================================
LossFn::LossFn()
  : mLoss(tl::nullopt),
    mLossAndGrad(tl::nullopt),
    mLowerBound(-std::numeric_limits<double>::infinity()),
    mUpperBound(std::numeric_limits<double>::infinity())
{
}

//==============================================================================
LossFn::LossFn(TrajectoryLossFn loss)
  : mLoss(loss),
    mLossAndGrad(tl::nullopt),
    mLowerBound(-std::numeric_limits<double>::infinity()),
    mUpperBound(std::numeric_limits<double>::infinity())
{
}

//==============================================================================
LossFn::LossFn(TrajectoryLossFn loss, TrajectoryLossFnAndGrad lossAndGrad)
  : mLoss(loss),
    mLossAndGrad(lossAndGrad),
    mLowerBound(-std::numeric_limits<double>::infinity()),
    mUpperBound(std::numeric_limits<double>::infinity())
{
}

//==============================================================================
LossFn::~LossFn()
{
}

//==============================================================================
double LossFn::getLoss(
    const TrajectoryRollout* rollout, PerformanceLog* perflog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_LOSS_FN
  if (perflog != nullptr)
  {
    thisLog = perflog->startRun("LossFn.getLoss");
  }
#endif

  double loss = 0.0;

  if (mLoss)
  {
    loss = mLoss.value()(rollout);
  }

#ifdef LOG_PERFORMANCE_LOSS_FN
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return loss;
}

//==============================================================================
double LossFn::getLossAndGradient(
    const TrajectoryRollout* rollout,
    /* OUT */ TrajectoryRollout* gradWrtRollout,
    PerformanceLog* perflog)
{
  PerformanceLog* thisLog = nullptr;
#ifdef LOG_PERFORMANCE_LOSS_FN
  if (perflog != nullptr)
  {
    thisLog = perflog->startRun("LossFn.getLossAndGradient");
  }
#endif

  double loss = 0.0;

  if (mLossAndGrad)
  {
    loss = mLossAndGrad.value()(rollout, gradWrtRollout);
  }
  else if (mLoss)
  {
    TrajectoryRolloutReal rolloutCopy = TrajectoryRolloutReal(rollout);
    double originalLoss = mLoss.value()(&rolloutCopy);

    const double EPS = 1e-7;

    for (int i = 0; i < rolloutCopy.getMasses().size(); i++)
    {
      rolloutCopy.getMasses()(i) += EPS;
      double lossPos = mLoss.value()(&rolloutCopy);
      rolloutCopy.getMasses()(i) -= EPS;

      rolloutCopy.getMasses()(i) -= EPS;
      double lossNeg = mLoss.value()(&rolloutCopy);
      rolloutCopy.getMasses()(i) += EPS;

      gradWrtRollout->getMasses()(i) = (lossPos - lossNeg) / (2 * EPS);
    }

    for (std::string key : rolloutCopy.getMappings())
    {
      for (int row = 0; row < rolloutCopy.getPoses(key).rows(); row++)
      {
        for (int col = 0; col < rolloutCopy.getPoses(key).cols(); col++)
        {
          rolloutCopy.getPoses(key)(row, col) += EPS;
          double lossPos = mLoss.value()(&rolloutCopy);
          rolloutCopy.getPoses(key)(row, col) -= EPS;

          rolloutCopy.getPoses(key)(row, col) -= EPS;
          double lossNeg = mLoss.value()(&rolloutCopy);
          rolloutCopy.getPoses(key)(row, col) += EPS;

          gradWrtRollout->getPoses(key)(row, col)
              = (lossPos - lossNeg) / (2 * EPS);
        }
      }
      for (int row = 0; row < rolloutCopy.getVels(key).rows(); row++)
      {
        for (int col = 0; col < rolloutCopy.getVels(key).cols(); col++)
        {
          rolloutCopy.getVels(key)(row, col) += EPS;
          double lossVel = mLoss.value()(&rolloutCopy);
          rolloutCopy.getVels(key)(row, col) -= EPS;

          rolloutCopy.getVels(key)(row, col) -= EPS;
          double lossNeg = mLoss.value()(&rolloutCopy);
          rolloutCopy.getVels(key)(row, col) += EPS;

          gradWrtRollout->getVels(key)(row, col)
              = (lossVel - lossNeg) / (2 * EPS);
        }
      }
      for (int row = 0; row < rolloutCopy.getForces(key).rows(); row++)
      {
        for (int col = 0; col < rolloutCopy.getForces(key).cols(); col++)
        {
          rolloutCopy.getForces(key)(row, col) += EPS;
          double lossForce = mLoss.value()(&rolloutCopy);
          rolloutCopy.getForces(key)(row, col) -= EPS;

          rolloutCopy.getForces(key)(row, col) -= EPS;
          double lossNeg = mLoss.value()(&rolloutCopy);
          rolloutCopy.getForces(key)(row, col) += EPS;

          gradWrtRollout->getForces(key)(row, col)
              = (lossForce - lossNeg) / (2 * EPS);
        }
      }
    }

    loss = originalLoss;
  }
  else
  {
    // Default to 0
    for (std::string key : gradWrtRollout->getMappings())
    {
      gradWrtRollout->getPoses(key).setZero();
      gradWrtRollout->getVels(key).setZero();
      gradWrtRollout->getForces(key).setZero();
    }
    loss = 0.0;
  }

#ifdef LOG_PERFORMANCE_LOSS_FN
  if (thisLog != nullptr)
  {
    thisLog->end();
  }
#endif

  return loss;
}

//==============================================================================
/// If this LossFn is being used as a constraint, this gets the lower bound
/// it's allowed to reach
double LossFn::getLowerBound() const
{
  return mLowerBound;
}

//==============================================================================
/// If this LossFn is being used as a constraint, this sets the lower bound
/// it's allowed to reach
void LossFn::setLowerBound(double lowerBound)
{
  mLowerBound = lowerBound;
}

//==============================================================================
/// If this LossFn is being used as a constraint, this gets the upper bound
/// it's allowed to reach
double LossFn::getUpperBound() const
{
  return mUpperBound;
}

//==============================================================================
/// If this LossFn is being used as a constraint, this sets the upper bound
/// it's allowed to reach
void LossFn::setUpperBound(double upperBound)
{
  mUpperBound = upperBound;
}

} // namespace trajectory
} // namespace dart