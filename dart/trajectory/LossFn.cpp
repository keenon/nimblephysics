#include "dart/trajectory/LossFn.hpp"

using namespace dart;

namespace dart {
namespace trajectory {

//==============================================================================
LossFn::LossFn()
  : mLoss(std::nullopt),
    mLossAndGrad(std::nullopt),
    mLowerBound(-std::numeric_limits<double>::infinity()),
    mUpperBound(std::numeric_limits<double>::infinity())
{
}

//==============================================================================
LossFn::LossFn(TrajectoryLossFn loss)
  : mLoss(loss),
    mLossAndGrad(std::nullopt),
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
    const Eigen::Ref<const Eigen::MatrixXd>& poses,
    const Eigen::Ref<const Eigen::MatrixXd>& vels,
    const Eigen::Ref<const Eigen::MatrixXd>& forces)
{
  if (mLoss)
  {
    return mLoss.value()(poses, vels, forces);
  }
  // Default to 0
  return 0.0;
}

//==============================================================================
double LossFn::getLossAndGradient(
    const Eigen::Ref<const Eigen::MatrixXd>& poses,
    const Eigen::Ref<const Eigen::MatrixXd>& vels,
    const Eigen::Ref<const Eigen::MatrixXd>& forces,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtPoses,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtVels,
    /* OUT */ Eigen::Ref<Eigen::MatrixXd> gradWrtForces)
{
  if (mLossAndGrad)
  {
    return mLossAndGrad.value()(
        poses, vels, forces, gradWrtPoses, gradWrtVels, gradWrtForces);
  }
  else if (mLoss)
  {
    Eigen::MatrixXd posesCopy = poses;
    Eigen::MatrixXd velsCopy = vels;
    Eigen::MatrixXd forcesCopy = forces;
    double originalLoss = mLoss.value()(poses, vels, forces);

    const double EPS = 1e-7;

    for (int row = 0; row < posesCopy.rows(); row++)
    {
      // Only test the last step
      for (int col = 0; col < posesCopy.cols(); col++)
      {
        posesCopy(row, col) += EPS;
        double lossPos = mLoss.value()(posesCopy, vels, forces);
        posesCopy(row, col) -= EPS;
        gradWrtPoses(row, col) = (lossPos - originalLoss) / EPS;

        velsCopy(row, col) += EPS;
        double lossVel = mLoss.value()(poses, velsCopy, forces);
        velsCopy(row, col) -= EPS;
        gradWrtVels(row, col) = (lossVel - originalLoss) / EPS;

        forcesCopy(row, col) += EPS;
        double lossForce = mLoss.value()(poses, vels, forcesCopy);
        forcesCopy(row, col) -= EPS;
        gradWrtForces(row, col) = (lossForce - originalLoss) / EPS;
      }
    }

    return originalLoss;
  }

  // Default to 0
  gradWrtPoses.setZero();
  gradWrtVels.setZero();
  gradWrtForces.setZero();
  return 0.0;
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