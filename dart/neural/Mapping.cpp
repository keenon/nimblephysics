#include "dart/neural/Mapping.hpp"

using namespace dart;

namespace dart {
namespace neural {

//==============================================================================
Mapping::Mapping()
{
}

//==============================================================================
Mapping::~Mapping()
{
}

//==============================================================================
Eigen::VectorXd Mapping::getPositions(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXd poses = Eigen::VectorXd::Zero(getPosDim());
  getPositionsInPlace(world, poses);
  return poses;
}

//==============================================================================
Eigen::VectorXd Mapping::getVelocities(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXd vels = Eigen::VectorXd::Zero(getVelDim());
  getVelocitiesInPlace(world, vels);
  return vels;
}

//==============================================================================
Eigen::VectorXd Mapping::getForces(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXd forces = Eigen::VectorXd::Zero(getForceDim());
  getForcesInPlace(world, forces);
  return forces;
}

//==============================================================================
Eigen::VectorXd Mapping::getMasses(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXd masses = Eigen::VectorXd::Zero(getMassDim());
  getMassesInPlace(world, masses);
  return masses;
}

//==============================================================================
Eigen::VectorXd Mapping::getPositionLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXd::Ones(getPosDim())
         * -std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd Mapping::getPositionUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXd::Ones(getPosDim())
         * std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd Mapping::getVelocityLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXd::Ones(getVelDim())
         * -std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd Mapping::getVelocityUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXd::Ones(getVelDim())
         * std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd Mapping::getForceLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXd::Ones(getForceDim())
         * -std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd Mapping::getForceUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXd::Ones(getForceDim())
         * std::numeric_limits<double>::infinity();
}

//==============================================================================
Eigen::VectorXd Mapping::getMassLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  // Non-zero lower bound by default
  return Eigen::VectorXd::Ones(getMassDim()) * 1e-7;
}

//==============================================================================
Eigen::VectorXd Mapping::getMassUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXd::Ones(getMassDim())
         * std::numeric_limits<double>::infinity();
}

} // namespace neural
} // namespace dart