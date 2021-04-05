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
Eigen::VectorXs Mapping::getPositions(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs poses = Eigen::VectorXs::Zero(getPosDim());
  getPositionsInPlace(world, poses);
  return poses;
}

//==============================================================================
Eigen::VectorXs Mapping::getVelocities(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs vels = Eigen::VectorXs::Zero(getVelDim());
  getVelocitiesInPlace(world, vels);
  return vels;
}

//==============================================================================
Eigen::VectorXs Mapping::getForces(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs forces = Eigen::VectorXs::Zero(getForceDim());
  getForcesInPlace(world, forces);
  return forces;
}

//==============================================================================
Eigen::VectorXs Mapping::getMasses(std::shared_ptr<simulation::World> world)
{
  Eigen::VectorXs masses = Eigen::VectorXs::Zero(getMassDim());
  getMassesInPlace(world, masses);
  return masses;
}

//==============================================================================
Eigen::VectorXs Mapping::getPositionLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getPosDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getPositionUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getPosDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getVelocityLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getVelDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getVelocityUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getVelDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getForceLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getForceDim())
         * -std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getForceUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getForceDim())
         * std::numeric_limits<s_t>::infinity();
}

//==============================================================================
Eigen::VectorXs Mapping::getMassLowerLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  // Non-zero lower bound by default
  return Eigen::VectorXs::Ones(getMassDim()) * 1e-7;
}

//==============================================================================
Eigen::VectorXs Mapping::getMassUpperLimits(
    std::shared_ptr<simulation::World> /* world */)
{
  return Eigen::VectorXs::Ones(getMassDim())
         * std::numeric_limits<s_t>::infinity();
}

} // namespace neural
} // namespace dart