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

} // namespace neural
} // namespace dart