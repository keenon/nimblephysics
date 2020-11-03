#ifndef DART_NEURAL_WRT_HPP_
#define DART_NEURAL_WRT_HPP_

#include <memory>

#include <Eigen/Dense>

namespace dart {
namespace simulation {
class World;
}

namespace dynamics {
class Skeleton;
}

namespace neural {

class WithRespectToPosition;
class WithRespectToVelocity;
class WithRespectToForce;

class WithRespectTo
{
public:
  virtual ~WithRespectTo();

  /// This returns this WRT from the world as a vector
  virtual Eigen::VectorXd get(simulation::World* world) = 0;

  /// This returns this WRT from a skeleton as a vector
  virtual Eigen::VectorXd get(dynamics::Skeleton* skel) = 0;

  /// This sets the world's state based on our WRT
  virtual void set(simulation::World* world, Eigen::VectorXd value) = 0;

  /// This sets the skeleton's state based on our WRT
  virtual void set(dynamics::Skeleton* skel, Eigen::VectorXd value) = 0;

  /// This gives the dimensions of the WRT in a whole world
  virtual int dim(simulation::World* world) = 0;

  /// This gives the dimensions of the WRT in a single skeleton
  virtual int dim(dynamics::Skeleton* skel) = 0;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  virtual Eigen::VectorXd upperBound(simulation::World* world) = 0;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  virtual Eigen::VectorXd lowerBound(simulation::World* world) = 0;

  static WithRespectToPosition* POSITION;
  static WithRespectToVelocity* VELOCITY;
  static WithRespectToForce* FORCE;
};

class WithRespectToPosition : public WithRespectTo
{
public:
  WithRespectToPosition();

  /// This returns this WRT from the world as a vector
  Eigen::VectorXd get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXd get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXd value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXd value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXd upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXd lowerBound(simulation::World* world) override;
};

class WithRespectToVelocity : public WithRespectTo
{
public:
  WithRespectToVelocity();

  /// This returns this WRT from the world as a vector
  Eigen::VectorXd get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXd get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXd value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXd value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXd upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXd lowerBound(simulation::World* world) override;
};

class WithRespectToForce : public WithRespectTo
{
public:
  WithRespectToForce();

  /// This returns this WRT from the world as a vector
  Eigen::VectorXd get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXd get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXd value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXd value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXd upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXd lowerBound(simulation::World* world) override;
};

} // namespace neural
} // namespace dart

#endif