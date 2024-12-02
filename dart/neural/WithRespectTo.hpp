#ifndef DART_NEURAL_WRT_HPP_
#define DART_NEURAL_WRT_HPP_

#include <memory>

#include <Eigen/Dense>

#include "dart/math/MathTypes.hpp"

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
class WithRespectToAcceleration;
class WithRespectToGroupScales;
class WithRespectToGroupMasses;
class WithRespectToLinearizedMasses;
class WithRespectToGroupCOMs;
class WithRespectToGroupInertias;

class WithRespectTo
{
public:
  virtual ~WithRespectTo();

  /// A printable name for this WRT object
  virtual std::string name() = 0;

  /// This returns this WRT from the world as a vector
  virtual Eigen::VectorXs get(simulation::World* world) = 0;

  /// This returns this WRT from a skeleton as a vector
  virtual Eigen::VectorXs get(dynamics::Skeleton* skel) = 0;

  /// This sets the world's state based on our WRT
  virtual void set(simulation::World* world, Eigen::VectorXs value) = 0;

  /// This sets the skeleton's state based on our WRT
  virtual void set(dynamics::Skeleton* skel, Eigen::VectorXs value) = 0;

  /// This gives the dimensions of the WRT in a whole world
  virtual int dim(simulation::World* world) = 0;

  /// This gives the dimensions of the WRT in a single skeleton
  virtual int dim(dynamics::Skeleton* skel) = 0;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  virtual Eigen::VectorXs upperBound(simulation::World* world) = 0;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  virtual Eigen::VectorXs lowerBound(simulation::World* world) = 0;

  static WithRespectToPosition* POSITION;
  static WithRespectToVelocity* VELOCITY;
  static WithRespectToForce* FORCE;
  static WithRespectToAcceleration* ACCELERATION;
  static WithRespectToGroupScales* GROUP_SCALES;
  static WithRespectToGroupMasses* GROUP_MASSES;
  static WithRespectToLinearizedMasses* LINEARIZED_MASSES;
  static WithRespectToGroupCOMs* GROUP_COMS;
  static WithRespectToGroupInertias* GROUP_INERTIAS;
};

class WithRespectToPosition : public WithRespectTo
{
public:
  WithRespectToPosition();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToVelocity : public WithRespectTo
{
public:
  WithRespectToVelocity();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToAcceleration : public WithRespectTo
{
public:
  WithRespectToAcceleration();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToGroupScales : public WithRespectTo
{
public:
  WithRespectToGroupScales();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToGroupMasses : public WithRespectTo
{
public:
  WithRespectToGroupMasses();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToLinearizedMasses : public WithRespectTo
{
public:
  WithRespectToLinearizedMasses();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToGroupCOMs : public WithRespectTo
{
public:
  WithRespectToGroupCOMs();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToGroupInertias : public WithRespectTo
{
public:
  WithRespectToGroupInertias();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

class WithRespectToForce : public WithRespectTo
{
public:
  WithRespectToForce();

  /// A printable name for this WRT object
  std::string name() override;

  /// This returns this WRT from the world as a vector
  Eigen::VectorXs get(simulation::World* world) override;

  /// This returns this WRT from a skeleton as a vector
  Eigen::VectorXs get(dynamics::Skeleton* skel) override;

  /// This sets the world's state based on our WRT
  void set(simulation::World* world, Eigen::VectorXs value) override;

  /// This sets the skeleton's state based on our WRT
  void set(dynamics::Skeleton* skel, Eigen::VectorXs value) override;

  /// This gives the dimensions of the WRT
  int dim(simulation::World* world) override;

  /// This gives the dimensions of the WRT in a single skeleton
  int dim(dynamics::Skeleton* skel) override;

  /// This gives a vector of upper bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs upperBound(simulation::World* world) override;

  /// This gives a vector of lower bound values for this WRT, given state in the
  /// world
  Eigen::VectorXs lowerBound(simulation::World* world) override;
};

} // namespace neural
} // namespace dart

#endif