/*
 * Copyright (c) 2011-2019, The DART development contributors
 * All rights reserved.
 *
 * The list of contributors can be found at:
 *   https://github.com/dartsim/dart/blob/master/LICENSE
 *
 * This file is provided under the following "BSD-style" License:
 *   Redistribution and use in source and binary forms, with or
 *   without modification, are permitted provided that the following
 *   conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 *   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *   POSSIBILITY OF SUCH DAMAGE.
 */

#include <dart/simulation/World.hpp>
#include <dart/trajectory/Problem.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

/*
  /// This updates the loss function for this trajectory
  void setLoss(LossFn loss);

  /// Add a custom constraint function to the trajectory
  void addConstraint(LossFn loss);

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
  virtual void switchRepresentationMapping(
      std::shared_ptr<simulation::World> world, const std::string& mapping);

  /// This adds a mapping through which the loss function can interpret the
  /// output. We can have multiple loss mappings at the same time, and loss can
  /// use arbitrary combinations of multiple views, as long as it can provide
  /// gradients.
  virtual void addMapping(
      const std::string& key, std::shared_ptr<neural::Mapping> mapping);

  /// This returns true if there is a loss mapping at the specified key
  bool hasMapping(const std::string& key);

  /// This returns the loss mapping at the specified key
  std::shared_ptr<neural::Mapping> getMapping(const std::string& key);

  /// This returns a reference to all the mappings in this shot
  std::unordered_map<std::string, std::shared_ptr<neural::Mapping>>&
  getMappings();

  /// This removes the loss mapping at a particular key
  virtual void removeMapping(const std::string& key);

  /// Returns the sum of posDim() + velDim() for the current representation
  /// mapping
  int getRepresentationStateSize() const;

  const std::string& getRepresentationName() const;

  /// Returns the representation currently being used
  const std::shared_ptr<neural::Mapping> getRepresentation() const;

  /// Returns the length of the flattened problem state
  virtual int getFlatProblemDim() const = 0;

  /// Returns the length of the knot-point constraint vector
  virtual int getConstraintDim() const;

  /// This copies a shot down into a single flat vector
  virtual void flatten(Eigen::Ref<Eigen::VectorXs> flat) const = 0;

/// This gets the parameters out of a flat vector
virtual void unflatten(const Eigen::Ref<const Eigen::VectorXs>& flat) = 0;

/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
virtual void getUpperBounds(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> flat) const = 0;

/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
virtual void getLowerBounds(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> flat) const = 0;

/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
virtual void getConstraintUpperBounds(
    Eigen::Ref<Eigen::VectorXs> flat) const;

/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
virtual void getConstraintLowerBounds(
    Eigen::Ref<Eigen::VectorXs> flat) const;

/// This returns the initial guess for the values of X when running an
/// optimization
virtual void getInitialGuess(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> flat) const = 0;

/// This computes the values of the constraints
virtual void computeConstraints(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> constraints);

/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
virtual void backpropJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::MatrixXs> jac);

/// This computes the gradient in the flat problem space, automatically
/// computing the gradients of the loss function as part of the call
void backpropGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> grad);

/// Get the loss for the rollout
s_t getLoss(std::shared_ptr<simulation::World> world);

/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
virtual void backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout& gradWrtRollout,
    Eigen::Ref<Eigen::VectorXs> grad)
    = 0;

/// This populates the passed in matrices with the values from this trajectory
virtual void getStates(
    std::shared_ptr<simulation::World> world,
    TrajectoryRollout& rollout,
    bool useKnots = true)
    = 0;

const TrajectoryRollout& getRolloutCache(
    std::shared_ptr<simulation::World> world, bool useKnots = true);

TrajectoryRollout& getGradientWrtRolloutCache(
    std::shared_ptr<simulation::World> world, bool useKnots = true);

/// This returns the concatenation of (start pos, start vel) for convenience
virtual Eigen::VectorXs getStartState() = 0;

/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
virtual Eigen::VectorXs getFinalState(std::shared_ptr<simulation::World> world)
    = 0;

int getNumSteps();

/// This returns the debugging name of a given DOF
virtual std::string getFlatDimName(int dim) = 0;

/// This gets the number of non-zero entries in the Jacobian
virtual int getNumberNonZeroJacobian();

/// This gets the structure of the non-zero entries in the Jacobian
virtual void getJacobianSparsityStructure(
    Eigen::Ref<Eigen::VectorXi> rows, Eigen::Ref<Eigen::VectorXi> cols);

/// This writes the Jacobian to a sparse vector
virtual void getSparseJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> sparse);

//////////////////////////////////////////////////////////////////////////////
// For Testing
//////////////////////////////////////////////////////////////////////////////

/// This computes finite difference Jacobians analagous to backpropJacobians()
void finiteDifferenceJacobian(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXs> jac);

/// This computes finite difference Jacobians analagous to
/// backpropGradient()
void finiteDifferenceGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXs> grad);

/// This computes the Jacobians that relate each timestep to the endpoint of
/// the trajectory. For a timestep at time t, this will relate quantities like
/// v_t -> p_end, for example.
TimestepJacobians backpropStartStateJacobians(
    std::shared_ptr<simulation::World> world);

/// This computes finite difference Jacobians analagous to
/// backpropStartStateJacobians()
TimestepJacobians finiteDifferenceStartStateJacobians(
    std::shared_ptr<simulation::World> world, s_t EPS);
*/

void Problem(py::module& m)
{
  ::py::class_<dart::trajectory::Problem>(m, "Problem")
      .def("setLoss", &dart::trajectory::Problem::setLoss, ::py::arg("loss"))
      .def(
          "setExploreAlternateStrategies",
          &dart::trajectory::Problem::setExploreAlternateStrategies,
          ::py::arg("flag"))
      .def(
          "getExploreAlternateStrategies",
          &dart::trajectory::Problem::getExploreAlternateStrategies)
      .def(
          "addConstraint",
          &dart::trajectory::Problem::addConstraint,
          ::py::arg("constraint"))
      .def(
          "pinForce",
          &dart::trajectory::Problem::pinForce,
          ::py::arg("time"),
          ::py::arg("value"))
      .def(
          "getPinnedForce",
          &dart::trajectory::Problem::getPinnedForce,
          ::py::arg("time"))
      .def(
          "addMapping",
          &dart::trajectory::Problem::addMapping,
          ::py::arg("key"),
          ::py::arg("mapping"))
      .def(
          "hasMapping",
          &dart::trajectory::Problem::hasMapping,
          ::py::arg("key"))
      .def(
          "getMapping",
          &dart::trajectory::Problem::getMapping,
          ::py::arg("key"))
      .def("getMappings", &dart::trajectory::Problem::getMappings)
      .def(
          "removeMapping",
          &dart::trajectory::Problem::removeMapping,
          ::py::arg("key"))
      .def(
          "getRepresentationStateSize",
          &dart::trajectory::Problem::getRepresentationStateSize)
      .def(
          "getFlatProblemDim",
          &dart::trajectory::Problem::getFlatProblemDim,
          ::py::arg("world"))
      .def("getConstraintDim", &dart::trajectory::Problem::getConstraintDim)
      .def("getStartState", &dart::trajectory::Problem::getStartState)
      .def(
          "getFinalState",
          &dart::trajectory::Problem::getFinalState,
          ::py::arg("world"),
          ::py::arg("perfLog") = nullptr)
      .def("getNumSteps", &dart::trajectory::Problem::getNumSteps)
      .def(
          "getFlatDimName",
          &dart::trajectory::Problem::getFlatDimName,
          ::py::arg("world"),
          ::py::arg("dim"))
      .def(
          "getLoss",
          &dart::trajectory::Problem::getLoss,
          ::py::arg("world"),
          ::py::arg("perfLog") = nullptr)
      .def(
          "getRolloutCache",
          &dart::trajectory::Problem::getRolloutCache,
          ::py::arg("world"),
          ::py::arg("perfLog") = nullptr,
          ::py::arg("useKnots") = true,
          ::py::return_value_policy::reference)
      .def(
          "setStates",
          &dart::trajectory::Problem::setStates,
          ::py::arg("world"),
          ::py::arg("rollout"),
          ::py::arg("perfLog") = nullptr)
      .def(
          "setControlForcesRaw",
          &dart::trajectory::Problem::setControlForcesRaw,
          ::py::arg("forces"),
          ::py::arg("perfLog") = nullptr)
      .def(
          "updateWithForces",
          &dart::trajectory::Problem::updateWithForces,
          ::py::arg("world"),
          ::py::arg("forces"),
          ::py::arg("perfLog") = nullptr);
  /*
.def(
  "getRepresentation",
  &dart::trajectory::Problem::getRepresentation)
.def(
  "flatten",
  &dart::trajectory::Problem::flatten,
  ::py::arg("flat"))
.def(
  "unflatten",
  &dart::trajectory::Problem::unflatten,
  ::py::arg("flat"))
.def(
  "getUpperBounds",
  &dart::trajectory::Problem::getUpperBounds,
  ::py::arg("world"),
  ::py::arg("flat"))
.def(
  "getLowerBounds",
  &dart::trajectory::Problem::getLowerBounds,
  ::py::arg("world"),
  ::py::arg("flat"))
.def(
  "getConstraintUpperBounds",
  &dart::trajectory::Problem::getConstraintUpperBounds,
  ::py::arg("flat"))
.def(
  "getConstraintLowerBounds",
  &dart::trajectory::Problem::getConstraintLowerBounds,
  ::py::arg("flat"))
.def(
  "getInitialGuess",
  &dart::trajectory::Problem::getInitialGuess,
  ::py::arg("world"),
  ::py::arg("flat"))
.def(
  "computeConstraints",
  &dart::trajectory::Problem::computeConstraints,
  ::py::arg("world"),
  ::py::arg("constraints"))
.def(
  "backpropJacobian",
  &dart::trajectory::Problem::backpropJacobian,
  ::py::arg("world"),
  ::py::arg("jac"))
.def(
  "backpropGradient",
  &dart::trajectory::Problem::backpropGradient,
  ::py::arg("world"),
  ::py::arg("grad"))
.def(
  "getLoss",
  &dart::trajectory::Problem::getLoss,
  ::py::arg("world"))
.def(
  "backpropGradientWrt",
  &dart::trajectory::Problem::backpropGradientWrt,
  ::py::arg("world"),
  ::py::arg("gradWrtRollout"),
  ::py::arg("grad"))
.def(
  "getStates",
  &dart::trajectory::Problem::getStates,
  ::py::arg("world"),
  ::py::arg("rollout"),
  ::py::arg("perfLog") = nullptr,
  ::py::arg("useKnots") = true)
.def(
  "getGradientWrtRolloutCache",
  &dart::trajectory::Problem::getGradientWrtRolloutCache,
  ::py::arg("world"),
  ::py::arg("useKnots") = true)
.def(
  "getNumberNonZeroJacobian",
  &dart::trajectory::Problem::getNumberNonZeroJacobian)
.def(
  "getJacobianSparsityStructure",
  &dart::trajectory::Problem::getJacobianSparsityStructure,
  ::py::arg("rows"),
  ::py::arg("cols"))
.def(
  "getSparseJacobian",
  &dart::trajectory::Problem::getSparseJacobian,
  ::py::arg("sparse"))
.def(
  "finiteDifferenceJacobian",
  &dart::trajectory::Problem::finiteDifferenceJacobian,
  ::py::arg("world"),
  ::py::arg("jac"))
.def(
  "finiteDifferenceGradient",
  &dart::trajectory::Problem::finiteDifferenceGradient,
  ::py::arg("world"),
  ::py::arg("grad"))
.def(
  "backpropStartStateJacobians",
  &dart::trajectory::Problem::backpropStartStateJacobians,
  ::py::arg("world"))
.def(
  "finiteDifferenceStartStateJacobians",
  &dart::trajectory::Problem::finiteDifferenceStartStateJacobians,
  ::py::arg("world"),
  ::py::arg("EPS"));
  */
}

} // namespace python
} // namespace dart
