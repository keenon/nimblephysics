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

#include <dart/dart.hpp>
#include <dart/trajectory/AbstractShot.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

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
  virtual void flatten(Eigen::Ref<Eigen::VectorXd> flat) const = 0;

/// This gets the parameters out of a flat vector
virtual void unflatten(const Eigen::Ref<const Eigen::VectorXd>& flat) = 0;

/// This gets the fixed upper bounds for a flat vector, used during
/// optimization
virtual void getUpperBounds(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> flat) const = 0;

/// This gets the fixed lower bounds for a flat vector, used during
/// optimization
virtual void getLowerBounds(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> flat) const = 0;

/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
virtual void getConstraintUpperBounds(
    Eigen::Ref<Eigen::VectorXd> flat) const;

/// This gets the bounds on the constraint functions (both knot points and any
/// custom constraints)
virtual void getConstraintLowerBounds(
    Eigen::Ref<Eigen::VectorXd> flat) const;

/// This returns the initial guess for the values of X when running an
/// optimization
virtual void getInitialGuess(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> flat) const = 0;

/// This computes the values of the constraints
virtual void computeConstraints(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> constraints);

/// This computes the Jacobian that relates the flat problem to the end state.
/// This returns a matrix that's (getConstraintDim(), getFlatProblemDim()).
virtual void backpropJacobian(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::MatrixXd> jac);

/// This computes the gradient in the flat problem space, automatically
/// computing the gradients of the loss function as part of the call
void backpropGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> grad);

/// Get the loss for the rollout
double getLoss(std::shared_ptr<simulation::World> world);

/// This computes the gradient in the flat problem space, taking into accounts
/// incoming gradients with respect to any of the shot's values.
virtual void backpropGradientWrt(
    std::shared_ptr<simulation::World> world,
    const TrajectoryRollout& gradWrtRollout,
    Eigen::Ref<Eigen::VectorXd> grad)
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
virtual Eigen::VectorXd getStartState() = 0;

/// This unrolls the shot, and returns the (pos, vel) state concatenated at
/// the end of the shot
virtual Eigen::VectorXd getFinalState(std::shared_ptr<simulation::World> world)
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
    Eigen::Ref<Eigen::VectorXd> sparse);

//////////////////////////////////////////////////////////////////////////////
// For Testing
//////////////////////////////////////////////////////////////////////////////

/// This computes finite difference Jacobians analagous to backpropJacobians()
void finiteDifferenceJacobian(
    std::shared_ptr<simulation::World> world, Eigen::Ref<Eigen::MatrixXd> jac);

/// This computes finite difference Jacobians analagous to
/// backpropGradient()
void finiteDifferenceGradient(
    std::shared_ptr<simulation::World> world,
    Eigen::Ref<Eigen::VectorXd> grad);

/// This computes the Jacobians that relate each timestep to the endpoint of
/// the trajectory. For a timestep at time t, this will relate quantities like
/// v_t -> p_end, for example.
TimestepJacobians backpropStartStateJacobians(
    std::shared_ptr<simulation::World> world);

/// This computes finite difference Jacobians analagous to
/// backpropStartStateJacobians()
TimestepJacobians finiteDifferenceStartStateJacobians(
    std::shared_ptr<simulation::World> world, double EPS);
*/

void AbstractShot(py::module& m)
{
  ::py::class_<dart::trajectory::AbstractShot>(m, "AbstractShot")
      .def(
          "setLoss",
          &dart::trajectory::AbstractShot::setLoss,
          ::py::arg("loss"))
      .def(
          "addConstraint",
          &dart::trajectory::AbstractShot::addConstraint,
          ::py::arg("constraint"))
      .def(
          "switchRepresentationMapping",
          &dart::trajectory::AbstractShot::switchRepresentationMapping,
          ::py::arg("world"),
          ::py::arg("representation"))
      .def(
          "addMapping",
          &dart::trajectory::AbstractShot::addMapping,
          ::py::arg("key"),
          ::py::arg("mapping"))
      .def(
          "hasMapping",
          &dart::trajectory::AbstractShot::hasMapping,
          ::py::arg("key"))
      .def(
          "getMapping",
          &dart::trajectory::AbstractShot::getMapping,
          ::py::arg("key"))
      .def("getMappings", &dart::trajectory::AbstractShot::getMappings)
      .def(
          "removeMapping",
          &dart::trajectory::AbstractShot::removeMapping,
          ::py::arg("key"))
      .def(
          "getRepresentationStateSize",
          &dart::trajectory::AbstractShot::getRepresentationStateSize)
      .def(
          "getRepresentationName",
          &dart::trajectory::AbstractShot::getRepresentationName)
      .def(
          "getFlatProblemDim",
          &dart::trajectory::AbstractShot::getFlatProblemDim)
      .def(
          "getConstraintDim", &dart::trajectory::AbstractShot::getConstraintDim)
      .def("getStartState", &dart::trajectory::AbstractShot::getStartState)
      .def(
          "getFinalState",
          &dart::trajectory::AbstractShot::getFinalState,
          ::py::arg("world"))
      .def("getNumSteps", &dart::trajectory::AbstractShot::getNumSteps)
      .def(
          "getFlatDimName",
          &dart::trajectory::AbstractShot::getFlatDimName,
          ::py::arg("dim"))
      .def(
          "getLoss",
          &dart::trajectory::AbstractShot::getLoss,
          ::py::arg("world"))
      .def(
          "getRolloutCache",
          &dart::trajectory::AbstractShot::getRolloutCache,
          ::py::arg("world"),
          ::py::arg("useKnots") = true,
          ::py::return_value_policy::reference);
  /*
.def(
  "getRepresentation",
  &dart::trajectory::AbstractShot::getRepresentation)
.def(
  "flatten",
  &dart::trajectory::AbstractShot::flatten,
  ::py::arg("flat"))
.def(
  "unflatten",
  &dart::trajectory::AbstractShot::unflatten,
  ::py::arg("flat"))
.def(
  "getUpperBounds",
  &dart::trajectory::AbstractShot::getUpperBounds,
  ::py::arg("world"),
  ::py::arg("flat"))
.def(
  "getLowerBounds",
  &dart::trajectory::AbstractShot::getLowerBounds,
  ::py::arg("world"),
  ::py::arg("flat"))
.def(
  "getConstraintUpperBounds",
  &dart::trajectory::AbstractShot::getConstraintUpperBounds,
  ::py::arg("flat"))
.def(
  "getConstraintLowerBounds",
  &dart::trajectory::AbstractShot::getConstraintLowerBounds,
  ::py::arg("flat"))
.def(
  "getInitialGuess",
  &dart::trajectory::AbstractShot::getInitialGuess,
  ::py::arg("world"),
  ::py::arg("flat"))
.def(
  "computeConstraints",
  &dart::trajectory::AbstractShot::computeConstraints,
  ::py::arg("world"),
  ::py::arg("constraints"))
.def(
  "backpropJacobian",
  &dart::trajectory::AbstractShot::backpropJacobian,
  ::py::arg("world"),
  ::py::arg("jac"))
.def(
  "backpropGradient",
  &dart::trajectory::AbstractShot::backpropGradient,
  ::py::arg("world"),
  ::py::arg("grad"))
.def(
  "getLoss",
  &dart::trajectory::AbstractShot::getLoss,
  ::py::arg("world"))
.def(
  "backpropGradientWrt",
  &dart::trajectory::AbstractShot::backpropGradientWrt,
  ::py::arg("world"),
  ::py::arg("gradWrtRollout"),
  ::py::arg("grad"))
.def(
  "getStates",
  &dart::trajectory::AbstractShot::getStates,
  ::py::arg("world"),
  ::py::arg("rollout"),
  ::py::arg("useKnots") = true)
.def(
  "getRolloutCache",
  &dart::trajectory::AbstractShot::getRolloutCache,
  ::py::arg("world"),
  ::py::arg("useKnots") = true)
.def(
  "getGradientWrtRolloutCache",
  &dart::trajectory::AbstractShot::getGradientWrtRolloutCache,
  ::py::arg("world"),
  ::py::arg("useKnots") = true)
.def(
  "getNumberNonZeroJacobian",
  &dart::trajectory::AbstractShot::getNumberNonZeroJacobian)
.def(
  "getJacobianSparsityStructure",
  &dart::trajectory::AbstractShot::getJacobianSparsityStructure,
  ::py::arg("rows"),
  ::py::arg("cols"))
.def(
  "getSparseJacobian",
  &dart::trajectory::AbstractShot::getSparseJacobian,
  ::py::arg("sparse"))
.def(
  "finiteDifferenceJacobian",
  &dart::trajectory::AbstractShot::finiteDifferenceJacobian,
  ::py::arg("world"),
  ::py::arg("jac"))
.def(
  "finiteDifferenceGradient",
  &dart::trajectory::AbstractShot::finiteDifferenceGradient,
  ::py::arg("world"),
  ::py::arg("grad"))
.def(
  "backpropStartStateJacobians",
  &dart::trajectory::AbstractShot::backpropStartStateJacobians,
  ::py::arg("world"))
.def(
  "finiteDifferenceStartStateJacobians",
  &dart::trajectory::AbstractShot::finiteDifferenceStartStateJacobians,
  ::py::arg("world"),
  ::py::arg("EPS"));
  */
}

} // namespace python
} // namespace dart
