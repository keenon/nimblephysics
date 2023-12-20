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

#include <memory>

#include <dart/exo/ExoSolverPinnedContact.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void ExoSolverPinnedContact(py::module& m)
{
  ::py::class_<dart::exo::ExoSolverPinnedContact>(m, "ExoSolverPinnedContact")
      .def(
          ::py::init<
              std::shared_ptr<dynamics::Skeleton>,
              std::shared_ptr<dynamics::Skeleton>>(),
          ::py::arg("realSkel"),
          ::py::arg("virtualSkel"),
          "Both the real and virtual skeletons must be identical in their "
          "number of DOFs and their structure (names of bodies, etc). The key "
          "difference is the masses, inertias, gravity, and spring forces of "
          "the virtual skeleton.")
      .def(
          "addMotorDof",
          &dart::exo::ExoSolverPinnedContact::addMotorDof,
          ::py::arg("dofIndex"))
      .def(
          "setPositions",
          &dart::exo::ExoSolverPinnedContact::setPositions,
          ::py::arg("q"))
      .def(
          "getExoToJointTorquesJacobian",
          &dart::exo::ExoSolverPinnedContact::getExoToJointTorquesJacobian)
      .def(
          "setContactPins",
          &dart::exo::ExoSolverPinnedContact::setContactPins,
          ::py::arg("pins"),
          "Set the contact points that we will use when solving inverse "
          "dynamics.")
      .def(
          "getContactJacobian",
          &dart::exo::ExoSolverPinnedContact::getContactJacobian,
          "Get the Jacobian relating world space velocity of the contact "
          "points to joint velocities.")
      .def(
          "finiteDifferenceContactJacobian",
          &dart::exo::ExoSolverPinnedContact::finiteDifferenceContactJacobian,
          "This is only used for testing: Get the Jacobian relating world "
          "space velocity of the contact points to joint velocities, by finite "
          "differencing.")
      .def(
          "analyticalForwardDynamics",
          &dart::exo::ExoSolverPinnedContact::analyticalForwardDynamics,
          ::py::arg("dq"),
          ::py::arg("tau"),
          ::py::arg("exoTorques"),
          ::py::arg("contactForces"),
          "This is only used for testing, to allow us to compare the "
          "analytical solution to the numerical solution.")
      .def(
          "implicitForwardDynamics",
          &dart::exo::ExoSolverPinnedContact::implicitForwardDynamics,
          ::py::arg("dq"),
          ::py::arg("tau"),
          ::py::arg("exoTorques"),
          ::py::arg("contactForces"),
          "This is only used for testing, to allow us to compare the "
          "analytical solution to the numerical solution.")
      .def(
          "estimateHumanTorques",
          &dart::exo::ExoSolverPinnedContact::estimateHumanTorques,
          ::py::arg("dq"),
          ::py::arg("ddq"),
          ::py::arg("contactForces"),
          ::py::arg("lastExoTorques"),
          "This is part of the main exoskeleton solver. It takes in the "
          "current joint velocities and accelerations, and the last "
          "exoskeleton torques, and returns the estimated human pilot joint "
          "torques.")
      .def(
          "getPinnedVirtualDynamics",
          &dart::exo::ExoSolverPinnedContact::getPinnedVirtualDynamics,
          ::py::arg("dq"),
          ::py::arg("tau"),
          "This is part of the main exoskeleton solver. It takes in the "
          "current estimated human pilot joint torques, and computes the "
          "accelerations we would see on the virtual skeleton if we applied "
          "those same torques, with the contacts pinned at the CoPs.")
      .def(
          "getPinnedVirtualDynamicsLinearMap",
          &dart::exo::ExoSolverPinnedContact::getPinnedVirtualDynamicsLinearMap,
          ::py::arg("dq"),
          "This does the same thing as getPinndVirtualDynamics, but returns "
          "the Ax + b values A and b such that Ax + b = ddq, accounting for "
          "the pin constraints.")
      .def(
          "getPinnedRealDynamics",
          &dart::exo::ExoSolverPinnedContact::getPinnedRealDynamics,
          ::py::arg("dq"),
          ::py::arg("tau"),
          "This is not part of the main exoskeleton solver, but is useful for "
          "the inverse problem of analyzing the human pilot's joint torques "
          "under different assistance strategies.")
      .def(
          "getPinnedRealDynamicsLinearMap",
          &dart::exo::ExoSolverPinnedContact::getPinnedRealDynamicsLinearMap,
          ::py::arg("dq"),
          "This does the same thing as getPinndRealDynamics, but returns the "
          "Ax + b values A and b such that Ax + b = ddq, accounting for the "
          "pin constraints.")
      .def(
          "getPinnedTotalTorques",
          &dart::exo::ExoSolverPinnedContact::getPinnedTotalTorques,
          ::py::arg("dq"),
          ::py::arg("ddqDesired"),
          "This is part of the main exoskeleton solver. It takes in how the "
          "digital twin of the exo pilot is accelerating, and attempts to "
          "solve for the torques that the exo needs to apply to get as close "
          "to that as possible.")
      .def(
          "getPinnedTotalTorquesLinearMap",
          &dart::exo::ExoSolverPinnedContact::getPinnedTotalTorquesLinearMap,
          ::py::arg("dq"),
          "This does the same thing as getPinnedTotalTorques, but returns the "
          "Ax + b values A and b such that Ax + b = tau, accounting for the "
          "pin constraints.")
      .def(
          "projectTorquesToExoControlSpace",
          &dart::exo::ExoSolverPinnedContact::projectTorquesToExoControlSpace,
          ::py::arg("torques"),
          "This is part of the main exoskeleton solver. It takes in the "
          "desired torques for the exoskeleton, and returns the torques on the "
          "actuated DOFs that can be used to drive the exoskeleton.")
      .def(
          "projectTorquesToExoControlSpaceLinearMap",
          &dart::exo::ExoSolverPinnedContact::
              projectTorquesToExoControlSpaceLinearMap,
          "This does the same thing as projectTorquesToExoControlSpace, but "
          "returns the matrix to multiply by the torques to get the exo "
          "torques.")
      .def(
          "solveFromAccelerations",
          &dart::exo::ExoSolverPinnedContact::solveFromAccelerations,
          ::py::arg("dq"),
          ::py::arg("ddq"),
          ::py::arg("lastExoTorques"),
          ::py::arg("contactForces"),
          "This runs the entire exoskeleton solver pipeline, spitting out the "
          "torques to apply to the exoskeleton actuators.")
      .def(
          "solveFromBiologicalTorques",
          &dart::exo::ExoSolverPinnedContact::solveFromBiologicalTorques,
          ::py::arg("dq"),
          ::py::arg("tau"),
          "This is a subset of the steps in solveFromAccelerations, which can "
          "take the biological joint torques directly, and solve for the exo "
          "torques.")
      .def(
          "getExoTorquesLinearMap",
          &dart::exo::ExoSolverPinnedContact::getExoTorquesLinearMap,
          ::py::arg("dq"),
          "This is the same as solveFromBiologicalTorques, but returns the Ax "
          "+ b values A and b such that Ax + b = exo_tau, accounting for the "
          "pin constraints.")
      .def(
          "getPinnedForwardDynamicsForExoAndHuman",
          &dart::exo::ExoSolverPinnedContact::
              getPinnedForwardDynamicsForExoAndHuman,
          ::py::arg("dq"),
          ::py::arg("humanTau"),
          "This does a simple forward dynamics step, given the current human "
          "joint torques, factoring in how the exoskeleton will respond to "
          "those torques. This returns the `ddq` that we would see on the "
          "human, and the contact forces we see at the pin constraints, and "
          "the exo torques we would get.")
      .def(
          "getPinnedForwardDynamicsForExoAndHumanLinearMap",
          &dart::exo::ExoSolverPinnedContact::
              getPinnedForwardDynamicsForExoAndHumanLinearMap,
          ::py::arg("dq"),
          "This does the same thing as getPinnedForwardDynamicsForExoAndHuman, "
          "but returns the Ax + b values A and b such that Ax + b = ddq, "
          "accounting for the pin constraints.")
      .def(
          "getHumanAndExoTorques",
          &dart::exo::ExoSolverPinnedContact::getHumanAndExoTorques,
          ::py::arg("dq"),
          ::py::arg("ddq"),
          "Given the desired end-kinematics, after the human and exoskeleton "
          "have finished \"negotiating\" how they will collaborate, this "
          "computes the resulting human and exoskeleton torques.");
}

} // namespace python
} // namespace dart
