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

#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/Frame.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace dart {
namespace python {

void Shape(py::module& sm);

void Entity(
    py::module& sm,
    ::py::class_<
        dart::dynamics::Entity,
        std::shared_ptr<dart::dynamics::Entity>>& entity);
void Frame(
    py::module& sm,
    ::py::class_<
        dart::dynamics::Frame,
        dart::dynamics::Entity,
        std::shared_ptr<dart::dynamics::Frame>>& frame);
void ShapeFrame(py::module& sm);
void SimpleFrame(py::module& sm);

void Node(py::module& sm);
void JacobianNode(py::module& sm);
void ShapeNode(py::module& sm);

void DegreeOfFreedom(py::module& sm);

void BodyNode(
    py::module& sm,
    ::py::class_<dart::dynamics::detail::BodyNodeAspectProperties>&
        bodyNodeAspectProps,
    ::py::class_<dart::dynamics::BodyNode::Properties>& bodyNodeProps,
    ::py::class_<
        dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>,
        dart::dynamics::JacobianNode,
        std::shared_ptr<
            dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>>>&
        templatedJacobianBodyNode,
    ::py::class_<
        dart::dynamics::BodyNode,
        dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>,
        dart::dynamics::Frame,
        std::shared_ptr<dart::dynamics::BodyNode>>& bodyNode);
void Inertia(py::module& sm);

void Joint(
    py::module& sm,
    ::py::class_<dart::dynamics::detail::JointProperties>& jointProps,
    ::py::class_<
        dart::dynamics::Joint,
        dart::common::Subject,
        dart::common::EmbedProperties<
            dart::dynamics::Joint,
            dart::dynamics::detail::JointProperties>,
        std::shared_ptr<dart::dynamics::Joint>>& joint);
void ZeroDofJoint(py::module& sm);
void WeldJoint(py::module& sm);
void GenericJoint(py::module& sm);
void RevoluteJoint(py::module& sm);
void PrismaticJoint(py::module& sm);
void ScrewJoint(py::module& sm);
void UniversalJoint(py::module& sm);
void TranslationalJoint2D(py::module& sm);
void PlanarJoint(py::module& sm);
void EulerJoint(py::module& sm);
void EulerFreeJoint(py::module& sm);
void ScapulathoracicJoint(py::module& sm);
void CustomJoint(py::module& sm);
void BallJoint(py::module& sm);
void TranslationalJoint(py::module& sm);
void FreeJoint(py::module& sm);

void MetaSkeleton(
    py::module& sm,
    ::py::class_<
        dart::dynamics::MetaSkeleton,
        std::shared_ptr<dart::dynamics::MetaSkeleton>>& metaSkeleton);
void ReferentialSkeleton(py::module& sm);
void Linkage(py::module& sm);
void Chain(py::module& sm);
void Skeleton(
    py::module& sm,
    ::py::class_<
        dart::dynamics::Skeleton,
        dart::dynamics::MetaSkeleton,
        std::shared_ptr<dart::dynamics::Skeleton>>& skeleton);

void dart_dynamics(py::module& m)
{
  auto sm = m.def_submodule("dynamics");

  Shape(sm);

  //////////////////////////////////////////////////////////////////////////////////
  auto entity = ::py::
      class_<dart::dynamics::Entity, std::shared_ptr<dart::dynamics::Entity>>(
          sm, "Entity");
  auto frame = ::py::class_<
      dart::dynamics::Frame,
      dart::dynamics::Entity,
      std::shared_ptr<dart::dynamics::Frame>>(sm, "Frame");
  //////////////////////////////////////////////////////////////////////////////////

  Entity(sm, entity);
  Frame(sm, frame);
  ShapeFrame(sm);
  SimpleFrame(sm);

  /////////////////////////////////////////////////////////////////////////////////
  // Predefine classes involved in circular references
  auto metaSkeleton = ::py::class_<
      dart::dynamics::MetaSkeleton,
      std::shared_ptr<dart::dynamics::MetaSkeleton>>(sm, "MetaSkeleton");

  auto skeleton = ::py::class_<
      dart::dynamics::Skeleton,
      dart::dynamics::MetaSkeleton,
      std::shared_ptr<dart::dynamics::Skeleton>>(sm, "Skeleton");
  /////////////////////////////////////////////////////////////////////////////////

  Node(sm);
  JacobianNode(sm);
  ShapeNode(sm);

  /////////////////////////////////////////////////////////////////////////////////
  // Predefine classes involved in circular references
  auto jointProps = ::py::class_<dart::dynamics::detail::JointProperties>(
      sm, "JointProperties");

  ::py::class_<
      dart::common::SpecializedForAspect<dart::common::EmbeddedPropertiesAspect<
          dart::dynamics::Joint,
          dart::dynamics::detail::JointProperties>>,
      dart::common::Composite,
      std::shared_ptr<dart::common::SpecializedForAspect<
          dart::common::EmbeddedPropertiesAspect<
              dart::dynamics::Joint,
              dart::dynamics::detail::JointProperties>>>>(
      sm, "SpecializedForAspect_EmbeddedPropertiesAspect_Joint_JointProperties")
      .def(::py::init<>());

  ::py::class_<
      dart::common::RequiresAspect<dart::common::EmbeddedPropertiesAspect<
          dart::dynamics::Joint,
          dart::dynamics::detail::JointProperties>>,
      dart::common::SpecializedForAspect<dart::common::EmbeddedPropertiesAspect<
          dart::dynamics::Joint,
          dart::dynamics::detail::JointProperties>>,
      std::shared_ptr<
          dart::common::RequiresAspect<dart::common::EmbeddedPropertiesAspect<
              dart::dynamics::Joint,
              dart::dynamics::detail::JointProperties>>>>(
      sm, "RequiresAspect_EmbeddedPropertiesAspect_Joint_JointProperties")
      .def(::py::init<>());

  ::py::class_<
      dart::common::EmbedProperties<
          dart::dynamics::Joint,
          dart::dynamics::detail::JointProperties>,
      dart::common::RequiresAspect<dart::common::EmbeddedPropertiesAspect<
          dart::dynamics::Joint,
          dart::dynamics::detail::JointProperties>>,
      std::shared_ptr<dart::common::EmbedProperties<
          dart::dynamics::Joint,
          dart::dynamics::detail::JointProperties>>>(
      sm, "EmbedProperties_Joint_JointProperties");

  auto joint = ::py::class_<
      dart::dynamics::Joint,
      dart::common::Subject,
      dart::common::EmbedProperties<
          dart::dynamics::Joint,
          dart::dynamics::detail::JointProperties>,
      std::shared_ptr<dart::dynamics::Joint>>(sm, "Joint");

  auto bodyNodeAspectProps
      = ::py::class_<dart::dynamics::detail::BodyNodeAspectProperties>(
          sm, "BodyNodeAspectProperties");

  auto bodyNodeProps = ::py::class_<dart::dynamics::BodyNode::Properties>(
      sm, "BodyNodeProperties");

  auto templatedJacobianBodyNode = ::py::class_<
      dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>,
      dart::dynamics::JacobianNode,
      std::shared_ptr<
          dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>>>(
      sm, "TemplatedJacobianBodyNode");

  auto bodyNode = ::py::class_<
      dart::dynamics::BodyNode,
      dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>,
      dart::dynamics::Frame,
      std::shared_ptr<dynamics::BodyNode>>(sm, "BodyNode");
  /////////////////////////////////////////////////////////////////////////////////

  DegreeOfFreedom(sm);

  Inertia(sm);

  Joint(sm, jointProps, joint);
  ZeroDofJoint(sm);
  WeldJoint(sm);
  GenericJoint(sm);
  RevoluteJoint(sm);
  PrismaticJoint(sm);
  ScrewJoint(sm);
  UniversalJoint(sm);
  PlanarJoint(sm);
  TranslationalJoint2D(sm);
  EulerJoint(sm);
  EulerFreeJoint(sm);
  ScapulathoracicJoint(sm);
  CustomJoint(sm);
  BallJoint(sm);
  TranslationalJoint(sm);
  FreeJoint(sm);

  BodyNode(
      sm,
      bodyNodeAspectProps,
      bodyNodeProps,
      templatedJacobianBodyNode,
      bodyNode);

  MetaSkeleton(sm, metaSkeleton);
  ReferentialSkeleton(sm);
  //   Linkage(sm);
  //   Chain(sm);
  Skeleton(sm, skeleton);
}

} // namespace python
} // namespace dart
