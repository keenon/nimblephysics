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

#include <dart/dynamics/BallJoint.hpp>
#include <dart/dynamics/BodyNode.hpp>
#include <dart/dynamics/EulerJoint.hpp>
#include <dart/dynamics/FreeJoint.hpp>
#include <dart/dynamics/PlanarJoint.hpp>
#include <dart/dynamics/PrismaticJoint.hpp>
#include <dart/dynamics/RevoluteJoint.hpp>
#include <dart/dynamics/ScrewJoint.hpp>
#include <dart/dynamics/TranslationalJoint.hpp>
#include <dart/dynamics/TranslationalJoint2D.hpp>
#include <dart/dynamics/UniversalJoint.hpp>
#include <dart/dynamics/WeldJoint.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "eigen_geometry_pybind.h"
#include "eigen_pybind.h"

PYBIND11_DECLARE_HOLDER_TYPE(T, dart::dynamics::TemplateBodyNodePtr<T>, true);

namespace py = pybind11;

#define DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(joint_type)        \
  .def(                                                                        \
      "create" #joint_type "AndBodyNodePair",                                  \
      +[](dart::dynamics::BodyNode* self)                                      \
          -> std::                                                             \
              pair<dart::dynamics::joint_type*, dart::dynamics::BodyNode*> {   \
                return self->createChildJointAndBodyNodePair<                  \
                    dart::dynamics::joint_type,                                \
                    dart::dynamics::BodyNode>();                               \
              },                                                               \
      ::py::return_value_policy::reference_internal)                           \
      .def(                                                                    \
          "create" #joint_type "AndBodyNodePair",                              \
          +[](dart::dynamics::BodyNode* self,                                  \
              const dart::dynamics::joint_type::Properties& jointProperties)   \
              -> std::pair<                                                    \
                  dart::dynamics::joint_type*,                                 \
                  dart::dynamics::BodyNode*> {                                 \
            return self->createChildJointAndBodyNodePair<                      \
                dart::dynamics::joint_type,                                    \
                dart::dynamics::BodyNode>(jointProperties);                    \
          },                                                                   \
          ::py::return_value_policy::reference_internal,                       \
          ::py::arg("jointProperties"))                                        \
      .def(                                                                    \
          "create" #joint_type "AndBodyNodePair",                              \
          +[](dart::dynamics::BodyNode* self,                                  \
              const dart::dynamics::joint_type::Properties& jointProperties,   \
              const dart::dynamics::BodyNode::Properties& bodyProperties)      \
              -> std::pair<                                                    \
                  dart::dynamics::joint_type*,                                 \
                  dart::dynamics::BodyNode*> {                                 \
            return self->createChildJointAndBodyNodePair<                      \
                dart::dynamics::joint_type,                                    \
                dart::dynamics::BodyNode>(jointProperties, bodyProperties);    \
          },                                                                   \
          ::py::return_value_policy::reference_internal,                       \
          ::py::arg("jointProperties"),                                        \
          ::py::arg("bodyProperties"))

namespace dart {
namespace python {

void BodyNode(py::module& m)
{
  ::py::class_<dart::dynamics::detail::BodyNodeAspectProperties>(
      m, "BodyNodeAspectProperties")
      .def(::py::init<>())
      .def(::py::init<const std::string&>(), ::py::arg("name"))
      .def(
          ::py::init<const std::string&, const dart::dynamics::Inertia&>(),
          ::py::arg("name"),
          ::py::arg("inertia"))
      .def(
          ::py::
              init<const std::string&, const dart::dynamics::Inertia&, bool>(),
          ::py::arg("name"),
          ::py::arg("inertia"),
          ::py::arg("isCollidable"))
      .def(
          ::py::init<
              const std::string&,
              const dart::dynamics::Inertia&,
              bool,
              s_t>(),
          ::py::arg("name"),
          ::py::arg("inertia"),
          ::py::arg("isCollidable"),
          ::py::arg("frictionCoeff"))
      .def(
          ::py::init<
              const std::string&,
              const dart::dynamics::Inertia&,
              bool,
              s_t,
              s_t>(),
          ::py::arg("name"),
          ::py::arg("inertia"),
          ::py::arg("isCollidable"),
          ::py::arg("frictionCoeff"),
          ::py::arg("restitutionCoeff"))
      .def(
          ::py::init<
              const std::string&,
              const dart::dynamics::Inertia&,
              bool,
              s_t,
              s_t,
              bool>(),
          ::py::arg("name"),
          ::py::arg("inertia"),
          ::py::arg("isCollidable"),
          ::py::arg("frictionCoeff"),
          ::py::arg("restitutionCoeff"),
          ::py::arg("gravityMode"))
      .def_readwrite(
          "mName", &dart::dynamics::detail::BodyNodeAspectProperties::mName)
      .def_readwrite(
          "mInertia",
          &dart::dynamics::detail::BodyNodeAspectProperties::mInertia)
      .def_readwrite(
          "mIsCollidable",
          &dart::dynamics::detail::BodyNodeAspectProperties::mIsCollidable)
      .def_readwrite(
          "mFrictionCoeff",
          &dart::dynamics::detail::BodyNodeAspectProperties::mFrictionCoeff)
      .def_readwrite(
          "mRestitutionCoeff",
          &dart::dynamics::detail::BodyNodeAspectProperties::mRestitutionCoeff)
      .def_readwrite(
          "mGravityMode",
          &dart::dynamics::detail::BodyNodeAspectProperties::mGravityMode);

  ::py::class_<dart::dynamics::BodyNode::Properties>(m, "BodyNodeProperties")
      .def(::py::init<>())
      .def(
          ::py::init<const dart::dynamics::detail::BodyNodeAspectProperties&>(),
          ::py::arg("aspectProperties"));

  ::py::class_<dart::dynamics::BodyNode::MovingVertex>(m, "MovingVertex")
      .def(
          ::py::init<
              Eigen::Vector3s,
              Eigen::Vector3s,
              Eigen::Vector3s,
              dart::dynamics::BodyNode*,
              int>(),
          ::py::arg("pos"),
          ::py::arg("vel"),
          ::py::arg("accel"),
          ::py::arg("bodyNode"),
          ::py::arg("timestep"))
      .def_readwrite("pos", &dart::dynamics::BodyNode::MovingVertex::pos)
      .def_readwrite("vel", &dart::dynamics::BodyNode::MovingVertex::vel)
      .def_readwrite("accel", &dart::dynamics::BodyNode::MovingVertex::accel)
      .def_readwrite(
          "bodyNode", &dart::dynamics::BodyNode::MovingVertex::bodyNode)
      .def_readwrite(
          "timestep", &dart::dynamics::BodyNode::MovingVertex::timestep);

  ::py::class_<
      dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>,
      dart::dynamics::JacobianNode,
      std::shared_ptr<
          dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>>>(
      m, "TemplatedJacobianBodyNode")
      .def(
          "getJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobian(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset) -> dart::math::Jacobian {
            return self->getJacobian(_offset);
          },
          ::py::arg("offset"))
      .def(
          "getJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobian(_offset, _inCoordinatesOf);
          },
          ::py::arg("offset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getWorldJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset) -> dart::math::Jacobian {
            return self->getWorldJacobian(_offset);
          },
          ::py::arg("offset"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
               dart::dynamics::BodyNode>* self) -> dart::math::LinearJacobian {
            return self->getLinearJacobian();
          })
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobian(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset) -> dart::math::LinearJacobian {
            return self->getLinearJacobian(_offset);
          },
          ::py::arg("offset"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobian(_offset, _inCoordinatesOf);
          },
          ::py::arg("offset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getAngularJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
               dart::dynamics::BodyNode>* self) -> dart::math::AngularJacobian {
            return self->getAngularJacobian();
          })
      .def(
          "getAngularJacobian",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::AngularJacobian {
            return self->getAngularJacobian(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobianSpatialDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianSpatialDeriv(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobianSpatialDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset) -> dart::math::Jacobian {
            return self->getJacobianSpatialDeriv(_offset);
          },
          ::py::arg("offset"))
      .def(
          "getJacobianSpatialDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianSpatialDeriv(_offset, _inCoordinatesOf);
          },
          ::py::arg("offset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobianClassicDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianClassicDeriv(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobianClassicDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset) -> dart::math::Jacobian {
            return self->getJacobianClassicDeriv(_offset);
          },
          ::py::arg("offset"))
      .def(
          "getJacobianClassicDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianClassicDeriv(_offset, _inCoordinatesOf);
          },
          ::py::arg("offset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
               dart::dynamics::BodyNode>* self) -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv();
          })
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset) -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv(_offset);
          },
          ::py::arg("offset"))
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const Eigen::Vector3s& _offset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv(_offset, _inCoordinatesOf);
          },
          ::py::arg("offset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getAngularJacobianDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
               dart::dynamics::BodyNode>* self) -> dart::math::AngularJacobian {
            return self->getAngularJacobianDeriv();
          })
      .def(
          "getAngularJacobianDeriv",
          +[](const dart::dynamics::TemplatedJacobianNode<
                  dart::dynamics::BodyNode>* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::AngularJacobian {
            return self->getAngularJacobianDeriv(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"));

  ::py::class_<
      dart::dynamics::BodyNode,
      dart::dynamics::TemplatedJacobianNode<dart::dynamics::BodyNode>,
      dart::dynamics::Frame,
      dart::dynamics::BodyNodePtr>(m, "BodyNode")
      .def(
          "setAllNodeStates",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode::AllNodeStates& states) {
            self->setAllNodeStates(states);
          },
          ::py::arg("states"))
      .def(
          "getAllNodeStates",
          +[](const dart::dynamics::BodyNode* self)
              -> dart::dynamics::BodyNode::AllNodeStates {
            return self->getAllNodeStates();
          })
      .def(
          "setAllNodeProperties",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode::AllNodeProperties& properties) {
            self->setAllNodeProperties(properties);
          },
          ::py::arg("properties"))
      .def(
          "getAllNodeProperties",
          +[](const dart::dynamics::BodyNode* self)
              -> dart::dynamics::BodyNode::AllNodeProperties {
            return self->getAllNodeProperties();
          })
      .def(
          "setProperties",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode::CompositeProperties&
                  _properties) { self->setProperties(_properties); },
          ::py::arg("properties"))
      .def(
          "setProperties",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode::AspectProperties& _properties) {
            self->setProperties(_properties);
          },
          ::py::arg("properties"))
      .def(
          "setAspectState",
          +[](dart::dynamics::BodyNode* self,
              const dart::common::EmbedStateAndPropertiesOnTopOf<
                  dart::dynamics::BodyNode,
                  dart::dynamics::detail::BodyNodeState,
                  dart::dynamics::detail::BodyNodeAspectProperties,
                  dart::common::RequiresAspect<
                      dart::common::ProxyStateAndPropertiesAspect<
                          dart::dynamics::BodyNode,
                          dart::common::ProxyCloneable<
                              dart::common::Aspect::State,
                              dart::dynamics::BodyNode,
                              dart::common::CloneableMap<std::map<
                                  std::type_index,
                                  std::unique_ptr<
                                      dart::common::CloneableVector<
                                          std::unique_ptr<
                                              dart::dynamics::Node::State,
                                              std::default_delete<
                                                  dart::dynamics::Node::
                                                      State>>>,
                                      std::default_delete<
                                          dart::common::CloneableVector<
                                              std::unique_ptr<
                                                  dart::dynamics::Node::State,
                                                  std::default_delete<
                                                      dart::dynamics::Node::
                                                          State>>>>>,
                                  std::less<std::type_index>,
                                  std::allocator<std::pair<
                                      const std::type_index,
                                      std::unique_ptr<
                                          dart::common::CloneableVector<
                                              std::unique_ptr<
                                                  dart::dynamics::Node::State,
                                                  std::default_delete<
                                                      dart::dynamics::Node::
                                                          State>>>,
                                          std::default_delete<
                                              dart::common::CloneableVector<
                                                  std::unique_ptr<
                                                      dart::dynamics::Node::
                                                          State,
                                                      std::default_delete<
                                                          dart::dynamics::Node::
                                                              State>>>>>>>>>,
                              &dart::dynamics::detail::setAllNodeStates,
                              &dart::dynamics::detail::getAllNodeStates>,
                          dart::common::ProxyCloneable<
                              dart::common::Aspect::Properties,
                              dart::dynamics::BodyNode,
                              dart::common::CloneableMap<std::map<
                                  std::type_index,
                                  std::unique_ptr<
                                      dart::common::CloneableVector<
                                          std::unique_ptr<
                                              dart::dynamics::Node::Properties,
                                              std::default_delete<
                                                  dart::dynamics::Node::
                                                      Properties>>>,
                                      std::default_delete<
                                          dart::common::CloneableVector<
                                              std::unique_ptr<
                                                  dart::dynamics::Node::
                                                      Properties,
                                                  std::default_delete<
                                                      dart::dynamics::Node::
                                                          Properties>>>>>,
                                  std::less<std::type_index>,
                                  std::allocator<std::pair<
                                      const std::type_index,
                                      std::unique_ptr<
                                          dart::common::CloneableVector<
                                              std::unique_ptr<
                                                  dart::dynamics::Node::
                                                      Properties,
                                                  std::default_delete<
                                                      dart::dynamics::Node::
                                                          Properties>>>,
                                          std::default_delete<
                                              dart::common::CloneableVector<
                                                  std::unique_ptr<
                                                      dart::dynamics::Node::
                                                          Properties,
                                                      std::default_delete<
                                                          dart::dynamics::Node::
                                                              Properties>>>>>>>>>,
                              &dart::dynamics::detail::setAllNodeProperties,
                              &dart::dynamics::detail::
                                  getAllNodeProperties>>>>::AspectState&
                  state) { self->setAspectState(state); },
          ::py::arg("state"))
      .def(
          "setAspectProperties",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode::AspectProperties& properties) {
            self->setAspectProperties(properties);
          },
          ::py::arg("properties"))
      .def(
          "getBodyNodeProperties",
          +[](const dart::dynamics::BodyNode* self)
              -> dart::dynamics::BodyNode::Properties {
            return self->getBodyNodeProperties();
          })
      .def(
          "copy",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode& otherBodyNode) {
            self->copy(otherBodyNode);
          },
          ::py::arg("otherBodyNode"))
      .def(
          "copy",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode* otherBodyNode) {
            self->copy(otherBodyNode);
          },
          ::py::arg("otherBodyNode"))
      .def(
          "duplicateNodes",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode* otherBodyNode) {
            self->duplicateNodes(otherBodyNode);
          },
          ::py::arg("otherBodyNode"))
      .def(
          "matchNodes",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::BodyNode* otherBodyNode) {
            self->matchNodes(otherBodyNode);
          },
          ::py::arg("otherBodyNode"))
      .def(
          "setName",
          +[](dart::dynamics::BodyNode* self, const std::string& _name)
              -> const std::string& { return self->setName(_name); },
          ::py::return_value_policy::reference_internal,
          ::py::arg("name"))
      .def(
          "getName",
          +[](const dart::dynamics::BodyNode* self) -> const std::string& {
            return self->getName();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "setGravityMode",
          +[](dart::dynamics::BodyNode* self, bool _gravityMode) {
            self->setGravityMode(_gravityMode);
          },
          ::py::arg("gravityMode"))
      .def(
          "getGravityMode",
          +[](const dart::dynamics::BodyNode* self)
              -> bool { return self->getGravityMode(); })
      .def(
          "isCollidable",
          +[](const dart::dynamics::BodyNode* self)
              -> bool { return self->isCollidable(); })
      .def(
          "setCollidable",
          +[](dart::dynamics::BodyNode* self, bool _isCollidable) {
            self->setCollidable(_isCollidable);
          },
          ::py::arg("isCollidable"))
      .def(
          "setScale",
          &dart::dynamics::BodyNode::setScale,
          ::py::arg("scale"),
          ::py::arg("silentlyClamp") = false)
      .def("getScale", &dart::dynamics::BodyNode::getScale)
      .def(
          "setScaleLowerBound",
          &dart::dynamics::BodyNode::setScaleLowerBound,
          ::py::arg("scale"))
      .def("getScaleLowerBound", &dart::dynamics::BodyNode::getScaleLowerBound)
      .def(
          "setScaleUpperBound",
          &dart::dynamics::BodyNode::setScaleUpperBound,
          ::py::arg("scale"))
      .def("getScaleUpperBound", &dart::dynamics::BodyNode::getScaleUpperBound)
      .def(
          "setMass",
          +[](dart::dynamics::BodyNode* self,
              s_t mass) { self->setMass(mass); },
          ::py::arg("mass"))
      .def(
          "getMass",
          +[](const dart::dynamics::BodyNode* self)
              -> s_t { return self->getMass(); })
      .def(
          "setMomentOfInertia",
          +[](dart::dynamics::BodyNode* self, s_t _Ixx, s_t _Iyy, s_t _Izz) {
            self->setMomentOfInertia(_Ixx, _Iyy, _Izz);
          },
          ::py::arg("Ixx"),
          ::py::arg("Iyy"),
          ::py::arg("Izz"))
      .def(
          "setMomentOfInertia",
          +[](dart::dynamics::BodyNode* self,
              s_t _Ixx,
              s_t _Iyy,
              s_t _Izz,
              s_t _Ixy) { self->setMomentOfInertia(_Ixx, _Iyy, _Izz, _Ixy); },
          ::py::arg("Ixx"),
          ::py::arg("Iyy"),
          ::py::arg("Izz"),
          ::py::arg("Ixy"))
      .def(
          "setMomentOfInertia",
          +[](dart::dynamics::BodyNode* self,
              s_t _Ixx,
              s_t _Iyy,
              s_t _Izz,
              s_t _Ixy,
              s_t _Ixz) {
            self->setMomentOfInertia(_Ixx, _Iyy, _Izz, _Ixy, _Ixz);
          },
          ::py::arg("Ixx"),
          ::py::arg("Iyy"),
          ::py::arg("Izz"),
          ::py::arg("Ixy"),
          ::py::arg("Ixz"))
      .def(
          "setMomentOfInertia",
          +[](dart::dynamics::BodyNode* self,
              s_t _Ixx,
              s_t _Iyy,
              s_t _Izz,
              s_t _Ixy,
              s_t _Ixz,
              s_t _Iyz) {
            self->setMomentOfInertia(_Ixx, _Iyy, _Izz, _Ixy, _Ixz, _Iyz);
          },
          ::py::arg("Ixx"),
          ::py::arg("Iyy"),
          ::py::arg("Izz"),
          ::py::arg("Ixy"),
          ::py::arg("Ixz"),
          ::py::arg("Iyz"))
      .def(
          "getMomentOfInertia",
          +[](const dart::dynamics::BodyNode* self,
              s_t& _Ixx,
              s_t& _Iyy,
              s_t& _Izz,
              s_t& _Ixy,
              s_t& _Ixz,
              s_t& _Iyz) {
            self->getMomentOfInertia(_Ixx, _Iyy, _Izz, _Ixy, _Ixz, _Iyz);
          },
          ::py::arg("Ixx"),
          ::py::arg("Iyy"),
          ::py::arg("Izz"),
          ::py::arg("Ixy"),
          ::py::arg("Ixz"),
          ::py::arg("Iyz"))
      .def(
          "setInertia",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::Inertia& inertia) {
            self->setInertia(inertia);
          },
          ::py::arg("inertia"))
      .def(
          "setLocalCOM",
          +[](dart::dynamics::BodyNode* self, const Eigen::Vector3s& _com) {
            self->setLocalCOM(_com);
          },
          ::py::arg("com"))
      .def(
          "getLocalCOM",
          +[](const dart::dynamics::BodyNode* self) -> const Eigen::Vector3s& {
            return self->getLocalCOM();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "getCOM",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector3s {
            return self->getCOM();
          })
      .def(
          "getCOM",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::Frame* _withRespectTo) -> Eigen::Vector3s {
            return self->getCOM(_withRespectTo);
          },
          ::py::arg("withRespectTo"))
      .def(
          "getCOMLinearVelocity",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector3s {
            return self->getCOMLinearVelocity();
          })
      .def(
          "getCOMLinearVelocity",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::Frame* _relativeTo) -> Eigen::Vector3s {
            return self->getCOMLinearVelocity(_relativeTo);
          },
          ::py::arg("relativeTo"))
      .def(
          "getCOMLinearVelocity",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector3s {
            return self->getCOMLinearVelocity(_relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMSpatialVelocity",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector6s {
            return self->getCOMSpatialVelocity();
          })
      .def(
          "getCOMSpatialVelocity",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector6s {
            return self->getCOMSpatialVelocity(_relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMLinearAcceleration",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector3s {
            return self->getCOMLinearAcceleration();
          })
      .def(
          "getCOMLinearAcceleration",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::Frame* _relativeTo) -> Eigen::Vector3s {
            return self->getCOMLinearAcceleration(_relativeTo);
          },
          ::py::arg("relativeTo"))
      .def(
          "getCOMLinearAcceleration",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector3s {
            return self->getCOMLinearAcceleration(
                _relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMSpatialAcceleration",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector6s {
            return self->getCOMSpatialAcceleration();
          })
      .def(
          "getCOMSpatialAcceleration",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector6s {
            return self->getCOMSpatialAcceleration(
                _relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "setFrictionCoeff",
          +[](dart::dynamics::BodyNode* self,
              s_t _coeff) { self->setFrictionCoeff(_coeff); },
          ::py::arg("coeff"))
      .def(
          "getFrictionCoeff",
          +[](const dart::dynamics::BodyNode* self)
              -> s_t { return self->getFrictionCoeff(); })
      .def(
          "setRestitutionCoeff",
          +[](dart::dynamics::BodyNode* self, s_t _coeff) {
            self->setRestitutionCoeff(_coeff);
          },
          ::py::arg("coeff"))
      .def(
          "getRestitutionCoeff",
          +[](const dart::dynamics::BodyNode* self) -> s_t {
            return self->getRestitutionCoeff();
          })
      .def(
          "getClosestVerticesToMarker",
          &dynamics::BodyNode::getClosestVerticesToMarker,
          ::py::arg("marker"))
      .def(
          "getDistToClosestVerticesToMarker",
          &dynamics::BodyNode::getDistToClosestVerticesToMarker,
          ::py::arg("marker"))
      .def(
          "getGradientOfDistToClosestVerticesToMarkerWrtMarker",
          &dynamics::BodyNode::
              getGradientOfDistToClosestVerticesToMarkerWrtMarker,
          ::py::arg("marker"))
      .def(
          "getGradientOfDistToClosestVerticesToMarkerWrtBodyScale",
          &dynamics::BodyNode::
              getGradientOfDistToClosestVerticesToMarkerWrtBodyScale,
          ::py::arg("marker"))
      .def(
          "getIndexInSkeleton",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getIndexInSkeleton();
          })
      .def(
          "getIndexInTree",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getIndexInTree();
          })
      .def(
          "getTreeIndex",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getTreeIndex();
          })
      .def(
          "remove",
          +[](dart::dynamics::BodyNode* self) -> dart::dynamics::SkeletonPtr {
            return self->remove();
          })
      .def(
          "remove",
          +[](dart::dynamics::BodyNode* self, const std::string& _name)
              -> dart::dynamics::SkeletonPtr { return self->remove(_name); },
          ::py::arg("name"))
      .def(
          "moveTo",
          +[](dart::dynamics::BodyNode* self,
              dart::dynamics::BodyNode* _newParent) -> bool {
            return self->moveTo(_newParent);
          },
          ::py::arg("newParent"))
      .def(
          "moveTo",
          +[](dart::dynamics::BodyNode* self,
              const dart::dynamics::SkeletonPtr& _newSkeleton,
              dart::dynamics::BodyNode* _newParent) -> bool {
            return self->moveTo(_newSkeleton, _newParent);
          },
          ::py::arg("newSkeleton"),
          ::py::arg("newParent"))
      .def(
          "split",
          +[](dart::dynamics::BodyNode* self,
              const std::string& _skeletonName) -> dart::dynamics::SkeletonPtr {
            return self->split(_skeletonName);
          },
          ::py::arg("skeletonName"))
      .def(
          "copyTo",
          +[](dart::dynamics::BodyNode* self,
              dart::dynamics::BodyNode* _newParent)
              -> std::pair<dart::dynamics::Joint*, dart::dynamics::BodyNode*> {
            return self->copyTo(_newParent);
          },
          ::py::arg("newParent"))
      .def(
          "copyTo",
          +[](dart::dynamics::BodyNode* self,
              dart::dynamics::BodyNode* _newParent,
              bool _recursive)
              -> std::pair<dart::dynamics::Joint*, dart::dynamics::BodyNode*> {
            return self->copyTo(_newParent, _recursive);
          },
          ::py::arg("newParent"),
          ::py::arg("recursive"))
      .def(
          "copyTo",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::SkeletonPtr& _newSkeleton,
              dart::dynamics::BodyNode* _newParent)
              -> std::pair<dart::dynamics::Joint*, dart::dynamics::BodyNode*> {
            return self->copyTo(_newSkeleton, _newParent);
          },
          ::py::arg("newSkeleton"),
          ::py::arg("newParent"))
      .def(
          "copyTo",
          +[](const dart::dynamics::BodyNode* self,
              const dart::dynamics::SkeletonPtr& _newSkeleton,
              dart::dynamics::BodyNode* _newParent,
              bool _recursive)
              -> std::pair<dart::dynamics::Joint*, dart::dynamics::BodyNode*> {
            return self->copyTo(_newSkeleton, _newParent, _recursive);
          },
          ::py::arg("newSkeleton"),
          ::py::arg("newParent"),
          ::py::arg("recursive"))
      .def(
          "copyAs",
          +[](const dart::dynamics::BodyNode* self,
              const std::string& _skeletonName) -> dart::dynamics::SkeletonPtr {
            return self->copyAs(_skeletonName);
          },
          ::py::arg("skeletonName"))
      .def(
          "copyAs",
          +[](const dart::dynamics::BodyNode* self,
              const std::string& _skeletonName,
              bool _recursive) -> dart::dynamics::SkeletonPtr {
            return self->copyAs(_skeletonName, _recursive);
          },
          ::py::arg("skeletonName"),
          ::py::arg("recursive"))
      .def(
          "getSkeleton",
          +[](dart::dynamics::BodyNode* self) -> dart::dynamics::SkeletonPtr {
            return self->getSkeleton();
          })
      .def(
          "getSkeleton",
          +[](const dart::dynamics::BodyNode* self)
              -> dart::dynamics::ConstSkeletonPtr {
            return self->getSkeleton();
          })
      .def(
          "getParentJoint",
          +[](dart::dynamics::BodyNode* self) -> dart::dynamics::Joint* {
            return self->getParentJoint();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "getParentBodyNode",
          +[](dart::dynamics::BodyNode* self) -> dart::dynamics::BodyNode* {
            return self->getParentBodyNode();
          },
          ::py::return_value_policy::reference_internal)
      // clang-format off
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(WeldJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(RevoluteJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(PrismaticJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(ScrewJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(UniversalJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(TranslationalJoint2D)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(PlanarJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(EulerJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(BallJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(TranslationalJoint)
      DARTPY_DEFINE_CREATE_CHILD_JOINT_AND_BODY_NODE_PAIR(FreeJoint)
      // clang-format on
      .def(
          "getNumChildBodyNodes",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getNumChildBodyNodes();
          })
      .def(
          "getNumChildJoints",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getNumChildJoints();
          })
      .def(
          "getNumShapeNodes",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getNumShapeNodes();
          })
      .def(
          "getShapeNode",
          +[](dart::dynamics::BodyNode* self,
              std::size_t index) -> dart::dynamics::ShapeNode* {
            return self->getShapeNode(index);
          },
          ::py::return_value_policy::reference_internal,
          ::py::arg("index"))
      .def(
          "createShapeNode",
          +[](dart::dynamics::BodyNode* self,
              dart::dynamics::ShapePtr shape) -> dart::dynamics::ShapeNode* {
            return self->createShapeNode(shape);
          },
          ::py::return_value_policy::reference_internal,
          ::py::arg("shape"))
      .def(
          "createShapeNode",
          +[](dart::dynamics::BodyNode* self,
              dart::dynamics::ShapePtr shape,
              const std::string& name) -> dart::dynamics::ShapeNode* {
            return self->createShapeNode(shape, name);
          },
          ::py::return_value_policy::reference_internal,
          ::py::arg("shape"),
          ::py::arg("name"))
      .def(
          "getShapeNodes",
          +[](dart::dynamics::BodyNode* self)
              -> const std::vector<dart::dynamics::ShapeNode*> {
            return self->getShapeNodes();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "removeAllShapeNodes",
          +[](dart::dynamics::BodyNode* self) { self->removeAllShapeNodes(); })
      .def(
          "getLocalVertices",
          +[](const dart::dynamics::BodyNode* self)
              -> const std::vector<Eigen::Vector3s> {
            return self->getLocalVertices();
          })
      .def(
          "getMovingVerticesInWorldSpace",
          +[](const dart::dynamics::BodyNode* self, int timestep)
              -> std::vector<dart::dynamics::BodyNode::MovingVertex> {
            return self->getMovingVerticesInWorldSpace(timestep);
          },
          ::py::arg("timestep") = -1)
      .def(
          "getNumEndEffectors",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getNumEndEffectors();
          })
      .def(
          "getNumMarkers",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getNumMarkers();
          })
      .def(
          "dependsOn",
          +[](const dart::dynamics::BodyNode* self, std::size_t _genCoordIndex)
              -> bool { return self->dependsOn(_genCoordIndex); },
          ::py::arg("genCoordIndex"))
      .def(
          "getNumDependentGenCoords",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getNumDependentGenCoords();
          })
      .def(
          "getDependentGenCoordIndex",
          +[](const dart::dynamics::BodyNode* self,
              std::size_t _arrayIndex) -> std::size_t {
            return self->getDependentGenCoordIndex(_arrayIndex);
          },
          ::py::arg("arrayIndex"))
      .def(
          "getNumDependentDofs",
          +[](const dart::dynamics::BodyNode* self) -> std::size_t {
            return self->getNumDependentDofs();
          })
      .def(
          "getChainDofs",
          +[](const dart::dynamics::BodyNode* self)
              -> const std::vector<const dart::dynamics::DegreeOfFreedom*> {
            return self->getChainDofs();
          })
      .def(
          "addExtForce",
          +[](dart::dynamics::BodyNode* self, const Eigen::Vector3s& _force) {
            self->addExtForce(_force);
          },
          ::py::arg("force"))
      .def(
          "addExtForce",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _force,
              const Eigen::Vector3s& _offset) {
            self->addExtForce(_force, _offset);
          },
          ::py::arg("force"),
          ::py::arg("offset"))
      .def(
          "addExtForce",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _force,
              const Eigen::Vector3s& _offset,
              bool _isForceLocal) {
            self->addExtForce(_force, _offset, _isForceLocal);
          },
          ::py::arg("force"),
          ::py::arg("offset"),
          ::py::arg("isForceLocal"))
      .def(
          "addExtForce",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _force,
              const Eigen::Vector3s& _offset,
              bool _isForceLocal,
              bool _isOffsetLocal) {
            self->addExtForce(_force, _offset, _isForceLocal, _isOffsetLocal);
          },
          ::py::arg("force"),
          ::py::arg("offset"),
          ::py::arg("isForceLocal"),
          ::py::arg("isOffsetLocal"))
      .def(
          "setExtForce",
          +[](dart::dynamics::BodyNode* self, const Eigen::Vector3s& _force) {
            self->setExtForce(_force);
          },
          ::py::arg("force"))
      .def(
          "setExtForce",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _force,
              const Eigen::Vector3s& _offset) {
            self->setExtForce(_force, _offset);
          },
          ::py::arg("force"),
          ::py::arg("offset"))
      .def(
          "setExtForce",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _force,
              const Eigen::Vector3s& _offset,
              bool _isForceLocal) {
            self->setExtForce(_force, _offset, _isForceLocal);
          },
          ::py::arg("force"),
          ::py::arg("offset"),
          ::py::arg("isForceLocal"))
      .def(
          "setExtForce",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _force,
              const Eigen::Vector3s& _offset,
              bool _isForceLocal,
              bool _isOffsetLocal) {
            self->setExtForce(_force, _offset, _isForceLocal, _isOffsetLocal);
          },
          ::py::arg("force"),
          ::py::arg("offset"),
          ::py::arg("isForceLocal"),
          ::py::arg("isOffsetLocal"))
      .def(
          "addExtTorque",
          +[](dart::dynamics::BodyNode* self, const Eigen::Vector3s& _torque) {
            self->addExtTorque(_torque);
          },
          ::py::arg("torque"))
      .def(
          "addExtTorque",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _torque,
              bool _isLocal) { self->addExtTorque(_torque, _isLocal); },
          ::py::arg("torque"),
          ::py::arg("isLocal"))
      .def(
          "setExtTorque",
          +[](dart::dynamics::BodyNode* self, const Eigen::Vector3s& _torque) {
            self->setExtTorque(_torque);
          },
          ::py::arg("torque"))
      .def(
          "setExtTorque",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _torque,
              bool _isLocal) { self->setExtTorque(_torque, _isLocal); },
          ::py::arg("torque"),
          ::py::arg("isLocal"))
      .def(
          "setExtWrench",
          +[](dart::dynamics::BodyNode* self, const Eigen::Vector6s& wrench) {
            self->setExtWrench(wrench);
          },
          ::py::arg("wrench"))
      .def(
          "clearExternalForces",
          +[](dart::dynamics::BodyNode* self) { self->clearExternalForces(); })
      .def(
          "clearInternalForces",
          +[](dart::dynamics::BodyNode* self) { self->clearInternalForces(); })
      .def(
          "getExternalForceLocal",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector6s {
            return self->getExternalForceLocal();
          })
      .def(
          "getExternalForceGlobal",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector6s {
            return self->getExternalForceGlobal();
          })
      .def(
          "isReactive",
          +[](const dart::dynamics::BodyNode* self)
              -> bool { return self->isReactive(); })
      .def(
          "getConstraintImpulse",
          +[](const dart::dynamics::BodyNode* self) -> const Eigen::Vector6s& {
            return self->getConstraintImpulse();
          })
      .def(
          "setConstraintImpulse",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector6s& _constImp) {
            self->setConstraintImpulse(_constImp);
          },
          ::py::arg("constImp"))
      .def(
          "addConstraintImpulse",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector6s& _constImp) {
            self->addConstraintImpulse(_constImp);
          },
          ::py::arg("constImp"))
      .def(
          "addConstraintImpulse",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _constImp,
              const Eigen::Vector3s& _offset) {
            self->addConstraintImpulse(_constImp, _offset);
          },
          ::py::arg("constImp"),
          ::py::arg("offset"))
      .def(
          "addConstraintImpulse",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _constImp,
              const Eigen::Vector3s& _offset,
              bool _isImpulseLocal) {
            self->addConstraintImpulse(_constImp, _offset, _isImpulseLocal);
          },
          ::py::arg("constImp"),
          ::py::arg("offset"),
          ::py::arg("isImpulseLocal"))
      .def(
          "addConstraintImpulse",
          +[](dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& _constImp,
              const Eigen::Vector3s& _offset,
              bool _isImpulseLocal,
              bool _isOffsetLocal) {
            self->addConstraintImpulse(
                _constImp, _offset, _isImpulseLocal, _isOffsetLocal);
          },
          ::py::arg("constImp"),
          ::py::arg("offset"),
          ::py::arg("isImpulseLocal"),
          ::py::arg("isOffsetLocal"))
      .def(
          "clearConstraintImpulse",
          +[](dart::dynamics::BodyNode*
                  self) { self->clearConstraintImpulse(); })
      .def(
          "computeLagrangian",
          +[](const dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& gravity) -> s_t {
            return self->computeLagrangian(gravity);
          },
          ::py::arg("gravity"))
      .def(
          "computeKineticEnergy",
          +[](const dart::dynamics::BodyNode* self) -> s_t {
            return self->computeKineticEnergy();
          })
      .def(
          "computePotentialEnergy",
          +[](const dart::dynamics::BodyNode* self,
              const Eigen::Vector3s& gravity) -> s_t {
            return self->computePotentialEnergy(gravity);
          },
          ::py::arg("gravity"))
      .def(
          "getLinearMomentum",
          +[](const dart::dynamics::BodyNode* self) -> Eigen::Vector3s {
            return self->getLinearMomentum();
          })
      .def(
          "getAngularMomentum",
          +[](dart::dynamics::BodyNode* self) -> Eigen::Vector3s {
            return self->getAngularMomentum();
          })
      .def(
          "getAngularMomentum",
          +[](dart::dynamics::BodyNode* self, const Eigen::Vector3s& _pivot)
              -> Eigen::Vector3s { return self->getAngularMomentum(_pivot); },
          ::py::arg("pivot"))
      .def(
          "dirtyTransform",
          +[](dart::dynamics::BodyNode* self) { self->dirtyTransform(); })
      .def(
          "dirtyVelocity",
          +[](dart::dynamics::BodyNode* self) { self->dirtyVelocity(); })
      .def(
          "dirtyAcceleration",
          +[](dart::dynamics::BodyNode* self) { self->dirtyAcceleration(); })
      .def(
          "dirtyArticulatedInertia",
          +[](dart::dynamics::BodyNode*
                  self) { self->dirtyArticulatedInertia(); })
      .def(
          "dirtyExternalForces",
          +[](dart::dynamics::BodyNode* self) { self->dirtyExternalForces(); })
      .def(
          "dirtyCoriolisForces",
          +[](dart::dynamics::BodyNode* self) { self->dirtyCoriolisForces(); });
}

} // namespace python
} // namespace dart
