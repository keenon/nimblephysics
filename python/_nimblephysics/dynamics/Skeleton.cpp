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
#include <dart/dynamics/PointMass.hpp>
#include <dart/dynamics/PrismaticJoint.hpp>
#include <dart/dynamics/RevoluteJoint.hpp>
#include <dart/dynamics/ScrewJoint.hpp>
#include <dart/dynamics/Skeleton.hpp>
#include <dart/dynamics/TranslationalJoint.hpp>
#include <dart/dynamics/TranslationalJoint2D.hpp>
#include <dart/dynamics/UniversalJoint.hpp>
#include <dart/dynamics/WeldJoint.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dart/math/MathTypes.hpp"

#include "eigen_geometry_pybind.h"
#include "eigen_pybind.h"

namespace py = pybind11;

#define DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(joint_type)              \
  .def(                                                                        \
      "create" #joint_type "AndBodyNodePair",                                  \
      +[](dart::dynamics::Skeleton* self)                                      \
          -> std::                                                             \
              pair<dart::dynamics::joint_type*, dart::dynamics::BodyNode*> {   \
                return self->createJointAndBodyNodePair<                       \
                    dart::dynamics::joint_type,                                \
                    dart::dynamics::BodyNode>();                               \
              },                                                               \
      ::py::return_value_policy::reference_internal)                           \
      .def(                                                                    \
          "create" #joint_type "AndBodyNodePair",                              \
          +[](dart::dynamics::Skeleton* self,                                  \
              dart::dynamics::BodyNode* parent)                                \
              -> std::pair<                                                    \
                  dart::dynamics::joint_type*,                                 \
                  dart::dynamics::BodyNode*> {                                 \
            return self->createJointAndBodyNodePair<                           \
                dart::dynamics::joint_type,                                    \
                dart::dynamics::BodyNode>(parent);                             \
          },                                                                   \
          ::py::return_value_policy::reference_internal,                       \
          ::py::arg("parent"))                                                 \
      .def(                                                                    \
          "create" #joint_type "AndBodyNodePair",                              \
          +[](dart::dynamics::Skeleton* self,                                  \
              dart::dynamics::BodyNode* parent,                                \
              const dart::dynamics::joint_type::Properties& jointProperties)   \
              -> std::pair<                                                    \
                  dart::dynamics::joint_type*,                                 \
                  dart::dynamics::BodyNode*> {                                 \
            return self->createJointAndBodyNodePair<                           \
                dart::dynamics::joint_type,                                    \
                dart::dynamics::BodyNode>(parent, jointProperties);            \
          },                                                                   \
          ::py::return_value_policy::reference_internal,                       \
          ::py::arg("parent"),                                                 \
          ::py::arg("jointProperties"))                                        \
      .def(                                                                    \
          "create" #joint_type "AndBodyNodePair",                              \
          +[](dart::dynamics::Skeleton* self,                                  \
              dart::dynamics::BodyNode* parent,                                \
              const dart::dynamics::joint_type::Properties& jointProperties,   \
              const dart::dynamics::BodyNode::Properties& bodyProperties)      \
              -> std::pair<                                                    \
                  dart::dynamics::joint_type*,                                 \
                  dart::dynamics::BodyNode*> {                                 \
            return self->createJointAndBodyNodePair<                           \
                dart::dynamics::joint_type,                                    \
                dart::dynamics::BodyNode>(                                     \
                parent, jointProperties, bodyProperties);                      \
          },                                                                   \
          ::py::return_value_policy::reference_internal,                       \
          ::py::arg("parent").none(true),                                      \
          ::py::arg("jointProperties"),                                        \
          ::py::arg("bodyProperties"))

namespace dart {
namespace python {

void Skeleton(
    py::module& m,
    ::py::class_<
        dart::dynamics::Skeleton,
        dart::dynamics::MetaSkeleton,
        std::shared_ptr<dart::dynamics::Skeleton>>& skeleton)
{
  ::py::class_<dart::dynamics::BodyScaleGroup>(m, "BodyScaleGroup")
      .def_readwrite("nodes", &dart::dynamics::BodyScaleGroup::nodes)
      .def_readwrite("flipAxis", &dart::dynamics::BodyScaleGroup::flipAxis)
      .def_readwrite(
          "uniformScaling", &dart::dynamics::BodyScaleGroup::uniformScaling);
  ::py::class_<dart::dynamics::Skeleton::ContactInverseDynamicsResult>(
      m, "ContactInverseDynamicsResult")
      .def(::py::init<>())
      .def_readwrite(
          "skel", &dynamics::Skeleton::ContactInverseDynamicsResult::skel)
      .def_readwrite(
          "contactBody",
          &dynamics::Skeleton::ContactInverseDynamicsResult::contactBody)
      .def_readwrite(
          "contactWrench",
          &dynamics::Skeleton::ContactInverseDynamicsResult::contactWrench)
      .def_readwrite(
          "jointTorques",
          &dynamics::Skeleton::ContactInverseDynamicsResult::jointTorques)
      .def_readwrite(
          "pos", &dynamics::Skeleton::ContactInverseDynamicsResult::pos)
      .def_readwrite(
          "vel", &dynamics::Skeleton::ContactInverseDynamicsResult::vel)
      .def_readwrite(
          "acc", &dynamics::Skeleton::ContactInverseDynamicsResult::acc)
      .def(
          "sumError",
          &dynamics::Skeleton::ContactInverseDynamicsResult::sumError);

  ::py::class_<dart::dynamics::Skeleton::MultipleContactInverseDynamicsResult>(
      m, "MultipleContactInverseDynamicsResult")
      .def(::py::init<>())
      .def_readwrite(
          "skel",
          &dynamics::Skeleton::MultipleContactInverseDynamicsResult::skel)
      .def_readwrite(
          "contactBodies",
          &dynamics::Skeleton::MultipleContactInverseDynamicsResult::
              contactBodies)
      .def_readwrite(
          "contactWrenches",
          &dynamics::Skeleton::MultipleContactInverseDynamicsResult::
              contactWrenches)
      .def_readwrite(
          "contactWrenchGuesses",
          &dynamics::Skeleton::MultipleContactInverseDynamicsResult::
              contactWrenchGuesses)
      .def_readwrite(
          "jointTorques",
          &dynamics::Skeleton::MultipleContactInverseDynamicsResult::
              jointTorques)
      .def_readwrite(
          "pos", &dynamics::Skeleton::MultipleContactInverseDynamicsResult::pos)
      .def_readwrite(
          "vel", &dynamics::Skeleton::MultipleContactInverseDynamicsResult::vel)
      .def_readwrite(
          "acc", &dynamics::Skeleton::MultipleContactInverseDynamicsResult::acc)
      .def(
          "sumError",
          &dynamics::Skeleton::MultipleContactInverseDynamicsResult::sumError)
      .def(
          "computeGuessLoss",
          &dynamics::Skeleton::MultipleContactInverseDynamicsResult::
              computeGuessLoss);

  ::py::class_<
      dart::dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult>(
      m, "MultipleContactInverseDynamicsOverTimeResult")
      .def(::py::init<>())
      .def_readwrite(
          "skel",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              skel)
      .def_readwrite(
          "timesteps",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              timesteps)
      .def_readwrite(
          "contactBodies",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              contactBodies)
      .def_readwrite(
          "contactWrenches",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              contactWrenches)
      .def_readwrite(
          "jointTorques",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              jointTorques)
      .def_readwrite(
          "positions",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              positions)
      .def_readwrite(
          "velocities",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              velocities)
      .def_readwrite(
          "accelerations",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              accelerations)
      .def_readwrite(
          "prevContactForces",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              prevContactForces)
      .def(
          "sumError",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              sumError)
      .def(
          "computePrevForceLoss",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              computePrevForceLoss)
      .def(
          "computeSmoothnessLoss",
          &dynamics::Skeleton::MultipleContactInverseDynamicsOverTimeResult::
              computeSmoothnessLoss);

  /*
    typedef struct EnergyAccountingFrame
    {
      // The current amount of energy in each of the body segments, split across
      // both categories.
      Eigen::VectorXs bodyKineticEnergy;
      Eigen::VectorXs bodyPotentialEnergy;
      std::vector<Eigen::Vector3s> bodyCenters;

      // Each joint energy transfer
      std::vector<JointEnergyTransmitter> joints;
      // Each contact energy transfer
      std::vector<ContactEnergyTransmitter> contacts;
    } EnergyAccountingFrame;
  */
  ::py::class_<dart::dynamics::Skeleton::JointEnergyTransmitter>(
      m, "JointEnergyTransmitter")
      .def(::py::init<>())
      .def_readwrite("name", &dynamics::Skeleton::JointEnergyTransmitter::name)
      .def_readwrite(
          "worldCenter",
          &dynamics::Skeleton::JointEnergyTransmitter::worldCenter)
      .def_readwrite(
          "parentBody", &dynamics::Skeleton::JointEnergyTransmitter::parentBody)
      .def_readwrite(
          "parentCenter",
          &dynamics::Skeleton::JointEnergyTransmitter::parentCenter)
      .def_readwrite(
          "childBody", &dynamics::Skeleton::JointEnergyTransmitter::childBody)
      .def_readwrite(
          "childCenter",
          &dynamics::Skeleton::JointEnergyTransmitter::childCenter)
      .def_readwrite(
          "powerToParent",
          &dynamics::Skeleton::JointEnergyTransmitter::powerToParent)
      .def_readwrite(
          "powerToChild",
          &dynamics::Skeleton::JointEnergyTransmitter::powerToChild);

  ::py::class_<dart::dynamics::Skeleton::ContactEnergyTransmitter>(
      m, "ContactEnergyTransmitter")
      .def(::py::init<>())
      .def_readwrite(
          "worldCenter",
          &dynamics::Skeleton::ContactEnergyTransmitter::worldCenter)
      .def_readwrite(
          "worldForce",
          &dynamics::Skeleton::ContactEnergyTransmitter::worldForce)
      .def_readwrite(
          "worldMoment",
          &dynamics::Skeleton::ContactEnergyTransmitter::worldMoment)
      .def_readwrite(
          "contactBody",
          &dynamics::Skeleton::ContactEnergyTransmitter::contactBody)
      .def_readwrite(
          "contactBodyCenter",
          &dynamics::Skeleton::ContactEnergyTransmitter::contactBodyCenter)
      .def_readwrite(
          "powerToBody",
          &dynamics::Skeleton::ContactEnergyTransmitter::powerToBody);

  ::py::class_<dart::dynamics::Skeleton::EnergyAccountingFrame>(
      m, "EnergyAccountingFrame")
      .def(::py::init<>())
      .def_readwrite(
          "bodyCenters",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyCenters)
      .def_readwrite(
          "bodyKineticEnergy",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyKineticEnergy)
      .def_readwrite(
          "bodyPotentialEnergy",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyPotentialEnergy)
      .def_readwrite(
          "bodyKineticEnergyDeriv",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyKineticEnergyDeriv)
      .def_readwrite(
          "bodyPotentialEnergyDeriv",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyPotentialEnergyDeriv)
      .def_readwrite(
          "bodyParentJointPower",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyParentJointPower)
      .def_readwrite(
          "bodyGravityPower",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyGravityPower)
      .def_readwrite(
          "bodyExternalForcePower",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyExternalForcePower)
      .def_readwrite(
          "bodyChildJointPowerSum",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyChildJointPowerSum)
      .def_readwrite(
          "bodyChildJointPowers",
          &dynamics::Skeleton::EnergyAccountingFrame::bodyChildJointPowers)
      .def_readwrite(
          "contacts", &dynamics::Skeleton::EnergyAccountingFrame::contacts)
      .def_readwrite(
          "joints", &dynamics::Skeleton::EnergyAccountingFrame::joints);

  skeleton
      .def(::py::init(+[]() -> dart::dynamics::SkeletonPtr {
        return dart::dynamics::Skeleton::create();
      }))
      .def(
          ::py::init(
              +[](const std::string& _name) -> dart::dynamics::SkeletonPtr {
                return dart::dynamics::Skeleton::create(_name);
              }),
          ::py::arg("name"))
      .def(
          "getPtr",
          +[](dart::dynamics::Skeleton* self) -> dart::dynamics::SkeletonPtr {
            return self->getPtr();
          })
      .def(
          "getPtr",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::dynamics::ConstSkeletonPtr { return self->getPtr(); })
      .def(
          "getSkeleton",
          +[](dart::dynamics::Skeleton* self) -> dart::dynamics::SkeletonPtr {
            return self->getSkeleton();
          })
      .def(
          "getSkeleton",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::dynamics::ConstSkeletonPtr {
            return self->getSkeleton();
          })
      /*
      .def(
          "getLockableReference",
          +[](const dart::dynamics::Skeleton* self)
              -> std::unique_ptr<dart::common::LockableReference> {
            return self->getLockableReference();
          })
    */
      .def(
          "clone",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::dynamics::SkeletonPtr { return self->cloneSkeleton(); })
      .def(
          "clone",
          +[](const dart::dynamics::Skeleton* self,
              const std::string& cloneName) -> dart::dynamics::SkeletonPtr {
            return self->cloneSkeleton(cloneName);
          },
          ::py::arg("cloneName"))
      .def(
          "simplifySkeleton",
          +[](const dart::dynamics::Skeleton* self,
              const std::string& cloneName,
              std::map<std::string, std::string> mergeBodiesInto)
              -> dart::dynamics::SkeletonPtr {
            return self->simplifySkeleton(cloneName, mergeBodiesInto);
          },
          ::py::arg("cloneName"),
          ::py::arg("mergeBodiesInto"))
      /*
      .def(
          "setConfiguration",
          +[](dart::dynamics::Skeleton* self,
              const dart::dynamics::Skeleton::Configuration& configuration)
              -> void { return self->setConfiguration(configuration); },
          ::py::arg("configuration"))
      .def(
          "getConfiguration",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::dynamics::Skeleton::Configuration {
            return self->getConfiguration();
          })
      .def(
          "getConfiguration",
          +[](const dart::dynamics::Skeleton* self,
              int flags) -> dart::dynamics::Skeleton::Configuration {
            return self->getConfiguration(flags);
          },
          ::py::arg("flags"))
      .def(
          "getConfiguration",
          +[](const dart::dynamics::Skeleton* self,
              const std::vector<std::size_t>& indices)
              -> dart::dynamics::Skeleton::Configuration {
            return self->getConfiguration(indices);
          },
          ::py::arg("indices"))
      .def(
          "getConfiguration",
          +[](const dart::dynamics::Skeleton* self,
              const std::vector<std::size_t>& indices,
              int flags) -> dart::dynamics::Skeleton::Configuration {
            return self->getConfiguration(indices, flags);
          },
          ::py::arg("indices"),
          ::py::arg("flags"))
      .def(
          "setState",
          +[](dart::dynamics::Skeleton* self,
              const dart::dynamics::Skeleton::State& state) -> void {
            return self->setState(state);
          },
          ::py::arg("state"))
      .def(
          "getState",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::dynamics::Skeleton::State { return self->getState(); })
      */
      .def(
          "setProperties",
          +[](dart::dynamics::Skeleton* self,
              const dart::dynamics::Skeleton::Properties& properties) -> void {
            return self->setProperties(properties);
          },
          ::py::arg("properties"))
      .def(
          "getProperties",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::dynamics::Skeleton::Properties {
            return self->getProperties();
          })
      .def(
          "setProperties",
          +[](dart::dynamics::Skeleton* self,
              const dart::dynamics::Skeleton::Properties& properties) -> void {
            return self->setProperties(properties);
          },
          ::py::arg("properties"))
      .def(
          "setName",
          +[](dart::dynamics::Skeleton* self, const std::string& _name)
              -> const std::string& { return self->setName(_name); },
          ::py::return_value_policy::reference_internal,
          ::py::arg("name"))
      .def(
          "getName",
          +[](const dart::dynamics::Skeleton* self) -> const std::string& {
            return self->getName();
          },
          ::py::return_value_policy::reference_internal)
      .def(
          "setSelfCollisionCheck",
          +[](dart::dynamics::Skeleton* self, bool enable) -> void {
            return self->setSelfCollisionCheck(enable);
          },
          ::py::arg("enable"))
      .def(
          "getSelfCollisionCheck",
          +[](const dart::dynamics::Skeleton* self) -> bool {
            return self->getSelfCollisionCheck();
          })
      .def(
          "enableSelfCollisionCheck",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->enableSelfCollisionCheck();
          })
      .def(
          "disableSelfCollisionCheck",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->disableSelfCollisionCheck();
          })
      .def(
          "isEnabledSelfCollisionCheck",
          +[](const dart::dynamics::Skeleton* self) -> bool {
            return self->isEnabledSelfCollisionCheck();
          })
      .def(
          "setAdjacentBodyCheck",
          +[](dart::dynamics::Skeleton* self, bool enable) -> void {
            return self->setAdjacentBodyCheck(enable);
          },
          ::py::arg("enable"))
      .def(
          "getAdjacentBodyCheck",
          +[](const dart::dynamics::Skeleton* self) -> bool {
            return self->getAdjacentBodyCheck();
          })
      .def(
          "enableAdjacentBodyCheck",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->enableAdjacentBodyCheck();
          })
      .def(
          "disableAdjacentBodyCheck",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->disableAdjacentBodyCheck();
          })
      .def(
          "isEnabledAdjacentBodyCheck",
          +[](const dart::dynamics::Skeleton* self) -> bool {
            return self->isEnabledAdjacentBodyCheck();
          })
      .def(
          "setMobile",
          +[](dart::dynamics::Skeleton* self, bool _isMobile) -> void {
            return self->setMobile(_isMobile);
          },
          ::py::arg("isMobile"))
      .def(
          "isMobile",
          +[](const dart::dynamics::Skeleton* self) -> bool {
            return self->isMobile();
          })
      .def(
          "setTimeStep",
          +[](dart::dynamics::Skeleton* self, s_t _timeStep) -> void {
            return self->setTimeStep(_timeStep);
          },
          ::py::arg("timeStep"))
      .def(
          "getTimeStep",
          +[](const dart::dynamics::Skeleton* self) -> s_t {
            return self->getTimeStep();
          })
      .def(
          "setGravity",
          +[](dart::dynamics::Skeleton* self, const Eigen::Vector3s& _gravity)
              -> void { return self->setGravity(_gravity); },
          ::py::arg("gravity"))
      .def(
          "getGravity",
          +[](const dart::dynamics::Skeleton* self) -> const Eigen::Vector3s& {
            return self->getGravity();
          },
          ::py::return_value_policy::reference_internal)
      // clang-format off
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(WeldJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(RevoluteJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(PrismaticJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(ScrewJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(UniversalJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(TranslationalJoint2D)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(PlanarJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(EulerJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(BallJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(TranslationalJoint)
      DARTPY_DEFINE_CREATE_JOINT_AND_BODY_NODE_PAIR(FreeJoint)
      // clang-format on
      .def(
          "getNumBodyNodes",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumBodyNodes();
          })
      .def(
          "getNumRigidBodyNodes",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumRigidBodyNodes();
          })
      .def(
          "getNumSoftBodyNodes",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumSoftBodyNodes();
          })
      .def(
          "getNumTrees",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumTrees();
          })
      .def(
          "getRootBodyNode",
          +[](dart::dynamics::Skeleton* self) -> dart::dynamics::BodyNode* {
            return self->getRootBodyNode();
          },
          py::return_value_policy::reference)
      .def(
          "getRootBodyNode",
          +[](dart::dynamics::Skeleton* self,
              std::size_t index) -> dart::dynamics::BodyNode* {
            return self->getRootBodyNode(index);
          },
          ::py::arg("treeIndex"),
          py::return_value_policy::reference)
      .def(
          "getRootJoint",
          +[](dart::dynamics::Skeleton* self) -> dart::dynamics::Joint* {
            return self->getRootJoint();
          },
          py::return_value_policy::reference_internal)
      .def(
          "getRootJoint",
          +[](dart::dynamics::Skeleton* self, std::size_t index)
              -> dart::dynamics::Joint* { return self->getRootJoint(index); },
          ::py::arg("treeIndex"),
          py::return_value_policy::reference_internal)
      .def(
          "getBodyNodes",
          +[](dart::dynamics::Skeleton* self)
              -> const std::vector<dart::dynamics::BodyNode*>& {
            return self->getBodyNodes();
          },
          py::return_value_policy::reference_internal)
      .def(
          "hasBodyNode",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::BodyNode* bodyNode) -> bool {
            return self->hasBodyNode(bodyNode);
          },
          ::py::arg("bodyNode"))
      .def(
          "getIndexOf",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::BodyNode* _bn) -> std::size_t {
            return self->getIndexOf(_bn);
          },
          ::py::arg("bn"))
      .def(
          "getIndexOf",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::BodyNode* _bn,
              bool _warning) -> std::size_t {
            return self->getIndexOf(_bn, _warning);
          },
          ::py::arg("bn"),
          ::py::arg("warning"))
      .def(
          "getTreeBodyNodes",
          +[](const dart::dynamics::Skeleton* self, std::size_t _treeIdx)
              -> std::vector<const dart::dynamics::BodyNode*> {
            return self->getTreeBodyNodes(_treeIdx);
          },
          ::py::arg("treeIdx"))
      .def(
          "getNumJoints",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumJoints();
          })
      .def(
          "getJoint",
          +[](dart::dynamics::Skeleton* self, std::size_t _idx)
              -> dart::dynamics::Joint* { return self->getJoint(_idx); },
          ::py::arg("idx"),
          py::return_value_policy::reference_internal)
      .def(
          "getJoint",
          +[](dart::dynamics::Skeleton* self, const std::string& name)
              -> dart::dynamics::Joint* { return self->getJoint(name); },
          ::py::arg("name"),
          py::return_value_policy::reference_internal)
      /*
      // These methods all crash because pybind11 tries to take ownership of
      // the joints within the list
      // When the list of joints is freed on the Python side, it attempts to
      // free the joint pointers too.
      // This seems to be a limitation of pybind11, it's challenging to assign
      // different return policies
      // to a vector, and the contents of that vector:
      // https://github.com/pybind/pybind11/issues/637
    .def( "getJoints",
        +[](dart::dynamics::Skeleton* self)
            -> std::vector<dart::dynamics::Joint*> {
          return self->getJoints();
        })
    .def(
        "getJoints",
        +[](const dart::dynamics::Skeleton* self)
            -> std::vector<const dart::dynamics::Joint*> {
          return self->getJoints();
        })
    .def(
        "getJoints",
        +[](dart::dynamics::Skeleton* self,
            const std::string& name) -> std::vector<dart::dynamics::Joint*> {
          return self->getJoints(name);
        },
        ::py::arg("name"))
    .def(
        "getJoints",
        +[](const dart::dynamics::Skeleton* self, const std::string& name)
            -> std::vector<const dart::dynamics::Joint*> {
          return self->getJoints(name);
        },
        ::py::arg("name"))
        */
      .def(
          "hasJoint",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Joint* joint) -> bool {
            return self->hasJoint(joint);
          },
          ::py::arg("joint"))
      .def(
          "getIndexOf",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Joint* _joint) -> std::size_t {
            return self->getIndexOf(_joint);
          },
          ::py::arg("joint"))
      .def(
          "getIndexOf",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Joint* _joint,
              bool _warning) -> std::size_t {
            return self->getIndexOf(_joint, _warning);
          },
          ::py::arg("joint"),
          ::py::arg("warning"))
      .def(
          "getNumDofs",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumDofs();
          })
      .def(
          "getDof",
          +[](dart::dynamics::Skeleton* self,
              const std::string& name) -> dart::dynamics::DegreeOfFreedom* {
            return self->getDof(name);
          },
          ::py::return_value_policy::reference_internal,
          ::py::arg("name"))
      .def(
          "getDofByIndex",
          +[](dart::dynamics::Skeleton* self, int i)
              -> dart::dynamics::DegreeOfFreedom* { return self->getDof(i); },
          ::py::return_value_policy::reference_internal,
          ::py::arg("index"))
      .def(
          "getIndexOf",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::DegreeOfFreedom* _dof) -> std::size_t {
            return self->getIndexOf(_dof);
          },
          ::py::arg("dof"))
      .def(
          "getIndexOf",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::DegreeOfFreedom* _dof,
              bool _warning) -> std::size_t {
            return self->getIndexOf(_dof, _warning);
          },
          ::py::arg("dof"),
          ::py::arg("warning"))
      .def(
          "checkIndexingConsistency",
          +[](const dart::dynamics::Skeleton* self) -> bool {
            return self->checkIndexingConsistency();
          })
      .def(
          "getNumMarkers",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumMarkers();
          })
      .def(
          "getNumMarkers",
          +[](const dart::dynamics::Skeleton* self, std::size_t treeIndex)
              -> std::size_t { return self->getNumMarkers(treeIndex); },
          ::py::arg("treeIndex"))
      .def(
          "getNumShapeNodes",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumShapeNodes();
          })
      .def(
          "getNumShapeNodes",
          +[](const dart::dynamics::Skeleton* self, std::size_t treeIndex)
              -> std::size_t { return self->getNumShapeNodes(treeIndex); },
          ::py::arg("treeIndex"))
      .def(
          "getNumEndEffectors",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getNumEndEffectors();
          })
      .def(
          "getNumEndEffectors",
          +[](const dart::dynamics::Skeleton* self, std::size_t treeIndex)
              -> std::size_t { return self->getNumEndEffectors(treeIndex); },
          ::py::arg("treeIndex"))
      .def(
          "getHeight",
          &dart::dynamics::Skeleton::getHeight,
          ::py::arg("pos"),
          ::py::arg("up") = Eigen::Vector3s::UnitY())
      .def(
          "getGradientOfHeightWrtBodyScales",
          &dart::dynamics::Skeleton::getGradientOfHeightWrtBodyScales,
          ::py::arg("pos"),
          ::py::arg("up") = Eigen::Vector3s::UnitY())
      .def(
          "getLowestPoint",
          &dart::dynamics::Skeleton::getLowestPoint,
          ::py::arg("up") = Eigen::Vector3s::UnitY())
      .def(
          "getGradientOfLowestPointWrtBodyScales",
          &dart::dynamics::Skeleton::getGradientOfLowestPointWrtBodyScales,
          ::py::arg("up") = Eigen::Vector3s::UnitY())
      .def(
          "getGradientOfLowestPointWrtJoints",
          &dart::dynamics::Skeleton::getGradientOfLowestPointWrtJoints,
          ::py::arg("up") = Eigen::Vector3s::UnitY())
      .def("getRandomPose", &dart::dynamics::Skeleton::getRandomPose)
      .def(
          "getRandomPoseForJoints",
          &dart::dynamics::Skeleton::getRandomPoseForJoints,
          ::py::arg("joints"))
      .def(
          "getControlForceUpperLimits",
          +[](dart::dynamics::Skeleton* self) -> Eigen::VectorXs {
            return self->getControlForceUpperLimits();
          })
      .def(
          "getControlForceLowerLimits",
          +[](dart::dynamics::Skeleton* self) -> Eigen::VectorXs {
            return self->getControlForceLowerLimits();
          })
      .def(
          "getPositionLowerLimits",
          +[](dart::dynamics::Skeleton* self) -> Eigen::VectorXs {
            return self->getPositionLowerLimits();
          })
      .def(
          "getPositionUpperLimits",
          +[](dart::dynamics::Skeleton* self) -> Eigen::VectorXs {
            return self->getPositionUpperLimits();
          })
      .def(
          "getVelocityLowerLimits",
          +[](dart::dynamics::Skeleton* self) -> Eigen::VectorXs {
            return self->getVelocityLowerLimits();
          })
      .def(
          "getVelocityUpperLimits",
          +[](dart::dynamics::Skeleton* self) -> Eigen::VectorXs {
            return self->getVelocityUpperLimits();
          })
      .def(
          "setControlForcesUpperLimits",
          +[](dart::dynamics::Skeleton* self, Eigen::VectorXs limits) -> void {
            self->setControlForceUpperLimits(limits);
          })
      .def(
          "setControlForcesLowerLimits",
          +[](dart::dynamics::Skeleton* self, Eigen::VectorXs limits) -> void {
            self->setControlForceLowerLimits(limits);
          })
      .def(
          "setPositionUpperLimits",
          +[](dart::dynamics::Skeleton* self, Eigen::VectorXs limits) -> void {
            self->setPositionUpperLimits(limits);
          })
      .def(
          "setPositionLowerLimits",
          +[](dart::dynamics::Skeleton* self, Eigen::VectorXs limits) -> void {
            self->setPositionLowerLimits(limits);
          })
      .def("getBodyScales", &dart::dynamics::Skeleton::getBodyScales)
      .def(
          "setBodyScales",
          &dart::dynamics::Skeleton::setBodyScales,
          ::py::arg("scales"))
      .def(
          "clampPositionsToLimits",
          &dart::dynamics::Skeleton::clampPositionsToLimits)
      .def(
          "getBodyForMesh",
          &dart::dynamics::Skeleton::getBodyForMesh,
          ::py::arg("meshFileName"),
          ::py::return_value_policy::reference_internal)
      .def(
          "getTransformFromMeshToParentBody",
          &dart::dynamics::Skeleton::getTransformFromMeshToParentBody,
          ::py::arg("meshFileName"),
          ::py::arg("relativeToGeometry"))
      .def(
          "getTranslationFromMeshToParentBody",
          &dart::dynamics::Skeleton::getTranslationFromMeshToParentBody,
          ::py::arg("meshFileName"),
          ::py::arg("relativeToGeometry"))
      .def(
          "getRotationFromMeshToParentBody",
          &dart::dynamics::Skeleton::getRotationFromMeshToParentBody,
          ::py::arg("meshFileName"),
          ::py::arg("relativeToGeometry"))
      .def("getBodyScaleGroups", &dart::dynamics::Skeleton::getBodyScaleGroups)
      .def(
          "getBodyScaleGroup",
          &dart::dynamics::Skeleton::getBodyScaleGroup,
          ::py::arg("index"))
      .def(
          "getScaleGroupLowerBound",
          &dart::dynamics::Skeleton::getScaleGroupLowerBound,
          ::py::arg("index"))
      .def(
          "getScaleGroupUpperBound",
          &dart::dynamics::Skeleton::getScaleGroupUpperBound,
          ::py::arg("index"))
      .def(
          "getScaleGroupIndex",
          &dart::dynamics::Skeleton::getScaleGroupIndex,
          ::py::arg("bodyNode"))
      .def(
          "mergeScaleGroups",
          &dart::dynamics::Skeleton::mergeScaleGroups,
          ::py::arg("bodyNodeA"),
          ::py::arg("bodyNodeB"))
      .def(
          "mergeScaleGroupsByIndex",
          &dart::dynamics::Skeleton::mergeScaleGroupsByIndex,
          ::py::arg("groupA"),
          ::py::arg("groupB"))
      .def(
          "autogroupSymmetricSuffixes",
          &dart::dynamics::Skeleton::autogroupSymmetricSuffixes,
          ::py::arg("leftSuffix") = "_l",
          ::py::arg("rightSuffix") = "_r")
      .def(
          "autogroupSymmetricPrefixes",
          &dart::dynamics::Skeleton::autogroupSymmetricPrefixes,
          ::py::arg("firstPrefix") = "radius",
          ::py::arg("secondPrefix") = "ulna")
      .def("getGroupScaleDim", &dart::dynamics::Skeleton::getGroupScaleDim)
      .def(
          "setScaleGroupUniformScaling",
          &dart::dynamics::Skeleton::setScaleGroupUniformScaling,
          ::py::arg("bodyNode"),
          ::py::arg("uniform") = true)
      .def(
          "setLinkMasses",
          &dart::dynamics::Skeleton::setLinkMasses,
          ::py::arg("masses"))
      .def("getLinkMasses", &dart::dynamics::Skeleton::getLinkMasses)
      .def(
          "setGroupScales",
          &dart::dynamics::Skeleton::setGroupScales,
          ::py::arg("scales"),
          ::py::arg("silentlyClamp") = false)
      .def("getGroupScales", &dart::dynamics::Skeleton::getGroupScales)
      .def(
          "setGroupMasses",
          &dart::dynamics::Skeleton::setGroupMasses,
          ::py::arg("masses"))
      .def("getGroupMasses", &dart::dynamics::Skeleton::getGroupMasses)
      .def(
          "getGroupMassesUpperBound",
          &dart::dynamics::Skeleton::getGroupMassesUpperBound)
      .def(
          "getGroupMassesLowerBound",
          &dart::dynamics::Skeleton::getGroupMassesLowerBound)
      .def(
          "setGroupInertias",
          &dart::dynamics::Skeleton::setGroupInertias,
          ::py::arg("inertias"))
      .def("getGroupInertias", &dart::dynamics::Skeleton::getGroupInertias)
      .def(
          "getGroupInertiasUpperBound",
          &dart::dynamics::Skeleton::getGroupInertiasUpperBound)
      .def(
          "getGroupInertiasLowerBound",
          &dart::dynamics::Skeleton::getGroupInertiasLowerBound)
      .def(
          "setGroupCOMs",
          &dart::dynamics::Skeleton::setGroupCOMs,
          ::py::arg("coms"))
      .def("getGroupCOMs", &dart::dynamics::Skeleton::getGroupCOMs)
      .def(
          "getGroupCOMUpperBound",
          &dart::dynamics::Skeleton::getGroupCOMUpperBound)
      .def(
          "getGroupCOMLowerBound",
          &dart::dynamics::Skeleton::getGroupCOMLowerBound)
      .def(
          "setLinearizedMasses",
          &dart::dynamics::Skeleton::setLinearizedMasses,
          ::py::arg("masses"))
      .def(
          "getLinearizedMasses", &dart::dynamics::Skeleton::getLinearizedMasses)
      .def(
          "getJointWorldPositionsJacobianWrtGroupScales",
          &dart::dynamics::Skeleton::
              getJointWorldPositionsJacobianWrtGroupScales,
          ::py::arg("joints"))
      .def(
          "getJointWorldPositions",
          &dart::dynamics::Skeleton::getJointWorldPositions,
          ::py::arg("joints"))
      .def(
          "getJointWorldPositionsMap",
          &dart::dynamics::Skeleton::getJointWorldPositionsMap)
      .def(
          "getJointWorldPositionsJacobianWrtJointPositions",
          &dart::dynamics::Skeleton::
              getJointWorldPositionsJacobianWrtJointPositions,
          ::py::arg("joints"))
      .def(
          "getJointWorldPositionsJacobianWrtBodyScales",
          &dart::dynamics::Skeleton::
              getJointWorldPositionsJacobianWrtBodyScales,
          ::py::arg("joints"))
      .def(
          "getMarkerWorldPositions",
          &dart::dynamics::Skeleton::getMarkerWorldPositions,
          ::py::arg("markers"))
      .def(
          "getMarkerMapWorldPositions",
          &dart::dynamics::Skeleton::getMarkerMapWorldPositions,
          ::py::arg("markerMap"))
      .def(
          "convertMarkerMap",
          &dart::dynamics::Skeleton::convertMarkerMap,
          ::py::arg("markerMap"),
          ::py::arg("warnOnDrop") = true)
      .def(
          "fitJointsToWorldPositions",
          +[](dart::dynamics::Skeleton* self,
              const std::vector<dynamics::Joint*>& positionJoints,
              Eigen::VectorXs targetPositions,
              bool scaleBodies,
              double convergenceThreshold,
              int maxStepCount,
              double leastSquaresDamping,
              bool lineSearch,
              bool logOutput) -> double {
            return self->fitJointsToWorldPositions(
                positionJoints,
                targetPositions,
                scaleBodies,
                math::IKConfig()
                    .setConvergenceThreshold(convergenceThreshold)
                    .setMaxStepCount(maxStepCount)
                    .setLeastSquaresDamping(leastSquaresDamping)
                    .setLineSearch(lineSearch)
                    .setLogOutput(logOutput));
          },
          ::py::arg("positionJoints"),
          ::py::arg("targetPositions"),
          ::py::arg("scaleBodies") = false,
          ::py::arg("convergenceThreshold") = 1e-7,
          ::py::arg("maxStepCount") = 100,
          ::py::arg("leastSquaresDamping") = 0.01,
          ::py::arg("lineSearch") = true,
          ::py::arg("logOutput") = false)
      .def(
          "fitMarkersToWorldPositions",
          +[](dart::dynamics::Skeleton* self,
              const std::vector<
                  std::pair<dynamics::BodyNode*, Eigen::Vector3s>>& markers,
              Eigen::VectorXs targetPositions,
              Eigen::VectorXs markerWeights,
              bool scaleBodies,
              double convergenceThreshold,
              int maxStepCount,
              double leastSquaresDamping,
              bool lineSearch,
              bool logOutput) -> double {
            return self->fitMarkersToWorldPositions(
                markers,
                targetPositions,
                markerWeights,
                scaleBodies,
                math::IKConfig()
                    .setConvergenceThreshold(convergenceThreshold)
                    .setMaxStepCount(maxStepCount)
                    .setLeastSquaresDamping(leastSquaresDamping)
                    .setLineSearch(lineSearch)
                    .setLogOutput(logOutput));
          },
          ::py::arg("markers"),
          ::py::arg("targetPositions"),
          ::py::arg("markerWeights"),
          ::py::arg("scaleBodies") = false,
          ::py::arg("convergenceThreshold") = 1e-7,
          ::py::arg("maxStepCount") = 100,
          ::py::arg("leastSquaresDamping") = 0.01,
          ::py::arg("lineSearch") = true,
          ::py::arg("logOutput") = false)
      .def(
          "getGyroMapReadings",
          &dart::dynamics::Skeleton::getGyroMapReadings,
          ::py::arg("gyros"),
          R"docs(
These are a set of bodies, and offsets in local body space where gyros
are mounted on the body
    )docs")
      .def(
          "getAccMapReadings",
          &dart::dynamics::Skeleton::getAccMapReadings,
          ::py::arg("accelerometers"),
          R"docs(
These are a set of bodies, and offsets in local body space where gyros
are mounted on the body
    )docs")
      .def(
          "convertSensorMap",
          &dart::dynamics::Skeleton::convertSensorMap,
          ::py::arg("sensorMap"),
          ::py::arg("warnOnDrop") = true,
          R"docs(
This converts markers from a source skeleton to the current, doing a
simple mapping based on body node names. Any markers that don't find a
body node in the current skeleton with the same name are dropped.
    )docs")
      .def(
          "getGyroReadings",
          &dart::dynamics::Skeleton::getGyroReadings,
          ::py::arg("gyros"),
          R"docs(
These are a set of bodies, and offsets in local body space where gyros are mounted on the body.
    )docs")
      .def(
          "getGyroReadingsJacobianWrt",
          &dart::dynamics::Skeleton::getGyroReadingsJacobianWrt,
          ::py::arg("gyros"),
          ::py::arg("wrt"),
          R"docs(
This returns the Jacobian relating changes in the `wrt` quantity to changes in gyro readings.
    )docs")
      .def(
          "getAccelerometerReadings",
          &dart::dynamics::Skeleton::getAccelerometerReadings,
          ::py::arg("accelerometers"),
          R"docs(
These are a set of bodies, and offsets in local body space where accs are mounted on the body.
    )docs")
      .def(
          "getAccelerometerReadingsJacobianWrt",
          &dart::dynamics::Skeleton::getAccelerometerReadingsJacobianWrt,
          ::py::arg("accs"),
          ::py::arg("wrt"),
          R"docs(
This returns the Jacobian relating changes in the `wrt` quantity to changes in acc readings.
    )docs")
      .def(
          "getMagnetometerReadings",
          &dart::dynamics::Skeleton::getMagnetometerReadings,
          ::py::arg("mags"),
          ::py::arg("magneticField"),
          R"docs(
These are a set of bodies, and offsets in local body space where magnetometers are mounted on the body.
    )docs")
      .def(
          "getMagnetometerReadingsJacobianWrt",
          &dart::dynamics::Skeleton::getMagnetometerReadingsJacobianWrt,
          ::py::arg("mags"),
          ::py::arg("magneticField"),
          ::py::arg("wrt"),
          R"docs(
This returns the Jacobian relating changes in the `wrt` quantity to changes in mag readings.
    )docs")
      .def(
          "getMagnetometerReadingsJacobianWrtMagneticField",
          &dart::dynamics::Skeleton::
              getMagnetometerReadingsJacobianWrtMagneticField,
          ::py::arg("mags"),
          ::py::arg("magneticField"),
          R"docs(
This returns the Jacobian relating changes in the magnetic field to changes in mag readings.
    )docs")
      .def(
          "getBodyLocalVelocities",
          &dart::dynamics::Skeleton::getBodyLocalVelocities)
      .def(
          "getBodyLocalAccelerations",
          &dart::dynamics::Skeleton::getBodyLocalAccelerations)
      .def(
          "setVelocityUpperLimits",
          +[](dart::dynamics::Skeleton* self, Eigen::VectorXs limits) -> void {
            self->setVelocityUpperLimits(limits);
          })
      .def(
          "setVelocityLowerLimits",
          +[](dart::dynamics::Skeleton* self, Eigen::VectorXs limits) -> void {
            self->setVelocityLowerLimits(limits);
          })
      .def(
          "integratePositions",
          +[](dart::dynamics::Skeleton* self, s_t _dt) -> void {
            return self->integratePositions(_dt);
          },
          ::py::arg("dt"))
      .def(
          "integratePositionsExplicit",
          +[](dart::dynamics::Skeleton* self,
              Eigen::VectorXs _pos,
              Eigen::VectorXs _vel,
              s_t _dt) -> Eigen::VectorXs {
            return self->integratePositionsExplicit(_pos, _vel, _dt);
          },
          ::py::arg("pos"),
          ::py::arg("vel"),
          ::py::arg("dt"))
      .def(
          "integrateVelocities",
          +[](dart::dynamics::Skeleton* self, s_t _dt) -> void {
            return self->integrateVelocities(_dt);
          },
          ::py::arg("dt"))

      .def(
          "getPositionDifferences",
          +[](const dart::dynamics::Skeleton* self,
              const Eigen::VectorXs& _q2,
              const Eigen::VectorXs& _q1) -> Eigen::VectorXs {
            return self->getPositionDifferences(_q2, _q1);
          },
          ::py::arg("q2"),
          ::py::arg("q1"))
      .def(
          "getVelocityDifferences",
          +[](const dart::dynamics::Skeleton* self,
              const Eigen::VectorXs& _dq2,
              const Eigen::VectorXs& _dq1) -> Eigen::VectorXs {
            return self->getVelocityDifferences(_dq2, _dq1);
          },
          ::py::arg("dq2"),
          ::py::arg("dq1"))
      .def(
          "unwrapPositionToNearest",
          &dart::dynamics::Skeleton::unwrapPositionToNearest,
          ::py::arg("thisPos"),
          ::py::arg("lastPos"))
      .def(
          "getInverseDynamics",
          &dart::dynamics::Skeleton::getInverseDynamics,
          ::py::arg("accelerations"))
      .def(
          "getInverseDynamicsFromPredictions",
          &dart::dynamics::Skeleton::getInverseDynamicsFromPredictions,
          ::py::arg("accelerations"),
          ::py::arg("contactBodies"),
          ::py::arg("rootFrameContactWrench"),
          ::py::arg("rootResiduals"))
      .def(
          "getContactInverseDynamics",
          &dart::dynamics::Skeleton::getContactInverseDynamics,
          ::py::arg("accelerations"),
          ::py::arg("contactBody"))
      .def(
          "getMultipleContactInverseDynamics",
          &dart::dynamics::Skeleton::getMultipleContactInverseDynamics,
          ::py::arg("accelerations"),
          ::py::arg("contactBodies"),
          ::py::arg("bodyWrenchGuesses") = std::vector<Eigen::Vector6s>())
      .def(
          "getMultipleContactInverseDynamicsOverTime",
          &dart::dynamics::Skeleton::getMultipleContactInverseDynamicsOverTime,
          ::py::arg("positions"),
          ::py::arg("contactBodies"),
          ::py::arg("smoothingWeight"),
          ::py::arg("minTorqueWeight"),
          ::py::arg(
              "velocityPenalty"), // Having a default arg lambda seems to break
                                  // pybind11? = [](double) { return 0.0; },
          ::py::arg("prevContactForces") = std::vector<Eigen::Vector6s>(),
          ::py::arg("prevContactWeight") = 0.0,
          ::py::arg("magnitudeCosts") = dart::dynamics::Skeleton::EMPTY)
      .def(
          "getEnergyAccounting",
          &dart::dynamics::Skeleton::getEnergyAccounting,
          ::py::arg("heightAtZeroPoint") = 0,
          ::py::arg("referenceFrameVelocity") = Eigen::Vector3s::Zero(),
          ::py::arg("contactBodies") = std::vector<dynamics::BodyNode*>(),
          ::py::arg("cops") = std::vector<Eigen::Vector3s>(),
          ::py::arg("forces") = std::vector<Eigen::Vector3s>(),
          ::py::arg("moments") = std::vector<Eigen::Vector3s>())
      .def(
          "getSupportVersion",
          +[](const dart::dynamics::Skeleton* self) -> std::size_t {
            return self->getSupportVersion();
          })
      .def(
          "getSupportVersion",
          +[](const dart::dynamics::Skeleton* self, std::size_t _treeIdx)
              -> std::size_t { return self->getSupportVersion(_treeIdx); },
          ::py::arg("treeIdx"))
      .def(
          "computeForwardKinematics",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->computeForwardKinematics();
          })
      .def(
          "computeForwardKinematics",
          +[](dart::dynamics::Skeleton* self, bool _updateTransforms) -> void {
            return self->computeForwardKinematics(_updateTransforms);
          },
          ::py::arg("updateTransforms"))
      .def(
          "computeForwardKinematics",
          +[](dart::dynamics::Skeleton* self,
              bool _updateTransforms,
              bool _updateVels) -> void {
            return self->computeForwardKinematics(
                _updateTransforms, _updateVels);
          },
          ::py::arg("updateTransforms"),
          ::py::arg("updateVels"))
      .def(
          "computeForwardKinematics",
          +[](dart::dynamics::Skeleton* self,
              bool _updateTransforms,
              bool _updateVels,
              bool _updateAccs) -> void {
            return self->computeForwardKinematics(
                _updateTransforms, _updateVels, _updateAccs);
          },
          ::py::arg("updateTransforms"),
          ::py::arg("updateVels"),
          ::py::arg("updateAccs"))
      .def(
          "computeForwardDynamics",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->computeForwardDynamics();
          })
      .def(
          "computeInverseDynamics",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->computeInverseDynamics();
          })
      .def(
          "computeInverseDynamics",
          +[](dart::dynamics::Skeleton* self,
              bool _withExternalForces) -> void {
            return self->computeInverseDynamics(_withExternalForces);
          },
          ::py::arg("withExternalForces"))
      .def(
          "computeInverseDynamics",
          +[](dart::dynamics::Skeleton* self,
              bool _withExternalForces,
              bool _withDampingForces) -> void {
            return self->computeInverseDynamics(
                _withExternalForces, _withDampingForces);
          },
          ::py::arg("withExternalForces"),
          ::py::arg("withDampingForces"))
      .def(
          "computeInverseDynamics",
          +[](dart::dynamics::Skeleton* self,
              bool _withExternalForces,
              bool _withDampingForces,
              bool _withSpringForces) -> void {
            return self->computeInverseDynamics(
                _withExternalForces, _withDampingForces, _withSpringForces);
          },
          ::py::arg("withExternalForces"),
          ::py::arg("withDampingForces"),
          ::py::arg("withSpringForces"))
      .def(
          "clearConstraintImpulses",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->clearConstraintImpulses();
          })
      .def(
          "updateBiasImpulse",
          +[](dart::dynamics::Skeleton* self,
              dart::dynamics::BodyNode* _bodyNode) -> void {
            return self->updateBiasImpulse(_bodyNode);
          },
          ::py::arg("bodyNode"))
      .def(
          "updateBiasImpulse",
          +[](dart::dynamics::Skeleton* self,
              dart::dynamics::BodyNode* _bodyNode,
              const Eigen::Vector6s& _imp) -> void {
            return self->updateBiasImpulse(_bodyNode, _imp);
          },
          ::py::arg("bodyNode"),
          ::py::arg("imp"))
      .def(
          "updateBiasImpulse",
          +[](dart::dynamics::Skeleton* self,
              dart::dynamics::BodyNode* _bodyNode1,
              const Eigen::Vector6s& _imp1,
              dart::dynamics::BodyNode* _bodyNode2,
              const Eigen::Vector6s& _imp2) -> void {
            return self->updateBiasImpulse(
                _bodyNode1, _imp1, _bodyNode2, _imp2);
          },
          ::py::arg("bodyNode1"),
          ::py::arg("imp1"),
          ::py::arg("bodyNode2"),
          ::py::arg("imp2"))
      /*
        .def(
            "updateBiasImpulse",
            +[](dart::dynamics::Skeleton* self,
                dart::dynamics::SoftBodyNode* _softBodyNode,
                dart::dynamics::PointMass* _pointMass,
                const Eigen::Vector3s& _imp) -> void {
              return self->updateBiasImpulse(_softBodyNode, _pointMass, _imp);
            },
            ::py::arg("softBodyNode"),
            ::py::arg("pointMass"),
            ::py::arg("imp"))
            */
      .def(
          "updateVelocityChange",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->updateVelocityChange();
          })
      .def(
          "setImpulseApplied",
          +[](dart::dynamics::Skeleton* self, bool _val) -> void {
            return self->setImpulseApplied(_val);
          },
          ::py::arg("val"))
      .def(
          "isImpulseApplied",
          +[](const dart::dynamics::Skeleton* self) -> bool {
            return self->isImpulseApplied();
          })
      .def(
          "computeImpulseForwardDynamics",
          +[](dart::dynamics::Skeleton* self) -> void {
            return self->computeImpulseForwardDynamics();
          })
      .def(
          "getJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::Jacobian { return self->getJacobian(_node); },
          ::py::arg("node"))
      .def(
          "getJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobian(_node, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset) -> dart::math::Jacobian {
            return self->getJacobian(_node, _localOffset);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"))
      .def(
          "getJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobian(_node, _localOffset, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getWorldJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::Jacobian { return self->getWorldJacobian(_node); },
          ::py::arg("node"))
      .def(
          "getWorldJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset) -> dart::math::Jacobian {
            return self->getWorldJacobian(_node, _localOffset);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobian(_node);
          },
          ::py::arg("node"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobian(_node, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobian(_node, _localOffset);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobian(
                _node, _localOffset, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getLinearJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset,
              const Eigen::Matrix3s& _localRotation)
              -> dart::math::LinearJacobian {
            return _localRotation
                   * self->getLinearJacobian(_node, _localOffset);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"),
          ::py::arg("rotation"))
      .def(
          "getAngularJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::AngularJacobian {
            return self->getAngularJacobian(_node);
          },
          ::py::arg("node"))
      .def(
          "getAngularJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::AngularJacobian {
            return self->getAngularJacobian(_node, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getAngularJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Matrix3s& _rotation) -> dart::math::AngularJacobian {
            return _rotation * self->getAngularJacobian(_node);
          },
          ::py::arg("node"),
          ::py::arg("rotation"))
      .def(
          "getJacobianSpatialDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::Jacobian {
            return self->getJacobianSpatialDeriv(_node);
          },
          ::py::arg("node"))
      .def(
          "getJacobianSpatialDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianSpatialDeriv(_node, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobianSpatialDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset) -> dart::math::Jacobian {
            return self->getJacobianSpatialDeriv(_node, _localOffset);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"))
      .def(
          "getJacobianSpatialDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianSpatialDeriv(
                _node, _localOffset, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobianClassicDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::Jacobian {
            return self->getJacobianClassicDeriv(_node);
          },
          ::py::arg("node"))
      .def(
          "getJacobianClassicDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianClassicDeriv(_node, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getJacobianClassicDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset) -> dart::math::Jacobian {
            return self->getJacobianClassicDeriv(_node, _localOffset);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"))
      .def(
          "getJacobianClassicDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getJacobianClassicDeriv(
                _node, _localOffset, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv(_node);
          },
          ::py::arg("node"))
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv(_node, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv(_node, _localOffset);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"))
      .def(
          "getLinearJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const Eigen::Vector3s& _localOffset,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getLinearJacobianDeriv(
                _node, _localOffset, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("localOffset"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getAngularJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node)
              -> dart::math::AngularJacobian {
            return self->getAngularJacobianDeriv(_node);
          },
          ::py::arg("node"))
      .def(
          "getAngularJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::JacobianNode* _node,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::AngularJacobian {
            return self->getAngularJacobianDeriv(_node, _inCoordinatesOf);
          },
          ::py::arg("node"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getMass",
          +[](const dart::dynamics::Skeleton* self)
              -> s_t { return self->getMass(); })
      .def(
          "getMassMatrix",
          +[](const dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::MatrixXs& {
            return self->getMassMatrix(treeIndex);
          })
      // TODO(JS): Redefining get[~]() that are already defined in MetaSkeleton.
      // We need this because the methods with same name (but different
      // arguments) are hidden. Update (or remove) once following issue is
      // resolved: https://github.com/pybind/pybind11/issues/974
      .def(
          "getMassMatrix",
          +[](const dart::dynamics::Skeleton* self) -> const Eigen::MatrixXs& {
            return self->getMassMatrix();
          })
      .def(
          "getAugMassMatrix",
          +[](const dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::MatrixXs& {
            return self->getAugMassMatrix(treeIndex);
          })
      .def(
          "getAugMassMatrix",
          +[](const dart::dynamics::Skeleton* self) -> const Eigen::MatrixXs& {
            return self->getAugMassMatrix();
          })
      .def(
          "getInvMassMatrix",
          +[](const dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::MatrixXs& {
            return self->getInvMassMatrix(treeIndex);
          })
      .def(
          "getInvMassMatrix",
          +[](const dart::dynamics::Skeleton* self) -> const Eigen::MatrixXs& {
            return self->getInvMassMatrix();
          })
      .def(
          "getCoriolisForces",
          +[](dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::VectorXs& {
            return self->getCoriolisForces(treeIndex);
          })
      .def(
          "getCoriolisForces",
          +[](dart::dynamics::Skeleton* self) -> const Eigen::VectorXs& {
            return self->getCoriolisForces();
          })
      .def(
          "getGravityForces",
          +[](dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::VectorXs& {
            return self->getCoriolisForces(treeIndex);
          })
      .def(
          "getGravityForces",
          +[](dart::dynamics::Skeleton* self) -> const Eigen::VectorXs& {
            return self->getCoriolisForces();
          })
      .def(
          "getCoriolisAndGravityForces",
          +[](dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::VectorXs& {
            return self->getCoriolisAndGravityForces(treeIndex);
          })
      .def(
          "getCoriolisAndGravityForces",
          +[](dart::dynamics::Skeleton* self) -> const Eigen::VectorXs& {
            return self->getCoriolisAndGravityForces();
          })
      .def(
          "getExternalForces",
          +[](dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::VectorXs& {
            return self->getCoriolisAndGravityForces(treeIndex);
          })
      .def(
          "getExternalForces",
          +[](dart::dynamics::Skeleton* self) -> const Eigen::VectorXs& {
            return self->getCoriolisAndGravityForces();
          })
      .def(
          "getConstraintForces",
          +[](dart::dynamics::Skeleton* self,
              std::size_t treeIndex) -> const Eigen::VectorXs& {
            return self->getConstraintForces(treeIndex);
          })
      .def(
          "getConstraintForces",
          +[](dart::dynamics::Skeleton* self) -> const Eigen::VectorXs& {
            return self->getConstraintForces();
          })
      .def(
          "clearExternalForces",
          +[](dart::dynamics::Skeleton* self)
              -> void { return self->clearExternalForces(); })
      .def(
          "clearInternalForces",
          +[](dart::dynamics::Skeleton* self)
              -> void { return self->clearInternalForces(); })
      //      .def("notifyArticulatedInertiaUpdate",
      //      +[](dart::dynamics::Skeleton *self, std::size_t _treeIdx) -> void
      //      { return self->notifyArticulatedInertiaUpdate(_treeIdx); },
      //      ::py::arg("treeIdx"))
      .def(
          "dirtyArticulatedInertia",
          +[](dart::dynamics::Skeleton* self, std::size_t _treeIdx) -> void {
            return self->dirtyArticulatedInertia(_treeIdx);
          },
          ::py::arg("treeIdx"))
      //      .def("notifySupportUpdate", +[](dart::dynamics::Skeleton *self,
      //      std::size_t _treeIdx) -> void { return
      //      self->notifySupportUpdate(_treeIdx); },
      //      ::py::arg("treeIdx"))
      .def(
          "dirtySupportPolygon",
          +[](dart::dynamics::Skeleton* self, std::size_t _treeIdx) -> void {
            return self->dirtySupportPolygon(_treeIdx);
          },
          ::py::arg("treeIdx"))
      .def(
          "computeKineticEnergy",
          +[](const dart::dynamics::Skeleton* self) -> s_t {
            return self->computeKineticEnergy();
          })
      .def(
          "computePotentialEnergy",
          +[](const dart::dynamics::Skeleton* self) -> s_t {
            return self->computePotentialEnergy();
          })
      .def(
          "clearCollidingBodies",
          +[](dart::dynamics::Skeleton* self)
              -> void { return self->clearCollidingBodies(); })
      .def(
          "getCOM",
          +[](const dart::dynamics::Skeleton* self) -> Eigen::Vector3s {
            return self->getCOM();
          })
      .def(
          "getCOM",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _withRespectTo) -> Eigen::Vector3s {
            return self->getCOM(_withRespectTo);
          },
          ::py::arg("withRespectTo"))
      .def(
          "getCOMSpatialVelocity",
          +[](const dart::dynamics::Skeleton* self) -> Eigen::Vector6s {
            return self->getCOMSpatialVelocity();
          })
      .def(
          "getCOMSpatialVelocity",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo) -> Eigen::Vector6s {
            return self->getCOMSpatialVelocity(_relativeTo);
          },
          ::py::arg("relativeTo"))
      .def(
          "getCOMSpatialVelocity",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector6s {
            return self->getCOMSpatialVelocity(_relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMLinearVelocity",
          +[](const dart::dynamics::Skeleton* self) -> Eigen::Vector3s {
            return self->getCOMLinearVelocity();
          })
      .def(
          "getCOMLinearVelocity",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo) -> Eigen::Vector3s {
            return self->getCOMLinearVelocity(_relativeTo);
          },
          ::py::arg("relativeTo"))
      .def(
          "getCOMLinearVelocity",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector3s {
            return self->getCOMLinearVelocity(_relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMSpatialAcceleration",
          +[](const dart::dynamics::Skeleton* self) -> Eigen::Vector6s {
            return self->getCOMSpatialAcceleration();
          })
      .def(
          "getCOMSpatialAcceleration",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo) -> Eigen::Vector6s {
            return self->getCOMSpatialAcceleration(_relativeTo);
          },
          ::py::arg("relativeTo"))
      .def(
          "getCOMSpatialAcceleration",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector6s {
            return self->getCOMSpatialAcceleration(
                _relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMLinearAcceleration",
          +[](const dart::dynamics::Skeleton* self) -> Eigen::Vector3s {
            return self->getCOMLinearAcceleration();
          })
      .def(
          "getCOMLinearAcceleration",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo) -> Eigen::Vector3s {
            return self->getCOMLinearAcceleration(_relativeTo);
          },
          ::py::arg("relativeTo"))
      .def(
          "getCOMLinearAcceleration",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _relativeTo,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> Eigen::Vector3s {
            return self->getCOMLinearAcceleration(
                _relativeTo, _inCoordinatesOf);
          },
          ::py::arg("relativeTo"),
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMJacobian",
          +[](const dart::dynamics::Skeleton* self) -> dart::math::Jacobian {
            return self->getCOMJacobian();
          })
      .def(
          "getCOMJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getCOMJacobian(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMLinearJacobian",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::math::LinearJacobian {
            return self->getCOMLinearJacobian();
          })
      .def(
          "getCOMLinearJacobian",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getCOMLinearJacobian(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMJacobianSpatialDeriv",
          +[](const dart::dynamics::Skeleton* self) -> dart::math::Jacobian {
            return self->getCOMJacobianSpatialDeriv();
          })
      .def(
          "getCOMJacobianSpatialDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::Jacobian {
            return self->getCOMJacobianSpatialDeriv(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "getCOMLinearJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self)
              -> dart::math::LinearJacobian {
            return self->getCOMLinearJacobianDeriv();
          })
      .def(
          "getCOMLinearJacobianDeriv",
          +[](const dart::dynamics::Skeleton* self,
              const dart::dynamics::Frame* _inCoordinatesOf)
              -> dart::math::LinearJacobian {
            return self->getCOMLinearJacobianDeriv(_inCoordinatesOf);
          },
          ::py::arg("inCoordinatesOf"))
      .def(
          "resetUnion",
          +[](dart::dynamics::Skeleton* self)
              -> void { return self->resetUnion(); })
      //   .def_readwrite(
      //       "mUnionRootSkeleton",
      //       &dart::dynamics::Skeleton::mUnionRootSkeleton)
      .def_readwrite("mUnionSize", &dart::dynamics::Skeleton::mUnionSize)
      .def_readwrite("mUnionIndex", &dart::dynamics::Skeleton::mUnionIndex);
}

} // namespace python
} // namespace dart
