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

#ifndef DART_DYNAMICS_SKELETON_HPP_
#define DART_DYNAMICS_SKELETON_HPP_

#include <functional>
#include <memory>
#include <mutex>

#include "dart/common/NameManager.hpp"
#include "dart/common/VersionCounter.hpp"
#include "dart/dynamics/EndEffector.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Marker.hpp"
#include "dart/dynamics/MetaSkeleton.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/SmartPointer.hpp"
#include "dart/dynamics/SpecializedNodeManager.hpp"
#include "dart/dynamics/detail/BodyNodeAspect.hpp"
#include "dart/dynamics/detail/SkeletonAspect.hpp"
#include "dart/math/IKSolver.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/neural/WithRespectTo.hpp"

namespace dart {

namespace neural {
class ConstrainedGroupGradientMatrices;
}

namespace dynamics {

typedef std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
    MarkerMap;

typedef std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>
    SensorMap;

struct BodyScaleGroup
{
  std::vector<dynamics::BodyNode*> nodes;
  std::vector<Eigen::Vector3s> flipAxis;
  bool uniformScaling;
};

struct BodyScaleGroupAndIndex
{
  struct BodyScaleGroup& group;
  int axis;
  BodyScaleGroupAndIndex(BodyScaleGroup& group, int axis);
};

/// class Skeleton
class Skeleton : public virtual common::VersionCounter,
                 public MetaSkeleton,
                 public SkeletonSpecializedFor<ShapeNode, EndEffector, Marker>,
                 public detail::SkeletonAspectBase
{
public:
  static Eigen::Matrix<s_t, Eigen::Dynamic, Eigen::Dynamic> EMPTY;

  // Some of non-virtual functions of MetaSkeleton are hidden because of the
  // functions of the same name in this class. We expose those functions as
  // follows.
  using MetaSkeleton::getAngularJacobian;
  using MetaSkeleton::getAngularJacobianDeriv;
  using MetaSkeleton::getJacobian;
  using MetaSkeleton::getJacobianClassicDeriv;
  using MetaSkeleton::getJacobianSpatialDeriv;
  using MetaSkeleton::getLinearJacobian;
  using MetaSkeleton::getLinearJacobianDeriv;

  using AspectPropertiesData = detail::SkeletonAspectProperties;
  using AspectProperties = common::Aspect::MakeProperties<AspectPropertiesData>;

  using State = common::Composite::State;
  using Properties = common::Composite::Properties;

  enum ConfigFlags
  {
    CONFIG_NOTHING = 0,
    CONFIG_POSITIONS = 1 << 1,
    CONFIG_VELOCITIES = 1 << 2,
    CONFIG_ACCELERATIONS = 1 << 3,
    CONFIG_FORCES = 1 << 4,
    CONFIG_COMMANDS = 1 << 5,
    CONFIG_ALL = 0xFF
  };

  /// The Configuration struct represents the joint configuration of a Skeleton.
  /// The size of each Eigen::VectorXs member in this struct must be equal to
  /// the number of degrees of freedom in the Skeleton or it must be zero. We
  /// assume that any Eigen::VectorXs member with zero entries should be
  /// ignored.
  struct Configuration
  {
    Configuration(
        const Eigen::VectorXs& positions = Eigen::VectorXs(),
        const Eigen::VectorXs& velocities = Eigen::VectorXs(),
        const Eigen::VectorXs& accelerations = Eigen::VectorXs(),
        const Eigen::VectorXs& forces = Eigen::VectorXs(),
        const Eigen::VectorXs& commands = Eigen::VectorXs());

    Configuration(
        const std::vector<std::size_t>& indices,
        const Eigen::VectorXs& positions = Eigen::VectorXs(),
        const Eigen::VectorXs& velocities = Eigen::VectorXs(),
        const Eigen::VectorXs& accelerations = Eigen::VectorXs(),
        const Eigen::VectorXs& forces = Eigen::VectorXs(),
        const Eigen::VectorXs& commands = Eigen::VectorXs());

    /// A list of degree of freedom indices that each entry in the
    /// Eigen::VectorXs members correspond to.
    std::vector<std::size_t> mIndices;

    /// Joint positions
    Eigen::VectorXs mPositions;

    /// Joint velocities
    Eigen::VectorXs mVelocities;

    /// Joint accelerations
    Eigen::VectorXs mAccelerations;

    /// Joint forces
    Eigen::VectorXs mControlForces;

    /// Joint commands
    Eigen::VectorXs mCommands;

    /// Equality comparison operator
    bool operator==(const Configuration& other) const;

    /// Inequality comparison operator
    bool operator!=(const Configuration& other) const;
  };

  //----------------------------------------------------------------------------
  /// \{ \name Constructor and Destructor
  //----------------------------------------------------------------------------

  /// Create a new Skeleton inside of a shared_ptr
  static SkeletonPtr create(const std::string& _name = "Skeleton");

  /// Create a new Skeleton inside of a shared_ptr
  static SkeletonPtr create(const AspectPropertiesData& properties);

  /// Get the shared_ptr that manages this Skeleton
  SkeletonPtr getPtr();

  /// Get the shared_ptr that manages this Skeleton
  ConstSkeletonPtr getPtr() const;

  /// Same as getPtr(), but this allows Skeleton to have a similar interface as
  /// BodyNode and Joint for template programming.
  SkeletonPtr getSkeleton();

  /// Same as getPtr(), but this allows Skeleton to have a similar interface as
  /// BodyNode and Joint for template programming.
  ConstSkeletonPtr getSkeleton() const;

  /// Get the mutex that protects the state of this Skeleton
  std::mutex& getMutex() const;

  /// Get the mutex that protects the state of this Skeleton
  std::unique_ptr<common::LockableReference> getLockableReference()
      const override;

  Skeleton(const Skeleton&) = delete;

  /// Destructor
  virtual ~Skeleton();

  /// Remove copy operator
  Skeleton& operator=(const Skeleton& _other) = delete;

  /// Create an identical clone of this Skeleton.
  /// \deprecated Deprecated in DART 6.7. Please use cloneSkeleton() instead.
  DART_DEPRECATED(6.7)
  SkeletonPtr clone() const;
  // TODO: In DART 7, change this function to override MetaSkeleton::clone()
  // that returns MetaSkeletonPtr

  /// Create an identical clone of this Skeleton, except that it has a new name.
  /// \deprecated Deprecated in DART 6.7. Please use cloneSkeleton() instead.
  DART_DEPRECATED(6.7)
  SkeletonPtr clone(const std::string& cloneName) const;
  // TODO: In DART 7, change this function to override MetaSkeleton::clone()
  // that returns MetaSkeletonPtr

  /// Creates and returns a clone of this Skeleton.
  SkeletonPtr cloneSkeleton() const;

  /// Creates and returns a clone of this Skeleton.
  SkeletonPtr cloneSkeleton(const std::string& cloneName) const;

  /// Creates and returns a clone of this Skeleton, where we merge the provided
  /// bodies together and approximate the CustomJoints with simpler joint types.
  SkeletonPtr simplifySkeleton(
      const std::string& cloneName,
      std::map<std::string, std::string> mergeBodiesInto) const;

  // Documentation inherited
  MetaSkeletonPtr cloneMetaSkeleton(
      const std::string& cloneName) const override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Configuration
  //----------------------------------------------------------------------------

  /// Set the configuration of this Skeleton
  void setConfiguration(const Configuration& configuration);

  /// Get the configuration of this Skeleton
  Configuration getConfiguration(int flags = CONFIG_ALL) const;

  /// Get the configuration of the specified indices in this Skeleton
  Configuration getConfiguration(
      const std::vector<std::size_t>& indices, int flags = CONFIG_ALL) const;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name State
  //----------------------------------------------------------------------------

  /// Set the State of this Skeleton [alias for setCompositeState(~)]
  void setState(const State& state);

  /// Get the State of this Skeleton [alias for getCompositeState()]
  State getState() const;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Properties
  //----------------------------------------------------------------------------

  /// Set all properties of this Skeleton
  void setProperties(const Properties& properties);

  /// Get all properties of this Skeleton
  Properties getProperties() const;

  /// Set the Properties of this Skeleton
  void setProperties(const AspectProperties& properties);

  /// Get the Properties of this Skeleton
  DART_DEPRECATED(6.0)
  const AspectProperties& getSkeletonProperties() const;

  /// Set the AspectProperties of this Skeleton
  void setAspectProperties(const AspectProperties& properties);

  /// Set name.
  const std::string& setName(const std::string& _name) override;

  /// Get name.
  const std::string& getName() const override;

  /// Deprecated. Please use enableSelfCollisionCheck() and
  /// setAdjacentBodyCheck() instead.
  DART_DEPRECATED(6.0)
  void enableSelfCollision(bool enableAdjacentBodyCheck = false);

  /// Deprecated. Please use disableSelfCollisionCheck() instead.
  DART_DEPRECATED(6.0)
  void disableSelfCollision();

  /// Set whether to check self-collision.
  void setSelfCollisionCheck(bool enable);

  /// Return whether self-collision check is enabled.
  bool getSelfCollisionCheck() const;

  /// Enable self-collision check.
  void enableSelfCollisionCheck();

  /// Disable self-collision check.
  void disableSelfCollisionCheck();

  /// Return true if self-collision check is enabled
  bool isEnabledSelfCollisionCheck() const;

  /// Set whether to check adjacent bodies. This option is effective only when
  /// the self-collision check is enabled.
  void setAdjacentBodyCheck(bool enable);

  /// Return whether adjacent body check is enabled.
  bool getAdjacentBodyCheck() const;

  /// Enable collision check for adjacent bodies. This option is effective only
  /// when the self-collision check is enabled.
  void enableAdjacentBodyCheck();

  /// Disable collision check for adjacent bodies. This option is effective only
  /// when the self-collision check is enabled.
  void disableAdjacentBodyCheck();

  /// Return true if self-collision check is enabled including adjacent bodies.
  bool isEnabledAdjacentBodyCheck() const;

  /// Set whether this skeleton will be updated by forward dynamics.
  /// \param[in] _isMobile True if this skeleton is mobile.
  void setMobile(bool _isMobile);

  /// Get whether this skeleton will be updated by forward dynamics.
  /// \return True if this skeleton is mobile.
  bool isMobile() const;

  /// Set time step. This timestep is used for implicit joint damping
  /// force.
  void setTimeStep(s_t _timeStep);

  /// Get time step.
  s_t getTimeStep() const;

  /// Set 3-dim gravitational acceleration. The gravity is used for
  /// calculating gravity force vector of the skeleton.
  void setGravity(const Eigen::Vector3s& _gravity);

  /// Get 3-dim gravitational acceleration.
  const Eigen::Vector3s& getGravity() const;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Structural Properties
  //----------------------------------------------------------------------------

#ifdef _WIN32
  template <typename JointType>
  static typename JointType::Properties createJointProperties()
  {
    return typename JointType::Properties();
  }

  template <typename NodeType>
  static typename NodeType::Properties createBodyNodeProperties()
  {
    return typename NodeType::Properties();
  }
#endif
  // TODO: Workaround for MSVC bug on template function specialization with
  // default argument. Please see #487 for detail

  /// Create a Joint and child BodyNode pair of the given types. When creating
  /// a root (parentless) BodyNode, pass in nullptr for the _parent argument.
  template <class JointType, class NodeType = BodyNode>
  std::pair<JointType*, NodeType*> createJointAndBodyNodePair(
      BodyNode* _parent = nullptr,
#ifdef _WIN32
      const typename JointType::Properties& _jointProperties
      = Skeleton::createJointProperties<JointType>(),
      const typename NodeType::Properties& _bodyProperties
      = Skeleton::createBodyNodeProperties<NodeType>());
#else
      const typename JointType::Properties& _jointProperties
      = typename JointType::Properties(),
      const typename NodeType::Properties& _bodyProperties
      = typename NodeType::Properties());
#endif
  // TODO: Workaround for MSVC bug on template function specialization with
  // default argument. Please see #487 for detail

  // Documentation inherited
  std::size_t getNumBodyNodes() const override;

  /// Get number of rigid body nodes.
  std::size_t getNumRigidBodyNodes() const;

  /// Get number of soft body nodes.
  std::size_t getNumSoftBodyNodes() const;

  /// Get the number of independent trees that this Skeleton contains
  std::size_t getNumTrees() const;

  /// Get the root BodyNode of the tree whose index in this Skeleton is _treeIdx
  BodyNode* getRootBodyNode(std::size_t _treeIdx = 0);

  /// Get the const root BodyNode of the tree whose index in this Skeleton is
  /// _treeIdx
  const BodyNode* getRootBodyNode(std::size_t _treeIdx = 0) const;

  /// Get the root Joint of the tree whose index in this Skeleton is treeIdx
  Joint* getRootJoint(std::size_t treeIdx = 0u);

  /// Get the const root Joint of the tree whose index in this Skeleton is
  /// treeIdx
  const Joint* getRootJoint(std::size_t treeIdx = 0u) const;

  // Documentation inherited
  BodyNode* getBodyNode(std::size_t _idx) override;

  // Documentation inherited
  const BodyNode* getBodyNode(std::size_t _idx) const override;

  /// Get SoftBodyNode whose index is _idx
  SoftBodyNode* getSoftBodyNode(std::size_t _idx);

  /// Get const SoftBodyNode whose index is _idx
  const SoftBodyNode* getSoftBodyNode(std::size_t _idx) const;

  // Documentation inherited
  BodyNode* getBodyNode(const std::string& name) override;

  // Documentation inherited
  const BodyNode* getBodyNode(const std::string& name) const override;

  /// Get soft body node whose name is _name
  SoftBodyNode* getSoftBodyNode(const std::string& _name);

  /// Get const soft body node whose name is _name
  const SoftBodyNode* getSoftBodyNode(const std::string& _name) const;

  // Documentation inherited
  const std::vector<BodyNode*>& getBodyNodes() override;

  // Documentation inherited
  const std::vector<const BodyNode*>& getBodyNodes() const override;

  /// \copydoc MetaSkeleton::getBodyNodes(const std::string&).
  ///
  /// \note Skeleton always guarantees name uniqueness for BodyNodes and Joints.
  /// So this function returns the single BodyNode of the given name if it
  /// exists.
  std::vector<BodyNode*> getBodyNodes(const std::string& name) override;

  /// \copydoc MetaSkeleton::getBodyNodes(const std::string&).
  ///
  /// \note Skeleton always guarantees name uniqueness for BodyNodes and Joints.
  /// So this function returns the single BodyNode of the given name if it
  /// exists.
  std::vector<const BodyNode*> getBodyNodes(
      const std::string& name) const override;

  // Documentation inherited
  bool hasBodyNode(const BodyNode* bodyNode) const override;

  // Documentation inherited
  std::size_t getIndexOf(
      const BodyNode* _bn, bool _warning = true) const override;

  /// Get the BodyNodes belonging to a tree in this Skeleton
  const std::vector<BodyNode*>& getTreeBodyNodes(std::size_t _treeIdx);

  /// Get the BodyNodes belonging to a tree in this Skeleton
  std::vector<const BodyNode*> getTreeBodyNodes(std::size_t _treeIdx) const;

  // Documentation inherited
  std::size_t getNumJoints() const override;

  // Documentation inherited
  Joint* getJoint(std::size_t _idx) override;

  // Gets the index in the skeleton where this joint lives
  int getJointIndex(const Joint* joint);

  // Documentation inherited
  const Joint* getJoint(std::size_t _idx) const override;

  // Documentation inherited
  Joint* getJoint(const std::string& name) override;

  // Documentation inherited
  const Joint* getJoint(const std::string& name) const override;

  // Documentation inherited
  std::vector<Joint*> getJoints() override;

  // Documentation inherited
  std::vector<const Joint*> getJoints() const override;

  /// \copydoc MetaSkeleton::getJoints(const std::string&).
  ///
  /// \note Skeleton always guarantees name uniqueness for BodyNodes and Joints.
  /// So this function returns the single Joint of the given name if it exists.
  std::vector<Joint*> getJoints(const std::string& name) override;

  /// \copydoc MetaSkeleton::getJoints(const std::string&).
  ///
  /// \note Skeleton always guarantees name uniqueness for BodyNodes and Joints.
  /// So this function returns the single Joint of the given name if it exists.
  std::vector<const Joint*> getJoints(const std::string& name) const override;

  // Documentation inherited
  bool hasJoint(const Joint* joint) const override;

  // Documentation inherited
  std::size_t getIndexOf(
      const Joint* _joint, bool _warning = true) const override;

  // Documentation inherited
  std::size_t getNumDofs() const override;

  /// Returns the number of degrees of freedom of a subtree.
  std::size_t getNumDofs(std::size_t treeIndex) const;

  // Documentation inherited
  DegreeOfFreedom* getDof(std::size_t _idx) override;

  // Documentation inherited
  const DegreeOfFreedom* getDof(std::size_t _idx) const override;

  /// Get degree of freedom (aka generalized coordinate) whose name is _name
  DegreeOfFreedom* getDof(const std::string& _name);

  /// Get degree of freedom (aka generalized coordinate) whose name is _name
  const DegreeOfFreedom* getDof(const std::string& _name) const;

  // Documentation inherited
  const std::vector<DegreeOfFreedom*>& getDofs() override;

  // Documentation inherited
  std::vector<const DegreeOfFreedom*> getDofs() const override;

  // Documentation inherited
  std::size_t getIndexOf(
      const DegreeOfFreedom* _dof, bool _warning = true) const override;

  /// Get the DegreesOfFreedom belonging to a tree in this Skeleton
  const std::vector<DegreeOfFreedom*>& getTreeDofs(std::size_t _treeIdx);

  /// Get the DegreesOfFreedom belonging to a tree in this Skeleton
  const std::vector<const DegreeOfFreedom*>& getTreeDofs(
      std::size_t _treeIdx) const;

  /// This function is only meant for debugging purposes. It will verify that
  /// all objects held in the Skeleton have the correct information about their
  /// indexing.
  bool checkIndexingConsistency() const;

  DART_BAKE_SPECIALIZED_NODE_SKEL_DECLARATIONS(Marker)

  DART_BAKE_SPECIALIZED_NODE_SKEL_DECLARATIONS(ShapeNode)

  DART_BAKE_SPECIALIZED_NODE_SKEL_DECLARATIONS(EndEffector)

  /// This returns a square (N x N) matrix, filled with 1s and 0s. This can be
  /// interpreted as:
  ///
  /// getDofParentMap(i,j) == 1: Dof[i] is a parent of Dof[j]
  /// getDofParentMap(i,j) == 0: Dof[i] is NOT a parent of Dof[j]
  ///
  /// This is computed in bulk, and cached in the skeleton.
  const Eigen::MatrixXi& getDofParentMap();

  /// This returns a square (N x N) matrix, filled with 1s and 0s. This can be
  /// interpreted as:
  ///
  /// getJointParentMap(i,j) == 1: Joint[i] is a parent of Joint[j]
  /// getJointParentMap(i,j) == 0: Joint[i] is NOT a parent of Joint[j]
  ///
  /// This is computed in bulk, and cached in the skeleton.
  const Eigen::MatrixXi& getJointParentMap();

  /// \}

  //----------------------------------------------------------------------------
  // Gradients
  //----------------------------------------------------------------------------

  /// This resets the gradient constraint matrices
  void clearGradientConstraintMatrices();

  /// Get a shared pointer to the saved gradient matrices for the constrained
  /// group this Skeleton was part of in the last LCP solve.
  std::shared_ptr<neural::ConstrainedGroupGradientMatrices>
  getGradientConstraintMatrices();

  void setGradientConstraintMatrices(
      std::shared_ptr<neural::ConstrainedGroupGradientMatrices>
          gradientMatrices);

  /// This gives the vel-X Jacobian (in the absence of constraints) for this
  /// skeleton. This is useful for backprop if this skeleton isn't part of
  /// constrained group.
  Eigen::MatrixXs getUnconstrainedVelJacobianWrt(
      s_t dt, neural::WithRespectTo* wrt);

  /// This gives the unconstrained Jacobian giving the difference in C(pos, vel)
  Eigen::MatrixXs getVelCJacobian();

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  struct DiffC
  {
    struct Data
    {
      math::Jacobian S;

      Eigen::Vector6s V;
      Eigen::Vector6s dV;
      Eigen::Vector6s F;
      Eigen::VectorXs tau;

      void init();
    };

    struct Node
    {
      Data data;
      std::vector<Data> derivs;
    };

    std::vector<Node> nodes;

    std::vector<Node> nodes_numeric;

    void init(size_t numBodies, size_t numDofs);
    void print();
  };

  DiffC mDiffC;
#endif

  /// This gives the unconstrained Jacobian of C(pos, vel) using the derivative
  /// f the inverse dynamics
  Eigen::MatrixXs getJacobianOfC(neural::WithRespectTo* wrt);

  /// This gives the unconstrained Jacobian of M*x using the derivative of the
  /// inverse dynamics
  Eigen::MatrixXs getJacobianOfM(
      const Eigen::VectorXs& x, neural::WithRespectTo* wrt);

  /// This gives the unconstrained Jacobian of M*x using the derivative of the
  /// inverse dynamics
  /// @warning SLOW: Only for testing
  Eigen::MatrixXs getJacobianOfID(
      const Eigen::VectorXs& x, neural::WithRespectTo* wrt);

#ifdef DART_DEBUG_ANALYTICAL_DERIV
  struct DiffMinv
  {
    struct Data
    {
      math::Jacobian S;

      math::Inertia AI;
      Eigen::Vector6s AB;
      Eigen::MatrixXs psi;
      math::Inertia Pi;
      Eigen::VectorXs alpha;
      Eigen::Vector6s beta;

      Eigen::VectorXs ddq;
      Eigen::Vector6s dV;

      void init();
    };

    struct Node
    {
      Data data;
      std::vector<Data> derivs;
    };

    std::vector<Node> nodes;

    std::vector<Node> nodes_numeric;

    void init(size_t numBodies, size_t numDofs);
    void print();
  };

  DiffMinv mDiffMinv;
#endif

  /// This gives the unconstrained Jacobian of M^{-1}f
  Eigen::MatrixXs getJacobianOfMinv(
      const Eigen::VectorXs& f, neural::WithRespectTo* wrt);

  /// This gives the unconstrained Jacobian of M^{-1}f using the derivative of
  /// the inverse dynamics.
  /// @note This function is about 2.33 times faster than
  /// getJacobianOfMinv_Direct() for a 10 degrees-of-freedom serial chain robot.
  Eigen::MatrixXs getJacobianOfMinv_ID(
      const Eigen::VectorXs& f, neural::WithRespectTo* wrt);

  /// This gives the unconstrained Jacobian of M^{-1}f using the derivative of
  /// the forward dynamics (supposedly slower than getJacobianOfMinv_ID).
  Eigen::MatrixXs getJacobianOfMinv_Direct(
      const Eigen::VectorXs& f, neural::WithRespectTo* wrt);

  /// This gives the unconstrained Jacobian of the forward dynamics.
  /// @warning SLOW: Only for testing
  Eigen::MatrixXs getJacobianOfFD(neural::WithRespectTo* wrt);

  /// This gives the jacobian of damping and spring forces
  /// @warning SLOW: Only for testing
  Eigen::MatrixXs getJacobianOfDampSpring(neural::WithRespectTo* wrt);

  /// VERY SLOW: Only for testing. This computes the unconstrained Jacobian
  /// giving the difference in M(pos) for finite changes
  Eigen::MatrixXs finiteDifferenceJacobianOfM(
      const Eigen::VectorXs& f,
      neural::WithRespectTo* wrt,
      bool useRidders = true);

  /// VERY SLOW: Only for testing. This computes the unconstrained Jacobian
  /// giving the difference in C(pos, vel) for finite changes
  Eigen::MatrixXs finiteDifferenceJacobianOfC(
      neural::WithRespectTo* wrt, bool useRidders = true);

  /// VERY SLOW: Only for testing. This computes the unconstrained Jacobian
  /// giving the difference in M*f + C(pos, vel) for finite changes
  Eigen::MatrixXs finiteDifferenceJacobianOfID(
      const Eigen::VectorXs& f,
      neural::WithRespectTo* wrt,
      bool useRidders = true);

  /// VERY SLOW: Only for testing. This computes the unconstrained Jacobian
  /// giving the difference in M^{-1}f for finite changes
  Eigen::MatrixXs finiteDifferenceJacobianOfMinv(
      const Eigen::VectorXs& f,
      neural::WithRespectTo* wrt,
      bool useRidders = true);

  /// VERY SLOW: Only for testing. This computes the unconstrained Jacobian
  /// giving the difference in C(pos, vel) for finite changes in vel
  Eigen::MatrixXs finiteDifferenceVelCJacobian(bool useRidders = true);

  Eigen::MatrixXs finiteDifferenceJacobianOfFD(
      neural::WithRespectTo* wrt, bool useRidders = true);

  Eigen::VectorXs getDynamicsForces();

  //----------------------------------------------------------------------------
  // Differentiable Sensors
  //----------------------------------------------------------------------------

  /// This computes the distance (along the `up` vector) from the highest vertex
  /// to the lowest vertex on the model, when positioned at `pose`
  s_t getHeight(
      Eigen::VectorXs pose, Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the gradient of the height
  Eigen::VectorXs getGradientOfHeightWrtBodyScales(
      Eigen::VectorXs pose, Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the gradient of the height
  Eigen::VectorXs finiteDifferenceGradientOfHeightWrtBodyScales(
      Eigen::VectorXs pose, Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the gradient of the height
  Eigen::VectorXs getGradientOfHeightWrtGroupScales(
      Eigen::VectorXs pose, Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This returns a marker set with at least one marker in it, that each
  /// represents the lowest point on the body, measure by the `up` vector, in
  /// the specified position. If there are no ties, this will be of length 1. If
  /// there are more than one tied lowest point, then this is of length > 1.
  std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>
  getLowestPointMarkers(Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the lowest point on the colliders, as measured by the `up`
  /// vector. This is useful in order to apply constraints that a model can't
  /// penetrate the ground.
  s_t getLowestPoint(Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the gradient of the lowest point wrt body scales
  Eigen::VectorXs getGradientOfLowestPointWrtBodyScales(
      Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the gradient of the lowest point wrt body scales
  Eigen::VectorXs finiteDifferenceGradientOfLowestPointWrtBodyScales(
      Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the gradient of the lowest point wrt body scales
  Eigen::VectorXs getGradientOfLowestPointWrtJoints(
      Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  /// This computes the gradient of the lowest point wrt body scales
  Eigen::VectorXs finiteDifferenceGradientOfLowestPointWrtJoints(
      Eigen::Vector3s up = Eigen::Vector3s::UnitY());

  //----------------------------------------------------------------------------
  // Randomness
  //----------------------------------------------------------------------------

  /// This gets a random pose that's valid within joint limits
  Eigen::VectorXs getRandomPose();

  /// This gets a random velocity that's valid within joint limits
  Eigen::VectorXs getRandomVelocity();

  /// This gets a random pose that's valid within joint limits, but only changes
  /// the specified joints. All unspecified joints are left as 0.
  Eigen::VectorXs getRandomPoseForJoints(std::vector<dynamics::Joint*> joints);

  //----------------------------------------------------------------------------
  // Trajectory optimization
  //----------------------------------------------------------------------------

  // This gives the vector of force upper limits for all the DOFs in this
  // skeleton
  Eigen::VectorXs getControlForceUpperLimits();

  // This gives the vector of force lower limits for all the DOFs in this
  // skeleton
  Eigen::VectorXs getControlForceLowerLimits();

  // This gives the vector of position upper limits for all the DOFs in this
  // skeleton
  Eigen::VectorXs getPositionUpperLimits();

  // This gives the vector of position lower limits for all the DOFs in this
  // skeleton
  Eigen::VectorXs getPositionLowerLimits();

  // This gives the vector of velocity upper limits for all the DOFs in this
  // skeleton
  Eigen::VectorXs getVelocityUpperLimits();

  // This gives the vector of velocity lower limits for all the DOFs in this
  // skeleton
  Eigen::VectorXs getVelocityLowerLimits();

  /// Returns the size of the getLinkCOMs() vector
  std::size_t getLinkCOMDims();

  /// Returns the size of the getLinkMOIs() vector
  std::size_t getLinkMOIDims();

  /// Returns the size of the getMasses() vector
  std::size_t getLinkMassesDims();

  // Get link coms using mu
  Eigen::VectorXs getLinkMUs();

  // Get Beta Related
  Eigen::VectorXs getLinkBetas();

  Eigen::Vector3s getLinkBetaIndex(size_t index);

  s_t getLinkMUIndex(size_t index);

  // This gets all the inertia center-of-mass vectors for all the links in this
  // skeleton concatenated together
  Eigen::VectorXs getLinkCOMs();

  // This gets particular center-of-mass vectors of a body node
  Eigen::Vector3s getLinkCOMIndex(size_t index);

  // This gets all the inertia moment-of-inertia paremeters for all the links in
  // this skeleton concatenated together
  Eigen::VectorXs getLinkMOIs();

  // This get particular moment of inertia of a body node
  Eigen::Vector6s getLinkMOIIndex(size_t index);

  // This returns a vector of all the link masses for all the links in this
  // skeleton concatenated into a flat vector.
  Eigen::VectorXs getLinkMasses();

  // Sets the upper limits of all the joints from a single vector
  void setControlForceUpperLimits(Eigen::VectorXs limits);

  // Sets the lower limits of all the joints from a single vector
  void setControlForceLowerLimits(Eigen::VectorXs limits);

  // Sets the upper limits of all the joints from a single vector
  void setPositionUpperLimits(Eigen::VectorXs limits);

  // Sets the lower limits of all the joints from a single vector
  void setPositionLowerLimits(Eigen::VectorXs limits);

  // Sets the upper limits of all the joints from a single vector
  void setVelocityUpperLimits(Eigen::VectorXs limits);

  // Sets the lower limits of all the joints from a single vector
  void setVelocityLowerLimits(Eigen::VectorXs limits);

  void setLinkMUs(Eigen::VectorXs mus);
  // This will set COM of a particular link according to mu
  void setLinkMUIndex(s_t mu, size_t index);

  void setLinkBetaIndex(Eigen::Vector3s beta, size_t index);

  void setLinkBetas(Eigen::VectorXs betas);

  // This sets all the inertia center-of-mass vectors for all the links in this
  // skeleton concatenated together
  void setLinkCOMs(Eigen::VectorXs coms);

  void setLinkCOMIndex(Eigen::Vector3s com, size_t index);

  // This sets all the inertia moment-of-inertia paremeters for all the links in
  // this skeleton concatenated together
  void setLinkMOIs(Eigen::VectorXs mois);

  void setLinkMOIIndex(Eigen::Vector6s moi, size_t index);

  // This returns a vector of all the link masses for all the links in this
  // skeleton concatenated into a flat vector.
  void setLinkMasses(Eigen::VectorXs masses);

  // This returns a vector of all the link scales for all the links in the
  // skeleton concatenated into a flat vector
  Eigen::VectorXs getBodyScales();

  // Sets all the link scales for the skeleton, from a flat vector
  void setBodyScales(Eigen::VectorXs scales);

  // This sets all the positions of the joints to within their limit range, if
  // they're currently outside it.
  void clampPositionsToLimits();

  /// There is an annoying tendency for custom joints to encode the linear
  /// offset of the bone in their custom functions. We don't want that, so we
  /// want to move any relative transform caused by custom functions into the
  /// parent transform.
  void zeroTranslationInCustomFunctions();

  //----------------------------------------------------------------------------
  // Utilities for featurizing diverse skeletons consistently
  //----------------------------------------------------------------------------

  /// This is a utility that will find a body with this mesh attached, if one
  /// exists, and also the transform which relates the geometry frame to the
  /// body frame.
  ///
  /// WARNING: Do not expose this to Python, because the pybind11 ownership
  /// scheme can't handle mixed ownership, and it'll try to free the BodyNode
  /// pointer when the pair goes out of use in Python, which is bad. Expose the
  /// below methods instead.
  std::pair<dynamics::BodyNode*, Eigen::Isometry3s>* getBodyAndTransformForMesh(
      std::string meshFileName);

  /// This is a utility that will find a body with this mesh attached, if one
  /// exists.
  dynamics::BodyNode* getBodyForMesh(std::string meshFileName);

  /// This is a utility that will transform an isometry in the frame of a mesh
  /// to a transform in the parent body's frame.
  Eigen::Isometry3s getTransformFromMeshToParentBody(
      std::string meshFileName, Eigen::Isometry3s relativeToGeometry);

  /// This is a utility that will transform a translation in the frame of a mesh
  /// to a translation in the parent body's frame.
  Eigen::Vector3s getTranslationFromMeshToParentBody(
      std::string meshFileName, Eigen::Vector3s relativeToGeometry);

  /// This is a utility that will transform a rotation in the frame of a mesh
  /// to a rotation in the parent body's frame.
  Eigen::Matrix3s getRotationFromMeshToParentBody(
      std::string meshFileName, Eigen::Matrix3s relativeToGeometry);

  //----------------------------------------------------------------------------
  // Constraining links to have the same scale
  //----------------------------------------------------------------------------

  const std::vector<BodyScaleGroup>& getBodyScaleGroups() const;

  const BodyScaleGroup& getBodyScaleGroup(int index) const;

  /// This creates scale groups for any body nodes that may've been added since
  /// we last interacted with the body scale group APIs
  void ensureBodyScaleGroups();

  /// This returns the index of the group that this body node corresponds to
  int getScaleGroupIndex(dynamics::BodyNode* bodyNode);

  /// This returns the axis flips of this body in the scale group that this body
  /// node corresponds to
  Eigen::Vector3s getScaleGroupFlips(dynamics::BodyNode* bodyNode);

  /// This takes two scale groups and merges their contents into a single group.
  /// After this operation, there is one fewer scale group.
  void mergeScaleGroups(dynamics::BodyNode* a, dynamics::BodyNode* b);

  /// The scale group axis flips
  void autodetectScaleGroupAxisFlips(int symmetryAxis);

  /// This finds all the pairs of bodies that share the same prefix, and
  /// different suffixes (for example "a_body_l" and "a_body_r", sharing "_l"
  /// and "_r")
  void autogroupSymmetricSuffixes(
      std::string leftSuffix = "_l", std::string rightSuffix = "_r");

  /// This finds all the pairs of bodies that share the same suffix, and
  /// different prefixes (for example "ulna_l" and "radius_l", sharing "_l")
  void autogroupSymmetricPrefixes(
      std::string firstPrefix = "ulna", std::string secondPrefix = "radius");

  /// This means that we'll scale a group along all three axis equally. This
  /// constrains scaling.
  void setScaleGroupUniformScaling(dynamics::BodyNode* a, bool uniform = true);

  /// This returns the number of scale groups
  int getNumScaleGroups();

  /// This returns the dimension of the scale group
  int getScaleGroupDim(int groupIndex);

  /// This gets the scale upper bound for the first body in a group, by index
  Eigen::VectorXs getScaleGroupUpperBound(int groupIndex);

  /// This gets the scale lower bound for the first body in a group, by index
  Eigen::VectorXs getScaleGroupLowerBound(int groupIndex);

  /// This takes two scale groups and merges their contents into a single group.
  /// After this operation, there is one fewer scale group.
  void mergeScaleGroupsByIndex(int a, int b);

  /// This returns the dimensions of the grouped scale vector.
  int getGroupScaleDim();

  /// This precomputes the array of group scale indices that we need for
  /// getGroupScaleIndexDetails()
  void updateGroupScaleIndices();

  /// This grabs the details for what a group scale index corresponds to
  const BodyScaleGroupAndIndex& getGroupScaleIndexDetails(int index) const;

  /// This produces a human-readable description of the group scale vector index
  std::string debugGroupScaleIndex(int groupIdx);

  /// This returns the vector relating changing a group scale parameter to the
  /// location of the center of a body.
  Eigen::Vector3s getGroupScaleMovementOnBodyInWorldSpace(
      int groupIdx, int bodyIdx);

  /// This returns the vector relating changing a group scale parameter to the
  /// location of the center of a body.
  Eigen::Vector3s finiteDifferenceGroupScaleMovementOnBodyInWorldSpace(
      int groupIdx, int bodyIdx);

  /// This returns the vector relating changing a group scale parameter to the
  /// location of the center of a joint.
  Eigen::Vector3s getGroupScaleMovementOnJointInWorldSpace(
      int groupIdx, int bodyIdx);

  /// This returns the vector relating changing a group scale parameter to the
  /// location of the center of a joint.
  Eigen::Vector3s finiteDifferenceGroupScaleMovementOnJointInWorldSpace(
      int groupIdx, int bodyIdx);

  /// This sets the scales of all the body nodes according to their group
  /// membership. The `scale` vector is expected to be the same size as the
  /// number of groups.
  void setGroupScales(Eigen::VectorXs scale, bool silentlyClamp = false);

  /// This gets the scales of the first body in each scale group.
  Eigen::VectorXs getGroupScales();

  /// This converts a map of body scales back into group scales, interpreting
  /// everything as gradients.
  Eigen::VectorXs getGroupScaleGradientsFromMap(
      std::map<std::string, Eigen::Vector3s> bodyScales);

  /// This returns the upper bound values for each index in the group scales
  /// vector
  Eigen::VectorXs getGroupScalesUpperBound();

  /// This returns the upper bound values for each index in the group scales
  /// vector
  Eigen::VectorXs getGroupScalesLowerBound();

  /// This gets the masses of each scale group, concatenated
  Eigen::VectorXs getGroupMasses();

  /// This sets the masses of each scale group, concatenated
  void setGroupMasses(Eigen::VectorXs masses);

  /// This gets the upper bound for each group's mass, concatenated
  Eigen::VectorXs getGroupMassesUpperBound();

  /// This gets the lower bound for each group's mass, concatenated
  Eigen::VectorXs getGroupMassesLowerBound();

  /// This gets a vector of [(1/m), p_0, ..., p_n], where the first entry is the
  /// inverse of the total mass of the skeleton, and all subsequent entries are
  /// the percentages of the total for each link mass.
  Eigen::VectorXs getLinearizedMasses();

  /// This sets a vector of [(1/m), p_0, ..., p_n], where the first entry is the
  /// inverse of the total mass of the skeleton, and all subsequent entries are
  /// the percentages of the total for each link mass, and maps back into the
  /// mass list.
  void setLinearizedMasses(Eigen::VectorXs masses);

  /// This gets the upper bound for the linearized mass, concatenated
  Eigen::VectorXs getLinearizedMassesUpperBound();

  /// This gets the lower bound for the linearized mass, concatenated
  Eigen::VectorXs getLinearizedMassesLowerBound();

  /// This maps a linearized masses vector to an "unnormalized COM." This means
  /// that the percentages in the linearized mass vector need not add to 1.
  /// Relaxing this constraint, and then applying it later in the optimizer, can
  /// make certain dynamics problems into linear.
  Eigen::Vector3s getUnnormalizedCOM(Eigen::VectorXs linearizedMasses);

  /// This maps a linearized masses vector to an "unnormalized COM
  /// acceleration." This means that the percentages in the linearized mass
  /// vector need not add to 1. Relaxing this constraint, and then applying it
  /// later in the optimizer, can make certain dynamics problems into linear.
  Eigen::Vector3s getUnnormalizedCOMAcceleration(
      Eigen::VectorXs linearizedMasses);

  /// This maps a linearized masses vector to an "unnormalized COM
  /// acceleration." This means that the percentages in the linearized mass
  /// vector need not add to 1. Relaxing this constraint, and then applying it
  /// later in the optimizer, can make certain dynamics problems into linear.
  Eigen::Vector3s getUnnormalizedCOMFDAcceleration(
      Eigen::VectorXs linearizedMasses);

  /// This maps to the difference for unnormalized (analytical COM acc - fd COM
  /// acc)
  Eigen::Vector3s getUnnormalizedCOMAccelerationOffset(
      Eigen::VectorXs linearizedMasses);

  /// This gets the analytical jacobian relating the linearized masses to link
  /// masses
  Eigen::MatrixXs getGroupMassesJacobianWrtLinearizedMasses();

  /// This gets the finite difference'd jacobian relating the linearized masses
  /// to link masses
  Eigen::MatrixXs finiteDifferenceGroupMassesJacobianWrtLinearizedMasses();

  /// This gets the analytical jacobian relating changes in the linearized
  /// masses to changes in the COM position, neglecting normalization to
  /// guarantee that total percentages sum to 1.0
  Eigen::MatrixXs getUnnormalizedCOMJacobianWrtLinearizedMasses();

  /// This gets the analytical jacobian relating changes in the linearized
  /// masses to changes in the COM position, neglecting normalization to
  /// guarantee that total percentages sum to 1.0
  Eigen::MatrixXs finiteDifferenceUnnormalizedCOMJacobianWrtLinearizedMasses();

  /// This relates changes to the linearized masses to changes to the finite
  /// difference'd formula for COM acceleration, (c[t-1] - 2*c[t] +
  /// c[t+1])/(dt*dt)
  Eigen::MatrixXs getUnnormalizedCOMFDAccJacobianWrtLinearizedMasses();

  /// This relates changes to the linearized masses to changes to the finite
  /// difference'd formula for COM acceleration, (c[t-1] - 2*c[t] +
  /// c[t+1])/(dt*dt)
  Eigen::MatrixXs
  finiteDifferenceUnnormalizedCOMFDAccJacobianWrtLinearizedMasses();

  /// This relates changes to the linearized masses to changes to the analytical
  /// formula for COM acceleration
  Eigen::MatrixXs getUnnormalizedCOMAnalyticalAccJacobianWrtLinearizedMasses();

  /// This relates changes to the linearized masses to changes to the analytical
  /// formula for COM acceleration
  Eigen::MatrixXs
  finiteDifferenceUnnormalizedCOMAnalyticalAccJacobianWrtLinearizedMasses();

  /// This relates changes to the linearized masses to changes in the
  /// unnormalized (analytical COM acc - fd COM acc) quantity.
  Eigen::MatrixXs
  getUnnormalizedCOMAccelerationOffsetJacobianWrtLinearizedMasses();

  /// This relates changes to the linearized masses to changes in the
  /// unnormalized (analytical COM acc - fd COM acc) quantity.
  Eigen::MatrixXs
  finiteDifferenceUnnormalizedCOMAccelerationOffsetJacobianWrtLinearizedMasses();

  /// This gets the COMs of each scale group, concatenated
  Eigen::VectorXs getGroupCOMs();

  /// This gets the upper bound for each axis of each group's COM, concatenated
  Eigen::VectorXs getGroupCOMUpperBound();

  /// This gets the lower bound for each axis of each group's COM, concatenated
  Eigen::VectorXs getGroupCOMLowerBound();

  /// This sets the COMs of each scale group, concatenated
  void setGroupCOMs(Eigen::VectorXs coms);

  /// This gets the Inertias of each scale group (the 6 vector), concatenated
  Eigen::VectorXs getGroupInertias();

  /// This sets the Inertias of each scale group (the 6 vector), concatenated
  void setGroupInertias(Eigen::VectorXs inertias);

  /// This gets the upper bound for each axis of each group's inertias,
  /// concatenated
  Eigen::VectorXs getGroupInertiasUpperBound();

  /// This gets the lower bound for each axis of each group's inertias,
  /// concatenated
  Eigen::VectorXs getGroupInertiasLowerBound();

  /// This is a general purpose utility to convert a Gradient wrt Body scales to
  /// one wrt Group scales
  Eigen::VectorXs convertBodyScalesGradientToGroupScales(
      Eigen::VectorXs bodyScalesGrad);

  /// This is a general purpose utility to convert a Jacobian wrt Body scales to
  /// one wrt Group scales
  Eigen::MatrixXs convertBodyScalesJacobianToGroupScales(
      Eigen::MatrixXs bodyScalesJac);

  /// This returns the Jacobian of the joint positions wrt the scales of the
  /// groups
  Eigen::MatrixXs getJointWorldPositionsJacobianWrtGroupScales(
      const std::vector<dynamics::Joint*>& joints);

  /// This returns the Jacobian of the joint positions wrt the scales of the
  /// groups
  Eigen::MatrixXs finiteDifferenceJointWorldPositionsJacobianWrtGroupScales(
      const std::vector<dynamics::Joint*>& joints);

  /// This returns the Jacobian relating changes in body scales to changes in
  /// marker world positions.
  Eigen::MatrixXs getMarkerWorldPositionsJacobianWrtGroupScales(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This returns the Jacobian relating changes in body scales to changes in
  /// marker world positions.
  Eigen::MatrixXs finiteDifferenceMarkerWorldPositionsJacobianWrtGroupScales(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to group
  /// scales
  Eigen::MatrixXs getMarkerWorldPositionsSecondJacobianWrtJointWrtGroupScales(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to group
  /// scales
  Eigen::MatrixXs
  finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtGroupScales(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This returns the gradient of the distance measurement, with respect to
  /// group scales
  Eigen::VectorXs getGradientOfDistanceWrtGroupScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB);

  /// This returns the gradient of the distance measurement, with respect to
  /// group scales
  Eigen::VectorXs finiteDifferenceGradientOfDistanceWrtGroupScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB);

  /// This returns the gradient of the distance measurement, with respect to
  /// group scales
  Eigen::VectorXs getGradientOfDistanceAlongAxisWrtGroupScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
      Eigen::Vector3s axis);

  /// This returns the gradient of the distance measurement, with respect to
  /// group scales
  Eigen::VectorXs finiteDifferenceGradientOfDistanceAlongAxisWrtGroupScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
      Eigen::Vector3s axis);

  //----------------------------------------------------------------------------
  // Converting EulerJoints->BallJoints and EulerFreeJoints->FreeJoints
  //
  // This allows us to do computations like IK in a gimbal-lock-free space.
  //----------------------------------------------------------------------------

  // This creates a fresh skeleton, which is a copy of this one EXCEPT that
  // EulerJoints are BallJoints, and EulerFreeJoints are FreeJoints. This means
  // the configuration spaces are different, so you need to use
  // `convertPositionsToBallSpace()` and `convertPositionsFromBallSpace()` to
  // transform positions to and from the new skeleton's configuration.
  std::shared_ptr<dynamics::Skeleton> convertSkeletonToBallJoints();

  // This converts the position vector from Euler space to Ball space for any
  // joints that need to be converted. This needs to be called on a skeleton
  // with EulerJoints and/or EulerFreeJoints or it will just return the passed
  // in vector unchanged.
  Eigen::VectorXs convertPositionsToBallSpace(Eigen::VectorXs pos);

  // This converts the position vector from Ball space to Euler space for any
  // joints that need to be converted. This needs to be called on a skeleton
  // with EulerJoints and/or EulerFreeJoints or it will just return the passed
  // in vector unchanged.
  Eigen::VectorXs convertPositionsFromBallSpace(Eigen::VectorXs pos);

  //----------------------------------------------------------------------------
  // IK for retargetting (especially between similar but not identical human
  // skeletons)
  //----------------------------------------------------------------------------

  /// This returns the concatenated 3-vectors for world positions of each joint
  /// in 3D world space, for the registered joints.
  Eigen::VectorXs getJointWorldPositions(
      const std::vector<dynamics::Joint*>& joints) const;

  /// This returns the world position of a joint
  Eigen::Vector3s getJointWorldPosition(int idx) const;

  /// This returns a map with the world positions of each joint, keyed by joint
  /// name
  std::map<std::string, Eigen::Vector3s> getJointWorldPositionsMap() const;

  /// This returns the concatenated 3-vectors for world angle of each joint's
  /// child space in 3D world space, for the registered joints.
  Eigen::VectorXs getJointWorldAngles(
      const std::vector<dynamics::Joint*>& joints) const;

  /// This returns the Jacobian relating changes in source skeleton joint
  /// positions to changes in source joint world positions.
  Eigen::MatrixXs getJointWorldPositionsJacobianWrtJointPositions(
      const std::vector<dynamics::Joint*>& joints) const;

  /// This returns the Jacobian relating changes in source skeleton joint
  /// positions to changes in source joint world positions.
  Eigen::MatrixXs finiteDifferenceJointWorldPositionsJacobianWrtJointPositions(
      const std::vector<dynamics::Joint*>& joints);

  /// This returns the Jacobian relating changes in source skeleton joint
  /// positions to changes in source joint world positions.
  Eigen::MatrixXs getJointWorldPositionsJacobianWrtJointChildAngles(
      const std::vector<dynamics::Joint*>& joints) const;

  /// This returns the Jacobian relating changes in source skeleton joint
  /// positions to changes in source joint world positions.
  Eigen::MatrixXs
  finiteDifferenceJointWorldPositionsJacobianWrtJointChildAngles(
      const std::vector<dynamics::Joint*>& joints);

  /// This returns the Jacobian relating changes in source skeleton body scales
  /// to changes in source joint world positions.
  Eigen::MatrixXs getJointWorldPositionsJacobianWrtBodyScales(
      const std::vector<dynamics::Joint*>& joints);

  /// This returns the Jacobian relating changes in source skeleton body scales
  /// to changes in source joint world positions.
  Eigen::MatrixXs finiteDifferenceJointWorldPositionsJacobianWrtBodyScales(
      const std::vector<dynamics::Joint*>& joints);

  /// This returns a score for a force field, which runs a simple non-linear
  /// function over the joint distance
  s_t getJointForceFieldToOtherJoints(
      const std::vector<dynamics::Joint*>& joints,
      int jointIndex,
      s_t barrierDistance,
      s_t softness);

  /// This returns a score for a force field, which runs a simple non-linear
  /// function over the joint distance
  Eigen::VectorXs getJointForceFieldToOtherJointsGradient(
      const std::vector<dynamics::Joint*>& joints,
      int jointIndex,
      s_t barrierDistance,
      s_t softness,
      neural::WithRespectTo* wrt);

  /// This returns a score for a force field, which runs a simple non-linear
  /// function over the joint distance
  Eigen::VectorXs finiteDifferenceJointForceFieldToOtherJointsGradient(
      const std::vector<dynamics::Joint*>& joints,
      int jointIndex,
      s_t barrierDistance,
      s_t softness,
      neural::WithRespectTo* wrt);

  /// This returns a score for a force field, which runs a simple non-linear
  /// function over the joint distance with respect to body scales, which for
  /// historical reasons is not supported in the neural::WithRespectTo API.
  /// TODO: correct that
  Eigen::VectorXs getJointForceFieldToOtherJointsGradientWrtBodyScales(
      const std::vector<dynamics::Joint*>& joints,
      int jointIndex,
      s_t barrierDistance,
      s_t softness);

  /// This returns a score for a force field, which runs a simple non-linear
  /// function over the joint distance with respect to body scales, which for
  /// historical reasons is not supported in the neural::WithRespectTo API.
  /// TODO: correct that
  Eigen::VectorXs
  finiteDifferenceJointForceFieldToOtherJointsGradientWrtBodyScales(
      const std::vector<dynamics::Joint*>& joints,
      int jointIndex,
      s_t barrierDistance,
      s_t softness);

  /// This returns a vector of the distance to all joints after the joint
  Eigen::VectorXs getJointDistanceToOtherJoints(
      const std::vector<dynamics::Joint*>& joints, int jointIndex);

  /// This gets the Jacobian relating changes in the joint world positions
  /// (IMPORTANT: NOT JOINT ANGLE - physical position of joint centers in the
  /// world) to the changes in the joint distance measurement vector for joint
  /// `jointIndex`.
  Eigen::MatrixXs getJointDistanceToOtherJointsJacobianWrtJointWorldPositions(
      const std::vector<dynamics::Joint*>& joints, int jointIndex);

  /// This gets the Jacobian relating changes in the joint world positions
  /// (IMPORTANT: NOT JOINT ANGLE - physical position of joint centers in the
  /// world) to the changes in the joint distance measurement vector for joint
  /// `jointIndex`.
  Eigen::MatrixXs
  finiteDifferenceJointDistanceToOtherJointsJacobianWrtJointWorldPositions(
      const std::vector<dynamics::Joint*>& joints, int jointIndex);

  /// This returns a Jacobian of the distance to every joint in the body with
  /// respect to WRT
  Eigen::MatrixXs getJointDistanceToOtherJointsJacobianWrt(
      const std::vector<dynamics::Joint*>& joints,
      int jointIndex,
      neural::WithRespectTo* wrt);

  /// This returns a Jacobian of the distance to every joint in the body with
  /// respect to WRT
  Eigen::MatrixXs finiteDifferenceJointDistanceToOtherJointsJacobianWrt(
      const std::vector<dynamics::Joint*>& joints,
      int jointIndex,
      neural::WithRespectTo* wrt);

  /// This returns a Jacobian of the distance to every joint in the body with
  /// respect to body scales, which for historical reasons is not supported in
  /// the neural::WithRespectTo API. TODO: correct that
  Eigen::MatrixXs getJointDistanceToOtherJointsJacobianWrtBodyScales(
      const std::vector<dynamics::Joint*>& joints, int jointIndex);

  /// This returns a Jacobian of the distance to every joint in the body with
  /// respect to body scales, which for historical reasons is not supported in
  /// the neural::WithRespectTo API. TODO: correct that
  Eigen::MatrixXs
  finiteDifferenceJointDistanceToOtherJointsJacobianWrtBodyScales(
      const std::vector<dynamics::Joint*>& joints, int jointIndex);

  /// These are a set of bodies, and offsets in local body space where markers
  /// are mounted on the body
  std::map<std::string, Eigen::Vector3s> getMarkerMapWorldPositions(
      const MarkerMap& markers);

  /// This converts markers from a source skeleton to the current, doing a
  /// simple mapping based on body node names. Any markers that don't find a
  /// body node in the current skeleton with the same name are dropped.
  MarkerMap convertMarkerMap(
      const MarkerMap& markerMap, bool warnOnDrop = true);

  /// These are a set of bodies, and offsets in local body space where markers
  /// are mounted on the body
  Eigen::VectorXs getMarkerWorldPositions(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This returns the Jacobian relating changes in joint
  /// positions to changes in marker world positions.
  Eigen::MatrixXs getMarkerWorldPositionsJacobianWrtJointPositions(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers) const;

  /// This returns the Jacobian relating changes in joint
  /// positions to changes in marker world positions.
  Eigen::MatrixXs finiteDifferenceMarkerWorldPositionsJacobianWrtJointPositions(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This returns the Jacobian relating changes in body scales to changes in
  /// marker world positions.
  Eigen::MatrixXs getMarkerWorldPositionsJacobianWrtBodyScales(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This returns the Jacobian relating changes in body scales to changes in
  /// marker world positions.
  Eigen::MatrixXs finiteDifferenceMarkerWorldPositionsJacobianWrtBodyScales(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This returns the Jacobian relating changes in marker offsets to changes in
  /// marker world positions.
  Eigen::MatrixXs getMarkerWorldPositionsJacobianWrtMarkerOffsets(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers) const;

  /// This returns the Jacobian relating changes in marker offsets to changes in
  /// marker world positions.
  Eigen::MatrixXs finiteDifferenceMarkerWorldPositionsJacobianWrtMarkerOffsets(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This gets the gradient of the ||f(q) - x|| function with respect to q
  Eigen::VectorXs getMarkerWorldPositionDiffToGoalGradientWrtJointPos(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs goal);

  /// This gets the gradient of the ||f(q) - x|| function with respect to q
  Eigen::VectorXs
  finiteDifferenceMarkerWorldPositionDiffToGoalGradientWrtJointPos(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs goal);

  /// This should be equivalent to
  /// `getMarkerWorldPositionsJacobianWrtJointPositions`, just slower. This is
  /// here so there's a simple non-recursive formula for the Jacobian to take
  /// derivatives against.
  Eigen::MatrixXs getScrewsMarkerWorldPositionsJacobianWrtJointPositions(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  /// This gets the derivative of the Jacobian of the markers wrt joint
  /// positions, with respect to a single joint index
  Eigen::MatrixXs getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtJoints(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int index);

  /// This gets the derivative of the Jacobian of the markers wrt joint
  /// positions, with respect to a single joint index
  Eigen::MatrixXs
  finiteDifferenceMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtJoints(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int index);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to joint
  /// positions
  Eigen::MatrixXs
  getMarkerWorldPositionsSecondJacobianWrtJointWrtJointPositions(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to joint
  /// positions
  Eigen::MatrixXs
  finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtJointPositions(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This gets the derivative of the Jacobian of the markers wrt joint
  /// positions, with respect to a single body scaling
  Eigen::MatrixXs
  getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtBodyScale(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int index,
      int axis,
      const Eigen::MatrixXs& markerWrtScaleJac);

  Eigen::MatrixXs scratch(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers);

  Eigen::MatrixXs scratchAnalytical(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int index,
      int axis);

  Eigen::MatrixXs scratchFd(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int index,
      int axis);

  /// This gets the derivative of the Jacobian of the markers wrt joint
  /// positions, with respect to a single body scaling
  Eigen::MatrixXs
  finiteDifferenceMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtBodyScale(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int index,
      int axis);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to body
  /// scales
  Eigen::MatrixXs getMarkerWorldPositionsSecondJacobianWrtJointWrtBodyScale(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to body
  /// scales
  Eigen::MatrixXs
  finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtBodyScale(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This gets the derivative of the Jacobian of the markers wrt joint
  /// positions, with respect to a single marker offset
  Eigen::MatrixXs
  getMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtMarkerOffsets(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int marker,
      int axis,
      const Eigen::MatrixXs& markerWrtMarkerJac);

  /// This gets the derivative of the Jacobian of the markers wrt joint
  /// positions, with respect to a single marker offset
  Eigen::MatrixXs
  finiteDifferenceMarkerWorldPositionsDerivativeOfJacobianWrtJointsWrtMarkerOffsets(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      int marker,
      int axis);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to
  /// marker offsets
  Eigen::MatrixXs getMarkerWorldPositionsSecondJacobianWrtJointWrtMarkerOffsets(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This gets the Jacobian of leftMultiply.transpose()*J with respect to
  /// marker offsets
  Eigen::MatrixXs
  finiteDifferenceMarkerWorldPositionsSecondJacobianWrtJointWrtMarkerOffsets(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs leftMultiply);

  /// This runs IK, attempting to fit the world positions of the passed in
  /// joints to the vector of (concatenated) target positions. This can
  /// optionally also rescale the skeleton.
  s_t fitJointsToWorldPositions(
      const std::vector<dynamics::Joint*>& positionJoints,
      Eigen::VectorXs targetPositions,
      bool scaleBodies = false,
      math::IKConfig config = math::IKConfig());

  /// This runs IK, attempting to fit the world positions of the passed in
  /// markers to the vector of (concatenated) target positions.
  s_t fitMarkersToWorldPositions(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Vector3s>>&
          markers,
      Eigen::VectorXs targetPositions,
      Eigen::VectorXs markerWeights,
      bool scaleBodies = false,
      math::IKConfig config = math::IKConfig());

  /// These are a set of bodies, and offsets in local body space where gyros
  /// are mounted on the body
  std::map<std::string, Eigen::Vector3s> getGyroMapReadings(
      const SensorMap& gyros);

  /// These are a set of bodies, and offsets in local body space where
  /// accelerometers are mounted on the body
  std::map<std::string, Eigen::Vector3s> getAccMapReadings(
      const SensorMap& accs);

  /// This returns the world positions and orientations of the sensors
  std::map<std::string, Eigen::Isometry3s> getSensorWorldPositions(
      const SensorMap& sensors);

  /// This converts markers from a source skeleton to the current, doing a
  /// simple mapping based on body node names. Any markers that don't find a
  /// body node in the current skeleton with the same name are dropped.
  SensorMap convertSensorMap(
      const SensorMap& sensorMap, bool warnOnDrop = true);

  /// These are a set of bodies, and offsets in local body space where gyros
  /// are mounted on the body
  Eigen::VectorXs getGyroReadings(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>&
          gyros);

  /// This returns the Jacobian relating changes in joint
  /// positions to changes in gyro readings
  Eigen::MatrixXs getGyroReadingsJacobianWrt(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>&
          gyros,
      neural::WithRespectTo* wrt);

  /// This returns the Jacobian relating changes in joint
  /// positions to changes in gyro readings
  Eigen::MatrixXs finiteDifferenceGyroReadingsJacobianWrt(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>&
          gyros,
      neural::WithRespectTo* wrt);

  /// These are a set of bodies, and offsets in local body space where accs
  /// are mounted on the body
  Eigen::VectorXs getAccelerometerReadings(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>&
          accs);

  /// This returns the Jacobian relating changes in joint
  /// positions to changes in acc readings
  Eigen::MatrixXs getAccelerometerReadingsJacobianWrt(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>&
          accs,
      neural::WithRespectTo* wrt);

  /// This returns the Jacobian relating changes in joint
  /// positions to changes in acc readings
  Eigen::MatrixXs finiteDifferenceAccelerometerReadingsJacobianWrt(
      const std::vector<std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>&
          accs,
      neural::WithRespectTo* wrt);

  //----------------------------------------------------------------------------
  // Handling anthropometric measurements
  //----------------------------------------------------------------------------

  /// This measures the distance between two markers in world space, at the
  /// current configuration and scales.
  s_t getDistanceInWorldSpace(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB);

  /// This returns the gradient of the distance measurement, with respect to
  /// body scales
  Eigen::VectorXs getGradientOfDistanceWrtBodyScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB);

  /// This returns the gradient of the distance measurement, with respect to
  /// body scales
  Eigen::VectorXs finiteDifferenceGradientOfDistanceWrtBodyScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB);

  /// This measures the distance between two markers in world space **along a
  /// specific axis**, at the current configuration and scales. For example, if
  /// the axis is the Y axis, we're just measuring the Y distance between
  /// markers.
  s_t getDistanceAlongAxis(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
      Eigen::Vector3s axis);

  /// This returns the gradient of the distance measurement, with respect to
  /// body scales
  Eigen::VectorXs getGradientOfDistanceAlongAxisWrtBodyScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
      Eigen::Vector3s axis);

  /// This returns the gradient of the distance measurement, with respect to
  /// body scales
  Eigen::VectorXs finiteDifferenceGradientOfDistanceAlongAxisWrtBodyScales(
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerA,
      std::pair<dynamics::BodyNode*, Eigen::Vector3s> markerB,
      Eigen::Vector3s axis);

  //----------------------------------------------------------------------------
  // Handling translation of velocities and accelerations into body space
  //----------------------------------------------------------------------------

  /// This returns the spatial velocities (6 vecs) of the bodies each in local
  /// body space, concatenated
  Eigen::VectorXs getBodyLocalVelocities();

  /// This returns the spatial accelerations (6 vecs) of the bodies in each in
  /// local body space, concatenated
  Eigen::VectorXs getBodyLocalAccelerations();

  /// This returns the linear accelerations due to gravity (3 vecs) of the
  /// bodies in each in local body space, concatenated
  Eigen::VectorXs getBodyLocalGravityVectors();

  /// This computes the jacobian of the local velocities for each body with
  /// respect to `wrt`
  Eigen::MatrixXs getBodyLocalVelocitiesJacobian(neural::WithRespectTo* wrt);

  /// This brute forces our local velocities jacobian
  Eigen::MatrixXs finiteDifferenceBodyLocalVelocitiesJacobian(
      neural::WithRespectTo* wrt);

  /// This computes the jacobian of the local accelerations for each body with
  /// respect to `wrt`
  Eigen::MatrixXs getBodyLocalAccelerationsJacobian(neural::WithRespectTo* wrt);

  /// This brute forces our local accelerations jacobian
  Eigen::MatrixXs finiteDifferenceBodyLocalAccelerationsJacobian(
      neural::WithRespectTo* wrt);

  /// This computes the jacobian of the local gravity vectors for each body with
  /// respect to `wrt`
  Eigen::MatrixXs getBodyLocalGravityVectorsJacobian(
      neural::WithRespectTo* wrt);

  /// This brute forces our local gravity vectors jacobian
  Eigen::MatrixXs finiteDifferenceBodyLocalGravityVectorsJacobian(
      neural::WithRespectTo* wrt);

  /// This returns the spatial velocities (6 vecs) of the bodies in world space,
  /// concatenated
  Eigen::VectorXs getBodyWorldVelocities();

  /// This returns the spatial accelerations (6 vecs) of the bodies in world
  /// space, concatenated
  Eigen::VectorXs getBodyWorldAccelerations();

  /// This computes the jacobian of the world velocities for each body with
  /// respect to `wrt`
  Eigen::MatrixXs getBodyWorldVelocitiesJacobian(neural::WithRespectTo* wrt);

  /// This brute forces our world velocities jacobian
  Eigen::MatrixXs finiteDifferenceBodyWorldVelocitiesJacobian(
      neural::WithRespectTo* wrt);

  /// This computes the jacobian of the world accelerations for each body with
  /// respect to `wrt`
  Eigen::MatrixXs getBodyWorldAccelerationsJacobian(neural::WithRespectTo* wrt);

  /// This brute forces our world accelerations jacobian
  Eigen::MatrixXs finiteDifferenceBodyWorldAccelerationsJacobian(
      neural::WithRespectTo* wrt);

  /// This returns the spatial velocities (6 vecs) of the COMs of each body in
  /// world space, concatenated
  Eigen::VectorXs getCOMWorldVelocities();

  /// This returns the spatial accelerations (6 vecs) of the COMs of each body
  /// in world space, concatenated
  Eigen::VectorXs getCOMWorldAccelerations();

  /// This returns the linear accelerations (3 vecs) of the COMs of each body in
  /// world space.
  Eigen::VectorXs getCOMWorldLinearAccelerations();

  /// This computes the jacobian of the world velocities for each body with
  /// respect to `wrt`
  Eigen::MatrixXs getCOMWorldVelocitiesJacobian(neural::WithRespectTo* wrt);

  /// This brute forces our world velocities jacobian
  Eigen::MatrixXs finiteDifferenceCOMWorldVelocitiesJacobian(
      neural::WithRespectTo* wrt);

  /// This computes the jacobian of the world accelerations for each body with
  /// respect to `wrt`
  Eigen::MatrixXs getCOMWorldAccelerationsJacobian(neural::WithRespectTo* wrt);

  /// This brute forces our world accelerations jacobian
  Eigen::MatrixXs finiteDifferenceCOMWorldAccelerationsJacobian(
      neural::WithRespectTo* wrt);

  Eigen::MatrixXs getCOMWorldLinearAccelerationsJacobian(
      neural::WithRespectTo* wrt);

  /// This brute forces our world linear accelerations jacobian
  Eigen::MatrixXs finiteDifferenceCOMWorldLinearAccelerationsJacobian(
      neural::WithRespectTo* wrt);

  //----------------------------------------------------------------------------
  // Integration and finite difference
  //----------------------------------------------------------------------------

  // Documentation inherited
  void integratePositions(s_t _dt);

  // This will do whatever math is necessary to move pos by vel*dt. This isn't
  // always a straight linear addition, in we're using spatial coordinates for
  // some of the joints.
  Eigen::VectorXs integratePositionsExplicit(
      Eigen::VectorXs pos, Eigen::VectorXs vel, s_t dt);

  // This is d/dpos integratePositionsExplicit()
  Eigen::MatrixXs getPosPosJac(
      Eigen::VectorXs pos, Eigen::VectorXs vel, s_t dt);

  // This is d/dvel integratePositionsExplicit()
  Eigen::MatrixXs getVelPosJac(
      Eigen::VectorXs pos, Eigen::VectorXs vel, s_t dt);

  // Documentation inherited
  void integrateVelocities(s_t _dt);

  /// Return the difference of two generalized positions which are measured in
  /// the configuration space of this Skeleton. If the configuration space is
  /// Euclidean space, this function returns _q2 - _q1. Otherwise, it depends on
  /// the type of the configuration space.
  Eigen::VectorXs getPositionDifferences(
      const Eigen::VectorXs& _q2, const Eigen::VectorXs& _q1) const;

  /// Return the difference of two generalized velocities or accelerations which
  /// are measured in the tangent space at the identity. Since the tangent
  /// spaces are vector spaces, this function always returns _dq2 - _dq1.
  Eigen::VectorXs getVelocityDifferences(
      const Eigen::VectorXs& _dq2, const Eigen::VectorXs& _dq1) const;

  //----------------------------------------------------------------------------
  // Inverse Dynamics for Contacts
  //----------------------------------------------------------------------------

  struct ContactInverseDynamicsResult
  {
    dynamics::Skeleton* skel;
    const dynamics::BodyNode* contactBody;
    Eigen::Vector6s contactWrench;
    Eigen::VectorXs jointTorques;

    // These are the setup of the inverse dynamics problem
    Eigen::VectorXs pos;
    Eigen::VectorXs vel;
    Eigen::VectorXs acc;

    /// This computes how much the actual dynamics we get when we apply this
    /// solution differ from the goal solution.
    s_t sumError();
  };

  Eigen::VectorXs getInverseDynamics(const Eigen::VectorXs& accelerations);

  /// This solves the inverse dynamics problem to figure out what forces we
  /// would need to apply (in our _current state_) in order to get the desired
  /// next velocity. This includes arbitrary forces and moments at the
  /// `contactBody`, which can be post-processed down to individual contact
  /// results.
  ContactInverseDynamicsResult getContactInverseDynamics(
      const Eigen::VectorXs& accelerations,
      const dynamics::BodyNode* contactBody);

  struct MultipleContactInverseDynamicsResult
  {
    dynamics::Skeleton* skel;
    std::vector<const dynamics::BodyNode*> contactBodies;
    std::vector<Eigen::Vector6s> contactWrenches;
    std::vector<Eigen::Vector6s> contactWrenchGuesses;
    Eigen::VectorXs jointTorques;

    // These are the setup of the inverse dynamics problem
    Eigen::VectorXs pos;
    Eigen::VectorXs vel;
    Eigen::VectorXs acc;

    /// This computes how much the actual dynamics we get when we apply this
    /// solution differ from the goal solution.
    s_t sumError();

    /// This computes the difference between the guess and the closest valid
    /// solution
    s_t computeGuessLoss();
  };

  /// If you pass in multiple simultaneous contacts, with guesses about the
  /// contact wrenches for each body, this method will find the least-squares
  /// closest solution for contact wrenches on each body that will satisfying
  /// the next velocity constraint. This is intended to be useful for EM loops
  /// for learning rich contact models.
  ///
  /// Without initial guesses (pass in an empty array for guesses), the solution
  /// is not unique, so in order to use this method to get useful inverse
  /// dynamics we use a heuristic: we find the inverse dynamics that minimizes
  /// the joint torques.
  MultipleContactInverseDynamicsResult getMultipleContactInverseDynamics(
      const Eigen::VectorXs& accelerations,
      std::vector<const dynamics::BodyNode*> bodies,
      std::vector<Eigen::Vector6s> bodyWrenchGuesses);

  struct MultipleContactCoPProblem
  {
    std::vector<const dynamics::BodyNode*> bodies;
    Eigen::VectorXs massTorques;
    Eigen::VectorXs coriolisAndGravity;
    Eigen::MatrixXs jacs;
    Eigen::MatrixXs jacBlock;
    Eigen::FullPivLU<Eigen::MatrixXs> lu;
    Eigen::MatrixXs J_null_space;
    Eigen::Vector6s rootTorque;
    std::vector<Eigen::Vector9s> copWrenchGuesses;
    s_t groundHeight;
    int verticalAxis;

    s_t weightForceToMeters;

    Eigen::VectorXs getInitialGuess();
    s_t getLoss(const Eigen::VectorXs& x);
    s_t getAvgCoPDistance(const Eigen::VectorXs& x);
    Eigen::VectorXs getUnconstrainedGradient(const Eigen::VectorXs& x);
    Eigen::VectorXs finiteDifferenceUnconstrainedGradient(
        const Eigen::VectorXs& x);
    Eigen::VectorXs projectToNullSpace(const Eigen::VectorXs& x);
    Eigen::VectorXs clampToNearestLegalValues(const Eigen::VectorXs& x);
    Eigen::Vector6s getConstraintErrors(const Eigen::VectorXs& x);
  };

  /// This is just broken out to make testing easier, this creates an
  /// optimization problem object that can be used by
  /// getMultipleContactInverseDynamicsNearCoP()
  MultipleContactCoPProblem createMultipleContactInverseDynamicsNearCoPProblem(
      const Eigen::VectorXs& accelerations,
      std::vector<const dynamics::BodyNode*> bodies,
      std::vector<Eigen::Vector9s> copWrenchGuesses,
      s_t groundHeight,
      int verticalAxis);

  /// This performs a similar task to getMultipleContactInverseDynamics(), but
  /// it resolves ambiguity by attempting to find contact forces that are as
  /// closes as possible to the center-of-pressure (CoP) guesses.
  MultipleContactInverseDynamicsResult getMultipleContactInverseDynamicsNearCoP(
      const Eigen::VectorXs& accelerations,
      std::vector<const dynamics::BodyNode*> bodies,
      std::vector<Eigen::Vector6s> bodyWrenchGuesses,
      s_t groundHeight,
      int verticalAxis,
      s_t weightForceToMeters = 0.001,
      bool logOutput = false);

  struct MultipleContactInverseDynamicsOverTimeResult
  {
    dynamics::Skeleton* skel;
    std::vector<const dynamics::BodyNode*> contactBodies;

    int timesteps;

    // One entry / column per timestep
    std::vector<std::vector<Eigen::Vector6s>> contactWrenches;
    Eigen::MatrixXs jointTorques;

    // One column per timestep
    Eigen::MatrixXs positions;
    Eigen::MatrixXs velocities;
    Eigen::MatrixXs accelerations;

    // Problem setup
    std::vector<Eigen::Vector6s> prevContactForces;

    /// This computes how much the actual dynamics we get when we apply this
    /// solution differ from the goal solution.
    s_t sumError();

    /// This computes the (unweighted) smoothness loss for this problem
    s_t computeSmoothnessLoss();

    /// This computes the (unweighted) prev force loss for this problem
    s_t computePrevForceLoss();
  };

  /// This sets up and solves a QP that tracks multiple contacts over a
  /// time-series of positions. This has two blending factors to control the
  /// solution, a `smoothingWeight` and a `minTorqueWeight`. Increasing the
  /// smoothing weight will prioritize a smoother (less time varying) set of
  /// contact forces. Increasing the minimize torques weight will prioritize
  /// solutions at each timestep that minimize the torque-component of the
  /// contact forces at each body.
  ///
  /// This will not provide a contact solution at the last two timesteps passed
  /// in, because it cannot compute a velocity and acceleration at those
  /// timesteps.
  MultipleContactInverseDynamicsOverTimeResult
  getMultipleContactInverseDynamicsOverTime(
      const Eigen::MatrixXs& positions,
      std::vector<const dynamics::BodyNode*> bodies,
      // This allows us to penalize non-smooth GRFs
      s_t smoothingWeight,
      // This allows us to penalize large torques in our GRFs
      s_t minTorqueWeight,
      // This allows us to penalize GRFs on rapidly moving bodies
      std::function<s_t(s_t)> velocityPenalty = [](s_t) { return 0.0; },
      // This allows us to specify exactly what we want the initial forces to be
      std::vector<Eigen::Vector6s> prevContactForces
      = std::vector<Eigen::Vector6s>(),
      s_t prevContactWeight = 0.0,
      // This allows us to specify how we'd like to penalize magnitudes of
      // different contact forces frame-by-frame
      Eigen::MatrixXs magnitudeCosts = EMPTY);

  //----------------------------------------------------------------------------
  /// \{ \name Support Polygon
  //----------------------------------------------------------------------------

  /// Get the support polygon of this Skeleton, which is computed based on the
  /// gravitational projection of the support geometries of all EndEffectors
  /// in this Skeleton that are currently in support mode.
  const math::SupportPolygon& getSupportPolygon() const;

  /// Same as getSupportPolygon(), but it will only use EndEffectors within the
  /// specified tree within this Skeleton
  const math::SupportPolygon& getSupportPolygon(std::size_t _treeIdx) const;

  /// Get a list of the EndEffector indices that correspond to each of the
  /// points in the support polygon.
  const std::vector<std::size_t>& getSupportIndices() const;

  /// Same as getSupportIndices(), but it corresponds to the support polygon of
  /// the specified tree within this Skeleton
  const std::vector<std::size_t>& getSupportIndices(std::size_t _treeIdx) const;

  /// Get the axes that correspond to each component in the support polygon.
  /// These axes are needed in order to map the points on a support polygon
  /// into 3D space. If gravity is along the z-direction, then these axes will
  /// simply be <1,0,0> and <0,1,0>.
  const std::pair<Eigen::Vector3s, Eigen::Vector3s>& getSupportAxes() const;

  /// Same as getSupportAxes(), but it corresponds to the support polygon of the
  /// specified tree within this Skeleton
  const std::pair<Eigen::Vector3s, Eigen::Vector3s>& getSupportAxes(
      std::size_t _treeIdx) const;

  /// Get the centroid of the support polygon for this Skeleton. If the support
  /// polygon is an empty set, the components of this vector will be nan.
  const Eigen::Vector2s& getSupportCentroid() const;

  /// Get the centroid of the support polygon for a tree in this Skeleton. If
  /// the support polygon is an empty set, the components of this vector will be
  /// nan.
  const Eigen::Vector2s& getSupportCentroid(std::size_t _treeIdx) const;

  /// The version number of a support polygon will be incremented each time the
  /// support polygon needs to be recomputed. This number can be used to
  /// immediately determine whether the support polygon has changed since the
  /// last time you asked for it, allowing you to be more efficient in how you
  /// handle the data.
  std::size_t getSupportVersion() const;

  /// Same as getSupportVersion(), but it corresponds to the support polygon of
  /// the specified tree within this Skeleton
  std::size_t getSupportVersion(std::size_t _treeIdx) const;

  /// \}

  //----------------------------------------------------------------------------
  // Kinematics algorithms
  //----------------------------------------------------------------------------

  /// Compute forward kinematics
  ///
  /// In general, this function doesn't need to be called for forward kinematics
  /// to update. Forward kinematics will always be computed when it's needed and
  /// will only perform the computations that are necessary for what the user
  /// requests. This works by performing some bookkeeping internally with dirty
  /// flags whenever a position, velocity, or acceleration is set, either
  /// internally or by the user.
  ///
  /// On one hand, this results in some overhead due to the extra effort of
  /// bookkeeping, but on the other hand we have much greater code safety, and
  /// in some cases performance can be dramatically improved with the auto-
  /// updating. For example, this function is inefficient when only one portion
  /// of the BodyNodes needed to be updated rather than the entire Skeleton,
  /// which is common when performing inverse kinematics on a limb or on some
  /// subsection of a Skeleton.
  ///
  /// This function might be useful in a case where the user wants to perform
  /// all the forward kinematics computations during a particular time window
  /// rather than waiting for it to be computed at the exact time that it's
  /// needed.
  ///
  /// One example would be a real time controller. Let's say a controller gets
  /// encoder data at time t0 but needs to wait until t1 before it receives the
  /// force-torque sensor data that it needs in order to compute the output for
  /// an operational space controller. Instead of being idle from t0 to t1, it
  /// could use that time to compute the forward kinematics by calling this
  /// function.
  void computeForwardKinematics(
      bool _updateTransforms = true,
      bool _updateVels = true,
      bool _updateAccs = true);

  //----------------------------------------------------------------------------
  // Dynamics algorithms
  //----------------------------------------------------------------------------

  /// Compute forward dynamics
  void computeForwardDynamics();

  /// Compute inverse dynamics
  void computeInverseDynamics(
      bool _withExternalForces = false,
      bool _withDampingForces = false,
      bool _withSpringForces = false);

  //----------------------------------------------------------------------------
  // Impulse-based dynamics algorithms
  //----------------------------------------------------------------------------

  /// Clear constraint impulses and cache data used for impulse-based forward
  /// dynamics algorithm, where the constraint impulses are spatial constraints
  /// on the BodyNodes and generalized constraints on the Joints.
  void clearConstraintImpulses();

  /// Update bias impulses
  void updateBiasImpulse(BodyNode* _bodyNode);

  /// \brief Update bias impulses due to impulse [_imp] on body node [_bodyNode]
  /// \param _bodyNode Body node contraint impulse, _imp, is applied
  /// \param _imp Constraint impulse expressed in body frame of _bodyNode
  void updateBiasImpulse(BodyNode* _bodyNode, const Eigen::Vector6s& _imp);

  /// \brief Update bias impulses due to impulse [_imp] on body node [_bodyNode]
  /// \param _bodyNode1 Body node contraint impulse, _imp1, is applied
  /// \param _imp1 Constraint impulse expressed in body frame of _bodyNode1
  /// \param _bodyNode2 Body node contraint impulse, _imp2, is applied
  /// \param _imp2 Constraint impulse expressed in body frame of _bodyNode2
  void updateBiasImpulse(
      BodyNode* _bodyNode1,
      const Eigen::Vector6s& _imp1,
      BodyNode* _bodyNode2,
      const Eigen::Vector6s& _imp2);

  /// \brief Update bias impulses due to impulse[_imp] on body node [_bodyNode]
  void updateBiasImpulse(
      SoftBodyNode* _softBodyNode,
      PointMass* _pointMass,
      const Eigen::Vector3s& _imp);

  /// \brief Update velocity changes in body nodes and joints due to applied
  /// impulse
  void updateVelocityChange();

  // TODO(JS): Better naming
  /// Set whether this skeleton is constrained. ConstraintSolver will
  ///  mark this.
  void setImpulseApplied(bool _val);

  /// Get whether this skeleton is constrained
  bool isImpulseApplied() const;

  /// Compute impulse-based forward dynamics
  void computeImpulseForwardDynamics();

  //----------------------------------------------------------------------------
  /// \{ \name Jacobians
  //----------------------------------------------------------------------------

  // Documentation inherited
  math::Jacobian getJacobian(const JacobianNode* _node) const override;

  // Documentation inherited
  math::Jacobian getJacobianInPositionSpace(const JacobianNode* _node) const;

  // Documentation inherited
  math::Jacobian getJacobian(
      const JacobianNode* _node, const Frame* _inCoordinatesOf) const override;

  // Documentation inherited
  math::Jacobian getJacobian(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset) const override;

  // Documentation inherited
  math::Jacobian getJacobian(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset,
      const Frame* _inCoordinatesOf) const override;

  // Documentation inherited
  math::Jacobian getWorldPositionJacobian(const JacobianNode* _node) const;

  math::Jacobian getWorldPositionJacobian(
      const JacobianNode* _node, const Eigen::Vector3s& _localOffset) const;

  // Documentation inherited
  math::Jacobian finiteDifferenceWorldPositionJacobian(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset,
      bool useRidders = true);

  // Documentation inherited
  math::Jacobian getWorldJacobian(const JacobianNode* _node) const override;

  // Documentation inherited
  math::Jacobian getWorldJacobian(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset) const override;

  // Documentation inherited
  math::LinearJacobian getLinearJacobian(
      const JacobianNode* _node,
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  // Documentation inherited
  math::LinearJacobian getLinearJacobian(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset,
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  // Documentation inherited
  math::AngularJacobian getAngularJacobian(
      const JacobianNode* _node,
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  // Documentation inherited
  math::Jacobian getJacobianSpatialDeriv(
      const JacobianNode* _node) const override;

  // Documentation inherited
  math::Jacobian getJacobianSpatialDeriv(
      const JacobianNode* _node, const Frame* _inCoordinatesOf) const override;

  // Documentation inherited
  math::Jacobian getJacobianSpatialDeriv(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset) const override;

  // Documentation inherited
  math::Jacobian getJacobianSpatialDeriv(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset,
      const Frame* _inCoordinatesOf) const override;

  // Documentation inherited
  math::Jacobian getJacobianClassicDeriv(
      const JacobianNode* _node) const override;

  // Documentation inherited
  math::Jacobian getJacobianClassicDeriv(
      const JacobianNode* _node, const Frame* _inCoordinatesOf) const override;

  // Documentation inherited
  math::Jacobian getJacobianClassicDeriv(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset,
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  // Documentation inherited
  math::LinearJacobian getLinearJacobianDeriv(
      const JacobianNode* _node,
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  // Documentation inherited
  math::LinearJacobian getLinearJacobianDeriv(
      const JacobianNode* _node,
      const Eigen::Vector3s& _localOffset,
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  // Documentation inherited
  math::AngularJacobian getAngularJacobianDeriv(
      const JacobianNode* _node,
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Equations of Motion
  //----------------------------------------------------------------------------

  /// Get total mass of the skeleton. The total mass is calculated as BodyNodes
  /// are added and is updated as BodyNode mass is changed, so this is a
  /// constant-time O(1) operation for the Skeleton class.
  s_t getMass() const override;

  /// Get the mass matrix of a specific tree in the Skeleton
  const Eigen::MatrixXs& getMassMatrix(std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::MatrixXs& getMassMatrix() const override;

  /// Get the augmented mass matrix of a specific tree in the Skeleton
  const Eigen::MatrixXs& getAugMassMatrix(std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::MatrixXs& getAugMassMatrix() const override;

  /// Get the inverse mass matrix of a specific tree in the Skeleton
  const Eigen::MatrixXs& getInvMassMatrix(std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::MatrixXs& getInvMassMatrix() const override;

  /// Get the inverse augmented mass matrix of a tree
  const Eigen::MatrixXs& getInvAugMassMatrix(std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::MatrixXs& getInvAugMassMatrix() const override;

  // Returns the value of M*x, left multiplying x by the mass matrix. This is
  // O(n) compared with O(n^2) to form the complete mass matrix and then
  // multiply.
  Eigen::VectorXs multiplyByImplicitMassMatrix(Eigen::VectorXs x);

  // Returns the value of M_inv*x, left multiplying x by the inverse mass
  // matrix. This is O(n) compared with O(n^2) to form the complete inverse mass
  // matrix and then multiply.
  Eigen::VectorXs multiplyByImplicitInvMassMatrix(Eigen::VectorXs x);

  /// Get the Coriolis force vector of a tree in this Skeleton
  const Eigen::VectorXs& getCoriolisForces(std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::VectorXs& getCoriolisForces() const override;

  /// Get the gravity forces for a tree in this Skeleton
  const Eigen::VectorXs& getGravityForces(std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::VectorXs& getGravityForces() const override;

  /// Get the combined vector of Coriolis force and gravity force of a tree
  const Eigen::VectorXs& getCoriolisAndGravityForces(
      std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::VectorXs& getCoriolisAndGravityForces() const override;

  /// Get the external force vector of a tree in the Skeleton
  const Eigen::VectorXs& getExternalForces(std::size_t _treeIdx) const;

  // Documentation inherited
  const Eigen::VectorXs& getExternalForces() const override;

  // Get damping coefficients
  Eigen::VectorXs getDampingCoeffVector();

  // Get damping force of the skeleton.
  Eigen::VectorXs getDampingForce();

  // Get spring coefficients
  Eigen::VectorXs getSpringStiffVector();

  // Get rest positions
  Eigen::VectorXs getRestPositions();

  // Get Spring Forces
  Eigen::VectorXs getSpringForce();

  /// Get constraint force vector for a tree
  const Eigen::VectorXs& getConstraintForces(std::size_t _treeIdx) const;

  /// Get constraint force vector
  const Eigen::VectorXs& getConstraintForces() const override;

  // Documentation inherited
  void clearExternalForces() override;

  // Documentation inherited
  void clearInternalForces() override;

  /// Notify that the articulated inertia and everything that depends on it
  /// needs to be updated
  DART_DEPRECATED(6.2)
  void notifyArticulatedInertiaUpdate(std::size_t _treeIdx);

  /// Notify that the articulated inertia and everything that depends on it
  /// needs to be updated
  void dirtyArticulatedInertia(std::size_t _treeIdx);

  /// Notify that the support polygon of a tree needs to be updated
  DART_DEPRECATED(6.2)
  void notifySupportUpdate(std::size_t _treeIdx);

  /// Notify that the support polygon of a tree needs to be updated
  void dirtySupportPolygon(std::size_t _treeIdx);

  // Documentation inherited
  s_t computeKineticEnergy() const override;

  // Documentation inherited
  s_t computePotentialEnergy() const override;

  // Documentation inherited
  DART_DEPRECATED(6.0)
  void clearCollidingBodies() override;

  /// \}

  //----------------------------------------------------------------------------
  /// \{ \name Center of Mass Jacobian
  //----------------------------------------------------------------------------

  /// Get the Skeleton's COM with respect to any Frame (default is World Frame)
  Eigen::Vector3s getCOM(
      const Frame* _withRespectTo = Frame::World()) const override;

  /// Get the Skeleton's COM spatial velocity in terms of any Frame (default is
  /// World Frame)
  Eigen::Vector6s getCOMSpatialVelocity(
      const Frame* _relativeTo = Frame::World(),
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// Get the Skeleton's COM linear velocity in terms of any Frame (default is
  /// World Frame)
  Eigen::Vector3s getCOMLinearVelocity(
      const Frame* _relativeTo = Frame::World(),
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// Get the Skeleton's COM spatial acceleration in terms of any Frame (default
  /// is World Frame)
  Eigen::Vector6s getCOMSpatialAcceleration(
      const Frame* _relativeTo = Frame::World(),
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// Get the Skeleton's COM linear acceleration in terms of any Frame (default
  /// is World Frame)
  Eigen::Vector3s getCOMLinearAcceleration(
      const Frame* _relativeTo = Frame::World(),
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// Get the Skeleton's COM Jacobian in terms of any Frame (default is World
  /// Frame)
  math::Jacobian getCOMJacobian(
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// Get the Skeleton's COM Position Jacobian. This is different from the
  /// standard
  // COM Jacobian because of FreeJoint and BallJoints.
  math::Jacobian getCOMPositionJacobian() const;

  /// Get the Skeleton's COM Linear Jacobian in terms of any Frame (default is
  /// World Frame)
  math::LinearJacobian getCOMLinearJacobian(
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// Get the Skeleton's COM Jacobian spatial time derivative in terms of any
  /// Frame (default is World Frame).
  ///
  /// NOTE: Since this is a spatial time derivative, it is only meant to be used
  /// with spatial acceleration vectors. If you are using classical linear
  /// vectors, then use getCOMLinearJacobianDeriv() instead.
  math::Jacobian getCOMJacobianSpatialDeriv(
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// Get the Skeleton's COM Linear Jacobian time derivative in terms of any
  /// Frame (default is World Frame).
  ///
  /// NOTE: Since this is a classical time derivative, it is only meant to be
  /// used with classical acceleration vectors. If you are using spatial
  /// vectors, then use getCOMJacobianSpatialDeriv() instead.
  math::LinearJacobian getCOMLinearJacobianDeriv(
      const Frame* _inCoordinatesOf = Frame::World()) const override;

  /// \}

  //----------------------------------------------------------------------------
  // Friendship
  //----------------------------------------------------------------------------
  friend class BodyNode;
  friend class SoftBodyNode;
  friend class Joint;
  template <class>
  friend class GenericJoint;
  friend class DegreeOfFreedom;
  friend class Node;
  friend class ShapeNode;
  friend class EndEffector;

protected:
  struct DataCache;

  /// Constructor called by create()
  Skeleton(const AspectPropertiesData& _properties);

  /// Setup this Skeleton with its shared_ptr
  void setPtr(const SkeletonPtr& _ptr);

  /// Construct a new tree in the Skeleton
  void constructNewTree();

  /// Register a BodyNode with the Skeleton. Internal use only.
  void registerBodyNode(BodyNode* _newBodyNode);

  /// Register a Joint with the Skeleton. Internal use only.
  void registerJoint(Joint* _newJoint);

  /// Register a Node with the Skeleton. Internal use only.
  void registerNode(NodeMap& nodeMap, Node* _newNode, std::size_t& _index);

  /// Register a Node with the Skeleton. Internal use only.
  void registerNode(Node* _newNode);

  /// Remove an old tree from the Skeleton
  void destructOldTree(std::size_t tree);

  /// Remove a BodyNode from the Skeleton. Internal use only.
  void unregisterBodyNode(BodyNode* _oldBodyNode);

  /// Remove a Joint from the Skeleton. Internal use only.
  void unregisterJoint(Joint* _oldJoint);

  /// Remove a Node from the Skeleton. Internal use only.
  void unregisterNode(NodeMap& nodeMap, Node* _oldNode, std::size_t& _index);

  /// Remove a Node from the Skeleton. Internal use only.
  void unregisterNode(Node* _oldNode);

  /// Move a subtree of BodyNodes from this Skeleton to another Skeleton
  bool moveBodyNodeTree(
      Joint* _parentJoint,
      BodyNode* _bodyNode,
      SkeletonPtr _newSkeleton,
      BodyNode* _parentNode);

  /// Move a subtree of BodyNodes from this Skeleton to another Skeleton while
  /// changing the Joint type of the top parent Joint.
  ///
  /// Returns a nullptr if the move failed for any reason.
  template <class JointType>
  JointType* moveBodyNodeTree(
      BodyNode* _bodyNode,
      const SkeletonPtr& _newSkeleton,
      BodyNode* _parentNode,
      const typename JointType::Properties& _joint);

  /// Copy a subtree of BodyNodes onto another Skeleton while leaving the
  /// originals intact
  std::pair<Joint*, BodyNode*> cloneBodyNodeTree(
      Joint* _parentJoint,
      const BodyNode* _bodyNode,
      const SkeletonPtr& _newSkeleton,
      BodyNode* _parentNode,
      bool _recursive) const;

  /// Copy a subtree of BodyNodes onto another Skeleton while leaving the
  /// originals intact, but alter the top parent Joint to a new type
  template <class JointType>
  std::pair<JointType*, BodyNode*> cloneBodyNodeTree(
      const BodyNode* _bodyNode,
      const SkeletonPtr& _newSkeleton,
      BodyNode* _parentNode,
      const typename JointType::Properties& _joint,
      bool _recursive) const;

  /// Create a vector representation of a subtree of BodyNodes
  std::vector<const BodyNode*> constructBodyNodeTree(
      const BodyNode* _bodyNode) const;

  std::vector<BodyNode*> constructBodyNodeTree(BodyNode* _bodyNode);

  /// Create a vector representation of a subtree of BodyNodes and remove that
  /// subtree from this Skeleton without deleting them
  std::vector<BodyNode*> extractBodyNodeTree(BodyNode* _bodyNode);

  /// Take in and register a subtree of BodyNodes
  void receiveBodyNodeTree(const std::vector<BodyNode*>& _tree);

  /// Update the computation for total mass
  void updateTotalMass();

  /// Update the dimensions for a specific cache
  void updateCacheDimensions(DataCache& _cache);

  /// Update the dimensions for a tree's cache
  void updateCacheDimensions(std::size_t _treeIdx);

  /// Update the articulated inertia of a tree
  void updateArticulatedInertia(std::size_t _tree) const;

  /// Update the articulated inertias of the skeleton
  void updateArticulatedInertia() const;

  /// Update the mass matrix of a tree
  void updateMassMatrix(std::size_t _treeIdx) const;

  /// Update mass matrix of the skeleton.
  void updateMassMatrix() const;

  void updateAugMassMatrix(std::size_t _treeIdx) const;

  /// Update augmented mass matrix of the skeleton.
  void updateAugMassMatrix() const;

  /// Update the inverse mass matrix of a tree
  void updateInvMassMatrix(std::size_t _treeIdx) const;

  /// Update inverse of mass matrix of the skeleton.
  void updateInvMassMatrix() const;

  /// Update the inverse augmented mass matrix of a tree
  void updateInvAugMassMatrix(std::size_t _treeIdx) const;

  /// Update inverse of augmented mass matrix of the skeleton.
  void updateInvAugMassMatrix() const;

  /// Update Coriolis force vector for a tree in the Skeleton
  void updateCoriolisForces(std::size_t _treeIdx) const;

  /// Update Coriolis force vector of the skeleton.
  void updateCoriolisForces() const;

  /// Update the gravity force vector of a tree
  void updateGravityForces(std::size_t _treeIdx) const;

  /// Update gravity force vector of the skeleton.
  void updateGravityForces() const;

  /// Update the combined vector for a tree in this Skeleton
  void updateCoriolisAndGravityForces(std::size_t _treeIdx) const;

  /// Update combined vector of the skeleton.
  void updateCoriolisAndGravityForces() const;

  /// Update external force vector to generalized forces for a tree
  void updateExternalForces(std::size_t _treeIdx) const;

  // TODO(JS): Not implemented yet
  /// update external force vector to generalized forces.
  void updateExternalForces() const;

  /// Compute the constraint force vector for a tree
  const Eigen::VectorXs& computeConstraintForces(DataCache& cache) const;

  //  /// Update damping force vector.
  //  virtual void updateDampingForceVector();

  /// Add a BodyNode to the BodyNode NameManager
  const std::string& addEntryToBodyNodeNameMgr(BodyNode* _newNode);

  /// Add a Joint to to the Joint NameManager
  const std::string& addEntryToJointNameMgr(
      Joint* _newJoint, bool _updateDofNames = true);

  /// Add a SoftBodyNode to the SoftBodyNode NameManager
  void addEntryToSoftBodyNodeNameMgr(SoftBodyNode* _newNode);

protected:
  /// The resource-managing pointer to this Skeleton
  std::weak_ptr<Skeleton> mPtr;

  /// List of Soft body node list in the skeleton
  std::vector<SoftBodyNode*> mSoftBodyNodes;

  /// NameManager for tracking BodyNodes
  dart::common::NameManager<BodyNode*> mNameMgrForBodyNodes;

  /// NameManager for tracking Joints
  dart::common::NameManager<Joint*> mNameMgrForJoints;

  /// NameManager for tracking DegreesOfFreedom
  dart::common::NameManager<DegreeOfFreedom*> mNameMgrForDofs;

  /// NameManager for tracking SoftBodyNodes
  dart::common::NameManager<SoftBodyNode*> mNameMgrForSoftBodyNodes;

  /// The groups that constrain the scales of body nodes to be equal
  std::vector<BodyScaleGroup> mBodyScaleGroups;

  /// This is a cache for the data around our group scales
  std::vector<BodyScaleGroupAndIndex> mGroupScaleIndices;

  /// This is a cache for looking up meshes attached to bodies
  std::map<std::string, std::pair<dynamics::BodyNode*, Eigen::Isometry3s>>
      mMeshBodyCache;
  std::map<std::string, bool> mMeshExistsCache;

  struct DirtyFlags
  {
    /// Default constructor
    DirtyFlags();

    /// Dirty flag for articulated body inertia
    bool mArticulatedInertia;

    /// Dirty flag for the mass matrix.
    bool mMassMatrix;

    /// Dirty flag for the mass matrix.
    bool mAugMassMatrix;

    /// Dirty flag for the inverse of mass matrix.
    bool mInvMassMatrix;

    /// Dirty flag for the inverse of augmented mass matrix.
    bool mInvAugMassMatrix;

    /// Dirty flag for the gravity force vector.
    bool mGravityForces;

    /// Dirty flag for the Coriolis force vector.
    bool mCoriolisForces;

    /// Dirty flag for the combined vector of Coriolis and gravity.
    bool mCoriolisAndGravityForces;

    /// Dirty flag for the external force vector.
    bool mExternalForces;

    /// Dirty flag for the damping force vector.
    bool mDampingForces;

    /// Dirty flag for the support polygon
    bool mSupport;

    /// Dirty flag for the parent maps
    bool mDofParentMap;
    bool mJointParentMap;

    /// Increments each time a new support polygon is computed to help keep
    /// track of changes in the support polygon
    std::size_t mSupportVersion;
  };

  struct DataCache
  {
    DirtyFlags mDirty;

    /// BodyNodes belonging to this tree
    std::vector<BodyNode*> mBodyNodes;

    /// Cache for const BodyNodes, for the sake of the API
    std::vector<const BodyNode*> mConstBodyNodes;

    /// Degrees of Freedom belonging to this tree
    std::vector<DegreeOfFreedom*> mDofs;

    /// Cache for const Degrees of Freedom, for the sake of the API
    std::vector<const DegreeOfFreedom*> mConstDofs;

    /// Mass matrix cache
    Eigen::MatrixXs mM;

    /// Mass matrix for the skeleton.
    Eigen::MatrixXs mAugM;

    /// Inverse of mass matrix for the skeleton.
    Eigen::MatrixXs mInvM;

    /// Inverse of augmented mass matrix for the skeleton.
    Eigen::MatrixXs mInvAugM;

    /// Coriolis vector for the skeleton which is C(q,dq)*dq.
    Eigen::VectorXs mCvec;

    /// Gravity vector for the skeleton; computed in nonrecursive
    /// dynamics only.
    Eigen::VectorXs mG;

    /// Combined coriolis and gravity vector which is C(q, dq)*dq + g(q).
    Eigen::VectorXs mCg;

    /// External force vector for the skeleton.
    Eigen::VectorXs mFext;

    /// Constraint force vector.
    Eigen::VectorXs mFc;

    /// Support polygon
    math::SupportPolygon mSupportPolygon;

    /// A map of which EndEffectors correspond to the individual points in the
    /// support polygon
    std::vector<std::size_t> mSupportIndices;

    /// A pair of vectors which map the 2D coordinates of the support polygon
    /// into 3D space
    std::pair<Eigen::Vector3s, Eigen::Vector3s> mSupportAxes;

    /// Support geometry -- only used for temporary storage purposes
    math::SupportGeometry mSupportGeometry;

    /// Centroid of the support polygon
    Eigen::Vector2s mSupportCentroid;

    /// A map of the parent relationships between dofs in this skeleton.
    Eigen::MatrixXi mDofParentMap;
    Eigen::MatrixXi mJointParentMap;

    /// A shared pointer to the saved gradient matrices for the ConstrainedGroup
    /// this skeleton was part of in the last LCP solve.
    std::shared_ptr<neural::ConstrainedGroupGradientMatrices>
        mGradientConstraintMatrices;

    // To get byte-aligned Eigen vectors
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  mutable common::aligned_vector<DataCache> mTreeCache;

  mutable DataCache mSkelCache;

  using SpecializedTreeNodes
      = std::map<std::type_index, std::vector<NodeMap::iterator>*>;

  SpecializedTreeNodes mSpecializedTreeNodes;

  /// Total mass.
  s_t mTotalMass;

  // TODO(JS): Better naming
  /// Flag for status of impulse testing.
  bool mIsImpulseApplied;

  mutable std::mutex mMutex;

public:
  //--------------------------------------------------------------------------
  // Union finding
  //--------------------------------------------------------------------------
  ///
  void resetUnion()
  {
    mUnionRootSkeleton = mPtr;
    mUnionSize = 1;
  }

  ///
  std::weak_ptr<Skeleton> mUnionRootSkeleton;

  ///
  std::size_t mUnionSize;

  ///
  std::size_t mUnionIndex;
};

} // namespace dynamics
} // namespace dart

#include "dart/dynamics/detail/Skeleton.hpp"

#endif // DART_DYNAMICS_SKELETON_HPP_
