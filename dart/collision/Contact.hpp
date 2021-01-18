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

#ifndef DART_COLLISION_CONTACT_HPP_
#define DART_COLLISION_CONTACT_HPP_

#include <Eigen/Dense>

#include "dart/collision/SmartPointer.hpp"
#include "dart/dynamics/SmartPointer.hpp"

namespace dart {
namespace collision {

enum ContactType
{
  UNSUPPORTED
  = 0, // This is the default, and means that gradients won't attempt to model
       // how the contact point changes as we move the skeletons.

  // Mesh-mesh collisions
  VERTEX_FACE = 1,
  FACE_VERTEX = 2,
  EDGE_EDGE = 3,

  // Sphere-box collisions
  SPHERE_BOX = 4,
  BOX_SPHERE = 5,

  // Sphere-sphere collisions
  SPHERE_SPHERE = 6,

  // Sphere-mesh collisions
  SPHERE_VERTEX = 7,
  SPHERE_EDGE = 8,
  SPHERE_FACE = 9,
  VERTEX_SPHERE = 10,
  EDGE_SPHERE = 11,
  FACE_SPHERE = 12,

  // Capsule-capsule and capsule-sphere collisions
  PIPE_SPHERE = 13,
  SPHERE_PIPE = 14,
  PIPE_PIPE = 15,

  // Pipe-mesh collisions
  PIPE_VERTEX = 16,
  PIPE_EDGE = 17,
  VERTEX_PIPE = 18,
  EDGE_PIPE = 19,
};

/// Contact information
struct Contact
{
  /// Default constructor
  Contact();

  /// Contact point w.r.t. the world frame
  Eigen::Vector3d point;

  /// Contact normal vector from bodyNode2 to bodyNode1 w.r.t. the world frame
  Eigen::Vector3d normal;

  /// Contact force acting on bodyNode1 w.r.t. the world frame
  ///
  /// The contact force acting on bodyNode2 is -force, which is the opposite
  /// direction of the force.
  Eigen::Vector3d force;

  /// First colliding collision object
  CollisionObject* collisionObject1;

  /// Second colliding collision object
  CollisionObject* collisionObject2;

  /// Penetration depth
  double penetrationDepth;

  // TODO(JS): triID1 will be deprecated when we don't use fcl_mesh
  /// \brief
  int triID1;

  // TODO(JS): triID2 will be deprecated when we don't use fcl_mesh
  /// \brief
  int triID2;

  // TODO(JS): userData is an experimental variable.
  /// \brief User data.
  void* userData;

  /// This holds the amount of force the LCP solved for for this contact
  double lcpResult;

  /// This is necessary for computing gradients. This tells us what type of
  /// contact generated these contacts.
  ContactType type;

  /// This is only filled for (type == EDGE_EDGE) contacts. This is the closest
  /// point on edge A to edge B.
  Eigen::Vector3d edgeAClosestPoint;

  /// This is only filled for (type == EDGE_EDGE) contacts. This is an arbitrary
  /// fixed point on edge A.
  Eigen::Vector3d edgeAFixedPoint;

  /// This is only filled for (type == EDGE_EDGE) contacts. This is the
  /// direction of edge A
  Eigen::Vector3d edgeADir;

  /// This is only filled for (type == EDGE_EDGE) contacts. This is the closest
  /// point on edge B to edge A.
  Eigen::Vector3d edgeBClosestPoint;

  /// This is only filled for (type == EDGE_EDGE) contacts. This is an arbitrary
  /// fixed point on edge B.
  Eigen::Vector3d edgeBFixedPoint;

  /// This is only filled for (type == EDGE_EDGE) contacts. This is the
  /// direction of edge B
  Eigen::Vector3d edgeBDir;

  /// This is only filled for (type == SPHERE_BOX || type == BOX_SPHERE)
  /// contacts. This holds the center-point of the sphere
  Eigen::Vector3d sphereCenter;

  /// This is useful for SPHERE_FACE, SPHERE_EDGE, SPHERE_VERTEX, etc
  double sphereRadius;

  /// This is only filled for (type == PIPE_SPHERE || type == SPHERE_PIPE)
  Eigen::Vector3d pipeDir;

  /// This is only filled for (type == PIPE_SPHERE || type == SPHERE_PIPE)
  Eigen::Vector3d pipeClosestPoint;

  /// This is only filled for (type == PIPE_SPHERE || type == SPHERE_PIPE)
  Eigen::Vector3d pipeFixedPoint;

  /// This is only filled for (type == PIPE_SPHERE || type == SPHERE_PIPE)
  double pipeRadius;

  /// This is useful for SPHERE_VERTEX, PIPE_VERTEX, etc
  Eigen::Vector3d vertexPoint;

  /// This is only useful for (type == SPHERE_BOX || type == BOX_SPHERE)
  /// contacts. If this is true, then the contact point is clamped on face1.
  bool face1Locked;

  /// This is only filled for (type == SPHERE_BOX || type == BOX_SPHERE)
  /// contacts. If (face1Locked) is true, then this means we don't move the
  /// contact point along this normal.
  Eigen::Vector3d face1Normal;

  /// This is only useful for (type == SPHERE_BOX || type == BOX_SPHERE)
  /// contacts. If this is true, then the contact point is clamped on face2.
  bool face2Locked;

  /// This is only filled for (type == SPHERE_BOX || type == BOX_SPHERE)
  /// contacts. If (face2Locked) is true, then this means we don't move the
  /// contact point along this normal.
  Eigen::Vector3d face2Normal;

  /// This is only useful for (type == SPHERE_BOX || type == BOX_SPHERE)
  /// contacts. If this is true, then the contact point is clamped on face3.
  bool face3Locked;

  /// This is only filled for (type == SPHERE_BOX || type == BOX_SPHERE)
  /// contacts. If (face3Locked) is true, then this means we don't move the
  /// contact point along this normal.
  Eigen::Vector3d face3Normal;

  /// This is only filled for (type == SPHERE_SPHERE) contacts. It's the center
  /// of the A sphere.
  Eigen::Vector3d centerA;

  /// This is only filled for (type == SPHERE_SPHERE) contacts. It's the radius
  /// of the A sphere.
  double radiusA;

  /// This is only filled for (type == SPHERE_SPHERE) contacts. It's the center
  /// of the B sphere.
  Eigen::Vector3d centerB;

  /// This is only filled for (type == SPHERE_SPHERE) contacts. It's the radius
  /// of the B sphere.
  double radiusB;

  /// Returns the epsilon to be used for determination of zero-length normal.
  constexpr static double getNormalEpsilon();

  /// Returns the squired epsilon to be used for determination of zero-length
  /// normal.
  constexpr static double getNormalEpsilonSquared();

  /// Returns true if the length of a normal is less than the epsilon.
  static bool isZeroNormal(const Eigen::Vector3d& normal);

  /// Returns !isZeroNormal().
  static bool isNonZeroNormal(const Eigen::Vector3d& normal);
};

} // namespace collision
} // namespace dart

#include "dart/collision/detail/Contact-impl.hpp"

#endif // DART_COLLISION_CONTACT_HPP_
