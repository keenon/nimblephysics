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

#ifndef DART_COLLISION_DART_DARTCOLLIDE_HPP_
#define DART_COLLISION_DART_DARTCOLLIDE_HPP_

#include <vector>

#include <Eigen/Dense>
#include <assimp/scene.h>
#include <ccd/vec3.h>

#include "dart/collision/CollisionDetector.hpp"

namespace dart {
namespace collision {

int collide(CollisionObject* o1, CollisionObject* o2, CollisionResult& result);

int collideBoxBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result);

int collideBoxSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    const double& r1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result);

int collideSphereBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& r0,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result);

int collideSphereSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& r0,
    const Eigen::Isometry3d& c0,
    const double& r1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result);

int collideBoxBoxAsMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result);

int collideMeshBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result);

int collideBoxMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const aiScene* mesh1,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result);

int collideMeshSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const double& r1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result);

int collideMeshMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const aiScene* mesh1,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result);

int collideCylinderSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& cyl_rad,
    const double& half_height,
    const Eigen::Isometry3d& T0,
    const double& sphere_rad,
    const Eigen::Isometry3d& T1,
    CollisionResult& result);

int collideCylinderPlane(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& cyl_rad,
    const double& half_height,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& plane_normal,
    const Eigen::Isometry3d& T1,
    CollisionResult& result);

/////////////////////////////////////////////////////////////////////
// Interface with libccd:
/////////////////////////////////////////////////////////////////////

// We need to define structs for each object type that we pass to libccd, with
// all relevant info about the object.
struct ccdBox
{
  const Eigen::Vector3d* size;
  const Eigen::Isometry3d* transform;
};

struct ccdSphere
{
  double radius;
  const Eigen::Isometry3d* transform;
};

struct ccdMesh
{
  const aiScene* mesh;
  const Eigen::Isometry3d* transform;
  const Eigen::Vector3d* scale;
};

// We also need to define "support" functions that will find the furthest point
// in the object along the direction "_dir", and return it in "_vec" for each
// type of object.

void ccdSupportBox(const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out);
void ccdSupportSphere(
    const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out);
void ccdSupportMesh(const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out);

// Finally, we need to define the "center" function for objects. This returns
// the approximate center of each object.

void ccdCenterBox(const void* _obj, ccd_vec3_t* _center);
void ccdCenterSphere(const void* _obj, ccd_vec3_t* _center);
void ccdCenterMesh(const void* _obj, ccd_vec3_t* _center);

// In order to differentiate between different types of contact, we need to be
// able to get all the vertices that are within some small epsilon of being on
// the "witness" plane returned by ccd.

std::vector<Eigen::Vector3d> ccdPointsAtWitnessBox(
    ccdBox* box, ccd_vec3_t* dir, bool neg);
std::vector<Eigen::Vector3d> ccdPointsAtWitnessMesh(
    ccdMesh* mesh, ccd_vec3_t* dir, bool neg);

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createMeshMeshContacts(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const std::vector<Eigen::Vector3d>& pointsAWitness,
    const std::vector<Eigen::Vector3d>& pointsBWitness);

/// This is necessary preparation for rapidly checking if another point is
/// contained within the convex shape. This sorts the shape by angle from the
/// center, and trims out any points that lie inside the convex polygon.
void prepareConvex2DShape(std::vector<Eigen::Vector2d>& shape);

/// This checks whether a 2D shape contains a point. This assumes that shape was
/// sorted using sortConvex2DShape().
bool convex2DShapeContains(
    const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& shape);

double angle2D(const Eigen::Vector2d& from, const Eigen::Vector2d& to);

// This implements the "2D cross product" as redefined here:
// https://stackoverflow.com/a/565282/13177487
inline double crossProduct2D(
    const Eigen::Vector2d& v, const Eigen::Vector2d& w);

/// This returns true if the two line segments defined by (p0,p1) and (q0,q1)
/// intersect. If they do intersect, this writes the intersection point to
/// `out`.
bool get2DLineIntersection(
    const Eigen::Vector2d& p0,
    const Eigen::Vector2d& p1,
    const Eigen::Vector2d& q0,
    const Eigen::Vector2d& q1,
    Eigen::Vector2d& out);

} // namespace collision
} // namespace dart

#endif // DART_COLLISION_DART_DARTCOLLIDE_HPP_
