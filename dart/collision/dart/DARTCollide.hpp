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

#include <thread>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <assimp/scene.h>
#include <ccd/ccd.h>
#include <ccd/vec3.h>

#include "dart/collision/CollisionDetector.hpp"

namespace dart {
namespace collision {

int collide(
    CollisionObject* o1,
    CollisionObject* o2,
    const CollisionOption& option,
    CollisionResult& result);

/// This is for when we use the sphere collision routines for capsule-ends. If
/// we have a capsule in deep inter-penetration with another object, we want to
/// only detect collisions on one half of the sphere. This is easy to decide,
/// because it's the Z coordinate of the collision in the local space for the
/// sphere. TOP will allow only Z > 0, BOTTOM will allow only Z < 0, and BOTH
/// will allow either configuration.
enum ClipSphereHalfspace
{
  BOTH = 0,
  TOP = 1,
  BOTTOM = 2
};

/// This is for pipe-edge collisions, when found by the face-face collision
/// algorithm. We want to be able to ensure that all contacts lie exactly in the
/// plane of the mesh/box that is responsible for the collision.
enum PinToFace
{
  AVERAGE = 0,
  FACE_A = 1,
  FACE_B = 2
};

enum SmoothNormWRT
{
    POINT_1 = 0,
    POINT_2 = 1,
    POINT_3 = 2,
};

int collideBoxBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideBoxSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    const s_t& r1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace halfspace = ClipSphereHalfspace::BOTH);

int collideSphereBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const s_t& r0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace halfspace = ClipSphereHalfspace::BOTH);

int collideSphereSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const s_t& r0,
    const Eigen::Isometry3s& c0,
    const s_t& r1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace halfspace0 = ClipSphereHalfspace::BOTH,
    ClipSphereHalfspace halfspace1 = ClipSphereHalfspace::BOTH);

int collideBoxBoxAsMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideMeshBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result);

int collideBoxMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const aiScene* mesh1,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result);

int collideMeshSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const s_t& r1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace halfspace = ClipSphereHalfspace::BOTH);

int collideSphereMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const s_t& r0,
    const Eigen::Isometry3s& c0,
    const aiScene* mesh1,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace halfspace = ClipSphereHalfspace::BOTH);

int collideMeshMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const aiScene* mesh1,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result);

int collideCapsuleCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t height0,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    s_t height1,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideSphereCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    s_t height1,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideCapsuleSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t height0,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideBoxCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    s_t height1,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideCapsuleBox(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t height0,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideMeshCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    s_t height1,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideCapsuleMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t height0,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    const aiScene* mesh1,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

int collideCylinderSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const s_t& cyl_rad,
    const s_t& half_height,
    const Eigen::Isometry3s& T0,
    const s_t& sphere_rad,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace halfspace = ClipSphereHalfspace::BOTH);

int collideCylinderPlane(
    CollisionObject* o1,
    CollisionObject* o2,
    const s_t& cyl_rad,
    const s_t& half_height,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& plane_normal,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result);

/////////////////////////////////////////////////////////////////////
// Interface with libccd:
/////////////////////////////////////////////////////////////////////

// Get the `pos` vec for CCD for this pair of objects
ccd_vec3_t& getCachedCcdPos(CollisionObject* o1, CollisionObject* o2);

// Get the `dir` vec for CCD for this pair of objects
ccd_vec3_t& getCachedCcdDir(CollisionObject* o1, CollisionObject* o2);

// We need to define structs for each object type that we pass to libccd, with
// all relevant info about the object.
struct ccdBox
{
  const Eigen::Vector3s* size;
  const Eigen::Isometry3s* transform;
};

struct ccdSphere
{
  s_t radius;
  const Eigen::Isometry3s* transform;
};

struct ccdMesh
{
  const aiScene* mesh;
  const Eigen::Isometry3s* transform;
  const Eigen::Vector3s* scale;
};

struct ccdCapsule
{
  s_t radius;
  s_t height;
  const Eigen::Isometry3s* transform;
};

// We also need to define "support" functions that will find the furthest point
// in the object along the direction "_dir", and return it in "_vec" for each
// type of object.

void ccdSupportBox(const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out);
void ccdSupportSphere(
    const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out);
void ccdSupportMesh(const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out);
void ccdSupportCapsule(
    const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out);

// Finally, we need to define the "center" function for objects. This returns
// the approximate center of each object.

void ccdCenterBox(const void* _obj, ccd_vec3_t* _center);
void ccdCenterSphere(const void* _obj, ccd_vec3_t* _center);
void ccdCenterMesh(const void* _obj, ccd_vec3_t* _center);
void ccdCenterCapsule(const void* _obj, ccd_vec3_t* _center);

// In order to differentiate between different types of contact, we need to be
// able to get all the vertices that are within some small epsilon of being on
// the "witness" plane returned by ccd.

std::vector<Eigen::Vector3s> ccdPointsAtWitnessBox(
    ccdBox* box, ccd_vec3_t* dir, bool neg);
std::vector<Eigen::Vector3s> ccdPointsAtWitnessMesh(
    ccdMesh* mesh, ccd_vec3_t* dir, bool neg);

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createMeshMeshContacts(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const std::vector<Eigen::Vector3s>& pointsAWitness,
    const std::vector<Eigen::Vector3s>& pointsBWitness);

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createMeshSphereContact(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const std::vector<Eigen::Vector3s>& meshPointsWitness,
    const Eigen::Vector3s& sphereCenter,
    s_t sphereRadius,
    ClipSphereHalfspace halfspace = ClipSphereHalfspace::BOTH);

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createSphereMeshContact(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const Eigen::Vector3s& sphereCenter,
    s_t sphereRadius,
    const std::vector<Eigen::Vector3s>& meshPointsWitness,
    ClipSphereHalfspace halfspace = ClipSphereHalfspace::BOTH);

/*
/// This is necessary preparation for rapidly checking if another point is
/// contained within the convex shape. This sorts the shape by angle from
the
/// center, and trims out any points that lie inside the convex polygon.
void prepareConvex2DShape(std::vector<Eigen::Vector2s>& shape);

/// This checks whether a 2D shape contains a point. This assumes that shape
was
/// sorted using prepareConvex2DShape().
bool convex2DShapeContains(
    const Eigen::Vector2s& point, const std::vector<Eigen::Vector2s>&
shape);
*/

/// This trims out any points that lie inside the convex polygon, without
/// changing the order.
void keepOnlyConvex2DHull(
    std::vector<Eigen::Vector3s>& shape,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY);

/// This is necessary preparation for rapidly checking if another point is
/// contained within the convex shape. This sorts the shape by angle from
/// the center, and trims out any points that lie inside the convex polygon.
void prepareConvex2DShape(
    std::vector<Eigen::Vector3s>& shape,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY);

/// This checks whether a 2D shape contains a point. This assumes that shape was
/// sorted using prepareConvex2DShape().
bool convex2DShapeContains(
    const Eigen::Vector3s& point,
    const std::vector<Eigen::Vector3s>& shape,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY);

/// This transforms a 3D point down to a 2D point in the given 3D plane
Eigen::Vector2s pointInPlane(
    const Eigen::Vector3s& point,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY);

s_t angle2D(const Eigen::Vector2s& from, const Eigen::Vector2s& to);

// This implements the "2D cross product" as redefined here:
// https://stackoverflow.com/a/565282/13177487
inline s_t crossProduct2D(const Eigen::Vector2s& v, const Eigen::Vector2s& w);

// This check whether 3D point p lay in the triangle abc
// If true return 0 , if false return -1
int pointInTriangle(Eigen::Vector3s a, Eigen::Vector3s b, Eigen::Vector3s c, Eigen::Vector3s p);

// This compute the area of triangle from three vertex point based on Helen formula
s_t computeArea(Eigen::Vector3s a, Eigen::Vector3s b, Eigen::Vector3s c);

// This compute the perimeter of triangle from three vertices
s_t computePerimeter(Eigen::Vector3s a, Eigen:: Vector3s b, Eigen::Vector3s c);

// This compute the multiplication of all vectors norm
s_t computeVectorSqrNormMul(std::vector<Eigen::Vector3s> vectors);

// This compute the weight of inverse squared distance interpolation of a point
std::vector<std::vector<Eigen::Vector3s>> computeISDSubcomponent(
    std::vector<Eigen::Vector3s> points,
    Eigen::Vector3s point,
    int index);

s_t computeDenominator(std::vector<std::vector<Eigen::Vector3s>> sub_components);

s_t computeISDWeight(
    std::vector<std::vector<Eigen::Vector3s>> sub_components,
    Eigen::Vector3s point,
    int index);

Eigen::VectorXs ISDInterpolate(
  std::vector<Eigen::VectorXs> vectors,
  std::vector<Eigen::Vector3s> positions,
  Eigen::Vector3s p
);
// This compute the interpolation result of some vector field at point p
Eigen::VectorXs barycentricInterpolate(
    std::vector<Eigen::VectorXs>& vectors,
    std::vector<Eigen::Vector3s>& positions,
    Eigen::Vector3s p
);

/// This returns true if the two line segments defined by (p0,p1) and (q0,q1)
/// intersect. If they do intersect, this writes the intersection point to
/// `out`.
bool get2DLineIntersection(
    const Eigen::Vector2s& p0,
    const Eigen::Vector2s& p1,
    const Eigen::Vector2s& q0,
    const Eigen::Vector2s& q1,
    Eigen::Vector2s& out);

/// This sets the default settings for CCD in a single spot in the DARTCollide
/// code, so it's easy to tweak settings across all collision pairs.
inline void setCcdDefaultSettings(ccd_t& ccd);

/// This allows us to prevent weird effects where we don't want to carry over
/// cacheing
void clearCcdCache();

/// This is the static cache for all the CCD collision search data
static std::unordered_map<std::thread::id, std::unordered_map<long, ccd_vec3_t>>
    _ccdDirCache;
static std::unordered_map<std::thread::id, std::unordered_map<long, ccd_vec3_t>>
    _ccdPosCache;

} // namespace collision
} // namespace dart

#endif // DART_COLLISION_DART_DARTCOLLIDE_HPP_
