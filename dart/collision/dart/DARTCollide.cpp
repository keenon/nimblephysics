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

#include "dart/collision/dart/DARTCollide.hpp"

#include <memory>

#include "dart/collision/CollisionObject.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CapsuleShape.hpp"
#include "dart/dynamics/CylinderShape.hpp"
#include "dart/dynamics/EllipsoidShape.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/math/Helpers.hpp"

namespace dart {
namespace collision {

// point : world coordinate vector
// normal : normal vector from right to left 0 <- 1
// penetration : real positive means penetration

#define DART_COLLISION_WITNESS_PLANE_DEPTH 0.01
#define DART_COLLISION_EPS 1E-6
static const int MAX_CYLBOX_CLIP_POINTS = 16;
static const int nCYLINDER_AXIS = 2;
// Number of segment of cylinder base circle.
// Must be divisible by 4.
static const int nCYLINDER_SEGMENT = 8;

typedef double dVector3[4];
typedef double dVector3[4];
typedef double dVector4[4];
typedef double dMatrix3[12];
typedef double dMatrix4[16];
typedef double dMatrix6[48];
typedef double dQuaternion[4];

inline void convVector(const Eigen::Vector3d& p0, dVector3& p1)
{
  p1[0] = p0[0];
  p1[1] = p0[1];
  p1[2] = p0[2];
}

inline void convMatrix(const Eigen::Isometry3d& T0, dMatrix3& R0)
{
  R0[0] = T0(0, 0);
  R0[1] = T0(0, 1);
  R0[2] = T0(0, 2);
  R0[3] = T0(0, 3);
  R0[4] = T0(1, 0);
  R0[5] = T0(1, 1);
  R0[6] = T0(1, 2);
  R0[7] = T0(1, 3);
  R0[8] = T0(2, 0);
  R0[9] = T0(2, 1);
  R0[10] = T0(2, 2);
  R0[11] = T0(2, 3);
}

struct dContactGeom
{
  dVector3 pos;
  double depth;
};

inline double Inner(const double* a, const double* b)
{
  return ((a)[0] * (b)[0] + (a)[1] * (b)[1] + (a)[2] * (b)[2]);
}

inline double Inner14(const double* a, const double* b)
{
  return ((a)[0] * (b)[0] + (a)[1] * (b)[4] + (a)[2] * (b)[8]);
}

inline double Inner41(const double* a, const double* b)
{
  return ((a)[0] * (b)[0] + (a)[4] * (b)[1] + (a)[8] * (b)[2]);
}

inline double Inner44(const double* a, const double* b)
{
  return ((a)[0] * (b)[0] + (a)[4] * (b)[4] + (a)[8] * (b)[8]);
}

#define dMULTIPLYOP0_331(A, op, B, C)                                          \
  (A)[0] op Inner((B), (C));                                                   \
  (A)[1] op Inner((B + 4), (C));                                               \
  (A)[2] op Inner((B + 8), (C));

#define dMULTIPLYOP1_331(A, op, B, C)                                          \
  (A)[0] op Inner41((B), (C));                                                 \
  (A)[1] op Inner41((B + 1), (C));                                             \
  (A)[2] op Inner41((B + 2), (C));

inline void dMULTIPLY0_331(double* A, const double* B, const double* C)
{
  dMULTIPLYOP0_331(A, =, B, C)
}

inline void dMULTIPLY1_331(double* A, const double* B, const double* C)
{
  dMULTIPLYOP1_331(A, =, B, C)
}

#define dRecip(x) (1.0 / (x))

// given n points in the plane (array p, of size 2*n), generate m points that
// best represent the whole set. the definition of 'best' here is not
// predetermined - the idea is to select points that give good box-box
// collision detection behavior. the chosen point indexes are returned in the
// array iret (of size m). 'i0' is always the first entry in the array.
// n must be in the range [1..8]. m must be in the range [1..n]. i0 must be
// in the range [0..n-1].

void cullPoints(int n, double p[], int m, int i0, int iret[])
{
  // compute the centroid of the polygon in cx,cy
  int i, j;
  double a, cx, cy, q;
  if (n == 1)
  {
    cx = p[0];
    cy = p[1];
  }
  else if (n == 2)
  {
    cx = 0.5 * (p[0] + p[2]);
    cy = 0.5 * (p[1] + p[3]);
  }
  else
  {
    a = 0;
    cx = 0;
    cy = 0;
    for (i = 0; i < (n - 1); i++)
    {
      q = p[i * 2] * p[i * 2 + 3] - p[i * 2 + 2] * p[i * 2 + 1];
      a += q;
      cx += q * (p[i * 2] + p[i * 2 + 2]);
      cy += q * (p[i * 2 + 1] + p[i * 2 + 3]);
    }
    q = p[n * 2 - 2] * p[1] - p[0] * p[n * 2 - 1];
    a = dRecip(3.0 * (a + q));
    cx = a * (cx + q * (p[n * 2 - 2] + p[0]));
    cy = a * (cy + q * (p[n * 2 - 1] + p[1]));
  }

  // compute the angle of each point w.r.t. the centroid
  double A[8];
  for (i = 0; i < n; i++)
    A[i] = atan2(p[i * 2 + 1] - cy, p[i * 2] - cx);

  // search for points that have angles closest to A[i0] + i*(2*pi/m).
  int avail[8];
  for (i = 0; i < n; i++)
    avail[i] = 1;
  avail[i0] = 0;
  iret[0] = i0;
  iret++;
  for (j = 1; j < m; j++)
  {
    a = double(j) * (2 * math::constantsd::pi() / m) + A[i0];
    if (a > math::constantsd::pi())
      a -= 2 * math::constantsd::pi();
    double maxdiff = 1e9, diff;
    for (i = 0; i < n; i++)
    {
      if (avail[i])
      {
        diff = fabs(A[i] - a);
        if (diff > math::constantsd::pi())
          diff = 2 * math::constantsd::pi() - diff;
        if (diff < maxdiff)
        {
          maxdiff = diff;
          *iret = i;
        }
      }
    }
    avail[*iret] = 0;
    iret++;
  }
}

void dLineClosestApproach(
    const dVector3 pa,
    const dVector3 ua,
    const dVector3 pb,
    const dVector3 ub,
    double* alpha,
    double* beta)
{
  dVector3 p;
  p[0] = pb[0] - pa[0];
  p[1] = pb[1] - pa[1];
  p[2] = pb[2] - pa[2];
  double uaub = Inner(ua, ub);
  double q1 = Inner(ua, p);
  double q2 = -Inner(ub, p);
  double d = 1 - uaub * uaub;
  if (d <= 0)
  {
    // @@@ this needs to be made more robust
    *alpha = 0;
    *beta = 0;
  }
  else
  {
    d = dRecip(d);
    *alpha = (q1 + uaub * q2) * d;
    *beta = (uaub * q1 + q2) * d;
  }
}

// Adapted from Source: http://geomalgorithms.com/a07-_distance.html
void dSegmentsClosestApproach(
    const Eigen::Vector3d& pa,
    const Eigen::Vector3d& ua,
    const Eigen::Vector3d& pb,
    const Eigen::Vector3d& ub,
    double* alpha,
    double* beta)
{
  Eigen::Vector3d u = pb - pa;
  Eigen::Vector3d v = ub - ua;
  Eigen::Vector3d w = pa - ua;
  double a = u.dot(u); // always >= 0
  double b = u.dot(v);
  double c = v.dot(v); // always >= 0
  double d = u.dot(w);
  double e = v.dot(w);
  double D = a * c - b * b; // always >= 0
  double sc, sN, sD = D;    // sc = sN / sD, default sD = D >= 0
  double tc, tN, tD = D;    // tc = tN / tD, default tD = D >= 0

  const double SMALL_NUM = 1e-15;

  // compute the line parameters of the two closest points
  if (D < SMALL_NUM)
  {           // the lines are almost parallel
    sN = 0.0; // force using point P0 on segment S1
    sD = 1.0; // to prevent possible division by 0.0 later
    tN = e;
    tD = c;
  }
  else
  { // get the closest points on the infinite lines
    sN = (b * e - c * d);
    tN = (a * e - b * d);
    if (sN < 0.0)
    { // sc < 0 => the s=0 edge is visible
      sN = 0.0;
      tN = e;
      tD = c;
    }
    else if (sN > sD)
    { // sc > 1  => the s=1 edge is visible
      sN = sD;
      tN = e + b;
      tD = c;
    }
  }

  if (tN < 0.0)
  { // tc < 0 => the t=0 edge is visible
    tN = 0.0;
    // recompute sc for this edge
    if (-d < 0.0)
      sN = 0.0;
    else if (-d > a)
      sN = sD;
    else
    {
      sN = -d;
      sD = a;
    }
  }
  else if (tN > tD)
  { // tc > 1  => the t=1 edge is visible
    tN = tD;
    // recompute sc for this edge
    if ((-d + b) < 0.0)
      sN = 0;
    else if ((-d + b) > a)
      sN = sD;
    else
    {
      sN = (-d + b);
      sD = a;
    }
  }
  // finally do the division to get alpha and beta
  *alpha = (abs(sN) < SMALL_NUM ? 0.0 : sN / sD);
  *beta = (abs(tN) < SMALL_NUM ? 0.0 : tN / tD);
}

// Adapted from Source:
// http://geomalgorithms.com/a02-_lines.html#Distance-to-Ray-or-Segment
double dDistPointToSegment(
    const Eigen::Vector3d& p,
    const Eigen::Vector3d& ua,
    const Eigen::Vector3d& ub,
    double* alpha)
{
  Eigen::Vector3d v = ub - ua;
  Eigen::Vector3d w = p - ua;

  double c1 = w.dot(v);
  if (c1 <= 0)
  {
    *alpha = 0;
    return (p - ua).norm();
  }

  double c2 = v.dot(v);
  if (c2 <= c1)
  {
    *alpha = 1;
    return (p - ub).norm();
  }

  *alpha = c1 / c2;
  Eigen::Vector3d Pb = ua + *alpha * v;
  return (p - Pb).norm();
}

int intersectRectQuad(double h[2], double p[8], double ret[16])
{
  // q (and r) contain nq (and nr) coordinate points for the current (and
  // chopped) polygons
  int nq = 4, nr = 0;
  double buffer[16];
  double* q = p;
  double* r = ret;
  for (int dir = 0; dir <= 1; dir++)
  {
    // direction notation: xy[0] = x axis, xy[1] = y axis
    for (int sign = -1; sign <= 1; sign += 2)
    {
      // chop q along the line xy[dir] = sign*h[dir]
      double* pq = q;
      double* pr = r;
      nr = 0;
      for (int i = nq; i > 0; i--)
      {
        // go through all points in q and all lines between adjacent points
        if (sign * pq[dir] < h[dir])
        {
          // this point is inside the chopping line
          pr[0] = pq[0];
          pr[1] = pq[1];
          pr += 2;
          nr++;
          if (nr & 8)
          {
            q = r;
            goto done;
          }
        }
        double* nextq = (i > 1) ? pq + 2 : q;
        if ((sign * pq[dir] < h[dir]) ^ (sign * nextq[dir] < h[dir]))
        {
          // this line crosses the chopping line
          pr[1 - dir] = pq[1 - dir]
                        + (nextq[1 - dir] - pq[1 - dir])
                              / (nextq[dir] - pq[dir])
                              * (sign * h[dir] - pq[dir]);
          pr[dir] = sign * h[dir];
          pr += 2;
          nr++;
          if (nr & 8)
          {
            q = r;
            goto done;
          }
        }
        pq += 2;
      }
      q = r;
      r = (q == ret) ? buffer : ret;
      nq = nr;
    }
  }
done:
  if (q != ret)
    memcpy(ret, q, nr * 2 * sizeof(double));
  return nr;
}

// a simple root finding algorithm is used to find the value of 't' that
// satisfies:
//		d|D(t)|^2/dt = 0
// where:
//		|D(t)| = |p(t)-b(t)|
// where p(t) is a point on the line parameterized by t:
//		p(t) = p1 + t*(p2-p1)
// and b(t) is that same point clipped to the boundary of the box. in box-
// relative coordinates d|D(t)|^2/dt is the sum of three x,y,z components
// each of which looks like this:
//
//	    t_lo     /
//	      ______/    -->t
//	     /     t_hi
//	    /
//
// t_lo and t_hi are the t values where the line passes through the planes
// corresponding to the sides of the box. the algorithm computes d|D(t)|^2/dt
// in a piecewise fashion from t=0 to t=1, stopping at the point where
// d|D(t)|^2/dt crosses from negative to positive.

void dClosestLineBoxPoints(
    const dVector3 p1,
    const dVector3 p2,
    const dVector3 c,
    const dMatrix3 R,
    const dVector3 side,
    dVector3 lret,
    dVector3 bret)
{
  int i;

  // compute the start and delta of the line p1-p2 relative to the box.
  // we will do all subsequent computations in this box-relative coordinate
  // system. we have to do a translation and rotation for each point.
  dVector3 tmp, s, v;
  tmp[0] = p1[0] - c[0];
  tmp[1] = p1[1] - c[1];
  tmp[2] = p1[2] - c[2];
  dMULTIPLY1_331(s, R, tmp);
  tmp[0] = p2[0] - p1[0];
  tmp[1] = p2[1] - p1[1];
  tmp[2] = p2[2] - p1[2];
  dMULTIPLY1_331(v, R, tmp);

  // mirror the line so that v has all components >= 0
  dVector3 sign;
  for (i = 0; i < 3; i++)
  {
    if (v[i] < 0)
    {
      s[i] = -s[i];
      v[i] = -v[i];
      sign[i] = -1;
    }
    else
      sign[i] = 1;
  }

  // compute v^2
  dVector3 v2;
  v2[0] = v[0] * v[0];
  v2[1] = v[1] * v[1];
  v2[2] = v[2] * v[2];

  // compute the half-sides of the box
  double h[3];
  h[0] = side[0];
  h[1] = side[1];
  h[2] = side[2];

  // region is -1,0,+1 depending on which side of the box planes each
  // coordinate is on. tanchor is the next t value at which there is a
  // transition, or the last one if there are no more.
  int region[3];
  double tanchor[3];

  // find the region and tanchor values for p1
  for (i = 0; i < 3; i++)
  {
    if (v[i] > 0)
    {
      if (s[i] < -h[i])
      {
        region[i] = -1;
        tanchor[i] = (-h[i] - s[i]) / v[i];
      }
      else
      {
        region[i] = (s[i] > h[i]);
        tanchor[i] = (h[i] - s[i]) / v[i];
      }
    }
    else
    {
      region[i] = 0;
      tanchor[i] = 2; // this will never be a valid tanchor
    }
  }

  // compute d|d|^2/dt for t=0. if it's >= 0 then p1 is the closest point
  double t = 0;
  double dd2dt = 0;
  for (i = 0; i < 3; i++)
    dd2dt -= (region[i] ? v2[i] : 0) * tanchor[i];
  if (dd2dt >= 0)
    goto got_answer;

  do
  {
    // find the point on the line that is at the next clip plane boundary
    double next_t = 1;
    for (i = 0; i < 3; i++)
    {
      if (tanchor[i] > t && tanchor[i] < 1 && tanchor[i] < next_t)
        next_t = tanchor[i];
    }

    // compute d|d|^2/dt for the next t
    double next_dd2dt = 0;
    for (i = 0; i < 3; i++)
    {
      next_dd2dt += (region[i] ? v2[i] : 0) * (next_t - tanchor[i]);
    }

    // if the sign of d|d|^2/dt has changed, solution = the crossover point
    if (next_dd2dt >= 0)
    {
      double m = (next_dd2dt - dd2dt) / (next_t - t);
      t -= dd2dt / m;
      goto got_answer;
    }

    // advance to the next anchor point / region
    for (i = 0; i < 3; i++)
    {
      if (tanchor[i] == next_t)
      {
        tanchor[i] = (h[i] - s[i]) / v[i];
        region[i]++;
      }
    }
    t = next_t;
    dd2dt = next_dd2dt;
  } while (t < 1);
  t = 1;

got_answer:

  // compute closest point on the line
  for (i = 0; i < 3; i++)
    lret[i] = p1[i] + t * tmp[i]; // note: tmp=p2-p1

  // compute closest point on the box
  for (i = 0; i < 3; i++)
  {
    tmp[i] = sign[i] * (s[i] + t * v[i]);
    if (tmp[i] < -h[i])
      tmp[i] = -h[i];
    else if (tmp[i] > h[i])
      tmp[i] = h[i];
  }
  dMULTIPLY0_331(s, R, tmp);
  for (i = 0; i < 3; i++)
    bret[i] = s[i] + c[i];
}

// given two boxes (p1,R1,side1) and (p2,R2,side2), collide them together and
// generate contact points. this returns 0 if there is no contact otherwise
// it returns the number of contacts generated.
// `normal' returns the contact normal.
// `depth' returns the maximum penetration depth along that normal.
// `return_code' returns a number indicating the type of contact that was
// detected:
//        1,2,3 = box 2 intersects with a face of box 1
//        4,5,6 = box 1 intersects with a face of box 2
//        7..15 = edge-edge contact
// `maxc' is the maximum number of contacts allowed to be generated, i.e.
// the size of the `contact' array.
// `contact' and `skip' are the contact array information provided to the
// collision functions. this function only fills in the position and depth
// fields.
int dBoxBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const dVector3 p1,
    const dMatrix3 R1,
    const dVector3 side1,
    const dVector3 p2,
    const dMatrix3 R2,
    const dVector3 side2,
    CollisionResult& result)
{
  const double fudge_factor = 1.05;
  dVector3 p, pp, normalC = {0.0, 0.0, 0.0, 0.0};
  const double* normalR = 0;
  double A[3], B[3], R11, R12, R13, R21, R22, R23, R31, R32, R33, Q11, Q12, Q13,
      Q21, Q22, Q23, Q31, Q32, Q33, s, s2, l;
  int i, j, invert_normal, code;

  // get vector from centers of box 1 to box 2, relative to box 1
  p[0] = p2[0] - p1[0];
  p[1] = p2[1] - p1[1];
  p[2] = p2[2] - p1[2];
  dMULTIPLY1_331(pp, R1, p); // get pp = p relative to body 1

  // get side lengths / 2
  A[0] = side1[0];
  A[1] = side1[1];
  A[2] = side1[2];
  B[0] = side2[0];
  B[1] = side2[1];
  B[2] = side2[2];

  // Rij is R1'*R2, i.e. the relative rotation between R1 and R2
  R11 = Inner44(R1 + 0, R2 + 0);
  R12 = Inner44(R1 + 0, R2 + 1);
  R13 = Inner44(R1 + 0, R2 + 2);
  R21 = Inner44(R1 + 1, R2 + 0);
  R22 = Inner44(R1 + 1, R2 + 1);
  R23 = Inner44(R1 + 1, R2 + 2);
  R31 = Inner44(R1 + 2, R2 + 0);
  R32 = Inner44(R1 + 2, R2 + 1);
  R33 = Inner44(R1 + 2, R2 + 2);

  Q11 = std::abs(R11);
  Q12 = std::abs(R12);
  Q13 = std::abs(R13);
  Q21 = std::abs(R21);
  Q22 = std::abs(R22);
  Q23 = std::abs(R23);
  Q31 = std::abs(R31);
  Q32 = std::abs(R32);
  Q33 = std::abs(R33);

  // for all 15 possible separating axes:
  //   * see if the axis separates the boxes. if so, return 0.
  //   * find the depth of the penetration along the separating axis (s2)
  //   * if this is the largest depth so far, record it.
  // the normal vector will be set to the separating axis with the smallest
  // depth. note: normalR is set to point to a column of R1 or R2 if that is
  // the smallest depth normal so far. otherwise normalR is 0 and normalC is
  // set to a vector relative to body 1. invert_normal is 1 if the sign of
  // the normal should be flipped.

#define TST(expr1, expr2, norm, cc)                                            \
  s2 = std::abs(expr1) - (expr2);                                              \
  if (s2 > s)                                                                  \
  {                                                                            \
    s = s2;                                                                    \
    normalR = norm;                                                            \
    invert_normal = ((expr1) < 0);                                             \
    code = (cc);                                                               \
  }

  s = -1E12;
  invert_normal = 0;
  code = 0;

  // separating axis = u1,u2,u3
  TST(pp[0], (A[0] + B[0] * Q11 + B[1] * Q12 + B[2] * Q13), R1 + 0, 1);
  TST(pp[1], (A[1] + B[0] * Q21 + B[1] * Q22 + B[2] * Q23), R1 + 1, 2);
  TST(pp[2], (A[2] + B[0] * Q31 + B[1] * Q32 + B[2] * Q33), R1 + 2, 3);

  // separating axis = v1,v2,v3
  TST(Inner41(R2 + 0, p),
      (A[0] * Q11 + A[1] * Q21 + A[2] * Q31 + B[0]),
      R2 + 0,
      4);
  TST(Inner41(R2 + 1, p),
      (A[0] * Q12 + A[1] * Q22 + A[2] * Q32 + B[1]),
      R2 + 1,
      5);
  TST(Inner41(R2 + 2, p),
      (A[0] * Q13 + A[1] * Q23 + A[2] * Q33 + B[2]),
      R2 + 2,
      6);

  // note: cross product axes need to be scaled when s is computed.
  // normal (n1,n2,n3) is relative to box 1.
#undef TST
#define TST(expr1, expr2, n1, n2, n3, cc)                                      \
  s2 = std::abs(expr1) - (expr2);                                              \
  l = sqrt((n1) * (n1) + (n2) * (n2) + (n3) * (n3));                           \
  if (l > 0)                                                                   \
  {                                                                            \
    s2 /= l;                                                                   \
    if (s2 * fudge_factor > s)                                                 \
    {                                                                          \
      s = s2;                                                                  \
      normalR = 0;                                                             \
      normalC[0] = (n1) / l;                                                   \
      normalC[1] = (n2) / l;                                                   \
      normalC[2] = (n3) / l;                                                   \
      invert_normal = ((expr1) < 0);                                           \
      code = (cc);                                                             \
    }                                                                          \
  }

  // separating axis = u1 x (v1,v2,v3)
  TST(pp[2] * R21 - pp[1] * R31,
      (A[1] * Q31 + A[2] * Q21 + B[1] * Q13 + B[2] * Q12),
      0,
      -R31,
      R21,
      7);
  TST(pp[2] * R22 - pp[1] * R32,
      (A[1] * Q32 + A[2] * Q22 + B[0] * Q13 + B[2] * Q11),
      0,
      -R32,
      R22,
      8);
  TST(pp[2] * R23 - pp[1] * R33,
      (A[1] * Q33 + A[2] * Q23 + B[0] * Q12 + B[1] * Q11),
      0,
      -R33,
      R23,
      9);

  // separating axis = u2 x (v1,v2,v3)
  TST(pp[0] * R31 - pp[2] * R11,
      (A[0] * Q31 + A[2] * Q11 + B[1] * Q23 + B[2] * Q22),
      R31,
      0,
      -R11,
      10);
  TST(pp[0] * R32 - pp[2] * R12,
      (A[0] * Q32 + A[2] * Q12 + B[0] * Q23 + B[2] * Q21),
      R32,
      0,
      -R12,
      11);
  TST(pp[0] * R33 - pp[2] * R13,
      (A[0] * Q33 + A[2] * Q13 + B[0] * Q22 + B[1] * Q21),
      R33,
      0,
      -R13,
      12);

  // separating axis = u3 x (v1,v2,v3)
  TST(pp[1] * R11 - pp[0] * R21,
      (A[0] * Q21 + A[1] * Q11 + B[1] * Q33 + B[2] * Q32),
      -R21,
      R11,
      0,
      13);
  TST(pp[1] * R12 - pp[0] * R22,
      (A[0] * Q22 + A[1] * Q12 + B[0] * Q33 + B[2] * Q31),
      -R22,
      R12,
      0,
      14);
  TST(pp[1] * R13 - pp[0] * R23,
      (A[0] * Q23 + A[1] * Q13 + B[0] * Q32 + B[1] * Q31),
      -R23,
      R13,
      0,
      15);

#undef TST

  if (!code)
    return 0;
  if (s > 0.0)
    return 0;

  // if we get to this point, the boxes interpenetrate. compute the normal
  // in global coordinates.

  Eigen::Vector3d normal;
  Eigen::Vector3d point_vec;
  double penetration;

  if (normalR)
  {
    normal << normalR[0], normalR[4], normalR[8];
  }
  else
  {
    normal << Inner((R1), (normalC)), Inner((R1 + 4), (normalC)),
        Inner((R1 + 8), (normalC));
    // dMULTIPLY0_331 (normal,R1,normalC);
  }
  if (invert_normal)
  {
    normal *= -1.0;
  }

  // compute contact point(s)

  // single point
  if (code > 6)
  {
    // an edge from box 1 touches an edge from box 2.
    // find a point pa on the intersecting edge of box 1
    dVector3 pa;
    double sign;
    for (i = 0; i < 3; i++)
      pa[i] = p1[i];
    for (j = 0; j < 3; j++)
    {
#define TEMP_INNER14(a, b) (a[0] * (b)[0] + a[1] * (b)[4] + a[2] * (b)[8])
      sign = (TEMP_INNER14(normal, R1 + j) > 0) ? 1.0 : -1.0;

      // sign = (Inner14(normal,R1+j) > 0) ? 1.0 : -1.0;

      for (i = 0; i < 3; i++)
        pa[i] += sign * A[j] * R1[i * 4 + j];
    }

    // find a point pb on the intersecting edge of box 2
    dVector3 pb;
    for (i = 0; i < 3; i++)
      pb[i] = p2[i];
    for (j = 0; j < 3; j++)
    {
      sign = (TEMP_INNER14(normal, R2 + j) > 0) ? -1.0 : 1.0;
#undef TEMP_INNER14
      for (i = 0; i < 3; i++)
        pb[i] += sign * B[j] * R2[i * 4 + j];
    }

    double alpha, beta;
    dVector3 ua, ub;
    for (i = 0; i < 3; i++)
      ua[i] = R1[((code)-7) / 3 + i * 4];
    for (i = 0; i < 3; i++)
      ub[i] = R2[((code)-7) % 3 + i * 4];

    dLineClosestApproach(pa, ua, pb, ub, &alpha, &beta);
    Eigen::Vector3d edgeAFixedPoint = Eigen::Vector3d(pa[0], pa[1], pa[2]);
    Eigen::Vector3d edgeBFixedPoint = Eigen::Vector3d(pb[0], pb[1], pb[2]);

    // After this, pa and pb represent the closest point
    for (i = 0; i < 3; i++)
      pa[i] += ua[i] * alpha;
    for (i = 0; i < 3; i++)
      pb[i] += ub[i] * beta;

    {
      // This is the average of the closest point on the A edge and the B edge
      point_vec << 0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]),
          0.5 * (pa[2] + pb[2]);
      penetration = -s;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = point_vec;
      contact.normal = normal;
      contact.penetrationDepth = penetration;
      contact.type = ContactType::EDGE_EDGE;
      contact.edgeAClosestPoint = Eigen::Vector3d(pa[0], pa[1], pa[2]);
      contact.edgeAFixedPoint = edgeAFixedPoint;
      contact.edgeADir = Eigen::Vector3d(ua[0], ua[1], ua[2]);
      contact.edgeBClosestPoint = Eigen::Vector3d(pb[0], pb[1], pb[2]);
      contact.edgeBFixedPoint = edgeBFixedPoint;
      contact.edgeBDir = Eigen::Vector3d(ub[0], ub[1], ub[2]);
      result.addContact(contact);
    }
    return 1;
  }

  // okay, we have a face-something intersection (because the separating
  // axis is perpendicular to a face). define face 'a' to be the reference
  // face (i.e. the normal vector is perpendicular to this) and face 'b' to be
  // the incident face (the closest face of the other box).

  const double *Ra, *Rb, *pa, *pb, *Sa, *Sb;
  ContactType type;
  if (code <= 3)
  {
    Ra = R1;
    Rb = R2;
    pa = p1;
    pb = p2;
    Sa = A;
    Sb = B;
    type = ContactType::VERTEX_FACE;
  }
  else
  {
    Ra = R2;
    Rb = R1;
    pa = p2;
    pb = p1;
    Sa = B;
    Sb = A;
    type = ContactType::FACE_VERTEX;
  }

  // nr = normal vector of reference face dotted with axes of incident box.
  // anr = absolute values of nr.
  dVector3 normal2, nr, anr;
  if (code <= 3)
  {
    normal2[0] = normal[0];
    normal2[1] = normal[1];
    normal2[2] = normal[2];
  }
  else
  {
    normal2[0] = -normal[0];
    normal2[1] = -normal[1];
    normal2[2] = -normal[2];
  }
  dMULTIPLY1_331(nr, Rb, normal2);
  anr[0] = fabs(nr[0]);
  anr[1] = fabs(nr[1]);
  anr[2] = fabs(nr[2]);

  // find the largest compontent of anr: this corresponds to the normal
  // for the indident face. the other axis numbers of the indicent face
  // are stored in a1,a2.
  int lanr, a1, a2;
  if (anr[1] > anr[0])
  {
    if (anr[1] > anr[2])
    {
      a1 = 0;
      lanr = 1;
      a2 = 2;
    }
    else
    {
      a1 = 0;
      a2 = 1;
      lanr = 2;
    }
  }
  else
  {
    if (anr[0] > anr[2])
    {
      lanr = 0;
      a1 = 1;
      a2 = 2;
    }
    else
    {
      a1 = 0;
      a2 = 1;
      lanr = 2;
    }
  }

  // compute center point of incident face, in reference-face coordinates
  dVector3 center;
  if (nr[lanr] < 0)
  {
    for (i = 0; i < 3; i++)
      center[i] = pb[i] - pa[i] + Sb[lanr] * Rb[i * 4 + lanr];
  }
  else
  {
    for (i = 0; i < 3; i++)
      center[i] = pb[i] - pa[i] - Sb[lanr] * Rb[i * 4 + lanr];
  }

  // find the normal and non-normal axis numbers of the reference box
  int codeN, code1, code2;
  if (code <= 3)
    codeN = code - 1;
  else
    codeN = code - 4;
  if (codeN == 0)
  {
    code1 = 1;
    code2 = 2;
  }
  else if (codeN == 1)
  {
    code1 = 0;
    code2 = 2;
  }
  else
  {
    code1 = 0;
    code2 = 1;
  }

  // find the four corners of the incident face, in reference-face coordinates
  double quad[8]; // 2D coordinate of incident face (x,y pairs)
  double c1, c2, m11, m12, m21, m22;
  c1 = Inner14(center, Ra + code1);
  c2 = Inner14(center, Ra + code2);
  // optimize this? - we have already computed this data above, but it is not
  // stored in an easy-to-index format. for now it's quicker just to recompute
  // the four dot products.
  m11 = Inner44(Ra + code1, Rb + a1);
  m12 = Inner44(Ra + code1, Rb + a2);
  m21 = Inner44(Ra + code2, Rb + a1);
  m22 = Inner44(Ra + code2, Rb + a2);
  {
    double k1 = m11 * Sb[a1];
    double k2 = m21 * Sb[a1];
    double k3 = m12 * Sb[a2];
    double k4 = m22 * Sb[a2];
    quad[0] = c1 - k1 - k3;
    quad[1] = c2 - k2 - k4;
    quad[2] = c1 - k1 + k3;
    quad[3] = c2 - k2 + k4;
    quad[4] = c1 + k1 + k3;
    quad[5] = c2 + k2 + k4;
    quad[6] = c1 + k1 - k3;
    quad[7] = c2 + k2 - k4;
  }

  // find the size of the reference face
  double rect[2];
  rect[0] = Sa[code1];
  rect[1] = Sa[code2];

  // intersect the incident and reference faces
  double ret[16];
  int n = intersectRectQuad(rect, quad, ret);
  if (n < 1)
    return 0; // this should never happen

  // convert the intersection points into reference-face coordinates,
  // and compute the contact position and depth for each point. only keep
  // those points that have a positive (penetrating) depth. delete points in
  // the 'ret' array as necessary so that 'point' and 'ret' correspond.
  // real point[3*8];		// penetrating contact points
  double point[24]; // penetrating contact points
  double dep[8];    // depths for those points
  double det1 = dRecip(m11 * m22 - m12 * m21);
  m11 *= det1;
  m12 *= det1;
  m21 *= det1;
  m22 *= det1;
  int cnum = 0; // number of penetrating contact points found
  for (j = 0; j < n; j++)
  {
    double k1 = m22 * (ret[j * 2] - c1) - m12 * (ret[j * 2 + 1] - c2);
    double k2 = -m21 * (ret[j * 2] - c1) + m11 * (ret[j * 2 + 1] - c2);
    for (i = 0; i < 3; i++)
    {
      point[cnum * 3 + i]
          = center[i] + k1 * Rb[i * 4 + a1] + k2 * Rb[i * 4 + a2];
    }
    dep[cnum] = Sa[codeN] - Inner(normal2, point + cnum * 3);
    if (dep[cnum] >= 0)
    {
      ret[cnum * 2] = ret[j * 2];
      ret[cnum * 2 + 1] = ret[j * 2 + 1];
      cnum++;
    }
  }
  if (cnum < 1)
    return 0; // this should never happen

  // we can't generate more contacts than we actually have
  int maxc = 4;
  if (maxc > cnum)
    maxc = cnum;
  // if (maxc < 1) maxc = 1;

  if (cnum <= maxc)
  {
    // we have less contacts than we need, so we use them all
    for (j = 0; j < cnum; j++)
    {
      point_vec << point[j * 3 + 0] + pa[0], point[j * 3 + 1] + pa[1],
          point[j * 3 + 2] + pa[2];

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = point_vec;
      contact.normal = normal;
      contact.penetrationDepth = dep[j];
      contact.type = type;
      result.addContact(contact);
    }
  }
  else
  {
    // we have more contacts than are wanted, some of them must be culled.
    // find the deepest point, it is always the first contact.
    int i1 = 0;
    double maxdepth = dep[0];
    for (i = 1; i < cnum; i++)
    {
      if (dep[i] > maxdepth)
      {
        maxdepth = dep[i];
        i1 = i;
      }
    }

    int iret[8];
    cullPoints(cnum, ret, maxc, i1, iret);

    cnum = maxc;
    for (j = 0; j < cnum; j++)
    {
      point_vec << point[iret[j] * 3 + 0] + pa[0],
          point[iret[j] * 3 + 1] + pa[1], point[iret[j] * 3 + 2] + pa[2];

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = point_vec;
      contact.normal = normal;
      contact.penetrationDepth = dep[iret[j]];
      contact.type = type;
      result.addContact(contact);
    }
  }
  return cnum;
}

int collideBoxBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  dVector3 halfSize0;
  dVector3 halfSize1;

  convVector(0.5 * size0, halfSize0);
  convVector(0.5 * size1, halfSize1);

  dMatrix3 R0, R1;

  convMatrix(T0, R0);
  convMatrix(T1, R1);

  dVector3 p0;
  dVector3 p1;

  convVector(T0.translation(), p0);
  convVector(T1.translation(), p1);

  return dBoxBox(o1, o2, p1, R1, halfSize1, p0, R0, halfSize0, result);
}

int collideBoxSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    const double& r1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  Eigen::Vector3d halfSize = 0.5 * size0;
  bool inside_box = true;

  // clipping a center of the sphere to a boundary of the box
  // Vec3 c0(&T0[9]);
  Eigen::Vector3d c0 = T1.translation();
  Eigen::Vector3d p = T0.inverse() * c0;

  Contact contact;
  contact.sphereCenter = c0;
  contact.collisionObject1 = o1;
  contact.collisionObject2 = o2;
  contact.type = BOX_SPHERE;

  if (p[0] < -halfSize[0])
  {
    contact.face1Normal = T0.linear().col(0);
    contact.face1Locked = true;
    p[0] = -halfSize[0];
    inside_box = false;
  }
  if (p[0] > halfSize[0])
  {
    contact.face1Normal = T0.linear().col(0);
    contact.face1Locked = true;
    p[0] = halfSize[0];
    inside_box = false;
  }

  if (p[1] < -halfSize[1])
  {
    contact.face2Normal = T0.linear().col(1);
    contact.face2Locked = true;
    p[1] = -halfSize[1];
    inside_box = false;
  }
  if (p[1] > halfSize[1])
  {
    contact.face2Normal = T0.linear().col(1);
    contact.face2Locked = true;
    p[1] = halfSize[1];
    inside_box = false;
  }

  if (p[2] < -halfSize[2])
  {
    contact.face3Normal = T0.linear().col(2);
    contact.face3Locked = true;
    p[2] = -halfSize[2];
    inside_box = false;
  }
  if (p[2] > halfSize[2])
  {
    contact.face3Normal = T0.linear().col(2);
    contact.face3Locked = true;
    p[2] = halfSize[2];
    inside_box = false;
  }

  Eigen::Vector3d normal(0.0, 0.0, 0.0);
  double penetration;

  if (inside_box)
  {
    // find nearest side from the sphere center
    double min = halfSize[0] - std::abs(p[0]);
    double tmin = halfSize[1] - std::abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - std::abs(p[2]);
    if (tmin < min)
    {
      min = tmin;
      idx = 2;
    }

    // normal[idx] = (p[idx] > 0.0 ? 1.0 : -1.0);
    normal[idx] = (p[idx] > 0.0 ? -1.0 : 1.0);
    normal = T0.linear() * normal;
    penetration = min + r1;

    // In this special case, it actually behaves as though it's just a raw
    // vertex-face collision for gradients, so don't reinvent the wheel
    contact.type = FACE_VERTEX;
    contact.point = c0;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
    return 1;
  }

  Eigen::Vector3d contactpt = T0 * p;
  // normal = c0 - contactpt;
  normal = contactpt - c0;
  double mag = normal.norm();
  penetration = r1 - mag;

  if (penetration < 0.0)
  {
    return 0;
  }

  if (mag > DART_COLLISION_EPS)
  {
    normal *= (1.0 / mag);

    contact.point = contactpt;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
  }
  else
  {
    double min = halfSize[0] - std::abs(p[0]);
    double tmin = halfSize[1] - std::abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - std::abs(p[2]);
    if (tmin < min)
    {
      min = tmin;
      idx = 2;
    }
    normal.setZero();
    // normal[idx] = (p[idx] > 0.0 ? 1.0 : -1.0);
    normal[idx] = (p[idx] > 0.0 ? -1.0 : 1.0);
    normal = T0.linear() * normal;

    contact.point = contactpt;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
  }
  return 1;
}

int collideSphereBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& r0,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  Eigen::Vector3d halfSize = 0.5 * size1;
  bool inside_box = true;

  // clipping a center of the sphere to a boundary of the box
  Eigen::Vector3d c0 = T0.translation();
  Eigen::Vector3d p = T1.inverse() * c0;

  Contact contact;
  contact.sphereCenter = c0;
  contact.collisionObject1 = o1;
  contact.collisionObject2 = o2;
  contact.type = SPHERE_BOX;

  if (p[0] < -halfSize[0])
  {
    contact.face1Normal = T1.linear().col(0);
    contact.face1Locked = true;
    p[0] = -halfSize[0];
    inside_box = false;
  }
  if (p[0] > halfSize[0])
  {
    contact.face1Normal = T1.linear().col(0);
    contact.face1Locked = true;
    p[0] = halfSize[0];
    inside_box = false;
  }

  if (p[1] < -halfSize[1])
  {
    contact.face2Normal = T1.linear().col(1);
    contact.face2Locked = true;
    p[1] = -halfSize[1];
    inside_box = false;
  }
  if (p[1] > halfSize[1])
  {
    contact.face2Normal = T1.linear().col(1);
    contact.face2Locked = true;
    p[1] = halfSize[1];
    inside_box = false;
  }

  if (p[2] < -halfSize[2])
  {
    contact.face3Normal = T1.linear().col(2);
    contact.face3Locked = true;
    p[2] = -halfSize[2];
    inside_box = false;
  }
  if (p[2] > halfSize[2])
  {
    contact.face3Normal = T1.linear().col(2);
    contact.face3Locked = true;
    p[2] = halfSize[2];
    inside_box = false;
  }

  Eigen::Vector3d normal(0.0, 0.0, 0.0);
  double penetration;

  if (inside_box)
  {
    // find nearest side from the sphere center
    double min = halfSize[0] - std::abs(p[0]);
    double tmin = halfSize[1] - std::abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - std::abs(p[2]);
    if (tmin < min)
    {
      min = tmin;
      idx = 2;
    }

    normal[idx] = (p[idx] > 0.0 ? 1.0 : -1.0);
    normal = T1.linear() * normal;
    penetration = min + r0;

    // In this special case, it actually behaves as though it's just a raw
    // vertex-face collision for gradients, so don't reinvent the wheel
    contact.type = VERTEX_FACE;
    contact.point = c0;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
    return 1;
  }

  Eigen::Vector3d contactpt = T1 * p;
  normal = c0 - contactpt;
  double mag = normal.norm();
  penetration = r0 - mag;

  if (penetration < 0.0)
  {
    return 0;
  }

  if (mag > DART_COLLISION_EPS)
  {
    normal *= (1.0 / mag);

    contact.point = contactpt;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
  }
  else
  {
    double min = halfSize[0] - std::abs(p[0]);
    double tmin = halfSize[1] - std::abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - std::abs(p[2]);
    if (tmin < min)
    {
      min = tmin;
      idx = 2;
    }
    normal.setZero();
    normal[idx] = (p[idx] > 0.0 ? 1.0 : -1.0);
    normal = T1.linear() * normal;

    contact.point = contactpt;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
  }
  return 1;
}

int collideSphereSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& _r0,
    const Eigen::Isometry3d& c0,
    const double& _r1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result)
{
  double r0 = _r0;
  double r1 = _r1;
  double rsum = r0 + r1;
  Eigen::Vector3d normal = c0.translation() - c1.translation();
  double normal_sqr = normal.squaredNorm();

  if (normal_sqr > rsum * rsum)
  {
    return 0;
  }

  r0 /= rsum;
  r1 /= rsum;

  Eigen::Vector3d point = r1 * c0.translation() + r0 * c1.translation();
  double penetration;

  if (normal_sqr < DART_COLLISION_EPS)
  {
    normal.setZero();
    penetration = rsum;

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = point;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    contact.type = SPHERE_SPHERE;
    contact.centerA = c0.translation();
    contact.radiusA = r0 * rsum;
    contact.centerB = c1.translation();
    contact.radiusB = r1 * rsum;
    result.addContact(contact);
    return 1;
  }

  normal_sqr = sqrt(normal_sqr);
  normal *= (1.0 / normal_sqr);
  penetration = rsum - normal_sqr;

  Contact contact;
  contact.type = SPHERE_SPHERE;
  contact.centerA = c0.translation();
  contact.radiusA = r0 * rsum;
  contact.centerB = c1.translation();
  contact.radiusB = r1 * rsum;
  contact.collisionObject1 = o1;
  contact.collisionObject2 = o2;
  contact.point = point;
  contact.normal = normal;
  contact.penetrationDepth = penetration;
  result.addContact(contact);
  return 1;
}

/// libccd support function for a box
void ccdSupportBox(const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out)
{
  // assume that obj_t is user-defined structure that holds info about
  // object (in this case box: x, y, z, pos, quat - dimensions of box,
  // position and rotation)
  ccdBox* box = (ccdBox*)_obj;
  Eigen::Map<const Eigen::Vector3d> dir(_dir->v);
  Eigen::Map<Eigen::Vector3d> out(_out->v);

  // apply rotation on direction vector
  Eigen::Vector3d localDir = box->transform->linear().transpose() * dir;

  // compute support point in specified direction (in box coordinates)
  Eigen::Vector3d clipped = Eigen::Vector3d(
      ccdSign(localDir(0)) * (*box->size)(0) * CCD_REAL(0.5),
      ccdSign(localDir(1)) * (*box->size)(1) * CCD_REAL(0.5),
      ccdSign(localDir(2)) * (*box->size)(2) * CCD_REAL(0.5));

  // transform support point according to position and rotation of object
  out = *(box->transform) * clipped;
}

/// libccd support function for a sphere
void ccdSupportSphere(
    const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out)
{
  ccdSphere* sphere = (ccdSphere*)_obj;

  Eigen::Map<const Eigen::Vector3d> dir(_dir->v);
  Eigen::Map<Eigen::Vector3d> out(_out->v);

  // apply rotation on direction vector
  Eigen::Vector3d localDir = sphere->transform->linear().transpose() * dir;
  localDir *= sphere->radius / localDir.norm();

  // transform support point according to position and rotation of object
  out = *(sphere->transform) * localDir;
}

/// libccd support function for a mesh
void ccdSupportMesh(const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out)
{
  ccdMesh* mesh = (ccdMesh*)_obj;

  Eigen::Map<const Eigen::Vector3d> dir(_dir->v);
  Eigen::Map<Eigen::Vector3d> out(_out->v);

  // apply rotation on direction vector
  Eigen::Vector3d localDir = mesh->transform->linear().transpose() * dir;
  localDir(0) /= (*mesh->scale)(0);
  localDir(1) /= (*mesh->scale)(1);
  localDir(2) /= (*mesh->scale)(2);

  double maxDot = -std::numeric_limits<double>::infinity();
  Eigen::Vector3d maxDotPoint = Eigen::Vector3d::Zero();

  for (int i = 0; i < mesh->mesh->mNumMeshes; i++)
  {
    aiMesh* m = mesh->mesh->mMeshes[i];
    for (int k = 0; k < m->mNumVertices; k++)
    {
      double dot = m->mVertices[k].x * localDir(0)
                   + m->mVertices[k].y * localDir(1)
                   + m->mVertices[k].z * localDir(2);
      if (dot > maxDot)
      {
        maxDot = dot;
        maxDotPoint(0) = m->mVertices[k].x;
        maxDotPoint(1) = m->mVertices[k].y;
        maxDotPoint(2) = m->mVertices[k].z;
      }
    }
  }

  maxDotPoint(0) *= (*mesh->scale)(0);
  maxDotPoint(1) *= (*mesh->scale)(1);
  maxDotPoint(2) *= (*mesh->scale)(2);

  // transform support point according to position and rotation of object
  out = *(mesh->transform) * maxDotPoint;
}

/// libccd support function for a capsule
void ccdSupportCapsule(
    const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out)
{
  ccdCapsule* capsule = (ccdCapsule*)_obj;

  Eigen::Map<const Eigen::Vector3d> dir(_dir->v);
  Eigen::Map<Eigen::Vector3d> out(_out->v);

  // apply rotation on direction vector
  Eigen::Vector3d localDir = capsule->transform->linear().transpose() * dir;
  localDir.normalize();
  localDir *= capsule->radius;

  if (std::abs(localDir(2)) < 1e-10)
  {
    out = *(capsule->transform) * localDir;
  }
  else if (localDir(2) > 0)
  {
    out = *(capsule->transform)
          * (localDir + Eigen::Vector3d(0, 0, capsule->height / 2));
  }
  else if (localDir(2) < 0)
  {
    out = *(capsule->transform)
          * (localDir + Eigen::Vector3d(0, 0, -capsule->height / 2));
  }
  else
  {
    assert(false && "This should be impossible to read");
  }
}

/// libccd support function for a box
void ccdCenterBox(const void* _obj, ccd_vec3_t* _center)
{
  ccdBox* box = (ccdBox*)_obj;
  Eigen::Map<Eigen::Vector3d> center(_center->v);
  center = box->transform->translation();
}

/// libccd support function for a sphere
void ccdCenterSphere(const void* _obj, ccd_vec3_t* _center)
{
  ccdSphere* sphere = (ccdSphere*)_obj;
  Eigen::Map<Eigen::Vector3d> center(_center->v);
  center = sphere->transform->translation();
}

/// libccd support function for a mesh
void ccdCenterMesh(const void* _obj, ccd_vec3_t* _center)
{
  ccdMesh* mesh = (ccdMesh*)_obj;
  Eigen::Map<Eigen::Vector3d> center(_center->v);
  center = mesh->transform->translation();
}

/// libccd support function for a capsule
void ccdCenterCapsule(const void* _obj, ccd_vec3_t* _center)
{
  ccdCapsule* capsule = (ccdCapsule*)_obj;
  Eigen::Map<Eigen::Vector3d> center(_center->v);
  center = capsule->transform->translation();
}

/// Find all the vertices within epsilon of lying on the witness plane
std::vector<Eigen::Vector3d> ccdPointsAtWitnessBox(
    ccdBox* box, ccd_vec3_t* _dir, bool neg)
{
  Eigen::Map<const Eigen::Vector3d> dir(_dir->v);

  // apply rotation on direction vector
  Eigen::Vector3d localDir = box->transform->linear().transpose() * dir;

  std::vector<Eigen::Vector3d> localPoints;

  std::vector<double> boundsX;
  std::vector<double> boundsY;
  std::vector<double> boundsZ;

  boundsX.push_back((*box->size)(0) * 0.5);
  boundsX.push_back((*box->size)(0) * -0.5);
  boundsY.push_back((*box->size)(1) * 0.5);
  boundsY.push_back((*box->size)(1) * -0.5);
  boundsZ.push_back((*box->size)(2) * 0.5);
  boundsZ.push_back((*box->size)(2) * -0.5);

  for (double x : boundsX)
  {
    for (double y : boundsY)
    {
      for (double z : boundsZ)
      {
        localPoints.push_back(Eigen::Vector3d(x, y, z));
      }
    }
  }

  double negMult = neg ? -1 : 1;

  double maxDot = -1 * std::numeric_limits<double>::infinity();
  for (Eigen::Vector3d& localPoint : localPoints)
  {
    double dot = negMult * localPoint.dot(localDir);
    if (dot > maxDot)
      maxDot = dot;
  }

  std::vector<Eigen::Vector3d> points;
  for (Eigen::Vector3d& localPoint : localPoints)
  {
    double dot = negMult * localPoint.dot(localDir);
    if (maxDot - dot < DART_COLLISION_WITNESS_PLANE_DEPTH)
    {
      points.push_back((*box->transform) * localPoint);
    }
  }

  return points;
}

/// Find all the vertices within epsilon of lying on the witness plane
std::vector<Eigen::Vector3d> ccdPointsAtWitnessMesh(
    ccdMesh* mesh, ccd_vec3_t* _dir, bool neg)
{
  Eigen::Map<const Eigen::Vector3d> dir(_dir->v);

  // apply rotation on direction vector
  Eigen::Vector3d localDir = mesh->transform->linear().transpose() * dir;
  localDir(0) /= (*mesh->scale)(0);
  localDir(1) /= (*mesh->scale)(1);
  localDir(2) /= (*mesh->scale)(2);

  std::vector<Eigen::Vector3d> points;

  double maxDot = (neg ? 1 : -1) * std::numeric_limits<double>::infinity();

  // 1. Find the max dot
  for (int i = 0; i < mesh->mesh->mNumMeshes; i++)
  {
    aiMesh* m = mesh->mesh->mMeshes[i];
    for (int k = 0; k < m->mNumVertices; k++)
    {
      double dot = m->mVertices[k].x * localDir(0) * (*mesh->scale)(0)
                       * (*mesh->scale)(0)
                   + m->mVertices[k].y * localDir(1) * (*mesh->scale)(1)
                         * (*mesh->scale)(1)
                   + m->mVertices[k].z * localDir(2) * (*mesh->scale)(2)
                         * (*mesh->scale)(2);
      if (((dot > maxDot) && !neg) || ((dot < maxDot) && neg))
      {
        maxDot = dot;
      }
    }
  }

  // 2. Use the max dot to find vertices at the contact plane
  for (int i = 0; i < mesh->mesh->mNumMeshes; i++)
  {
    aiMesh* m = mesh->mesh->mMeshes[i];
    for (int k = 0; k < m->mNumVertices; k++)
    {
      double dot = m->mVertices[k].x * localDir(0) * (*mesh->scale)(0)
                       * (*mesh->scale)(0)
                   + m->mVertices[k].y * localDir(1) * (*mesh->scale)(1)
                         * (*mesh->scale)(1)
                   + m->mVertices[k].z * localDir(2) * (*mesh->scale)(2)
                         * (*mesh->scale)(2);
      // If we're on the witness plane with our "maxDot" vector, then add us to
      // the list
      if (std::abs(dot - maxDot) < DART_COLLISION_WITNESS_PLANE_DEPTH)
      {
        Eigen::Vector3d proposedPoint
            = *(mesh->transform)
              * Eigen::Vector3d(
                  m->mVertices[k].x * (*mesh->scale)(0),
                  m->mVertices[k].y * (*mesh->scale)(1),
                  m->mVertices[k].z * (*mesh->scale)(2));
        // It's possible for meshes to contain duplicate vertices, so filter
        // those out
        bool foundDuplicate = false;
        for (int i = 0; i < points.size(); i++)
        {
          if ((points[i] - proposedPoint).squaredNorm() < 1e-6)
          {
            foundDuplicate = true;
            break;
          }
        }
        if (!foundDuplicate)
        {
          points.push_back(proposedPoint);
        }
      }
    }
  }

  return points;
}

/// This is a helper for creating contacts between a pair of faces, or face-edge
/// or edge-face pairs. This shows up several times in mesh-mesh collisions, as
/// well as capsule-mesh collisions, so is factored out as its own method.
void createFaceFaceContacts(
    std::vector<Contact>& collisionsOut,
    CollisionObject* o1,
    CollisionObject* o2,
    ccd_vec3_t* dir,
    const std::vector<Eigen::Vector3d>& pointsAWitnessSorted,
    const std::vector<Eigen::Vector3d>& pointsBWitnessSorted)
{
  assert(pointsAWitnessSorted.size() > 2 || pointsBWitnessSorted.size() > 2);
  assert(pointsAWitnessSorted.size() >= 2 && pointsBWitnessSorted.size() >= 2);

  Eigen::Map<Eigen::Vector3d> dirVec(dir->v);

  // All the pointsAWitness vectors are co-planar, so we choose the closest
  // [0], [1], and [2] to cross to get a precise normal
  Eigen::Vector3d normalA
      = (pointsAWitnessSorted[0] - pointsAWitnessSorted[1])
            .cross(
                pointsAWitnessSorted[1]
                - (pointsAWitnessSorted.size() > 2 ? pointsAWitnessSorted[2]
                                                   : dirVec))
            .normalized();
  // Likewise for the pointsBWitness vectors
  Eigen::Vector3d normalB
      = (pointsBWitnessSorted[0] - pointsBWitnessSorted[1])
            .cross(
                pointsBWitnessSorted[1]
                - (pointsBWitnessSorted.size() > 2 ? pointsBWitnessSorted[2]
                                                   : dirVec))
            .normalized();

  // If the norm of a given normal is 0, then the points were colinear. If the
  // normal direction is too far off the original direction, that's also sus,
  // likely numerical issues from having points super close together.
  bool aBroken = abs(normalA.squaredNorm() - 1) > 1e-10
                 || std::min(
                        (normalA - dirVec).squaredNorm(),
                        (-normalA - dirVec).squaredNorm())
                        > 1e-5;
  bool bBroken = abs(normalB.squaredNorm() - 1) > 1e-10
                 || std::min(
                        (normalB - dirVec).squaredNorm(),
                        (-normalB - dirVec).squaredNorm())
                        > 1e-5;
  if (aBroken && !bBroken)
  {
    normalA = normalB;
  }
  else if (!aBroken && bBroken)
  {
    normalB = normalA;
  }
  else if (aBroken && bBroken)
  {
    // Default to dir, if both faces are colinear or broken for other reasons
    normalA = -Eigen::Vector3d(dir->v[0], dir->v[1], dir->v[2]);
    normalB = normalA;
  }

  // Ensure that the normal is in the opposite direction as `dir`, so we're
  // still pointing from B to A.
  double normalADot = normalA(0) * dir->v[0] + normalA(1) * dir->v[1]
                      + normalA(2) * dir->v[2];
  if (normalADot > 0)
    normalA *= -1;
  double normalBDot = normalB(0) * dir->v[0] + normalB(1) * dir->v[1]
                      + normalB(2) * dir->v[2];
  if (normalBDot > 0)
    normalB *= -1;

  Eigen::Vector3d normal = ((normalA + normalB) / 2).normalized();

  // This will the origin for our 2D plane we're going to use to compute
  // collision geometry. We use different origins for object A and B, because
  // they're slightly offset from each other in space due to penetration
  // distance.
  Eigen::Vector3d originA = normal * (pointsAWitnessSorted[0].dot(normal));
  Eigen::Vector3d originB = normal * (pointsBWitnessSorted[0].dot(normal));

  Eigen::Vector3d origin = (originA + originB) / 2;

  Eigen::Vector3d tmp = normal.cross(Eigen::Vector3d::UnitZ());
  if (tmp.squaredNorm() < 1e-4)
  {
    tmp = normal.cross(Eigen::Vector3d::UnitX());
  }

  // These are the basis for our 2D plane (with an origin at `origin2d` in 3D
  // space)
  Eigen::Vector3d basis2dX = normalA.cross(tmp);
  Eigen::Vector3d basis2dY = normalA.cross(basis2dX);

  std::vector<Eigen::Vector3d> pointsAConvex = pointsAWitnessSorted;
  prepareConvex2DShape(pointsAConvex, origin, basis2dX, basis2dY);
  std::vector<Eigen::Vector3d> pointsBConvex = pointsBWitnessSorted;
  prepareConvex2DShape(pointsBConvex, origin, basis2dX, basis2dY);

  int numContacts = 0;

  // All vertices that lie inside the other shape's convex hull are
  // vertex-face contacts.

  // Start with points from object A. We'll later do the symmetric thing for
  // object B.
  for (int i = 0; i < pointsAConvex.size(); i++)
  {
    Eigen::Vector3d vertexA = pointsAConvex[i];
    if (convex2DShapeContains(
            vertexA, pointsBConvex, origin, basis2dX, basis2dY))
    {
      numContacts++;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = vertexA;

      // Make sure that the normal vector is pointed from B to A, which should
      // mean that distA > distB because A's vertex is penetrating B
      double distA = vertexA.dot(normalB);
      double distB = pointsBWitnessSorted[0].dot(normalB);
      contact.normal = normalB;
      contact.penetrationDepth = distB - distA;

      contact.type = VERTEX_FACE;

      collisionsOut.push_back(contact);
    }
  }

  // Now we need to repeat analagous logic for the vertices in shape B
  for (int i = 0; i < pointsBConvex.size(); i++)
  {
    Eigen::Vector3d vertexB = pointsBConvex[i];
    if (convex2DShapeContains(
            vertexB, pointsAConvex, origin, basis2dX, basis2dY))
    {
      numContacts++;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = vertexB;

      // Make sure that the normal vector is pointed from B to A, which should
      // mean that distB > distA because B's vertex is penetrating A
      double distA = pointsAWitnessSorted[0].dot(normalA);
      double distB = vertexB.dot(normalA);
      contact.normal = normalA;
      contact.penetrationDepth = distB - distA;

      contact.type = FACE_VERTEX;

      collisionsOut.push_back(contact);
    }
  }

  // Now finally we check every pair of edges in shape A and shape B for
  // collisions.
  for (int i = 0; i < pointsAConvex.size(); i++)
  {
    // Skip the last vertex loop-around if this is just a single edge, since we
    // don't want to count the same edge twice
    if (i == pointsAConvex.size() - 1 && pointsAConvex.size() == 2)
    {
      continue;
    }
    Eigen::Vector3d a1World = pointsAConvex[i];
    Eigen::Vector3d a2World
        = pointsAConvex[i == pointsAConvex.size() - 1 ? 0 : i + 1];
    Eigen::Vector2d a1 = pointInPlane(a1World, origin, basis2dX, basis2dY);
    Eigen::Vector2d a2 = pointInPlane(a2World, origin, basis2dX, basis2dY);
    for (int j = 0; j < pointsBConvex.size(); j++)
    {
      // Skip the last vertex loop-around if this is just a single edge, since
      // we don't want to count the same edge twice
      if (j == pointsBConvex.size() - 1 && pointsBConvex.size() == 2)
      {
        continue;
      }
      Eigen::Vector3d b1World = pointsBConvex[j];
      Eigen::Vector3d b2World
          = pointsBConvex[j == pointsBConvex.size() - 1 ? 0 : j + 1];
      Eigen::Vector2d b1 = pointInPlane(b1World, origin, basis2dX, basis2dY);
      Eigen::Vector2d b2 = pointInPlane(b2World, origin, basis2dX, basis2dY);

      Eigen::Vector2d out;
      if (get2DLineIntersection(a1, a2, b1, b2, out))
      {
        // We found an edge-edge collision at "out"!
        numContacts++;

        // Get the relevant points in 3D space
        Eigen::Vector3d edgeAClosestPoint
            = originA + out(0) * basis2dX + out(1) * basis2dY;
        Eigen::Vector3d edgeBClosestPoint
            = originB + out(0) * basis2dX + out(1) * basis2dY;

        Contact contact;
        contact.collisionObject1 = o1;
        contact.collisionObject2 = o2;
        contact.point = (edgeAClosestPoint + edgeBClosestPoint) / 2;
        contact.type = EDGE_EDGE;
        contact.edgeAClosestPoint = edgeAClosestPoint;
        contact.edgeAFixedPoint = edgeAClosestPoint;
        contact.edgeADir = a2World - a1World;
        contact.edgeBClosestPoint = edgeBClosestPoint;
        contact.edgeBFixedPoint = edgeBClosestPoint;
        contact.edgeBDir = b2World - b1World;
        // Arbitrarily tie break normal, cause we're not using either face
        // precisely
        double distA = contact.edgeAClosestPoint.dot(normalA);
        double distB = contact.edgeBClosestPoint.dot(normalA);
        contact.normal = normalA;
        contact.penetrationDepth = distB - distA;

        collisionsOut.push_back(contact);
      }
    }
  }
}

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createMeshMeshContacts(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const std::vector<Eigen::Vector3d>& pointsAWitness,
    const std::vector<Eigen::Vector3d>& pointsBWitness)
{
  if (pointsAWitness.size() == 0 && pointsBWitness.size() == 0)
  {
    std::cout
        << "Attempting to create a mesh-mesh contact with no witness points!"
        << std::endl;
  }
  assert(pointsAWitness.size() > 0 && pointsBWitness.size() > 0);

  std::vector<Eigen::Vector3d> pointsAWitnessSorted = pointsAWitness;
  std::vector<Eigen::Vector3d> pointsBWitnessSorted = pointsBWitness;

  // `dir` points from A to B, so we want the highest dot product of dir at the
  // front of A
  std::sort(
      pointsAWitnessSorted.begin(),
      pointsAWitnessSorted.end(),
      [dir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
        double aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
        double bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
        return aDot < bDot;
      });
  // `dir` points from A to B, so we want the lowest dot product of dir at the
  // front of B
  std::sort(
      pointsBWitnessSorted.begin(),
      pointsBWitnessSorted.end(),
      [dir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
        double aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
        double bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
        return aDot > bDot;
      });

  // Single vertex-face collision
  if (pointsAWitness.size() == 1 && pointsBWitness.size() > 2)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = pointsAWitness[0];

    // All the pointsBWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3d normal
        = (pointsBWitnessSorted[0] - pointsBWitnessSorted[1])
              .cross(pointsBWitnessSorted[1] - pointsBWitnessSorted[2])
              .normalized();

    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    double normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;
    contact.normal = normal;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distA > distB because A's vertex is penetrating B
    double distA = pointsAWitness[0].dot(normal);
    double distB = pointsBWitnessSorted[0].dot(normal);
    // Normal is fine as it is
    contact.penetrationDepth = abs(distA - distB);

    contact.type = VERTEX_FACE;
    result.addContact(contact);

    return 1;
  }
  // Single face-vertex collision
  else if (pointsAWitness.size() > 2 && pointsBWitness.size() == 1)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = pointsBWitness[0];

    // All the pointsAWitness vectors are co-planar, so we choose the closest
    // [0], [1], and [2] to cross to get a precise normal
    Eigen::Vector3d normal
        = (pointsAWitnessSorted[0] - pointsAWitnessSorted[1])
              .cross(pointsAWitnessSorted[1] - pointsAWitnessSorted[2])
              .normalized();

    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    double normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;
    contact.normal = normal;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distB > distA because B's vertex is penetrating A
    double distA = pointsAWitnessSorted[0].dot(normal);
    double distB = pointsBWitness[0].dot(normal);
    // Normal is fine as it is
    contact.penetrationDepth = abs(distA - distB);

    contact.type = FACE_VERTEX;
    result.addContact(contact);

    return 1;
  }
  // Single edge-edge collision
  else if (pointsAWitness.size() == 2 && pointsBWitness.size() == 2)
  {
    Eigen::Vector3d ua = (pointsAWitness[0] - pointsAWitness[1]).normalized();
    Eigen::Vector3d ub = (pointsBWitness[0] - pointsBWitness[1]).normalized();
    Eigen::Vector3d pa = pointsAWitness[0];
    Eigen::Vector3d pb = pointsBWitness[0];

    double alpha, beta;
    dLineClosestApproach(
        pa.data(), ua.data(), pb.data(), ub.data(), &alpha, &beta);

    // After this, pa and pb represent the closest point
    for (int i = 0; i < 3; i++)
      pa[i] += ua[i] * alpha;
    for (int i = 0; i < 3; i++)
      pb[i] += ub[i] * beta;

    {
      // This is the average of the closest point on the A edge and the B edge
      Eigen::Vector3d point = Eigen::Vector3d(
          0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2]));

      Eigen::Vector3d normal = ua.cross(ub);
      double normalDot = normal(0) * dir->v[0] + normal(1) * dir->v[1]
                         + normal(2) * dir->v[2];
      if (normalDot > 0)
        normal *= -1;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = point;
      contact.type = ContactType::EDGE_EDGE;
      contact.edgeAClosestPoint = Eigen::Vector3d(pa[0], pa[1], pa[2]);
      contact.edgeAFixedPoint = pa;
      contact.edgeADir = Eigen::Vector3d(ua[0], ua[1], ua[2]);
      contact.edgeBClosestPoint = Eigen::Vector3d(pb[0], pb[1], pb[2]);
      contact.edgeBFixedPoint = pb;
      contact.edgeBDir = Eigen::Vector3d(ub[0], ub[1], ub[2]);
      contact.normal = normal;

      double distA = contact.edgeAClosestPoint.dot(normal);
      double distB = contact.edgeBClosestPoint.dot(normal);
      contact.penetrationDepth = abs(distB - distA);

      result.addContact(contact);
    }

    return 1;
  }
  // Edge-face collision, results in two collisions
  else if (pointsAWitness.size() == 2 && pointsBWitness.size() > 2)
  {
    std::vector<Contact> contacts;
    createFaceFaceContacts(
        contacts, o1, o2, dir, pointsAWitnessSorted, pointsBWitnessSorted);
    for (Contact& contact : contacts)
    {
      result.addContact(contact);
    }
    assert(contacts.size() == 2);
    return contacts.size();
  }
  // Face-edge collision, results in two collisions
  else if (pointsAWitness.size() > 2 && pointsBWitness.size() == 2)
  {
    std::vector<Contact> contacts;
    createFaceFaceContacts(
        contacts, o1, o2, dir, pointsAWitnessSorted, pointsBWitnessSorted);
    for (Contact& contact : contacts)
    {
      result.addContact(contact);
    }
    assert(contacts.size() == 2);
    return contacts.size();
  }
  // Single vertex-edge collision, awkward special case. Pretend it's
  // VERTEX_FACE, but don't compute the exact normal analytically (because we
  // don't have enough vertices to do so)
  else if (pointsAWitness.size() == 1 && pointsBWitness.size() == 2)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = pointsAWitness[0];

    Eigen::Vector3d normal = Eigen::Vector3d(dir->v[0], dir->v[1], dir->v[2]);
    // Ensure the normal is orthogonal to edge B, at least
    Eigen::Vector3d edgeB
        = (pointsBWitness[0] - pointsBWitness[1]).normalized();
    normal -= normal.dot(edgeB) * edgeB;
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    double normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distA > distB because A's vertex is penetrating B
    double distA = pointsAWitness[0].dot(normal);
    double distB = pointsBWitness[0].dot(normal);
    contact.normal = normal;
    contact.penetrationDepth = abs(distA - distB);

    contact.type = VERTEX_FACE;
    result.addContact(contact);

    return 1;
  }
  // Single edge-vertex collision, awkward special case. Pretend it's
  // FACE_VERTEX, but don't compute the exact normal analytically (because we
  // don't have enough vertices to do so)
  else if (pointsAWitness.size() == 2 && pointsBWitness.size() == 1)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = pointsBWitness[0];

    Eigen::Vector3d normal = Eigen::Vector3d(dir->v[0], dir->v[1], dir->v[2]);
    // Ensure the normal is orthogonal to edge A, at least
    Eigen::Vector3d edgeA
        = (pointsAWitness[1] - pointsAWitness[1]).normalized();
    normal -= normal.dot(edgeA) * edgeA;
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    double normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distB > distA because B's vertex is penetrating A
    double distA = pointsAWitness[0].dot(normal);
    double distB = pointsBWitness[0].dot(normal);
    contact.normal = normal;
    contact.penetrationDepth = abs(distA - distB);

    contact.type = FACE_VERTEX;
    result.addContact(contact);

    return 1;
  }
  // A vertex-vertex collision. Totally weird special case, but technically
  // possible. Pretend it's FACE_VERTEX, but don't compute the exact normal
  // analytically (because we don't have enough vertices to do so). Arbitrarily
  // choose point B to be the face, cause why not?
  else if (pointsAWitness.size() == 1 && pointsBWitness.size() == 1)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = pointsBWitness[0];

    // point in the opposite direction of dir to get from A -> B
    Eigen::Vector3d normal
        = Eigen::Vector3d(dir->v[0], dir->v[1], dir->v[2]) * -1;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distB > distA because B's vertex is penetrating A
    double distA = pointsAWitness[0].dot(normal);
    double distB = pointsBWitness[0].dot(normal);
    contact.normal = normal;
    contact.penetrationDepth = abs(distB - distA);

    contact.type = FACE_VERTEX;
    result.addContact(contact);

    return 1;
  }
  // This will be multiple collisions. This is definitely not simple. Really, we
  // want to know which convex face subsumes the other, so that the smaller face
  // can be handled as the vertices, and the larger face can be the face of a
  // vertex-face collision.
  //
  // Because both shapes are convex, any linear slice of them must also be
  // convex. We can exploit this, and the fact that all the witness points from
  // A and B are all co-planar, so we can do 2D logic to find the vertices from
  // A contained in B, the vertices from B contained in A, and the edge-edge
  // collisions where the lines on the edges of the convex shapes cross.
  else
  {
    assert(pointsAWitness.size() > 2 && pointsBWitness.size() > 2);

    std::vector<Contact> contacts;
    createFaceFaceContacts(
        contacts, o1, o2, dir, pointsAWitnessSorted, pointsBWitnessSorted);
    for (Contact& contact : contacts)
    {
      result.addContact(contact);
    }
    return contacts.size();
  }
  // We should never reach here
  assert(false);
  return 0;
}

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createMeshSphereContact(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const std::vector<Eigen::Vector3d>& meshPointsWitness,
    const Eigen::Vector3d& sphereCenter,
    double sphereRadius)
{
  if (meshPointsWitness.size() == 0)
  {
    std::cout
        << "Attempting to create a mesh-sphere contact with no witness points!"
        << std::endl;
  }
  assert(meshPointsWitness.size() > 0);
  // vertex-sphere collision
  if (meshPointsWitness.size() == 1)
  {
    // normal is (vertex) -> (sphere center)
    Eigen::Vector3d normal = (meshPointsWitness[0] - sphereCenter).normalized();
    Eigen::Vector3d contactPoint = sphereCenter + (sphereRadius * normal);
    Contact contact;
    contact.point = contactPoint;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.penetrationDepth
        = sphereRadius - (meshPointsWitness[0] - sphereCenter).norm();
    contact.type = VERTEX_SPHERE;
    result.addContact(contact);
  }
  // edge-sphere collision
  else if (meshPointsWitness.size() == 2)
  {
    // Find nearest point on the edge to sphere center
    // normal is (nearest point) -> (sphere center)
    Eigen::Vector3d edge = meshPointsWitness[1] - meshPointsWitness[0];
    double offset = edge.dot(meshPointsWitness[0]);
    double relativeSphere = edge.dot(sphereCenter) - offset;
    Eigen::Vector3d closestPoint = meshPointsWitness[0] + relativeSphere * edge;
    Eigen::Vector3d normal = (closestPoint - sphereCenter).normalized();
    Eigen::Vector3d contactPoint = sphereCenter + (sphereRadius * normal);
    Contact contact;
    contact.point = contactPoint;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.type = EDGE_SPHERE;
    contact.edgeAClosestPoint = closestPoint;
    contact.edgeADir = edge;
    contact.penetrationDepth
        = sphereRadius - (closestPoint - sphereCenter).norm();
    result.addContact(contact);
  }
  // face-sphere collision
  else if (meshPointsWitness.size() > 2)
  {
    std::vector<Eigen::Vector3d> pointsWitnessSorted = meshPointsWitness;
    // `dir` points from A to B, so we want the highest dot product of dir at
    // the front of A
    std::sort(
        pointsWitnessSorted.begin(),
        pointsWitnessSorted.end(),
        [dir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
          double aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
          double bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
          return aDot < bDot;
        });
    // All the meshPointsWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3d normal
        = (pointsWitnessSorted[0] - pointsWitnessSorted[1])
              .cross(pointsWitnessSorted[1] - pointsWitnessSorted[2])
              .normalized();
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    double normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // We want to find the furthest point on the surface of the sphere, which is
    // just whatever point is facing the normal

    Eigen::Vector3d point = sphereCenter + normal * sphereRadius;

    Contact contact;
    contact.point = point;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.penetrationDepth
        = sphereRadius
          - (normal.dot(pointsWitnessSorted[0]) - normal.dot(sphereCenter));
    contact.type = FACE_SPHERE;
    result.addContact(contact);
  }
  // We always create exactly one contact, if we don't error
  return 1;
}

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createSphereMeshContact(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const Eigen::Vector3d& sphereCenter,
    double sphereRadius,
    const std::vector<Eigen::Vector3d>& meshPointsWitness)
{
  if (meshPointsWitness.size() == 0)
  {
    std::cout
        << "Attempting to create a mesh-sphere contact with no witness points!"
        << std::endl;
  }
  assert(meshPointsWitness.size() > 0);
  // vertex-sphere collision
  if (meshPointsWitness.size() == 1)
  {
    Eigen::Vector3d normal = (sphereCenter - meshPointsWitness[0]).normalized();
    Eigen::Vector3d contactPoint = sphereCenter - (sphereRadius * normal);
    Contact contact;
    contact.point = contactPoint;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.penetrationDepth
        = sphereRadius - (meshPointsWitness[0] - sphereCenter).norm();
    contact.type = SPHERE_VERTEX;
    result.addContact(contact);
  }
  // edge-sphere collision
  else if (meshPointsWitness.size() == 2)
  {
    // Find nearest point on the edge to sphere center
    // normal is (nearest point) -> (sphere center)
    Eigen::Vector3d edge = meshPointsWitness[1] - meshPointsWitness[0];
    double offset = edge.dot(meshPointsWitness[0]);
    double relativeSphere = edge.dot(sphereCenter) - offset;
    Eigen::Vector3d closestPoint = meshPointsWitness[0] + relativeSphere * edge;
    Eigen::Vector3d normal = (closestPoint - sphereCenter).normalized();
    Eigen::Vector3d contactPoint = sphereCenter + (sphereRadius * normal);
    Contact contact;
    contact.point = contactPoint;
    contact.normal = normal * -1;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.type = SPHERE_EDGE;
    contact.edgeAClosestPoint = closestPoint;
    contact.edgeADir = edge;
    contact.penetrationDepth
        = sphereRadius - (closestPoint - sphereCenter).norm();
    result.addContact(contact);
  }
  // face-sphere collision
  else if (meshPointsWitness.size() > 2)
  {
    std::vector<Eigen::Vector3d> pointsWitnessSorted = meshPointsWitness;
    // `dir` points from A to B, so we want the highest dot product of dir at
    // the front of A
    std::sort(
        pointsWitnessSorted.begin(),
        pointsWitnessSorted.end(),
        [dir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
          double aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
          double bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
          return aDot < bDot;
        });
    // All the meshPointsWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3d normal
        = (pointsWitnessSorted[0] - pointsWitnessSorted[1])
              .cross(pointsWitnessSorted[1] - pointsWitnessSorted[2])
              .normalized();
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    double normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // We want to find the furthest point on the surface of the sphere, which is
    // just whatever point is facing the normal

    Eigen::Vector3d point = sphereCenter - normal * sphereRadius;

    Contact contact;
    contact.point = point;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.type = SPHERE_FACE;
    contact.penetrationDepth
        = sphereRadius
          - (normal.dot(sphereCenter) - normal.dot(pointsWitnessSorted[0]));
    result.addContact(contact);
  }
  // We always create exactly one contact, if we don't error
  return 1;
}

/// This is responsible for creating and annotating all the contact objects with
/// all the metadata we need in order to get accurate gradients.
int createCapsuleMeshContact(
    CollisionObject* o1,
    CollisionObject* o2,
    CollisionResult& result,
    ccd_vec3_t* dir,
    const Eigen::Vector3d& capsuleA,
    const Eigen::Vector3d& capsuleB,
    double capsuleRadius,
    const std::vector<Eigen::Vector3d>& meshPointsWitness,
    bool flipObjectOrder)
{
  if (meshPointsWitness.size() == 0)
  {
    std::cout
        << "Attempting to create a mesh-sphere contact with no witness points!"
        << std::endl;
  }
  assert(meshPointsWitness.size() > 0);

  // vertex-pipe collision
  if (meshPointsWitness.size() == 1)
  {
    double alpha;
    dDistPointToSegment(meshPointsWitness[0], capsuleA, capsuleB, &alpha);
    Eigen::Vector3d nearestPoint = capsuleA + (capsuleB - capsuleA) * alpha;

    Eigen::Vector3d normal = (nearestPoint - meshPointsWitness[0]).normalized();
    Eigen::Vector3d contactPoint = meshPointsWitness[0];

    Contact contact;
    contact.point = contactPoint;
    if (flipObjectOrder)
    {
      contact.collisionObject1 = o2;
      contact.collisionObject2 = o1;
      contact.normal = normal * -1;
      contact.edgeBClosestPoint = nearestPoint;
      contact.edgeBDir = (capsuleB - capsuleA);
      contact.type = VERTEX_PIPE;
    }
    else
    {
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.normal = normal;
      contact.edgeAClosestPoint = nearestPoint;
      contact.edgeADir = (capsuleB - capsuleA);
      contact.type = PIPE_VERTEX;
    }
    contact.penetrationDepth
        = capsuleRadius - (meshPointsWitness[0] - nearestPoint).norm();
    result.addContact(contact);

    return 1;
  }
  // edge-pipe collision
  else if (meshPointsWitness.size() == 2)
  {
    Eigen::Vector3d pipeDir = (capsuleB - capsuleA).normalized();
    Eigen::Vector3d edgeDir
        = (meshPointsWitness[1] - meshPointsWitness[0]).normalized();
    bool parallel = std::abs(1.0 - std::abs(pipeDir.dot(edgeDir))) < 1e-5;

    // Special case, if the edge is parallel to the pipe. This generates two
    // contacts, and has several different edge cases.
    if (parallel)
    {
      double edgeALinear = edgeDir.dot(meshPointsWitness[0]);
      double edgeBLinear = edgeDir.dot(meshPointsWitness[1]);
      double capsuleALinear = edgeDir.dot(capsuleA);
      double capsuleBLinear = edgeDir.dot(capsuleB);

      double edgeMin = edgeALinear < edgeBLinear ? edgeALinear : edgeBLinear;
      Eigen::Vector3d edgeMinPoint = edgeALinear < edgeBLinear
                                         ? meshPointsWitness[0]
                                         : meshPointsWitness[1];
      double edgeMax = edgeALinear < edgeBLinear ? edgeBLinear : edgeALinear;
      Eigen::Vector3d edgeMaxPoint = edgeALinear < edgeBLinear
                                         ? meshPointsWitness[1]
                                         : meshPointsWitness[0];
      double capsuleMin
          = capsuleALinear < capsuleBLinear ? capsuleALinear : capsuleBLinear;
      Eigen::Vector3d capsuleMinPoint
          = capsuleALinear < capsuleBLinear ? capsuleA : capsuleB;
      double capsuleMax
          = capsuleALinear < capsuleBLinear ? capsuleBLinear : capsuleALinear;
      Eigen::Vector3d capsuleMaxPoint
          = capsuleALinear < capsuleBLinear ? capsuleB : capsuleA;

      Eigen::Vector3d normal = capsuleA - meshPointsWitness[0];
      normal -= edgeDir * normal.dot(edgeDir);
      double dist = normal.norm();
      normal.normalize();

      // This means our min-end contact point is the edgeMin, resulting in a
      // VERTEX-EDGE collision
      if (capsuleMin < edgeMin)
      {
        Contact contact;
        contact.point = edgeMinPoint;
        if (flipObjectOrder)
        {
          contact.collisionObject1 = o2;
          contact.collisionObject2 = o1;
          contact.normal = normal * -1;
          contact.type = VERTEX_PIPE;
          contact.edgeBClosestPoint = edgeMinPoint + normal * capsuleRadius;
          contact.edgeBDir = (capsuleB - capsuleA);
          contact.radiusB = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = PIPE_VERTEX;
          contact.edgeAClosestPoint = edgeMinPoint + normal * capsuleRadius;
          contact.edgeADir = (capsuleB - capsuleA);
          contact.radiusA = capsuleRadius;
        }
        contact.penetrationDepth = capsuleRadius - dist;
        result.addContact(contact);
      }
      // This means our min-edge contact is the capsuleMin, resulting in a
      // SPHERE-EDGE collision
      else
      {
        Contact contact;
        contact.point = capsuleMinPoint - normal * capsuleRadius;
        if (flipObjectOrder)
        {
          contact.collisionObject1 = o2;
          contact.collisionObject2 = o1;
          contact.normal = normal * -1;
          contact.type = EDGE_SPHERE;
          contact.edgeAClosestPoint = capsuleMinPoint - normal * capsuleRadius;
          contact.edgeADir = edgeDir;
          contact.radiusB = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = SPHERE_EDGE;
          contact.radiusA = capsuleRadius;
          contact.edgeBClosestPoint = capsuleMinPoint - normal * capsuleRadius;
          contact.edgeBDir = edgeDir;
        }
        contact.penetrationDepth = capsuleRadius - dist;
        result.addContact(contact);
      }

      // This means our max-end contact point is the edgeMax, resulting in a
      // VERTEX-EDGE collision
      if (capsuleMax > edgeMax)
      {
        Contact contact;
        contact.point = edgeMaxPoint;
        if (flipObjectOrder)
        {
          contact.collisionObject1 = o2;
          contact.collisionObject2 = o1;
          contact.normal = normal * -1;
          contact.type = VERTEX_PIPE;
          contact.edgeBClosestPoint = edgeMaxPoint + normal * capsuleRadius;
          contact.edgeBDir = (capsuleB - capsuleA);
          contact.radiusB = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = PIPE_VERTEX;
          contact.edgeAClosestPoint = edgeMaxPoint + normal * capsuleRadius;
          contact.edgeADir = (capsuleB - capsuleA);
          contact.radiusA = capsuleRadius;
        }
        contact.penetrationDepth = capsuleRadius - dist;
        result.addContact(contact);
      }
      // This means our min-edge contact is the capsuleMin, resulting in a
      // SPHERE-EDGE collision
      else
      {
        Contact contact;
        contact.point = capsuleMaxPoint - normal * capsuleRadius;
        if (flipObjectOrder)
        {
          contact.collisionObject1 = o2;
          contact.collisionObject2 = o1;
          contact.normal = normal * -1;
          contact.type = EDGE_SPHERE;
          contact.edgeAClosestPoint = capsuleMaxPoint - normal * capsuleRadius;
          contact.edgeADir = edgeDir;
          contact.radiusB = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = SPHERE_EDGE;
          contact.radiusA = capsuleRadius;
          contact.edgeBClosestPoint = capsuleMaxPoint - normal * capsuleRadius;
          contact.edgeBDir = edgeDir;
        }
        contact.penetrationDepth = capsuleRadius - dist;
        result.addContact(contact);
      }

      return 2;
    }
    // Standard case, edge and capsule are not parallel
    // Find nearest point on both edges and generate a single contact
    else
    {
      double alpha;
      double beta;
      dSegmentsClosestApproach(
          meshPointsWitness[0],
          capsuleA,
          meshPointsWitness[1],
          capsuleB,
          &alpha,
          &beta);

      Eigen::Vector3d edgeClosestPoint
          = meshPointsWitness[0]
            + alpha * (meshPointsWitness[1] - meshPointsWitness[0]);
      Eigen::Vector3d pipeClosestPoint
          = capsuleA + beta * (capsuleB - capsuleA);
      Eigen::Vector3d normal
          = (edgeClosestPoint - pipeClosestPoint).normalized();
      Eigen::Vector3d contactPoint = edgeClosestPoint;

      Contact contact;
      contact.point = contactPoint;
      if (flipObjectOrder)
      {
        contact.collisionObject1 = o2;
        contact.collisionObject2 = o1;
        contact.normal = normal;
        contact.type = EDGE_PIPE;
        contact.edgeAClosestPoint = edgeClosestPoint;
        contact.edgeADir = (meshPointsWitness[1] - meshPointsWitness[0]);
        contact.edgeBClosestPoint = pipeClosestPoint;
        contact.edgeBDir = (capsuleB - capsuleA);
        contact.radiusB = capsuleRadius;
      }
      else
      {
        contact.collisionObject1 = o1;
        contact.collisionObject2 = o2;
        contact.normal = normal * -1;
        contact.type = PIPE_EDGE;
        contact.edgeAClosestPoint = pipeClosestPoint;
        contact.edgeADir = (capsuleB - capsuleA);
        contact.radiusA = capsuleRadius;
        contact.edgeBClosestPoint = edgeClosestPoint;
        contact.edgeBDir = (meshPointsWitness[1] - meshPointsWitness[0]);
      }
      contact.penetrationDepth
          = capsuleRadius - (edgeClosestPoint - pipeClosestPoint).norm();
      result.addContact(contact);

      return 1;
    }
  }
  // face-pipe collision
  else if (meshPointsWitness.size() > 2)
  {
    std::vector<Eigen::Vector3d> pointsWitnessSorted = meshPointsWitness;
    // `dir` points from A to B, so we want the highest dot product of dir at
    // the front of A
    std::sort(
        pointsWitnessSorted.begin(),
        pointsWitnessSorted.end(),
        [dir](Eigen::Vector3d& a, Eigen::Vector3d& b) {
          double aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
          double bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
          return aDot < bDot;
        });
    // All the meshPointsWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3d normal
        = (pointsWitnessSorted[0] - pointsWitnessSorted[1])
              .cross(pointsWitnessSorted[1] - pointsWitnessSorted[2])
              .normalized();
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    double normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // We want to find the closest points to the mesh plane
    std::vector<Eigen::Vector3d> capsulePointsWitness;

    // Now we need to process the two shapes to find intersection points
    std::vector<Contact> contacts;

    if (flipObjectOrder)
    {
      capsulePointsWitness.push_back(capsuleA + normal * capsuleRadius);
      capsulePointsWitness.push_back(capsuleB + normal * capsuleRadius);
      createFaceFaceContacts(
          contacts, o2, o1, dir, pointsWitnessSorted, capsulePointsWitness);
      for (Contact& contact : contacts)
      {
        assert(contact.type != VERTEX_FACE);

        // This happens if the edge of the capsule is inside the face, in 2D
        // space
        if (contact.type == FACE_VERTEX)
        {
          contact.type = FACE_SPHERE;
          contact.radiusB = capsuleRadius;
          // Fill the sphere center value depending on which endpoint this
          // contact is nearest
          if ((contact.point - capsuleA).squaredNorm()
              < (contact.point - capsuleB).squaredNorm())
          {
            contact.sphereCenter = capsuleA;
          }
          else
          {
            contact.sphereCenter = capsuleB;
          }
        }
        else if (contact.type == EDGE_EDGE)
        {
          contact.type = EDGE_PIPE;
          contact.edgeBClosestPoint += normal * capsuleRadius;
          contact.radiusB = capsuleRadius;
        }
        else
        {
          assert(
              "Got an unexpected contact type in createCapsuleMeshContacts"
              && false);
        }

        result.addContact(contact);
      }

      return contacts.size();
    }
    else
    {
      capsulePointsWitness.push_back(capsuleA - normal * capsuleRadius);
      capsulePointsWitness.push_back(capsuleB - normal * capsuleRadius);
      createFaceFaceContacts(
          contacts, o1, o2, dir, capsulePointsWitness, pointsWitnessSorted);
      for (Contact& contact : contacts)
      {
        assert(contact.type != FACE_VERTEX);

        // This happens if the edge of the capsule is inside the face, in 2D
        // space
        if (contact.type == VERTEX_FACE)
        {
          contact.type = SPHERE_FACE;
          contact.radiusA = capsuleRadius;
          // Fill the sphere center value depending on which endpoint this
          // contact is nearest
          if ((contact.point - capsuleA).squaredNorm()
              < (contact.point - capsuleB).squaredNorm())
          {
            contact.sphereCenter = capsuleA;
          }
          else
          {
            contact.sphereCenter = capsuleB;
          }
        }
        else if (contact.type == EDGE_EDGE)
        {
          contact.type = PIPE_EDGE;
          contact.edgeAClosestPoint += normal * capsuleRadius;
          contact.radiusA = capsuleRadius;
        }
        else
        {
          assert(
              "Got an unexpected contact type in createCapsuleMeshContacts"
              && false);
        }

        result.addContact(contact);
      }

      return contacts.size();
    }
  }

  assert(false);
  return 0;
}

/// This is necessary preparation for rapidly checking if another point is
/// contained within the convex shape. This sorts the shape by angle from the
/// center, and trims out any points that lie inside the convex polygon.
void prepareConvex2DShape(
    std::vector<Eigen::Vector3d>& shape,
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& basis2dX,
    const Eigen::Vector3d& basis2dY)
{
  /*
  #ifndef NDEBUG
    int originalSize = shape.size();
  #endif
  */

  // We need to throw out any points on the inside of the convex shape
  // TODO: there's gotta be a better algorithm than O(n^3).
  while (shape.size() > 0)
  {
    bool foundAnyToRemove = false;
    for (int i = 0; i < shape.size(); i++)
    {
      bool foundBoundaryPlane = false;
      Eigen::Vector2d shapeI
          = pointInPlane(shape[i], origin, basis2dX, basis2dY);

      for (int j = 0; j < shape.size(); j++)
      {
        if (i == j)
          continue;
        Eigen::Vector2d shapeJ
            = pointInPlane(shape[j], origin, basis2dX, basis2dY);

        Eigen::Vector2d plane
            = Eigen::Vector2d(shapeI(1) - shapeJ(1), shapeJ(0) - shapeI(0))
                  .normalized();
        double b = -plane.dot(shapeI);

        bool isBoundaryPlane = true;

        int side = 0;
        for (int k = 0; k < shape.size(); k++)
        {
          double measure
              = plane.dot(pointInPlane(shape[k], origin, basis2dX, basis2dY))
                + b;
          int kSide = ccdSign(measure);

          if (std::abs(measure) < 1e-3)
          {
            // This is colinear with the proposed boundary plane, ignore it
          }
          else if (side == 0)
          {
            side = kSide;
          }
          else if (side != kSide)
          {
            isBoundaryPlane = false;
            break;
          }
        }

        if (isBoundaryPlane)
        {
          foundBoundaryPlane = true;
          break;
        }
      }

      if (!foundBoundaryPlane)
      {
        // This point is not convex, and needs to be thrown out!
        shape.erase(shape.begin() + i);
        foundAnyToRemove = true;
        break;
      }
    }
    if (!foundAnyToRemove)
      break;
  }

  /*
  #ifndef NDEBUG
    int reducedSize = shape.size();
    if (reducedSize < originalSize)
    {
      std::cout << "Went from " << originalSize << " to " << reducedSize
                << " using:" << std::endl;
      std::cout << "Origin: " << std::endl << origin << std::endl;
      std::cout << "BasisX: " << std::endl << basis2dX << std::endl;
      std::cout << "BasisY: " << std::endl << basis2dY << std::endl;
    }
  #endif
  */

  // Sort the shape in clockwise order around some internal point (choose the
  // average).
  Eigen::Vector2d avg = Eigen::Vector2d::Zero();
  for (Eigen::Vector3d pt : shape)
  {
    avg += pointInPlane(pt, origin, basis2dX, basis2dY);
  }
  avg /= shape.size();
  std::sort(
      shape.begin(),
      shape.end(),
      [&avg, &origin, &basis2dX, &basis2dY](
          Eigen::Vector3d& a, Eigen::Vector3d& b) {
        return angle2D(avg, pointInPlane(a, origin, basis2dX, basis2dY))
               < angle2D(avg, pointInPlane(b, origin, basis2dX, basis2dY));
      });
}

/// This transforms a 3D point down to a 2D point in the given 3D plane
Eigen::Vector2d pointInPlane(
    const Eigen::Vector3d& point,
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& basis2dX,
    const Eigen::Vector3d& basis2dY)
{
  return Eigen::Vector2d(
      (point - origin).dot(basis2dX), (point - origin).dot(basis2dY));
}

/*
/// This is necessary preparation for rapidly checking if another point is
/// contained within the convex shape.
void prepareConvex2DShape(std::vector<Eigen::Vector2d>& shape)
{
  // We need to throw out any points on the inside of the convex shape
  // TODO: there's gotta be a better algorithm than O(n^3).
  while (shape.size() > 0)
  {
    bool foundAnyToRemove = false;
    for (int i = 0; i < shape.size(); i++)
    {
      bool foundBoundaryPlane = false;

      for (int j = 0; j < shape.size(); j++)
      {
        if (i == j)
          continue;

        Eigen::Vector2d plane
            = Eigen::Vector2d(
                  shape[i](1) - shape[j](1), shape[j](0) - shape[i](0))
                  .normalized();
        double b = -plane.dot(shape[i]);

        bool isBoundaryPlane = true;
        for (int k = 0; k < shape.size(); k++)
        {
          double measure = plane.dot(shape[k]) + b;
          if (measure < 0)
          {
            isBoundaryPlane = false;
            break;
          }
        }

        if (isBoundaryPlane)
        {
          foundBoundaryPlane = true;
          break;
        }
      }

      if (!foundBoundaryPlane)
      {
        // This point is not convex, and needs to be thrown out!
        shape.erase(shape.begin() + i);
        foundAnyToRemove = true;
        break;
      }
    }
    if (!foundAnyToRemove)
      break;
  }

  // Sort the shape in clockwise order around some internal point (choose the
  // average).
  Eigen::Vector2d avg = Eigen::Vector2d::Zero();
  for (Eigen::Vector2d pt : shape)
  {
    avg += pt;
  }
  avg /= shape.size();
  std::sort(
      shape.begin(),
      shape.end(),
      [&avg](Eigen::Vector2d& a, Eigen::Vector2d& b) {
        return angle2D(avg, a) < angle2D(avg, b);
      });
}
*/

// This implements the "2D cross product" as redefined here:
// https://stackoverflow.com/a/565282/13177487
inline double crossProduct2D(const Eigen::Vector2d& v, const Eigen::Vector2d& w)
{
  return v(0) * w(1) - v(1) * w(0);
}

inline void setCcdDefaultSettings(ccd_t& ccd)
{
  ccd.mpr_tolerance = 0.0001;
  ccd.epa_tolerance = 0.0001;
  ccd.dist_tolerance = 0.001;
  ccd.max_iterations = 100;
}

/*
/// This checks whether a 2D shape contains a point. This assumes that shape was
/// sorted using sortConvex2DShape().
///
/// Source:
///
https://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
/// A better source:
/// https://inginious.org/course/competitive-programming/geometry-pointinconvex
bool convex2DShapeContains(
    const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& shape)
{
  int side = 0;
  for (int i = 0; i < shape.size(); i++)
  {
    const Eigen::Vector2d& a = shape[i];
    const Eigen::Vector2d& b = shape[(i + 1) % shape.size()];
    int thisSide = ccdSign(crossProduct2D(point - a, b - a));
    if (i == 0)
      side = thisSide;
    else if (thisSide == 0)
      continue;
    else if (side == 0 && thisSide != 0)
      side = thisSide;
    else if (side != thisSide && side != 0)
      return false;
  }

  return true;
}
*/

/// This checks whether a 2D shape contains a point. This assumes that shape was
/// sorted using sortConvex2DShape().
///
/// Source:
/// https://demonstrations.wolfram.com/AnEfficientTestForAPointToBeInAConvexPolygon/
/// A better source:
/// https://inginious.org/course/competitive-programming/geometry-pointinconvex
bool convex2DShapeContains(
    const Eigen::Vector3d& point,
    const std::vector<Eigen::Vector3d>& shape,
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& basis2dX,
    const Eigen::Vector3d& basis2dY)
{
  Eigen::Vector2d point2d = pointInPlane(point, origin, basis2dX, basis2dY);

  int side = 0;
  for (int i = 0; i < shape.size(); i++)
  {
    Eigen::Vector2d a = pointInPlane(shape[i], origin, basis2dX, basis2dY);
    Eigen::Vector2d b = pointInPlane(
        shape[(i + 1) % shape.size()], origin, basis2dX, basis2dY);
    int thisSide = ccdSign(crossProduct2D(point2d - a, b - a));
    if (i == 0)
      side = thisSide;
    else if (thisSide == 0)
      continue;
    else if (side == 0 && thisSide != 0)
      side = thisSide;
    else if (side != thisSide && side != 0)
      return false;
  }

  return true;
}

double angle2D(const Eigen::Vector2d& from, const Eigen::Vector2d& to)
{
  return atan2(to(1) - from(1), to(0) - from(0));
}

// Returns 1 if the lines intersect, otherwise 0. In addition, if the lines
// intersect the intersection point may be stored in the floats i(0) and i(1).
//
// Adapted from sources: https://stackoverflow.com/a/1968345/13177487 and
// https://stackoverflow.com/a/565282/13177487
bool get2DLineIntersection(
    const Eigen::Vector2d& p,
    const Eigen::Vector2d& p1,
    const Eigen::Vector2d& q,
    const Eigen::Vector2d& q1,
    Eigen::Vector2d& out)
{
  Eigen::Vector2d r = p1 - p;
  Eigen::Vector2d s = q1 - q;

  // If r × s = 0 and (q − p) × r = 0, then the two lines are collinear.
  if (crossProduct2D(r, s) == 0 && crossProduct2D(q - p, r) == 0)
  {
    // In this case, express the endpoints of the second segment (q and q + s)
    // in terms of the equation of the first line segment (p + t r)

    double t0 = (q - p).dot(r) / r.dot(r);
    double t1 = (q + s - p).dot(r) / r.dot(r);

    // If the interval between t0 and t1 intersects the interval [0, 1] then
    // the line segments are collinear and overlapping; otherwise they are
    // collinear and disjoint. Note that if s and r point in opposite
    // directions, then s · r < 0 and so the interval to be checked is [t1,
    // t0] rather than [t0, t1].

    if (t0 >= 0 && t0 <= 1)
    {
      out = p + s * t0;
      return true;
    }
    else if (t1 >= 0 && t1 <= 1)
    {
      out = p + s * t1;
      return true;
    }
    else
    {
      return false;
    }
  }
  else if (crossProduct2D(r, s) == 0)
  {
    // If r × s = 0 and (q − p) × r ≠ 0, then the two lines are parallel and
    // non-intersecting.
    // If r × s = 0 by itself, then the two lines are non-parallel and
    // non-intersecting
    return false;
  }

  // If 0 ≤ t ≤ 1 and 0 ≤ u ≤ 1, the two line segments meet at the
  // point p + t r = q + u s.

  double t = crossProduct2D(q - p, s) / crossProduct2D(r, s);
  double u = crossProduct2D(p - q, r) / crossProduct2D(s, r);

  if (t >= 0 && t <= 1 && u >= 0 && u <= 1)
  {
    out = p + t * r;
    return true;
  }

  /*
  // Simpler proposed version
  double s, t;
  s = (-s1(1) * (p0(0) - q0(0)) + s1(0) * (p0(1) - q0(1)))
      / (-s2(0) * s1(1) + s1(0) * s2(1));
  t = (s2(0) * (p0(1) - q0(1)) - s2(1) * (p0(0) - q0(0)))
      / (-s2(0) * s1(1) + s1(0) * s2(1));

  if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
  {
    // Collision detected
    out(0) = p0(0) + (t * s1(0));
    out(1) = p0(1) + (t * s1(1));
    return true;
  }
  */

  return false; // No collision
}

int collideBoxBoxAsMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox; // support function for first object
  ccd.support2 = ccdSupportBox; // support function for second object
  ccd.center1 = ccdCenterBox;   // center function for first object
  ccd.center2 = ccdCenterBox;   // center function for second object
  setCcdDefaultSettings(ccd);   // maximal tolerance

  ccdBox box1;
  box1.size = &size0;
  box1.transform = &T0;

  ccdBox box2;
  box2.size = &size1;
  box2.transform = &T1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3d> pointsA
        = ccdPointsAtWitnessBox(&box1, &dir, false);
    std::vector<Eigen::Vector3d> pointsB
        = ccdPointsAtWitnessBox(&box2, &dir, true);

    return createMeshMeshContacts(o1, o2, result, &dir, pointsA, pointsB);
  }
  return 0;
}

int collideMeshBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh; // support function for first object
  ccd.support2 = ccdSupportBox;  // support function for second object
  ccd.center1 = ccdCenterMesh;   // center function for first object
  ccd.center2 = ccdCenterBox;    // center function for second object
  setCcdDefaultSettings(ccd);    // maximal tolerance

  ccdMesh mesh1;
  mesh1.mesh = mesh0;
  mesh1.transform = &c0;
  mesh1.scale = &size0;

  ccdBox box2;
  box2.size = &size1;
  box2.transform = &c1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&mesh1, &box2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3d> pointsA
        = ccdPointsAtWitnessMesh(&mesh1, &dir, false);
    std::vector<Eigen::Vector3d> pointsB
        = ccdPointsAtWitnessBox(&box2, &dir, true);

    return createMeshMeshContacts(o1, o2, result, &dir, pointsA, pointsB);
  }
  return 0;
}

int collideBoxMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const aiScene* m1,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox;  // support function for first object
  ccd.support2 = ccdSupportMesh; // support function for second object
  ccd.center1 = ccdCenterBox;    // center function for first object
  ccd.center2 = ccdCenterMesh;   // center function for second object
  setCcdDefaultSettings(ccd);    // maximal tolerance

  ccdBox box1;
  box1.size = &size0;
  box1.transform = &c0;

  ccdMesh mesh2;
  mesh2.mesh = m1;
  mesh2.transform = &c1;
  mesh2.scale = &size1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&box1, &mesh2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3d> pointsA
        = ccdPointsAtWitnessBox(&box1, &dir, false);
    std::vector<Eigen::Vector3d> pointsB
        = ccdPointsAtWitnessMesh(&mesh2, &dir, true);

    if (pointsB.size() == 0)
    {
      std::vector<Eigen::Vector3d> pointsB
          = ccdPointsAtWitnessMesh(&mesh2, &dir, true);
    }

    return createMeshMeshContacts(o1, o2, result, &dir, pointsA, pointsB);
  }
  return 0;
}

int collideMeshSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const double& r1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  ccdMesh mesh;
  mesh.mesh = mesh0;
  mesh.transform = &c0;
  mesh.scale = &size0;

  ccdSphere sphere;
  sphere.radius = r1;
  sphere.transform = &c1;

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh;   // support function for first object
  ccd.support2 = ccdSupportSphere; // support function for second object
  ccd.center1 = ccdCenterMesh;     // center function for first object
  ccd.center2 = ccdCenterSphere;   // center function for second object
  setCcdDefaultSettings(ccd);      // maximal tolerance

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&mesh, &sphere, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3d> meshPoints
        = ccdPointsAtWitnessMesh(&mesh, &dir, false);

    return createMeshSphereContact(
        o1, o2, result, &dir, meshPoints, c1.translation(), r1);
  }
  return 0;
}

int collideSphereMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& r0,
    const Eigen::Isometry3d& c0,
    const aiScene* mesh1,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  ccdSphere sphere;
  sphere.radius = r0;
  sphere.transform = &c0;

  ccdMesh mesh;
  mesh.mesh = mesh1;
  mesh.transform = &c1;
  mesh.scale = &size1;

  // set up ccd_t struct
  ccd.support1 = ccdSupportSphere; // support function for first object
  ccd.support2 = ccdSupportMesh;   // support function for second object
  ccd.center1 = ccdCenterSphere;   // center function for first object
  ccd.center2 = ccdCenterMesh;     // center function for second object
  setCcdDefaultSettings(ccd);      // maximal tolerance

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&sphere, &mesh, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3d> meshPoints
        = ccdPointsAtWitnessMesh(&mesh, &dir, true);

    return createSphereMeshContact(
        o1, o2, result, &dir, c0.translation(), r0, meshPoints);
  }
  return 0;
}

int collideMeshMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* m0,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& c0,
    const aiScene* m1,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& c1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh; // support function for first object
  ccd.support2 = ccdSupportMesh; // support function for second object
  ccd.center1 = ccdCenterMesh;   // center function for first object
  ccd.center2 = ccdCenterMesh;   // center function for second object
  setCcdDefaultSettings(ccd);    // maximal tolerance

  ccdMesh mesh1;
  mesh1.mesh = m0;
  mesh1.transform = &c0;
  mesh1.scale = &size0;

  ccdMesh mesh2;
  mesh2.mesh = m1;
  mesh2.transform = &c1;
  mesh2.scale = &size1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&mesh1, &mesh2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3d> pointsA
        = ccdPointsAtWitnessMesh(&mesh1, &dir, false);
    std::vector<Eigen::Vector3d> pointsB
        = ccdPointsAtWitnessMesh(&mesh2, &dir, true);

    return createMeshMeshContacts(o1, o2, result, &dir, pointsA, pointsB);
  }
  return 0;
}

int collideCapsuleCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    double height0,
    double radius0,
    const Eigen::Isometry3d& T0,
    double height1,
    double radius1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  Eigen::Vector3d pa = T0 * (Eigen::Vector3d::UnitZ() * -(height0 / 2));
  Eigen::Vector3d pb = T0 * (Eigen::Vector3d::UnitZ() * (height0 / 2));
  Eigen::Vector3d ua = T1 * (Eigen::Vector3d::UnitZ() * -(height1 / 2));
  Eigen::Vector3d ub = T1 * (Eigen::Vector3d::UnitZ() * (height1 / 2));
  double alpha, beta;
  dSegmentsClosestApproach(pa, ua, pb, ub, &alpha, &beta);
  if (alpha < 0)
    alpha = 0;
  if (alpha > 1)
    alpha = 1;
  if (beta < 0)
    beta = 0;
  if (beta > 1)
    beta = 1;

  Eigen::Vector3d closest0 = pa + (pb - pa) * alpha;
  Eigen::Vector3d closest1 = ua + (ub - ua) * beta;

  double dist = (closest0 - closest1).norm();
  double rsum = radius0 + radius1;
  if (dist <= rsum)
  {
    // There's a contact!
    radius0 /= rsum;
    radius1 /= rsum;

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth = rsum - dist;
    contact.point = (closest0 * radius1) + (closest1 * radius0);
    contact.normal = (closest0 - closest1).normalized();

    const double SPHERE_THRESHOLD = 1e-8;

    contact.radiusA = radius0 * rsum;
    contact.radiusB = radius1 * rsum;
    bool isSphere0 = std::abs(alpha) < SPHERE_THRESHOLD
                     || std::abs(1 - alpha) < SPHERE_THRESHOLD;
    bool isSphere1 = std::abs(beta) < SPHERE_THRESHOLD
                     || std::abs(1 - beta) < SPHERE_THRESHOLD;

    if (isSphere0 && isSphere1)
    {
      contact.type = SPHERE_SPHERE;
      contact.centerA = closest0;
      contact.centerB = closest1;
    }
    else if (isSphere0)
    {
      contact.type = SPHERE_PIPE;
      contact.centerA = closest0;
      contact.edgeBClosestPoint = closest1;
      contact.edgeBDir = ub - ua;
    }
    else if (isSphere1)
    {
      contact.type = PIPE_SPHERE;
      contact.edgeAClosestPoint = closest0;
      contact.edgeADir = pb - pa;
      contact.centerB = closest1;
    }
    else
    {
      contact.type = PIPE_PIPE;
      contact.edgeADir = pb - pa;
      contact.centerB = closest1;
      contact.edgeBClosestPoint = closest1;
      contact.edgeBDir = ub - ua;
    }

    result.addContact(contact);

    return 1;
  }
  // No contact
  return 0;
}

int collideSphereCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    double radius0,
    const Eigen::Isometry3d& T0,
    double height1,
    double radius1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  double alpha;
  Eigen::Vector3d center0 = T0.translation();
  Eigen::Vector3d ua = T1 * (Eigen::Vector3d::UnitZ() * -(height1 / 2));
  Eigen::Vector3d ub = T1 * (Eigen::Vector3d::UnitZ() * (height1 / 2));

  double dist = dDistPointToSegment(center0, ua, ub, &alpha);
  if (dist < radius0 + radius1)
  {
    // There's a contact!

    Eigen::Vector3d closest1 = ua + (ub - ua) * alpha;
    double rsum = radius0 + radius1;

    radius0 /= rsum;
    radius1 /= rsum;

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth = rsum - dist;
    contact.point = (center0 * radius1) + (closest1 * radius0);
    contact.normal = (center0 - closest1).normalized();

    const double SPHERE_THRESHOLD = 1e-8;

    contact.radiusA = radius0 * rsum;
    contact.radiusB = radius1 * rsum;
    bool isSphere1 = std::abs(alpha) < SPHERE_THRESHOLD
                     || std::abs(1 - alpha) < SPHERE_THRESHOLD;

    if (isSphere1)
    {
      contact.type = SPHERE_SPHERE;
      contact.centerA = center0;
      contact.centerB = closest1;
    }
    else
    {
      contact.type = SPHERE_PIPE;
      contact.centerA = center0;
      contact.edgeBClosestPoint = closest1;
      contact.edgeBDir = ub - ua;
    }

    result.addContact(contact);

    return 1;
  }

  return 0;
}

int collideCapsuleSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    double height0,
    double radius0,
    const Eigen::Isometry3d& T0,
    double radius1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  double alpha;
  Eigen::Vector3d ua = T0 * (Eigen::Vector3d::UnitZ() * -(height0 / 2));
  Eigen::Vector3d ub = T0 * (Eigen::Vector3d::UnitZ() * (height0 / 2));
  Eigen::Vector3d center1 = T1.translation();

  double dist = dDistPointToSegment(center1, ua, ub, &alpha);
  if (dist < radius0 + radius1)
  {
    // There's a contact!

    Eigen::Vector3d closest0 = ua + (ub - ua) * alpha;
    double rsum = radius0 + radius1;

    radius0 /= rsum;
    radius1 /= rsum;

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth = rsum - dist;
    contact.point = (closest0 * radius1) + (center1 * radius0);
    contact.normal = (closest0 - center1).normalized();

    const double SPHERE_THRESHOLD = 1e-8;

    contact.radiusA = radius0 * rsum;
    contact.radiusB = radius1 * rsum;
    bool isSphere1 = std::abs(alpha) < SPHERE_THRESHOLD
                     || std::abs(1 - alpha) < SPHERE_THRESHOLD;

    if (isSphere1)
    {
      contact.type = SPHERE_SPHERE;
      contact.centerA = closest0;
      contact.centerB = center1;
    }
    else
    {
      contact.type = PIPE_SPHERE;
      contact.edgeAClosestPoint = closest0;
      contact.edgeADir = ub - ua;
      contact.centerB = center1;
    }

    result.addContact(contact);

    return 1;
  }

  return 0;
}

int collideBoxCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    double height1,
    double radius1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportBox;     // support function for first object
  ccd.support2 = ccdSupportCapsule; // support function for second object
  ccd.center1 = ccdCenterBox;       // center function for first object
  ccd.center2 = ccdCenterCapsule;   // center function for second object
  setCcdDefaultSettings(ccd);       // maximal tolerance

  ccdBox box1;
  box1.size = &size0;
  box1.transform = &T0;

  ccdCapsule capsule2;
  capsule2.height = height1;
  capsule2.radius = radius1;
  capsule2.transform = &T1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&box1, &capsule2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    Eigen::Map<Eigen::Vector3d> posMap(pos.v);
    Eigen::Vector3d localPos = T1.inverse() * posMap;
    if (localPos(2) > height1 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, height1 / 2);
      Eigen::Vector3d sphereCenter = T1 * Eigen::Vector3d(0, 0, height1 / 2);
      Eigen::Vector3d transCenter = (T1 * sphereTransform).translation();
      return collideBoxSphere(
          o1, o2, size0, T0, radius1, T1 * sphereTransform, result);
    }
    else if (localPos(2) < -height1 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, -height1 / 2);
      Eigen::Vector3d sphereCenter = T1 * Eigen::Vector3d(0, 0, -height1 / 2);
      Eigen::Vector3d transCenter = (T1 * sphereTransform).translation();
      return collideBoxSphere(
          o1, o2, size0, T0, radius1, T1 * sphereTransform, result);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3d> meshPoints
        = ccdPointsAtWitnessBox(&box1, &dir, false);

    return createCapsuleMeshContact(
        o1,
        o2,
        result,
        &dir,
        T1 * Eigen::Vector3d(0, 0, height1 / 2),
        T1 * Eigen::Vector3d(0, 0, -height1 / 2),
        radius1,
        meshPoints,
        true);
  }
  return 0;
}

int collideCapsuleBox(
    CollisionObject* o1,
    CollisionObject* o2,
    double height0,
    double radius0,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportCapsule; // support function for first object
  ccd.support2 = ccdSupportBox;     // support function for second object
  ccd.center1 = ccdCenterCapsule;   // center function for first object
  ccd.center2 = ccdCenterBox;       // center function for second object
  setCcdDefaultSettings(ccd);

  ccdCapsule capsule1;
  capsule1.height = height0;
  capsule1.radius = radius0;
  capsule1.transform = &T0;

  ccdBox box2;
  box2.size = &size1;
  box2.transform = &T1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect = ccdMPRPenetration(&capsule1, &box2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    Eigen::Map<Eigen::Vector3d> posMap(pos.v);
    Eigen::Vector3d localPos = T0.inverse() * posMap;
    if (localPos(2) > height0 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, height0 / 2);
      return collideSphereBox(
          o1, o2, radius0, T0 * sphereTransform, size1, T1, result);
    }
    else if (localPos(2) < -height0 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, -height0 / 2);
      return collideSphereBox(
          o1, o2, radius0, T0 * sphereTransform, size1, T1, result);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3d> meshPoints
        = ccdPointsAtWitnessBox(&box2, &dir, true);

    return createCapsuleMeshContact(
        o1,
        o2,
        result,
        &dir,
        T0 * Eigen::Vector3d(0, 0, height0 / 2),
        T0 * Eigen::Vector3d(0, 0, -height0 / 2),
        radius0,
        meshPoints,
        false);
  }
  return 0;
}

int collideMeshCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* m0,
    const Eigen::Vector3d& size0,
    const Eigen::Isometry3d& T0,
    double height1,
    double radius1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportMesh;    // support function for first object
  ccd.support2 = ccdSupportCapsule; // support function for second object
  ccd.center1 = ccdCenterMesh;      // center function for first object
  ccd.center2 = ccdCenterCapsule;   // center function for second object
  setCcdDefaultSettings(ccd);       // maximal tolerance

  ccdMesh mesh1;
  mesh1.mesh = m0;
  mesh1.transform = &T0;
  mesh1.scale = &size0;

  ccdCapsule capsule2;
  capsule2.height = height1;
  capsule2.radius = radius1;
  capsule2.transform = &T1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect
      = ccdMPRPenetration(&mesh1, &capsule2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    Eigen::Map<Eigen::Vector3d> posMap(pos.v);
    Eigen::Vector3d localPos = T1.inverse() * posMap;
    if (localPos(2) > height1 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, height1 / 2);
      Eigen::Vector3d sphereCenter = T1 * Eigen::Vector3d(0, 0, height1 / 2);
      Eigen::Vector3d transCenter = (T1 * sphereTransform).translation();
      return collideMeshSphere(
          o1, o2, m0, size0, T0, radius1, T1 * sphereTransform, result);
    }
    else if (localPos(2) < -height1 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, -height1 / 2);
      Eigen::Vector3d sphereCenter = T1 * Eigen::Vector3d(0, 0, -height1 / 2);
      Eigen::Vector3d transCenter = (T1 * sphereTransform).translation();
      return collideMeshSphere(
          o1, o2, m0, size0, T0, radius1, T1 * sphereTransform, result);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3d> meshPoints
        = ccdPointsAtWitnessMesh(&mesh1, &dir, false);

    return createCapsuleMeshContact(
        o1,
        o2,
        result,
        &dir,
        T1 * Eigen::Vector3d(0, 0, height1 / 2),
        T1 * Eigen::Vector3d(0, 0, -height1 / 2),
        radius1,
        meshPoints,
        true);
  }
  return 0;
}

int collideCapsuleMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    double height0,
    double radius0,
    const Eigen::Isometry3d& T0,
    const aiScene* m1,
    const Eigen::Vector3d& size1,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  ccd_t ccd;
  CCD_INIT(&ccd); // initialize ccd_t struct

  // set up ccd_t struct
  ccd.support1 = ccdSupportCapsule; // support function for first object
  ccd.support2 = ccdSupportMesh;    // support function for second object
  ccd.center1 = ccdCenterCapsule;   // center function for first object
  ccd.center2 = ccdCenterMesh;      // center function for second object
  setCcdDefaultSettings(ccd);       // maximal tolerance

  ccdCapsule capsule1;
  capsule1.height = height0;
  capsule1.radius = radius0;
  capsule1.transform = &T0;

  ccdMesh mesh2;
  mesh2.mesh = m1;
  mesh2.scale = &size1;
  mesh2.transform = &T1;

  ccd_real_t depth;
  long key = (long)o1 ^ (long)o2;
  ccd_vec3_t& dir = _ccdDirCache[key];
  ccd_vec3_t& pos = _ccdPosCache[key];
  int intersect
      = ccdMPRPenetration(&capsule1, &mesh2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    Eigen::Map<Eigen::Vector3d> posMap(pos.v);
    Eigen::Vector3d localPos = T0.inverse() * posMap;
    if (localPos(2) > height0 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, height0 / 2);
      return collideSphereMesh(
          o1, o2, radius0, T0 * sphereTransform, m1, size1, T1, result);
    }
    else if (localPos(2) < -height0 / 2)
    {
      Eigen::Isometry3d sphereTransform = Eigen::Isometry3d::Identity();
      sphereTransform.translation() = Eigen::Vector3d(0, 0, -height0 / 2);
      return collideSphereMesh(
          o1, o2, radius0, T0 * sphereTransform, m1, size1, T1, result);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3d> meshPoints
        = ccdPointsAtWitnessMesh(&mesh2, &dir, true);

    return createCapsuleMeshContact(
        o1,
        o2,
        result,
        &dir,
        T0 * Eigen::Vector3d(0, 0, height0 / 2),
        T0 * Eigen::Vector3d(0, 0, -height0 / 2),
        radius0,
        meshPoints,
        false);
  }
  return 0;
}

int collideCylinderSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& cyl_rad,
    const double& half_height,
    const Eigen::Isometry3d& T0,
    const double& sphere_rad,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  Eigen::Vector3d center = T0.inverse() * T1.translation();

  double dist = sqrt(center[0] * center[0] + center[1] * center[1]);

  if (dist < cyl_rad && std::abs(center[2]) < half_height + sphere_rad)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth
        = 0.5 * (half_height + sphere_rad - math::sign(center[2]) * center[2]);
    contact.point
        = T0
          * Eigen::Vector3d(
              center[0], center[1], half_height - contact.penetrationDepth);
    contact.normal
        = T0.linear() * Eigen::Vector3d(0.0, 0.0, math::sign(center[2]));
    result.addContact(contact);
    return 1;
  }
  else
  {
    double penetration = 0.5 * (cyl_rad + sphere_rad - dist);
    if (penetration > 0.0)
    {
      if (std::abs(center[2]) > half_height)
      {
        Eigen::Vector3d point
            = (Eigen::Vector3d(center[0], center[1], 0.0).normalized());
        point *= cyl_rad;
        point[2] = math::sign(center[2]) * half_height;
        Eigen::Vector3d normal = point - center;
        penetration = sphere_rad - normal.norm();
        normal = (T0.linear() * normal).normalized();
        point = T0 * point;

        if (penetration > 0.0)
        {
          Contact contact;
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.point = point;
          contact.normal = normal;
          contact.penetrationDepth = penetration;
          result.addContact(contact);
          return 1;
        }
      }
      else // if( center[2] >= -half_height && center[2] <= half_height )
      {
        Eigen::Vector3d point
            = (Eigen::Vector3d(center[0], center[1], 0.0)).normalized();
        Eigen::Vector3d normal = -(T0.linear() * point);
        point *= (cyl_rad - penetration);
        point[2] = center[2];
        point = T0 * point;

        Contact contact;
        contact.collisionObject1 = o1;
        contact.collisionObject2 = o2;
        contact.point = point;
        contact.normal = normal;
        contact.penetrationDepth = penetration;
        result.addContact(contact);
        return 1;
      }
    }
  }
  return 0;
}

int collideCylinderPlane(
    CollisionObject* o1,
    CollisionObject* o2,
    const double& cyl_rad,
    const double& half_height,
    const Eigen::Isometry3d& T0,
    const Eigen::Vector3d& plane_normal,
    const Eigen::Isometry3d& T1,
    CollisionResult& result)
{
  Eigen::Vector3d normal = T1.linear() * plane_normal;
  Eigen::Vector3d Rx = T0.linear().rightCols(1);
  Eigen::Vector3d Ry = normal - normal.dot(Rx) * Rx;
  double mag = Ry.norm();
  Ry.normalize();
  if (mag < DART_COLLISION_EPS)
  {
    if (std::abs(Rx[2]) > 1.0 - DART_COLLISION_EPS)
      Ry = Eigen::Vector3d::UnitX();
    else
      Ry = (Eigen::Vector3d(Rx[1], -Rx[0], 0.0)).normalized();
  }

  Eigen::Vector3d Rz = Rx.cross(Ry);
  Eigen::Isometry3d T;
  T.linear().col(0) = Rx;
  T.linear().col(1) = Ry;
  T.linear().col(2) = Rz;
  T.translation() = T0.translation();

  Eigen::Vector3d nn = T.linear().transpose() * normal;
  Eigen::Vector3d pn = T.inverse() * T1.translation();

  // four corners c0 = ( -h/2, -r ), c1 = ( +h/2, -r ), c2 = ( +h/2, +r ), c3
  // = ( -h/2, +r )
  Eigen::Vector3d c[4]
      = {Eigen::Vector3d(-half_height, -cyl_rad, 0.0),
         Eigen::Vector3d(+half_height, -cyl_rad, 0.0),
         Eigen::Vector3d(+half_height, +cyl_rad, 0.0),
         Eigen::Vector3d(-half_height, +cyl_rad, 0.0)};

  double depth[4]
      = {(pn - c[0]).dot(nn),
         (pn - c[1]).dot(nn),
         (pn - c[2]).dot(nn),
         (pn - c[3]).dot(nn)};

  double penetration = -1.0;
  int found = -1;
  for (int i = 0; i < 4; i++)
  {
    if (depth[i] > penetration)
    {
      penetration = depth[i];
      found = i;
    }
  }

  Eigen::Vector3d point;

  if (std::abs(depth[found] - depth[(found + 1) % 4]) < DART_COLLISION_EPS)
    point = T * (0.5 * (c[found] + c[(found + 1) % 4]));
  else if (std::abs(depth[found] - depth[(found + 3) % 4]) < DART_COLLISION_EPS)
    point = T * (0.5 * (c[found] + c[(found + 3) % 4]));
  else
    point = T * c[found];

  if (penetration > 0.0)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = point;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
    return 1;
  }

  return 0;
}

//==============================================================================
int collide(CollisionObject* o1, CollisionObject* o2, CollisionResult& result)
{
  // TODO(JS): We could make the contact point computation as optional for
  // the case that we want only binary check.

  const auto& shape1 = o1->getShape();
  const auto& shape2 = o2->getShape();

  const auto& shapeType1 = shape1->getType();
  const auto& shapeType2 = shape2->getType();

  const Eigen::Isometry3d& T1 = o1->getTransform();
  const Eigen::Isometry3d& T2 = o2->getTransform();

  if (dynamics::SphereShape::getStaticType() == shapeType1)
  {
    const auto* sphere0
        = static_cast<const dynamics::SphereShape*>(shape1.get());

    if (dynamics::SphereShape::getStaticType() == shapeType2)
    {
      const auto* sphere1
          = static_cast<const dynamics::SphereShape*>(shape2.get());

      return collideSphereSphere(
          o1, o2, sphere0->getRadius(), T1, sphere1->getRadius(), T2, result);
    }
    else if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      return collideSphereBox(
          o1, o2, sphere0->getRadius(), T1, box1->getSize(), T2, result);
    }
    else if (dynamics::EllipsoidShape::getStaticType() == shapeType2)
    {
      const auto* ellipsoid1
          = static_cast<const dynamics::EllipsoidShape*>(shape2.get());

      return collideSphereSphere(
          o1,
          o2,
          sphere0->getRadius(),
          T1,
          ellipsoid1->getRadii()[0],
          T2,
          result);
    }
    else if (dynamics::MeshShape::getStaticType() == shapeType2)
    {
      const auto* mesh1 = static_cast<const dynamics::MeshShape*>(shape2.get());

      return collideSphereMesh(
          o1,
          o2,
          sphere0->getRadius(),
          T1,
          mesh1->getMesh(),
          mesh1->getScale(),
          T2,
          result);
    }
    else if (dynamics::CapsuleShape::getStaticType() == shapeType2)
    {
      const auto* capsule1
          = static_cast<const dynamics::CapsuleShape*>(shape2.get());

      return collideSphereCapsule(
          o1,
          o2,
          sphere0->getRadius(),
          T1,
          capsule1->getHeight(),
          capsule1->getRadius(),
          T2,
          result);
    }
  }
  else if (dynamics::BoxShape::getStaticType() == shapeType1)
  {
    const auto* box0 = static_cast<const dynamics::BoxShape*>(shape1.get());

    if (dynamics::SphereShape::getStaticType() == shapeType2)
    {
      const auto* sphere1
          = static_cast<const dynamics::SphereShape*>(shape2.get());

      return collideBoxSphere(
          o1, o2, box0->getSize(), T1, sphere1->getRadius(), T2, result);
    }
    else if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      // Used to be collideBoxBox(), but that doesn't annotate its collision
      // points properly for complex face-face collisions.
      return collideBoxBoxAsMesh(
          o1, o2, box0->getSize(), T1, box1->getSize(), T2, result);
    }
    else if (dynamics::EllipsoidShape::getStaticType() == shapeType2)
    {
      const auto* ellipsoid1
          = static_cast<const dynamics::EllipsoidShape*>(shape2.get());

      return collideBoxSphere(
          o1, o2, box0->getSize(), T1, ellipsoid1->getRadii()[0], T2, result);
    }
    else if (dynamics::MeshShape::getStaticType() == shapeType2)
    {
      const auto* mesh1 = static_cast<const dynamics::MeshShape*>(shape2.get());

      return collideBoxMesh(
          o1,
          o2,
          box0->getSize(),
          T1,
          mesh1->getMesh(),
          mesh1->getScale(),
          T2,
          result);
    }
    else if (dynamics::CapsuleShape::getStaticType() == shapeType2)
    {
      const auto* capsule1
          = static_cast<const dynamics::CapsuleShape*>(shape2.get());

      return collideBoxCapsule(
          o1,
          o2,
          box0->getSize(),
          T1,
          capsule1->getHeight(),
          capsule1->getRadius(),
          T2,
          result);
    }
  }
  else if (dynamics::EllipsoidShape::getStaticType() == shapeType1)
  {
    const auto* ellipsoid0
        = static_cast<const dynamics::EllipsoidShape*>(shape1.get());

    if (dynamics::SphereShape::getStaticType() == shapeType2)
    {
      const auto* sphere1
          = static_cast<const dynamics::SphereShape*>(shape2.get());

      return collideSphereSphere(
          o1,
          o2,
          ellipsoid0->getRadii()[0],
          T1,
          sphere1->getRadius(),
          T2,
          result);
    }
    else if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      return collideSphereBox(
          o1, o2, ellipsoid0->getRadii()[0], T1, box1->getSize(), T2, result);
    }
    else if (dynamics::EllipsoidShape::getStaticType() == shapeType2)
    {
      const auto* ellipsoid1
          = static_cast<const dynamics::EllipsoidShape*>(shape2.get());

      return collideSphereSphere(
          o1,
          o2,
          ellipsoid0->getRadii()[0],
          T1,
          ellipsoid1->getRadii()[0],
          T2,
          result);
    }
    else if (dynamics::MeshShape::getStaticType() == shapeType2)
    {
      const auto* mesh1 = static_cast<const dynamics::MeshShape*>(shape2.get());

      return collideSphereMesh(
          o1,
          o2,
          ellipsoid0->getRadii()[0],
          T1,
          mesh1->getMesh(),
          mesh1->getScale(),
          T2,
          result);
    }
    else if (dynamics::CapsuleShape::getStaticType() == shapeType2)
    {
      const auto* capsule1
          = static_cast<const dynamics::CapsuleShape*>(shape2.get());

      return collideSphereCapsule(
          o1,
          o2,
          ellipsoid0->getRadii()[0],
          T1,
          capsule1->getHeight(),
          capsule1->getRadius(),
          T2,
          result);
    }
  }
  else if (dynamics::MeshShape::getStaticType() == shapeType1)
  {
    const auto* mesh0 = static_cast<const dynamics::MeshShape*>(shape1.get());

    if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      return collideMeshBox(
          o1,
          o2,
          mesh0->getMesh(),
          mesh0->getScale(),
          T1,
          box1->getSize(),
          T2,
          result);
    }
    else if (dynamics::SphereShape::getStaticType() == shapeType2)
    {
      const auto* sphere1
          = static_cast<const dynamics::SphereShape*>(shape2.get());

      return collideMeshSphere(
          o1,
          o2,
          mesh0->getMesh(),
          mesh0->getScale(),
          T1,
          sphere1->getRadius(),
          T2,
          result);
    }
    else if (dynamics::EllipsoidShape::getStaticType() == shapeType2)
    {
      const auto* ellipsoid1
          = static_cast<const dynamics::EllipsoidShape*>(shape2.get());

      return collideMeshSphere(
          o1,
          o2,
          mesh0->getMesh(),
          mesh0->getScale(),
          T1,
          ellipsoid1->getRadii()[0],
          T2,
          result);
    }
    else if (dynamics::MeshShape::getStaticType() == shapeType2)
    {
      const auto* mesh1 = static_cast<const dynamics::MeshShape*>(shape2.get());

      return collideMeshMesh(
          o1,
          o2,
          mesh0->getMesh(),
          mesh0->getScale(),
          T1,
          mesh1->getMesh(),
          mesh1->getScale(),
          T2,
          result);
    }
    else if (dynamics::CapsuleShape::getStaticType() == shapeType2)
    {
      const auto* capsule1
          = static_cast<const dynamics::CapsuleShape*>(shape2.get());

      return collideMeshCapsule(
          o1,
          o2,
          mesh0->getMesh(),
          mesh0->getScale(),
          T1,
          capsule1->getHeight(),
          capsule1->getRadius(),
          T2,
          result);
    }
  }
  else if (dynamics::CapsuleShape::getStaticType() == shapeType1)
  {
    const auto* capsule0
        = static_cast<const dynamics::CapsuleShape*>(shape1.get());

    if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      return collideCapsuleBox(
          o1,
          o2,
          capsule0->getHeight(),
          capsule0->getRadius(),
          T1,
          box1->getSize(),
          T2,
          result);
    }
    else if (dynamics::SphereShape::getStaticType() == shapeType2)
    {
      const auto* sphere1
          = static_cast<const dynamics::SphereShape*>(shape2.get());

      return collideCapsuleSphere(
          o1,
          o2,
          capsule0->getHeight(),
          capsule0->getRadius(),
          T1,
          sphere1->getRadius(),
          T2,
          result);
    }
    else if (dynamics::EllipsoidShape::getStaticType() == shapeType2)
    {
      const auto* ellipsoid1
          = static_cast<const dynamics::EllipsoidShape*>(shape2.get());

      return collideCapsuleSphere(
          o1,
          o2,
          capsule0->getHeight(),
          capsule0->getRadius(),
          T1,
          ellipsoid1->getRadii()[0],
          T2,
          result);
    }
    else if (dynamics::MeshShape::getStaticType() == shapeType2)
    {
      const auto* mesh1 = static_cast<const dynamics::MeshShape*>(shape2.get());

      return collideCapsuleMesh(
          o1,
          o2,
          capsule0->getHeight(),
          capsule0->getRadius(),
          T1,
          mesh1->getMesh(),
          mesh1->getScale(),
          T2,
          result);
    }
    else if (dynamics::CapsuleShape::getStaticType() == shapeType2)
    {
      const auto* capsule1
          = static_cast<const dynamics::CapsuleShape*>(shape2.get());

      return collideCapsuleCapsule(
          o1,
          o2,
          capsule0->getHeight(),
          capsule0->getRadius(),
          T1,
          capsule1->getHeight(),
          capsule1->getRadius(),
          T2,
          result);
    }
  }
  // collideCapsuleCapsule

  dterr << "[DARTCollisionDetector] Attempting to check for an "
        << "unsupported shape pair: [" << shape1->getType() << "] - ["
        << shape2->getType() << "]. Returning false.\n";

  return false;
}

} // namespace collision
} // namespace dart
