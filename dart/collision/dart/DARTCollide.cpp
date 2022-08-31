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
#include <thread>

#include "dart/collision/CollisionObject.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/CapsuleShape.hpp"
#include "dart/dynamics/CylinderShape.hpp"
#include "dart/dynamics/EllipsoidShape.hpp"
#include "dart/dynamics/MeshShape.hpp"
#include "dart/dynamics/SphereShape.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"

namespace dart {

using namespace math;
namespace collision {

// point : world coordinate vector
// normal : normal vector from right to left 0 <- 1
// penetration : real positive means penetration

#define DART_COLLISION_WITNESS_PLANE_DEPTH 0.01
#define DART_COLLISION_EPS 1E-6
// static const int MAX_CYLBOX_CLIP_POINTS = 16;
// static const int nCYLINDER_AXIS = 2;
// Number of segment of cylinder base circle.
// Must be divisible by 4.
// static const int nCYLINDER_SEGMENT = 8;

#ifdef DART_USE_ARBITRARY_PRECISION
_ccd_inline int ccdIsZero(s_t val)
{
  return CCD_FABS(val) < CCD_EPS;
}

/*
_ccd_inline int ccdEq(s_t _a, s_t _b)
{
  s_t ab;
  s_t a, b;

  ab = CCD_FABS(_a - _b);
  if (CCD_FABS(ab) < CCD_EPS)
    return 1;

  a = CCD_FABS(_a);
  b = CCD_FABS(_b);
  if (b > a)
  {
    return ab < CCD_EPS * b;
  }
  else
  {
    return ab < CCD_EPS * a;
  }
}
*/

_ccd_inline int ccdSign(s_t val)
{
  if (ccdIsZero(val))
  {
    return 0;
  }
  else if (val < CCD_ZERO)
  {
    return -1;
  }
  return 1;
}
#endif

typedef s_t dVector3[4];
typedef s_t dVector3[4];
typedef s_t dVector4[4];
typedef s_t dMatrix3[12];
typedef s_t dMatrix4[16];
typedef s_t dMatrix6[48];
typedef s_t dQuaternion[4];

inline void convVector(const Eigen::Vector3s& p0, dVector3& p1)
{
  p1[0] = p0[0];
  p1[1] = p0[1];
  p1[2] = p0[2];
}

inline void convMatrix(const Eigen::Isometry3s& T0, dMatrix3& R0)
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
  s_t depth;
};

inline s_t Inner(const s_t* a, const s_t* b)
{
  return ((a)[0] * (b)[0] + (a)[1] * (b)[1] + (a)[2] * (b)[2]);
}

inline s_t Inner14(const s_t* a, const s_t* b)
{
  return ((a)[0] * (b)[0] + (a)[1] * (b)[4] + (a)[2] * (b)[8]);
}

inline s_t Inner41(const s_t* a, const s_t* b)
{
  return ((a)[0] * (b)[0] + (a)[4] * (b)[1] + (a)[8] * (b)[2]);
}

inline s_t Inner44(const s_t* a, const s_t* b)
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

inline void dMULTIPLY0_331(s_t* A, const s_t* B, const s_t* C)
{
  dMULTIPLYOP0_331(A, =, B, C)
}

inline void dMULTIPLY1_331(s_t* A, const s_t* B, const s_t* C)
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

void cullPoints(int n, s_t p[], int m, int i0, int iret[])
{
  // compute the centroid of the polygon in cx,cy
  int i, j;
  s_t a, cx, cy, q;
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
  s_t A[8];
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
    a = s_t(j) * (2 * math::constantsd::pi() / m) + A[i0];
    if (a > math::constantsd::pi())
      a -= 2 * math::constantsd::pi();
    s_t maxdiff = 1e9, diff;
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

/// pa: a point on line A
/// ua: a unit vector in the direction of line A
/// pb: a point on line B
/// ub: a unit vector in the direction of B
void dLineClosestApproach(
    const dVector3 pa,
    const dVector3 ua,
    const dVector3 pb,
    const dVector3 ub,
    s_t* alpha,
    s_t* beta)
{
  dVector3 p;
  p[0] = pb[0] - pa[0];
  p[1] = pb[1] - pa[1];
  p[2] = pb[2] - pa[2];
  s_t uaub = Inner(ua, ub);
  s_t q1 = Inner(ua, p);
  s_t q2 = -Inner(ub, p);
  s_t d = 1 - uaub * uaub;
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
    const Eigen::Vector3s& pa,
    const Eigen::Vector3s& ua,
    const Eigen::Vector3s& pb,
    const Eigen::Vector3s& ub,
    s_t* alpha,
    s_t* beta)
{
  Eigen::Vector3s u = pb - pa;
  Eigen::Vector3s v = ub - ua;
  Eigen::Vector3s w = pa - ua;
  s_t a = u.dot(u); // always >= 0
  s_t b = u.dot(v);
  s_t c = v.dot(v); // always >= 0
  s_t d = u.dot(w);
  s_t e = v.dot(w);
  s_t D = a * c - b * b; // always >= 0
  s_t sN, sD = D;        // sc = sN / sD, default sD = D >= 0
  s_t tN, tD = D;        // tc = tN / tD, default tD = D >= 0

  const s_t SMALL_NUM = 1e-15;

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
s_t dDistPointToSegment(
    const Eigen::Vector3s& p,
    const Eigen::Vector3s& ua,
    const Eigen::Vector3s& ub,
    s_t* alpha)
{
  Eigen::Vector3s v = ub - ua;
  Eigen::Vector3s w = p - ua;

  s_t c1 = w.dot(v);
  if (c1 <= 0)
  {
    *alpha = 0;
    return (p - ua).norm();
  }

  s_t c2 = v.dot(v);
  if (c2 <= c1)
  {
    *alpha = 1;
    return (p - ub).norm();
  }

  *alpha = c1 / c2;
  Eigen::Vector3s Pb = ua + *alpha * v;
  return (p - Pb).norm();
}

/*
/// This method intersects a rectangle with an arbitrary quad.
///
/// The rectangle is centered at (0,0) and extends along h[0] in the X axis and
/// h[1] in the Y axis. The quad in encoded in p, and consists of 4 points at
/// (p[0], p[1]), (p[1], p[2]), etc.
///
/// The intersection points are stored in ret, at (ret[0], ret[1]), (ret[2],
/// ret[3]), etc. The number of intersection points is the return value of this
/// function.
int intersectRectQuad(
    s_t rectDimensions[2], s_t quadPoints[8], s_t ret[16])
{
  // q (and r) contain nq (and nr) coordinate points for the current (and
  // chopped) polygons
  int srcPointsNum = 4;
  int dstPointsNum = 0;

  s_t retBuffer[16];

  s_t* srcPoints = quadPoints;
  s_t* dstPoints = ret;

  // Iterate over X (dir == 0) and Y (dir == 1) axis
  for (int dir = 0; dir <= 1; dir++)
  {
    // Iterate over -dir and +dir, to get all 4 sides of our rectangle
    for (int sign = -1; sign <= 1; sign += 2)
    {
      // Inside this loop we're comparing all the `srcPoints` to the barrier
      // we've set up. If a point is beyond (rectDimensions[dir] * sign), then
      // it's out of bounds and needs to be clipped. If an edge between two
      // subsequent points on the quad crosses the barrier, then we need to
      // introduce an edge-edge contact at that clipping point. Otherwise, we
      // can keep the all points inside our barrier.

      s_t* srcPointsCursor = srcPoints;
      s_t* dstPointsCursor = dstPoints;
      dstPointsNum = 0;

#define INCREMENT_DST_POINTS_CURSOR                                            \
  dstPointsCursor += 2;                                                        \
  dstPointsNum++;                                                              \
  if (dstPointsNum & 8)                                                        \
  {                                                                            \
    srcPoints = dstPoints;                                                     \
    goto done;                                                                 \
  }

      // This loop increments the `srcPointsCursor` at the end
      for (int i = srcPointsNum; i > 0; i--)
      {
        // Check 1: If this point is within our boundary along `dir`, then we
        // can keep it
        if (sign * srcPointsCursor[dir] < rectDimensions[dir])
        {
          // this point is inside the chopping line
          dstPointsCursor[0] = srcPointsCursor[0];
          dstPointsCursor[1] = srcPointsCursor[1];
          INCREMENT_DST_POINTS_CURSOR
        }
        // Check 2: If the edge from this point to the next point crosses our
        // boundary, introduce a point at the crossing point
        s_t* nextSrcPointsCursor = (i > 1) ? srcPointsCursor + 2 : srcPoints;
        if ((sign * srcPointsCursor[dir] < rectDimensions[dir])
            ^ (sign * nextSrcPointsCursor[dir] < rectDimensions[dir]))
        {
          // this line crosses the chopping line
          dstPointsCursor[1 - dir]
              = srcPointsCursor[1 - dir]
                + (nextSrcPointsCursor[1 - dir] - srcPointsCursor[1 - dir])
                      / (nextSrcPointsCursor[dir] - srcPointsCursor[dir])
                      * (sign * rectDimensions[dir] - srcPointsCursor[dir]);
          dstPointsCursor[dir] = sign * rectDimensions[dir];
          INCREMENT_DST_POINTS_CURSOR
        }

        // Increment the src pointer
        srcPointsCursor += 2;
      }

#undef INCREMENT_DST_POINTS_CURSOR

      // Swap `dst` into `src`
      srcPoints = dstPoints;
      srcPointsNum = dstPointsNum;

      // Point `dst` at whatever unused buffers we've got
      dstPoints = (srcPoints == ret) ? retBuffer : ret;
    }
  }
done:
  if (srcPoints != ret)
    memcpy(ret, srcPoints, dstPointsNum * 2 * sizeof(s_t));
  return dstPointsNum;
}
*/

// See the commented version above for more explanation. Should be equivalent,
// but don't want to rock the boat with changes right now. TODO(keenon): Swap in
// the commented version for clarity.
int intersectRectQuad(s_t h[2], s_t p[8], s_t ret[16])
{
  // q (and r) contain nq (and nr) coordinate points for the current (and
  // chopped) polygons
  int nq = 4, nr = 0;
  s_t buffer[16];
  s_t* q = p;
  s_t* r = ret;
  for (int dir = 0; dir <= 1; dir++)
  {
    // direction notation: xy[0] = x axis, xy[1] = y axis
    for (int sign = -1; sign <= 1; sign += 2)
    {
      // chop q along the line xy[dir] = sign*h[dir]
      s_t* pq = q;
      s_t* pr = r;
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
        s_t* nextq = (i > 1) ? pq + 2 : q;
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
  {
    // memcpy(ret, q, nr * 2 * sizeof(s_t));
    for (int i = 0; i < nr * 2; i++)
    {
      ret[i] = q[i];
    }
  }
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
  s_t h[3];
  h[0] = side[0];
  h[1] = side[1];
  h[2] = side[2];

  // region is -1,0,+1 depending on which side of the box planes each
  // coordinate is on. tanchor is the next t value at which there is a
  // transition, or the last one if there are no more.
  int region[3];
  s_t tanchor[3];

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
  s_t t = 0;
  s_t dd2dt = 0;
  for (i = 0; i < 3; i++)
    dd2dt -= (region[i] ? v2[i] : 0) * tanchor[i];
  if (dd2dt >= 0)
    goto got_answer;

  do
  {
    // find the point on the line that is at the next clip plane boundary
    s_t next_t = 1;
    for (i = 0; i < 3; i++)
    {
      if (tanchor[i] > t && tanchor[i] < 1 && tanchor[i] < next_t)
        next_t = tanchor[i];
    }

    // compute d|d|^2/dt for the next t
    s_t next_dd2dt = 0;
    for (i = 0; i < 3; i++)
    {
      next_dd2dt += (region[i] ? v2[i] : 0) * (next_t - tanchor[i]);
    }

    // if the sign of d|d|^2/dt has changed, solution = the crossover point
    if (next_dd2dt >= 0)
    {
      s_t m = (next_dd2dt - dd2dt) / (next_t - t);
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
    const CollisionOption& option,
    CollisionResult& result)
{
  const s_t fudge_factor = 1.05;
  dVector3 p, pp, normalC = {0.0, 0.0, 0.0, 0.0};
  const s_t* normalR = 0;
  s_t A[3], B[3], R11, R12, R13, R21, R22, R23, R31, R32, R33, Q11, Q12, Q13,
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

  Q11 = abs(R11);
  Q12 = abs(R12);
  Q13 = abs(R13);
  Q21 = abs(R21);
  Q22 = abs(R22);
  Q23 = abs(R23);
  Q31 = abs(R31);
  Q32 = abs(R32);
  Q33 = abs(R33);

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
  s2 = abs(expr1) - (expr2);                                                   \
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
  s2 = abs(expr1) - (expr2);                                                   \
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

  Eigen::Vector3s normal;
  Eigen::Vector3s point_vec;
  s_t penetration;

  if (normalR)
  {
    normal << normalR[0], normalR[4], normalR[8];
  }
  else
  {
    normal << Inner((R1), (normalC)), Inner((R1 + 4), (normalC)),
        Inner((R1 + 8), (normalC));
    normal.normalize();
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
    s_t sign;
    for (i = 0; i < 3; i++)
      pa[i] = p1[i];
    for (j = 0; j < 3; j++)
    {
#define TEMP_INNER14(a, b) (a[0] * (b)[0] + a[1] * (b)[4] + a[2] * (b)[8])
      s_t val = TEMP_INNER14(normal, R1 + j);
      // we want to do val > 0, but there's numerical issues when normal is
      // perpendicular to R1.col(j), so add a very small negative buffer to keep
      // things stable for finite differencing
      sign = (val > -1e-10) ? 1.0 : -1.0;

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
      s_t val = TEMP_INNER14(normal, R2 + j);
      // we want to do val > 0, but there's numerical issues when normal is
      // perpendicular to R1.col(j), so add a very small negative buffer to keep
      // things stable for finite differencing
      sign = (val > -1e-3) ? -1.0 : 1.0;
#undef TEMP_INNER14
      for (i = 0; i < 3; i++)
        pb[i] += sign * B[j] * R2[i * 4 + j];
    }

    s_t alpha, beta;
    dVector3 ua, ub;
    for (i = 0; i < 3; i++)
      ua[i] = R1[((code)-7) / 3 + i * 4];
    for (i = 0; i < 3; i++)
      ub[i] = R2[((code)-7) % 3 + i * 4];

    dLineClosestApproach(pa, ua, pb, ub, &alpha, &beta);
    Eigen::Vector3s edgeAFixedPoint = Eigen::Vector3s(pa[0], pa[1], pa[2]);
    Eigen::Vector3s edgeBFixedPoint = Eigen::Vector3s(pb[0], pb[1], pb[2]);

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
      if (penetration > option.contactClippingDepth)
        return 0;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = point_vec;
      contact.normal = normal * -1;
      contact.penetrationDepth = penetration;
      contact.type = ContactType::EDGE_EDGE;
      contact.edgeAClosestPoint = Eigen::Vector3s(pa[0], pa[1], pa[2]);
      contact.edgeAFixedPoint = edgeAFixedPoint;
      contact.edgeADir = Eigen::Vector3s(ua[0], ua[1], ua[2]).normalized();
      contact.edgeBClosestPoint = Eigen::Vector3s(pb[0], pb[1], pb[2]);
      contact.edgeBFixedPoint = edgeBFixedPoint;
      contact.edgeBDir = Eigen::Vector3s(ub[0], ub[1], ub[2]).normalized();
      result.addContact(contact);
    }
    return 1;
  }

  // okay, we have a face-something intersection (because the separating
  // axis is perpendicular to a face). define face 'a' to be the reference
  // face (i.e. the normal vector is perpendicular to this) and face 'b' to be
  // the incident face (the closest face of the other box).

  const s_t *Ra, *Rb, *pa, *pb, *Sa, *Sb;
  bool flipGradientMetadataOrder = false;
  if (code <= 3)
  {
    Ra = R1;
    Rb = R2;
    pa = p1;
    pb = p2;
    Sa = A;
    Sb = B;
    flipGradientMetadataOrder = false;
  }
  else
  {
    Ra = R2;
    Rb = R1;
    pa = p2;
    pb = p1;
    Sa = B;
    Sb = A;
    flipGradientMetadataOrder = true;
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
  s_t quad[8]; // 2D coordinate of incident face (x,y pairs)
  s_t c1, c2, m11, m12, m21, m22;
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
    s_t k1 = m11 * Sb[a1];
    s_t k2 = m21 * Sb[a1];
    s_t k3 = m12 * Sb[a2];
    s_t k4 = m22 * Sb[a2];
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
  s_t rect[2];
  rect[0] = Sa[code1];
  rect[1] = Sa[code2];

  // intersect the incident and reference faces
  s_t ret[16];
  int n = intersectRectQuad(rect, quad, ret);
  if (n < 1)
    return 0; // this should never happen

  // convert the intersection points into reference-face coordinates,
  // and compute the contact position and depth for each point. only keep
  // those points that have a positive (penetrating) depth. delete points in
  // the 'ret' array as necessary so that 'point' and 'ret' correspond.
  // real point[3*8];		// penetrating contact points
  s_t point[24]; // penetrating contact points
  s_t dep[8];    // depths for those points
  s_t det1 = dRecip(m11 * m22 - m12 * m21);
  m11 *= det1;
  m12 *= det1;
  m21 *= det1;
  m22 *= det1;
  int cnum = 0; // number of penetrating contact points found
  for (j = 0; j < n; j++)
  {
    s_t k1 = m22 * (ret[j * 2] - c1) - m12 * (ret[j * 2 + 1] - c2);
    s_t k2 = -m21 * (ret[j * 2] - c1) + m11 * (ret[j * 2 + 1] - c2);
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

  // Compute the normal of the incident face (because our current normal is wrt
  // the reference face). We may need to use the incident face normal for some
  // of our contacts.
  Eigen::Vector3s otherNormal
      = Eigen::Vector3s(Rb[lanr], Rb[4 + lanr], Rb[8 + lanr]);
  if (otherNormal.dot(normal) < 0)
    otherNormal *= -1;

  // These are the two basis vectors that are mutually perpendicular to the
  // `otherNormal` for the incident box.
  Eigen::Vector3s ortho1 = Eigen::Vector3s(Rb[a1], Rb[4 + a1], Rb[8 + a1]);
  Eigen::Vector3s ortho2 = Eigen::Vector3s(Rb[a2], Rb[4 + a2], Rb[8 + a2]);
  Eigen::Vector3s centerB = Eigen::Vector3s(pb[0], pb[1], pb[2]);
  Eigen::Vector3s faceCenter = centerB - Sb[lanr] * otherNormal;

  // we have less contacts than we need, so we use them all
  for (j = 0; j < cnum; j++)
  {
    point_vec << point[j * 3 + 0] + pa[0], point[j * 3 + 1] + pa[1],
        point[j * 3 + 2] + pa[2];

    s_t* point2dBuf = ret + (j * 2);
    s_t x = point2dBuf[0];
    s_t y = point2dBuf[1];
    s_t rectWidthX = rect[0];
    s_t rectWidthY = rect[1];

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = point_vec;
    contact.normal = normal * -1;
    contact.penetrationDepth = dep[j];

    bool onEdgeX = abs(x) == rectWidthX;
    bool onEdgeY = abs(y) == rectWidthY;

    if (onEdgeX && onEdgeY)
    {

      // point_vec at this point lays on our reference face. Since we've
      // determined at this point that this is a vertex from another box
      // colliding with our reference face, we need to move the collision point
      // to the exact depth of the contact vertex.

      // We're at a corner of our reference rectangle
      if (flipGradientMetadataOrder)
      {
        contact.type = ContactType::FACE_VERTEX;
        contact.point += contact.normal * contact.penetrationDepth;
      }
      else
      {
        contact.type = ContactType::VERTEX_FACE;
        contact.point -= contact.normal * contact.penetrationDepth;
        // Use the normal from the other face
        // contact.normal = otherNormal;
      }
    }
    else if (!onEdgeX && !onEdgeY)
    {
      // We're inside our reference rectangle
      if (flipGradientMetadataOrder)
      {
        contact.type = ContactType::VERTEX_FACE;
      }
      else
      {
        contact.type = ContactType::FACE_VERTEX;
        // Use the normal from the other face
        // contact.normal = otherNormal;
      }
    }
    else
    {
      // We're on an edge, but not at a corner.
      contact.type = ContactType::EDGE_EDGE;

      // Get the nearest corner, as a fixed point for our edge
      s_t faceX = x > 0 ? rectWidthX : -rectWidthX;
      s_t faceY = y > 0 ? rectWidthY : -rectWidthY;
      Eigen::Vector3s centerA = Eigen::Vector3s(pa[0], pa[1], pa[2]);
      Eigen::Vector3s faceCenterA = centerA + Sa[codeN] * normal;
      Eigen::Vector3s ortho1A
          = Eigen::Vector3s(Ra[code1], Ra[4 + code1], Ra[8 + code1]);
      Eigen::Vector3s ortho2A
          = Eigen::Vector3s(Ra[code2], Ra[4 + code2], Ra[8 + code2]);

      contact.edgeAFixedPoint = faceCenterA + faceX * ortho1A + faceY * ortho2A;
      contact.edgeADir = (contact.point - contact.edgeAFixedPoint).normalized();
      contact.edgeAClosestPoint = contact.point;

      // Map the point into relative space on the incident face, rather than the
      // reference face
      s_t incidentFaceX = ortho1.dot(contact.point) - ortho1.dot(centerB);
      s_t incidentFaceY = ortho2.dot(contact.point) - ortho2.dot(centerB);
      s_t signX
          = incidentFaceX == 0 ? 1.0 : (incidentFaceX / abs(incidentFaceX));
      s_t signY
          = incidentFaceY == 0 ? 1.0 : (incidentFaceY / abs(incidentFaceY));
      Eigen::Vector3s nearestBCorner
          = (signX * Sb[a1]) * ortho1 + (signY * Sb[a2]) * ortho2 + faceCenter;

      s_t distX = abs(abs(incidentFaceX) - Sb[a1]);
      s_t distY = abs(abs(incidentFaceY) - Sb[a2]);

      Eigen::Vector3s otherBCorner;
      if (distX < distY)
      {
        // If we're on the x-edge, then grab the corner on the opposite Y
        otherBCorner = (signX * Sb[a1]) * ortho1
                       + (-1 * signY * Sb[a2]) * ortho2 + faceCenter;
      }
      else
      {
        // If we're on the y-edge, then grab the corner on the opposite X
        otherBCorner = (-1 * signX * Sb[a1]) * ortho1
                       + (signY * Sb[a2]) * ortho2 + faceCenter;
      }

      contact.edgeBDir = (nearestBCorner - otherBCorner).normalized();
      contact.edgeBFixedPoint = nearestBCorner;

      // Flip the order of the metadata.
      if (flipGradientMetadataOrder)
      {
        Eigen::Vector3s buf = contact.edgeADir;
        contact.edgeADir = contact.edgeBDir;
        contact.edgeBDir = buf;
        buf = contact.edgeAFixedPoint;
        contact.edgeAFixedPoint = contact.edgeBFixedPoint;
        contact.edgeBFixedPoint = buf;
      }
    }

    result.addContact(contact);
  }

  // TODO: We once limited the number of contact points to 4. This introduces
  // problems when we're finite differencing, because we can get
  // discontinuities. If we have more than 4 contacts at an equal depth, and we
  // rotate the contact by any tiny EPS, we can get different contact points.

  /*
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
    s_t maxdepth = dep[0];
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
  */
  return cnum;
}

int collideBoxBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
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

  return dBoxBox(o1, o2, p0, R0, halfSize0, p1, R1, halfSize1, option, result);
}

int collideBoxSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    const s_t& r1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace halfspace)
{
  Eigen::Vector3s halfSize = 0.5 * size0;
  bool inside_box = true;

  // clipping a center of the sphere to a boundary of the box
  // Vec3 c0(&T0[9]);
  Eigen::Vector3s c0 = T1.translation();
  Eigen::Vector3s p = T0.inverse() * c0;

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

  Eigen::Vector3s normal(0.0, 0.0, 0.0);
  s_t penetration;

  if (inside_box)
  {
    // find nearest side from the sphere center
    s_t min = halfSize[0] - abs(p[0]);
    s_t tmin = halfSize[1] - abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - abs(p[2]);
    if (tmin < min)
    {
      min = tmin;
      idx = 2;
    }

    // normal[idx] = (p[idx] > 0.0 ? 1.0 : -1.0);
    normal[idx] = (p[idx] > 0.0 ? -1.0 : 1.0);
    normal = T0.linear() * normal;
    penetration = min + r1;
    if (penetration > option.contactClippingDepth)
      return 0;

    // In this special case, it actually behaves as though it's just a raw
    // vertex-face collision for gradients, so don't reinvent the wheel
    contact.type = FACE_VERTEX;
    contact.point = c0;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
    return 1;
  }

  Eigen::Vector3s contactpt = T0 * p;
  // normal = c0 - contactpt;
  normal = contactpt - c0;
  s_t mag = normal.norm();
  penetration = r1 - mag;
  if (penetration > option.contactClippingDepth)
    return 0;

  // Enforce halfspace clipping on the sphere, if we're not in the BOTH setting
  if (halfspace == ClipSphereHalfspace::BOTTOM
      && (T1.inverse() * contactpt)(2) >= 0)
  {
    return 0;
  }
  if (halfspace == ClipSphereHalfspace::TOP
      && (T1.inverse() * contactpt)(2) <= 0)
  {
    return 0;
  }

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
    s_t min = halfSize[0] - abs(p[0]);
    s_t tmin = halfSize[1] - abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - abs(p[2]);
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
    const s_t& r0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace /* halfspace */)
{
  Eigen::Vector3s halfSize = 0.5 * size1;
  bool inside_box = true;

  // clipping a center of the sphere to a boundary of the box
  Eigen::Vector3s c0 = T0.translation();
  Eigen::Vector3s p = T1.inverse() * c0;

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

  Eigen::Vector3s normal(0.0, 0.0, 0.0);
  s_t penetration;

  if (inside_box)
  {
    // find nearest side from the sphere center
    s_t min = halfSize[0] - abs(p[0]);
    s_t tmin = halfSize[1] - abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - abs(p[2]);
    if (tmin < min)
    {
      min = tmin;
      idx = 2;
    }

    normal[idx] = (p[idx] > 0.0 ? 1.0 : -1.0);
    normal = T1.linear() * normal;
    penetration = min + r0;
    if (penetration > option.contactClippingDepth)
      return 0;

    // In this special case, it actually behaves as though it's just a raw
    // vertex-face collision for gradients, so don't reinvent the wheel
    contact.type = VERTEX_FACE;
    contact.point = c0;
    contact.normal = normal;
    contact.penetrationDepth = penetration;
    result.addContact(contact);
    return 1;
  }

  Eigen::Vector3s contactpt = T1 * p;
  normal = c0 - contactpt;
  s_t mag = normal.norm();
  penetration = r0 - mag;
  if (penetration > option.contactClippingDepth)
    return 0;

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
    s_t min = halfSize[0] - abs(p[0]);
    s_t tmin = halfSize[1] - abs(p[1]);
    int idx = 0;

    if (tmin < min)
    {
      min = tmin;
      idx = 1;
    }
    tmin = halfSize[2] - abs(p[2]);
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
    const s_t& _r0,
    const Eigen::Isometry3s& c0,
    const s_t& _r1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace /* halfspace0 */,
    ClipSphereHalfspace /* halfspace1 */)
{
  s_t r0 = _r0;
  s_t r1 = _r1;
  s_t rsum = r0 + r1;
  Eigen::Vector3s normal = c0.translation() - c1.translation();
  s_t normal_sqr = normal.squaredNorm();

  if (normal_sqr > rsum * rsum)
  {
    return 0;
  }

  r0 /= rsum;
  r1 /= rsum;

  Eigen::Vector3s point = r1 * c0.translation() + r0 * c1.translation();
  s_t penetration;

  if (normal_sqr < DART_COLLISION_EPS)
  {
    normal.setZero();
    penetration = rsum;
    if (penetration > option.contactClippingDepth)
      return 0;

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
  if (penetration > option.contactClippingDepth)
    return 0;

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
  Eigen::Vector3s dir;
  dir(0) = static_cast<s_t>(_dir->v[0]);
  dir(1) = static_cast<s_t>(_dir->v[1]);
  dir(2) = static_cast<s_t>(_dir->v[2]);

  // apply rotation on direction vector
  Eigen::Vector3s localDir = box->transform->linear().transpose() * dir;

  // compute support point in specified direction (in box coordinates)
  Eigen::Vector3s clipped = Eigen::Vector3s(
      ccdSign(localDir(0)) * (*box->size)(0) * CCD_REAL(0.5),
      ccdSign(localDir(1)) * (*box->size)(1) * CCD_REAL(0.5),
      ccdSign(localDir(2)) * (*box->size)(2) * CCD_REAL(0.5));

  // transform support point according to position and rotation of object
  Eigen::Vector3s out = *(box->transform) * clipped;
  _out->v[0] = static_cast<ccd_real_t>(out(0));
  _out->v[1] = static_cast<ccd_real_t>(out(1));
  _out->v[2] = static_cast<ccd_real_t>(out(2));
}

/// libccd support function for a sphere
void ccdSupportSphere(
    const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out)
{
  ccdSphere* sphere = (ccdSphere*)_obj;

  Eigen::Vector3s dir;
  dir(0) = static_cast<s_t>(_dir->v[0]);
  dir(1) = static_cast<s_t>(_dir->v[1]);
  dir(2) = static_cast<s_t>(_dir->v[2]);

  // apply rotation on direction vector
  Eigen::Vector3s localDir = sphere->transform->linear().transpose() * dir;
  localDir *= sphere->radius / localDir.norm();

  // transform support point according to position and rotation of object
  Eigen::Vector3s out = *(sphere->transform) * localDir;
  _out->v[0] = static_cast<ccd_real_t>(out(0));
  _out->v[1] = static_cast<ccd_real_t>(out(1));
  _out->v[2] = static_cast<ccd_real_t>(out(2));
}

/// libccd support function for a mesh
void ccdSupportMesh(const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out)
{
  ccdMesh* mesh = (ccdMesh*)_obj;

  Eigen::Vector3s dir;
  dir(0) = static_cast<s_t>(_dir->v[0]);
  dir(1) = static_cast<s_t>(_dir->v[1]);
  dir(2) = static_cast<s_t>(_dir->v[2]);

  // apply rotation on direction vector
  Eigen::Vector3s localDir = mesh->transform->linear().transpose() * dir;
  localDir(0) /= (*mesh->scale)(0);
  localDir(1) /= (*mesh->scale)(1);
  localDir(2) /= (*mesh->scale)(2);

  s_t maxDot = -std::numeric_limits<s_t>::infinity();
  Eigen::Vector3s maxDotPoint = Eigen::Vector3s::Zero();

  for (int i = 0; i < mesh->mesh->mNumMeshes; i++)
  {
    aiMesh* m = mesh->mesh->mMeshes[i];
    for (int k = 0; k < m->mNumVertices; k++)
    {
      s_t dot = m->mVertices[k].x * localDir(0)
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
  Eigen::Vector3s out = *(mesh->transform) * maxDotPoint;
  _out->v[0] = static_cast<ccd_real_t>(out(0));
  _out->v[1] = static_cast<ccd_real_t>(out(1));
  _out->v[2] = static_cast<ccd_real_t>(out(2));
}

/// libccd support function for a capsule
void ccdSupportCapsule(
    const void* _obj, const ccd_vec3_t* _dir, ccd_vec3_t* _out)
{
  ccdCapsule* capsule = (ccdCapsule*)_obj;

  Eigen::Vector3s dir;
  dir(0) = static_cast<s_t>(_dir->v[0]);
  dir(1) = static_cast<s_t>(_dir->v[1]);
  dir(2) = static_cast<s_t>(_dir->v[2]);

  // apply rotation on direction vector
  Eigen::Vector3s localDir = capsule->transform->linear().transpose() * dir;
  localDir.normalize();
  localDir *= capsule->radius;

  Eigen::Vector3s out;
  if (abs(localDir(2)) < 1e-10)
  {
    out = *(capsule->transform) * localDir;
  }
  else if (localDir(2) > 0)
  {
    out = *(capsule->transform)
          * (localDir + Eigen::Vector3s(0, 0, capsule->height / 2));
  }
  else if (localDir(2) < 0)
  {
    out = *(capsule->transform)
          * (localDir + Eigen::Vector3s(0, 0, -capsule->height / 2));
  }
  else
  {
    assert(false && "This should be impossible to read");
    out.setZero();
  }
  _out->v[0] = static_cast<ccd_real_t>(out(0));
  _out->v[1] = static_cast<ccd_real_t>(out(1));
  _out->v[2] = static_cast<ccd_real_t>(out(2));
}

/// libccd support function for a box
void ccdCenterBox(const void* _obj, ccd_vec3_t* _center)
{
  ccdBox* box = (ccdBox*)_obj;
  _center->v[0] = static_cast<ccd_real_t>(box->transform->translation()(0));
  _center->v[1] = static_cast<ccd_real_t>(box->transform->translation()(1));
  _center->v[2] = static_cast<ccd_real_t>(box->transform->translation()(2));
}

/// libccd support function for a sphere
void ccdCenterSphere(const void* _obj, ccd_vec3_t* _center)
{
  ccdSphere* sphere = (ccdSphere*)_obj;
  _center->v[0] = static_cast<ccd_real_t>(sphere->transform->translation()(0));
  _center->v[1] = static_cast<ccd_real_t>(sphere->transform->translation()(1));
  _center->v[2] = static_cast<ccd_real_t>(sphere->transform->translation()(2));
}

/// libccd support function for a mesh
void ccdCenterMesh(const void* _obj, ccd_vec3_t* _center)
{
  ccdMesh* mesh = (ccdMesh*)_obj;
  _center->v[0] = static_cast<ccd_real_t>(mesh->transform->translation()(0));
  _center->v[1] = static_cast<ccd_real_t>(mesh->transform->translation()(1));
  _center->v[2] = static_cast<ccd_real_t>(mesh->transform->translation()(2));
}

/// libccd support function for a capsule
void ccdCenterCapsule(const void* _obj, ccd_vec3_t* _center)
{
  ccdCapsule* capsule = (ccdCapsule*)_obj;
  _center->v[0] = static_cast<ccd_real_t>(capsule->transform->translation()(0));
  _center->v[1] = static_cast<ccd_real_t>(capsule->transform->translation()(1));
  _center->v[2] = static_cast<ccd_real_t>(capsule->transform->translation()(2));
}

/// Find all the vertices within epsilon of lying on the witness plane
std::vector<Eigen::Vector3s> ccdPointsAtWitnessBox(
    ccdBox* box, ccd_vec3_t* _dir, bool neg)
{
  Eigen::Vector3s dir;
  dir(0) = static_cast<s_t>(_dir->v[0]);
  dir(1) = static_cast<s_t>(_dir->v[1]);
  dir(2) = static_cast<s_t>(_dir->v[2]);

  // apply rotation on direction vector
  Eigen::Vector3s localDir = box->transform->linear().transpose() * dir;

  std::vector<Eigen::Vector3s> localPoints;

  std::vector<s_t> boundsX;
  std::vector<s_t> boundsY;
  std::vector<s_t> boundsZ;

  boundsX.push_back((*box->size)(0) * 0.5);
  boundsX.push_back((*box->size)(0) * -0.5);
  boundsY.push_back((*box->size)(1) * 0.5);
  boundsY.push_back((*box->size)(1) * -0.5);
  boundsZ.push_back((*box->size)(2) * 0.5);
  boundsZ.push_back((*box->size)(2) * -0.5);

  for (s_t x : boundsX)
  {
    for (s_t y : boundsY)
    {
      for (s_t z : boundsZ)
      {
        localPoints.push_back(Eigen::Vector3s(x, y, z));
      }
    }
  }

  s_t negMult = neg ? -1 : 1;

  s_t maxDot = -1 * std::numeric_limits<s_t>::infinity();
  for (Eigen::Vector3s& localPoint : localPoints)
  {
    s_t dot = negMult * localPoint.dot(localDir);
    if (dot > maxDot)
      maxDot = dot;
  }

  std::vector<Eigen::Vector3s> points;
  for (Eigen::Vector3s& localPoint : localPoints)
  {
    s_t dot = negMult * localPoint.dot(localDir);
    if (maxDot - dot < DART_COLLISION_WITNESS_PLANE_DEPTH)
    {
      points.push_back((*box->transform) * localPoint);
    }
  }

  return points;
}

/// Find all the vertices within epsilon of lying on the witness plane
std::vector<Eigen::Vector3s> ccdPointsAtWitnessMesh(
    ccdMesh* mesh, ccd_vec3_t* _dir, bool neg)
{
  Eigen::Vector3s dir;
  dir(0) = static_cast<s_t>(_dir->v[0]);
  dir(1) = static_cast<s_t>(_dir->v[1]);
  dir(2) = static_cast<s_t>(_dir->v[2]);

  // apply rotation on direction vector
  Eigen::Vector3s localDir = mesh->transform->linear().transpose() * dir;
  localDir(0) /= (*mesh->scale)(0);
  localDir(1) /= (*mesh->scale)(1);
  localDir(2) /= (*mesh->scale)(2);

  std::vector<Eigen::Vector3s> points;

  s_t maxDot = (neg ? 1 : -1) * std::numeric_limits<s_t>::infinity();

  // 1. Find the max dot
  for (int i = 0; i < mesh->mesh->mNumMeshes; i++)
  {
    aiMesh* m = mesh->mesh->mMeshes[i];
    for (int k = 0; k < m->mNumVertices; k++)
    {
      s_t dot = m->mVertices[k].x * localDir(0) * (*mesh->scale)(0)
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
      s_t dot = m->mVertices[k].x * localDir(0) * (*mesh->scale)(0)
                    * (*mesh->scale)(0)
                + m->mVertices[k].y * localDir(1) * (*mesh->scale)(1)
                      * (*mesh->scale)(1)
                + m->mVertices[k].z * localDir(2) * (*mesh->scale)(2)
                      * (*mesh->scale)(2);
      // If we're on the witness plane with our "maxDot" vector, then add us to
      // the list
      if (abs(dot - maxDot) < DART_COLLISION_WITNESS_PLANE_DEPTH)
      {
        Eigen::Vector3s proposedPoint
            = *(mesh->transform)
              * Eigen::Vector3s(
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
    const std::vector<Eigen::Vector3s>& pointsAWitnessSorted,
    const std::vector<Eigen::Vector3s>& pointsBWitnessSorted,
    PinToFace pinToFace = PinToFace::AVERAGE)
{
  assert(pointsAWitnessSorted.size() > 2 || pointsBWitnessSorted.size() > 2);
  assert(pointsAWitnessSorted.size() >= 2 && pointsBWitnessSorted.size() >= 2);

  // Eigen::Map<Eigen::Vector3s> dirVec(dir->v);
  // Make a copy so we can see it in the debugger
  Eigen::Vector3s dirVec;
  dirVec(0) = static_cast<s_t>(dir->v[0]);
  dirVec(1) = static_cast<s_t>(dir->v[1]);
  dirVec(2) = static_cast<s_t>(dir->v[2]);

  // All the pointsAWitness vectors are co-planar, so we choose the closest
  // [0], [1], and [2] to cross to get a precise normal
  Eigen::Vector3s normalA
      = (pointsAWitnessSorted[0] - pointsAWitnessSorted[1])
            .cross(
                pointsAWitnessSorted[1]
                - (pointsAWitnessSorted.size() > 2 ? pointsAWitnessSorted[2]
                                                   : dirVec))
            .normalized();
  // Likewise for the pointsBWitness vectors
  Eigen::Vector3s normalB
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
                        > 0.2;
  bool bBroken = abs(normalB.squaredNorm() - 1) > 1e-10
                 || std::min(
                        (normalB - dirVec).squaredNorm(),
                        (-normalB - dirVec).squaredNorm())
                        > 0.2;
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
    normalA = -Eigen::Vector3s(dir->v[0], dir->v[1], dir->v[2]);
    normalB = normalA;
  }

  // Ensure that the normal is in the opposite direction as `dir`, so we're
  // still pointing from B to A.
  s_t normalADot = normalA(0) * dir->v[0] + normalA(1) * dir->v[1]
                   + normalA(2) * dir->v[2];
  if (normalADot > 0)
    normalA *= -1;
  s_t normalBDot = normalB(0) * dir->v[0] + normalB(1) * dir->v[1]
                   + normalB(2) * dir->v[2];
  if (normalBDot > 0)
    normalB *= -1;

  Eigen::Vector3s normal = ((normalA + normalB) / 2).normalized();

  // This will the origin for our 2D plane we're going to use to compute
  // collision geometry. We use different origins for object A and B, because
  // they're slightly offset from each other in space due to penetration
  // distance.
  Eigen::Vector3s originA = normal * (pointsAWitnessSorted[0].dot(normal));
  Eigen::Vector3s originB = normal * (pointsBWitnessSorted[0].dot(normal));

  Eigen::Vector3s origin = (originA + originB) / 2;
  if (pinToFace == PinToFace::FACE_A)
  {
    origin = originA;
    normal = normalA;
  }
  else if (pinToFace == PinToFace::FACE_B)
  {
    origin = originB;
    normal = normalB;
  }

  Eigen::Vector3s tmp = normal.cross(Eigen::Vector3s::UnitZ());
  if (tmp.squaredNorm() < 1e-4)
  {
    tmp = normal.cross(Eigen::Vector3s::UnitX());
  }

  // These are the basis for our 2D plane (with an origin at `origin2d` in 3D
  // space)
  Eigen::Vector3s basis2dX = normalA.cross(tmp);
  Eigen::Vector3s basis2dY = normalA.cross(basis2dX);

  // We have to be very careful to preserve the order of the points so that the
  // order of our contacts is preserved even under perturbations.
  std::vector<Eigen::Vector3s> pointsAConvexHull = pointsAWitnessSorted;
  keepOnlyConvex2DHull(pointsAConvexHull, origin, basis2dX, basis2dY);
  std::vector<Eigen::Vector3s> pointsAConvexSorted = pointsAConvexHull;
  prepareConvex2DShape(pointsAConvexSorted, origin, basis2dX, basis2dY);

  // We have to be very careful to preserve the order of the points so that the
  // order of our contacts is preserved even under perturbations.
  std::vector<Eigen::Vector3s> pointsBConvexHull = pointsBWitnessSorted;
  keepOnlyConvex2DHull(pointsBConvexHull, origin, basis2dX, basis2dY);
  std::vector<Eigen::Vector3s> pointsBConvexSorted = pointsBConvexHull;
  prepareConvex2DShape(pointsBConvexSorted, origin, basis2dX, basis2dY);

  int numContacts = 0;

  // All vertices that lie inside the other shape's convex hull are
  // vertex-face contacts.

  // Start with points from object A. We'll later do the symmetric thing for
  // object B.
  for (int i = 0; i < pointsAConvexHull.size(); i++)
  {
    Eigen::Vector3s vertexA = pointsAConvexHull[i];
    if (convex2DShapeContains(
            vertexA, pointsBConvexSorted, origin, basis2dX, basis2dY))
    {
      numContacts++;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = vertexA;
      if (pinToFace == PinToFace::FACE_B)
      {
        contact.point = originB + (basis2dX * basis2dX.dot(contact.point))
                        + (basis2dY * basis2dY.dot(contact.point));
      }

      // Make sure that the normal vector is pointed from B to A, which should
      // mean that distA > distB because A's vertex is penetrating B
      s_t distA = vertexA.dot(normalB);
      s_t distB = pointsBWitnessSorted[0].dot(normalB);
      if (pinToFace == PinToFace::FACE_B)
      {
        contact.normal = normalA;
      }
      else
      {
        contact.normal = normalB;
      }
      contact.penetrationDepth = distB - distA;

      contact.type = VERTEX_FACE;

      collisionsOut.push_back(contact);
    }
  }

  // Now we need to repeat analagous logic for the vertices in shape B
  for (int i = 0; i < pointsBConvexHull.size(); i++)
  {
    Eigen::Vector3s vertexB = pointsBConvexHull[i];
    if (convex2DShapeContains(
            vertexB, pointsAConvexSorted, origin, basis2dX, basis2dY))
    {
      numContacts++;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = vertexB;
      if (pinToFace == PinToFace::FACE_A)
      {
        contact.point = originA + (basis2dX * basis2dX.dot(contact.point))
                        + (basis2dY * basis2dY.dot(contact.point));
      }

      // Make sure that the normal vector is pointed from B to A, which should
      // mean that distB > distA because B's vertex is penetrating A
      s_t distA = pointsAWitnessSorted[0].dot(normalA);
      s_t distB = vertexB.dot(normalA);
      if (pinToFace == PinToFace::FACE_A)
      {
        contact.normal = normalB;
      }
      else
      {
        contact.normal = normalA;
      }
      contact.penetrationDepth = distB - distA;

      contact.type = FACE_VERTEX;

      collisionsOut.push_back(contact);
    }
  }

  // Now finally we check every pair of edges in shape A and shape B for
  // collisions.
  for (int i = 0; i < pointsAConvexSorted.size(); i++)
  {
    // Skip the last vertex loop-around if this is just a single edge, since we
    // don't want to count the same edge twice
    if (i == pointsAConvexSorted.size() - 1 && pointsAConvexSorted.size() == 2)
    {
      continue;
    }
    Eigen::Vector3s a1World = pointsAConvexSorted[i];
    Eigen::Vector3s a2World
        = pointsAConvexSorted[i == pointsAConvexSorted.size() - 1 ? 0 : i + 1];
    Eigen::Vector2s a1 = pointInPlane(a1World, origin, basis2dX, basis2dY);
    Eigen::Vector2s a2 = pointInPlane(a2World, origin, basis2dX, basis2dY);
    for (int j = 0; j < pointsBConvexSorted.size(); j++)
    {
      // Skip the last vertex loop-around if this is just a single edge, since
      // we don't want to count the same edge twice
      if (j == pointsBConvexSorted.size() - 1
          && pointsBConvexSorted.size() == 2)
      {
        continue;
      }
      Eigen::Vector3s b1World = pointsBConvexSorted[j];
      Eigen::Vector3s b2World = pointsBConvexSorted
          [j == pointsBConvexSorted.size() - 1 ? 0 : j + 1];
      Eigen::Vector2s b1 = pointInPlane(b1World, origin, basis2dX, basis2dY);
      Eigen::Vector2s b2 = pointInPlane(b2World, origin, basis2dX, basis2dY);

      Eigen::Vector2s out;
      if (get2DLineIntersection(a1, a2, b1, b2, out))
      {
        // We found an edge-edge collision at "out"!
        numContacts++;

        // Get the relevant points in 3D space
        Eigen::Vector3s edgeAClosestPoint
            = originA + out(0) * basis2dX + out(1) * basis2dY;
        Eigen::Vector3s edgeBClosestPoint
            = originB + out(0) * basis2dX + out(1) * basis2dY;

        Contact contact;
        contact.collisionObject1 = o1;
        contact.collisionObject2 = o2;
        contact.type = EDGE_EDGE;
        contact.edgeAClosestPoint = edgeAClosestPoint;
        contact.edgeAFixedPoint = a1World;
        contact.edgeADir = (a2World - a1World).normalized();
        contact.edgeBClosestPoint = edgeBClosestPoint;
        contact.edgeBFixedPoint = b1World;
        contact.edgeBDir = (b2World - b1World).normalized();

        // Construct the normal as exactly perpendicular to both edges.
        contact.normal = contact.edgeADir.cross(contact.edgeBDir);
        // Ensure contact.normal points in roughly the same direction as
        // normalA.
        if (contact.normal.dot(normalA) < 0)
        {
          contact.normal *= -1;
        }
        // Compute penetration depth
        s_t distA = contact.edgeAClosestPoint.dot(contact.normal);
        s_t distB = contact.edgeBClosestPoint.dot(contact.normal);
        contact.penetrationDepth = distB - distA;
        if (contact.penetrationDepth < 0)
        {
          contact.normal *= -1;
          contact.penetrationDepth *= -1;
        }

        // Construct our contact point using the standard geometry routine
        s_t radiusA = 1.0;
        s_t radiusB = 1.0;
        if (pinToFace == PinToFace::FACE_A)
        {
          radiusA = 0.0;
        }
        else if (pinToFace == PinToFace::FACE_B)
        {
          radiusB = 0.0;
        }
        contact.point = math::getContactPoint(
            contact.edgeAFixedPoint,
            contact.edgeADir,
            contact.edgeBFixedPoint,
            contact.edgeBDir,
            radiusA,
            radiusB);

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
    const std::vector<Eigen::Vector3s>& pointsAWitness,
    const std::vector<Eigen::Vector3s>& pointsBWitness)
{
  if (pointsAWitness.size() == 0 && pointsBWitness.size() == 0)
  {
    std::cout
        << "Attempting to create a mesh-mesh contact with no witness points!"
        << std::endl;
  }
  assert(pointsAWitness.size() > 0 && pointsBWitness.size() > 0);

  std::vector<Eigen::Vector3s> pointsAWitnessSorted = pointsAWitness;
  std::vector<Eigen::Vector3s> pointsBWitnessSorted = pointsBWitness;

  /*
  // `dir` points from A to B, so we want the highest dot product of dir at the
  // front of A
  std::sort(
      pointsAWitnessSorted.begin(),
      pointsAWitnessSorted.end(),
      [dir](Eigen::Vector3s& a, Eigen::Vector3s& b) {
        s_t aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
        s_t bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
        return aDot < bDot;
      });
  // `dir` points from A to B, so we want the lowest dot product of dir at the
  // front of B
  std::sort(
      pointsBWitnessSorted.begin(),
      pointsBWitnessSorted.end(),
      [dir](Eigen::Vector3s& a, Eigen::Vector3s& b) {
        s_t aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
        s_t bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
        return aDot > bDot;
      });
      */

  // Single vertex-face collision
  if (pointsAWitness.size() == 1 && pointsBWitness.size() > 2)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.point = pointsAWitness[0];

    // All the pointsBWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3s normal
        = (pointsBWitnessSorted[0] - pointsBWitnessSorted[1])
              .cross(pointsBWitnessSorted[1] - pointsBWitnessSorted[2])
              .normalized();

    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    s_t normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;
    contact.normal = normal;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distA > distB because A's vertex is penetrating B
    s_t distA = pointsAWitness[0].dot(normal);
    s_t distB = pointsBWitnessSorted[0].dot(normal);
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
    Eigen::Vector3s normal
        = (pointsAWitnessSorted[0] - pointsAWitnessSorted[1])
              .cross(pointsAWitnessSorted[1] - pointsAWitnessSorted[2])
              .normalized();

    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    s_t normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;
    contact.normal = normal;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distB > distA because B's vertex is penetrating A
    s_t distA = pointsAWitnessSorted[0].dot(normal);
    s_t distB = pointsBWitness[0].dot(normal);
    // Normal is fine as it is
    contact.penetrationDepth = abs(distA - distB);

    contact.type = FACE_VERTEX;
    result.addContact(contact);

    return 1;
  }
  // Single edge-edge collision
  else if (pointsAWitness.size() == 2 && pointsBWitness.size() == 2)
  {
    Eigen::Vector3s ua = (pointsAWitness[0] - pointsAWitness[1]).normalized();
    Eigen::Vector3s ub = (pointsBWitness[0] - pointsBWitness[1]).normalized();
    Eigen::Vector3s pa = pointsAWitness[0];
    Eigen::Vector3s pb = pointsBWitness[0];

    s_t alpha, beta;
    dLineClosestApproach(
        pa.data(), ua.data(), pb.data(), ub.data(), &alpha, &beta);

    // After this, pa and pb represent the closest point
    for (int i = 0; i < 3; i++)
      pa[i] += ua[i] * alpha;
    for (int i = 0; i < 3; i++)
      pb[i] += ub[i] * beta;

    {
      // This is the average of the closest point on the A edge and the B edge
      Eigen::Vector3s point = Eigen::Vector3s(
          0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2]));

      Eigen::Vector3s normal = ua.cross(ub);
      s_t normalDot = normal(0) * dir->v[0] + normal(1) * dir->v[1]
                      + normal(2) * dir->v[2];
      if (normalDot > 0)
        normal *= -1;

      Contact contact;
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.point = point;
      contact.type = ContactType::EDGE_EDGE;
      contact.edgeAClosestPoint = pa;
      contact.edgeAFixedPoint = pointsAWitness[0];
      contact.edgeADir = ua;
      contact.edgeBClosestPoint = pb;
      contact.edgeBFixedPoint = pointsBWitness[0];
      contact.edgeBDir = ub;
      contact.normal = normal;

      s_t distA = contact.edgeAClosestPoint.dot(normal);
      s_t distB = contact.edgeBClosestPoint.dot(normal);
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
    // TODO(keenon): This isn't always true when we have deep inter-penetration
    // assert(contacts.size() == 2);
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
    // TODO(keenon): This isn't always true when we have deep inter-penetration
    // assert(contacts.size() == 2);
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

    Eigen::Vector3s normal = Eigen::Vector3s(dir->v[0], dir->v[1], dir->v[2]);
    // Ensure the normal is orthogonal to edge B, at least
    Eigen::Vector3s edgeB
        = (pointsBWitness[0] - pointsBWitness[1]).normalized();
    normal -= normal.dot(edgeB) * edgeB;
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    s_t normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distA > distB because A's vertex is penetrating B
    s_t distA = pointsAWitness[0].dot(normal);
    s_t distB = pointsBWitness[0].dot(normal);
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

    Eigen::Vector3s normal = Eigen::Vector3s(dir->v[0], dir->v[1], dir->v[2]);
    // Ensure the normal is orthogonal to edge A, at least
    Eigen::Vector3s edgeA
        = (pointsAWitness[1] - pointsAWitness[1]).normalized();
    normal -= normal.dot(edgeA) * edgeA;
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    s_t normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distB > distA because B's vertex is penetrating A
    s_t distA = pointsAWitness[0].dot(normal);
    s_t distB = pointsBWitness[0].dot(normal);
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
    Eigen::Vector3s normal
        = Eigen::Vector3s(dir->v[0], dir->v[1], dir->v[2]) * -1;

    // Make sure that the normal vector is pointed from B to A, which should
    // mean that distB > distA because B's vertex is penetrating A
    s_t distA = pointsAWitness[0].dot(normal);
    s_t distB = pointsBWitness[0].dot(normal);
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
    const std::vector<Eigen::Vector3s>& meshPointsWitness,
    const Eigen::Vector3s& sphereCenter,
    s_t sphereRadius,
    ClipSphereHalfspace /* halfspace */)
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
    Eigen::Vector3s normal = (meshPointsWitness[0] - sphereCenter).normalized();
    Eigen::Vector3s contactPoint = meshPointsWitness[0];
    Contact contact;
    contact.point = contactPoint;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.vertexPoint = meshPointsWitness[0];
    contact.sphereRadius = sphereRadius;
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
    Eigen::Vector3s edge
        = (meshPointsWitness[1] - meshPointsWitness[0]).normalized();
    Eigen::Vector3s closestPoint
        = math::closestPointOnLine(meshPointsWitness[0], edge, sphereCenter);
    Eigen::Vector3s normal = (closestPoint - sphereCenter).normalized();
    Contact contact;
    contact.point = closestPoint;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.sphereRadius = sphereRadius;
    contact.type = EDGE_SPHERE;
    contact.edgeAFixedPoint = meshPointsWitness[0];
    contact.edgeAClosestPoint = closestPoint;
    contact.edgeADir = edge;
    contact.penetrationDepth
        = sphereRadius - (closestPoint - sphereCenter).norm();
    result.addContact(contact);
  }
  // face-sphere collision
  else if (meshPointsWitness.size() > 2)
  {
    std::vector<Eigen::Vector3s> pointsWitnessSorted = meshPointsWitness;
    /*
    // `dir` points from A to B, so we want the highest dot product of dir at
    // the front of A
    std::sort(
        pointsWitnessSorted.begin(),
        pointsWitnessSorted.end(),
        [dir](Eigen::Vector3s& a, Eigen::Vector3s& b) {
          s_t aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
          s_t bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
          return aDot < bDot;
        });
        */
    // All the meshPointsWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3s normal
        = (pointsWitnessSorted[0] - pointsWitnessSorted[1])
              .cross(pointsWitnessSorted[1] - pointsWitnessSorted[2])
              .normalized();
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    s_t normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // We want to find the furthest point on the surface of the sphere, which is
    // just whatever point is facing the normal

    Eigen::Vector3s point = sphereCenter + normal * sphereRadius;

    Contact contact;
    contact.point = point;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.sphereRadius = sphereRadius;
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
    const Eigen::Vector3s& sphereCenter,
    s_t sphereRadius,
    const std::vector<Eigen::Vector3s>& meshPointsWitness,
    ClipSphereHalfspace /* halfspace */)
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
    Eigen::Vector3s normal = (sphereCenter - meshPointsWitness[0]).normalized();
    Eigen::Vector3s contactPoint = meshPointsWitness[0];
    Contact contact;
    contact.point = contactPoint;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.vertexPoint = meshPointsWitness[0];
    contact.sphereRadius = sphereRadius;
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
    Eigen::Vector3s edge
        = (meshPointsWitness[1] - meshPointsWitness[0]).normalized();
    Eigen::Vector3s closestPoint
        = math::closestPointOnLine(meshPointsWitness[0], edge, sphereCenter);
    Eigen::Vector3s normal = (closestPoint - sphereCenter).normalized();
    Contact contact;
    contact.point = closestPoint;
    contact.normal = normal * -1;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.type = SPHERE_EDGE;
    contact.edgeAClosestPoint = closestPoint;
    contact.edgeAFixedPoint = meshPointsWitness[0];
    contact.edgeADir = edge;
    contact.penetrationDepth
        = sphereRadius - (closestPoint - sphereCenter).norm();
    result.addContact(contact);
  }
  // face-sphere collision
  else if (meshPointsWitness.size() > 2)
  {
    std::vector<Eigen::Vector3s> pointsWitnessSorted = meshPointsWitness;
    /*
    // `dir` points from A to B, so we want the highest dot product of dir at
    // the front of A
    std::sort(
        pointsWitnessSorted.begin(),
        pointsWitnessSorted.end(),
        [dir](Eigen::Vector3s& a, Eigen::Vector3s& b) {
          s_t aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
          s_t bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
          return aDot < bDot;
        });
        */
    // All the meshPointsWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3s normal
        = (pointsWitnessSorted[0] - pointsWitnessSorted[1])
              .cross(pointsWitnessSorted[1] - pointsWitnessSorted[2])
              .normalized();
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    s_t normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // We want to find the furthest point on the surface of the sphere, which is
    // just whatever point is facing the normal

    Eigen::Vector3s point = sphereCenter - normal * sphereRadius;

    Contact contact;
    contact.point = point;
    contact.normal = normal;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.sphereCenter = sphereCenter;
    contact.sphereRadius = sphereRadius;
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
    std::vector<Contact>& contacts,
    ccd_vec3_t* dir,
    const Eigen::Vector3s& capsuleA,
    const Eigen::Vector3s& capsuleB,
    s_t capsuleRadius,
    const std::vector<Eigen::Vector3s>& meshPointsWitness,
    bool flipObjectOrder,
    const CollisionOption& option)
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
    s_t alpha;
    dDistPointToSegment(meshPointsWitness[0], capsuleA, capsuleB, &alpha);
    Eigen::Vector3s nearestPoint = capsuleA + (capsuleB - capsuleA) * alpha;

    Eigen::Vector3s normal = (nearestPoint - meshPointsWitness[0]).normalized();
    Eigen::Vector3s contactPoint = meshPointsWitness[0];

    Contact contact;
    contact.point = contactPoint;
    if (flipObjectOrder)
    {
      contact.collisionObject1 = o2;
      contact.collisionObject2 = o1;
      contact.normal = normal * -1;
      contact.pipeClosestPoint = nearestPoint;
      contact.pipeDir = (capsuleB - capsuleA).normalized();
      contact.pipeFixedPoint = capsuleA;
      contact.pipeRadius = capsuleRadius;
      contact.type = VERTEX_PIPE;
    }
    else
    {
      contact.collisionObject1 = o1;
      contact.collisionObject2 = o2;
      contact.normal = normal;
      contact.pipeClosestPoint = nearestPoint;
      contact.pipeDir = (capsuleB - capsuleA).normalized();
      contact.pipeFixedPoint = capsuleA;
      contact.pipeRadius = capsuleRadius;
      contact.type = PIPE_VERTEX;
    }
    contact.penetrationDepth
        = capsuleRadius - (meshPointsWitness[0] - nearestPoint).norm();
    if (contact.penetrationDepth > option.contactClippingDepth)
      return 0;
    contacts.push_back(contact);

    return 1;
  }
  // edge-pipe collision
  else if (meshPointsWitness.size() == 2)
  {
    Eigen::Vector3s pipeDir = (capsuleB - capsuleA).normalized();
    Eigen::Vector3s edgeDir
        = (meshPointsWitness[1] - meshPointsWitness[0]).normalized();
    bool parallel = abs(1.0 - abs(pipeDir.dot(edgeDir))) < 1e-5;

    // Special case, if the edge is parallel to the pipe. This generates two
    // contacts, and has several different edge cases.
    if (parallel)
    {
      int numContacts = 0;

      s_t edgeALinear = edgeDir.dot(meshPointsWitness[0]);
      s_t edgeBLinear = edgeDir.dot(meshPointsWitness[1]);
      s_t capsuleALinear = edgeDir.dot(capsuleA);
      s_t capsuleBLinear = edgeDir.dot(capsuleB);

      s_t edgeMin = edgeALinear < edgeBLinear ? edgeALinear : edgeBLinear;
      Eigen::Vector3s edgeMinPoint = edgeALinear < edgeBLinear
                                         ? meshPointsWitness[0]
                                         : meshPointsWitness[1];
      s_t edgeMax = edgeALinear < edgeBLinear ? edgeBLinear : edgeALinear;
      Eigen::Vector3s edgeMaxPoint = edgeALinear < edgeBLinear
                                         ? meshPointsWitness[1]
                                         : meshPointsWitness[0];
      s_t capsuleMin
          = capsuleALinear < capsuleBLinear ? capsuleALinear : capsuleBLinear;
      Eigen::Vector3s capsuleMinPoint
          = capsuleALinear < capsuleBLinear ? capsuleA : capsuleB;
      s_t capsuleMax
          = capsuleALinear < capsuleBLinear ? capsuleBLinear : capsuleALinear;
      Eigen::Vector3s capsuleMaxPoint
          = capsuleALinear < capsuleBLinear ? capsuleB : capsuleA;

      Eigen::Vector3s normal = capsuleA - meshPointsWitness[0];
      normal -= edgeDir * normal.dot(edgeDir);
      s_t dist = normal.norm();
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
          contact.pipeClosestPoint = edgeMinPoint + normal * capsuleRadius;
          contact.pipeFixedPoint = capsuleA;
          contact.pipeDir = (capsuleB - capsuleA).normalized();
          contact.pipeRadius = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = PIPE_VERTEX;
          contact.pipeClosestPoint = edgeMinPoint + normal * capsuleRadius;
          contact.pipeFixedPoint = capsuleB;
          contact.pipeDir = (capsuleB - capsuleA);
          contact.pipeRadius = capsuleRadius;
        }
        contact.penetrationDepth = capsuleRadius - dist;
        if (contact.penetrationDepth > 0
            && contact.penetrationDepth < option.contactClippingDepth)
        {
          numContacts++;
          contacts.push_back(contact);
        }
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
          contact.edgeAFixedPoint = meshPointsWitness[0];
          contact.sphereRadius = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = SPHERE_EDGE;
          contact.sphereRadius = capsuleRadius;
          contact.edgeAClosestPoint = capsuleMinPoint - normal * capsuleRadius;
          contact.edgeADir = edgeDir;
          contact.edgeAFixedPoint = meshPointsWitness[0];
        }
        contact.penetrationDepth = capsuleRadius - dist;
        if (contact.penetrationDepth > 0
            && contact.penetrationDepth < option.contactClippingDepth)
        {
          numContacts++;
          contacts.push_back(contact);
        }
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
          contact.pipeClosestPoint = edgeMaxPoint + normal * capsuleRadius;
          contact.pipeFixedPoint = capsuleA;
          contact.pipeDir = (capsuleB - capsuleA).normalized();
          contact.pipeRadius = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = PIPE_VERTEX;
          contact.pipeClosestPoint = edgeMaxPoint + normal * capsuleRadius;
          contact.pipeFixedPoint = capsuleA;
          contact.pipeDir = (capsuleB - capsuleA).normalized();
          contact.pipeRadius = capsuleRadius;
        }
        contact.penetrationDepth = capsuleRadius - dist;
        if (contact.penetrationDepth > 0
            && contact.penetrationDepth < option.contactClippingDepth)
        {
          numContacts++;
          contacts.push_back(contact);
        }
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
          contact.edgeAFixedPoint = meshPointsWitness[0];
          contact.sphereRadius = capsuleRadius;
        }
        else
        {
          contact.collisionObject1 = o1;
          contact.collisionObject2 = o2;
          contact.normal = normal;
          contact.type = SPHERE_EDGE;
          contact.sphereRadius = capsuleRadius;
          contact.edgeAClosestPoint = capsuleMaxPoint - normal * capsuleRadius;
          contact.edgeADir = edgeDir;
          contact.edgeAFixedPoint = meshPointsWitness[0];
        }
        contact.penetrationDepth = capsuleRadius - dist;
        if (contact.penetrationDepth > 0
            && contact.penetrationDepth < option.contactClippingDepth)
        {
          numContacts++;
          contacts.push_back(contact);
        }
      }

      return numContacts;
    }
    // Standard case, edge and capsule are not parallel
    // Find nearest point on both edges and generate a single contact
    else
    {
      s_t alpha;
      s_t beta;
      dSegmentsClosestApproach(
          meshPointsWitness[0],
          capsuleA,
          meshPointsWitness[1],
          capsuleB,
          &alpha,
          &beta);

      Eigen::Vector3s edgeClosestPoint
          = meshPointsWitness[0]
            + alpha * (meshPointsWitness[1] - meshPointsWitness[0]);
      Eigen::Vector3s pipeClosestPoint
          = capsuleA + beta * (capsuleB - capsuleA);
      Eigen::Vector3s normal
          = (edgeClosestPoint - pipeClosestPoint).normalized();
      Eigen::Vector3s contactPoint = edgeClosestPoint;

      Contact contact;
      contact.point = contactPoint;
      if (flipObjectOrder)
      {
        contact.collisionObject1 = o2;
        contact.collisionObject2 = o1;
        contact.normal = normal;
        contact.type = EDGE_PIPE;
        contact.edgeAClosestPoint = edgeClosestPoint;
        contact.edgeAFixedPoint = meshPointsWitness[0];
        contact.edgeADir
            = (meshPointsWitness[1] - meshPointsWitness[0]).normalized();
        contact.pipeClosestPoint = pipeClosestPoint;
        contact.pipeDir = (capsuleB - capsuleA).normalized();
        contact.pipeFixedPoint = capsuleA;
        contact.pipeRadius = capsuleRadius;
      }
      else
      {
        contact.collisionObject1 = o1;
        contact.collisionObject2 = o2;
        contact.normal = normal * -1;
        contact.type = PIPE_EDGE;
        contact.pipeClosestPoint = pipeClosestPoint;
        contact.pipeDir = (capsuleB - capsuleA).normalized();
        contact.pipeFixedPoint = capsuleA;
        contact.pipeRadius = capsuleRadius;
        contact.edgeAClosestPoint = edgeClosestPoint;
        contact.edgeAFixedPoint = meshPointsWitness[0];
        contact.edgeADir
            = (meshPointsWitness[1] - meshPointsWitness[0]).normalized();
      }
      contact.penetrationDepth
          = capsuleRadius - (edgeClosestPoint - pipeClosestPoint).norm();
      if (contact.penetrationDepth > option.contactClippingDepth)
      {
        return 0;
      }

      contacts.push_back(contact);

      return 1;
    }
  }
  // face-pipe collision
  else if (meshPointsWitness.size() > 2)
  {
    std::vector<Eigen::Vector3s> pointsWitnessSorted = meshPointsWitness;
    // `dir` points from A to B, so we want the highest dot product of dir at
    // the front of A
    /*
    std::sort(
        pointsWitnessSorted.begin(),
        pointsWitnessSorted.end(),
        [dir](Eigen::Vector3s& a, Eigen::Vector3s& b) {
          s_t aDot = a(0) * dir->v[0] + a(1) * dir->v[1] + a(2) * dir->v[2];
          s_t bDot = b(0) * dir->v[0] + b(1) * dir->v[1] + b(2) * dir->v[2];
          return aDot < bDot;
        });
        */
    // All the meshPointsWitness vectors are co-planar, so we choose the closest
    // [0], [1] and [2] to cross to get a precise normal
    Eigen::Vector3s normal
        = (pointsWitnessSorted[0] - pointsWitnessSorted[1])
              .cross(pointsWitnessSorted[1] - pointsWitnessSorted[2])
              .normalized();
    // Ensure that the normal is in the opposite direction as `dir`, so we're
    // still pointing from B to A.
    s_t normalDot
        = normal(0) * dir->v[0] + normal(1) * dir->v[1] + normal(2) * dir->v[2];
    if (normalDot > 0)
      normal *= -1;

    // We want to find the closest points to the mesh plane
    std::vector<Eigen::Vector3s> capsulePointsWitness;

    // Now we need to process the two shapes to find intersection points
    std::vector<Contact> faceContacts;

    if (flipObjectOrder)
    {
      capsulePointsWitness.push_back(capsuleA + normal * capsuleRadius);
      capsulePointsWitness.push_back(capsuleB + normal * capsuleRadius);
      createFaceFaceContacts(
          faceContacts,
          o2,
          o1,
          dir,
          pointsWitnessSorted,
          capsulePointsWitness,
          PinToFace::FACE_A);
      int numContacts = 0;
      for (Contact& contact : faceContacts)
      {
        assert(contact.type != VERTEX_FACE);

        // This happens if the edge of the capsule is inside the face, in 2D
        // space
        if (contact.type == FACE_VERTEX)
        {
          contact.type = FACE_SPHERE;
          contact.sphereRadius = capsuleRadius;
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
          contact.point = contact.sphereCenter + contact.normal * capsuleRadius;
        }
        else if (contact.type == EDGE_EDGE)
        {
          contact.type = EDGE_PIPE;
          contact.pipeFixedPoint
              = contact.edgeBFixedPoint - normal * capsuleRadius;
          contact.pipeClosestPoint
              = contact.edgeBClosestPoint - normal * capsuleRadius;
          contact.pipeDir = contact.edgeBDir;
          contact.pipeRadius = capsuleRadius;
          // the contact point is defined as the closest point on the edge
          // (always included as edge A) in these cases
          contact.point = contact.edgeAClosestPoint;
        }
        else
        {
          assert(
              "Got an unexpected contact type in createCapsuleMeshContacts"
              && false);
        }
        if (contact.penetrationDepth >= 0
            && contact.penetrationDepth < option.contactClippingDepth)
        {
          numContacts++;
          contacts.push_back(contact);
        }
      }

      return numContacts;
    }
    else
    {
      capsulePointsWitness.push_back(capsuleA - normal * capsuleRadius);
      capsulePointsWitness.push_back(capsuleB - normal * capsuleRadius);
      createFaceFaceContacts(
          faceContacts,
          o1,
          o2,
          dir,
          capsulePointsWitness,
          pointsWitnessSorted,
          PinToFace::FACE_B);
      int numContacts = 0;
      for (Contact& contact : faceContacts)
      {
        assert(contact.type != FACE_VERTEX);

        // This happens if the edge of the capsule is inside the face, in 2D
        // space
        if (contact.type == VERTEX_FACE)
        {
          contact.type = SPHERE_FACE;
          contact.sphereRadius = capsuleRadius;
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
          contact.point = contact.sphereCenter - contact.normal * capsuleRadius;
        }
        else if (contact.type == EDGE_EDGE)
        {
          contact.type = PIPE_EDGE;
          contact.pipeFixedPoint
              = contact.edgeAFixedPoint + normal * capsuleRadius;
          contact.pipeClosestPoint
              = contact.edgeAClosestPoint + normal * capsuleRadius;
          contact.pipeDir = contact.edgeADir;
          contact.pipeRadius = capsuleRadius;
          contact.edgeAFixedPoint = contact.edgeBFixedPoint;
          contact.edgeAClosestPoint = contact.edgeBClosestPoint;
          contact.edgeADir = contact.edgeBDir;
          // the contact point is defined as the closest point on the edge
          // (always included as edge A) in these cases
          contact.point = contact.edgeAClosestPoint;
        }
        else
        {
          assert(
              "Got an unexpected contact type in createCapsuleMeshContacts"
              && false);
        }

        if (contact.penetrationDepth >= 0
            && contact.penetrationDepth < option.contactClippingDepth)
        {
          numContacts++;
          contacts.push_back(contact);
        }
      }

      return numContacts;
    }
  }

  assert(false);
  return 0;
}

/// This trims out any points that lie inside the convex polygon, without
/// changing the order.
void keepOnlyConvex2DHull(
    std::vector<Eigen::Vector3s>& shape,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY)
{
  // We need to throw out any points on the inside of the convex shape
  // TODO: there's gotta be a better algorithm than O(n^3).
  while (shape.size() > 0)
  {
    bool foundAnyToRemove = false;
    for (int i = 0; i < shape.size(); i++)
    {
      bool foundBoundaryPlane = false;
      Eigen::Vector2s shapeI
          = pointInPlane(shape[i], origin, basis2dX, basis2dY);

      for (int j = 0; j < shape.size(); j++)
      {
        if (i == j)
          continue;
        Eigen::Vector2s shapeJ
            = pointInPlane(shape[j], origin, basis2dX, basis2dY);

        Eigen::Vector2s plane
            = Eigen::Vector2s(shapeI(1) - shapeJ(1), shapeJ(0) - shapeI(0))
                  .normalized();
        s_t b = -plane.dot(shapeI);

        bool isBoundaryPlane = true;

        int side = 0;
        for (int k = 0; k < shape.size(); k++)
        {
          s_t measure
              = plane.dot(pointInPlane(shape[k], origin, basis2dX, basis2dY))
                + b;
          int kSide = ccdSign(measure);

          if (abs(measure) < 1e-3)
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
}

/*
/// This is necessary preparation for rapidly checking if another point is
/// contained within the convex shape.
void prepareConvex2DShape(std::vector<Eigen::Vector2s>& shape)
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

        Eigen::Vector2s plane
            = Eigen::Vector2s(
                  shape[i](1) - shape[j](1), shape[j](0) - shape[i](0))
                  .normalized();
        s_t b = -plane.dot(shape[i]);

        bool isBoundaryPlane = true;
        for (int k = 0; k < shape.size(); k++)
        {
          s_t measure = plane.dot(shape[k]) + b;
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
  Eigen::Vector2s avg = Eigen::Vector2s::Zero();
  for (Eigen::Vector2s pt : shape)
  {
    avg += pt;
  }
  avg /= shape.size();
  std::sort(
      shape.begin(),
      shape.end(),
      [&avg](Eigen::Vector2s& a, Eigen::Vector2s& b) {
        return angle2D(avg, a) < angle2D(avg, b);
      });
}
*/

// This implements the "2D cross product" as redefined here:
// https://stackoverflow.com/a/565282/13177487
inline s_t crossProduct2D(const Eigen::Vector2s& v, const Eigen::Vector2s& w)
{
  return v(0) * w(1) - v(1) * w(0);
}

inline void setCcdDefaultSettings(ccd_t& ccd)
{
  ccd.mpr_tolerance = 0.0001;
  ccd.epa_tolerance = 0.0001;
  ccd.dist_tolerance = 0.001;
  ccd.max_iterations = 10000;
}

/// This allows us to prevent weird effects where we don't want to carry over
/// cacheing
void clearCcdCache()
{
  const std::thread::id tid = std::this_thread::get_id();
  _ccdDirCache[tid].clear();
  _ccdPosCache[tid].clear();
  // _ccdDirCache.clear();
  // _ccdPosCache.clear();
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
    const Eigen::Vector2s& point, const std::vector<Eigen::Vector2s>& shape)
{
  int side = 0;
  for (int i = 0; i < shape.size(); i++)
  {
    const Eigen::Vector2s& a = shape[i];
    const Eigen::Vector2s& b = shape[(i + 1) % shape.size()];
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
    const Eigen::Vector3s& point,
    const std::vector<Eigen::Vector3s>& shape,
    const Eigen::Vector3s& origin,
    const Eigen::Vector3s& basis2dX,
    const Eigen::Vector3s& basis2dY)
{
  Eigen::Vector2s point2d = pointInPlane(point, origin, basis2dX, basis2dY);

  int side = 0;
  for (int i = 0; i < shape.size(); i++)
  {
    Eigen::Vector2s a = pointInPlane(shape[i], origin, basis2dX, basis2dY);
    Eigen::Vector2s b = pointInPlane(
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

// Returns 1 if the lines intersect, otherwise 0. In addition, if the lines
// intersect the intersection point may be stored in the floats i(0) and i(1).
//
// Adapted from sources: https://stackoverflow.com/a/1968345/13177487 and
// https://stackoverflow.com/a/565282/13177487
bool get2DLineIntersection(
    const Eigen::Vector2s& p,
    const Eigen::Vector2s& p1,
    const Eigen::Vector2s& q,
    const Eigen::Vector2s& q1,
    Eigen::Vector2s& out)
{
  Eigen::Vector2s r = p1 - p;
  Eigen::Vector2s s = q1 - q;

  // If r × s = 0 and (q − p) × r = 0, then the two lines are collinear.
  if (crossProduct2D(r, s) == 0 && crossProduct2D(q - p, r) == 0)
  {
    // In this case, express the endpoints of the second segment (q and q + s)
    // in terms of the equation of the first line segment (p + t r)

    s_t t0 = (q - p).dot(r) / r.dot(r);
    s_t t1 = (q + s - p).dot(r) / r.dot(r);

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

  s_t t = crossProduct2D(q - p, s) / crossProduct2D(r, s);
  s_t u = crossProduct2D(p - q, r) / crossProduct2D(s, r);

  if (t >= 0 && t <= 1 && u >= 0 && u <= 1)
  {
    out = p + t * r;
    return true;
  }

  /*
  // Simpler proposed version
  s_t s, t;
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

// Get the `pos` vec for CCD for this pair of objects
ccd_vec3_t& getCachedCcdPos(CollisionObject* o1, CollisionObject* o2)
{
  long key = (long)o1 ^ (long)o2;
  const std::thread::id tid = std::this_thread::get_id();
  ccd_vec3_t& pos = _ccdPosCache[tid][key];
  return pos;
}

// Get the `dir` vec for CCD for this pair of objects
ccd_vec3_t& getCachedCcdDir(CollisionObject* o1, CollisionObject* o2)
{
  long key = (long)o1 ^ (long)o2;
  const std::thread::id tid = std::this_thread::get_id();
  ccd_vec3_t& dir = _ccdDirCache[tid][key];
  return dir;
}

int collideBoxBoxAsMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&box1, &box2, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3s> pointsA
        = ccdPointsAtWitnessBox(&box1, &dir, false);
    std::vector<Eigen::Vector3s> pointsB
        = ccdPointsAtWitnessBox(&box2, &dir, true);

    return createMeshMeshContacts(o1, o2, result, &dir, pointsA, pointsB);
  }
  return 0;
}

int collideMeshBox(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* mesh0,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&mesh1, &box2, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3s> pointsA
        = ccdPointsAtWitnessMesh(&mesh1, &dir, false);
    std::vector<Eigen::Vector3s> pointsB
        = ccdPointsAtWitnessBox(&box2, &dir, true);

    return createMeshMeshContacts(o1, o2, result, &dir, pointsA, pointsB);
  }
  return 0;
}

int collideBoxMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const aiScene* m1,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&box1, &mesh2, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3s> pointsA
        = ccdPointsAtWitnessBox(&box1, &dir, false);
    std::vector<Eigen::Vector3s> pointsB
        = ccdPointsAtWitnessMesh(&mesh2, &dir, true);

    if (pointsB.size() == 0)
    {
      std::vector<Eigen::Vector3s> pointsB
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
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const s_t& r1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
    CollisionResult& result,
    ClipSphereHalfspace /* halfspace */)
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&mesh, &sphere, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3s> meshPoints
        = ccdPointsAtWitnessMesh(&mesh, &dir, false);

    return createMeshSphereContact(
        o1, o2, result, &dir, meshPoints, c1.translation(), r1);
  }
  return 0;
}

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
    ClipSphereHalfspace /* halfspace */)
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&sphere, &mesh, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3s> meshPoints
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
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& c0,
    const aiScene* m1,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& c1,
    const CollisionOption& option,
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&mesh1, &mesh2, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    std::vector<Eigen::Vector3s> pointsA
        = ccdPointsAtWitnessMesh(&mesh1, &dir, false);
    std::vector<Eigen::Vector3s> pointsB
        = ccdPointsAtWitnessMesh(&mesh2, &dir, true);

    return createMeshMeshContacts(o1, o2, result, &dir, pointsA, pointsB);
  }
  return 0;
}

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
    CollisionResult& result)
{
  Eigen::Vector3s pa = T0 * (Eigen::Vector3s::UnitZ() * -(height0 / 2));
  Eigen::Vector3s pb = T0 * (Eigen::Vector3s::UnitZ() * (height0 / 2));
  Eigen::Vector3s ua = T1 * (Eigen::Vector3s::UnitZ() * -(height1 / 2));
  Eigen::Vector3s ub = T1 * (Eigen::Vector3s::UnitZ() * (height1 / 2));
  s_t alpha, beta;
  dSegmentsClosestApproach(pa, ua, pb, ub, &alpha, &beta);
  if (alpha < 0)
    alpha = 0;
  if (alpha > 1)
    alpha = 1;
  if (beta < 0)
    beta = 0;
  if (beta > 1)
    beta = 1;

  Eigen::Vector3s closest0 = pa + (pb - pa) * alpha;
  Eigen::Vector3s closest1 = ua + (ub - ua) * beta;

  s_t dist = (closest0 - closest1).norm();
  s_t rsum = radius0 + radius1;
  if (dist <= rsum)
  {
    // There's a contact!
    radius0 /= rsum;
    radius1 /= rsum;

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth = rsum - dist;
    if (contact.penetrationDepth > option.contactClippingDepth)
      return 0;
    contact.point = (closest0 * radius1) + (closest1 * radius0);
    contact.normal = (closest0 - closest1).normalized();

    const s_t SPHERE_THRESHOLD = 1e-8;

    contact.radiusA = radius0 * rsum;
    contact.radiusB = radius1 * rsum;
    bool isSphere0
        = abs(alpha) < SPHERE_THRESHOLD || abs(1 - alpha) < SPHERE_THRESHOLD;
    bool isSphere1
        = abs(beta) < SPHERE_THRESHOLD || abs(1 - beta) < SPHERE_THRESHOLD;

    if (isSphere0 && isSphere1)
    {
      contact.type = SPHERE_SPHERE;
      contact.centerA = closest0;
      contact.centerB = closest1;
    }
    else if (isSphere0)
    {
      contact.type = SPHERE_PIPE;
      contact.sphereRadius = radius0 * rsum;
      contact.sphereCenter = closest0;
      contact.pipeRadius = radius1 * rsum;
      contact.pipeClosestPoint = closest1;
      contact.pipeFixedPoint = ua;
      contact.pipeDir = (ub - ua).normalized();
    }
    else if (isSphere1)
    {
      contact.type = PIPE_SPHERE;
      contact.pipeRadius = radius0 * rsum;
      contact.pipeClosestPoint = closest0;
      contact.pipeFixedPoint = pa;
      contact.pipeDir = (pb - pa).normalized();
      contact.sphereRadius = radius1 * rsum;
      contact.sphereCenter = closest1;
    }
    else
    {
      contact.type = PIPE_PIPE;
      contact.radiusA = radius0;
      contact.radiusB = radius1;
      contact.edgeAFixedPoint = pa;
      contact.edgeAClosestPoint = closest0;
      contact.edgeADir = (pb - pa).normalized();
      contact.edgeBFixedPoint = ua;
      contact.edgeBClosestPoint = closest1;
      contact.edgeBDir = (ub - ua).normalized();
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
    s_t radius0,
    const Eigen::Isometry3s& T0,
    s_t height1,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result)
{
  s_t alpha;
  Eigen::Vector3s center0 = T0.translation();
  Eigen::Vector3s ua = T1 * (Eigen::Vector3s::UnitZ() * -(height1 / 2));
  Eigen::Vector3s ub = T1 * (Eigen::Vector3s::UnitZ() * (height1 / 2));

  s_t dist = dDistPointToSegment(center0, ua, ub, &alpha);
  if (dist < radius0 + radius1)
  {
    // There's a contact!

    Eigen::Vector3s closest1 = ua + (ub - ua) * alpha;
    s_t rsum = radius0 + radius1;

    radius0 /= rsum;
    radius1 /= rsum;

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth = rsum - dist;
    if (contact.penetrationDepth > option.contactClippingDepth)
      return 0;
    contact.point = (center0 * radius1) + (closest1 * radius0);
    contact.normal = (center0 - closest1).normalized();

    const s_t SPHERE_THRESHOLD = 1e-8;

    contact.radiusA = radius0 * rsum;
    contact.radiusB = radius1 * rsum;
    bool isSphere1
        = abs(alpha) < SPHERE_THRESHOLD || abs(1 - alpha) < SPHERE_THRESHOLD;

    if (isSphere1)
    {
      contact.type = SPHERE_SPHERE;
      contact.centerA = center0;
      contact.centerB = closest1;
    }
    else
    {
      contact.type = SPHERE_PIPE;
      contact.sphereRadius = radius0 * rsum;
      contact.pipeRadius = radius1 * rsum;
      contact.sphereCenter = center0;
      contact.pipeClosestPoint = closest1;
      contact.pipeFixedPoint = ua;
      contact.pipeDir = (ub - ua).normalized();
    }

    result.addContact(contact);

    return 1;
  }

  return 0;
}

int collideCapsuleSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t height0,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
    CollisionResult& result)
{
  s_t alpha;
  Eigen::Vector3s ua = T0 * (Eigen::Vector3s::UnitZ() * -(height0 / 2));
  Eigen::Vector3s ub = T0 * (Eigen::Vector3s::UnitZ() * (height0 / 2));
  Eigen::Vector3s center1 = T1.translation();

  s_t dist = dDistPointToSegment(center1, ua, ub, &alpha);
  if (dist < radius0 + radius1)
  {
    // There's a contact!

    Eigen::Vector3s closest0 = ua + (ub - ua) * alpha;
    s_t rsum = radius0 + radius1;

    radius0 /= rsum;
    radius1 /= rsum;

    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth = rsum - dist;
    if (contact.penetrationDepth > option.contactClippingDepth)
      return 0;
    contact.point = (closest0 * radius1) + (center1 * radius0);
    contact.normal = (closest0 - center1).normalized();

    const s_t SPHERE_THRESHOLD = 1e-8;

    contact.radiusA = radius0 * rsum;
    contact.radiusB = radius1 * rsum;
    bool isSphere1
        = abs(alpha) < SPHERE_THRESHOLD || abs(1 - alpha) < SPHERE_THRESHOLD;

    if (isSphere1)
    {
      contact.type = SPHERE_SPHERE;
      contact.centerA = closest0;
      contact.centerB = center1;
    }
    else
    {
      contact.type = PIPE_SPHERE;
      contact.pipeRadius = radius0 * rsum;
      contact.pipeClosestPoint = closest0;
      contact.pipeFixedPoint = ua;
      contact.pipeDir = (ub - ua).normalized();
      contact.sphereRadius = radius1 * rsum;
      contact.sphereCenter = center1;
    }

    result.addContact(contact);

    return 1;
  }

  return 0;
}

int collideBoxCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    s_t height1,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
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
  capsule2.height = static_cast<double>(height1);
  capsule2.radius = static_cast<double>(radius1);
  capsule2.transform = &T1;

  ccd_real_t depth;
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&box1, &capsule2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    if (depth > option.contactClippingDepth)
      return 0;
    Eigen::Map<Eigen::Vector3d> posMap(pos.v);
    Eigen::Vector3s localPos = T1.inverse() * posMap.cast<s_t>();
    if (localPos(2) > height1 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, height1 / 2);
      return collideBoxSphere(
          o1,
          o2,
          size0,
          T0,
          radius1,
          T1 * sphereTransform,
          option,
          result,
          ClipSphereHalfspace::TOP);
    }
    else if (localPos(2) < -height1 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, -height1 / 2);
      return collideBoxSphere(
          o1,
          o2,
          size0,
          T0,
          radius1,
          T1 * sphereTransform,
          option,
          result,
          ClipSphereHalfspace::BOTTOM);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3s> meshPoints
        = ccdPointsAtWitnessBox(&box1, &dir, false);

    std::vector<Contact> contacts;
    int numContacts = createCapsuleMeshContact(
        o1,
        o2,
        contacts,
        &dir,
        T1 * Eigen::Vector3s(0, 0, height1 / 2),
        T1 * Eigen::Vector3s(0, 0, -height1 / 2),
        radius1,
        meshPoints,
        true,
        option);
    (void)numContacts;
    assert(contacts.size() == numContacts);
    for (auto contact : contacts)
    {
      assert(contact.type != ContactType::SPHERE_FACE);
      if (contact.type == ContactType::FACE_SPHERE)
      {
        Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
        sphereTransform.translation() = contact.sphereCenter;
        collideBoxSphere(
            o1, o2, size0, T0, radius1, sphereTransform, option, result);
      }
      else
      {
        result.addContact(contact);
      }
    }
    return contacts.size();
  }
  return 0;
}

int collideCapsuleBox(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t height0,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect = ccdMPRPenetration(&capsule1, &box2, &ccd, &depth, &dir, &pos);
  if (intersect == 0)
  {
    if (depth > option.contactClippingDepth)
      return 0;
    Eigen::Vector3s posMap;
    posMap(0) = static_cast<s_t>(pos.v[0]);
    posMap(1) = static_cast<s_t>(pos.v[1]);
    posMap(2) = static_cast<s_t>(pos.v[2]);
    Eigen::Vector3s localPos = T0.inverse() * posMap;
    if (localPos(2) > height0 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, height0 / 2);
      return collideSphereBox(
          o1,
          o2,
          radius0,
          T0 * sphereTransform,
          size1,
          T1,
          option,
          result,
          ClipSphereHalfspace::TOP);
    }
    else if (localPos(2) < -height0 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, -height0 / 2);
      return collideSphereBox(
          o1,
          o2,
          radius0,
          T0 * sphereTransform,
          size1,
          T1,
          option,
          result,
          ClipSphereHalfspace::BOTTOM);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3s> meshPoints
        = ccdPointsAtWitnessBox(&box2, &dir, true);

    std::vector<Contact> contacts;
    int numContacts = createCapsuleMeshContact(
        o1,
        o2,
        contacts,
        &dir,
        T0 * Eigen::Vector3s(0, 0, height0 / 2),
        T0 * Eigen::Vector3s(0, 0, -height0 / 2),
        radius0,
        meshPoints,
        false,
        option);
    (void)numContacts;
    assert(contacts.size() == numContacts);
    for (auto contact : contacts)
    {
      assert(contact.type != ContactType::FACE_SPHERE);
      if (contact.type == ContactType::SPHERE_FACE)
      {
        Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
        sphereTransform.translation() = contact.sphereCenter;
        collideSphereBox(
            o1, o2, radius0, sphereTransform, size1, T1, option, result);
      }
      else
      {
        result.addContact(contact);
      }
    }
    return contacts.size();
  }
  return 0;
}

int collideMeshCapsule(
    CollisionObject* o1,
    CollisionObject* o2,
    const aiScene* m0,
    const Eigen::Vector3s& size0,
    const Eigen::Isometry3s& T0,
    s_t height1,
    s_t radius1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect
      = ccdMPRPenetration(&mesh1, &capsule2, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    Eigen::Vector3s posMap;
    posMap(0) = static_cast<s_t>(pos.v[0]);
    posMap(1) = static_cast<s_t>(pos.v[1]);
    posMap(2) = static_cast<s_t>(pos.v[2]);
    Eigen::Vector3s localPos = T1.inverse() * posMap;
    if (localPos(2) > height1 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, height1 / 2);
      return collideMeshSphere(
          o1,
          o2,
          m0,
          size0,
          T0,
          radius1,
          T1 * sphereTransform,
          option,
          result,
          ClipSphereHalfspace::TOP);
    }
    else if (localPos(2) < -height1 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, -height1 / 2);
      return collideMeshSphere(
          o1,
          o2,
          m0,
          size0,
          T0,
          radius1,
          T1 * sphereTransform,
          option,
          result,
          ClipSphereHalfspace::BOTTOM);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3s> meshPoints
        = ccdPointsAtWitnessMesh(&mesh1, &dir, false);

    std::vector<Contact> contacts;
    int numContacts = createCapsuleMeshContact(
        o1,
        o2,
        contacts,
        &dir,
        T1 * Eigen::Vector3s(0, 0, height1 / 2),
        T1 * Eigen::Vector3s(0, 0, -height1 / 2),
        radius1,
        meshPoints,
        true,
        option);
    (void)numContacts;
    assert(contacts.size() == numContacts);
    for (auto contact : contacts)
    {
      result.addContact(contact);
    }
    return contacts.size();
  }
  return 0;
}

int collideCapsuleMesh(
    CollisionObject* o1,
    CollisionObject* o2,
    s_t height0,
    s_t radius0,
    const Eigen::Isometry3s& T0,
    const aiScene* m1,
    const Eigen::Vector3s& size1,
    const Eigen::Isometry3s& T1,
    const CollisionOption& option,
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
  ccd_vec3_t& dir = getCachedCcdDir(o1, o2);
  ccd_vec3_t& pos = getCachedCcdPos(o1, o2);
  int intersect
      = ccdMPRPenetration(&capsule1, &mesh2, &ccd, &depth, &dir, &pos);
  if (depth > option.contactClippingDepth)
    return 0;
  if (intersect == 0)
  {
    Eigen::Vector3s posMap;
    posMap(0) = static_cast<s_t>(pos.v[0]);
    posMap(1) = static_cast<s_t>(pos.v[1]);
    posMap(2) = static_cast<s_t>(pos.v[2]);
    Eigen::Vector3s localPos = T0.inverse() * posMap;
    if (localPos(2) > height0 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, height0 / 2);
      return collideSphereMesh(
          o1,
          o2,
          radius0,
          T0 * sphereTransform,
          m1,
          size1,
          T1,
          option,
          result,
          ClipSphereHalfspace::TOP);
    }
    else if (localPos(2) < -height0 / 2)
    {
      Eigen::Isometry3s sphereTransform = Eigen::Isometry3s::Identity();
      sphereTransform.translation() = Eigen::Vector3s(0, 0, -height0 / 2);
      return collideSphereMesh(
          o1,
          o2,
          radius0,
          T0 * sphereTransform,
          m1,
          size1,
          T1,
          option,
          result,
          ClipSphereHalfspace::BOTTOM);
    }

    // Otherwise we're on an edge, and have to handle the pipe collisions
    // properly

    std::vector<Eigen::Vector3s> meshPoints
        = ccdPointsAtWitnessMesh(&mesh2, &dir, true);

    std::vector<Contact> contacts;
    int numContacts = createCapsuleMeshContact(
        o1,
        o2,
        contacts,
        &dir,
        T0 * Eigen::Vector3s(0, 0, height0 / 2),
        T0 * Eigen::Vector3s(0, 0, -height0 / 2),
        radius0,
        meshPoints,
        false,
        option);
    (void)numContacts;
    assert(contacts.size() == numContacts);
    for (auto contact : contacts)
    {
      result.addContact(contact);
    }
    return contacts.size();
  }
  return 0;
}

int collideCylinderSphere(
    CollisionObject* o1,
    CollisionObject* o2,
    const s_t& cyl_rad,
    const s_t& half_height,
    const Eigen::Isometry3s& T0,
    const s_t& sphere_rad,
    const Eigen::Isometry3s& T1,
    CollisionResult& result,
    const CollisionOption& /* option */,
    ClipSphereHalfspace /* halfspace */)
{
  Eigen::Vector3s center = T0.inverse() * T1.translation();

  s_t dist = sqrt(center[0] * center[0] + center[1] * center[1]);

  if (dist < cyl_rad && abs(center[2]) < half_height + sphere_rad)
  {
    Contact contact;
    contact.collisionObject1 = o1;
    contact.collisionObject2 = o2;
    contact.penetrationDepth
        = 0.5 * (half_height + sphere_rad - math::sign(center[2]) * center[2]);
    contact.point
        = T0
          * Eigen::Vector3s(
              center[0], center[1], half_height - contact.penetrationDepth);
    contact.normal
        = T0.linear() * Eigen::Vector3s(0.0, 0.0, math::sign(center[2]));
    result.addContact(contact);
    return 1;
  }
  else
  {
    s_t penetration = 0.5 * (cyl_rad + sphere_rad - dist);
    if (penetration > 0.0)
    {
      if (abs(center[2]) > half_height)
      {
        Eigen::Vector3s point
            = (Eigen::Vector3s(center[0], center[1], 0.0).normalized());
        point *= cyl_rad;
        point[2] = math::sign(center[2]) * half_height;
        Eigen::Vector3s normal = point - center;
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
        Eigen::Vector3s point
            = (Eigen::Vector3s(center[0], center[1], 0.0)).normalized();
        Eigen::Vector3s normal = -(T0.linear() * point);
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
    const s_t& cyl_rad,
    const s_t& half_height,
    const Eigen::Isometry3s& T0,
    const Eigen::Vector3s& plane_normal,
    const Eigen::Isometry3s& T1,
    const CollisionOption& /* option */,
    CollisionResult& result)
{
  Eigen::Vector3s normal = T1.linear() * plane_normal;
  Eigen::Vector3s Rx = T0.linear().rightCols(1);
  Eigen::Vector3s Ry = normal - normal.dot(Rx) * Rx;
  s_t mag = Ry.norm();
  Ry.normalize();
  if (mag < DART_COLLISION_EPS)
  {
    if (abs(Rx[2]) > 1.0 - DART_COLLISION_EPS)
      Ry = Eigen::Vector3s::UnitX();
    else
      Ry = (Eigen::Vector3s(Rx[1], -Rx[0], 0.0)).normalized();
  }

  Eigen::Vector3s Rz = Rx.cross(Ry);
  Eigen::Isometry3s T;
  T.linear().col(0) = Rx;
  T.linear().col(1) = Ry;
  T.linear().col(2) = Rz;
  T.translation() = T0.translation();

  Eigen::Vector3s nn = T.linear().transpose() * normal;
  Eigen::Vector3s pn = T.inverse() * T1.translation();

  // four corners c0 = ( -h/2, -r ), c1 = ( +h/2, -r ), c2 = ( +h/2, +r ), c3
  // = ( -h/2, +r )
  Eigen::Vector3s c[4]
      = {Eigen::Vector3s(-half_height, -cyl_rad, 0.0),
         Eigen::Vector3s(+half_height, -cyl_rad, 0.0),
         Eigen::Vector3s(+half_height, +cyl_rad, 0.0),
         Eigen::Vector3s(-half_height, +cyl_rad, 0.0)};

  s_t depth[4]
      = {(pn - c[0]).dot(nn),
         (pn - c[1]).dot(nn),
         (pn - c[2]).dot(nn),
         (pn - c[3]).dot(nn)};

  s_t penetration = -1.0;
  int found = -1;
  for (int i = 0; i < 4; i++)
  {
    if (depth[i] > penetration)
    {
      penetration = depth[i];
      found = i;
    }
  }

  Eigen::Vector3s point;

  if (abs(depth[found] - depth[(found + 1) % 4]) < DART_COLLISION_EPS)
    point = T * (0.5 * (c[found] + c[(found + 1) % 4]));
  else if (abs(depth[found] - depth[(found + 3) % 4]) < DART_COLLISION_EPS)
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
int collide(
    CollisionObject* o1,
    CollisionObject* o2,
    const CollisionOption& option,
    CollisionResult& result)
{
  // TODO(JS): We could make the contact point computation as optional for
  // the case that we want only binary check.

  const auto& shape1 = o1->getShape();
  const auto& shape2 = o2->getShape();

  const auto& shapeType1 = shape1->getType();
  const auto& shapeType2 = shape2->getType();

  const Eigen::Isometry3s& T1 = o1->getTransform();
  const Eigen::Isometry3s& T2 = o2->getTransform();

  if (dynamics::SphereShape::getStaticType() == shapeType1)
  {
    const auto* sphere0
        = static_cast<const dynamics::SphereShape*>(shape1.get());

    if (dynamics::SphereShape::getStaticType() == shapeType2)
    {
      const auto* sphere1
          = static_cast<const dynamics::SphereShape*>(shape2.get());

      return collideSphereSphere(
          o1,
          o2,
          sphere0->getRadius(),
          T1,
          sphere1->getRadius(),
          T2,
          option,
          result);
    }
    else if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      return collideSphereBox(
          o1,
          o2,
          sphere0->getRadius(),
          T1,
          box1->getSize(),
          T2,
          option,
          result);
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
          option,
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
          option,
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
          option,
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
          o1,
          o2,
          box0->getSize(),
          T1,
          sphere1->getRadius(),
          T2,
          option,
          result);
    }
    else if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      /*
#ifndef NDEBUG
      CollisionResult boxBoxResult;
      int boxCollides = collideBoxBox(
          o1, o2, box0->getSize(), T1, box1->getSize(), T2, boxBoxResult);
      CollisionResult meshMeshResult;
      int meshCollides = collideBoxBoxAsMesh(
          o1, o2, box0->getSize(), T1, box1->getSize(), T2, meshMeshResult);

      if ((boxCollides > 0) != (meshCollides > 0))
      {
        std::cout << "Breakpoint!" << std::endl;
        Eigen::Vector3s randDir = Eigen::Vector3s::Random();
        boxBoxResult.sortContacts(randDir);
        meshMeshResult.sortContacts(randDir);
        for (int i = 0; i < std::min(
                            boxBoxResult.getNumContacts(),
                            meshMeshResult.getNumContacts());
             i++)
        {
          std::cout << "Contact " << i << std::endl;
          std::cout << "Box-box Type: " << std::endl
                    << boxBoxResult.getContact(i).type << std::endl;
          std::cout << "Mesh-mesh Type: " << std::endl
                    << meshMeshResult.getContact(i).type << std::endl;
          std::cout << "Box-box Normal: " << std::endl
                    << boxBoxResult.getContact(i).normal << std::endl;
          std::cout << "Mesh-mesh Normal: " << std::endl
                    << meshMeshResult.getContact(i).normal << std::endl;
          std::cout << "Normal diff: " << std::endl
                    << (boxBoxResult.getContact(i).normal
                        - meshMeshResult.getContact(i).normal)
                    << std::endl;
          std::cout << "Box-box Point: " << std::endl
                    << boxBoxResult.getContact(i).point << std::endl;
          std::cout << "Mesh-mesh Point: " << std::endl
                    << meshMeshResult.getContact(i).point << std::endl;
          std::cout << "Point diff: " << std::endl
                    << (boxBoxResult.getContact(i).point
                        - meshMeshResult.getContact(i).point)
                    << std::endl;
        }
        for (int i = std::min(
                 boxBoxResult.getNumContacts(),
                 meshMeshResult.getNumContacts());
             i < boxBoxResult.getNumContacts();
             i++)
        {
          std::cout << "Box-box Contact " << i << std::endl;
          std::cout << "Box-box Type: " << std::endl
                    << boxBoxResult.getContact(i).type << std::endl;
          std::cout << "Box-box Normal: " << std::endl
                    << boxBoxResult.getContact(i).normal << std::endl;
          std::cout << "Box-box Point: " << std::endl
                    << boxBoxResult.getContact(i).point << std::endl;
        }
        for (int i = std::min(
                 boxBoxResult.getNumContacts(),
                 meshMeshResult.getNumContacts());
             i < meshMeshResult.getNumContacts();
             i++)
        {
          std::cout << "Mesh-mesh Contact " << i << std::endl;
          std::cout << "Mesh-mesh Type: " << std::endl
                    << meshMeshResult.getContact(i).type << std::endl;
          std::cout << "Mesh-mesh Normal: " << std::endl
                    << meshMeshResult.getContact(i).normal << std::endl;
          std::cout << "Mesh-mesh Point: " << std::endl
                    << meshMeshResult.getContact(i).point << std::endl;
        }
        std::cout << "To replicate:" << std::endl;
        std::cout << "////////////////////////////////////////" << std::endl;
        std::cout << "Eigen::Vector3s size0 = Eigen::Vector3s("
                  << box0->getSize()(0) << "," << box0->getSize()(1) << ","
                  << box0->getSize()(2) << ");" << std::endl;
        std::cout << "Eigen::Matrix4d M_T0;" << std::endl;
        std::cout << "// clang-format off" << std::endl;
        std::cout << "M_T0 << ";
        for (int row = 0; row < 4; row++)
        {
          for (int col = 0; col < 4; col++)
          {
            std::cout << T1.matrix()(row, col);
            if (row == 3 && col == 3)
            {
              std::cout << ";";
            }
            else
            {
              std::cout << ",";
            }
          }
          std::cout << std::endl;
          if (row < 3)
          {
            std::cout << "        ";
          }
        }
        std::cout << "// clang-format on" << std::endl;
        std::cout << "Eigen::Isometry3s T0(M_T0);" << std::endl;
        std::cout << "Eigen::Vector3s size1 = Eigen::Vector3s("
                  << box1->getSize()(0) << "," << box1->getSize()(1) << ","
                  << box1->getSize()(2) << ");" << std::endl;
        std::cout << "Eigen::Matrix4d M_T1;" << std::endl;
        std::cout << "// clang-format off" << std::endl;
        std::cout << "M_T1 << ";
        for (int row = 0; row < 4; row++)
        {
          for (int col = 0; col < 4; col++)
          {
            std::cout << T2.matrix()(row, col);
            if (row == 3 && col == 3)
            {
              std::cout << ";";
            }
            else
            {
              std::cout << ",";
            }
          }
          std::cout << std::endl;
          if (row < 3)
          {
            std::cout << "        ";
          }
        }
        std::cout << "// clang-format on" << std::endl;
        std::cout << "Eigen::Isometry3s T1(M_T1);" << std::endl;
        std::cout << "////////////////////////////////////////" << std::endl;
      }
#endif
      */
      return collideBoxBox(
          o1, o2, box0->getSize(), T1, box1->getSize(), T2, option, result);
    }
    else if (dynamics::EllipsoidShape::getStaticType() == shapeType2)
    {
      const auto* ellipsoid1
          = static_cast<const dynamics::EllipsoidShape*>(shape2.get());

      return collideBoxSphere(
          o1,
          o2,
          box0->getSize(),
          T1,
          ellipsoid1->getRadii()[0],
          T2,
          option,
          result);
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
          option,
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
          option,
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
          option,
          result);
    }
    else if (dynamics::BoxShape::getStaticType() == shapeType2)
    {
      const auto* box1 = static_cast<const dynamics::BoxShape*>(shape2.get());

      return collideSphereBox(
          o1,
          o2,
          ellipsoid0->getRadii()[0],
          T1,
          box1->getSize(),
          T2,
          option,
          result);
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
          option,
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
