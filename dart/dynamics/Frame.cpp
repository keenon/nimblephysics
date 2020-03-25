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

#include "dart/dynamics/Frame.hpp"

#include "dart/common/Console.hpp"
#include "dart/dynamics/Shape.hpp"

namespace dart {
namespace dynamics {

typedef std::set<Entity*> EntityPtrSet;
typedef std::set<Frame*> FramePtrSet;

//==============================================================================
Frame::~Frame()
{
  if (isWorld())
    return;

  changeParentFrame(nullptr);

  // Inform all child entities that this Frame is disappearing by setting their
  // reference frames to the World frame.
  EntityPtrSet::iterator it = mChildEntities.begin(),
                         end = mChildEntities.end();
  while (it != end)
    (*(it++))->changeParentFrame(Frame::World());
  // Note: When we instruct an Entity to change its parent Frame, it will erase
  // itself from this Frame's mChildEntities list. This would invalidate the
  // 'it' and clobber our attempt to iterate through the std::set if we applied
  // changeParentFrame() directly to the 'it' iterator. So instead we use the
  // post-increment operator to iterate 'it' forward and we apply
  // changeParentFrame() to the temporary iterator created by the
  // post-increment operator. Put simply: we increment 'it' forward once and
  // then apply changeParentFrame() to the pointer that 'it' held just before
  // it incremented.

  // The entity destructor takes care of informing the parent Frame that this
  // one is disappearing
}

//==============================================================================
Frame* Frame::World()
{
  static WorldFrame world;
  return &world;
}

//==============================================================================
std::shared_ptr<Frame> Frame::WorldShared()
{
  struct EnableMakeShared : WorldFrame
  {
    EnableMakeShared() : Entity(nullptr, true), WorldFrame()
    {
    }
  };
  static auto sharedWorld = std::make_shared<EnableMakeShared>();
  return sharedWorld;
}

//==============================================================================
const Eigen::Isometry3d& Frame::getWorldTransform() const
{
  if (mAmWorld)
    return mWorldTransform;

  if (mNeedTransformUpdate)
  {
    mWorldTransform
        = mParentFrame->getWorldTransform() * getRelativeTransform();
    mNeedTransformUpdate = false;
  }

  return mWorldTransform;
}

//==============================================================================
Eigen::Isometry3d Frame::getTransform(const Frame* _withRespectTo) const
{
  if (_withRespectTo->isWorld())
    return getWorldTransform();
  else if (_withRespectTo == mParentFrame)
    return getRelativeTransform();
  else if (_withRespectTo == this)
    return Eigen::Isometry3d::Identity();

  return _withRespectTo->getWorldTransform().inverse() * getWorldTransform();
}

//==============================================================================
Eigen::Isometry3d Frame::getTransform(
    const Frame* withRespectTo, const Frame* inCoordinatesOf) const
{
  assert(nullptr != withRespectTo);
  assert(nullptr != inCoordinatesOf);

  if (withRespectTo == inCoordinatesOf)
    return getTransform(withRespectTo);

  // Rotation from "inCoordinatesOf" to "withRespecTo"
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = inCoordinatesOf->getWorldTransform().linear().transpose()
               * withRespectTo->getWorldTransform().linear();

  return T * getTransform(withRespectTo);
}

//==============================================================================
const Eigen::Vector6d& Frame::getSpatialVelocity() const
{
  if (mAmWorld)
    return mVelocity;

  if (mNeedVelocityUpdate)
  {
#if DART_USE_WORLD_COORDINATES
    const Eigen::Vector6d& parentVel = getParentFrame()->getSpatialVelocity();
    const Eigen::Vector3d p = getParentFrame()->getWorldTransform().linear()
                              * getRelativeTransform().translation();
    mVelocity = parentVel;
    mVelocity.tail<3>().noalias() += parentVel.head<3>().cross(p);
    mVelocity.noalias()
        += math::AdR(getWorldTransform(), getRelativeSpatialVelocity());
#else
    mVelocity
        = math::AdInvT(
              getRelativeTransform(), getParentFrame()->getSpatialVelocity())
          + getRelativeSpatialVelocity();
#endif
    mNeedVelocityUpdate = false;
  }

  return mVelocity;
}

//==============================================================================
Eigen::Vector6d Frame::getSpatialVelocity(
    const Frame* relativeTo, const Frame* inCoordinatesOf) const
{
#if DART_USE_WORLD_COORDINATES
  if (this == relativeTo)
    return Eigen::Vector6d::Zero();

  if (relativeTo->isWorld())
  {
    if (inCoordinatesOf->isWorld())
      return getSpatialVelocity();

    if (this == inCoordinatesOf)
      return math::AdInvR(getWorldTransform(), getSpatialVelocity());

    return math::AdInvR(
        inCoordinatesOf->getWorldTransform(), getSpatialVelocity());
  }

  // Optimized code of
  //   V' = V1 - Ad([I p12; 0 1], V2)
  // or
  //   w' = w1 - w2
  //   v' = v1 - (v2 + [p12]w2)
  // where
  //   V': The result spatial velocity in the world coordinates.
  //   V1: The spatial velocity of this Frame in the world coordinates.
  //   V2: The spatial velocity of relativeTo Frame in the world coordinates.
  //   p12: The vector from body2 to body1 in the world coordinates.
  const Eigen::Vector3d p12 = getWorldTransform().translation()
                              - relativeTo->getWorldTransform().translation();
  Eigen::Vector6d result
      = getSpatialVelocity() - relativeTo->getSpatialVelocity();
  result.tail<3>().noalias()
      += p12.cross(relativeTo->getSpatialVelocity().head<3>());

  if (inCoordinatesOf->isWorld())
    return result;

  return math::AdInvR(inCoordinatesOf->getWorldTransform(), result);
#else
  if (this == relativeTo)
    return Eigen::Vector6d::Zero();

  if (relativeTo->isWorld())
  {
    if (this == inCoordinatesOf)
      return getSpatialVelocity();

    if (inCoordinatesOf->isWorld())
      return math::AdR(getWorldTransform(), getSpatialVelocity());

    return math::AdR(getTransform(inCoordinatesOf), getSpatialVelocity());
  }

  const Eigen::Vector6d& result = (getSpatialVelocity()
                                   - math::AdT(
                                         relativeTo->getTransform(this),
                                         relativeTo->getSpatialVelocity()))
                                      .eval();

  if (this == inCoordinatesOf)
    return result;

  return math::AdR(getTransform(inCoordinatesOf), result);
#endif
}

//==============================================================================
Eigen::Vector6d Frame::getSpatialVelocity(const Eigen::Vector3d& offset) const
{
  return getSpatialVelocity(offset, Frame::World(), this);
}

//==============================================================================
Eigen::Vector6d Frame::getSpatialVelocity(
    const Eigen::Vector3d& offset,
    const Frame* relativeTo,
    const Frame* inCoordinatesOf) const
{
#if DART_USE_WORLD_COORDINATES
  if (this == relativeTo)
    return Eigen::Vector6d::Zero();

  // Optimized code of
  //   V' = Ad([I -R1*offset; 0 1], V1)
  // or
  //   w' = w1
  //   v' = v1 - [R1*offset]w1
  // where
  //   V': The result spatial velocity in the world coordinates.
  //   V1: The spatial velocity of this Frame in the world coordinates.
  //   R1: The rotation matrix of this Frame.
  //   offset: The vector from the origin of this Frame to the offset in this
  //   Frame coordinates.
  const Eigen::Vector3d offsetInWorldCoords
      = getWorldTransform().linear() * offset;
  Eigen::Vector6d v = getSpatialVelocity();
  v.tail<3>().noalias() += v.head<3>().cross(offsetInWorldCoords);

  if (relativeTo->isWorld())
  {
    if (inCoordinatesOf->isWorld())
      return v;

    return math::AdInvR(inCoordinatesOf->getWorldTransform(), v);
  }

  // Optimized code of
  //   V" = V' - Ad([I -R1*offset; 0 1], Ad([I p1r; 0 1], Vrel))
  // or
  //   w" = w' - wr
  //   v" = v' - (vr - [R1*offset]wr + [p1r]wr)
  // where
  //   V': The result spatial velocity in the world coordinates.
  //   V": The result spatial velocity relative to relativeTo in the world
  //       coordinates.
  const Eigen::Vector3d p1r = relativeTo->getWorldTransform().translation()
                              - getWorldTransform().translation();
  Eigen::Vector6d v_0 = relativeTo->getSpatialVelocity();
  v_0.tail<3>().noalias() += v_0.head<3>().cross(offsetInWorldCoords - p1r);
  v = v - v_0;

  if (inCoordinatesOf->isWorld())
    return v;

  return math::AdInvR(inCoordinatesOf->getWorldTransform(), v);
#else
  if (this == relativeTo)
    return Eigen::Vector6d::Zero();

  Eigen::Vector6d v = getSpatialVelocity();
  v.tail<3>().noalias() += v.head<3>().cross(offset);

  if (relativeTo->isWorld())
  {
    if (this == inCoordinatesOf)
      return v;

    return math::AdR(getTransform(inCoordinatesOf), v);
  }

  Eigen::Vector6d v_0 = math::AdT(
      relativeTo->getTransform(this), relativeTo->getSpatialVelocity());
  v_0.tail<3>().noalias() += v_0.head<3>().cross(offset);

  v = v - v_0;

  if (this == inCoordinatesOf)
    return v;

  return math::AdR(getTransform(inCoordinatesOf), v);
#endif
}

//==============================================================================
Eigen::Vector3d Frame::getLinearVelocity(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getSpatialVelocity(_relativeTo, _inCoordinatesOf).tail<3>();
}

//==============================================================================
Eigen::Vector3d Frame::getLinearVelocity(
    const Eigen::Vector3d& _offset,
    const Frame* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  return getSpatialVelocity(_offset, _relativeTo, _inCoordinatesOf).tail<3>();
}

//==============================================================================
Eigen::Vector3d Frame::getAngularVelocity(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getSpatialVelocity(_relativeTo, _inCoordinatesOf).head<3>();
}

//==============================================================================
const Eigen::Vector6d& Frame::getSpatialAcceleration() const
{
  if (mAmWorld)
    return mAcceleration;

  if (mNeedAccelerationUpdate)
  {
#if DART_USE_WORLD_COORDINATES
    // V(i) = Ad([I p(i-1,i); 0 1]^{-1}, V(i-1))
    //      + ad(V(i), S(i)*dq(i)) + dS(i)*dq(i)
    //      + S(i)*ddq(i)
    // where
    //   p(i-1,i): The relative translation of this Frmae in world coordinates.
    //       S(i): The relative Jacobian of this Frame in world coordinates.
    //      dS(i): The time derivative of the relative Jacobian of this Frame in
    //             world coordiantes.
    //     V(i-1): The velocity of the parent Frame of this Frame in world
    //             coordinates.
    //       V(i): The velocity of this Frame in world coordinates.
    //      dq(i): The joint velocities of the parent joint of this Frame.
    //     ddq(i): The joint acceleration of the parent joint of this Frame.
    mAcceleration = math::AdInvT(
                        getRelativeTransform(),
                        getParentFrame()->getSpatialAcceleration())
                    + getPartialAcceleration()
                    + getPrimaryRelativeAcceleration();
#else
    // V(i) = Ad([R(i-1,i) p(i-1,i); 0 1]^{-1}, V(i-1))
    //      + ad(V(i), S(i)*dq(i)) + dS(i)*dq(i)
    //      + S(i)*ddq(i)
    mAcceleration = math::AdInvT(
                        getRelativeTransform(),
                        getParentFrame()->getSpatialAcceleration())
                    + getPartialAcceleration()
                    + getPrimaryRelativeAcceleration();
#endif

    mNeedAccelerationUpdate = false;
  }

  return mAcceleration;
}

//==============================================================================
Eigen::Vector6d Frame::getSpatialAcceleration(
    const Frame* relativeTo, const Frame* inCoordinatesOf) const
{
  // Frame 2: this, Frame 1: _relativeTo, Frame O: _inCoordinatesOf
  // Acceleration of Frame 2 relative to Frame 1 in coordinates of O: a_21[O]
  // Total acceleration of Frame 2 in coordinates of Frame 2: a_2[2]
  // Velocity of Frame 2 relative to Frame 1 in coordinates of Frame 2: v_21[2]

  // a_21[O] = R_O2*( a_2[2] - X_21*a_1[1] - v_2[2] x v_21[2] )

  if (this == relativeTo)
    return Eigen::Vector6d::Zero();

  if (relativeTo->isWorld())
  {
    if (this == inCoordinatesOf)
      return getSpatialAcceleration();

    if (inCoordinatesOf->isWorld())
      return math::AdR(getWorldTransform(), getSpatialAcceleration());

    return math::AdR(getTransform(inCoordinatesOf), getSpatialAcceleration());
  }

  const Eigen::Vector6d& result
      = (getSpatialAcceleration()
         - math::AdT(
               relativeTo->getTransform(this),
               relativeTo->getSpatialAcceleration())
         + math::ad(
               getSpatialVelocity(),
               math::AdT(
                   relativeTo->getTransform(this),
                   relativeTo->getSpatialVelocity())))
            .eval();

  if (this == inCoordinatesOf)
    return result;

  return math::AdR(getTransform(inCoordinatesOf), result);
}

//==============================================================================
Eigen::Vector6d Frame::getSpatialAcceleration(
    const Eigen::Vector3d& _offset) const
{
  return getSpatialAcceleration(_offset, Frame::World(), this);
}

//==============================================================================
Eigen::Vector6d Frame::getSpatialAcceleration(
    const Eigen::Vector3d& offset,
    const Frame* relativeTo,
    const Frame* inCoordinatesOf) const
{
  if (this == relativeTo)
    return Eigen::Vector6d::Zero();

  // Compute spatial acceleration of the point
  Eigen::Vector6d a = getSpatialAcceleration();
  a.tail<3>().noalias() += a.head<3>().cross(offset);

  if (relativeTo->isWorld())
  {
    if (this == inCoordinatesOf)
      return a;

    return math::AdR(getTransform(inCoordinatesOf), a);
  }

  // Compute the spatial velocity of the point
  Eigen::Vector6d v = getSpatialVelocity();
  v.tail<3>().noalias() += v.head<3>().cross(offset);

  // Compute the acceleration of the reference Frame
  Eigen::Vector6d a_ref = math::AdT(
      relativeTo->getTransform(this), relativeTo->getSpatialAcceleration());
  a_ref.tail<3>().noalias() += a_ref.head<3>().cross(offset);

  // Compute the relative velocity of the point
  const Eigen::Vector6d& v_rel = getSpatialVelocity(offset, relativeTo, this);

  a = a - a_ref - math::ad(v, v_rel);

  if (this == inCoordinatesOf)
    return a;

  return math::AdR(getTransform(inCoordinatesOf), a);
}

//==============================================================================
Eigen::Vector3d Frame::getLinearAcceleration(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  if (this == _relativeTo)
    return Eigen::Vector3d::Zero();

  const Eigen::Vector6d& v_rel = getSpatialVelocity(_relativeTo, this);

  // r'' = a + w x v
  const Eigen::Vector3d& a
      = (getSpatialAcceleration(_relativeTo, this).tail<3>()
         + v_rel.head<3>().cross(v_rel.tail<3>()))
            .eval();

  if (this == _inCoordinatesOf)
    return a;

  return getTransform(_inCoordinatesOf).linear() * a;
}

//==============================================================================
Eigen::Vector3d Frame::getLinearAcceleration(
    const Eigen::Vector3d& _offset,
    const Frame* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  if (this == _relativeTo)
    return Eigen::Vector3d::Zero();

  const Eigen::Vector6d& v_rel = getSpatialVelocity(_offset, _relativeTo, this);
  const Eigen::Vector3d& a
      = (getSpatialAcceleration(_offset, _relativeTo, this).tail<3>()
         + v_rel.head<3>().cross(v_rel.tail<3>()))
            .eval();

  if (this == _inCoordinatesOf)
    return a;

  return getTransform(_inCoordinatesOf).linear() * a;
}

//==============================================================================
Eigen::Vector3d Frame::getAngularAcceleration(
    const Frame* _relativeTo, const Frame* _inCoordinatesOf) const
{
  return getSpatialAcceleration(_relativeTo, _inCoordinatesOf).head<3>();
}

//==============================================================================
template <typename T>
static std::set<const T*> convertToConstSet(const std::set<T*>& _set)
{
  std::set<const T*> const_set;
  for (const auto& element : _set)
    const_set.insert(element);

  return const_set;
}

//==============================================================================
const std::set<Entity*>& Frame::getChildEntities()
{
  return mChildEntities;
}

//==============================================================================
const std::set<const Entity*> Frame::getChildEntities() const
{
  return convertToConstSet<Entity>(mChildEntities);
}

//==============================================================================
std::size_t Frame::getNumChildEntities() const
{
  return mChildEntities.size();
}

//==============================================================================
const std::set<Frame*>& Frame::getChildFrames()
{
  return mChildFrames;
}

//==============================================================================
std::set<const Frame*> Frame::getChildFrames() const
{
  return convertToConstSet<Frame>(mChildFrames);
}

//==============================================================================
std::size_t Frame::getNumChildFrames() const
{
  return mChildFrames.size();
}

//==============================================================================
bool Frame::isShapeFrame() const
{
  return mAmShapeFrame;
}

//==============================================================================
ShapeFrame* Frame::asShapeFrame()
{
  return nullptr;
}

//==============================================================================
const ShapeFrame* Frame::asShapeFrame() const
{
  return nullptr;
}

//==============================================================================
bool Frame::isWorld() const
{
  return mAmWorld;
}

//==============================================================================
void Frame::dirtyTransform()
{
  dirtyVelocity(); // Global Velocity depends on the Global Transform

  // Always trigger the signal, in case a new subscriber has registered in the
  // time since the last signal
  mTransformUpdatedSignal.raise(this);

  // If we already know we need to update, just quit
  if (mNeedTransformUpdate)
    return;

  mNeedTransformUpdate = true;

  for (Entity* entity : mChildEntities)
    entity->dirtyTransform();
}

//==============================================================================
void Frame::dirtyVelocity()
{
  dirtyAcceleration(); // Global Acceleration depends on Global Velocity

  // Always trigger the signal, in case a new subscriber has registered in the
  // time since the last signal
  mVelocityChangedSignal.raise(this);

  // If we already know we need to update, just quit
  if (mNeedVelocityUpdate)
    return;

  mNeedVelocityUpdate = true;

  for (Entity* entity : mChildEntities)
    entity->dirtyVelocity();
}

//==============================================================================
void Frame::dirtyAcceleration()
{
  // Always trigger the signal, in case a new subscriber has registered in the
  // time since the last signal
  mAccelerationChangedSignal.raise(this);

  // If we already know we need to update, just quit
  if (mNeedAccelerationUpdate)
    return;

  mNeedAccelerationUpdate = true;

  for (Entity* entity : mChildEntities)
    entity->dirtyAcceleration();
}

//==============================================================================
Frame::Frame(Frame* _refFrame)
  : Entity(ConstructFrame),
    mWorldTransform(Eigen::Isometry3d::Identity()),
    mVelocity(Eigen::Vector6d::Zero()),
    mAcceleration(Eigen::Vector6d::Zero()),
    mAmWorld(false),
    mAmShapeFrame(false)
{
  mAmFrame = true;
  changeParentFrame(_refFrame);
}

//==============================================================================
Frame::Frame() : Frame(ConstructAbstract)
{
  // Delegated to Frame(ConstructAbstract)
}

//==============================================================================
Frame::Frame(ConstructAbstractTag)
  : Entity(Entity::ConstructAbstract), mAmWorld(false), mAmShapeFrame(false)
{
  dterr << "[Frame::constructor] You are calling a constructor for the Frame "
        << "class which is only meant to be used by pure abstract classes. If "
        << "you are seeing this, then there is a bug!\n";
  assert(false);
}

//==============================================================================
void Frame::changeParentFrame(Frame* _newParentFrame)
{
  if (mParentFrame == _newParentFrame)
    return;

  if (_newParentFrame)
  {
    if (_newParentFrame->descendsFrom(this))
    {
      if (!(this->isWorld() && _newParentFrame->isWorld()))
      // We make an exception here for the World Frame, because it's
      // special/unique
      {
        dtwarn << "[Frame::changeParentFrame] Attempting to create a circular "
               << "kinematic dependency by making Frame '" << getName()
               << "' a child of Frame '" << _newParentFrame->getName() << "'. "
               << "This will not be allowed.\n";
        return;
      }
    }
  }

  if (mParentFrame && !mParentFrame->isWorld())
  {
    FramePtrSet::iterator it = mParentFrame->mChildFrames.find(this);
    if (it != mParentFrame->mChildFrames.end())
      mParentFrame->mChildFrames.erase(it);
  }

  if (nullptr == _newParentFrame)
  {
    Entity::changeParentFrame(_newParentFrame);
    return;
  }

  if (!mAmQuiet && !_newParentFrame->isWorld())
    _newParentFrame->mChildFrames.insert(this);

  Entity::changeParentFrame(_newParentFrame);
}

//==============================================================================
void Frame::processNewEntity(Entity*)
{
  // Do nothing
}

//==============================================================================
void Frame::processRemovedEntity(Entity*)
{
  // Do nothing
}

//==============================================================================
Frame::Frame(ConstructWorldTag)
  : Entity(this, true),
    mWorldTransform(Eigen::Isometry3d::Identity()),
    mVelocity(Eigen::Vector6d::Zero()),
    mAcceleration(Eigen::Vector6d::Zero()),
    mAmWorld(true),
    mAmShapeFrame(false)
{
  mAmFrame = true;
}

//==============================================================================
const Eigen::Vector6d WorldFrame::mZero = Eigen::Vector6d::Zero();

//==============================================================================
const Eigen::Isometry3d& WorldFrame::getRelativeTransform() const
{
  return mRelativeTf;
}

//==============================================================================
const Eigen::Vector6d& WorldFrame::getRelativeSpatialVelocity() const
{
  return mZero;
}

//==============================================================================
const Eigen::Vector6d& WorldFrame::getRelativeSpatialAcceleration() const
{
  return mZero;
}

//==============================================================================
const Eigen::Vector6d& WorldFrame::getPrimaryRelativeAcceleration() const
{
  return mZero;
}

//==============================================================================
const Eigen::Vector6d& WorldFrame::getPartialAcceleration() const
{
  return mZero;
}

//==============================================================================
const std::string& WorldFrame::setName(const std::string& name)
{
  dterr << "[WorldFrame::setName] attempting to change name of World frame to ["
        << name << "], but this is not allowed!\n";
  static const std::string worldName = "World";
  return worldName;
}

//==============================================================================
const std::string& WorldFrame::getName() const
{
  static const std::string worldName = "World";
  return worldName;
}

//==============================================================================
WorldFrame::WorldFrame()
  : Entity(nullptr, true),
    Frame(ConstructWorld),
    mRelativeTf(Eigen::Isometry3d::Identity())
{
  changeParentFrame(this);
}

} // namespace dynamics
} // namespace dart
