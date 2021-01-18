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

#include "dart/collision/CollisionResult.hpp"

#include "dart/collision/CollisionObject.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/ShapeFrame.hpp"
#include "dart/dynamics/ShapeNode.hpp"

namespace dart {
namespace collision {

//==============================================================================
void CollisionResult::addContact(const Contact& contact)
{
  mContacts.push_back(contact);
  if (contact.collisionObject1 == nullptr
      || contact.collisionObject2 == nullptr)
  {
    std::cout
        << "THIS IS AN ERROR IN PRODUCTION! Got a nullptr in "
           "CollisionResult::addContact() for collisionObject1 and/or "
           "collisionObject2. Ignoring the objects, in case we're running a "
           "unit test, but this could lead to invalid behavior downstream."
        << std::endl;
  }
  else
  {
    addObject(contact.collisionObject1);
    addObject(contact.collisionObject2);
  }
}

//==============================================================================
std::size_t CollisionResult::getNumContacts() const
{
  return mContacts.size();
}

//==============================================================================
Contact& CollisionResult::getContact(std::size_t index)
{
  assert(index < mContacts.size());

  return mContacts[index];
}

//==============================================================================
const Contact& CollisionResult::getContact(std::size_t index) const
{
  assert(index < mContacts.size());

  return mContacts[index];
}

//==============================================================================
/// This sorts the list of contacts by the contact position dotted with some
/// random direction. This makes it much easier to compare sets of
/// CollisionResults.
void CollisionResult::sortContacts(Eigen::Vector3d& randDirection)
{
  std::sort(mContacts.begin(), mContacts.end(), [&](Contact& a, Contact& b) {
    double dotA = a.point.dot(randDirection);
    double dotB = b.point.dot(randDirection);
    if (dotA == dotB)
    {
      dotA = a.normal.dot(randDirection);
      dotB = b.normal.dot(randDirection);
      if (dotA == dotB)
      {
        Eigen::Vector3d ortho1 = a.normal.cross(randDirection);
        dotA = a.point.dot(ortho1);
        dotB = b.point.dot(ortho1);
        if (dotA == dotB)
        {
          dotA = a.normal.dot(ortho1);
          dotB = b.normal.dot(ortho1);
          if (dotA == dotB)
          {
            Eigen::Vector3d ortho2 = randDirection.cross(ortho1);
            dotA = a.normal.dot(ortho2);
            dotB = b.normal.dot(ortho2);
          }
        }
      }
    }
    return dotA < dotB;
  });
}

//==============================================================================
const std::vector<Contact>& CollisionResult::getContacts() const
{
  return mContacts;
}

//==============================================================================
const std::unordered_set<const dynamics::BodyNode*>&
CollisionResult::getCollidingBodyNodes() const
{
  return mCollidingBodyNodes;
}

//==============================================================================
const std::unordered_set<const dynamics::ShapeFrame*>&
CollisionResult::getCollidingShapeFrames() const
{
  return mCollidingShapeFrames;
}

//==============================================================================
bool CollisionResult::inCollision(const dynamics::BodyNode* bn) const
{
  return (mCollidingBodyNodes.find(bn) != mCollidingBodyNodes.end());
}

//==============================================================================
bool CollisionResult::inCollision(const dynamics::ShapeFrame* frame) const
{
  return (mCollidingShapeFrames.find(frame) != mCollidingShapeFrames.end());
}

//==============================================================================
bool CollisionResult::isCollision() const
{
  return !mContacts.empty();
}

//==============================================================================
CollisionResult::operator bool() const
{
  return isCollision();
}

//==============================================================================
void CollisionResult::clear()
{
  mContacts.clear();
  mCollidingShapeFrames.clear();
  mCollidingBodyNodes.clear();
}

//==============================================================================
void CollisionResult::addObject(CollisionObject* object)
{
  if (!object)
  {
    dterr << "[CollisionResult::addObject] Attempting to add a collision with "
          << "a nullptr object to a CollisionResult instance. This is not "
          << "allowed. Please report this as a bug!";
    assert(false);
    return;
  }

  const dynamics::ShapeFrame* frame = object->getShapeFrame();
  mCollidingShapeFrames.insert(frame);

  if (frame->isShapeNode())
  {
    const dynamics::ShapeNode* node = frame->asShapeNode();
    mCollidingBodyNodes.insert(node->getBodyNodePtr());
  }
}

} // namespace collision
} // namespace dart
