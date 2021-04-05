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

#include "dart/dynamics/MetaSkeleton.hpp"

#include <algorithm>
#include "dart/common/Console.hpp"
#include "dart/dynamics/DegreeOfFreedom.hpp"
#include "dart/dynamics/JacobianNode.hpp"

namespace dart {
namespace dynamics {

//==============================================================================
static bool checkIndexArrayValidity(const MetaSkeleton* skel,
                                    const std::vector<std::size_t>& _indices,
                                    const std::string& _fname)
{
  std::size_t dofs = skel->getNumDofs();
  for(std::size_t i=0; i<_indices.size(); ++i)
  {
    if(_indices[i] >= dofs)
    {
      if(dofs > 0)
      {
        dterr << "[Skeleton::" << _fname << "] Invalid entry (" << i << ") in "
              << "_indices array: " << _indices[i] << ". Value must be less than "
              << dofs << " for the Skeleton named [" << skel->getName() << "] ("
              << skel << ")\n";
      }
      else
      {
        dterr << "[Skeleton::" << _fname << "] The Skeleton named ["
              << skel->getName() << "] (" << skel << ") is empty, but _indices "
              << "has entries in it. Nothing will be set!\n";
      }

      return false;
    }
  }
  return true;
}

//==============================================================================
static bool checkIndexArrayAgreement(const MetaSkeleton* skel,
                                     const std::vector<std::size_t>& _indices,
                                     const Eigen::VectorXs& _values,
                                     const std::string& _fname,
                                     const std::string& _vname)
{
  if( static_cast<int>(_indices.size()) != _values.size() )
  {
    dterr << "[Skeleton::" << _fname << "] Mismatch between _indices size ("
          << _indices.size() << ") and " << _vname << " size ("
          << _values.size() << ") for Skeleton named [" << skel->getName()
          << "] (" << skel << "). Nothing will be set!\n";
    assert(false);
    return false;
  }

  return checkIndexArrayValidity(skel, _indices, _fname);
}

//==============================================================================
template <void (DegreeOfFreedom::*setValue)(s_t _value)>
static void setValuesFromVector(MetaSkeleton* skel,
                                const std::vector<std::size_t>& _indices,
                                const Eigen::VectorXs& _values,
                                const std::string& _fname,
                                const std::string& _vname)
{
  if(!checkIndexArrayAgreement(skel, _indices, _values, _fname, _vname))
    return;

  for (std::size_t i=0; i<_indices.size(); ++i)
  {
    DegreeOfFreedom* dof = skel->getDof(_indices[i]);
    if(dof)
    {
      (dof->*setValue)(_values[i]);
    }
    else
    {
      dterr << "[MetaSkeleton::" << _fname << "] DegreeOfFreedom #"
            << _indices[i] << " (entry #" << i << " in " << _vname << ") has "
            << "expired! ReferentialSkeletons should call update() after "
            << "structural changes have been made to the BodyNodes they refer "
            << "to. Nothing will be set for this specific DegreeOfFreedom.\n";
      assert(false);
    }
  }
}

//==============================================================================
template <void (DegreeOfFreedom::*setValue)(s_t _value)>
static void setAllValuesFromVector(MetaSkeleton* skel,
                                   const Eigen::VectorXs& _values,
                                   const std::string& _fname,
                                   const std::string& _vname)
{
  std::size_t nDofs = skel->getNumDofs();
  if( _values.size() != static_cast<int>(skel->getNumDofs()) )
  {
    dterr << "[MetaSkeleton::" << _fname << "] Invalid number of entries ("
          << _values.size() << ") in " << _vname << " for MetaSkeleton named ["
          << skel->getName() << "] (" << skel << "). Must be equal to ("
          << skel->getNumDofs() << "). Nothing will be set!\n";
    assert(false);
    return;
  }

  for(std::size_t i=0; i < nDofs; ++i)
  {
    DegreeOfFreedom* dof = skel->getDof(i);
    if(dof)
    {
      (dof->*setValue)(_values[i]);
    }
    else
    {
      dterr << "[MetaSkeleton::" << _fname << "] DegreeOfFreedom #" << i
            << " in the MetaSkeleton named [" << skel->getName() << "] ("
            << skel << ") has expired! ReferentialSkeletons should call "
            << "update() after structural changes have been made to the "
            << "BodyNodes they refer to. Nothing will be set for this specific "
            << "DegreeOfFreedom.\n";
      assert(false);
    }
  }
}

//==============================================================================
template <s_t (DegreeOfFreedom::*getValue)() const>
static Eigen::VectorXs getValuesFromVector(
    const MetaSkeleton* skel, const std::vector<std::size_t>& _indices,
    const std::string& _fname)
{
  Eigen::VectorXs values(_indices.size());

  for(std::size_t i=0; i<_indices.size(); ++i)
  {
    const DegreeOfFreedom* dof = skel->getDof(_indices[i]);
    if(dof)
    {
      values[i] = (dof->*getValue)();
    }
    else
    {
      values[i] = 0.0;
      if(i < skel->getNumDofs())
      {
        dterr << "[MetaSkeleton::" << _fname << "] Requesting value for "
              << "DegreeOfFreedom #" << _indices[i] << " (" << "entry #" << i
              << " in _indices), but this index has expired! "
              << "ReferentialSkeletons should call update() after structural "
              << "changes have been made to the BodyNodes they refer to. The "
              << "return value for this entry will be zero.\n";
      }
      else
      {
        dterr << "[MetaSkeleton::" << _fname << "] Requesting out of bounds "
              << "DegreeOfFreedom #" << _indices[i] << " (entry #" << i
              << " in _indices) for MetaSkeleton named [" << skel->getName()
              << "] (" << skel << "). The max index is (" << skel->getNumDofs()
              << "). The return value for this entry will be zero.\n";
      }
      assert(false);
    }
  }

  return values;
}

//==============================================================================
template <s_t (DegreeOfFreedom::*getValue)() const>
static Eigen::VectorXs getValuesFromAllDofs(
    const MetaSkeleton* skel, const std::string& _fname)
{
  std::size_t nDofs = skel->getNumDofs();
  Eigen::VectorXs values(nDofs);

  for(std::size_t i=0; i<nDofs; ++i)
  {
    const DegreeOfFreedom* dof = skel->getDof(i);
    if(dof)
    {
      values[i] = (skel->getDof(i)->*getValue)();
    }
    else
    {
      dterr << "[MetaSkeleton::" << _fname << "] DegreeOfFreedom #" << i
            << " has expired! ReferentialSkeletons should call update() after "
            << "structural changes have been made to the BodyNodes they refer "
            << "to. The return value for this entry will be zero.\n";
      values[i] = 0.0;
      assert(false);
    }
  }

  return values;
}

//==============================================================================
template <void (DegreeOfFreedom::*apply)()>
static void applyToAllDofs(MetaSkeleton* skel)
{
  std::size_t nDofs = skel->getNumDofs();
  for(std::size_t i=0; i<nDofs; ++i)
  {
    DegreeOfFreedom* dof = skel->getDof(i);
    if(dof)
      (dof->*apply)();
  }
}

//==============================================================================
template <void (DegreeOfFreedom::*setValue)(s_t _value)>
static void setValueFromIndex(MetaSkeleton* skel, std::size_t _index, s_t _value,
                              const std::string& _fname)
{
  if(_index >= skel->getNumDofs())
  {
    if(skel->getNumDofs() > 0)
      dterr << "[MetaSkeleton::" << _fname << "] Out of bounds index ("
            << _index << ") for MetaSkeleton named [" << skel->getName()
            << "] (" << skel << "). Must be less than " << skel->getNumDofs()
            << "!\n";
    else
      dterr << "[MetaSkeleton::" << _fname << "] Index (" << _index
            << ") cannot be used on MetaSkeleton [" << skel->getName() << "] ("
            << skel << ") because it is empty!\n";
    assert(false);
    return;
  }

  DegreeOfFreedom* dof = skel->getDof(_index);
  if(dof)
  {
    (dof->*setValue)(_value);
  }
  else
  {
    dterr << "[MetaSkeleton::" << _fname << "] DegreeOfFreedom #" << _index
          << " in the MetaSkeleton named [" << skel->getName() << "] (" << skel
          << ") has expired! ReferentialSkeletons should call update() after "
          << "structural changes have been made to the BodyNodes they refer "
          << "to. Nothing will be set!\n";
    assert(false);
  }
}

//==============================================================================
template <s_t (DegreeOfFreedom::*getValue)() const>
static s_t getValueFromIndex(const MetaSkeleton* skel, std::size_t _index,
                                const std::string& _fname)
{
  if(_index >= skel->getNumDofs())
  {
    if(skel->getNumDofs() > 0)
      dterr << "[MetaSkeleton::" << _fname << "] Out of bounds index ("
            << _index << ") for MetaSkeleton named [" << skel->getName()
            << "] (" << skel << "). Must be less than " << skel->getNumDofs()
            << "! The return value will be zero.\n";
    else
      dterr << "[MetaSkeleton::" << _fname << "] Index (" << _index
            << ") cannot " << "be requested for MetaSkeleton ["
            << skel->getName() << "] (" << skel << ") because it is empty! "
            << "The return value will be zero.\n";
    assert(false);
    return 0.0;
  }

  const DegreeOfFreedom* dof = skel->getDof(_index);
  if(dof)
  {
    return (skel->getDof(_index)->*getValue)();
  }

  dterr << "[MetaSkeleton::" << _fname << "] DegreeOfFreedom #" << _index
        << "in the MetaSkeleton named [" << skel->getName() << "] (" << skel
        << ") has expired! ReferentialSkeletons should call update() after "
        << "structural changes have been made to the BodyNodes they refer to. "
        << "The return value will be zero.\n";
  assert(false);
  return 0.0;
}

//==============================================================================
MetaSkeletonPtr MetaSkeleton::cloneMetaSkeleton() const
{
  return cloneMetaSkeleton(getName());
}

//==============================================================================
void MetaSkeleton::setCommand(std::size_t _index, s_t _command)
{
  setValueFromIndex<&DegreeOfFreedom::setCommand>(
        this, _index, _command, "setCommand");
}

//==============================================================================
s_t MetaSkeleton::getCommand(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getCommand>(
        this, _index, "getCommand");
}

//==============================================================================
void MetaSkeleton::setCommands(const Eigen::VectorXs& _commands)
{
  setAllValuesFromVector<&DegreeOfFreedom::setCommand>(
        this, _commands, "setCommands", "_commands");
}

//==============================================================================
void MetaSkeleton::setCommands(const std::vector<std::size_t>& _indices,
                           const Eigen::VectorXs& _commands)
{
  setValuesFromVector<&DegreeOfFreedom::setCommand>(
        this, _indices, _commands, "setCommands", "_commands");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getCommands() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getCommand>(
        this, "getCommands");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getCommands(const std::vector<std::size_t>& _indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getCommand>(
        this, _indices, "getCommands");
}

//==============================================================================
void MetaSkeleton::resetCommands()
{
  applyToAllDofs<&DegreeOfFreedom::resetCommand>(this);
}

//==============================================================================
void MetaSkeleton::setPosition(std::size_t _index, s_t _position)
{
  setValueFromIndex<&DegreeOfFreedom::setPosition>(
        this, _index, _position, "setPosition");
}

//==============================================================================
s_t MetaSkeleton::getPosition(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getPosition>(
        this, _index, "getPosition");
}

//==============================================================================
void MetaSkeleton::setPositions(const Eigen::VectorXs& _positions)
{
  setAllValuesFromVector<&DegreeOfFreedom::setPosition>(
        this, _positions, "setPositions", "_positions");
}

//==============================================================================
void MetaSkeleton::setPositions(const std::vector<std::size_t>& _indices,
                            const Eigen::VectorXs& _positions)
{
  setValuesFromVector<&DegreeOfFreedom::setPosition>(
        this, _indices, _positions, "setPositions", "_positions");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getPositions() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getPosition>(
        this, "getPositions");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getPositions(const std::vector<std::size_t>& _indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getPosition>(
        this, _indices, "getPositions");
}

//==============================================================================
void MetaSkeleton::resetPositions()
{
  applyToAllDofs<&DegreeOfFreedom::resetPosition>(this);
}

//==============================================================================
void MetaSkeleton::setPositionLowerLimit(std::size_t _index, s_t _position)
{
  setValueFromIndex<&DegreeOfFreedom::setPositionLowerLimit>(
        this, _index, _position, "setPositionLowerLimit");
}

//==============================================================================
void MetaSkeleton::setPositionLowerLimits(const Eigen::VectorXs& positions)
{
  setAllValuesFromVector<&DegreeOfFreedom::setPositionLowerLimit>(
        this, positions, "setPositionLowerLimits", "positions");
}

//==============================================================================
void MetaSkeleton::setPositionLowerLimits(
    const std::vector<std::size_t>& indices, const Eigen::VectorXs& positions)
{
  setValuesFromVector<&DegreeOfFreedom::setPositionLowerLimit>(
        this, indices, positions, "setPositionLowerLimits", "positions");
}

//==============================================================================
s_t MetaSkeleton::getPositionLowerLimit(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getPositionLowerLimit>(
        this, _index, "getPositionLowerLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getPositionLowerLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getPositionLowerLimit>(
        this, "getPositionLowerLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getPositionLowerLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getPositionLowerLimit>(
        this, indices, "getPositionLowerLimits");
}

//==============================================================================
void MetaSkeleton::setPositionUpperLimit(std::size_t _index, s_t _position)
{
  setValueFromIndex<&DegreeOfFreedom::setPositionUpperLimit>(
        this, _index, _position, "setPositionUpperLimit");
}

//==============================================================================
void MetaSkeleton::setPositionUpperLimits(const Eigen::VectorXs& positions)
{
  setAllValuesFromVector<&DegreeOfFreedom::setPositionUpperLimit>(
        this, positions, "setPositionUpperLimits", "positions");
}

//==============================================================================
void MetaSkeleton::setPositionUpperLimits(
    const std::vector<std::size_t>& indices, const Eigen::VectorXs& positions)
{
  setValuesFromVector<&DegreeOfFreedom::setPositionUpperLimit>(
        this, indices, positions, "setPositionUpperLimits", "positions");
}

//==============================================================================
s_t MetaSkeleton::getPositionUpperLimit(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getPositionUpperLimit>(
        this, _index, "getPositionUpperLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getPositionUpperLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getPositionUpperLimit>(
        this, "getPositionUpperLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getPositionUpperLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getPositionUpperLimit>(
        this, indices, "getPositionUpperLimits");
}

//==============================================================================
void MetaSkeleton::setVelocity(std::size_t _index, s_t _velocity)
{
  setValueFromIndex<&DegreeOfFreedom::setVelocity>(
        this, _index, _velocity, "setVelocity");
}

//==============================================================================
s_t MetaSkeleton::getVelocity(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getVelocity>(
        this, _index, "getVelocity");
}

//==============================================================================
void MetaSkeleton::setVelocities(const Eigen::VectorXs& _velocities)
{
  setAllValuesFromVector<&DegreeOfFreedom::setVelocity>(
        this, _velocities, "setVelocities", "_velocities");
}

//==============================================================================
void MetaSkeleton::setVelocities(const std::vector<std::size_t>& _indices,
                             const Eigen::VectorXs& _velocities)
{
  setValuesFromVector<&DegreeOfFreedom::setVelocity>(
        this, _indices, _velocities, "setVelocities", "_velocities");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getVelocities() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getVelocity>(
        this, "getVelocities");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getVelocities(const std::vector<std::size_t>& _indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getVelocity>(
        this, _indices, "getVelocities");
}

//==============================================================================
void MetaSkeleton::resetVelocities()
{
  applyToAllDofs<&DegreeOfFreedom::resetVelocity>(this);
}

//==============================================================================
void MetaSkeleton::setVelocityLowerLimit(std::size_t _index, s_t _velocity)
{
  setValueFromIndex<&DegreeOfFreedom::setVelocityLowerLimit>(
        this, _index, _velocity, "setVelocityLowerLimit");
}

//==============================================================================
void MetaSkeleton::setVelocityLowerLimits(const Eigen::VectorXs& velocities)
{
  setAllValuesFromVector<&DegreeOfFreedom::setVelocityLowerLimit>(
        this, velocities, "setVelocityLowerLimits", "velocities");
}

//==============================================================================
void MetaSkeleton::setVelocityLowerLimits(
    const std::vector<std::size_t>& indices, const Eigen::VectorXs& velocities)
{
  setValuesFromVector<&DegreeOfFreedom::setVelocityLowerLimit>(
        this, indices, velocities, "setVelocityLowerLimits", "velocities");
}

//==============================================================================
s_t MetaSkeleton::getVelocityLowerLimit(std::size_t _index)
{
  return getValueFromIndex<&DegreeOfFreedom::getVelocityLowerLimit>(
        this, _index, "getVelocityLowerLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getVelocityLowerLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getVelocityLowerLimit>(
        this, "getVelocityLowerLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getVelocityLowerLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getVelocityLowerLimit>(
        this, indices, "getVelocityLowerLimits");
}

//==============================================================================
void MetaSkeleton::setVelocityUpperLimit(std::size_t _index, s_t _velocity)
{
  setValueFromIndex<&DegreeOfFreedom::setVelocityUpperLimit>(
        this, _index, _velocity, "setVelocityUpperLimit");
}

//==============================================================================
void MetaSkeleton::setVelocityUpperLimits(const Eigen::VectorXs& velocities)
{
  setAllValuesFromVector<&DegreeOfFreedom::setVelocityUpperLimit>(
        this, velocities, "setVelocityUpperLimits", "velocities");
}

//==============================================================================
void MetaSkeleton::setVelocityUpperLimits(
    const std::vector<std::size_t>& indices, const Eigen::VectorXs& velocities)
{
  setValuesFromVector<&DegreeOfFreedom::setVelocityUpperLimit>(
        this, indices, velocities, "setVelocityUpperLimits", "velocities");
}

//==============================================================================
s_t MetaSkeleton::getVelocityUpperLimit(std::size_t _index)
{
  return getValueFromIndex<&DegreeOfFreedom::getVelocityUpperLimit>(
        this, _index, "getVelocityUpperLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getVelocityUpperLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getVelocityUpperLimit>(
        this, "getVelocityUpperLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getVelocityUpperLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getVelocityUpperLimit>(
        this, indices, "getVelocityUpperLimits");
}

//==============================================================================
void MetaSkeleton::setAcceleration(std::size_t _index, s_t _acceleration)
{
  setValueFromIndex<&DegreeOfFreedom::setAcceleration>(
        this, _index, _acceleration, "setAcceleration");
}

//==============================================================================
s_t MetaSkeleton::getAcceleration(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getAcceleration>(
        this, _index, "getAcceleration");
}

//==============================================================================
void MetaSkeleton::setAccelerations(const Eigen::VectorXs& _accelerations)
{
  setAllValuesFromVector<&DegreeOfFreedom::setAcceleration>(
        this, _accelerations, "setAccelerations", "_accelerations");
}

//==============================================================================
void MetaSkeleton::setAccelerations(const std::vector<std::size_t>& _indices,
                                const Eigen::VectorXs& _accelerations)
{
  setValuesFromVector<&DegreeOfFreedom::setAcceleration>(
        this, _indices, _accelerations, "setAccelerations", "_accelerations");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getAccelerations() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getAcceleration>(
        this, "getAccelerations");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getAccelerations(
    const std::vector<std::size_t>& _indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getAcceleration>(
        this, _indices, "getAccelerations");
}

//==============================================================================
void MetaSkeleton::resetAccelerations()
{
  applyToAllDofs<&DegreeOfFreedom::resetAcceleration>(this);
}

//==============================================================================
void MetaSkeleton::setAccelerationLowerLimit(std::size_t _index, s_t _acceleration)
{
  setValueFromIndex<&DegreeOfFreedom::setAccelerationLowerLimit>(
        this, _index, _acceleration, "setAccelerationLowerLimit");
}

//==============================================================================
void MetaSkeleton::setAccelerationLowerLimits(
    const Eigen::VectorXs& accelerations)
{
  setAllValuesFromVector<&DegreeOfFreedom::setAccelerationLowerLimit>(
        this, accelerations, "setAccelerationLowerLimits", "accelerations");
}

//==============================================================================
void MetaSkeleton::setAccelerationLowerLimits(
    const std::vector<std::size_t>& indices,
    const Eigen::VectorXs& accelerations)
{
  setValuesFromVector<&DegreeOfFreedom::setAccelerationLowerLimit>(
        this, indices, accelerations, "setAccelerationLowerLimits",
        "accelerations");
}

//==============================================================================
s_t MetaSkeleton::getAccelerationLowerLimit(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getAccelerationLowerLimit>(
        this, _index, "getAccelerationLowerLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getAccelerationLowerLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getAccelerationLowerLimit>(
        this, "getAccelerationLowerLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getAccelerationLowerLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getAccelerationLowerLimit>(
        this, indices, "getAccelerationLowerLimits");
}

//==============================================================================
void MetaSkeleton::setAccelerationUpperLimit(std::size_t _index, s_t _acceleration)
{
  setValueFromIndex<&DegreeOfFreedom::setAccelerationUpperLimit>(
        this, _index, _acceleration, "setAccelerationUpperLimit");
}

//==============================================================================
void MetaSkeleton::setAccelerationUpperLimits(
    const Eigen::VectorXs& accelerations)
{
  setAllValuesFromVector<&DegreeOfFreedom::setAccelerationUpperLimit>(
        this, accelerations, "setAccelerationUpperLimits", "accelerations");
}

//==============================================================================
void MetaSkeleton::setAccelerationUpperLimits(
    const std::vector<std::size_t>& indices,
    const Eigen::VectorXs& accelerations)
{
  setValuesFromVector<&DegreeOfFreedom::setAccelerationUpperLimit>(
        this, indices, accelerations, "setAccelerationUpperLimits",
        "accelerations");
}

//==============================================================================
s_t MetaSkeleton::getAccelerationUpperLimit(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getAccelerationUpperLimit>(
        this, _index, "getAccelerationUpperLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getAccelerationUpperLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getAccelerationUpperLimit>(
        this, "getAccelerationUpperLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getAccelerationUpperLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getAccelerationUpperLimit>(
        this, indices, "getAccelerationUpperLimits");
}

//==============================================================================
void MetaSkeleton::setForce(std::size_t _index, s_t _force)
{
  setValueFromIndex<&DegreeOfFreedom::setForce>(
        this, _index, _force, "setForce");
}

//==============================================================================
s_t MetaSkeleton::getForce(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getForce>(
        this, _index, "getForce");
}

//==============================================================================
void MetaSkeleton::setForces(const Eigen::VectorXs& _forces)
{
  setAllValuesFromVector<&DegreeOfFreedom::setForce>(
        this, _forces, "setForces", "_forces");
}

//==============================================================================
void MetaSkeleton::setForces(const std::vector<std::size_t>& _indices,
                         const Eigen::VectorXs& _forces)
{
  setValuesFromVector<&DegreeOfFreedom::setForce>(
        this, _indices, _forces, "setForces", "_forces");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getForces() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getForce>(
        this, "getForces");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getForces(const std::vector<std::size_t>& _indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getForce>(
        this, _indices, "getForces");
}

//==============================================================================
void MetaSkeleton::resetGeneralizedForces()
{
  applyToAllDofs<&DegreeOfFreedom::resetForce>(this);
  // Note: This function used to clear the internal forces of SoftBodyNodes as
  // well. Now you should use clearInternalForces for that
}

//==============================================================================
void MetaSkeleton::setForceLowerLimit(std::size_t _index, s_t _force)
{
  setValueFromIndex<&DegreeOfFreedom::setForceLowerLimit>(
        this, _index, _force, "setForceLowerLimit");
}

//==============================================================================
void MetaSkeleton::setForceLowerLimits(const Eigen::VectorXs& forces)
{
  setAllValuesFromVector<&DegreeOfFreedom::setForceLowerLimit>(
        this, forces, "setForceLowerLimits", "forces");
}

//==============================================================================
void MetaSkeleton::setForceLowerLimits(const std::vector<std::size_t>& indices,
                                       const Eigen::VectorXs& forces)
{
  setValuesFromVector<&DegreeOfFreedom::setForceLowerLimit>(
        this, indices, forces, "setForceLowerLimits", "forces");
}

//==============================================================================
s_t MetaSkeleton::getForceLowerLimit(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getForceLowerLimit>(
        this, _index, "getForceLowerLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getForceLowerLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getForceLowerLimit>(
        this, "getForceLowerLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getForceLowerLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getForceLowerLimit>(
        this, indices, "getForceLowerLimits");
}

//==============================================================================
void MetaSkeleton::setForceUpperLimit(std::size_t _index, s_t _force)
{
  setValueFromIndex<&DegreeOfFreedom::setForceUpperLimit>(
        this, _index, _force, "setForceUpperLimit");
}

//==============================================================================
void MetaSkeleton::setForceUpperLimits(const Eigen::VectorXs& forces)
{
  setAllValuesFromVector<&DegreeOfFreedom::setForceUpperLimit>(
        this, forces, "setForceUpperLimits", "forces");
}

//==============================================================================
void MetaSkeleton::setForceUpperLimits(const std::vector<std::size_t>& indices,
                                       const Eigen::VectorXs& forces)
{
  setValuesFromVector<&DegreeOfFreedom::setForceUpperLimit>(
        this, indices, forces, "setForceUpperLimits", "forces");
}

//==============================================================================
s_t MetaSkeleton::getForceUpperLimit(std::size_t _index) const
{
  return getValueFromIndex<&DegreeOfFreedom::getForceUpperLimit>(
        this, _index, "getForceUpperLimit");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getForceUpperLimits() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getForceUpperLimit>(
        this, "getForceUpperLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getForceUpperLimits(
    const std::vector<std::size_t>& indices) const
{
  return getValuesFromVector<&DegreeOfFreedom::getForceUpperLimit>(
        this, indices, "getForceUpperLimits");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getVelocityChanges() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getVelocityChange>(
        this, "getVelocityChanges");
}

//==============================================================================
void MetaSkeleton::setJointConstraintImpulses(const Eigen::VectorXs& _impulses)
{
  setAllValuesFromVector<&DegreeOfFreedom::setConstraintImpulse>(
        this, _impulses, "setJointConstraintImpulses", "_impulses");
}

//==============================================================================
Eigen::VectorXs MetaSkeleton::getJointConstraintImpulses() const
{
  return getValuesFromAllDofs<&DegreeOfFreedom::getConstraintImpulse>(
        this, "getJointConstraintImpulses");
}

//==============================================================================
math::Jacobian MetaSkeleton::getJacobian(
    const JacobianNode* _node,
    const JacobianNode* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  if (_node == _relativeTo)
    return math::Jacobian::Zero(6, getNumDofs());

  const math::Jacobian J = getJacobian(_node);
  const math::Jacobian JRelTo = getJacobian(_relativeTo);
  const Eigen::Isometry3s T = _relativeTo->getTransform(_node);

  const math::Jacobian result = (J - math::AdTJac(T, JRelTo)).eval();

  if (_node == _inCoordinatesOf)
    return result;

  return math::AdRJac(_node->getTransform(_inCoordinatesOf), result);
}

//==============================================================================
math::Jacobian MetaSkeleton::getJacobian(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const JacobianNode* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  if (_node == _relativeTo)
    return math::Jacobian::Zero(6, getNumDofs());

  const math::Jacobian J = getJacobian(_node);
  const math::Jacobian JRelTo = getJacobian(_relativeTo);
  const Eigen::Isometry3s T = _relativeTo->getTransform(_node);

  math::Jacobian result = (J - math::AdTJac(T, JRelTo)).eval();
  result.bottomRows<3>() += result.topRows<3>().colwise().cross(_localOffset);

  if (_node == _inCoordinatesOf)
    return result;

  return math::AdRJac(_node->getTransform(_inCoordinatesOf), result);
}

//==============================================================================
math::LinearJacobian MetaSkeleton::getLinearJacobian(
    const JacobianNode* _node,
    const JacobianNode* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  return getJacobian(_node, _relativeTo, _inCoordinatesOf).bottomRows<3>();
}

//==============================================================================
math::LinearJacobian MetaSkeleton::getLinearJacobian(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const JacobianNode* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  return getJacobian(
        _node, _localOffset, _relativeTo, _inCoordinatesOf).bottomRows<3>();
}

//==============================================================================
math::AngularJacobian MetaSkeleton::getAngularJacobian(
    const JacobianNode* _node,
    const JacobianNode* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  return getJacobian(_node, _relativeTo, _inCoordinatesOf).topRows<3>();
}

//==============================================================================
math::Jacobian MetaSkeleton::getJacobianSpatialDeriv(
    const JacobianNode* _node,
    const JacobianNode* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  if (_node == _relativeTo)
    return math::Jacobian::Zero(6, getNumDofs());

  const math::Jacobian dJ = getJacobianSpatialDeriv(_node);
  const math::Jacobian JRelTo = getJacobian(_relativeTo);
  const math::Jacobian dJRelTo = getJacobianSpatialDeriv(_relativeTo);
  const Eigen::Isometry3s T = _relativeTo->getTransform(_node);
  const Eigen::Vector6s V = _relativeTo->getSpatialVelocity(_node, _relativeTo);
  const math::Jacobian adJRelTo = math::adJac(V, JRelTo);

  const math::Jacobian result = dJ - math::AdTJac(T, dJRelTo + adJRelTo);

  if (_node == _inCoordinatesOf)
    return result;

  return math::AdRJac(_node->getTransform(_inCoordinatesOf), result);
}

//==============================================================================
math::Jacobian MetaSkeleton::getJacobianSpatialDeriv(
    const JacobianNode* _node,
    const Eigen::Vector3s& _localOffset,
    const JacobianNode* _relativeTo,
    const Frame* _inCoordinatesOf) const
{
  if (_node == _relativeTo)
    return math::Jacobian::Zero(6, getNumDofs());

  const math::Jacobian dJ = getJacobianSpatialDeriv(_node);
  const math::Jacobian JRelTo = getJacobian(_relativeTo);
  const math::Jacobian dJRelTo = getJacobianSpatialDeriv(_relativeTo);
  const Eigen::Isometry3s T = _relativeTo->getTransform(_node);
  const Eigen::Vector6s V = _relativeTo->getSpatialVelocity(_node, _relativeTo);
  const math::Jacobian adJRelTo = math::adJac(V, JRelTo);

  math::Jacobian result = dJ - math::AdTJac(T, dJRelTo + adJRelTo);
  result.bottomRows<3>().noalias() += result.topRows<3>().colwise().cross(_localOffset);

  if (_node == _inCoordinatesOf)
    return result;

  return math::AdRJac(_node->getTransform(_inCoordinatesOf), result);
}

//==============================================================================
s_t MetaSkeleton::computeLagrangian() const
{
  return computeKineticEnergy() - computePotentialEnergy();
}

//==============================================================================
s_t MetaSkeleton::getKineticEnergy() const
{
  return computeKineticEnergy();
}

//==============================================================================
s_t MetaSkeleton::getPotentialEnergy() const
{
  return computePotentialEnergy();
}

//==============================================================================
MetaSkeleton::MetaSkeleton()
  : onNameChanged(mNameChangedSignal)
{
  // Do nothing
}

} // namespace dynamics
} // namespace dart



