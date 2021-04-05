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

#ifndef EXAMPLES_JOINTCONSTRAINTS_CONTROLLER_HPP_
#define EXAMPLES_JOINTCONSTRAINTS_CONTROLLER_HPP_

#include <vector>

#include <Eigen/Dense>

#include <dart/dart.hpp>

class Controller
{
public:
  Controller(
      const dart::dynamics::SkeletonPtr& _skel,
      dart::constraint::ConstraintSolver* _collisionSolver,
      s_t _t);
  virtual ~Controller()
  {
  }

  Eigen::VectorXs getTorques()
  {
    return mTorques;
  }
  s_t getTorque(int _index)
  {
    return mTorques[_index];
  }
  void setDesiredDof(int _index, s_t _val)
  {
    mDesiredDofs[_index] = _val;
  }
  void computeTorques(
      const Eigen::VectorXs& _dof, const Eigen::VectorXs& _dofVel);
  dart::dynamics::SkeletonPtr getSkel()
  {
    return mSkel;
  }
  Eigen::VectorXs getDesiredDofs()
  {
    return mDesiredDofs;
  }
  Eigen::MatrixXs getKp()
  {
    return mKp;
  }
  Eigen::MatrixXs getKd()
  {
    return mKd;
  }
  void setConstrForces(const Eigen::VectorXs& _constrForce)
  {
    mConstrForces = _constrForce;
  }

protected:
  bool computeCoP(dart::dynamics::BodyNode* _node, Eigen::Vector3s* _cop);
  Eigen::Vector3s evalLinMomentum(const Eigen::VectorXs& _dofVel);
  Eigen::Vector3s evalAngMomentum(const Eigen::VectorXs& _dofVel);
  Eigen::VectorXs adjustAngMomentum(
      Eigen::VectorXs _deltaMomentum, Eigen::VectorXs _controlledAxis);
  dart::dynamics::SkeletonPtr mSkel;
  dart::constraint::ConstraintSolver* mCollisionHandle;
  Eigen::VectorXs mTorques;
  Eigen::VectorXs mDesiredDofs;
  Eigen::MatrixXs mKp;
  Eigen::MatrixXs mKd;
  int mFrame;
  s_t mTimestep;
  s_t mPreOffset;
  Eigen::VectorXs
      mConstrForces; // SPD utilizes the current info about contact forces
};

#endif // EXAMPLES_JOINTCONSTRAINTS_CONTROLLER_HPP_
