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

#include <iostream>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "dart/common/Console.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/SimpleFrame.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/Helpers.hpp"
#include "dart/math/Random.hpp"
#include "dart/simulation/World.hpp"
#include "dart/utils/SkelParser.hpp"

#include "TestHelpers.hpp"

using namespace Eigen;
using namespace dart;

//==============================================================================
class DifferentialDynamics : public ::testing::Test
{
public:
  // Get Skel file URI to test.
  const std::vector<common::Uri>& getList() const;

  void compareEquationsOfMotion(const common::Uri& uri);

protected:
  // Sets up the test fixture.
  void SetUp() override;

  // Skel file list.
  std::vector<common::Uri> fileList;
};

//==============================================================================
void DifferentialDynamics::SetUp()
{
  // fileList.push_back("dart://sample/skel/test/single_pendulum_ball_joint.skel");
  // fileList.push_back("dart://sample/skel/test/s_t_pendulum_ball_joint.skel");
  // fileList.push_back("dart://sample/skel/test/single_rigid_body.skel");
  // Create a list of skel files to test with
  //  fileList.push_back("dart://sample/skel/test/chainwhipa.skel");
  fileList.push_back("dart://sample/skel/test/single_pendulum.skel");
  fileList.push_back(
      "dart://sample/skel/test/single_pendulum_euler_joint.skel");
  fileList.push_back("dart://sample/skel/test/single_pendulum_ball_joint.skel");
  fileList.push_back("dart://sample/skel/test/single_rigid_body.skel");
  fileList.push_back("dart://sample/skel/test/s_t_pendulum.skel");
  fileList.push_back(
      "dart://sample/skel/test/s_t_pendulum_euler_joint.skel");
  fileList.push_back("dart://sample/skel/test/s_t_pendulum_ball_joint.skel");
  fileList.push_back(
      "dart://sample/skel/test/serial_chain_revolute_joint.skel");
  //    fileList.push_back("dart://sample/skel/test/serial_chain_eulerxyz_joint.skel");
  //    fileList.push_back("dart://sample/skel/test/serial_chain_ball_joint.skel");
  //    fileList.push_back("dart://sample/skel/test/serial_chain_ball_joint_20.skel");
  //    fileList.push_back("dart://sample/skel/test/serial_chain_ball_joint_40.skel");
  fileList.push_back("dart://sample/skel/test/simple_tree_structure.skel");
  //    fileList.push_back("dart://sample/skel/test/simple_tree_structure_euler_joint.skel");
  //    fileList.push_back("dart://sample/skel/test/simple_tree_structure_ball_joint.skel");
  fileList.push_back("dart://sample/skel/test/tree_structure.skel");
  //    fileList.push_back("dart://sample/skel/test/tree_structure_euler_joint.skel");
  //    fileList.push_back("dart://sample/skel/test/tree_structure_ball_joint.skel");
  fileList.push_back("dart://sample/skel/test/tree_structure_root_free.skel");
  //  fileList.push_back("dart://sample/skel/fullbody1.skel");
}

//==============================================================================
const std::vector<common::Uri>& DifferentialDynamics::getList() const
{
  return fileList;
}

//==============================================================================
template <typename T>
void printComparisonError(
    const std::string& _comparison,
    const std::string& _name,
    const std::string& _frame,
    const T& fk,
    const T& jac)
{
  std::cout << "Disagreement between FK and Jacobian results for "
            << _comparison << " of '" << _name
            << "' with a reference Frame of '" << _frame << "'\n"
            << "FK:  " << fk.transpose() << "\n"
            << "Jac: " << jac.transpose() << "\n";
}

//==============================================================================
template <typename T>
void printComparisonError(
    const std::string& _comparison,
    const std::string& _nameBN,
    const std::string& _nameBNRelativeTo,
    const std::string& _frame,
    const T& fk,
    const T& jac)
{
  std::cout << "Disagreement between FK and relative Jacobian results for "
            << _comparison << " of '" << _nameBN << "' relative to '"
            << _nameBNRelativeTo << "' with a reference Frame of '" << _frame
            << "'\n"
            << "FK:  " << fk.transpose() << "\n"
            << "Jac: " << jac.transpose() << "\n";
}

//==============================================================================
TEST_F(DifferentialDynamics, compareEquationsOfMotion)
{
  using namespace std;
  using namespace Eigen;
  using namespace dart;
  using namespace math;
  using namespace dynamics;
  using namespace simulation;
  using namespace utils;

  //---------------------------- Settings --------------------------------------
  // Number of random state tests for each skeletons
#ifndef NDEBUG // Debug mode
  int nRandomItr = 1;
#else
  int nRandomItr = 100;
#endif

  const s_t pi = constantsd::pi();

  // Lower and upper bound of configuration for system
  s_t qLB = -0.25 * pi;
  s_t qUB = 0.25 * pi;
  s_t dqLB = -0.25 * pi;
  s_t dqUB = 0.25 * pi;
  s_t ddqLB = -0.25 * pi;
  s_t ddqUB = 0.25 * pi;

  simulation::WorldPtr world;

  srand(42);

  for (const auto& uri : getList())
  {
#ifndef NDEBUG
    dtdbg << uri.toString() << std::endl;
#endif

    // TODO(keenon): remove all this special treatment when we remove finite
    // differencing for FreeJoint and BallJoint
    s_t abs_tol_C = 1e-8;
    s_t rel_tol_C = 1e-3; // 0.1 %
    s_t abs_tol_invM = 1e-8;
    s_t rel_tol_invM = 1e-3; // 0.1 %
    /*
    if (uri.toString() == "dart://sample/skel/test/chainwhipa.skel")
    {
    }
    else if (uri.toString() == "dart://sample/skel/test/tree_structure.skel")
    {
      abs_tol_invM = 2e-5;
      rel_tol_invM = 5e-3; // 0.5 %
    }
    else if (
        uri.toString()
        == "dart://sample/skel/test/s_t_pendulum_ball_joint.skel")
    {
      abs_tol_invM = 2e-5;
      rel_tol_invM = 5e-3; // 0.5 %
    }
    else if (
        uri.toString()
        == "dart://sample/skel/test/serial_chain_revolute_joint.skel")
    {
      abs_tol_invM = 3e-3;
      rel_tol_invM = 5e-3; // 0.5 %
    }
    else if (
        uri.toString()
        == "dart://sample/skel/test/serial_chain_ball_joint.skel")
    {
      abs_tol_invM = 2e-5;
      rel_tol_invM = 5e-3; // 0.5 %
    }
    else if (
        uri.toString()
        == "dart://sample/skel/test/tree_structure_root_free.skel")
    {
      abs_tol_invM = 2e-5;
      rel_tol_invM = 5e-3; // 0.5 %
    }
    else if (
        uri.toString()
        == "dart://sample/skel/test/serial_chain_eulerxyz_joint.skel")
    {
    }
    */

    //----------------------------- Tests --------------------------------------
    // Check whether multiplication of mass matrix and its inverse is identity
    // matrix.
    world = utils::SkelParser::readWorld(uri);
    // world->setGravity(Eigen::Vector3s::Zero());  // TODO(JS): Remove
    EXPECT_TRUE(world != nullptr);

    for (std::size_t i = 0; i < world->getNumSkeletons(); ++i)
    {
      dynamics::SkeletonPtr skel = world->getSkeleton(i);
      ASSERT_TRUE(true);

      const int dof = static_cast<int>(skel->getNumDofs());

      // Generate a random state
      VectorXs q = VectorXs::Zero(dof);
      VectorXs dq = VectorXs::Zero(dof);
      VectorXs ddq = VectorXs::Zero(dof);

      for (int j = 0; j < nRandomItr; ++j)
      {
        for (int k = 0; k < dof; ++k)
        {
          q[k] = math::Random::uniform(qLB, qUB);
          dq[k] = math::Random::uniform(dqLB, dqUB);
          ddq[k] = math::Random::uniform(ddqLB, ddqUB);
        }
        skel->setPositions(q);
        skel->setVelocities(dq);
        skel->setAccelerations(ddq);

        for (int i = 0; i < skel->getNumJoints(); i++) {
          // Eigen::Isometry3s Tparent = Eigen::Isometry3s::Identity();
          // Tparent.translation()(2) = 1.0;
          // skel->getJoint(i)->setTransformFromParentBodyNode(Tparent);
          // skel->getJoint(i)->setTransformFromChildBodyNode(Tparent);
          // skel->getJoint(i)->debugRelativeJacobianInPositionSpace();
        }

        // Test derivative of DM/dq*x
        {
          Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
          Eigen::MatrixXs DMX_Dq_numerical = skel->finiteDifferenceJacobianOfM(
              x, neural::WithRespectTo::POSITION);
          Eigen::MatrixXs DMX_Dq_analytic
              = skel->getJacobianOfM(x, neural::WithRespectTo::POSITION);
          const bool res = equals(
              DMX_Dq_analytic, DMX_Dq_numerical, abs_tol_invM, rel_tol_invM);
          EXPECT_TRUE(res);
          if (!res)
          {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
            skel->mDiffMinv.print();
#endif
            for (int i = 0; i < skel->getNumBodyNodes(); i++)
            {
              skel->getBodyNode(i)->debugJacobianOfMForward(
                  neural::WithRespectTo::POSITION, x);
            }
            for (int i = skel->getNumBodyNodes() - 1; i >= 0; i--)
            {
              skel->getBodyNode(i)->debugJacobianOfMBackward(
                  neural::WithRespectTo::POSITION, x, DMX_Dq_numerical);
            }
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] DM/Dq * x analytic\n";
            cout << DMX_Dq_analytic << std::endl;

            cout << "[DEBUG] DM/Dq * x numerical\n";
            cout << DMX_Dq_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << DMX_Dq_analytic - DMX_Dq_numerical << std::endl;

            cout << "[DEBUG] range: "
                 << (DMX_Dq_analytic - DMX_Dq_numerical).minCoeff() << " - "
                 << (DMX_Dq_analytic - DMX_Dq_numerical).maxCoeff()
                 << std::endl;
            return;
          }

          EXPECT_TRUE(skel->getJacobianOfM(x, neural::WithRespectTo::VELOCITY)
                          .isZero());
          EXPECT_TRUE(
              skel->getJacobianOfM(x, neural::WithRespectTo::FORCE).isZero());
          EXPECT_TRUE(skel->finiteDifferenceJacobianOfM(
                              x, neural::WithRespectTo::VELOCITY)
                          .isZero());
          EXPECT_TRUE(
              skel->finiteDifferenceJacobianOfM(x, neural::WithRespectTo::FORCE)
                  .isZero());
        }

        // Test derivative of Coriolis matrix w.r.t. position
        {
          Eigen::MatrixXs C_q_numerical = skel->finiteDifferenceJacobianOfC(
              neural::WithRespectTo::POSITION);
          Eigen::MatrixXs C_q_analytic
              = skel->getJacobianOfC(neural::WithRespectTo::POSITION);

          const bool res
              = equals(C_q_analytic, C_q_numerical, abs_tol_C, rel_tol_C);
          EXPECT_TRUE(res);
          if (!res)
          {
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] DC/Dq analytic\n";
            cout << C_q_analytic << std::endl;

            cout << "[DEBUG] DC/Dq numerical\n";
            cout << C_q_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << C_q_analytic - C_q_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (C_q_analytic - C_q_numerical).maxCoeff() << std::endl;
            
            for (int i = 0; i < skel->getNumBodyNodes(); i++) {
              std::cout << "Checking body node " << i << ":" << std::endl;
              skel->getBodyNode(i)->debugJacobianOfCForward(neural::WithRespectTo::POSITION);
              skel->getBodyNode(i)->debugJacobianOfCBackward(neural::WithRespectTo::POSITION);
            }

            return;
          }
        }

        // Test derivative of inverse dynamics w.r.t. position
        {
          Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
          Eigen::MatrixXs DID_Dq_numerical = skel->finiteDifferenceJacobianOfID(
              x, neural::WithRespectTo::POSITION);
          Eigen::MatrixXs DID_Dq_analytic
              = skel->getJacobianOfID(x, neural::WithRespectTo::POSITION);
          const bool res = equals(
              DID_Dq_analytic, DID_Dq_numerical, abs_tol_invM, rel_tol_invM);
          EXPECT_TRUE(res);
          if (!res)
          {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
            skel->mDiffMinv.print();
#endif
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] DID/Dq * x analytic\n";
            cout << DID_Dq_analytic << std::endl;

            cout << "[DEBUG] DID/Dq * x numerical\n";
            cout << DID_Dq_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << DID_Dq_analytic - DID_Dq_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (DID_Dq_analytic - DID_Dq_numerical).maxCoeff()
                 << std::endl;
          }
        }

        // Test derivative of inverse dynamics w.r.t. velocity
        {
          Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
          Eigen::MatrixXs DID_Ddq_numerical
              = skel->finiteDifferenceJacobianOfID(
                  x, neural::WithRespectTo::VELOCITY);
          Eigen::MatrixXs DID_Ddq_analytic
              = skel->getJacobianOfID(x, neural::WithRespectTo::VELOCITY);
          const bool res = equals(
              DID_Ddq_analytic, DID_Ddq_numerical, abs_tol_invM, rel_tol_invM);
          EXPECT_TRUE(res);
          if (!res)
          {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
            skel->mDiffMinv.print();
#endif
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] DID/Ddq * x analytic\n";
            cout << DID_Ddq_analytic << std::endl;

            cout << "[DEBUG] DID/Ddq * x numerical\n";
            cout << DID_Ddq_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << DID_Ddq_analytic - DID_Ddq_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (DID_Ddq_analytic - DID_Ddq_numerical).maxCoeff()
                 << std::endl;
          }
        }

        // Test derivative of inverse dynamics w.r.t. force
        {
          Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
          Eigen::MatrixXs DID_Df_numerical = skel->finiteDifferenceJacobianOfID(
              x, neural::WithRespectTo::FORCE);
          Eigen::MatrixXs DID_Df_analytic
              = skel->getJacobianOfID(x, neural::WithRespectTo::FORCE);
          const bool res = equals(
              DID_Df_analytic, DID_Df_numerical, abs_tol_invM, rel_tol_invM);
          EXPECT_TRUE(res);
          if (!res)
          {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
            skel->mDiffMinv.print();
#endif
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] DID/Df * x analytic\n";
            cout << DID_Df_analytic << std::endl;

            cout << "[DEBUG] DID/Df * x numerical\n";
            cout << DID_Df_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << DID_Df_analytic - DID_Df_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (DID_Df_analytic - DID_Df_numerical).maxCoeff()
                 << std::endl;
          }
        }

        // Test derivative of Coriolis matrix w.r.t. velocity
        {
          Eigen::MatrixXs C_dq_analytic
              = skel->getJacobianOfC(neural::WithRespectTo::VELOCITY);
          Eigen::MatrixXs dC_numerical = skel->finiteDifferenceJacobianOfC(
              neural::WithRespectTo::VELOCITY);

          const bool res
              = equals(C_dq_analytic, dC_numerical, abs_tol_C, rel_tol_C);
          EXPECT_TRUE(res);
          if (!res)
          {
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] DC/Ddq analytic\n";
            cout << C_dq_analytic << std::endl;

            cout << "[DEBUG] DC/Ddq numerical\n";
            cout << dC_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << C_dq_analytic - dC_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (C_dq_analytic - dC_numerical).maxCoeff() << std::endl;
          }
        }

        // Test derivative of Coriolis matrix w.r.t. force
        EXPECT_TRUE(
            skel->getJacobianOfC(neural::WithRespectTo::FORCE).isZero());
        EXPECT_TRUE(
            skel->finiteDifferenceJacobianOfC(neural::WithRespectTo::FORCE)
                .isZero());

        /*
        // TODO: uncomment this if we ever start using the Direct Minv Jacobian again
        // Test derivative of M^{-1}x w.r.t. position/velocity/acceleration
        {
          Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
          Eigen::MatrixXs DMinvX_Dq_numerical
              = skel->finiteDifferenceJacobianOfMinv(
                  x, neural::WithRespectTo::POSITION);
          Eigen::MatrixXs DMinvX_Dq_analytic = skel->getJacobianOfMinv_Direct(
              x, neural::WithRespectTo::POSITION);
          const bool res = equals(
              DMinvX_Dq_analytic,
              DMinvX_Dq_numerical,
              abs_tol_invM,
              rel_tol_invM);
          EXPECT_TRUE(res);
          if (!res)
          {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
            skel->mDiffMinv.print();
#endif
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] D(M^{-1})/Dq * x analytic\n";
            cout << DMinvX_Dq_analytic << std::endl;

            cout << "[DEBUG] D(M^{-1})/Dq * x numerical\n";
            cout << DMinvX_Dq_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << DMinvX_Dq_analytic - DMinvX_Dq_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (DMinvX_Dq_analytic - DMinvX_Dq_numerical).maxCoeff()
                 << std::endl;
            
            for (int i = 0; i < skel->getNumBodyNodes(); i++) {
              std::cout << "Checking body node " << i << ":" << std::endl;
              skel->getBodyNode(i)->debugJacobianOfMinvXBackward(neural::WithRespectTo::POSITION, x);
              skel->getBodyNode(i)->debugJacobianOfMinvXForward(neural::WithRespectTo::POSITION, x);
            }
            // TODO
          }

          EXPECT_TRUE(
              skel->getJacobianOfMinv_Direct(x, neural::WithRespectTo::VELOCITY)
                  .isZero());
          EXPECT_TRUE(
              skel->getJacobianOfMinv_Direct(x, neural::WithRespectTo::FORCE)
                  .isZero());
          EXPECT_TRUE(skel->finiteDifferenceJacobianOfMinv(
                              x, neural::WithRespectTo::VELOCITY)
                          .isZero());
          EXPECT_TRUE(skel->finiteDifferenceJacobianOfMinv(
                              x, neural::WithRespectTo::FORCE)
                          .isZero());
        }
        */

        // Test derivative of M^{-1}x (ID) w.r.t. position/velocity/acceleration
        {
          Eigen::VectorXs x = Eigen::VectorXs::Random(dof);
          Eigen::MatrixXs DMinvX_Dq_numerical
              = skel->finiteDifferenceJacobianOfMinv(
                  x, neural::WithRespectTo::POSITION);
          Eigen::MatrixXs DMinvX_Dq_analytic
              = skel->getJacobianOfMinv_ID(x, neural::WithRespectTo::POSITION);
          const bool res = equals(
              DMinvX_Dq_analytic,
              DMinvX_Dq_numerical,
              abs_tol_invM,
              rel_tol_invM);
          EXPECT_TRUE(res);
          if (!res)
          {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
            skel->mDiffMinv.print();
#endif
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] D(M^{-1})/Dq * x analytic\n";
            cout << DMinvX_Dq_analytic << std::endl;

            cout << "[DEBUG] D(M^{-1})/Dq * x numerical\n";
            cout << DMinvX_Dq_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << DMinvX_Dq_analytic - DMinvX_Dq_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (DMinvX_Dq_analytic - DMinvX_Dq_numerical).maxCoeff()
                 << std::endl;
          }

          EXPECT_TRUE(
              skel->getJacobianOfMinv_ID(x, neural::WithRespectTo::VELOCITY)
                  .isZero());
          EXPECT_TRUE(
              skel->getJacobianOfMinv_ID(x, neural::WithRespectTo::FORCE)
                  .isZero());
          EXPECT_TRUE(skel->finiteDifferenceJacobianOfMinv(
                              x, neural::WithRespectTo::VELOCITY)
                          .isZero());
          EXPECT_TRUE(skel->finiteDifferenceJacobianOfMinv(
                              x, neural::WithRespectTo::FORCE)
                          .isZero());
        }

        // Test derivative of forward dynamics w.r.t. position
        {
          Eigen::MatrixXs DFD_Dq_numerical = skel->finiteDifferenceJacobianOfFD(
              neural::WithRespectTo::POSITION);
          Eigen::MatrixXs DFD_Dq_analytic
              = skel->getJacobianOfFD(neural::WithRespectTo::POSITION);
          const bool res = equals(
              DFD_Dq_analytic, DFD_Dq_numerical, abs_tol_invM, rel_tol_invM);
          EXPECT_TRUE(res);
          if (!res)
          {
#ifdef DART_DEBUG_ANALYTICAL_DERIV
            skel->mDiffMinv.print();
#endif
            cout << "[DEBUG] URI: " << uri.toString() << std::endl;

            cout << "[DEBUG] DFD/Dq * x analytic\n";
            cout << DFD_Dq_analytic << std::endl;

            cout << "[DEBUG] DFD/Dq * x numerical\n";
            cout << DFD_Dq_numerical << std::endl;

            cout << "[DEBUG] Diff\n";
            cout << DFD_Dq_analytic - DFD_Dq_numerical << std::endl;

            cout << "[DEBUG] max(Diff): "
                 << (DFD_Dq_analytic - DFD_Dq_numerical).maxCoeff()
                 << std::endl;
          }
        }
      }
    }
  }
}
