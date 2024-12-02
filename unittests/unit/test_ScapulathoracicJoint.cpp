#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/dart.hpp"
#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/EulerJoint.hpp"
#include "dart/dynamics/ScapulathoracicJoint.hpp"
#include "dart/dynamics/detail/BodyNodePtr.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/math/LinearFunction.hpp"
#include "dart/math/MathTypes.hpp"
#include "dart/server/GUIRecording.hpp"

#include "TestHelpers.hpp"

using namespace dart;

#define ALL_TESTS
// #define GUI_TESTS

//==============================================================================
// We know the Jacobians are correct for a number of joints, so we can check our
// finite differencing works correctly
bool verifyJacobianFiniteDifferencing(dynamics::Joint* shoulder)
{
  Eigen::VectorXs pos = shoulder->getPositions();
  Eigen::VectorXs vel = shoulder->getVelocities();

  math::Jacobian j = shoulder->getRelativeJacobian();
  math::Jacobian j_fd = shoulder->finiteDifferenceRelativeJacobian();

  if (!equals(j, j_fd, 1e-8))
  {
    std::cout << "relativeJacobian: " << std::endl;
    std::cout << "Analytical j: " << std::endl << j << std::endl;
    std::cout << "FD j: " << std::endl << j_fd << std::endl;
    std::cout << "Diff: " << std::endl << j - j_fd << std::endl;
    EXPECT_TRUE(equals(j, j_fd, 1e-8));
    return false;
  }
  return true;
}

//==============================================================================
bool verifyScapulathoracicJoint(
    dynamics::ScapulathoracicJoint* shoulder, s_t TEST_THRESHOLD)
{
  Eigen::VectorXs pos = shoulder->getPositions();
  Eigen::VectorXs vel = shoulder->getVelocities();

  math::Jacobian j = shoulder->getRelativeJacobian();
  math::Jacobian j_fd = shoulder->finiteDifferenceRelativeJacobian();

  if (!equals(j, j_fd, TEST_THRESHOLD))
  {
    std::cout << "relativeJacobian: " << std::endl;
    std::cout << "Analytical j: " << std::endl << j << std::endl;
    std::cout << "FD j: " << std::endl << j_fd << std::endl;
    std::cout << "Diff: " << std::endl << j - j_fd << std::endl;
    EXPECT_TRUE(equals(j, j_fd, TEST_THRESHOLD));
    return false;
  }

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      Eigen::MatrixXs scratch = shoulder->analyticalScratch(i, j);
      Eigen::MatrixXs scratch_fd = shoulder->finiteDifferenceScratch(i, j);
      if (!equals(scratch, scratch_fd, TEST_THRESHOLD))
      {
        std::cout << "Scatch failed for Jac wrt " << i << " wrt " << j
                  << std::endl;
        std::cout << "Analytical scratch: " << std::endl
                  << scratch << std::endl;
        std::cout << "FD scratch: " << std::endl << scratch_fd << std::endl;
        std::cout << "Diff: " << std::endl << scratch - scratch_fd << std::endl;
        EXPECT_TRUE(equals(scratch, scratch_fd, TEST_THRESHOLD));
        return false;
      }
    }

    math::Jacobian dj = shoulder->getRelativeJacobianDerivWrtPositionStatic(i);
    math::Jacobian dj_fd
        = shoulder->finiteDifferenceRelativeJacobianDerivWrtPosition(i);

    if (!equals(dj, dj_fd, TEST_THRESHOLD))
    {
      std::cout << "FD J: " << std::endl << j_fd << std::endl;

      std::cout << "relativeJacobianDeriv(index=" << i << "): " << std::endl;
      std::cout << "Analytical dj: " << std::endl << dj << std::endl;
      std::cout << "FD dj: " << std::endl << dj_fd << std::endl;
      std::cout << "Diff: " << std::endl << dj - dj_fd << std::endl;
      EXPECT_TRUE(equals(dj, dj_fd, TEST_THRESHOLD));
      return false;
    }
  }

  math::Jacobian dj_dt = shoulder->getRelativeJacobianTimeDeriv();
  math::Jacobian dj_dt_fd
      = shoulder->finiteDifferenceRelativeJacobianTimeDeriv();

  if (!equals(dj_dt, dj_dt_fd, TEST_THRESHOLD))
  {
    std::cout << "relativeJacobianTimeDeriv: " << std::endl;
    std::cout << "Analytical dj_dt: " << std::endl << dj_dt << std::endl;
    std::cout << "FD dj_dt: " << std::endl << dj_dt_fd << std::endl;
    std::cout << "Diff: " << std::endl << dj_dt - dj_dt_fd << std::endl;
    EXPECT_TRUE(equals(dj_dt, dj_dt_fd, TEST_THRESHOLD));
    return false;
  }

  for (int i = 0; i < shoulder->getNumDofs(); i++)
  {
    ////////////////////////////////////////////////////////////////////////////////
    // Check d/dt d/dx of relative Jacobians
    ////////////////////////////////////////////////////////////////////////////////

    math::Jacobian dj_dt_dp
        = shoulder->getRelativeJacobianTimeDerivDerivWrtPosition(i);
    math::Jacobian dj_dt_dp_fd
        = shoulder->finiteDifferenceRelativeJacobianTimeDerivDerivWrtPosition(
            i);

    if (!equals(dj_dt_dp, dj_dt_dp_fd, TEST_THRESHOLD))
    {
      std::cout << "getRelativeJacobianTimeDerivDerivWrtPosition(index=" << i
                << "): " << std::endl;
      std::cout << "Analytical dj dt dp: " << std::endl
                << dj_dt_dp << std::endl;
      std::cout << "FD dj dt dp: " << std::endl << dj_dt_dp_fd << std::endl;
      std::cout << "Diff: " << std::endl << dj_dt_dp - dj_dt_dp_fd << std::endl;
      EXPECT_TRUE(equals(dj_dt_dp, dj_dt_dp_fd, TEST_THRESHOLD));
      return false;
    }

    math::Jacobian dj_dt_dv
        = shoulder->getRelativeJacobianTimeDerivDerivWrtVelocity(i);
    math::Jacobian dj_dt_dv_fd
        = shoulder->finiteDifferenceRelativeJacobianTimeDerivDerivWrtVelocity(
            i);

    if (!equals(dj_dt_dv, dj_dt_dv_fd, TEST_THRESHOLD))
    {
      std::cout << "getRelativeJacobianTimeDerivDerivWrtVelocity(index=" << i
                << "): " << std::endl;
      std::cout << "Analytical dj dt dv: " << std::endl
                << dj_dt_dv << std::endl;
      std::cout << "FD dj dt dv: " << std::endl << dj_dt_dv_fd << std::endl;
      std::cout << "Diff: " << std::endl << dj_dt_dv - dj_dt_dv_fd << std::endl;
      EXPECT_TRUE(equals(dj_dt_dv, dj_dt_dv_fd, TEST_THRESHOLD));
      return false;
    }
  }

  return true;
}

//==============================================================================
#ifdef GUI_TESTS
TEST(ScapulathoracicJoint, LOAD_RAJAGOPAL_OPENSIM)
{
  auto osim = biomechanics::OpenSimParser::parseOsim(
      "dart://sample/osim/Rajagopal_ThoracoscapularShoulder/"
      "Rajagopal2015_CMUMarkerSet_neck_manual_v2_ThoracoscapularShoulder_"
      "torquedriven_ball_shoulders.osim");
  server::GUIRecording server;
  server.setFramesPerSecond(20);
  server.renderBasis();
  server.renderSkeleton(osim.skeleton);
  server.saveFrame();
  server.writeFramesJson("../../../javascript/src/data/movement2.bin");
}
#endif

//==============================================================================
#ifdef GUI_TESTS
TEST(ScapulathoracicJoint, LOAD_SHOULDER_OPENSIM)
{
  auto osim = biomechanics::OpenSimParser::parseOsim(
      "dart://sample/osim/ScapulaModel/"
      "ScapulothoracicJoint_Shoulder_NoConstraints_Offset.osim");

  server::GUIRecording server;
  server.setFramesPerSecond(20);
  server.renderBasis();
  osim.skeleton->getDof("scapula_abduction")->setPosition(M_PI / 4);
  // osim.skeleton->getDof("scapula_elevation")->setPosition(M_PI / 4);
  // osim.skeleton->getDof("scapula_upward_rot")->setPosition(M_PI / 4);
  // osim.skeleton->getDof("scapula_winging")->setPosition(M_PI / 4);
  osim.skeleton->getDof("scapula_winging")->setPosition(10.0 * M_PI / 180.0);
  // osim.skeleton->getDof("shoulder_elv")->setPosition(1.5);
  // osim.skeleton->getDof("plane_of_elev")->setPosition(1.5);
  // osim.skeleton->getDof("axial_rot")->setPosition(1.5);
  server.renderSkeleton(osim.skeleton);
  for (int i = 0; i < osim.skeleton->getNumJoints(); i++)
  {
    auto* joint = osim.skeleton->getJoint(i);
    if (joint->getType() == ScapulathorasicJoint::getStaticType())
    {
      dynamics::ScapulathorasicJoint* scap
          = static_cast<dynamics::ScapulathorasicJoint*>(joint);
      Eigen::Vector3s radii = scap->getEllipsoidRadii();
      Eigen::Vector3s worldPos = osim.skeleton->getJointWorldPosition(i);
      (void)worldPos;
      Eigen::Isometry3s worldT = scap->getParentBodyNode()->getWorldTransform()
                                 * scap->getTransformFromParentBodyNode();
      (void)worldT;
      server.createSphere(
          "joint" + std::to_string(i),
          radii,
          worldT.translation(),
          Eigen::Vector4s(1, 0, 0, 0.3));
      server.setObjectRotation(
          "joint" + std::to_string(i), math::matrixToEulerXYZ(worldT.linear()));
      server.renderBasis(
          0.1,
          "joint" + std::to_string(i),
          worldT.translation(),
          math::matrixToEulerXYZ(worldT.linear()));
    }
    if (joint->getType() == EulerJoint::getStaticType())
    {
      Eigen::Isometry3s worldT = joint->getParentBodyNode()->getWorldTransform()
                                 * joint->getTransformFromParentBodyNode();
      server.renderBasis(
          0.1,
          "joint" + std::to_string(i),
          worldT.translation(),
          math::matrixToEulerXYZ(worldT.linear()));
    }
  }
  server.saveFrame();
  server.writeFramesJson("../../../javascript/src/data/movement2.bin");
}
#endif

/*
//==============================================================================
TEST(ScapulathoracicJoint, DEBUG_RANGE_OF_MOTION_TO_GUI)
{
  server::GUIRecording server;
  server.setFramesPerSecond(20);

  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  auto pair
      = skel->createJointAndBodyNodePair<dynamics::ScapulathorasicJoint>();
  ScapulathorasicJoint* joint = pair.first;
  joint->setAxisOrder(EulerJoint::AxisOrder::XZY);
  BodyNode* body = pair.second;
  std::shared_ptr<dynamics::BoxShape> boxShape
      = std::make_shared<dynamics::BoxShape>(Eigen::Vector3s(0.1, 0.1, 0.01));
  body->createShapeNodeWith<dynamics::VisualAspect>(boxShape);

  // Render the ellipse that we'll be sliding the scapula along the top of
  server.createSphere(
      "scapula_ellipse",
      joint->getEllipsoidRadii(),
      Eigen::Vector3s::Zero(),
      Eigen::Vector4s(0.5, 0.5, 1.0, 0.5),
      "",
      false,
      false);

  // Do the whole range of motion
  for (int i = -10; i < 10; i++)
  {
    for (int j = -10; j < 10; j++)
    {
      for (int k = -10; k < 10; k++)
      {
        Eigen::Vector3s pos = Eigen::Vector3s(i * 0.1, j * 0.1, k * 0.1);
        skel->setPositions(pos);
        server.renderSkeleton(skel);
        server.saveFrame();
      }
    }
  }

  server.writeFramesJson("../../../javascript/src/data/scapula_test.bin");
}
*/

//==============================================================================
#ifdef ALL_TESTS
TEST(ScapulathoracicJoint, EulerJacobian)
{
  EulerJoint::Properties props;
  EulerJoint joint(props);
  joint.setAxisOrder(EulerJoint::AxisOrder::XZY);

  // Check at random positions
  for (int i = 0; i < 10; i++)
  {
    joint.setPositions(Eigen::VectorXs::Random(joint.getNumDofs()));
    joint.setVelocities(Eigen::VectorXs::Random(joint.getNumDofs()));

    /*
    joint.setPosition(0, 0.0);
    joint.setVelocity(0, 1.0);
    */
    std::cout << "Testing: " << joint.getPositions() << ".."
              << joint.getVelocities() << std::endl;

    if (!verifyJacobianFiniteDifferencing(&joint))
    {
      return;
    }
  }
}
#endif

//==============================================================================
#ifdef ALL_TESTS
TEST(ScapulathorasicJoint, ScapulothorasicJacobians)
{
  ScapulathoracicJoint::Properties props;
  ScapulathoracicJoint joint(props);
  joint.setAxisOrder(EulerJoint::AxisOrder::XZY);

  // TODO: special case around body scaling changing the ellipse scaling

  // Set the parameters of the example shoulder
  joint.setEllipsoidRadii(Eigen::Vector3s(0.07, 0.15, 0.07));
  // joint.setEllipsoidRadii(Eigen::Vector3s::Ones());
  Eigen::Isometry3s transformFromParent = Eigen::Isometry3s::Identity();
  transformFromParent.translation() = Eigen::Vector3s(-0.02, -0.0173, 0.07);
  transformFromParent.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3s(0, -0.87, 0));
  joint.setTransformFromParentBodyNode(transformFromParent);
  Eigen::Isometry3s transformFromChild = Eigen::Isometry3s::Identity();
  transformFromChild.translation()
      = Eigen::Vector3s(-0.05982, -0.03904, -0.056);
  transformFromChild.linear()
      = math::eulerXYZToMatrix(Eigen::Vector3s(-0.5181, -1.1416, -0.2854));

  // joint.setWingingAxisDirection(1);
  // joint.setWingingAxisOffset(Eigen::Vector2s::Ones());

  joint.setPositions(Eigen::Vector4s::Zero());
  joint.setVelocities(Eigen::Vector4s::Zero());
  std::cout << "Testing zero pos and zero vel, with _no_ child transform"
            << std::endl;

  if (!verifyScapulathoracicJoint(&joint, 1e-9))
  {
    return;
  }

  joint.setPositions(Eigen::Vector4s::Unit(3));
  std::cout << "Testing winging pos and zero vel, with _no_ child transform"
            << std::endl;
  if (!verifyScapulathoracicJoint(&joint, 1e-9))
  {
    return;
  }

  for (int i = 0; i < 3; i++)
  {
    joint.setPositions(Eigen::Vector4s::Unit(i));
    std::cout << "Testing euler pos(" << std::to_string(i)
              << ")=1, zero vel, with _no_ child transform" << std::endl;
    if (!verifyScapulathoracicJoint(&joint, 1e-9))
    {
      return;
    }
  }

  joint.setPositions(Eigen::Vector4s::Zero());
  joint.setVelocities(Eigen::Vector4s::Zero());
  joint.setTransformFromChildBodyNode(transformFromChild);
  std::cout << "Testing zero pos and zero vel, _with_ a child transform"
            << std::endl;

  if (!verifyScapulathoracicJoint(&joint, 1e-9))
  {
    return;
  }

  joint.setPositions(Eigen::Vector4s::Unit(3));
  std::cout << "Testing winging pos and zero vel, _with_ a child transform"
            << std::endl;
  if (!verifyScapulathoracicJoint(&joint, 1e-9))
  {
    return;
  }

  // Check at random positions
  for (int i = 0; i < 10; i++)
  {
    joint.setPositions(Eigen::Vector4s::Random());
    joint.setVelocities(Eigen::Vector4s::Random());
    // Eigen::Vector4s vel = Eigen::Vector4s::Random();
    // vel(3) = 0;
    // joint.setVelocities(vel);

    /*
    joint.setPosition(0, 0.0);
    joint.setVelocity(0, 1.0);
    */
    std::cout << "Testing: " << joint.getPositions() << ".."
              << joint.getVelocities() << std::endl;

    if (!verifyScapulathoracicJoint(&joint, 1e-9))
    {
      return;
    }
  }
}
#endif
