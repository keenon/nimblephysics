#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/math/Geometry.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"
#include "dart/utils/DartResourceRetriever.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

// #define ALL_TESTS

using namespace dart;
using namespace biomechanics;
using namespace server;
using namespace realtime;

std::shared_ptr<dynamics::Skeleton> getAmassSkeleton()
{
  /*
Joint positions:
[[-0.00220719 -0.24041004  0.02436019]
 [ 0.05199805 -0.3218177   0.00697495]
 [-0.05743078 -0.33038586  0.01046726]
 [ 0.00219637 -0.11644295 -0.0104045 ]
 [ 0.09824532 -0.71780986  0.01908565]
 [-0.10379907 -0.72144514  0.0096877 ]
 [ 0.00716769  0.02329072  0.0174972 ]
 [ 0.08478529 -1.1499848  -0.01584512]
 [-0.08532786 -1.1467853  -0.02244854]
 [ 0.00500904  0.07972068  0.01835739]
 [ 0.1303093  -1.2110176   0.10490999]
 [-0.11818279 -1.210165    0.10682374]
 [-0.00800311  0.2911238  -0.01365946]
 [ 0.07397135  0.19287294 -0.00326094]
 [-0.07602783  0.19166663 -0.00666485]
 [ 0.00191459  0.38311476  0.03824424]
 [ 0.19150157  0.23803937 -0.01963824]
 [-0.18496998  0.23789403 -0.01319679]
 [ 0.45029777  0.2226639  -0.03997875]
 [-0.4473554   0.22239119 -0.03933729]
 [ 0.7147937   0.2342671  -0.04937898]
 [-0.7177149   0.2315885  -0.04629372]]
Parents:
[-1  0  0  0  1  2  3  4  5  6  7  8  9  9  9 12 13 14 16 17 18 19]
  */
  Eigen::MatrixXs jointPositions = Eigen::MatrixXs(22, 3);
  // clang-format off
  jointPositions << 
 -0.00220719, -0.24041004,  0.02436019,
  0.05199805, -0.3218177,   0.00697495,
 -0.05743078, -0.33038586,  0.01046726,
  0.00219637, -0.11644295, -0.0104045 ,
  0.09824532, -0.71780986,  0.01908565,
 -0.10379907, -0.72144514,  0.0096877 ,
  0.00716769, 0.02329072,  0.0174972 ,
  0.08478529, -1.1499848,  -0.01584512,
 -0.08532786, -1.1467853,  -0.02244854,
  0.00500904, 0.07972068,  0.01835739,
  0.1303093, -1.2110176,   0.10490999,
 -0.11818279, -1.210165,    0.10682374,
 -0.00800311,  0.2911238,  -0.01365946,
  0.07397135,  0.19287294, -0.00326094,
 -0.07602783,  0.19166663, -0.00666485,
  0.00191459,  0.38311476,  0.03824424,
  0.19150157,  0.23803937, -0.01963824,
 -0.18496998,  0.23789403, -0.01319679,
  0.45029777,  0.2226639,  -0.03997875,
 -0.4473554,   0.22239119, -0.03933729,
  0.7147937,   0.2342671,  -0.04937898,
 -0.7177149,   0.2315885,  -0.04629372;
  // clang-format on
  Eigen::VectorXi parents = Eigen::VectorXi(22);
  parents << -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
      18, 19;

  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create();
  std::shared_ptr<dynamics::BoxShape> box
      = std::make_shared<dynamics::BoxShape>(Eigen::Vector3s::Ones() * 0.01);

  /*
    jointProps = nimble.dynamics.FreeJointProperties()
  bodyProps = nimble.dynamics.BodyNodeProperties(
      nimble.dynamics.BodyNodeAspectProperties('rootTransBody'))
  rootFreeJoint, rootBody = human.createFreeJointAndBodyNodePair(None)

  nodes.append(rootBody)
  rootShape = rootBody.createShapeNode(
      nimble.dynamics.MeshShape(
          np.array([1, 1, 1]),
          str(pathlib.Path(__file__).parent.absolute()) + '/meshes/mesh0.dae'))
  rootShape.createVisualAspect().setColor(color)
  rootShape.createCollisionAspect()
  totalVolume += rootShape.getShape().getVolume()
  */

  std::vector<dynamics::BodyNode*> nodes;

  auto rootPair = skel->createJointAndBodyNodePair<dynamics::FreeJoint>();
  dynamics::BodyNode* bodyNode = rootPair.second;
  nodes.push_back(bodyNode);
  bodyNode->createShapeNodeWith<dynamics::VisualAspect>(box);

  /*
  # child joints, each parented to the joint listed in the kintree
  for i in range(1, len(parents)):
    jointProps = nimble.dynamics.BallJointProperties()
    jointProps.mName = 'joint' + str(i)
    bodyProps = nimble.dynamics.BodyNodeProperties(
        nimble.dynamics.BodyNodeAspectProperties('body' + str(i)))
    joint, node = human.createBallJointAndBodyNodePair(nodes[parents[i]],
  jointProps, bodyProps)
  nodes.append(node) shape =
  node.createShapeNode(nimble.dynamics.MeshShape([1, 1, 1], str(
        pathlib.Path(__file__).parent.absolute()) + '/meshes/mesh' + str(i) +
  '.dae'))
  shape.createVisualAspect().setColor(color)
    shape.createCollisionAspect()
    totalVolume += shape.getShape().getVolume()

    childOffset = nimble.math.Isometry3()
    childOffset.set_translation(joint_positions[i] -
  joint_positions[parents[i]]) joint.setTransformFromParentBodyNode(childOffset)
  */
  for (int i = 1; i < parents.size(); i++)
  {
    dynamics::BallJoint::Properties props;
    props.mName = "joint" + std::to_string(i);
    auto childPair
        = nodes[parents(i)]
              ->createChildJointAndBodyNodePair<dynamics::BallJoint>(props);
    dynamics::BodyNode* childBody = childPair.second;
    childBody->createShapeNodeWith<dynamics::VisualAspect>(box);
    nodes.push_back(childBody);

    Eigen::Isometry3s T = Eigen::Isometry3s::Identity();
    T.translation() = jointPositions.row(i) - jointPositions.row(parents(i));
    childPair.first->setTransformFromParentBodyNode(T);
  }

  // Body node 4: left knee
  // Body node 5: right knee
  // Body node 7: left ankle
  // Body node 8: right ankle
  // Body node 10: left foot
  // Body node 11: right foot
  // Body node 12: lower neck
  // Body node 13: left clavicle
  // Body node 14: right clavicle
  // Body node 15: upper neck
  // Body node 16: left shoulder
  // Body node 17: right shoulder
  // Body node 18: left elbow
  // Body node 19: right elbow
  // Body node 20: left wrist
  // Body node 21: right wrist
  skel->getJoint(1)->setName("hip_l");
  skel->getJoint(2)->setName("hip_r");
  skel->getJoint(4)->setName("knee_l");
  skel->getJoint(5)->setName("knee_r");
  skel->getJoint(7)->setName("ankle_l");
  skel->getJoint(8)->setName("ankle_r");
  skel->getJoint(10)->setName("foot_l");
  skel->getJoint(11)->setName("foot_r");
  skel->getJoint(13)->setName("clavicle_l");
  skel->getJoint(14)->setName("clavicle_r");
  skel->getJoint(15)->setName("upper_neck");
  skel->getJoint(16)->setName("shoulder_l");
  skel->getJoint(17)->setName("shoulder_r");
  skel->getJoint(18)->setName("elbow_l");
  skel->getJoint(19)->setName("elbow_r");
  skel->getJoint(20)->setName("wrist_l");
  skel->getJoint(21)->setName("wrist_r");

  return skel;
}

/*
Eigen::VectorXs originalPos = Eigen::VectorXs(37);
originalPos << -0.0359295,
-1.02968,
1.51584,
1.22753,
0.848632,
0.296419,
0.960441,
-0.0517612,
-0.696987,
0.334791,
0.224439,
0.349066,
0.0702492,
1.56196,
0.136896,
0.413461,
0.541744,
0.069056,
-0.0169871,
0.153583,
1.31678,
0.0304888,
0.0545916,
-0.395601,
-0.113049,
1.5708,
0,
0,
0.0842056,
-0.212051,
-0.586747,
-0.157119,
-0.566219,
0.586258,
1.5708,
0.0651074,
-0.246306;
Eigen::VectorXs targetPos = Eigen::VectorXs(69);
targetPos << 0.00691438,
3.00543,
0.248381,
1.21846,
0.936188,
0.373096,
-0.330733,
-0.0641177,
0.0171745,
0.221141,
-0.0747513,
0.0135706,
0.301125,
0.111222,
0.0290149,
0.350494,
0.0271063,
-0.0961933,
0.199713,
0.12509,
0.0392563,
0.00241085,
-0.0178114,
0.00551967,
-0.121972,
0.156309,
-0.00531085,
-0.318662,
-0.0243452,
0.0107559,
0.0849869,
0.0208487,
-0.0171355,
0,
0,
0,
0,
0,
0,
0.0363008,
0.102775,
-0.0170745,
0.0412949,
0.0554488,
-0.496387,
0.0653812,
0.0477812,
0.491221,
0.261686,
0.000216491,
-0.0438005,
0.110739,
-0.157067,
-0.97183,
0.173904,
0.190508,
0.921755,
0.132264,
-0.487179,
0.171333,
-0.0410132,
0.339415,
-0.164047,
-0.120093,
-0.066295,
-0.0631069,
-0.104498,
0.0559579,
0.082662;
*/

#ifdef ALL_TESTS
TEST(SkeletonConverter, BROKEN_IK_TIMESTEP)
{
  std::shared_ptr<dynamics::Skeleton> amass = getAmassSkeleton();
  (void)amass;
  std::shared_ptr<dynamics::Skeleton> osim = OpenSimParser::parseOsim(
      "dart://sample/osim/FullBodyModel-4.0/Rajagopal2015.osim");
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(amass);
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->getBodyNode("tibia_l")->setScale(1.2);

  biomechanics::SkeletonConverter converter(osim, amass);
  converter.linkJoints(
      osim->getJoint("radius_hand_l"), amass->getJoint("wrist_l"));
  converter.linkJoints(
      osim->getJoint("radius_hand_r"), amass->getJoint("wrist_r"));
  converter.linkJoints(osim->getJoint("ankle_l"), amass->getJoint("ankle_l"));
  converter.linkJoints(osim->getJoint("ankle_r"), amass->getJoint("ankle_r"));
  converter.linkJoints(osim->getJoint("mtp_l"), amass->getJoint("foot_l"));
  converter.linkJoints(osim->getJoint("mtp_r"), amass->getJoint("foot_r"));
  converter.linkJoints(
      osim->getJoint("walker_knee_l"), amass->getJoint("knee_l"));
  converter.linkJoints(
      osim->getJoint("walker_knee_r"), amass->getJoint("knee_r"));
  converter.linkJoints(
      osim->getJoint("acromial_l"), amass->getJoint("shoulder_l"));
  converter.linkJoints(
      osim->getJoint("acromial_r"), amass->getJoint("shoulder_r"));
  converter.linkJoints(osim->getJoint("elbow_l"), amass->getJoint("elbow_l"));
  converter.linkJoints(osim->getJoint("elbow_r"), amass->getJoint("elbow_r"));
  converter.linkJoints(osim->getJoint("hip_l"), amass->getJoint("hip_l"));
  converter.linkJoints(osim->getJoint("hip_r"), amass->getJoint("hip_r"));

  converter.rescaleAndPrepTarget();

  Eigen::VectorXs originalPos = Eigen::VectorXs(osim->getNumDofs());
  originalPos << -0.0266395, -1.10191, 1.5708, 1.15151, 0.843671, 0.421078,
      1.27947, -0.161152, -0.385092, 0.478295, 0.193606, 0.0568404, -0.0317355,
      1.20856, 0.0493943, 0.385227, 0.299011, 0.0182451, 0.349066, -0.0612107,
      1.38634, -0.0346526, -0.0763902, -0.167112, -0.0989753, 1.5708,
      0.00210323, 2.59062e-05, -1.22173, 0.610865, -0.672659, 0.0578242,
      -0.710065, 0.846339, 0.11474, -0.0257645, 0.00456933;

  Eigen::VectorXs targetPos = Eigen::VectorXs(amass->getNumDofs());
  targetPos << -0.00592551, 3.12513, 0.227185, 1.15345, 0.941946, 0.487697,
      0.0887218, -0.0422028, -0.0360926, -0.0242451, -0.0240115, -0.00364244,
      0.234155, 0.029322, 0.0261914, 0.133144, -0.284175, -0.0398883, 0.366742,
      0.0072862, 0.0277357, 0.017676, 0.0107679, 0.0129669, -0.197533,
      0.0928309, 0.0595045, -0.251746, -0.10141, 0.040783, 0.0658296,
      -0.00191108, -0.00652646, 0, 0, 0, 0, 0, 0, 0.0098387, 0.0803666,
      0.0149037, 0.0310268, 0.0571799, -0.492749, 0.0295255, 0.0428593,
      0.508205, 0.226749, -0.0338132, -0.0626734, 0.102421, -0.160142,
      -0.962479, 0.193401, 0.211697, 0.935107, 0.108095, -0.435375, 0.161259,
      -0.0410188, 0.53933, -0.211976, -0.109802, -0.0535173, -0.0562375,
      -0.116836, 0.071374, 0.0773777;

  amass->setPositions(targetPos);
  osim->setPositions(originalPos);

  s_t error = converter.fitTarget(-1, 0.005);
  EXPECT_LE(error, 0.005);
}
#endif

// #ifdef ALL_TESTS
TEST(SkeletonConverter, BROKEN_IK_TIMESTEP_2)
{
  std::shared_ptr<dynamics::Skeleton> amass = getAmassSkeleton();
  (void)amass;
  std::shared_ptr<dynamics::Skeleton> osim = OpenSimParser::parseOsim(
      "dart://sample/osim/FullBodyModel-4.0/Rajagopal2015.osim");
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(amass);
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->getBodyNode("tibia_l")->setScale(1.2);

  biomechanics::SkeletonConverter converter(osim, amass);
  converter.linkJoints(
      osim->getJoint("radius_hand_l"), amass->getJoint("wrist_l"));
  converter.linkJoints(
      osim->getJoint("radius_hand_r"), amass->getJoint("wrist_r"));
  converter.linkJoints(osim->getJoint("ankle_l"), amass->getJoint("ankle_l"));
  converter.linkJoints(osim->getJoint("ankle_r"), amass->getJoint("ankle_r"));
  converter.linkJoints(osim->getJoint("mtp_l"), amass->getJoint("foot_l"));
  converter.linkJoints(osim->getJoint("mtp_r"), amass->getJoint("foot_r"));
  converter.linkJoints(
      osim->getJoint("walker_knee_l"), amass->getJoint("knee_l"));
  converter.linkJoints(
      osim->getJoint("walker_knee_r"), amass->getJoint("knee_r"));
  converter.linkJoints(
      osim->getJoint("acromial_l"), amass->getJoint("shoulder_l"));
  converter.linkJoints(
      osim->getJoint("acromial_r"), amass->getJoint("shoulder_r"));
  converter.linkJoints(osim->getJoint("elbow_l"), amass->getJoint("elbow_l"));
  converter.linkJoints(osim->getJoint("elbow_r"), amass->getJoint("elbow_r"));
  converter.linkJoints(osim->getJoint("hip_l"), amass->getJoint("hip_l"));
  converter.linkJoints(osim->getJoint("hip_r"), amass->getJoint("hip_r"));

  converter.rescaleAndPrepTarget();

  Eigen::VectorXs originalPos = Eigen::VectorXs(37);
  originalPos << -0.0359295, -1.02968, 1.51584, 1.22753, 0.848632, 0.296419,
      0.960441, -0.0517612, -0.696987, 0.334791, 0.224439, 0.349066, 0.0702492,
      1.56196, 0.136896, 0.413461, 0.541744, 0.069056, -0.0169871, 0.153583,
      1.31678, 0.0304888, 0.0545916, -0.395601, -0.113049, 1.5708, 0, 0,
      0.0842056, -0.212051, -0.586747, -0.157119, -0.566219, 0.586258, 1.5708,
      0.0651074, -0.246306;
  Eigen::VectorXs targetPos = Eigen::VectorXs(69);
  targetPos << 0.00691438, 3.00543, 0.248381, 1.21846, 0.936188, 0.373096,
      -0.330733, -0.0641177, 0.0171745, 0.221141, -0.0747513, 0.0135706,
      0.301125, 0.111222, 0.0290149, 0.350494, 0.0271063, -0.0961933, 0.199713,
      0.12509, 0.0392563, 0.00241085, -0.0178114, 0.00551967, -0.121972,
      0.156309, -0.00531085, -0.318662, -0.0243452, 0.0107559, 0.0849869,
      0.0208487, -0.0171355, 0, 0, 0, 0, 0, 0, 0.0363008, 0.102775, -0.0170745,
      0.0412949, 0.0554488, -0.496387, 0.0653812, 0.0477812, 0.491221, 0.261686,
      0.000216491, -0.0438005, 0.110739, -0.157067, -0.97183, 0.173904,
      0.190508, 0.921755, 0.132264, -0.487179, 0.171333, -0.0410132, 0.339415,
      -0.164047, -0.120093, -0.066295, -0.0631069, -0.104498, 0.0559579,
      0.082662;

  amass->setPositions(targetPos);
  osim->setPositions(originalPos);

  s_t error = converter.fitTarget(-1, 0.005);
  EXPECT_LE(error, 0.005);
}
// #endif

#ifdef ALL_TESTS
TEST(SkeletonConverter, RAJAGOPAL)
{
  std::shared_ptr<dynamics::Skeleton> amass = getAmassSkeleton();
  (void)amass;
  std::shared_ptr<dynamics::Skeleton> osim = OpenSimParser::parseOsim(
      "dart://sample/osim/FullBodyModel-4.0/Rajagopal2015.osim");
  (void)osim;
  std::shared_ptr<simulation::World> world = simulation::World::create();
  world->addSkeleton(amass);
  world->addSkeleton(osim);
  osim->setPosition(2, -3.14159 / 2);
  osim->setPosition(4, -0.2);
  osim->setPosition(5, 1.0);

  osim->getBodyNode("tibia_l")->setScale(1.2);

  biomechanics::SkeletonConverter converter(osim, amass);
  converter.linkJoints(
      osim->getJoint("radius_hand_l"), amass->getJoint("wrist_l"));
  converter.linkJoints(
      osim->getJoint("radius_hand_r"), amass->getJoint("wrist_r"));
  converter.linkJoints(osim->getJoint("ankle_l"), amass->getJoint("ankle_l"));
  converter.linkJoints(osim->getJoint("ankle_r"), amass->getJoint("ankle_r"));
  converter.linkJoints(osim->getJoint("mtp_l"), amass->getJoint("foot_l"));
  converter.linkJoints(osim->getJoint("mtp_r"), amass->getJoint("foot_r"));
  converter.linkJoints(
      osim->getJoint("walker_knee_l"), amass->getJoint("knee_l"));
  converter.linkJoints(
      osim->getJoint("walker_knee_r"), amass->getJoint("knee_r"));
  converter.linkJoints(
      osim->getJoint("acromial_l"), amass->getJoint("shoulder_l"));
  converter.linkJoints(
      osim->getJoint("acromial_r"), amass->getJoint("shoulder_r"));
  converter.linkJoints(osim->getJoint("elbow_l"), amass->getJoint("elbow_l"));
  converter.linkJoints(osim->getJoint("elbow_r"), amass->getJoint("elbow_r"));
  converter.linkJoints(osim->getJoint("hip_l"), amass->getJoint("hip_l"));
  converter.linkJoints(osim->getJoint("hip_r"), amass->getJoint("hip_r"));

  // Check the joint position Jacobian is accurate
  const s_t THRESHOLD = 1e-7;
  Eigen::MatrixXs posJac
      = osim->getJointWorldPositionsJacobianWrtJointPositions(
          converter.getSourceJoints());
  Eigen::MatrixXs posJac_fd
      = osim->finiteDifferenceJointWorldPositionsJacobianWrtJointPositions(
          converter.getSourceJoints());
  if (!equals(posJac, posJac_fd, THRESHOLD))
  {
    std::cout << "Analytical pos J: " << std::endl << posJac << std::endl;
    std::cout << "FD pos J: " << std::endl << posJac_fd << std::endl;
    std::cout << "Diff: " << std::endl << posJac - posJac_fd << std::endl;
    EXPECT_TRUE(equals(posJac, posJac_fd, THRESHOLD));
    return;
  }

  // Check the joint angle Jacobian is accurate
  Eigen::MatrixXs angleJac
      = osim->getJointWorldPositionsJacobianWrtJointChildAngles(
          converter.getSourceJoints());
  Eigen::MatrixXs angleJac_fd
      = osim->finiteDifferenceJointWorldPositionsJacobianWrtJointChildAngles(
          converter.getSourceJoints());
  if (!equals(angleJac, angleJac_fd, THRESHOLD))
  {
    std::cout << "Analytical angle J: " << std::endl << angleJac << std::endl;
    std::cout << "FD angle J: " << std::endl << angleJac_fd << std::endl;
    std::cout << "Diff: " << std::endl << angleJac - angleJac_fd << std::endl;
    EXPECT_TRUE(equals(angleJac, angleJac_fd, THRESHOLD));
    return;
  }

  // Check the body scale Jacobian is accurate
  Eigen::MatrixXs scaleJac = osim->getJointWorldPositionsJacobianWrtBodyScales(
      converter.getSourceJoints());
  Eigen::MatrixXs scaleJac_fd
      = osim->finiteDifferenceJointWorldPositionsJacobianWrtBodyScales(
          converter.getSourceJoints());
  if (!equals(scaleJac, scaleJac_fd, THRESHOLD))
  {
    for (int i = 0; i < scaleJac.cols(); i++)
    {
      for (int j = 0; j < scaleJac.rows() / 3; j++)
      {
        Eigen::Vector3s dpos_dscale = scaleJac.block(j * 3, i, 3, 1);
        Eigen::Vector3s dpos_dscale_fd = scaleJac_fd.block(j * 3, i, 3, 1);
        if (!equals(dpos_dscale, dpos_dscale_fd, THRESHOLD))
        {
          const dynamics::BodyNode* errorBody = osim->getBodyNode(i);
          Eigen::Matrix3s R = errorBody->getWorldTransform().linear();
          Eigen::Vector3s bodyToParentLocal
              = errorBody->getParentJoint()
                    ->getTransformFromChildBodyNode()
                    .translation();
          Eigen::Vector3s bodyToParentWorld = R * bodyToParentLocal;
          Eigen::Vector3s bodyToChildLocal
              = errorBody->getChildJoint(0)
                    ->getTransformFromParentBodyNode()
                    .translation();
          Eigen::Vector3s bodyToChildWorld = R * bodyToChildLocal;

          std::cout << "Error on scale body \"" << errorBody->getName()
                    << "\" -> joint position \""
                    << converter.getSourceJoints()[j]->getName() << "\""
                    << std::endl;
          std::cout << "Analytical scale J: " << std::endl
                    << dpos_dscale << std::endl;
          std::cout << "FD scale J: " << std::endl
                    << dpos_dscale_fd << std::endl;
          std::cout << "Diff: " << std::endl
                    << dpos_dscale - dpos_dscale_fd << std::endl;
          std::cout << "Diff ./ analytical: " << std::endl
                    << (dpos_dscale - dpos_dscale_fd).cwiseQuotient(dpos_dscale)
                    << std::endl;

          std::cout << "body->parent local: " << std::endl
                    << bodyToParentLocal << std::endl;
          std::cout << "body->parent world: " << std::endl
                    << bodyToParentWorld << std::endl;
          std::cout << "body->child local: " << std::endl
                    << bodyToChildLocal << std::endl;
          std::cout << "body->child world: " << std::endl
                    << bodyToChildWorld << std::endl;

          EXPECT_TRUE(equals(dpos_dscale, dpos_dscale_fd, THRESHOLD));
        }
      }
    }
    EXPECT_TRUE(equals(scaleJac, scaleJac_fd, THRESHOLD));
    return;
  }

  converter.rescaleAndPrepTarget();

  Eigen::VectorXs targetAngles = converter.getTargetJointWorldAngles();
  Eigen::VectorXs sourceAngles = converter.getSourceJointWorldAngles();
  if (!equals(targetAngles, sourceAngles, 1e-10))
  {
    std::cout << "Target angles (corrected): " << std::endl
              << targetAngles << std::endl;
    std::cout << "Source angles: " << std::endl << sourceAngles << std::endl;
    std::cout << "Diff: " << std::endl
              << sourceAngles - targetAngles << std::endl;
    EXPECT_TRUE(equals(targetAngles, sourceAngles, 1e-10));
    return;
  }

  auto retriever = utils::DartResourceRetriever::create();
  common::ResourcePtr ptr
      = retriever->retrieve("dart://sample/osim/amass_test_motion.csv");
  std::string contents = ptr->readAll();

  std::vector<std::vector<s_t>> trajectory;
  std::stringstream contentsStream(contents);
  std::string line;
  while (getline(contentsStream, line, '\n'))
  {
    std::vector<s_t> pose;
    std::stringstream lineStream(line);
    std::string token;
    while (getline(lineStream, token, ','))
    {
      s_t number = atof(token.c_str());
      pose.push_back(number);
    }
    trajectory.push_back(pose);
  }
  Eigen::MatrixXs poses
      = Eigen::MatrixXs::Zero(trajectory[0].size(), trajectory.size());
  for (int i = 0; i < trajectory.size(); i++)
  {
    for (int j = 0; j < trajectory[i].size(); j++)
    {
      poses(j, i) = trajectory[i][j];
    }
  }

  converter.convertMotion(poses, true, -1, 0.005);

  // Uncomment this for local testing
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderSkeleton(osim);
  server->renderSkeleton(amass);
  converter.debugToGUI(server);

  Ticker ticker = Ticker(0.01);

  int cursor = 0;
  ticker.registerTickListener([&](long /*now*/) {
    amass->setPositions(poses.col(cursor % poses.cols()));
    cursor++;
    server->renderSkeleton(amass);
  });
  server->registerConnectionListener([&]() { ticker.start(); });

  server->blockWhileServing();
}
#endif