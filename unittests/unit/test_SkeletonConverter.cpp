#include <gtest/gtest.h>

#include "dart/biomechanics/OpenSimParser.hpp"
#include "dart/biomechanics/SkeletonConverter.hpp"
#include "dart/dynamics/BallJoint.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/Joint.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/realtime/Ticker.hpp"
#include "dart/server/GUIWebsocketServer.hpp"

#include "GradientTestUtils.hpp"
#include "TestHelpers.hpp"

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
      = std::make_shared<dynamics::BoxShape>(Eigen::Vector3s::Ones() * 0.1);

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

  /*
  // Uncomment this for local testing
  std::shared_ptr<server::GUIWebsocketServer> server
      = std::make_shared<server::GUIWebsocketServer>();
  server->serve(8070);
  server->renderSkeleton(osim);
  server->renderSkeleton(amass);
  converter.debugToGUI(server);

  Ticker ticker = Ticker(0.01);
  ticker.registerTickListener([&](long now) {
    double progress = (now % 2000) / 2000.0;
    osim->getDof("knee_angle_r")
        ->setPosition(
            progress * osim->getDof("knee_angle_r")->getPositionUpperLimit());
    osim->getDof("knee_angle_l")
        ->setPosition(
            progress * osim->getDof("knee_angle_l")->getPositionUpperLimit());
    // osim->getDof("knee_angle_r_beta")->setPosition(progress);
    // osim->getDof("knee_angle_l_beta")->setPosition(progress);
    server->renderSkeleton(osim);
  });

  server->registerConnectionListener([&]() { ticker.start(); });

  server->blockWhileServing();
  */
}