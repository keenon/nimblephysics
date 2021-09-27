#include "dart/utils/amc/AMCParser.hpp"

#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/FreeJoint.hpp"
#include "dart/dynamics/RevoluteJoint.hpp"
#include "dart/dynamics/ShapeFrame.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/utils/amc/ReadSkeleton.hpp"

namespace dart {

using namespace dynamics;

namespace utils {
namespace amc {

std::pair<std::shared_ptr<dynamics::Skeleton>, Eigen::MatrixXs>
AMCParser::loadAMC(const std::string& asfPath, const std::string& amcPath)
{
  ::Library::Skeleton amcSkel;
  ::ReadSkeleton(asfPath, amcSkel);
  std::vector<double> amcAnimation;
  ::ReadAnimation(amcPath, amcSkel, amcAnimation);

  std::shared_ptr<dynamics::Skeleton> skel = dynamics::Skeleton::create("name");

  std::shared_ptr<BoxShape> box(new BoxShape(Eigen::Vector3s(0.1, 0.1, 0.1)));

  auto rootPair = skel->createJointAndBodyNodePair<dynamics::FreeJoint>();
  ShapeNode* rootShape
      = rootPair.second->createShapeNodeWith<VisualAspect>(box);
  rootShape->getVisualAspect()->setColor(Eigen::Vector3s(0.5, 0.5, 0.5));

  std::vector<dynamics::BodyNode*> nodes;
  for (::Library::Bone& bone : amcSkel.bones)
  {
    BodyNode* parent = bone.parent == -1 ? rootPair.second : nodes[bone.parent];

    auto linkPair
        = skel->createJointAndBodyNodePair<dynamics::FreeJoint>(parent);
    ShapeNode* linkShape
        = linkPair.second->createShapeNodeWith<VisualAspect>(box);
    linkShape->getVisualAspect()->setColor(Eigen::Vector3s(0.5, 0.5, 0.5));

    Eigen::Isometry3s transformFromParent = Eigen::Isometry3s::Identity();
    transformFromParent.translation() = Eigen::Vector3s(
        bone.direction.x * bone.length,
        bone.direction.y * bone.length,
        bone.direction.z * bone.length);
    linkPair.first->setTransformFromChildBodyNode(transformFromParent);

    std::cout << bone.name << ": " << bone.dof << " - " << bone.parent << " - "
              << bone.direction << std::endl;

    nodes.push_back(linkPair.second);
  }

  Eigen::MatrixXs animation = Eigen::MatrixXs::Zero(10, 10);
  return std::pair<std::shared_ptr<dynamics::Skeleton>, Eigen::MatrixXs>(
      skel, animation);
}

} // namespace amc
} // namespace utils
} // namespace dart