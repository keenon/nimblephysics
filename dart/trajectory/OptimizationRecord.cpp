#include "dart/trajectory/OptimizationRecord.hpp"

#include <sstream>
#include <unordered_map>
#include <vector>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/BoxShape.hpp"
#include "dart/dynamics/ShapeNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/simulation/World.hpp"

using namespace dart;

namespace dart {
namespace trajectory {

//==============================================================================
OptimizationRecord::OptimizationRecord() : mSuccess(false)
{
}

//==============================================================================
void OptimizationRecord::setSuccess(bool success)
{
  mSuccess = success;
}

//==============================================================================
void OptimizationRecord::registerIteration(
    int index,
    const TrajectoryRollout* rollout,
    double loss,
    double constraintViolation)
{
  mSteps.emplace_back(index, rollout, loss, constraintViolation);
}

//==============================================================================
/// Returns the number of steps that were registered
int OptimizationRecord::getNumSteps()
{
  return mSteps.size();
}

//==============================================================================
/// This returns the step record for this index
const OptimizationStep& OptimizationRecord::getStep(int index)
{
  return mSteps.at(index);
}

//==============================================================================
/// Small helper utility
void vec3ToJson(std::stringstream& json, const Eigen::Vector3d& vec)
{
  json << "[" << vec(0) << "," << vec(1) << "," << vec(2) << "]";
}

//==============================================================================
/// Small helper utility
void vecXToJson(std::stringstream& json, const Eigen::VectorXd& vec)
{
  json << "[";
  for (int i = 0; i < vec.size(); i++)
  {
    json << vec(i);
    if (i < vec.size() - 1)
      json << ",";
  }
  json << "]";
}

//==============================================================================
/// This converts this optimization record into a JSON blob we can display on
/// our web GUI
std::string OptimizationRecord::toJson(std::shared_ptr<simulation::World> world)
{
  std::stringstream json;

  json << "{";
  json << "\"world\": [";

  std::vector<dynamics::BodyNode*> bodies = world->getAllBodyNodes();

  for (int i = 0; i < bodies.size(); i++)
  {
    auto bodyNode = bodies[i];
    auto skel = bodyNode->getSkeleton();
    /*
    {
      name: "skel.node1",
      shapes: [
        {
          type: "box",
          size: [1, 2, 3],
          color: [1, 2, 3],
          pos: [0, 0, 0],
          angle: [0, 0, 0]
        }
      ],
      pos: [0, 0, 0],
      angle: [0, 0, 0]
    }
    */
    json << "{";
    std::string name = skel->getName() + "." + bodyNode->getName();
    json << "\"name\": \"" << name << "\",";
    json << "\"shapes\": [";
    const std::vector<dynamics::ShapeNode*> visualShapeNodes
        = bodyNode->getShapeNodesWith<dynamics::VisualAspect>();
    for (int j = 0; j < visualShapeNodes.size(); j++)
    {
      json << "{";
      auto shape = visualShapeNodes[j];
      dynamics::ShapePtr shapePtr = shape->getShape();

      if (shapePtr->is<dynamics::BoxShape>())
      {
        const auto box = static_cast<const dynamics::BoxShape*>(shapePtr.get());
        json << "\"type\": \"box\",";
        const Eigen::Vector3d& size = box->getSize();
        json << "\"size\": ";
        vec3ToJson(json, size);
        json << ",";
      }

      dynamics::VisualAspect* visual = shape->getVisualAspect(false);
      json << "\"color\": ";
      vec3ToJson(json, visual->getColor());
      json << ",";

      Eigen::Vector3d relativePos = shape->getRelativeTranslation();
      json << "\"pos\": ";
      vec3ToJson(json, relativePos);
      json << ",";

      Eigen::Vector3d relativeAngle
          = math::matrixToEulerXYZ(shape->getRelativeRotation());
      json << "\"angle\": ";
      vec3ToJson(json, relativeAngle);

      json << "}";
      if (j < visualShapeNodes.size() - 1)
      {
        json << ",";
      }
    }
    json << "],";
    const Eigen::Isometry3d& bodyTransform = bodyNode->getWorldTransform();
    json << "\"pos\":";
    vec3ToJson(json, bodyTransform.translation());
    json << ",";
    json << "\"angle\":";
    vec3ToJson(json, math::matrixToEulerXYZ(bodyTransform.linear()));
    json << "}";
    if (i < bodies.size() - 1)
    {
      json << ",";
    }
  }

  json << "]";

  Eigen::VectorXd originalWorldPos = world->getPositions();

  json << ",\"record\": [";
  for (int i = 0; i < getNumSteps(); i++)
  {
    json << "{";
    const trajectory::OptimizationStep& step = getStep(i);
    json << "\"index\": " << step.index << ",";
    json << "\"loss\": " << step.loss << ",";
    json << "\"constraintViolation\": " << step.constraintViolation << ",";
    int timesteps = step.rollout->getPoses("identity").cols();
    json << "\"timesteps\": " << timesteps << ",";
    json << "\"trajectory\": {";

    // Initialize a map to hold everything
    std::unordered_map<std::string, Eigen::MatrixXd> map;
    for (int i = 0; i < bodies.size(); i++)
    {
      auto bodyNode = bodies[i];
      std::string name
          = bodyNode->getSkeleton()->getName() + "." + bodyNode->getName();
      // 6 rows: pos_x, pos_y, pos_z, rot_x, rot_y, rot_z
      map[name] = Eigen::MatrixXd::Zero(6, timesteps);
    }

    // Fill the map with every timestep
    for (int t = 0; t < timesteps; t++)
    {
      world->setPositions(step.rollout->getPosesConst("identity").col(t));
      for (int i = 0; i < bodies.size(); i++)
      {
        auto bodyNode = bodies[i];
        std::string name
            = bodyNode->getSkeleton()->getName() + "." + bodyNode->getName();
        const Eigen::Isometry3d& bodyTransform = bodyNode->getWorldTransform();

        // 6 rows: pos_x, pos_y, pos_z, rot_x, rot_y, rot_z
        Eigen::Vector6d state = Eigen::Vector6d::Zero();
        state.head<3>() = bodyTransform.translation();
        state.tail<3>() = math::matrixToEulerXYZ(bodyTransform.linear());
        map[name].col(t) = state;
      }
    }

    // Print the map
    bool isFirst = true;
    for (const auto& pair : map)
    {
      if (isFirst)
        isFirst = false;
      else
        json << ",";
      json << "\"" << pair.first << "\": {";
      json << "\"pos_x\": ";
      vecXToJson(json, pair.second.row(0));
      json << ",\"pos_y\": ";
      vecXToJson(json, pair.second.row(1));
      json << ",\"pos_z\": ";
      vecXToJson(json, pair.second.row(2));
      json << ",\"rot_x\": ";
      vecXToJson(json, pair.second.row(3));
      json << ",\"rot_y\": ";
      vecXToJson(json, pair.second.row(4));
      json << ",\"rot_z\": ";
      vecXToJson(json, pair.second.row(5));
      json << "}";
    }

    json << "}";
    json << "}";
    if (i < getNumSteps() - 1)
      json << ",";
  }
  json << "]";

  world->setPositions(originalWorldPos);

  json << "}";

  return json.str();
}

} // namespace trajectory
} // namespace dart