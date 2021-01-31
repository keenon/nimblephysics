#include "dart/trajectory/TrajectoryRollout.hpp"

#include <sstream>

#include "dart/dynamics/BodyNode.hpp"
#include "dart/dynamics/Skeleton.hpp"
#include "dart/neural/RestorableSnapshot.hpp"
#include "dart/proto/SerializeEigen.hpp"
#include "dart/server/RawJsonUtils.hpp"
#include "dart/simulation/World.hpp"
#include "dart/trajectory/Problem.hpp"

// Make production builds happy with asserts
#define _unused(x) ((void)(x))

using namespace dart;

namespace dart {
namespace trajectory {

//==============================================================================
TrajectoryRollout::~TrajectoryRollout()
{
}

//==============================================================================
TrajectoryRolloutRef TrajectoryRollout::slice(int start, int len)
{
  return TrajectoryRolloutRef(this, start, len);
}

//==============================================================================
/// This returns a trajectory rollout ref, corresponding to a slice
/// of this trajectory rollout
const TrajectoryRolloutConstRef TrajectoryRollout::sliceConst(
    int start, int len) const
{
  return TrajectoryRolloutConstRef(this, start, len);
}

//==============================================================================
/// This returns a copy of the trajectory rollout
TrajectoryRollout* TrajectoryRollout::copy() const
{
  return new TrajectoryRolloutReal(this);
}

//==============================================================================
/// This formats the rollout as JSON, which can be sent to the frontend to be
/// parsed and displayed.
std::string TrajectoryRollout::toJson(
    std::shared_ptr<simulation::World> world) const
{
  neural::RestorableSnapshot snapshot(world);
  std::stringstream json;
  json << "{";

  int timesteps = getPosesConst().cols();

  std::vector<dynamics::BodyNode*> bodies = world->getAllBodyNodes();

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
    world->setPositions(getPosesConst("identity").col(t));
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

  snapshot.restore();

  return json.str();
}

//==============================================================================
/// This writes us out to a protobuf
void TrajectoryRollout::serialize(proto::TrajectoryRollout& proto) const
{
  proto.set_representationmapping(getRepresentationMapping());
  for (const std::string& mapping : getMappings())
  {
    proto::serializeMatrix(
        (*proto.mutable_pos())[mapping], getPosesConst(mapping));
    proto::serializeMatrix(
        (*proto.mutable_vel())[mapping], getVelsConst(mapping));
    proto::serializeMatrix(
        (*proto.mutable_force())[mapping], getForcesConst(mapping));
  }
  proto::serializeVector(*proto.mutable_mass(), getMassesConst());
  for (auto pair : getMetadataMap())
  {
    proto::serializeMatrix(
        (*proto.mutable_metadata())[pair.first], pair.second);
  }
}

//==============================================================================
/// This decodes a protobuf
TrajectoryRolloutReal TrajectoryRollout::deserialize(
    const proto::TrajectoryRollout& proto)
{
  std::string representationMapping = proto.representationmapping();
  std::unordered_map<std::string, Eigen::MatrixXd> pos;
  for (auto pair : proto.pos())
  {
    pos[pair.first] = proto::deserializeMatrix(pair.second);
  }
  std::unordered_map<std::string, Eigen::MatrixXd> vel;
  for (auto pair : proto.vel())
  {
    vel[pair.first] = proto::deserializeMatrix(pair.second);
  }
  std::unordered_map<std::string, Eigen::MatrixXd> force;
  for (auto pair : proto.force())
  {
    force[pair.first] = proto::deserializeMatrix(pair.second);
  }
  Eigen::VectorXd mass = proto::deserializeVector(proto.mass());
  std::unordered_map<std::string, Eigen::MatrixXd> metadata;
  for (auto pair : proto.metadata())
  {
    metadata[pair.first] = proto::deserializeMatrix(pair.second);
  }

  TrajectoryRolloutReal recovered = TrajectoryRolloutReal(
      representationMapping, pos, vel, force, mass, metadata);
  return recovered;
}

//==============================================================================
/// This creates a rollout from forces over time
TrajectoryRolloutReal TrajectoryRollout::fromForces(
    std::shared_ptr<simulation::World> world,
    Eigen::VectorXd startPos,
    Eigen::VectorXd startVel,
    std::vector<Eigen::VectorXd> forces)
{
  int steps = forces.size();
  int dofs = world->getNumDofs();
  Eigen::MatrixXd posMatrix = Eigen::MatrixXd::Zero(dofs, steps);
  Eigen::MatrixXd velMatrix = Eigen::MatrixXd::Zero(dofs, steps);
  Eigen::MatrixXd forceMatrix = Eigen::MatrixXd::Zero(dofs, steps);

  neural::RestorableSnapshot snapshot(world);
  world->setPositions(startPos);
  world->setVelocities(startVel);

  for (int i = 0; i < forces.size(); i++)
  {
    world->setExternalForces(forces[i]);
    world->step();
    posMatrix.col(i) = world->getPositions();
    velMatrix.col(i) = world->getVelocities();
    forceMatrix.col(i) = world->getExternalForces();
  }

  snapshot.restore();

  std::unordered_map<std::string, Eigen::MatrixXd> pos;
  pos["identity"] = posMatrix;
  std::unordered_map<std::string, Eigen::MatrixXd> vel;
  vel["identity"] = velMatrix;
  std::unordered_map<std::string, Eigen::MatrixXd> force;
  force["identity"] = forceMatrix;
  Eigen::VectorXd mass = world->getMasses();
  std::unordered_map<std::string, Eigen::MatrixXd> metadata;

  return TrajectoryRolloutReal("identity", pos, vel, force, mass, metadata);
}

//==============================================================================
/// This creates a rollout from poses over time
TrajectoryRolloutReal TrajectoryRollout::fromPoses(
    std::shared_ptr<simulation::World> world,
    std::vector<Eigen::VectorXd> poses)
{
  int steps = poses.size();
  int dofs = world->getNumDofs();
  Eigen::MatrixXd posMatrix = Eigen::MatrixXd::Zero(dofs, steps);
  Eigen::MatrixXd velMatrix = Eigen::MatrixXd::Zero(dofs, steps);
  Eigen::MatrixXd forceMatrix = Eigen::MatrixXd::Zero(dofs, steps);

  for (int i = 0; i < poses.size(); i++)
  {
    posMatrix.col(i) = poses[i];
  }

  std::unordered_map<std::string, Eigen::MatrixXd> pos;
  pos["identity"] = posMatrix;
  std::unordered_map<std::string, Eigen::MatrixXd> vel;
  vel["identity"] = velMatrix;
  std::unordered_map<std::string, Eigen::MatrixXd> force;
  force["identity"] = forceMatrix;
  Eigen::VectorXd mass = world->getMasses();
  std::unordered_map<std::string, Eigen::MatrixXd> metadata;

  return TrajectoryRolloutReal("identity", pos, vel, force, mass, metadata);
}

//==============================================================================
TrajectoryRolloutReal::TrajectoryRolloutReal(
    const std::unordered_map<std::string, std::shared_ptr<neural::Mapping>>
        mappings,
    int steps,
    std::string representationMapping,
    int massDim,
    const std::unordered_map<std::string, Eigen::MatrixXd> metadata)
  : mMetadata(metadata)
{
  mRepresentationMapping = representationMapping;
  for (auto pair : mappings)
  {
    mPoses[pair.first] = Eigen::MatrixXd::Zero(pair.second->getPosDim(), steps);
    mVels[pair.first] = Eigen::MatrixXd::Zero(pair.second->getVelDim(), steps);
    mForces[pair.first]
        = Eigen::MatrixXd::Zero(pair.second->getForceDim(), steps);
    mMappings.push_back(pair.first);
  }
  mMasses = Eigen::VectorXd::Zero(massDim);
}

//==============================================================================
TrajectoryRolloutReal::TrajectoryRolloutReal(Problem* shot)
  : TrajectoryRolloutReal(
      shot->getMappings(),
      shot->getNumSteps(),
      shot->getRepresentationName(),
      shot->getMassDims(),
      shot->getMetadataMap())
{
}

//==============================================================================
/// Raw constructor
TrajectoryRolloutReal::TrajectoryRolloutReal(
    std::string representationMapping,
    const std::unordered_map<std::string, Eigen::MatrixXd> pos,
    const std::unordered_map<std::string, Eigen::MatrixXd> vel,
    const std::unordered_map<std::string, Eigen::MatrixXd> force,
    const Eigen::VectorXd mass,
    const std::unordered_map<std::string, Eigen::MatrixXd> metadata)
  : mMasses(mass), mRepresentationMapping(representationMapping)
{
  for (auto pair : pos)
  {
    mMappings.push_back(pair.first);
  }
  for (const std::string& mapping : mMappings)
  {
    mPoses[mapping] = pos.at(mapping);
    mVels[mapping] = vel.at(mapping);
    mForces[mapping] = force.at(mapping);
  }
  for (auto pair : metadata)
  {
    mMetadata[pair.first] = pair.second;
  }
}

//==============================================================================
const std::string& TrajectoryRolloutReal::getRepresentationMapping() const
{
  return mRepresentationMapping;
}

//==============================================================================
const std::vector<std::string>& TrajectoryRolloutReal::getMappings() const
{
  return mMappings;
}

//==============================================================================
/// Deep copy constructor
TrajectoryRolloutReal::TrajectoryRolloutReal(const TrajectoryRollout* copy)
{
  mRepresentationMapping = copy->getRepresentationMapping();
  mMappings = copy->getMappings();
  for (std::string key : copy->getMappings())
  {
    mPoses[key] = copy->getPosesConst(key);
    mVels[key] = copy->getVelsConst(key);
    mForces[key] = copy->getForcesConst(key);
  }
  mMasses = copy->getMassesConst();
  mMetadata = copy->getMetadataMap();
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutReal::getPoses(
    const std::string& mapping)
{
  return mPoses.at(mapping);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutReal::getVels(
    const std::string& mapping)
{
  return mVels.at(mapping);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutReal::getForces(
    const std::string& mapping)
{
  return mForces.at(mapping);
}

//==============================================================================
Eigen::Ref<Eigen::VectorXd> TrajectoryRolloutReal::getMasses()
{
  return mMasses;
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutReal::getPosesConst(
    const std::string& mapping) const
{
  return mPoses.at(mapping);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutReal::getVelsConst(
    const std::string& mapping) const
{
  return mVels.at(mapping);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutReal::getForcesConst(
    const std::string& mapping) const
{
  return mForces.at(mapping);
}

//==============================================================================
const Eigen::Ref<const Eigen::VectorXd> TrajectoryRolloutReal::getMassesConst()
    const
{
  return mMasses;
}

//==============================================================================
const std::unordered_map<std::string, Eigen::MatrixXd>&
TrajectoryRolloutReal::getMetadataMap() const
{
  return mMetadata;
}

//==============================================================================
Eigen::MatrixXd TrajectoryRolloutReal::getMetadata(const std::string& key) const
{
  if (mMetadata.find(key) == mMetadata.end())
  {
    std::cout << "Warning: Asking TrajectoryRollout for metadata key \"" << key
              << "\" that doesn't exist! Keys that do exist:" << std::endl;
    for (auto pair : mMetadata)
    {
      std::cout << "   - \"" << pair.first << "\"" << std::endl;
    }
    return Eigen::MatrixXd::Zero(0, 0);
  }
  return mMetadata.at(key);
}

//==============================================================================
void TrajectoryRolloutReal::setMetadata(
    const std::string& key, Eigen::MatrixXd value)
{
  mMetadata[key] = value;
}

//==============================================================================
/// Slice constructor
TrajectoryRolloutRef::TrajectoryRolloutRef(
    TrajectoryRollout* toSlice, int start, int len)
  : mToSlice(toSlice), mStart(start), mLen(len)
{
}

//==============================================================================
const std::string& TrajectoryRolloutRef::getRepresentationMapping() const
{
  return mToSlice->getRepresentationMapping();
}

//==============================================================================
const std::vector<std::string>& TrajectoryRolloutRef::getMappings() const
{
  return mToSlice->getMappings();
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutRef::getPoses(
    const std::string& mapping)
{
  return mToSlice->getPoses(mapping).block(
      0, mStart, mToSlice->getPosesConst(mapping).rows(), mLen);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutRef::getVels(
    const std::string& mapping)
{
  return mToSlice->getVels(mapping).block(
      0, mStart, mToSlice->getVelsConst(mapping).rows(), mLen);
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutRef::getForces(
    const std::string& mapping)
{
  return mToSlice->getForces(mapping).block(
      0, mStart, mToSlice->getForcesConst(mapping).rows(), mLen);
}

//==============================================================================
Eigen::Ref<Eigen::VectorXd> TrajectoryRolloutRef::getMasses()
{
  return mToSlice->getMasses();
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutRef::getPosesConst(
    const std::string& mapping) const
{
  return mToSlice->getPosesConst(mapping).block(
      0, mStart, mToSlice->getPosesConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutRef::getVelsConst(
    const std::string& mapping) const
{
  return mToSlice->getVelsConst(mapping).block(
      0, mStart, mToSlice->getVelsConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutRef::getForcesConst(
    const std::string& mapping) const
{
  return mToSlice->getForcesConst(mapping).block(
      0, mStart, mToSlice->getForcesConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::VectorXd> TrajectoryRolloutRef::getMassesConst()
    const
{
  return mToSlice->getMassesConst();
}

//==============================================================================
const std::unordered_map<std::string, Eigen::MatrixXd>&
TrajectoryRolloutRef::getMetadataMap() const
{
  return mToSlice->getMetadataMap();
}

//==============================================================================
Eigen::MatrixXd TrajectoryRolloutRef::getMetadata(const std::string& key) const
{
  return mToSlice->getMetadata(key);
}

//==============================================================================
void TrajectoryRolloutRef::setMetadata(
    const std::string& key, Eigen::MatrixXd value)
{
  mToSlice->setMetadata(key, value);
}

//==============================================================================
/// Slice constructor
TrajectoryRolloutConstRef::TrajectoryRolloutConstRef(
    const TrajectoryRollout* toSlice, int start, int len)
  : mToSlice(toSlice), mStart(start), mLen(len)
{
}

//==============================================================================
const std::string& TrajectoryRolloutConstRef::getRepresentationMapping() const
{
  return mToSlice->getRepresentationMapping();
}

//==============================================================================
const std::vector<std::string>& TrajectoryRolloutConstRef::getMappings() const
{
  return mToSlice->getMappings();
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutConstRef::getPoses(
    const std::string& /* mapping */)
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
  throw std::runtime_error{"Execution should never reach this point"};
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutConstRef::getVels(
    const std::string& /* mapping */)
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
  throw std::runtime_error{"Execution should never reach this point"};
}

//==============================================================================
Eigen::Ref<Eigen::MatrixXd> TrajectoryRolloutConstRef::getForces(
    const std::string& /* mapping */)
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
  throw std::runtime_error{"Execution should never reach this point"};
}

//==============================================================================
Eigen::Ref<Eigen::VectorXd> TrajectoryRolloutConstRef::getMasses()
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
  throw std::runtime_error{"Execution should never reach this point"};
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd>
TrajectoryRolloutConstRef::getPosesConst(const std::string& mapping) const
{
  return mToSlice->getPosesConst(mapping).block(
      0, mStart, mToSlice->getPosesConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd> TrajectoryRolloutConstRef::getVelsConst(
    const std::string& mapping) const
{
  return mToSlice->getVelsConst(mapping).block(
      0, mStart, mToSlice->getVelsConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::MatrixXd>
TrajectoryRolloutConstRef::getForcesConst(const std::string& mapping) const
{
  return mToSlice->getForcesConst(mapping).block(
      0, mStart, mToSlice->getForcesConst(mapping).rows(), mLen);
}

//==============================================================================
const Eigen::Ref<const Eigen::VectorXd>
TrajectoryRolloutConstRef::getMassesConst() const
{
  return mToSlice->getMassesConst();
}

//==============================================================================
const std::unordered_map<std::string, Eigen::MatrixXd>&
TrajectoryRolloutConstRef::getMetadataMap() const
{
  return mToSlice->getMetadataMap();
}

//==============================================================================
Eigen::MatrixXd TrajectoryRolloutConstRef::getMetadata(
    const std::string& key) const
{
  return mToSlice->getMetadata(key);
}

//==============================================================================
void TrajectoryRolloutConstRef::setMetadata(
    const std::string& /* key */, Eigen::MatrixXd /* value */)
{
  assert(false && "It should be impossible to get a mutable reference from a TrajectorRolloutConstRef");
}

} // namespace trajectory
} // namespace dart