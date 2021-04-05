#ifndef DART_TRAJECTORY_ROLLOUT_HPP_
#define DART_TRAJECTORY_ROLLOUT_HPP_

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dart/neural/Mapping.hpp"
#include "dart/proto/TrajectoryRollout.pb.h"

namespace dart {

namespace simulation {
class World;
}

namespace trajectory {

class Problem;
class TrajectoryRolloutReal;
class TrajectoryRolloutRef;
class TrajectoryRolloutConstRef;

class TrajectoryRollout
{
public:
  virtual ~TrajectoryRollout();

  virtual const std::string& getRepresentationMapping() const = 0;
  virtual const std::vector<std::string>& getMappings() const = 0;

  virtual Eigen::Ref<Eigen::MatrixXs> getPoses(
      const std::string& mapping = "identity")
      = 0;
  virtual Eigen::Ref<Eigen::MatrixXs> getVels(
      const std::string& mapping = "identity")
      = 0;
  virtual Eigen::Ref<Eigen::MatrixXs> getForces(
      const std::string& mapping = "identity")
      = 0;
  virtual Eigen::Ref<Eigen::VectorXs> getMasses() = 0;

  virtual const Eigen::Ref<const Eigen::MatrixXs> getPosesConst(
      const std::string& mapping = "identity") const = 0;
  virtual const Eigen::Ref<const Eigen::MatrixXs> getVelsConst(
      const std::string& mapping = "identity") const = 0;
  virtual const Eigen::Ref<const Eigen::MatrixXs> getForcesConst(
      const std::string& mapping = "identity") const = 0;
  virtual const Eigen::Ref<const Eigen::VectorXs> getMassesConst() const = 0;

  virtual const std::unordered_map<std::string, Eigen::MatrixXs>&
  getMetadataMap() const = 0;
  virtual Eigen::MatrixXs getMetadata(const std::string& key) const = 0;
  virtual void setMetadata(const std::string& key, Eigen::MatrixXs value) = 0;

  /// This returns a trajectory rollout ref, corresponding to a slice
  /// of this trajectory rollout
  TrajectoryRolloutRef slice(int start, int len);

  /// This returns a trajectory rollout ref, corresponding to a slice
  /// of this trajectory rollout
  const TrajectoryRolloutConstRef sliceConst(int start, int len) const;

  /// This returns a copy of the trajectory rollout
  TrajectoryRollout* copy() const;

  /// This formats the rollout as JSON, which can be sent to the frontend to be
  /// parsed and displayed.
  std::string toJson(std::shared_ptr<simulation::World> world) const;

  /// This writes us out to a protobuf
  void serialize(proto::TrajectoryRollout& proto) const;

  /// This decodes a protobuf
  static TrajectoryRolloutReal deserialize(
      const proto::TrajectoryRollout& proto);

  /// This creates a rollout from forces over time
  static TrajectoryRolloutReal fromForces(
      std::shared_ptr<simulation::World> world,
      Eigen::VectorXs startPos,
      Eigen::VectorXs startVel,
      std::vector<Eigen::VectorXs> forces);

  /// This creates a rollout from poses over time
  static TrajectoryRolloutReal fromPoses(
      std::shared_ptr<simulation::World> world,
      std::vector<Eigen::VectorXs> poses);
};

class TrajectoryRolloutReal : public TrajectoryRollout
{
public:
  /// Fresh copy constructior
  TrajectoryRolloutReal(
      const std::unordered_map<std::string, std::shared_ptr<neural::Mapping>>
          mappings,
      int steps,
      std::string representationMapping,
      int massDim,
      const std::unordered_map<std::string, Eigen::MatrixXs> metadata);

  /// Create a fresh trajector rollout for a shot
  TrajectoryRolloutReal(Problem* shot);

  /// Deep copy constructor
  TrajectoryRolloutReal(const TrajectoryRollout* copy);

  /// Raw constructor
  TrajectoryRolloutReal(
      std::string representationMapping,
      const std::unordered_map<std::string, Eigen::MatrixXs> pos,
      const std::unordered_map<std::string, Eigen::MatrixXs> vel,
      const std::unordered_map<std::string, Eigen::MatrixXs> force,
      const Eigen::VectorXs mass,
      const std::unordered_map<std::string, Eigen::MatrixXs> metadata);

  const std::string& getRepresentationMapping() const override;
  const std::vector<std::string>& getMappings() const override;
  Eigen::Ref<Eigen::MatrixXs> getPoses(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::MatrixXs> getVels(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::MatrixXs> getForces(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::VectorXs> getMasses() override;
  const Eigen::Ref<const Eigen::MatrixXs> getPosesConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::MatrixXs> getVelsConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::MatrixXs> getForcesConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::VectorXs> getMassesConst() const override;

  virtual const std::unordered_map<std::string, Eigen::MatrixXs>&
  getMetadataMap() const override;
  virtual Eigen::MatrixXs getMetadata(const std::string& key) const override;
  virtual void setMetadata(
      const std::string& key, Eigen::MatrixXs value) override;

protected:
  std::unordered_map<std::string, Eigen::MatrixXs> mPoses;
  std::unordered_map<std::string, Eigen::MatrixXs> mVels;
  std::unordered_map<std::string, Eigen::MatrixXs> mForces;
  Eigen::VectorXs mMasses;
  std::unordered_map<std::string, Eigen::MatrixXs> mMetadata;
  std::string mRepresentationMapping;
  std::vector<std::string> mMappings;
};

class TrajectoryRolloutRef : public TrajectoryRollout
{
public:
  /// Slice constructor
  TrajectoryRolloutRef(TrajectoryRollout* toSlice, int start, int len);

  const std::string& getRepresentationMapping() const override;
  const std::vector<std::string>& getMappings() const override;
  Eigen::Ref<Eigen::MatrixXs> getPoses(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::MatrixXs> getVels(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::MatrixXs> getForces(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::VectorXs> getMasses() override;
  const Eigen::Ref<const Eigen::MatrixXs> getPosesConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::MatrixXs> getVelsConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::MatrixXs> getForcesConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::VectorXs> getMassesConst() const override;

  virtual const std::unordered_map<std::string, Eigen::MatrixXs>&
  getMetadataMap() const override;
  virtual Eigen::MatrixXs getMetadata(const std::string& key) const override;
  virtual void setMetadata(
      const std::string& key, Eigen::MatrixXs value) override;

protected:
  TrajectoryRollout* mToSlice;
  int mStart;
  int mLen;
};

class TrajectoryRolloutConstRef : public TrajectoryRollout
{
public:
  /// Slice constructor
  TrajectoryRolloutConstRef(
      const TrajectoryRollout* toSlice, int start, int len);

  const std::string& getRepresentationMapping() const override;
  const std::vector<std::string>& getMappings() const override;
  Eigen::Ref<Eigen::MatrixXs> getPoses(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::MatrixXs> getVels(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::MatrixXs> getForces(
      const std::string& mapping = "identity") override;
  Eigen::Ref<Eigen::VectorXs> getMasses() override;
  const Eigen::Ref<const Eigen::MatrixXs> getPosesConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::MatrixXs> getVelsConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::MatrixXs> getForcesConst(
      const std::string& mapping = "identity") const override;
  const Eigen::Ref<const Eigen::VectorXs> getMassesConst() const override;

  virtual const std::unordered_map<std::string, Eigen::MatrixXs>&
  getMetadataMap() const override;
  virtual Eigen::MatrixXs getMetadata(const std::string& key) const override;
  virtual void setMetadata(
      const std::string& key, Eigen::MatrixXs value) override;

protected:
  const TrajectoryRollout* mToSlice;
  int mStart;
  int mLen;
};

} // namespace trajectory
} // namespace dart

#endif