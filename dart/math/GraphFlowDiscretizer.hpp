#ifndef MATH_GRAPHFLOWDISCRETIZE_H_
#define MATH_GRAPHFLOWDISCRETIZE_H_

#include "dart/math/MathTypes.hpp"

//=============================================================================
//=============================================================================
namespace dart {
namespace math {

struct ParticlePath
{
  int startTime;
  std::vector<int> nodeHistory;
  s_t energyValue;

private:
  bool alreadyTransferred;

  friend class GraphFlowDiscretizer;
};

class GraphFlowDiscretizer
{
public:
  GraphFlowDiscretizer(
      int numNodes,
      std::vector<std::pair<int, int>> arcs,
      std::vector<bool> nodeAttachedToSink);

  /// This will find the least-squares closest rates of transfer across the arcs
  /// to end up with the energy levels at each node we got over time. The idea
  /// here is that arc rates may not perfectly reflect the observed changes in
  /// energy levels.
  Eigen::MatrixXs cleanUpArcRates(
      Eigen::MatrixXs energyLevels, Eigen::MatrixXs arcRates);

  /// This will attempt to create a set of ParticlePath objects that map the
  /// recorded graph node levels and flows as closely as possible. The particles
  /// can be created and destroyed within the arcs.
  std::vector<ParticlePath> discretize(
      int maxSimultaneousParticles,
      Eigen::MatrixXs energyLevels,
      Eigen::MatrixXs arcRates);

protected:
  int mNumNodes;
  std::vector<std::pair<int, int>> mArcs;
  std::vector<bool> mNodeAttachedToSink;
};

} // namespace math
} // namespace dart

#endif